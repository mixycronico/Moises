"""
Pruebas para evaluar la capacidad de recuperación del motor Genesis bajo picos de carga extremos.

Este módulo contiene pruebas diseñadas para someter el motor Genesis a picos
repentinos de carga extrema, evaluar su comportamiento durante estos picos,
y medir su capacidad para recuperarse después de estas situaciones de estrés.
"""

import asyncio
import logging
import pytest
import time
import random
import gc
import threading
from typing import Dict, Any, List, Optional, Set, Tuple, Callable, Coroutine

# Importamos utilidades para timeouts
from tests.utils.timeout_helpers import (
    emit_with_timeout,
    check_component_status,
    run_test_with_timing,
    cleanup_engine
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nos aseguramos que pytest detecte correctamente las pruebas asíncronas
pytestmark = pytest.mark.asyncio

# Importamos clases necesarias
from genesis.core.component import Component
from genesis.core.engine_dynamic_blocks import DynamicExpansionEngine


class LoadGeneratorComponent(Component):
    """
    Componente diseñado para generar carga sustancial en el sistema.
    
    Este componente puede generar diferentes tipos de carga:
    1. CPU-bound: operaciones que consumen CPU intensivamente
    2. Memory-bound: operaciones que consumen memoria intensivamente
    3. I/O-bound: operaciones que simulan bloqueos de I/O
    4. Mixed: combinación de todas las anteriores
    """
    
    def __init__(self, name: str, load_type: str = "mixed"):
        """
        Inicializar componente generador de carga.
        
        Args:
            name: Nombre del componente
            load_type: Tipo de carga a generar ("cpu", "memory", "io", "mixed")
        """
        super().__init__(name)
        self.load_type = load_type
        self.processing_count = 0
        self.error_count = 0
        self.events_seen = set()
        
        # Para medir rendimiento
        self.start_time = time.time()
        self.last_reset_time = self.start_time
        self.processing_times = []
        
        # Para modo burst
        self.burst_mode = False
        self.burst_recovery = False
        
        # Referencia al motor (añadido para resolver problema de registro)
        self._engine = None
        
    def set_engine(self, engine):
        """
        Establecer referencia al motor.
        
        Args:
            engine: Motor a asociar con este componente
        """
        self._engine = engine
        
    async def start(self) -> None:
        """Iniciar el componente."""
        logger.debug(f"LoadGeneratorComponent {self.name} iniciado")
        self.start_time = time.time()
        self.last_reset_time = self.start_time
        
    async def stop(self) -> None:
        """Detener el componente."""
        logger.debug(f"LoadGeneratorComponent {self.name} detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos generando carga según el tipo configurado.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta al evento o None
        """
        # Registrar que vimos este evento
        event_id = data.get("event_id", "unknown")
        self.events_seen.add(event_id)
        
        process_start = time.time()
        
        # Verificar si es un comando de control
        if event_type == "control_command":
            command = data.get("command", "")
            
            if command == "enable_burst_mode":
                self.burst_mode = True
                return {"success": True, "mode": "burst_enabled"}
                
            elif command == "disable_burst_mode":
                self.burst_mode = False
                return {"success": True, "mode": "burst_disabled"}
                
            elif command == "enable_recovery_mode":
                self.burst_recovery = True
                return {"success": True, "mode": "recovery_enabled"}
                
            elif command == "disable_recovery_mode":
                self.burst_recovery = False
                return {"success": True, "mode": "recovery_disabled"}
                
            elif command == "reset_metrics":
                self.processing_count = 0
                self.error_count = 0
                self.events_seen.clear()
                self.processing_times = []
                self.last_reset_time = time.time()
                return {"success": True, "action": "metrics_reset"}
                
            elif command == "get_metrics":
                return self._get_metrics()
        
        # Para los eventos de carga, aplicar la carga según configuración
        elif event_type == "generate_load":
            # En modo de recuperación, procesar más rápido
            if self.burst_recovery:
                await asyncio.sleep(0.001)
                self.processing_count += 1
                process_time = time.time() - process_start
                self.processing_times.append(process_time)
                return {
                    "success": True,
                    "load_type": self.load_type,
                    "process_time": process_time,
                    "event_id": event_id
                }
            
            # En modo burst, generar mucha más carga
            if self.burst_mode:
                await self._generate_extreme_load(data)
            else:
                await self._generate_normal_load(data)
            
            self.processing_count += 1
            process_time = time.time() - process_start
            self.processing_times.append(process_time)
            
            return {
                "success": True,
                "load_type": self.load_type,
                "process_time": process_time,
                "event_id": event_id
            }
        
        # Para cualquier otro tipo de evento
        else:
            await asyncio.sleep(0.01)  # Simulación de procesamiento básico
            self.processing_count += 1
            
            return {
                "event_type": event_type,
                "processed": True,
                "component": self.name
            }
    
    async def _generate_normal_load(self, data: Dict[str, Any]) -> None:
        """
        Generar carga normal según el tipo configurado.
        
        Args:
            data: Datos del evento
        """
        # Ajustar intensidad según datos
        intensity = data.get("intensity", 1.0)
        duration_factor = min(max(intensity, 0.1), 5.0)  # Limitar entre 0.1 y 5
        
        if self.load_type == "cpu" or self.load_type == "mixed":
            # Carga intensiva de CPU
            start = time.time()
            while time.time() - start < 0.05 * duration_factor:
                # Operación intensiva en CPU
                _ = [i ** 2 for i in range(1000)]
        
        if self.load_type == "memory" or self.load_type == "mixed":
            # Carga de memoria
            size = int(50000 * duration_factor)
            large_list = [random.random() for _ in range(size)]
            # Asegurar que se use
            sum_value = sum(large_list[:100])
            # Liberar después de usar
            del large_list
        
        if self.load_type == "io" or self.load_type == "mixed":
            # Simular carga de I/O con sleeps
            await asyncio.sleep(0.02 * duration_factor)
    
    async def _generate_extreme_load(self, data: Dict[str, Any]) -> None:
        """
        Generar carga extrema para simular picos de carga.
        
        Args:
            data: Datos del evento
        """
        # Factor de intensidad para carga extrema
        intensity = data.get("intensity", 1.0) * 5.0  # 5x más intenso que normal
        duration_factor = min(max(intensity, 1.0), 10.0)  # Limitar entre 1 y 10
        
        if self.load_type == "cpu" or self.load_type == "mixed":
            # Carga extrema de CPU
            start = time.time()
            while time.time() - start < 0.1 * duration_factor:
                # Operación muy intensiva en CPU
                _ = [i ** 3 for i in range(2000)]
        
        if self.load_type == "memory" or self.load_type == "mixed":
            # Carga extrema de memoria
            size = int(200000 * duration_factor)
            large_dict = {i: random.random() for i in range(size)}
            # Asegurar que se use
            sum_value = sum(v for k, v in list(large_dict.items())[:100])
            # Liberar después de usar
            del large_dict
            
            # Forzar recolección de basura para simular presión de memoria
            gc.collect()
        
        if self.load_type == "io" or self.load_type == "mixed":
            # Simular carga de I/O intensiva con sleeps más largos
            await asyncio.sleep(0.05 * duration_factor)
            
            # Simular algunos bloqueos ocasionales (5% de probabilidad)
            if random.random() < 0.05:
                await asyncio.sleep(0.5)  # Bloqueo sustancial
    
    def _get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas actuales del componente.
        
        Returns:
            Diccionario con métricas
        """
        current_time = time.time()
        total_time = current_time - self.last_reset_time
        
        # Calcular eventos por segundo
        events_per_second = self.processing_count / total_time if total_time > 0 else 0
        
        # Calcular tiempos de procesamiento
        avg_process_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        max_process_time = max(self.processing_times) if self.processing_times else 0
        min_process_time = min(self.processing_times) if self.processing_times else 0
        
        # Calcular percentiles 95 y 99 si hay suficientes datos
        p95 = p99 = 0
        if len(self.processing_times) >= 20:
            sorted_times = sorted(self.processing_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            p95 = sorted_times[p95_idx]
            p99 = sorted_times[p99_idx]
        
        return {
            "component": self.name,
            "load_type": self.load_type,
            "processing_count": self.processing_count,
            "error_count": self.error_count,
            "unique_events": len(self.events_seen),
            "events_per_second": events_per_second,
            "avg_process_time": avg_process_time,
            "max_process_time": max_process_time,
            "min_process_time": min_process_time,
            "p95_process_time": p95,
            "p99_process_time": p99,
            "burst_mode": self.burst_mode,
            "recovery_mode": self.burst_recovery,
            "uptime": current_time - self.start_time
        }


class BurstMonitorComponent(Component):
    """
    Componente para monitorear y gestionar picos de carga.
    
    Este componente detecta picos de carga anormales y puede
    activar modos de recuperación en los componentes LoadGenerator.
    """
    
    def __init__(self, name: str, load_generators: Optional[List[str]] = None, threshold: float = 0.8):
        """
        Inicializar monitor de picos.
        
        Args:
            name: Nombre del componente
            load_generators: Lista de componentes generadores de carga a monitorear (opcional)
            threshold: Umbral para considerar sobrecarga (0.0-1.0)
        """
        super().__init__(name)
        self.load_generators = load_generators if load_generators is not None else []
        self._engine = None  # Añadimos el atributo _engine inicializado
        self.threshold = threshold
        self.monitoring_active = False
        self.metrics_history = []
        self.recovery_count = 0
        self.last_recovery_time = 0
        self.max_metrics_history = 100
        
        # Detectores de anomalías
        self.anomaly_detected = False
        self.anomaly_start_time = 0
        
    def set_engine(self, engine):
        """Establecer referencia al motor de eventos."""
        self._engine = engine
        
    async def start(self) -> None:
        """Iniciar el componente de monitoreo."""
        logger.debug(f"BurstMonitorComponent {self.name} iniciado")
        self.monitoring_active = True
        
    async def stop(self) -> None:
        """Detener el componente de monitoreo."""
        logger.debug(f"BurstMonitorComponent {self.name} detenido")
        self.monitoring_active = False
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos para monitorear y gestionar picos de carga.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta al evento o None
        """
        if event_type == "monitor_command":
            command = data.get("command", "")
            
            if command == "start_monitoring":
                self.monitoring_active = True
                return {"success": True, "action": "monitoring_started"}
            
            elif command == "stop_monitoring":
                self.monitoring_active = False
                return {"success": True, "action": "monitoring_stopped"}
            
            elif command == "get_metrics":
                return self._get_monitor_metrics()
            
            elif command == "reset_metrics":
                self.metrics_history = []
                self.recovery_count = 0
                self.anomaly_detected = False
                return {"success": True, "action": "metrics_reset"}
            
            elif command == "activate_recovery":
                await self._activate_recovery_mode()
                return {"success": True, "action": "recovery_activated"}
            
            elif command == "deactivate_recovery":
                await self._deactivate_recovery_mode()
                return {"success": True, "action": "recovery_deactivated"}
        
        elif event_type == "collect_metrics" and self.monitoring_active:
            metrics = await self._collect_generator_metrics()
            
            # Guardar métricas en historial
            self.metrics_history.append({
                "timestamp": time.time(),
                "metrics": metrics
            })
            
            # Limitar tamaño del historial
            if len(self.metrics_history) > self.max_metrics_history:
                self.metrics_history = self.metrics_history[-self.max_metrics_history:]
            
            # Analizar métricas para detectar sobrecarga
            await self._analyze_metrics(metrics)
            
            return {
                "success": True,
                "collected": len(metrics),
                "anomaly_detected": self.anomaly_detected
            }
        
        return None
    
    async def _collect_generator_metrics(self) -> List[Dict[str, Any]]:
        """
        Recolectar métricas de todos los generadores de carga.
        
        Returns:
            Lista de métricas por generador
        """
        metrics = []
        
        for generator_name in self.load_generators:
            try:
                response = await emit_with_timeout(
                    self._engine,
                    "control_command",
                    {"command": "get_metrics"},
                    generator_name,
                    timeout=1.0
                )
                
                if response and len(response) > 0:
                    metrics.append(response[0])
            except Exception as e:
                logger.warning(f"Error al recolectar métricas de {generator_name}: {e}")
        
        return metrics
    
    async def _analyze_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """
        Analizar métricas para detectar condiciones anómalas.
        
        Args:
            metrics: Lista de métricas recolectadas
        """
        # Identificar potenciales sobrecargas
        high_process_times = []
        high_burst_counts = 0
        
        for metric in metrics:
            # Verificar tiempos de procesamiento extremos
            if metric.get("p95_process_time", 0) > 0.1:  # Más de 100ms es lento
                high_process_times.append((metric.get("component"), metric.get("p95_process_time")))
            
            # Contar componentes en modo burst
            if metric.get("burst_mode", False):
                high_burst_counts += 1
        
        # Decidir si hay una anomalía
        old_anomaly_state = self.anomaly_detected
        current_time = time.time()
        
        # Condiciones para considerar anomalía:
        # 1. Varios componentes con tiempos de proceso elevados o
        # 2. Muchos componentes en modo burst
        if len(high_process_times) >= len(metrics) // 2 or high_burst_counts > len(metrics) * 0.7:
            if not self.anomaly_detected:
                # Nueva anomalía detectada
                self.anomaly_detected = True
                self.anomaly_start_time = current_time
                logger.warning(f"Anomalía detectada por {self.name}: Activando recuperación")
                
                # Si es una nueva anomalía, activar recuperación
                await self._activate_recovery_mode()
                
        elif self.anomaly_detected and current_time - self.anomaly_start_time > 5.0:
            # Si ha pasado suficiente tiempo, desactivar el modo anomalía
            self.anomaly_detected = False
            logger.info(f"{self.name}: Sistema recuperado, desactivando modo recuperación")
            
            # Desactivar recuperación
            await self._deactivate_recovery_mode()
    
    async def _activate_recovery_mode(self) -> None:
        """Activar modo de recuperación en todos los generadores."""
        self.recovery_count += 1
        self.last_recovery_time = time.time()
        
        # Primero desactivar modo burst
        for generator_name in self.load_generators:
            try:
                await emit_with_timeout(
                    self._engine,
                    "control_command",
                    {"command": "disable_burst_mode"},
                    generator_name,
                    timeout=1.0
                )
            except Exception as e:
                logger.warning(f"Error al desactivar burst en {generator_name}: {e}")
        
        # Luego activar modo recuperación
        for generator_name in self.load_generators:
            try:
                await emit_with_timeout(
                    self._engine,
                    "control_command",
                    {"command": "enable_recovery_mode"},
                    generator_name,
                    timeout=1.0
                )
            except Exception as e:
                logger.warning(f"Error al activar recuperación en {generator_name}: {e}")
    
    async def _deactivate_recovery_mode(self) -> None:
        """Desactivar modo de recuperación en todos los generadores."""
        for generator_name in self.load_generators:
            try:
                await emit_with_timeout(
                    self._engine,
                    "control_command",
                    {"command": "disable_recovery_mode"},
                    generator_name,
                    timeout=1.0
                )
            except Exception as e:
                logger.warning(f"Error al desactivar recuperación en {generator_name}: {e}")
    
    def _get_monitor_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas actuales del monitor.
        
        Returns:
            Diccionario con métricas del monitor
        """
        current_time = time.time()
        
        # Analizar historial de métricas para calcular tendencias
        avg_process_times = []
        events_per_second = []
        
        for entry in self.metrics_history[-10:]:  # Últimas 10 entradas
            metrics = entry.get("metrics", [])
            for metric in metrics:
                avg_process_times.append(metric.get("avg_process_time", 0))
                events_per_second.append(metric.get("events_per_second", 0))
        
        avg_process_time = sum(avg_process_times) / len(avg_process_times) if avg_process_times else 0
        avg_events_per_second = sum(events_per_second) / len(events_per_second) if events_per_second else 0
        
        return {
            "component": self.name,
            "monitoring_active": self.monitoring_active,
            "metrics_collected": len(self.metrics_history),
            "recovery_count": self.recovery_count,
            "last_recovery_ago": current_time - self.last_recovery_time if self.last_recovery_time > 0 else -1,
            "anomaly_detected": self.anomaly_detected,
            "anomaly_duration": current_time - self.anomaly_start_time if self.anomaly_detected else 0,
            "avg_process_time": avg_process_time,
            "avg_events_per_second": avg_events_per_second,
            "load_generators": self.load_generators
        }


@pytest.mark.asyncio
async def test_peak_load_recovery():
    """
    Prueba para evaluar la recuperación del sistema tras picos de carga extremos.
    
    Esta prueba somete al motor a un flujo de eventos normal, seguido de un pico extremo de carga,
    y evalúa su capacidad para recuperarse después del pico.
    """
    # Crear motor con expansión dinámica para manejar picos
    engine = DynamicExpansionEngine(
        min_concurrent_blocks=2,
        max_concurrent_blocks=8,
        expansion_threshold=0.7,
        scale_cooldown=2.0
    )
    
    # Función interna para la prueba
    async def run_peak_load_test(engine):
        # Definir diferentes tipos de generadores de carga
        load_generators = [
            LoadGeneratorComponent("cpu_generator", "cpu"),
            LoadGeneratorComponent("memory_generator", "memory"),
            LoadGeneratorComponent("io_generator", "io"),
            LoadGeneratorComponent("mixed_generator_1", "mixed"),
            LoadGeneratorComponent("mixed_generator_2", "mixed")
        ]
        
        # Registrar componentes en el motor
        generator_names = []
        for generator in load_generators:
            # Establecer el motor en cada generador
            generator.set_engine(engine)
            await engine.register_component(generator)
            generator_names.append(generator.name)
        
        # Registrar monitor de picos
        monitor = BurstMonitorComponent("load_monitor", generator_names)
        # Usar método set_engine para establecer el motor en el monitor
        monitor.set_engine(engine)
        await engine.register_component(monitor)
        
        logger.info(f"Registrados {len(load_generators)} generadores de carga y 1 monitor")
        
        # Fase 1: Carga normal para establecer la línea base
        logger.info("Fase 1: Estableciendo carga normal (línea base)")
        baseline_metrics = await run_normal_load_phase(engine, generator_names, monitor.name, 100)
        
        # Fase 2: Pico extremo de carga
        logger.info("Fase 2: Generando pico extremo de carga")
        peak_metrics = await run_peak_load_phase(engine, generator_names, monitor.name, 200)
        
        # Fase 3: Período de recuperación
        logger.info("Fase 3: Evaluando recuperación después del pico")
        recovery_metrics = await run_recovery_phase(engine, generator_names, monitor.name, 100)
        
        # Calcular métricas de recuperación
        recovery_ratio = calculate_recovery_metrics(baseline_metrics, peak_metrics, recovery_metrics)
        
        logger.info(f"Ratio de recuperación: {recovery_ratio:.2f}x (1.0 = recuperación completa)")
        
        return {
            "baseline_metrics": baseline_metrics,
            "peak_metrics": peak_metrics,
            "recovery_metrics": recovery_metrics,
            "recovery_ratio": recovery_ratio,
            "test_result": "success" if recovery_ratio > 0.7 else "partial_success" if recovery_ratio > 0.4 else "failed"
        }
    
    # Fase de carga normal
    async def run_normal_load_phase(engine, generator_names, monitor_name, num_events):
        # Recolectar métricas iniciales
        await emit_with_timeout(
            engine,
            "monitor_command",
            {"command": "reset_metrics"},
            monitor_name,
            timeout=1.0
        )
        
        # Resetear métricas en generadores
        for generator in generator_names:
            await emit_with_timeout(
                engine,
                "control_command",
                {"command": "reset_metrics"},
                generator,
                timeout=1.0
            )
        
        # Generar carga normal distribuida
        tasks = []
        for i in range(num_events):
            # Distribuir eventos entre generadores
            generator = random.choice(generator_names)
            intensity = random.uniform(0.5, 1.5)  # Variación normal de intensidad
            
            task = emit_with_timeout(
                engine,
                "generate_load",
                {"event_id": f"normal_{i}", "intensity": intensity},
                generator,
                timeout=2.0
            )
            tasks.append(task)
            
            # Espaciar los eventos ligeramente
            if i % 10 == 0:
                await asyncio.sleep(0.01)
        
        # Esperar a que se completen todos los eventos
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Contar resultados exitosos
        success_count = sum(1 for r in results if not isinstance(r, Exception) and r)
        logger.info(f"Carga normal completada: {success_count}/{len(tasks)} eventos exitosos")
        
        # Recolectar métricas finales de esta fase
        await emit_with_timeout(
            engine,
            "collect_metrics",
            {},
            monitor_name,
            timeout=1.0
        )
        
        monitor_metrics = await emit_with_timeout(
            engine,
            "monitor_command",
            {"command": "get_metrics"},
            monitor_name,
            timeout=1.0
        )
        
        # Recolectar métricas de los generadores
        generator_metrics = []
        for generator in generator_names:
            metrics = await emit_with_timeout(
                engine,
                "control_command",
                {"command": "get_metrics"},
                generator,
                timeout=1.0
            )
            if metrics and len(metrics) > 0:
                generator_metrics.append(metrics[0])
        
        return {
            "monitor": monitor_metrics[0] if monitor_metrics and len(monitor_metrics) > 0 else {},
            "generators": generator_metrics,
            "success_rate": success_count / len(tasks) if tasks else 0
        }
    
    # Fase de pico de carga
    async def run_peak_load_phase(engine, generator_names, monitor_name, num_events):
        # Activar modo burst en generadores
        for generator in generator_names:
            await emit_with_timeout(
                engine,
                "control_command",
                {"command": "enable_burst_mode"},
                generator,
                timeout=1.0
            )
        
        # Generar pico de carga extremo (muchos eventos simultáneos)
        burst_tasks = []
        for i in range(num_events):
            # Distribuir eventos entre generadores con intensidad alta
            generator = random.choice(generator_names)
            intensity = random.uniform(2.0, 4.0)  # Intensidad muy alta para provocar sobrecarga
            
            task = emit_with_timeout(
                engine,
                "generate_load",
                {"event_id": f"burst_{i}", "intensity": intensity},
                generator,
                timeout=3.0  # Timeout más alto para picos
            )
            burst_tasks.append(task)
            
            # Acumular eventos rápidamente (sin pausas) para generar un pico real
        
        # Esperar a que se procesen (o fallen) todos los eventos del pico
        burst_results = await asyncio.gather(*burst_tasks, return_exceptions=True)
        
        # Contar resultados exitosos y fallos
        burst_success = sum(1 for r in burst_results if not isinstance(r, Exception) and r)
        burst_errors = sum(1 for r in burst_results if isinstance(r, Exception))
        
        logger.info(f"Pico de carga completado: {burst_success} exitosos, {burst_errors} errores de {len(burst_tasks)} eventos")
        
        # Recolectar métricas después del pico
        await emit_with_timeout(
            engine,
            "collect_metrics",
            {},
            monitor_name,
            timeout=1.0
        )
        
        # Obtener métricas del monitor
        monitor_metrics = await emit_with_timeout(
            engine,
            "monitor_command",
            {"command": "get_metrics"},
            monitor_name,
            timeout=1.0
        )
        
        # Recolectar métricas de los generadores
        generator_metrics = []
        for generator in generator_names:
            metrics = await emit_with_timeout(
                engine,
                "control_command",
                {"command": "get_metrics"},
                generator,
                timeout=1.0
            )
            if metrics and len(metrics) > 0:
                generator_metrics.append(metrics[0])
        
        return {
            "monitor": monitor_metrics[0] if monitor_metrics and len(monitor_metrics) > 0 else {},
            "generators": generator_metrics,
            "success_rate": burst_success / len(burst_tasks) if burst_tasks else 0,
            "error_rate": burst_errors / len(burst_tasks) if burst_tasks else 0
        }
    
    # Fase de recuperación
    async def run_recovery_phase(engine, generator_names, monitor_name, num_events):
        # Desactivar modo burst en generadores (para verificar recuperación natural)
        for generator in generator_names:
            await emit_with_timeout(
                engine,
                "control_command",
                {"command": "disable_burst_mode"},
                generator,
                timeout=1.0
            )
        
        # Dar tiempo para estabilización
        logger.info("Esperando 2 segundos para estabilización del sistema...")
        await asyncio.sleep(2)
        
        # Generar carga normal nuevamente para medir recuperación
        recovery_tasks = []
        for i in range(num_events):
            # Distribuir eventos entre generadores
            generator = random.choice(generator_names)
            intensity = random.uniform(0.5, 1.5)  # Volver a carga normal
            
            task = emit_with_timeout(
                engine,
                "generate_load",
                {"event_id": f"recovery_{i}", "intensity": intensity},
                generator,
                timeout=2.0
            )
            recovery_tasks.append(task)
            
            # Espaciar los eventos ligeramente
            if i % 10 == 0:
                await asyncio.sleep(0.01)
        
        # Esperar a que se completen todos los eventos
        recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
        
        # Contar resultados exitosos
        recovery_success = sum(1 for r in recovery_results if not isinstance(r, Exception) and r)
        logger.info(f"Fase de recuperación completada: {recovery_success}/{len(recovery_tasks)} eventos exitosos")
        
        # Recolectar métricas finales
        await emit_with_timeout(
            engine,
            "collect_metrics",
            {},
            monitor_name,
            timeout=1.0
        )
        
        monitor_metrics = await emit_with_timeout(
            engine,
            "monitor_command",
            {"command": "get_metrics"},
            monitor_name,
            timeout=1.0
        )
        
        # Recolectar métricas de los generadores
        generator_metrics = []
        for generator in generator_names:
            metrics = await emit_with_timeout(
                engine,
                "control_command",
                {"command": "get_metrics"},
                generator,
                timeout=1.0
            )
            if metrics and len(metrics) > 0:
                generator_metrics.append(metrics[0])
        
        return {
            "monitor": monitor_metrics[0] if monitor_metrics and len(monitor_metrics) > 0 else {},
            "generators": generator_metrics,
            "success_rate": recovery_success / len(recovery_tasks) if recovery_tasks else 0
        }
    
    # Calcular métricas de recuperación
    def calculate_recovery_metrics(baseline, peak, recovery):
        # Obtener tasas de éxito
        baseline_rate = baseline.get("success_rate", 0)
        peak_rate = peak.get("success_rate", 0)
        recovery_rate = recovery.get("success_rate", 0)
        
        # Calcular tiempos promedio de procesamiento
        baseline_times = [g.get("avg_process_time", 0) for g in baseline.get("generators", [])]
        peak_times = [g.get("avg_process_time", 0) for g in peak.get("generators", [])]
        recovery_times = [g.get("avg_process_time", 0) for g in recovery.get("generators", [])]
        
        baseline_avg_time = sum(baseline_times) / len(baseline_times) if baseline_times else 0
        peak_avg_time = sum(peak_times) / len(peak_times) if peak_times else 0
        recovery_avg_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        # Evitar división por cero
        if baseline_avg_time == 0:
            baseline_avg_time = 0.001
        if peak_avg_time == 0:
            peak_avg_time = 0.001
        
        # Calcular métricas de recuperación
        success_recovery = recovery_rate / baseline_rate if baseline_rate > 0 else 0
        time_recovery = baseline_avg_time / recovery_avg_time if recovery_avg_time > 0 else 0
        
        # Combinar métricas (promedio ponderado)
        recovery_ratio = 0.7 * success_recovery + 0.3 * time_recovery
        
        logger.info(f"Métricas de recuperación:")
        logger.info(f"- Tasa de éxito: {baseline_rate:.2%} → {peak_rate:.2%} → {recovery_rate:.2%}")
        logger.info(f"- Tiempo promedio: {baseline_avg_time*1000:.1f}ms → {peak_avg_time*1000:.1f}ms → {recovery_avg_time*1000:.1f}ms")
        logger.info(f"- Recuperación de tasa de éxito: {success_recovery:.2f}x")
        logger.info(f"- Recuperación de tiempo: {time_recovery:.2f}x")
        
        return recovery_ratio
    
    # Ejecutar y medir la prueba completa
    result = await run_test_with_timing(engine, "test_peak_load_recovery", run_peak_load_test)
    
    # Verificar que la prueba complete y el sistema se recupere adecuadamente
    assert isinstance(result, dict), "La prueba debería devolver un dict con resultados"
    
    # Evaluar recuperación
    recovery_ratio = result.get("recovery_ratio", 0)
    assert recovery_ratio > 0.4, f"Recuperación insuficiente tras pico de carga: {recovery_ratio:.2f}x"
    
    # Verificar bloques de expansión
    baseline_blocks = result.get("baseline_metrics", {}).get("monitor", {}).get("expansion_blocks", 0)
    peak_blocks = result.get("peak_metrics", {}).get("monitor", {}).get("expansion_blocks", 0)
    recovery_blocks = result.get("recovery_metrics", {}).get("monitor", {}).get("expansion_blocks", 0)
    
    # Verificar si hubo expansión durante el pico (no crítico, solo informativo)
    if peak_blocks > baseline_blocks:
        logger.info(f"Expansión dinámica activada durante el pico: {baseline_blocks} → {peak_blocks} bloques")
    
    # Limpiar recursos
    await cleanup_engine(engine)


@pytest.mark.asyncio
async def test_concurrent_load_distribution():
    """
    Prueba la distribución de carga concurrente entre los bloques paralelos.
    
    Esta prueba verifica que el sistema distribuya eficientemente la carga
    a través de los bloques paralelos disponibles.
    """
    # Crear motor con configuración avanzada
    engine = DynamicExpansionEngine(
        min_concurrent_blocks=2,
        max_concurrent_blocks=6,
        expansion_threshold=0.6,
        scale_cooldown=1.0
    )
    
    # Función interna para la prueba
    async def run_distribution_test(engine):
        # Crear componentes para simulación de carga
        load_generators = []
        generator_names = []
        
        # Crear varios generadores de cada tipo
        for load_type in ["cpu", "memory", "io", "mixed"]:
            for i in range(3):  # 3 generadores de cada tipo
                generator = LoadGeneratorComponent(f"{load_type}_gen_{i}", load_type)
                generator.set_engine(engine)  # Establecer el motor en cada generador
                load_generators.append(generator)
                generator_names.append(generator.name)
                await engine.register_component(generator)
        
        logger.info(f"Registrados {len(load_generators)} generadores de carga")
        
        # Función para enviar carga específica a un generador
        async def send_load_to_generator(generator, intensity, event_id):
            try:
                return await emit_with_timeout(
                    engine,
                    "generate_load",
                    {"event_id": event_id, "intensity": intensity},
                    generator,
                    timeout=2.0
                )
            except Exception as e:
                logger.warning(f"Error enviando carga a {generator}: {e}")
                return None
        
        # Función para enviar carga distribuida a los generadores
        async def distribute_load(num_events, intensity_range=(0.5, 1.5)):
            tasks = []
            event_ids = []
            
            for i in range(num_events):
                generator = random.choice(generator_names)
                intensity = random.uniform(*intensity_range)
                event_id = f"event_{i}"
                event_ids.append(event_id)
                
                task = send_load_to_generator(generator, intensity, event_id)
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start_time
            
            success_count = sum(1 for r in results if not isinstance(r, Exception) and r)
            error_count = sum(1 for r in results if isinstance(r, Exception))
            
            return {
                "total_events": len(tasks),
                "success_count": success_count,
                "error_count": error_count,
                "elapsed_time": elapsed,
                "events_per_second": len(tasks) / elapsed if elapsed > 0 else 0
            }
        
        # Fase 1: Carga moderada para establecer línea base
        logger.info("Fase 1: Carga moderada (línea base)")
        baseline_result = await distribute_load(100, (0.5, 1.0))
        
        # Obtener métricas iniciales
        generator_metrics_baseline = []
        for generator in generator_names:
            try:
                metrics = await emit_with_timeout(
                    engine,
                    "control_command",
                    {"command": "get_metrics"},
                    generator,
                    timeout=1.0
                )
                if metrics and len(metrics) > 0:
                    generator_metrics_baseline.append(metrics[0])
            except Exception:
                pass
        
        # Fase 2: Carga alta distribuida uniformemente
        logger.info("Fase 2: Carga alta distribuida")
        high_uniform_result = await distribute_load(200, (1.0, 2.0))
        
        # Fase 3: Carga mixta (algunos eventos pesados, otros ligeros)
        logger.info("Fase 3: Carga mixta (variada)")
        mixed_results = []
        
        # Ejecutar varias cargas mixtas
        for _ in range(3):
            # Generar carga mixta distribuyendo intensidades de manera desigual
            mix_tasks = []
            # 30% carga ligera
            for i in range(30):
                generator = random.choice(generator_names)
                task = send_load_to_generator(generator, random.uniform(0.2, 0.5), f"light_{i}")
                mix_tasks.append(task)
            
            # 50% carga media
            for i in range(50):
                generator = random.choice(generator_names)
                task = send_load_to_generator(generator, random.uniform(0.5, 1.5), f"medium_{i}")
                mix_tasks.append(task)
            
            # 20% carga pesada
            for i in range(20):
                generator = random.choice(generator_names)
                task = send_load_to_generator(generator, random.uniform(1.5, 3.0), f"heavy_{i}")
                mix_tasks.append(task)
            
            # Ejecutar todos los eventos mixtos concurrentemente
            start_time = time.time()
            mix_results = await asyncio.gather(*mix_tasks, return_exceptions=True)
            elapsed = time.time() - start_time
            
            success_count = sum(1 for r in mix_results if not isinstance(r, Exception) and r)
            error_count = sum(1 for r in mix_results if isinstance(r, Exception))
            
            mixed_results.append({
                "total_events": len(mix_tasks),
                "success_count": success_count,
                "error_count": error_count,
                "elapsed_time": elapsed,
                "events_per_second": len(mix_tasks) / elapsed if elapsed > 0 else 0
            })
        
        # Obtener métricas finales de los generadores
        generator_metrics_final = []
        for generator in generator_names:
            try:
                metrics = await emit_with_timeout(
                    engine,
                    "control_command",
                    {"command": "get_metrics"},
                    generator,
                    timeout=1.0
                )
                if metrics and len(metrics) > 0:
                    generator_metrics_final.append(metrics[0])
            except Exception:
                pass
        
        # Calcular estadísticas sobre distribución de carga
        distribution_stats = calculate_load_distribution(generator_metrics_baseline, generator_metrics_final)
        
        return {
            "baseline_result": baseline_result,
            "high_uniform_result": high_uniform_result,
            "mixed_results": mixed_results,
            "distribution_stats": distribution_stats
        }
    
    # Calcular estadísticas de distribución de carga
    def calculate_load_distribution(baseline_metrics, final_metrics):
        # Crear diccionario de métricas por tipo de carga
        metrics_by_type = {}
        
        # Inicializar estructura con los tipos de carga conocidos
        for load_type in ["cpu", "memory", "io", "mixed"]:
            metrics_by_type[load_type] = {
                "processing_counts": [],
                "processing_times": []
            }
        
        # Agrupar métricas finales por tipo
        for metric in final_metrics:
            component = metric.get("component", "")
            load_type = metric.get("load_type", "unknown")
            
            if load_type in metrics_by_type:
                metrics_by_type[load_type]["processing_counts"].append(metric.get("processing_count", 0))
                metrics_by_type[load_type]["processing_times"].append(metric.get("avg_process_time", 0))
        
        # Calcular desviación estándar de procesamiento por tipo
        distribution_stats = {}
        for load_type, data in metrics_by_type.items():
            counts = data["processing_counts"]
            times = data["processing_times"]
            
            if counts:
                avg_count = sum(counts) / len(counts)
                if len(counts) > 1:
                    std_count = (sum((c - avg_count) ** 2 for c in counts) / len(counts)) ** 0.5
                    variation_coeff = std_count / avg_count if avg_count > 0 else 0
                else:
                    std_count = 0
                    variation_coeff = 0
                
                distribution_stats[load_type] = {
                    "avg_count": avg_count,
                    "std_count": std_count,
                    "variation_coefficient": variation_coeff,
                    "avg_process_time": sum(times) / len(times) if times else 0,
                    "num_generators": len(counts)
                }
        
        # Calcular distribución general
        all_counts = sum((data["processing_counts"] for data in metrics_by_type.values()), [])
        if all_counts:
            avg_count = sum(all_counts) / len(all_counts)
            if len(all_counts) > 1:
                std_count = (sum((c - avg_count) ** 2 for c in all_counts) / len(all_counts)) ** 0.5
                variation_coeff = std_count / avg_count if avg_count > 0 else 0
            else:
                std_count = 0
                variation_coeff = 0
            
            distribution_stats["overall"] = {
                "avg_count": avg_count,
                "std_count": std_count,
                "variation_coefficient": variation_coeff,
                "num_generators": len(all_counts)
            }
        
        return distribution_stats
    
    # Ejecutar y medir la prueba completa
    result = await run_test_with_timing(engine, "test_concurrent_load_distribution", run_distribution_test)
    
    # Verificaciones
    assert isinstance(result, dict), "La prueba debería devolver un dict con resultados"
    
    # Verificar tasa de éxito en carga alta
    high_uniform = result.get("high_uniform_result", {})
    high_success_rate = high_uniform.get("success_count", 0) / high_uniform.get("total_events", 1)
    assert high_success_rate > 0.8, f"Baja tasa de éxito en carga alta: {high_success_rate:.2%}"
    
    # Verificar distribución de carga
    distribution_stats = result.get("distribution_stats", {})
    overall_stats = distribution_stats.get("overall", {})
    variation_coeff = overall_stats.get("variation_coefficient", 999)
    
    # Verificar que la carga esté razonablemente distribuida (coeficiente de variación menor a 0.5)
    assert variation_coeff < 0.5, f"Alta variación en distribución de carga: {variation_coeff:.2f}"
    
    logger.info(f"Tasa de éxito en carga alta: {high_success_rate:.2%}")
    logger.info(f"Coeficiente de variación en distribución de carga: {variation_coeff:.2f}")
    
    # Eventos por segundo
    events_per_second = high_uniform.get("events_per_second", 0)
    logger.info(f"Rendimiento en carga alta: {events_per_second:.2f} eventos/segundo")
    
    # Limpiar recursos
    await cleanup_engine(engine)


if __name__ == "__main__":
    # Para poder ejecutar este archivo directamente
    import pytest
    pytest.main(["-xvs", __file__])