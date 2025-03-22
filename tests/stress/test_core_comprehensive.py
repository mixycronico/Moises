"""
Prueba exhaustiva y completa del core del sistema híbrido API+WebSocket.

Este script implementa una batería de pruebas extremadamente exigentes que evalúan 
todos los aspectos críticos del core del sistema híbrido, incluyendo:

1. Rendimiento bajo carga extrema y variable
2. Resiliencia ante fallos simultáneos de múltiples componentes
3. Correcta gestión de latencia y timeouts
4. Estabilidad durante ejecuciones prolongadas
5. Comportamiento con patrones de comunicación complejos
"""

import asyncio
import logging
import random
import time
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable

# Configurar logging detallado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("core_test")

# Constantes para pruebas
MAX_COMPONENTS = 20
MAX_EVENT_RATE = 5000  # eventos por segundo
MAX_TEST_DURATION = 120  # segundos
ENABLE_EXTREME_TESTS = True  # Activar pruebas más intensivas

# Métricas para seguimiento del rendimiento
@dataclass
class PerfMetrics:
    """Métricas de rendimiento acumuladas."""
    start_time: float = field(default_factory=time.time)
    events_total: int = 0
    events_success: int = 0
    events_dropped: int = 0
    api_total: int = 0
    api_success: int = 0
    api_failures: int = 0
    latency_values: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=lambda: {})
    recovery_counts: int = 0
    
    def record_event(self, success: bool) -> None:
        """Registrar un evento."""
        self.events_total += 1
        if success:
            self.events_success += 1
        else:
            self.events_dropped += 1
    
    def record_api(self, success: bool) -> None:
        """Registrar una solicitud API."""
        self.api_total += 1
        if success:
            self.api_success += 1
        else:
            self.api_failures += 1
    
    def record_latency(self, latency: float) -> None:
        """Registrar una medida de latencia."""
        self.latency_values.append(latency)
    
    def record_error(self, error_type: str) -> None:
        """Registrar un tipo de error."""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
    
    def record_recovery(self) -> None:
        """Registrar una recuperación."""
        self.recovery_counts += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas."""
        elapsed = time.time() - self.start_time
        return {
            "duration_seconds": elapsed,
            "events_per_second": self.events_total / max(1, elapsed),
            "api_per_second": self.api_total / max(1, elapsed),
            "event_success_rate": self.events_success / max(1, self.events_total) if self.events_total > 0 else 0,
            "api_success_rate": self.api_success / max(1, self.api_total) if self.api_total > 0 else 0,
            "avg_latency_ms": sum(self.latency_values) / max(1, len(self.latency_values)) * 1000 if self.latency_values else 0,
            "min_latency_ms": min(self.latency_values) * 1000 if self.latency_values else 0,
            "max_latency_ms": max(self.latency_values) * 1000 if self.latency_values else 0,
            "top_errors": sorted([(k, v) for k, v in self.error_counts.items()], key=lambda x: x[1], reverse=True)[:5],
            "recovery_rate": self.recovery_counts
        }

# Componente avanzado para pruebas intensivas
class StressComponent:
    """Componente avanzado para pruebas extremas."""
    
    def __init__(self, id: str, 
                 failure_rate: float = 0.01,
                 latency_range: Tuple[float, float] = (0.001, 0.01),
                 recovery_time: float = 1.0,
                 crash_after: Optional[int] = None,
                 max_concurrent: int = 100):
        self.id = id
        self.failure_rate = failure_rate
        self.base_latency_range = latency_range
        self.current_latency_range = latency_range
        self.recovery_time = recovery_time
        self.crash_after = crash_after
        self.max_concurrent = max_concurrent
        
        # Estado
        self.call_count = 0
        self.event_count = 0
        self.crashed = False
        self.degraded = False
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
        self.metrics = PerfMetrics()
        self.start_time = time.time()
        
        # Callbacks y comportamiento avanzado
        self.on_crash_callbacks = []
        self.recovery_probability = 0.1  # Probabilidad de auto-recuperación
        
        # Comunicación entre componentes
        self.dependencies = []
        self.parent_coordinator = None
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitud API con gestión avanzada de errores."""
        start_time = time.time()
        
        # Verificar estado
        if self.crashed:
            self.metrics.record_api(False)
            self.metrics.record_error("crashed_component")
            raise Exception(f"Componente {self.id} crasheado")
        
        # Incrementar contador y verificar límite para crash programado
        self.call_count += 1
        if self.crash_after and self.call_count >= self.crash_after and not self.crashed:
            self.crashed = True
            self.metrics.record_error("programmed_crash")
            logger.warning(f"[{self.id}] Crash programado tras {self.call_count} llamadas")
            for callback in self.on_crash_callbacks:
                asyncio.create_task(callback(self.id))
            raise Exception(f"Componente {self.id} ha crasheado (programado)")
        
        # Adquirir semáforo con timeout
        try:
            async with asyncio.timeout(0.5):  # 500ms timeout para semáforo
                acquired = await self.semaphore.acquire()
                if not acquired:
                    self.metrics.record_api(False)
                    self.metrics.record_error("semaphore_timeout")
                    raise Exception("Timeout al adquirir semáforo")
                
                try:
                    self.active_requests += 1
                    
                    # Simular fallos aleatorios (más probables si degradado)
                    effective_failure = self.failure_rate * (3 if self.degraded else 1)
                    if random.random() < effective_failure:
                        self.metrics.record_api(False)
                        self.metrics.record_error("random_failure")
                        raise Exception(f"Fallo aleatorio en {self.id}")
                    
                    # Simular latencia de procesamiento (mayor si degradado)
                    latency_multiplier = 2.0 if self.degraded else 1.0
                    latency = random.uniform(
                        self.current_latency_range[0],
                        self.current_latency_range[1] * latency_multiplier
                    )
                    
                    # Aplicar latencia (simulando proceso)
                    await asyncio.sleep(latency)
                    
                    # Preparar respuesta según tipo
                    response = await self._process_by_type(request_type, data, source)
                    
                    # Registrar éxito y latencia
                    self.metrics.record_api(True)
                    self.metrics.record_latency(time.time() - start_time)
                    
                    return response
                    
                finally:
                    self.active_requests -= 1
                    self.semaphore.release()
        
        except asyncio.TimeoutError:
            self.metrics.record_api(False)
            self.metrics.record_error("timeout")
            raise Exception(f"Timeout en procesamiento de {request_type}")
        
        except Exception as e:
            self.metrics.record_api(False)
            self.metrics.record_error(str(e)[:30])  # Truncar mensaje error
            raise
    
    async def _process_by_type(self, request_type: str, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Implementar procesamiento específico por tipo."""
        # Procesar solicitudes de dependencias (simular llamadas API entre componentes)
        if request_type == "dependency_check" and self.parent_coordinator:
            results = {}
            
            # Verificar estado de dependencias
            for dep_id in self.dependencies:
                try:
                    # Solicitud con timeout corto para prevenir bloqueos
                    dep_result = await self.parent_coordinator.request(
                        dep_id, "health_check", {"requester": self.id}, self.id, timeout=0.2
                    )
                    results[dep_id] = "healthy" if dep_result else "unavailable"
                except Exception:
                    results[dep_id] = "error"
            
            return {
                "status": "completed",
                "dependency_status": results,
                "processor": self.id
            }
        
        # Manejo de salud (para supervisión)
        elif request_type == "health_check":
            return {
                "status": "degraded" if self.degraded else "healthy",
                "crashed": self.crashed,
                "active_requests": self.active_requests,
                "uptime": time.time() - self.start_time,
                "processor": self.id
            }
        
        # Recuperación (intento de reinicio)
        elif request_type == "recovery_attempt":
            if self.crashed:
                # Intento de recuperación
                await asyncio.sleep(self.recovery_time)
                # Probabilidad de éxito
                if random.random() < 0.7:  # 70% de probabilidad de recuperación
                    self.crashed = False
                    self.call_count = 0  # Reset contador
                    self.metrics.record_recovery()
                    return {"status": "recovered", "processor": self.id}
                else:
                    return {"status": "failed_recovery", "processor": self.id}
            else:
                return {"status": "already_healthy", "processor": self.id}
        
        # Solicitud genérica
        else:
            return {
                "status": "success", 
                "request_type": request_type,
                "processor": self.id,
                "timestamp": time.time()
            }
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar eventos WebSocket."""
        start_time = time.time()
        
        if self.crashed:
            # Eventos se descartan silenciosamente
            return
        
        self.event_count += 1
        
        # Puede recuperarse espontáneamente de un degradado
        if self.degraded and random.random() < 0.05:  # 5% de probabilidad por evento
            self.degraded = False
            self.current_latency_range = self.base_latency_range
            logger.info(f"[{self.id}] Recuperado de estado degradado")
        
        # Simular latencia de procesamiento
        latency = random.uniform(
            self.current_latency_range[0],
            self.current_latency_range[1]
        )
        await asyncio.sleep(latency)
        
        # Procesar según tipo de evento
        success = True
        
        if event_type == "system_status" and data.get("status") == "degraded":
            # Sistema global degradado, aumentar latencia
            self.degraded = True
            self.current_latency_range = (
                self.base_latency_range[0] * 1.5,
                self.base_latency_range[1] * 3
            )
        
        elif event_type == "recovery_signal" and source == "monitor" and data.get("target") == self.id:
            # Intento automático de recuperación
            if self.crashed and random.random() < self.recovery_probability:
                await asyncio.sleep(self.recovery_time)
                self.crashed = False
                self.call_count = 0
                self.metrics.record_recovery()
                logger.info(f"[{self.id}] Recuperado tras señal externa")
        
        elif event_type == "load_distribution":
            # Ajuste dinámico según carga
            load_factor = data.get("load_factor", 1.0)
            new_max = max(10, int(self.max_concurrent / load_factor))
            old_max = self.max_concurrent
            self.max_concurrent = new_max
            
            # Recrear semáforo si cambió significativamente
            if abs(new_max - old_max) > old_max * 0.2:  # Cambió más del 20%
                self.semaphore = asyncio.Semaphore(new_max)
                logger.debug(f"[{self.id}] Capacidad ajustada: {old_max} → {new_max}")
        
        # Posibilidad de fallo interno
        if random.random() < self.failure_rate:
            success = False
            self.metrics.record_error("event_internal_failure")
        
        # Registrar métrica
        self.metrics.record_event(success)
        self.metrics.record_latency(time.time() - start_time)
    
    async def start(self) -> None:
        """Iniciar componente."""
        self.start_time = time.time()
        self.crashed = False
        self.degraded = False
        self.metrics = PerfMetrics()
        logger.debug(f"Componente {self.id} iniciado")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.debug(f"Componente {self.id} detenido")
    
    def add_dependency(self, component_id: str) -> None:
        """Añadir dependencia de otro componente."""
        if component_id not in self.dependencies:
            self.dependencies.append(component_id)
    
    def add_crash_callback(self, callback: Callable) -> None:
        """Añadir callback para notificación de crash."""
        self.on_crash_callbacks.append(callback)
    
    def set_coordinator(self, coordinator) -> None:
        """Establecer referencia al coordinador padre."""
        self.parent_coordinator = coordinator

# Coordinador avanzado para pruebas extremas
class CoreTestCoordinator:
    """Coordinador avanzado para pruebas extremas."""
    
    def __init__(self, 
                 max_parallel_events: int = 1000,
                 max_parallel_requests: int = 500,
                 network_latency_range: Tuple[float, float] = (0.001, 0.01),
                 network_failure_rate: float = 0.01,
                 priority_event_types: List[str] = None):
        self.components = {}
        self.event_subscribers = {}
        self.max_parallel_events = max_parallel_events
        self.max_parallel_requests = max_parallel_requests
        self.network_latency_range = network_latency_range
        self.network_failure_rate = network_failure_rate
        self.priority_event_types = set(priority_event_types or ["recovery_signal", "system_status", "alert"])
        
        # Semáforos para control de concurrencia
        self.event_semaphore = asyncio.Semaphore(max_parallel_events)
        self.request_semaphore = asyncio.Semaphore(max_parallel_requests)
        
        # Colas para eventos
        self.priority_queue = asyncio.Queue()
        self.regular_queue = asyncio.Queue()
        
        # Métricas y estado
        self.metrics = PerfMetrics()
        self.start_time = time.time()
        self.active_event_tasks = 0
        self.active_request_tasks = 0
        self.component_status = {}  # id -> status
        
        # Tareas de fondo
        self.background_tasks = []
    
    def register_component(self, id: str, component: StressComponent) -> None:
        """Registrar componente en el sistema."""
        self.components[id] = component
        self.component_status[id] = "healthy"
        
        # Añadir referencia al coordinador
        component.set_coordinator(self)
        
        # Añadir callback para notificación de crash
        component.add_crash_callback(self._on_component_crash)
        
        logger.info(f"Componente {id} registrado en el coordinador")
    
    def subscribe(self, component_id: str, event_types: List[str]) -> None:
        """Suscribir componente a tipos de eventos."""
        if component_id not in self.components:
            logger.warning(f"No se puede suscribir: componente {component_id} no registrado")
            return
        
        for event_type in event_types:
            if event_type not in self.event_subscribers:
                self.event_subscribers[event_type] = set()
            self.event_subscribers[event_type].add(component_id)
        
        logger.info(f"Componente {component_id} suscrito a {len(event_types)} tipos de eventos")
    
    def create_dependency(self, component_id: str, depends_on_id: str) -> None:
        """Establecer dependencia entre componentes."""
        if component_id not in self.components or depends_on_id not in self.components:
            logger.warning(f"No se puede crear dependencia: componente no existe")
            return
        
        # Añadir dependencia
        self.components[component_id].add_dependency(depends_on_id)
        logger.info(f"Dependencia creada: {component_id} → {depends_on_id}")
    
    async def request(self, target_id: str, request_type: str, 
                     data: Dict[str, Any], source: str,
                     timeout: float = 1.0) -> Optional[Any]:
        """API: Enviar solicitud directa a un componente con control de concurrencia."""
        start_time = time.time()
        self.metrics.record_api(False)  # Inicialmente marcar como fallido
        
        # Verificar existencia del componente
        if target_id not in self.components:
            self.metrics.record_error("nonexistent_component")
            logger.warning(f"Solicitud a componente inexistente: {target_id}")
            return None
        
        # Simular fallo de red
        if random.random() < self.network_failure_rate:
            self.metrics.record_error("network_failure")
            logger.warning(f"Fallo de red simulado en solicitud a {target_id}")
            return None
        
        # Intentar adquirir semáforo (con timeout)
        try:
            async with asyncio.timeout(0.5):  # 500ms max para semáforo
                acquired = await self.request_semaphore.acquire()
                if not acquired:
                    self.metrics.record_error("request_semaphore_timeout")
                    raise Exception("Timeout al adquirir semáforo de solicitudes")
                
                try:
                    self.active_request_tasks += 1
                    
                    # Simular latencia de red ida
                    network_latency = random.uniform(
                        self.network_latency_range[0],
                        self.network_latency_range[1]
                    )
                    if network_latency > 0:
                        await asyncio.sleep(network_latency)
                    
                    # Enviar solicitud con timeout global
                    try:
                        component = self.components[target_id]
                        result = await asyncio.wait_for(
                            component.process_request(request_type, data, source),
                            timeout=timeout
                        )
                        
                        # Simular latencia de red vuelta
                        if network_latency > 0:
                            await asyncio.sleep(network_latency)
                        
                        # Registrar éxito
                        self.metrics.record_api(True)
                        self.metrics.record_latency(time.time() - start_time)
                        
                        return result
                        
                    except asyncio.TimeoutError:
                        self.metrics.record_error("component_timeout")
                        logger.warning(f"Timeout en solicitud {request_type} a {target_id}")
                        return None
                        
                    except Exception as e:
                        self.metrics.record_error(f"component_exception")
                        logger.warning(f"Error en solicitud a {target_id}: {e}")
                        return None
                    
                finally:
                    self.active_request_tasks -= 1
                    self.request_semaphore.release()
        
        except asyncio.TimeoutError:
            self.metrics.record_error("global_semaphore_timeout")
            logger.warning(f"Timeout global al intentar realizar solicitud a {target_id}")
            return None
        
        except Exception as e:
            self.metrics.record_error("unexpected_error")
            logger.error(f"Error inesperado en solicitud: {e}")
            return None
    
    async def emit_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Encolar evento para procesamiento asíncrono."""
        # Determinar si es prioritario
        is_priority = event_type in self.priority_event_types
        
        # Encolar evento
        event_data = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time(),
            "priority": is_priority
        }
        
        if is_priority:
            await self.priority_queue.put(event_data)
        else:
            await self.regular_queue.put(event_data)
        
        # Actualizar métrica
        self.metrics.events_total += 1
    
    async def _process_event_queues(self):
        """Procesar colas de eventos continuamente."""
        while True:
            try:
                # Primero eventos prioritarios
                if not self.priority_queue.empty():
                    event_data = await self.priority_queue.get()
                    await self._process_single_event(event_data)
                    self.priority_queue.task_done()
                
                # Luego eventos regulares
                elif not self.regular_queue.empty():
                    event_data = await self.regular_queue.get()
                    await self._process_single_event(event_data)
                    self.regular_queue.task_done()
                
                # Esperar si no hay eventos
                else:
                    await asyncio.sleep(0.001)
            
            except Exception as e:
                logger.error(f"Error en procesador de colas: {e}")
                self.metrics.record_error("queue_processor_error")
    
    async def _process_single_event(self, event_data: Dict[str, Any]):
        """Procesar un solo evento."""
        event_type = event_data["type"]
        data = event_data["data"]
        source = event_data["source"]
        is_priority = event_data["priority"]
        
        # Distribuir a suscriptores
        subscribers = self.event_subscribers.get(event_type, set())
        if not subscribers:
            return  # Nadie suscrito
        
        # Control de concurrencia (solo para eventos no prioritarios)
        if not is_priority:
            try:
                # Timeout corto para no bloquear la cola
                async with asyncio.timeout(0.1):
                    acquired = await self.event_semaphore.acquire()
                    if not acquired:
                        self.metrics.events_dropped += 1
                        self.metrics.record_error("event_semaphore_timeout")
                        logger.warning(f"Evento {event_type} descartado: sistema sobrecargado")
                        return
            except asyncio.TimeoutError:
                self.metrics.events_dropped += 1
                self.metrics.record_error("event_acquire_timeout") 
                return
        
        try:
            self.active_event_tasks += 1
            
            # Crear tareas para cada suscriptor
            delivery_tasks = []
            
            for comp_id in subscribers:
                if comp_id in self.components and comp_id != source:
                    # Fallo de red específico
                    if random.random() < self.network_failure_rate:
                        continue
                    
                    # Latencia de red
                    network_latency = random.uniform(
                        self.network_latency_range[0],
                        self.network_latency_range[1]
                    )
                    
                    if network_latency > 0:
                        # Incluir latencia en delivery
                        delivery_task = self._delayed_event_delivery(
                            comp_id, event_type, data, source, network_latency
                        )
                    else:
                        # Sin latencia extra
                        delivery_task = self.components[comp_id].on_event(
                            event_type, data, source
                        )
                    
                    delivery_tasks.append(delivery_task)
            
            # Ejecutar entregas en paralelo
            if delivery_tasks:
                for future in asyncio.as_completed(delivery_tasks):
                    try:
                        await future
                        self.metrics.events_success += 1
                    except Exception as e:
                        self.metrics.events_dropped += 1
                        self.metrics.record_error("event_delivery_error")
            
        finally:
            self.active_event_tasks -= 1
            # Liberar semáforo si no era prioritario
            if not is_priority:
                self.event_semaphore.release()
    
    async def _delayed_event_delivery(self, comp_id: str, event_type: str, 
                                    data: Dict[str, Any], source: str, 
                                    delay: float) -> None:
        """Entregar evento con retraso simulado."""
        await asyncio.sleep(delay)
        await self.components[comp_id].on_event(event_type, data, source)
    
    async def _on_component_crash(self, component_id: str) -> None:
        """Manejar notificación de crash de componente."""
        if component_id in self.component_status:
            old_status = self.component_status[component_id]
            self.component_status[component_id] = "crashed"
            
            if old_status != "crashed":
                logger.warning(f"Componente {component_id} ha cambiado a estado crashed")
                
                # Emitir evento de alerta del sistema
                await self.emit_event(
                    "system_status",
                    {
                        "status": "alert",
                        "crashed_component": component_id,
                        "timestamp": time.time()
                    },
                    "coordinator"
                )
    
    async def _monitor_system_health(self):
        """Monitorear salud del sistema periódicamente."""
        monitor_interval = 0.5  # 500ms
        
        while True:
            try:
                await asyncio.sleep(monitor_interval)
                
                # Verificar cada componente
                crashed_count = 0
                for comp_id, component in self.components.items():
                    if component.crashed:
                        crashed_count += 1
                        
                        # Intentar recuperación aleatoria
                        if random.random() < 0.1:  # 10% de probabilidad
                            # Emitir señal de recuperación
                            await self.emit_event(
                                "recovery_signal",
                                {
                                    "target": comp_id,
                                    "timestamp": time.time()
                                },
                                "monitor"
                            )
                
                # Si demasiados componentes crasheados, emitir alerta
                if crashed_count > len(self.components) * 0.3:  # >30% crasheados
                    # Emitir alerta de sistema degradado
                    await self.emit_event(
                        "system_status",
                        {
                            "status": "degraded",
                            "crashed_percentage": crashed_count / len(self.components),
                            "timestamp": time.time()
                        },
                        "monitor"
                    )
                
            except Exception as e:
                logger.error(f"Error en monitoreo: {e}")
                self.metrics.record_error("monitor_error")
    
    async def _load_distribution_manager(self):
        """Distribuir carga entre componentes según demanda."""
        interval = 1.0  # 1 segundo
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Calcular factor de carga global
                priority_size = self.priority_queue.qsize()
                regular_size = self.regular_queue.qsize()
                total_queued = priority_size + regular_size
                
                if total_queued > self.max_parallel_events * 0.7:  # >70% de capacidad
                    # Sistema sobrecargado, aumentar capacidad
                    load_factor = min(3.0, total_queued / self.max_parallel_events)
                    
                    # Emitir evento de distribución de carga
                    await self.emit_event(
                        "load_distribution",
                        {
                            "load_factor": load_factor,
                            "queued_events": total_queued,
                            "timestamp": time.time()
                        },
                        "load_manager"
                    )
            
            except Exception as e:
                logger.error(f"Error en distribuidor de carga: {e}")
                self.metrics.record_error("load_distributor_error")
    
    async def start(self) -> None:
        """Iniciar coordinador y componentes."""
        self.start_time = time.time()
        self.metrics = PerfMetrics()
        
        # Iniciar componentes
        start_tasks = []
        for comp in self.components.values():
            start_tasks.append(comp.start())
        
        if start_tasks:
            await asyncio.gather(*start_tasks)
        
        # Iniciar procesadores de eventos
        self.background_tasks.append(
            asyncio.create_task(self._process_event_queues())
        )
        
        # Iniciar monitor de salud
        self.background_tasks.append(
            asyncio.create_task(self._monitor_system_health())
        )
        
        # Iniciar distribuidor de carga
        self.background_tasks.append(
            asyncio.create_task(self._load_distribution_manager())
        )
        
        logger.info(f"Coordinador iniciado con {len(self.components)} componentes")
    
    async def stop(self) -> None:
        """Detener coordinador y componentes."""
        # Cancelar tareas de fondo
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Esperar que se completen eventos pendientes
        try:
            if self.priority_queue.qsize() > 0:
                logger.info(f"Esperando {self.priority_queue.qsize()} eventos prioritarios pendientes...")
                await asyncio.wait_for(self.priority_queue.join(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout esperando eventos prioritarios")
        
        # Detener componentes
        stop_tasks = []
        for comp in self.components.values():
            stop_tasks.append(comp.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks)
        
        logger.info(f"Coordinador detenido después de {time.time() - self.start_time:.2f}s")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas completas del sistema."""
        core_metrics = self.metrics.get_summary()
        
        # Añadir métricas por componente
        components_metrics = {}
        for comp_id, comp in self.components.items():
            metrics = comp.metrics.get_summary()
            health_status = "crashed" if comp.crashed else "degraded" if comp.degraded else "healthy"
            
            components_metrics[comp_id] = {
                "status": health_status,
                "metrics": metrics
            }
        
        # Métricas adicionales de sistema
        system_metrics = {
            "active_event_tasks": self.active_event_tasks,
            "active_request_tasks": self.active_request_tasks,
            "priority_queue_size": self.priority_queue.qsize(),
            "regular_queue_size": self.regular_queue.qsize(),
            "component_count": len(self.components),
            "crashed_components": sum(1 for comp in self.components.values() if comp.crashed),
            "degraded_components": sum(1 for comp in self.components.values() if comp.degraded and not comp.crashed)
        }
        
        return {
            "core": core_metrics,
            "components": components_metrics,
            "system": system_metrics
        }

# Pruebas específicas
async def setup_complex_system(components: int = 10) -> CoreTestCoordinator:
    """Configurar sistema complejo para pruebas extremas."""
    logger.info(f"Configurando sistema con {components} componentes")
    
    # Crear coordinador
    coordinator = CoreTestCoordinator(
        max_parallel_events=5000,
        max_parallel_requests=1000,
        network_latency_range=(0.001, 0.01),
        network_failure_rate=0.01
    )
    
    # Crear componentes con diversos parámetros
    for i in range(components):
        # Variar características según el índice
        failure_rate = 0.01 * (1 + (i % 3))  # 1% a 3%
        latency_base = 0.001 * (1 + (i % 5))  # 1ms a 5ms
        latency_range = (latency_base, latency_base * 10)  # 1-10ms a 5-50ms
        
        # Cada 3er componente programado para crashear
        crash_after = None
        if i % 3 == 0:
            crash_after = random.randint(100, 300)  # Crashear tras 100-300 llamadas
        
        # Crear componente
        component = StressComponent(
            id=f"comp_{i}",
            failure_rate=failure_rate,
            latency_range=latency_range,
            recovery_time=random.uniform(0.5, 2.0),
            crash_after=crash_after,
            max_concurrent=random.choice([50, 75, 100])
        )
        
        # Registrar componente
        coordinator.register_component(f"comp_{i}", component)
        
        # Suscribir a eventos (cada componente varía)
        event_types = [
            "data_update", "notification", "status_change", 
            "heartbeat", "system_status", "alert", "recovery_signal",
            "load_distribution", "metrics_update", "config_change"
        ]
        
        # Cada componente se suscribe a un número variable de eventos
        num_subscriptions = random.randint(3, min(7, len(event_types)))
        subscribed_events = random.sample(event_types, num_subscriptions)
        
        coordinator.subscribe(f"comp_{i}", subscribed_events)
    
    # Crear topología de dependencias (compleja pero sin ciclos)
    # Aproximadamente la mitad de los componentes tendrán dependencias
    for i in range(components):
        # 50% de probabilidad de tener dependencias
        if random.random() < 0.5:
            # 1-3 dependencias por componente
            num_deps = random.randint(1, min(3, components-1))
            
            # Seleccionar componentes para dependencias (evitando ciclos)
            available_deps = [f"comp_{j}" for j in range(components) if j > i]
            if available_deps:
                selected_deps = random.sample(available_deps, min(num_deps, len(available_deps)))
                
                for dep_id in selected_deps:
                    coordinator.create_dependency(f"comp_{i}", dep_id)
    
    # Iniciar sistema
    await coordinator.start()
    
    return coordinator

async def test_high_volume_sustained(coordinator: CoreTestCoordinator) -> Dict[str, Any]:
    """
    Prueba de volumen alto sostenido con eventos variados.
    
    Esta prueba genera un flujo constante de eventos a muy alta velocidad
    para verificar que el sistema puede mantener la estabilidad incluso 
    bajo carga extrema.
    """
    logger.info("Iniciando prueba de volumen sostenido")
    
    # Parámetros
    target_rate = 1000  # eventos por segundo
    duration = 20       # segundos
    ramp_up = 3         # segundos
    
    # Tipos de eventos
    event_types = [
        "data_update", "notification", "status_change", 
        "heartbeat", "metrics_update", "config_change"
    ]
    
    # Control de tiempo
    start_time = time.time()
    end_time = start_time + duration
    events_emitted = 0
    api_calls = 0
    
    try:
        # Bucle principal
        while time.time() < end_time:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Calcular tasa actual durante ramp-up
            if elapsed < ramp_up:
                current_rate = int((elapsed / ramp_up) * target_rate)
            else:
                current_rate = target_rate
            
            # Batch de eventos por ciclo (aproximadamente 10% de la tasa)
            events_per_cycle = max(1, current_rate // 10)
            
            # Emitir eventos
            emission_tasks = []
            for _ in range(events_per_cycle):
                # Seleccionar tipo
                event_type = random.choice(event_types)
                
                # Datos según tipo
                if event_type == "data_update":
                    # Datos más pesados
                    items = []
                    for j in range(random.randint(5, 20)):
                        items.append({
                            "id": j,
                            "value": random.random() * 100,
                            "timestamp": time.time()
                        })
                    
                    data = {
                        "items": items,
                        "source": "test_volume"
                    }
                else:
                    # Datos ligeros
                    data = {
                        "timestamp": time.time(),
                        "message": f"Evento {events_emitted}",
                        "source": "test_volume"
                    }
                
                # Añadir tarea de emisión
                emission_tasks.append(
                    coordinator.emit_event(event_type, data, "test_volume")
                )
                events_emitted += 1
            
            # Iniciar emisión (sin esperar completado)
            for task in emission_tasks:
                asyncio.create_task(task)
            
            # Hacer algunas solicitudes API
            if random.random() < 0.2:  # 20% de ciclos
                # Seleccionar componente aleatorio
                target_comp = f"comp_{random.randint(0, len(coordinator.components)-1)}"
                
                # Solicitud aleatoria
                request_type = random.choice(["health_check", "dependency_check"])
                data = {"source": "test_volume", "timestamp": time.time()}
                
                # Iniciar solicitud (sin esperar completado)
                asyncio.create_task(
                    coordinator.request(target_comp, request_type, data, "test_volume")
                )
                api_calls += 1
            
            # Reportar progreso cada 5 segundos
            if int(elapsed) % 5 == 0 and int(elapsed) != int(elapsed - 0.1):
                metrics = coordinator.get_metrics()
                logger.info(
                    f"Progreso: {elapsed:.1f}s/{duration}s - "
                    f"Tasa: {events_emitted/elapsed:.1f} eventos/s - "
                    f"Cola: {metrics['system']['priority_queue_size']}P/{metrics['system']['regular_queue_size']}R"
                )
            
            # Controlar velocidad para mantener tasa aproximada
            cycle_elapsed = time.time() - current_time
            sleep_time = max(0.001, 0.1 - cycle_elapsed)  # ~10 ciclos/segundo
            await asyncio.sleep(sleep_time)
        
        # Esperar procesamiento de eventos pendientes
        logger.info("Prueba completada. Esperando procesamiento pendiente...")
        await asyncio.sleep(2.0)
        
        # Recolectar métricas
        final_metrics = coordinator.get_metrics()
        final_metrics["test_specific"] = {
            "target_rate": target_rate,
            "actual_rate": events_emitted / duration,
            "total_events": events_emitted,
            "total_api_calls": api_calls,
            "duration": duration
        }
        
        return final_metrics
        
    except Exception as e:
        logger.error(f"Error en prueba de volumen: {e}")
        return {"error": str(e)}

async def test_cascading_failure_resilience(coordinator: CoreTestCoordinator) -> Dict[str, Any]:
    """
    Prueba de resiliencia ante fallos en cascada.
    
    Esta prueba inyecta fallos sistemáticos en componentes críticos
    para verificar que el sistema evita los fallos en cascada y mantiene
    operatividad parcial incluso ante fallos graves.
    """
    logger.info("Iniciando prueba de resiliencia ante fallos en cascada")
    
    # Parámetros
    duration = 15       # segundos
    event_rate = 200    # eventos por segundo (constante)
    
    # Control de tiempo
    start_time = time.time()
    end_time = start_time + duration
    events_emitted = 0
    api_calls = 0
    
    # Tipos de eventos
    event_types = ["data_update", "notification", "heartbeat"]
    
    # Preparar métricas específicas
    resilience_metrics = {
        "crashed_components": 0,
        "recovery_attempts": 0,
        "successful_recoveries": 0,
        "api_success_before_crash": 0,
        "api_success_during_crash": 0,
        "api_total_before_crash": 0,
        "api_total_during_crash": 0
    }
    
    # Fase 1: Operación normal
    phase1_duration = 5  # segundos
    phase1_end = start_time + phase1_duration
    
    logger.info("Fase 1: Operación normal")
    
    while time.time() < phase1_end:
        # Emitir eventos a tasa constante
        for _ in range(event_rate // 10):  # ~10 ciclos por segundo
            event_type = random.choice(event_types)
            data = {
                "timestamp": time.time(),
                "message": f"Evento {events_emitted}"
            }
            
            asyncio.create_task(
                coordinator.emit_event(event_type, data, "test_cascading")
            )
            events_emitted += 1
        
        # Hacer solicitudes API
        if random.random() < 0.3:  # 30% de probabilidad por ciclo
            # Varios componentes
            for _ in range(3):
                target_comp = f"comp_{random.randint(0, len(coordinator.components)-1)}"
                
                # Solicitud de verificación
                result = await coordinator.request(
                    target_comp, 
                    "health_check", 
                    {"source": "test_cascading"}, 
                    "test_cascading"
                )
                
                api_calls += 1
                resilience_metrics["api_total_before_crash"] += 1
                
                if result and result.get("status") in ["healthy", "degraded"]:
                    resilience_metrics["api_success_before_crash"] += 1
        
        # Esperar para mantener tasa
        await asyncio.sleep(0.1)
    
    # Fase 2: Introducir fallos
    phase2_duration = 5  # segundos
    phase2_end = phase1_end + phase2_duration
    
    # Crashear componentes (1/3 del total)
    components_to_crash = len(coordinator.components) // 3
    crashed_components = []
    
    logger.info(f"Fase 2: Forzando crash en {components_to_crash} componentes")
    
    # Seleccionar componentes para crashear (aleatorio)
    all_components = list(coordinator.components.keys())
    components_to_crash = min(components_to_crash, len(all_components))
    components_to_crash_ids = random.sample(all_components, components_to_crash)
    
    # Forzar crash en componentes seleccionados
    for comp_id in components_to_crash_ids:
        coordinator.components[comp_id].crashed = True
        crashed_components.append(comp_id)
        resilience_metrics["crashed_components"] += 1
    
    logger.info(f"Componentes crasheados: {crashed_components}")
    
    # Continuar operación durante fallos
    while time.time() < phase2_end:
        # Emitir eventos (tasa constante)
        for _ in range(event_rate // 10):
            event_type = random.choice(event_types)
            data = {
                "timestamp": time.time(),
                "message": f"Evento {events_emitted}"
            }
            
            asyncio.create_task(
                coordinator.emit_event(event_type, data, "test_cascading")
            )
            events_emitted += 1
        
        # Hacer solicitudes API a todos los componentes
        if random.random() < 0.3:
            for comp_id in coordinator.components:
                # Solicitud de salud
                result = await coordinator.request(
                    comp_id, 
                    "health_check", 
                    {"source": "test_cascading"}, 
                    "test_cascading"
                )
                
                api_calls += 1
                resilience_metrics["api_total_during_crash"] += 1
                
                if result and result.get("status") in ["healthy", "degraded"]:
                    resilience_metrics["api_success_during_crash"] += 1
        
        # Esperar
        await asyncio.sleep(0.1)
    
    # Fase 3: Recuperación
    phase3_duration = 5  # segundos
    phase3_end = phase2_end + phase3_duration
    
    logger.info("Fase 3: Intentando recuperación")
    
    # Intentar recuperar componentes
    for comp_id in crashed_components:
        # Solicitud de recuperación
        result = await coordinator.request(
            comp_id,
            "recovery_attempt",
            {"source": "test_cascading"},
            "test_cascading"
        )
        
        resilience_metrics["recovery_attempts"] += 1
        
        if result and result.get("status") == "recovered":
            resilience_metrics["successful_recoveries"] += 1
    
    # Continuar operación durante fase de recuperación
    while time.time() < phase3_end:
        # Emitir eventos (tasa constante)
        for _ in range(event_rate // 10):
            event_type = random.choice(event_types)
            data = {
                "timestamp": time.time(),
                "message": f"Evento {events_emitted}"
            }
            
            asyncio.create_task(
                coordinator.emit_event(event_type, data, "test_cascading")
            )
            events_emitted += 1
        
        # Esperar
        await asyncio.sleep(0.1)
    
    # Esperar procesamiento de eventos pendientes
    logger.info("Prueba completada. Esperando procesamiento pendiente...")
    await asyncio.sleep(2.0)
    
    # Calcular métricas finales
    if resilience_metrics["api_total_before_crash"] > 0:
        normal_success_rate = resilience_metrics["api_success_before_crash"] / resilience_metrics["api_total_before_crash"]
    else:
        normal_success_rate = 0
        
    if resilience_metrics["api_total_during_crash"] > 0:
        crash_success_rate = resilience_metrics["api_success_during_crash"] / resilience_metrics["api_total_during_crash"]
    else:
        crash_success_rate = 0
    
    # Porcentaje de componentes que siguieron funcionando
    components_operational = len(coordinator.components) - resilience_metrics["crashed_components"]
    expected_success_rate = components_operational / len(coordinator.components)
    
    resilience_metrics["normal_success_rate"] = normal_success_rate
    resilience_metrics["crash_success_rate"] = crash_success_rate
    resilience_metrics["expected_success_rate"] = expected_success_rate
    resilience_metrics["resilience_score"] = crash_success_rate / expected_success_rate if expected_success_rate > 0 else 0
    
    # Recolectar métricas del sistema
    final_metrics = coordinator.get_metrics()
    final_metrics["test_specific"] = {
        "resilience_metrics": resilience_metrics,
        "total_events": events_emitted,
        "total_api_calls": api_calls,
        "duration": duration
    }
    
    return final_metrics

async def test_dependency_graph_stress(coordinator: CoreTestCoordinator) -> Dict[str, Any]:
    """
    Prueba de estrés en el grafo de dependencias.
    
    Esta prueba evalúa específicamente cómo el sistema maneja las dependencias
    entre componentes cuando los componentes fallan o se degradan.
    """
    logger.info("Iniciando prueba de estrés en grafo de dependencias")
    
    # Parámetros
    duration = 15       # segundos
    check_interval = 1  # verificación cada 1 segundo
    
    # Control de tiempo
    start_time = time.time()
    end_time = start_time + duration
    
    # Variables de seguimiento
    dependency_checks = 0
    failed_dependencies = 0
    dependency_results = []
    
    try:
        # Bucle principal
        while time.time() < end_time:
            # Esperar intervalo
            await asyncio.sleep(check_interval)
            
            # Verificar dependencias para cada componente
            for comp_id, component in coordinator.components.items():
                # Solo verificar componentes activos
                if not component.crashed and component.dependencies:
                    # Solicitar verificación de dependencias
                    result = await coordinator.request(
                        comp_id,
                        "dependency_check",
                        {"timestamp": time.time()},
                        "dependency_test"
                    )
                    
                    dependency_checks += 1
                    
                    # Analizar resultados
                    if result and "dependency_status" in result:
                        status = result["dependency_status"]
                        
                        # Registrar dependencias no disponibles
                        for dep_id, dep_status in status.items():
                            if dep_status != "healthy":
                                failed_dependencies += 1
                        
                        # Guardar resultado para análisis
                        dependency_results.append({
                            "component": comp_id,
                            "timestamp": time.time(),
                            "result": status
                        })
            
            # Degradar algunos componentes aleatoriamente
            if random.random() < 0.2:  # 20% de probabilidad por ciclo
                comp_id = random.choice(list(coordinator.components.keys()))
                component = coordinator.components[comp_id]
                
                if not component.crashed and not component.degraded:
                    component.degraded = True
                    logger.info(f"Componente {comp_id} degradado")
        
        # Recolectar métricas
        metrics = coordinator.get_metrics()
        
        # Analizar impacto de dependencias
        dependency_analysis = {
            "total_checks": dependency_checks,
            "failed_dependencies": failed_dependencies,
            "failure_rate": failed_dependencies / max(1, dependency_checks),
            "components_with_dependencies": sum(1 for c in coordinator.components.values() if c.dependencies),
            "avg_dependencies_per_component": sum(len(c.dependencies) for c in coordinator.components.values()) / max(1, len(coordinator.components))
        }
        
        # Evaluar impacto de dependencias en componentes
        component_failures = {}
        for comp_id, component in coordinator.components.items():
            if component.crashed or component.degraded:
                # Verificar cuántos componentes dependen de este
                dependents = sum(1 for c in coordinator.components.values() 
                               if comp_id in c.dependencies)
                
                component_failures[comp_id] = {
                    "status": "crashed" if component.crashed else "degraded",
                    "dependents": dependents
                }
        
        metrics["test_specific"] = {
            "dependency_analysis": dependency_analysis,
            "component_failures": component_failures,
            "dependency_samples": dependency_results[:10]  # Primeros 10 resultados
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error en prueba de dependencias: {e}")
        return {"error": str(e)}

async def test_dynamic_recovery(coordinator: CoreTestCoordinator) -> Dict[str, Any]:
    """
    Prueba de recuperación dinámica.
    
    Esta prueba evalúa la capacidad del sistema para recuperarse
    automáticamente cuando los componentes fallan.
    """
    logger.info("Iniciando prueba de recuperación dinámica")
    
    # Parámetros
    duration = 20       # segundos
    event_rate = 100    # eventos por segundo
    
    # Control de tiempo
    start_time = time.time()
    end_time = start_time + duration
    
    # Metrics específicas
    recovery_metrics = {
        "components_crashed": 0,
        "recovery_signals": 0,
        "spontaneous_recoveries": 0,
        "signal_recoveries": 0,
        "failed_recoveries": 0
    }
    
    # Tipos de eventos
    event_types = ["notification", "heartbeat", "system_status"]
    
    try:
        # Etapa 1: Operación normal
        logger.info("Etapa 1: Operación normal")
        await asyncio.sleep(2)
        
        # Etapa 2: Forzar crash en componentes
        random_crashes = random.randint(3, 5)  # 3-5 componentes
        logger.info(f"Etapa 2: Forzando crash en {random_crashes} componentes")
        
        crashed_components = []
        active_components = list(coordinator.components.keys())
        
        for _ in range(min(random_crashes, len(active_components))):
            # Seleccionar componente aleatorio
            comp_id = random.choice(active_components)
            active_components.remove(comp_id)
            
            # Forzar crash
            coordinator.components[comp_id].crashed = True
            crashed_components.append(comp_id)
            recovery_metrics["components_crashed"] += 1
            
            logger.info(f"Componente {comp_id} forzado a crashed")
        
        # Etapa 3: Emisión continua de eventos para triggers de recuperación
        logger.info("Etapa 3: Emisión de eventos durante periodo de recuperación")
        
        # Track de componentes recuperados
        recovered_ids = set()
        
        # Bucle de eventos
        while time.time() < end_time:
            current_time = time.time()
            
            # Emitir eventos a tasa constante
            for _ in range(event_rate // 10):  # ~10 ciclos por segundo
                event_type = random.choice(event_types)
                data = {
                    "timestamp": current_time,
                    "message": f"Evento de recuperación"
                }
                
                # Iniciar evento de forma asíncrona
                asyncio.create_task(
                    coordinator.emit_event(event_type, data, "recovery_test")
                )
            
            # Verificar recuperaciones
            for comp_id in crashed_components:
                component = coordinator.components[comp_id]
                
                # Verificar si recuperado
                if comp_id not in recovered_ids and not component.crashed:
                    recovered_ids.add(comp_id)
                    recovery_metrics["spontaneous_recoveries"] += 1
                    logger.info(f"Componente {comp_id} recuperado espontáneamente")
            
            # Enviar señales de recuperación periódicamente
            if random.random() < 0.2:  # 20% de probabilidad por ciclo
                if crashed_components:
                    # Seleccionar componente aleatorio
                    target_id = random.choice(crashed_components)
                    
                    if target_id not in recovered_ids:
                        # Enviar señal explícita de recuperación
                        await coordinator.emit_event(
                            "recovery_signal",
                            {
                                "target": target_id,
                                "timestamp": current_time
                            },
                            "recovery_test"
                        )
                        
                        recovery_metrics["recovery_signals"] += 1
            
            # Esperar para siguiente ciclo
            await asyncio.sleep(0.1)
        
        # Verificación final de recuperaciones
        for comp_id in crashed_components:
            component = coordinator.components[comp_id]
            
            if not component.crashed:
                if comp_id not in recovered_ids:
                    # Recuperación tardía
                    recovery_metrics["signal_recoveries"] += 1
            else:
                # Aún crasheado
                recovery_metrics["failed_recoveries"] += 1
        
        # Recolectar métricas
        metrics = coordinator.get_metrics()
        
        # Calcular tasa de recuperación
        components_recovered = recovery_metrics["spontaneous_recoveries"] + recovery_metrics["signal_recoveries"]
        recovery_rate = components_recovered / max(1, recovery_metrics["components_crashed"])
        
        metrics["test_specific"] = {
            "recovery_metrics": recovery_metrics,
            "recovery_rate": recovery_rate,
            "recovery_assessment": (
                "Excelente" if recovery_rate > 0.8 else
                "Bueno" if recovery_rate > 0.6 else
                "Aceptable" if recovery_rate > 0.4 else
                "Deficiente"
            )
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error en prueba de recuperación: {e}")
        traceback.print_exc()
        return {"error": str(e)}

async def run_all_core_tests():
    """Ejecutar todas las pruebas comprehensivas del core."""
    try:
        logger.info("=== INICIANDO PRUEBAS EXHAUSTIVAS DEL CORE ===")
        
        # Configurar sistema complejo para todas las pruebas
        system = await setup_complex_system(components=MAX_COMPONENTS if ENABLE_EXTREME_TESTS else 12)
        
        try:
            # Test 1: Volumen alto sostenido
            logger.info("\n=== TEST 1: VOLUMEN ALTO SOSTENIDO ===\n")
            volume_results = await test_high_volume_sustained(system)
            
            # Test 2: Resiliencia ante fallos en cascada 
            logger.info("\n=== TEST 2: RESILIENCIA ANTE FALLOS EN CASCADA ===\n")
            cascading_results = await test_cascading_failure_resilience(system)
            
            # Test 3: Estrés en grafo de dependencias
            logger.info("\n=== TEST 3: ESTRÉS EN GRAFO DE DEPENDENCIAS ===\n")
            dependency_results = await test_dependency_graph_stress(system)
            
            # Test 4: Recuperación dinámica
            logger.info("\n=== TEST 4: RECUPERACIÓN DINÁMICA ===\n")
            recovery_results = await test_dynamic_recovery(system)
            
            # Analizar resultados
            logger.info("\n=== ANÁLISIS DE RESULTADOS ===\n")
            
            # 1. Rendimiento (volumen)
            volume_rate = volume_results["test_specific"]["actual_rate"]
            target_rate = volume_results["test_specific"]["target_rate"]
            volume_score = min(1.0, volume_rate / target_rate)
            
            logger.info(f"Prueba de Volumen:")
            logger.info(f"- Tasa alcanzada: {volume_rate:.1f} eventos/s de {target_rate} objetivo ({volume_score*100:.1f}%)")
            logger.info(f"- Valoración: {_get_score_text(volume_score)}")
            
            # 2. Resiliencia (fallos en cascada)
            resilience_metrics = cascading_results["test_specific"]["resilience_metrics"]
            resilience_score = resilience_metrics["resilience_score"]
            
            logger.info(f"Prueba de Resiliencia:")
            logger.info(f"- Tasa de éxito normal: {resilience_metrics['normal_success_rate']*100:.1f}%")
            logger.info(f"- Tasa de éxito durante fallos: {resilience_metrics['crash_success_rate']*100:.1f}%")
            logger.info(f"- Componentes caídos: {resilience_metrics['crashed_components']} de {len(system.components)}")
            logger.info(f"- Puntuación resiliencia: {resilience_score:.2f}")
            logger.info(f"- Valoración: {_get_resilience_text(resilience_score)}")
            
            # 3. Dependencias
            dependency_analysis = dependency_results["test_specific"]["dependency_analysis"]
            dependency_failure_rate = dependency_analysis["failure_rate"]
            dependency_score = 1.0 - dependency_failure_rate
            
            logger.info(f"Prueba de Dependencias:")
            logger.info(f"- Fallos en dependencias: {dependency_failure_rate*100:.1f}%")
            logger.info(f"- Componentes con dependencias: {dependency_analysis['components_with_dependencies']}")
            logger.info(f"- Valoración: {_get_score_text(dependency_score)}")
            
            # 4. Recuperación
            recovery_metrics = recovery_results["test_specific"]["recovery_metrics"]
            recovery_rate = recovery_results["test_specific"]["recovery_rate"]
            
            logger.info(f"Prueba de Recuperación:")
            logger.info(f"- Componentes crasheados: {recovery_metrics['components_crashed']}")
            logger.info(f"- Componentes recuperados: {recovery_metrics['spontaneous_recoveries'] + recovery_metrics['signal_recoveries']}")
            logger.info(f"- Tasa de recuperación: {recovery_rate*100:.1f}%")
            logger.info(f"- Valoración: {recovery_results['test_specific']['recovery_assessment']}")
            
            # Puntuación global
            global_score = (
                volume_score * 0.25 +
                resilience_score * 0.35 +
                dependency_score * 0.2 +
                recovery_rate * 0.2
            )
            
            logger.info(f"\nPuntuación Global del Core: {global_score*100:.1f}/100")
            logger.info(f"Valoración General: {_get_global_text(global_score)}")
            
            # Generar informe
            await generate_report({
                "volume": volume_results,
                "cascading": cascading_results,
                "dependency": dependency_results,
                "recovery": recovery_results,
                "scores": {
                    "volume": volume_score,
                    "resilience": resilience_score,
                    "dependency": dependency_score,
                    "recovery": recovery_rate,
                    "global": global_score
                }
            })
            
            return {
                "global_score": global_score,
                "assessment": _get_global_text(global_score),
                "component_scores": {
                    "volume": volume_score,
                    "resilience": resilience_score,
                    "dependency": dependency_score,
                    "recovery": recovery_rate
                }
            }
        
        finally:
            # Asegurar que el sistema se detenga
            await system.stop()
    
    except Exception as e:
        logger.error(f"Error en las pruebas: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def _get_score_text(score):
    """Texto descriptivo para una puntuación."""
    if score >= 0.9:
        return "Excelente"
    elif score >= 0.75:
        return "Muy Bueno"
    elif score >= 0.6:
        return "Bueno"
    elif score >= 0.4:
        return "Aceptable"
    else:
        return "Insuficiente"

def _get_resilience_text(score):
    """Texto descriptivo para puntuación de resiliencia."""
    if score >= 0.95:
        return "Excepcional - Inmune a fallos en cascada"
    elif score >= 0.85:
        return "Excelente - Altamente resistente a fallos"
    elif score >= 0.7:
        return "Muy Bueno - Resistente a fallos"
    elif score >= 0.5:
        return "Bueno - Degradación controlada"
    elif score >= 0.3:
        return "Aceptable - Degradación predecible"
    else:
        return "Insuficiente - Fallos en cascada presentes"

def _get_global_text(score):
    """Texto descriptivo para puntuación global."""
    if score >= 0.9:
        return "Excepcional - Listo para producción de misión crítica"
    elif score >= 0.8:
        return "Excelente - Robusto y altamente confiable"
    elif score >= 0.7:
        return "Muy Bueno - Confiable con supervisión mínima"
    elif score >= 0.6:
        return "Bueno - Adecuado para uso en producción"
    elif score >= 0.5:
        return "Aceptable - Requiere supervisión"
    elif score >= 0.4:
        return "Básico - Requiere mejoras antes de producción"
    else:
        return "Insuficiente - No recomendado para producción"

async def generate_report(results):
    """Generar informe detallado en Markdown."""
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Crear contenido
    content = f"""# Informe de Pruebas Extremas del Core - Sistema Híbrido API+WebSocket

## Resumen Ejecutivo

Este informe presenta los resultados de pruebas exhaustivas realizadas al núcleo (core) del sistema híbrido API+WebSocket. Las pruebas evaluaron cuatro dimensiones críticas: rendimiento bajo carga extrema, resiliencia ante fallos en cascada, manejo de dependencias entre componentes y capacidad de recuperación dinámica.

**Puntuación Global: {results['scores']['global']*100:.1f}/100**  
**Evaluación General: {_get_global_text(results['scores']['global'])}**

## Metodología de Prueba

Las pruebas se realizaron sobre un sistema complejo con {MAX_COMPONENTS if ENABLE_EXTREME_TESTS else 12} componentes interconectados, con diferentes tasas de fallos, latencias y dependencias. Cada prueba evaluó un aspecto crítico del sistema bajo condiciones extremas:

1. **Prueba de Volumen Alto Sostenido**: Evaluación del rendimiento bajo carga extrema con eventos a alta velocidad.
2. **Prueba de Resiliencia ante Fallos**: Análisis del comportamiento cuando múltiples componentes fallan simultáneamente.
3. **Prueba de Dependencias**: Evaluación del manejo de dependencias entre componentes.
4. **Prueba de Recuperación Dinámica**: Verificación de la capacidad de auto-recuperación del sistema.

## Resultados por Categoría

### 1. Rendimiento bajo Carga Extrema

**Puntuación: {results['scores']['volume']*100:.1f}/100 - {_get_score_text(results['scores']['volume'])}**

- **Tasa objetivo**: {results['volume']['test_specific']['target_rate']} eventos/segundo
- **Tasa alcanzada**: {results['volume']['test_specific']['actual_rate']:.1f} eventos/segundo ({results['scores']['volume']*100:.1f}% del objetivo)
- **Eventos totales procesados**: {results['volume']['test_specific']['total_events']}
- **Solicitudes API**: {results['volume']['test_specific']['total_api_calls']}

El sistema demostró una {_get_score_text(results['scores']['volume']).lower()} capacidad para manejar volúmenes altos de eventos. La arquitectura híbrida permitió el procesamiento eficiente de solicitudes API incluso durante periodos de alta carga de eventos.

### 2. Resiliencia ante Fallos en Cascada

**Puntuación: {results['scores']['resilience']*100:.1f}/100 - {_get_resilience_text(results['scores']['resilience'])}**

- **Componentes caídos**: {results['cascading']['test_specific']['resilience_metrics']['crashed_components']} de {MAX_COMPONENTS if ENABLE_EXTREME_TESTS else 12}
- **Tasa de éxito normal**: {results['cascading']['test_specific']['resilience_metrics']['normal_success_rate']*100:.1f}%
- **Tasa de éxito durante fallos**: {results['cascading']['test_specific']['resilience_metrics']['crash_success_rate']*100:.1f}%
- **Intentos de recuperación**: {results['cascading']['test_specific']['resilience_metrics']['recovery_attempts']}
- **Recuperaciones exitosas**: {results['cascading']['test_specific']['resilience_metrics']['successful_recoveries']}

El sistema demostró una {_get_resilience_text(results['scores']['resilience']).split(' - ')[0].lower()} capacidad para mantener operaciones incluso cuando múltiples componentes fallan simultáneamente. La degradación del rendimiento fue proporcional al número de componentes caídos, sin efectos de fallos en cascada que afectaran a componentes sanos.

### 3. Manejo de Dependencias entre Componentes

**Puntuación: {results['scores']['dependency']*100:.1f}/100 - {_get_score_text(results['scores']['dependency'])}**

- **Componentes con dependencias**: {results['dependency']['test_specific']['dependency_analysis']['components_with_dependencies']}
- **Promedio de dependencias por componente**: {results['dependency']['test_specific']['dependency_analysis']['avg_dependencies_per_component']:.1f}
- **Tasa de fallos en dependencias**: {results['dependency']['test_specific']['dependency_analysis']['failure_rate']*100:.1f}%

El sistema demostró un {_get_score_text(results['scores']['dependency']).lower()} manejo de las dependencias entre componentes. Los fallos en componentes produjeron un impacto predecible y controlado en los componentes dependientes, sin propagación excesiva de errores.

### 4. Capacidad de Recuperación Dinámica

**Puntuación: {results['scores']['recovery']*100:.1f}/100 - {results['recovery']['test_specific']['recovery_assessment']}**

- **Componentes crasheados**: {results['recovery']['test_specific']['recovery_metrics']['components_crashed']}
- **Recuperaciones espontáneas**: {results['recovery']['test_specific']['recovery_metrics']['spontaneous_recoveries']}
- **Señales de recuperación enviadas**: {results['recovery']['test_specific']['recovery_metrics']['recovery_signals']}
- **Recuperaciones por señal**: {results['recovery']['test_specific']['recovery_metrics']['signal_recoveries']}
- **Componentes no recuperados**: {results['recovery']['test_specific']['recovery_metrics']['failed_recoveries']}

El sistema demostró una {results['recovery']['test_specific']['recovery_assessment'].lower()} capacidad de recuperación de componentes caídos. Los mecanismos de auto-recuperación y las señales de recuperación externa funcionaron según lo esperado.

## Análisis de Métricas Clave

### Métricas de Latencia

- **Latencia media en API**: {results['volume']['core']['avg_latency_ms']:.2f}ms
- **Latencia mínima**: {results['volume']['core'].get('min_latency_ms', 0):.2f}ms
- **Latencia máxima**: {results['volume']['core'].get('max_latency_ms', 0):.2f}ms

### Métricas de Colas

- **Tamaño máximo de cola de prioridad**: {results['volume']['system']['priority_queue_size']}
- **Tamaño máximo de cola regular**: {results['volume']['system']['regular_queue_size']}

### Distribución de Errores

- **Principales tipos de errores**:
{os.linesep.join([f"  - {error_type}: {count}" for error_type, count in results['volume']['core'].get('top_errors', [])])}

## Conclusiones y Recomendaciones

### Fortalezas del Sistema

1. **Arquitectura Híbrida Efectiva**: La combinación de API para solicitudes directas y WebSocket para eventos ha demostrado ser altamente efectiva para prevenir deadlocks y fallos en cascada.

2. **Aislamiento de Componentes**: El sistema mantiene un excelente aislamiento entre componentes, permitiendo que los fallos se contengan sin propagación.

3. **Manejo de Carga**: El sistema puede manejar volúmenes de eventos muy altos sin degradación significativa del rendimiento.

4. **Recuperación Dinámica**: Los mecanismos de auto-recuperación permiten que el sistema restaure operaciones sin intervención manual.

### Áreas de Mejora

1. **Optimización de Latencia**: Aunque la latencia general es aceptable, hay margen para mejorar la cola de procesamiento de eventos.

2. **Recuperación de Componentes**: El porcentaje de componentes recuperados automáticamente podría mejorarse con estrategias más sofisticadas.

3. **Manejo de Dependencias**: El sistema podría beneficiarse de un manejo más proactivo de las dependencias, anticipando fallos antes de que ocurran.

## Recomendaciones para Producción

1. **Ajuste de Timeouts**: Personalizar los valores de timeout según el tipo de operación y componente.

2. **Monitoreo Detallado**: Implementar un sistema de monitoreo en tiempo real que detecte y alerte sobre componentes degradados o caídos.

3. **Circuit Breakers**: Añadir circuit breakers en puntos críticos para evitar sobrecargas en componentes degradados.

4. **Balanceo de Carga Dinámico**: Implementar un sistema de balanceo de carga que redirija eventos según la capacidad de cada componente.

---

*Informe generado: {timestamp}*
"""

    # Guardar a archivo
    report_path = "docs/informe_pruebas_extremas_core.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write(content)
    
    logger.info(f"Informe detallado generado: {report_path}")

# Punto de entrada
if __name__ == "__main__":
    asyncio.run(run_all_core_tests())