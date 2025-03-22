"""
Pruebas de estrés para el sistema híbrido Genesis API+WebSocket.

Este módulo contiene pruebas que evalúan el rendimiento y la resiliencia
del sistema híbrido bajo condiciones extremas, incluyendo:
- Alto volumen de eventos
- Variedad de tipos de eventos (incluyendo datos pesados)
- Fallos simulados en componentes
- Latencia de red simulada
- Ejecución extendida
"""

import asyncio
import logging
import random
import sys
import time
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple, Callable

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("stress_test")

# Métricas de rendimiento
@dataclass
class PerformanceMetrics:
    """Métricas acumuladas de rendimiento."""
    total_events: int = 0
    successful_events: int = 0
    failed_events: int = 0
    min_latency: float = float('inf')
    max_latency: float = 0
    total_latency: float = 0
    start_time: float = field(default_factory=time.time)
    component_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    event_counts: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    def record_event(self, event_type: str, latency: float, success: bool):
        """Registrar un evento procesado."""
        self.total_events += 1
        self.total_latency += latency
        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)
        
        if success:
            self.successful_events += 1
        else:
            self.failed_events += 1
        
        # Contar por tipo de evento
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
    
    def record_component_metric(self, component_id: str, metric_name: str, value: Any):
        """Registrar una métrica de componente."""
        if component_id not in self.component_metrics:
            self.component_metrics[component_id] = {}
        
        metrics = self.component_metrics[component_id]
        
        if metric_name in metrics:
            # Para métricas numéricas, hacemos un seguimiento del min/max/avg
            if isinstance(value, (int, float)) and isinstance(metrics[metric_name], dict):
                cur = metrics[metric_name]
                cur["count"] += 1
                cur["min"] = min(cur["min"], value)
                cur["max"] = max(cur["max"], value)
                cur["total"] += value
                cur["avg"] = cur["total"] / cur["count"]
            else:
                # Para métricas no numéricas, simplemente almacenamos el último valor
                metrics[metric_name] = value
        else:
            # Primera vez que vemos esta métrica
            if isinstance(value, (int, float)):
                metrics[metric_name] = {
                    "count": 1,
                    "min": value,
                    "max": value,
                    "total": value,
                    "avg": value
                }
            else:
                metrics[metric_name] = value
    
    def record_error(self, error_type: str):
        """Registrar un tipo de error."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def calculate_avg_latency(self) -> float:
        """Calcular latencia promedio."""
        if self.total_events == 0:
            return 0
        return self.total_latency / self.total_events
    
    def calculate_events_per_second(self) -> float:
        """Calcular eventos por segundo."""
        duration = time.time() - self.start_time
        if duration == 0:
            return 0
        return self.total_events / duration
    
    def get_success_rate(self) -> float:
        """Calcular tasa de éxito."""
        if self.total_events == 0:
            return 100.0
        return (self.successful_events / self.total_events) * 100.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas."""
        duration = time.time() - self.start_time
        return {
            "total_events": self.total_events,
            "successful_events": self.successful_events,
            "failed_events": self.failed_events,
            "success_rate": f"{self.get_success_rate():.2f}%",
            "events_per_second": f"{self.calculate_events_per_second():.2f}",
            "min_latency_ms": f"{self.min_latency * 1000:.2f}" if self.min_latency != float('inf') else "N/A",
            "max_latency_ms": f"{self.max_latency * 1000:.2f}",
            "avg_latency_ms": f"{self.calculate_avg_latency() * 1000:.2f}",
            "duration_seconds": f"{duration:.2f}",
            "top_5_events": self._get_top_n_events(5),
            "top_5_errors": self._get_top_n_errors(5)
        }
    
    def _get_top_n_events(self, n: int) -> List[Tuple[str, int]]:
        """Obtener los N tipos de eventos más comunes."""
        return sorted(self.event_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def _get_top_n_errors(self, n: int) -> List[Tuple[str, int]]:
        """Obtener los N tipos de errores más comunes."""
        return sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:n]

# Interfaces básicas pero mejoradas para estrés
class StressTestComponent:
    """Componente para pruebas de estrés."""
    
    def __init__(self, id: str, failure_rate: float = 0.0, 
                 latency_range: Tuple[float, float] = (0.0, 0.0),
                 crash_after_n_calls: Optional[int] = None):
        self.id = id
        self.received_events = []
        self.processed_requests = []
        self.failure_rate = failure_rate  # Probabilidad de fallo
        self.latency_range = latency_range  # Rango de latencia (min, max) en segundos
        self.call_count = 0
        self.crash_after_n_calls = crash_after_n_calls  # Simular crash después de N llamadas
        self.crashed = False
        self.metrics = PerformanceMetrics()
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitudes API (con simulación de fallos y latencia)."""
        if self.crashed:
            # Componente ha "crasheado", no responde
            self.metrics.record_error("component_crashed")
            logger.warning(f"[{self.id}] No se puede procesar solicitud: componente crasheado")
            return None
        
        self.call_count += 1
        start_time = time.time()
        
        self.processed_requests.append({
            "type": request_type,
            "data": data,
            "source": source,
            "timestamp": start_time
        })
        
        # Simular latencia
        latency = random.uniform(self.latency_range[0], self.latency_range[1])
        if latency > 0:
            await asyncio.sleep(latency)
        
        # Verificar si debemos crashear
        if self.crash_after_n_calls and self.call_count >= self.crash_after_n_calls:
            self.crashed = True
            logger.warning(f"[{self.id}] Componente simulando crash después de {self.call_count} llamadas")
            self.metrics.record_error("simulated_crash")
            raise Exception(f"Componente {self.id} ha crasheado (simulación)")
        
        # Simular fallos aleatorios
        if random.random() < self.failure_rate:
            self.metrics.record_error("random_failure")
            logger.warning(f"[{self.id}] Fallo aleatorio simulado para {request_type}")
            self.metrics.record_event(request_type, time.time() - start_time, False)
            raise Exception(f"Fallo aleatorio simulado en {self.id}")
        
        # Respuesta normal
        logger.debug(f"[{self.id}] Procesando solicitud {request_type} de {source}")
        result = {"status": "ok", "processed_by": self.id}
        
        # Añadir datos de respuesta específicos según el tipo
        if request_type == "echo":
            result["echo"] = data.get("message", "")
            
        elif request_type == "compute":
            # Simular procesamiento intensivo
            result["result"] = self._compute_intensive_task(data.get("complexity", 1))
            
        elif request_type == "query":
            # Simular consulta de datos
            result["data"] = self._generate_data(data.get("size", 1))
        
        # Registrar métrica de latencia
        latency = time.time() - start_time
        self.metrics.record_event(request_type, latency, True)
        self.metrics.record_component_metric(self.id, "request_latency", latency)
        
        return result
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar eventos recibidos (con simulación de fallos y latencia)."""
        if self.crashed:
            # Componente ha "crasheado", pero no generará error ya que WebSockets no esperan respuesta
            self.metrics.record_error("event_to_crashed_component")
            logger.warning(f"[{self.id}] Evento recibido pero componente crasheado: {event_type}")
            return
        
        start_time = time.time()
        
        # Registrar evento
        self.received_events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": start_time
        })
        
        # Simular latencia
        latency = random.uniform(self.latency_range[0], self.latency_range[1])
        if latency > 0:
            await asyncio.sleep(latency)
        
        # Para eventos, no generamos excepciones ya que son fire-and-forget,
        # pero registramos métricas de latencia y "fallos" internos
        failed = random.random() < self.failure_rate
        if failed:
            self.metrics.record_error("event_internal_failure")
            logger.warning(f"[{self.id}] Fallo interno al procesar evento {event_type}")
        
        # Procesamiento según tipo de evento
        if event_type == "notification" and not failed:
            logger.debug(f"[{self.id}] Procesando notificación: {data.get('message', '')}")
            
        elif event_type == "data_update" and not failed:
            logger.debug(f"[{self.id}] Actualizando datos: {len(data)} elementos")
            
        elif event_type == "large_payload" and not failed:
            size = len(json.dumps(data))
            logger.debug(f"[{self.id}] Recibido payload grande: {size/1024:.2f} KB")
            self.metrics.record_component_metric(self.id, "large_payload_size", size)
        
        # Registrar métrica de latencia de eventos
        latency = time.time() - start_time
        self.metrics.record_event(event_type, latency, not failed)
        self.metrics.record_component_metric(self.id, "event_latency", latency)
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.debug(f"Componente {self.id} iniciado")
        self.metrics.start_time = time.time()
        self.metrics.record_component_metric(self.id, "start_time", self.metrics.start_time)
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.debug(f"Componente {self.id} detenido")
        stop_time = time.time()
        self.metrics.record_component_metric(self.id, "stop_time", stop_time)
        self.metrics.record_component_metric(self.id, "uptime", stop_time - self.metrics.start_time)
    
    async def reset(self) -> None:
        """Reiniciar componente después de un crash."""
        if self.crashed:
            logger.info(f"Reiniciando componente {self.id} después de crash")
            self.crashed = False
            # Reiniciar contador de llamadas pero mantener métricas
            self.call_count = 0
            self.metrics.record_component_metric(self.id, "reset_count", 
                self.metrics.component_metrics[self.id].get("reset_count", {}).get("count", 0) + 1
                if self.id in self.metrics.component_metrics else 1)
    
    def _compute_intensive_task(self, complexity: int) -> int:
        """Simular tarea computacionalmente intensiva."""
        result = 0
        # Ajustar complejidad para evitar bloqueos demasiado largos en pruebas
        iterations = min(100000 * complexity, 1000000)
        for i in range(iterations):
            result += i * (i % 7)
        return result % 10000  # Devolver solo los últimos 4 dígitos
    
    def _generate_data(self, size: int) -> Dict[str, Any]:
        """Generar datos de respuesta simulados de tamaño variable."""
        # Generar diccionario con datos aleatorios
        size = min(size, 100)  # Limitar tamaño máximo
        result = {
            "timestamp": time.time(),
            "generator": self.id,
            "items": []
        }
        
        # Generar elementos según tamaño requerido
        for i in range(size):
            result["items"].append({
                "id": i,
                "value": random.random() * 100,
                "name": f"Item-{self.id}-{i}",
                "metadata": {
                    "type": random.choice(["A", "B", "C", "D"]),
                    "priority": random.randint(1, 10),
                    "tags": [f"tag{j}" for j in range(random.randint(1, 5))]
                }
            })
        
        return result

class StressTestCoordinator:
    """Coordinador mejorado para pruebas de estrés."""
    
    def __init__(self, simulated_network_latency: Tuple[float, float] = (0.0, 0.0),
                network_failure_rate: float = 0.0):
        self.components = {}
        self.event_subscribers = {}
        self.global_metrics = PerformanceMetrics()
        self.simulated_network_latency = simulated_network_latency
        self.network_failure_rate = network_failure_rate
        
    def register_component(self, id: str, component: StressTestComponent) -> None:
        """Registrar componente en el coordinador."""
        self.components[id] = component
        logger.debug(f"Componente {id} registrado")
        self.global_metrics.record_component_metric("coordinator", "registered_components", len(self.components))
    
    def subscribe(self, component_id: str, event_types: List[str]) -> None:
        """Suscribir componente a tipos de eventos."""
        if component_id not in self.components:
            logger.warning(f"No se puede suscribir: componente {component_id} no registrado")
            self.global_metrics.record_error("subscribe_non_existent_component")
            return
        
        for event_type in event_types:
            if event_type not in self.event_subscribers:
                self.event_subscribers[event_type] = set()
            self.event_subscribers[event_type].add(component_id)
            logger.debug(f"Componente {component_id} suscrito a eventos {event_type}")
        
        self.global_metrics.record_component_metric("coordinator", "total_subscriptions", 
                                                  sum(len(subs) for subs in self.event_subscribers.values()))
    
    async def request(self, target_id: str, request_type: str, data: Dict[str, Any], source: str, 
                     timeout: float = 5.0) -> Optional[Any]:
        """API: Enviar solicitud a componente con simulación de fallos de red."""
        start_time = time.time()
        
        if target_id not in self.components:
            logger.warning(f"Componente {target_id} no encontrado")
            self.global_metrics.record_error("request_to_non_existent_component")
            return None
        
        # Simular fallo de red
        if random.random() < self.network_failure_rate:
            logger.warning(f"Fallo de red simulado para solicitud a {target_id}")
            self.global_metrics.record_error("network_failure")
            self.global_metrics.record_event(request_type, time.time() - start_time, False)
            return None
        
        # Simular latencia de red
        network_latency = random.uniform(self.simulated_network_latency[0], self.simulated_network_latency[1])
        if network_latency > 0:
            await asyncio.sleep(network_latency / 2)  # Mitad de latencia antes de enviar
        
        try:
            # Solicitud con timeout para evitar bloqueos
            result = await asyncio.wait_for(
                self.components[target_id].process_request(request_type, data, source),
                timeout=timeout
            )
            
            # Simular latencia de vuelta
            if network_latency > 0:
                await asyncio.sleep(network_latency / 2)  # Mitad de latencia al recibir
            
            # Registrar métricas de solicitud
            latency = time.time() - start_time
            self.global_metrics.record_event(request_type, latency, True)
            self.global_metrics.record_component_metric("coordinator", "request_latency", latency)
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout en solicitud {request_type} a {target_id}")
            self.global_metrics.record_error("request_timeout")
            self.global_metrics.record_event(request_type, time.time() - start_time, False)
            return None
            
        except Exception as e:
            logger.error(f"Error en solicitud a {target_id}: {e}")
            self.global_metrics.record_error(f"component_exception")
            self.global_metrics.record_event(request_type, time.time() - start_time, False)
            return None
    
    async def emit_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """WebSocket: Emitir evento a suscriptores con simulación de fallos de red."""
        start_time = time.time()
        subscribers = self.event_subscribers.get(event_type, set())
        tasks = []
        
        # Registrar métricas básicas del evento
        self.global_metrics.record_component_metric("coordinator", "emit_event_subscribers", len(subscribers))
        
        # Para cada suscriptor, enviar evento
        for comp_id in subscribers:
            if comp_id in self.components and comp_id != source:
                
                # Simular fallo de red para este suscriptor
                if random.random() < self.network_failure_rate:
                    logger.warning(f"Fallo de red simulado para evento {event_type} a {comp_id}")
                    self.global_metrics.record_error("event_network_failure")
                    continue
                
                # Simular latencia de red
                network_latency = random.uniform(self.simulated_network_latency[0], self.simulated_network_latency[1])
                if network_latency > 0:
                    # Para eventos, la latencia se aplica al task completo ya que son asíncronos
                    delivery_task = self._delayed_event_delivery(
                        comp_id, event_type, data, source, network_latency
                    )
                else:
                    delivery_task = self.components[comp_id].on_event(event_type, data, source)
                
                tasks.append(delivery_task)
        
        # Ejecutar entregas en paralelo
        if tasks:
            # Para eventos, no esperamos las respuestas pero registramos excepciones sin propagar
            for task in asyncio.as_completed(tasks):
                try:
                    await task
                except Exception as e:
                    logger.error(f"Error en entrega de evento {event_type}: {e}")
                    self.global_metrics.record_error("event_delivery_error")
        
        # Registrar métrica de tiempo total de emisión (no incluye procesamiento en componentes)
        emit_latency = time.time() - start_time
        self.global_metrics.record_component_metric("coordinator", "emit_event_latency", emit_latency)
    
    async def _delayed_event_delivery(self, comp_id: str, event_type: str, data: Dict[str, Any], 
                                    source: str, delay: float) -> None:
        """Entregar evento con retraso simulado."""
        await asyncio.sleep(delay)
        await self.components[comp_id].on_event(event_type, data, source)
    
    async def start(self) -> None:
        """Iniciar coordinador y todos los componentes."""
        tasks = [comp.start() for comp in self.components.values()]
        await asyncio.gather(*tasks)
        logger.info("Coordinador iniciado con %d componentes", len(self.components))
        self.global_metrics.start_time = time.time()
    
    async def stop(self) -> None:
        """Detener coordinador y todos los componentes."""
        tasks = [comp.stop() for comp in self.components.values()]
        await asyncio.gather(*tasks)
        logger.info("Coordinador detenido")
    
    async def reset_crashed_components(self) -> int:
        """Reiniciar componentes que hayan crasheado."""
        reset_count = 0
        reset_tasks = []
        
        for comp_id, comp in self.components.items():
            if comp.crashed:
                logger.info(f"Reiniciando componente crasheado: {comp_id}")
                reset_tasks.append(comp.reset())
                reset_count += 1
        
        if reset_tasks:
            await asyncio.gather(*reset_tasks)
            self.global_metrics.record_component_metric("coordinator", "components_reset", reset_count)
        
        return reset_count

    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas globales y de componentes."""
        result = {
            "global": self.global_metrics.get_summary(),
            "components": {}
        }
        
        # Agregar métricas por componente
        for comp_id, comp in self.components.items():
            result["components"][comp_id] = {
                "metrics": comp.metrics.get_summary(),
                "status": "crashed" if comp.crashed else "active",
                "processed_requests": len(comp.processed_requests),
                "received_events": len(comp.received_events)
            }
        
        return result

# Pruebas de estrés
async def test_volume_scalability(coordinator: StressTestCoordinator, 
                                max_events_per_second: int, 
                                duration_seconds: int,
                                ramp_up_seconds: int = 10) -> Dict[str, Any]:
    """
    Prueba de escalabilidad de volumen.
    
    Aumenta gradualmente el número de eventos emitidos, desde 1 hasta max_events_per_second
    durante el tiempo de ramp_up, y luego mantiene el máximo por el resto de la duración.
    """
    logger.info("Iniciando prueba de escalabilidad de volumen")
    logger.info(f"- Máximo eventos/segundo: {max_events_per_second}")
    logger.info(f"- Duración total: {duration_seconds}s")
    logger.info(f"- Tiempo de escalado: {ramp_up_seconds}s")
    
    # Tipos de eventos para diversificar
    event_types = ["data_update", "notification", "status_change", "heartbeat", "alert"]
    
    # Control de tiempo
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    # Contar eventos totales emitidos
    events_emitted = 0
    
    try:
        # Bucle principal de emisión
        while time.time() < end_time:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Calcular tasa actual según fase de ramp-up
            if elapsed < ramp_up_seconds:
                # Durante ramp-up, aumentar linealmente
                current_rate = max(1, int((elapsed / ramp_up_seconds) * max_events_per_second))
            else:
                # Después de ramp-up, usar tasa máxima
                current_rate = max_events_per_second
            
            # Emitir eventos a la tasa actual
            logger.debug(f"Emitiendo a {current_rate} eventos/segundo (t+{elapsed:.1f}s)")
            
            # Calcular cuántos eventos emitir en este ciclo (aproximadamente 1 segundo)
            events_this_second = []
            for _ in range(current_rate):
                # Seleccionar tipo de evento
                event_type = random.choice(event_types)
                
                # Crear datos según tipo
                if event_type == "data_update":
                    # Eventos con datos más pesados
                    data_size = random.randint(10, 100)
                    data = {"items": [{"id": i, "value": random.random()} for i in range(data_size)]}
                else:
                    # Eventos ligeros
                    data = {
                        "timestamp": time.time(),
                        "priority": random.randint(1, 5),
                        "message": f"Evento {event_type} #{events_emitted}"
                    }
                
                events_this_second.append((event_type, data))
            
            # Emitir eventos en paralelo
            emission_tasks = []
            for event_type, data in events_this_second:
                emission_tasks.append(
                    coordinator.emit_event(event_type, data, "stress_test")
                )
                events_emitted += 1
            
            # Esperar que todos los eventos se emitan
            await asyncio.gather(*emission_tasks)
            
            # Calcular cuánto esperar para mantener la tasa deseada
            elapsed_in_cycle = time.time() - current_time
            sleep_time = max(0, 1.0 - elapsed_in_cycle)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        logger.info(f"Prueba completada - {events_emitted} eventos emitidos")
        
        # Recolectar métricas
        metrics = coordinator.get_metrics()
        metrics["test_specific"] = {
            "target_max_rate": max_events_per_second,
            "actual_rate": events_emitted / duration_seconds,
            "total_events_emitted": events_emitted,
            "test_duration": duration_seconds,
            "ramp_up_seconds": ramp_up_seconds
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error durante prueba de volumen: {e}")
        return {
            "error": str(e),
            "events_emitted": events_emitted,
            "elapsed_seconds": time.time() - start_time
        }

# Función para configurar sistema de prueba
async def setup_stress_test_system(
        num_components: int = 10,
        failure_rates: Tuple[float, float] = (0.0, 0.0),
        latency_range: Tuple[float, float] = (0.0, 0.0),
        crash_component_indices: List[int] = None,
        network_latency: Tuple[float, float] = (0.0, 0.0),
        network_failure_rate: float = 0.0) -> StressTestCoordinator:
    """
    Configurar sistema de prueba con componentes.
    
    Args:
        num_components: Número de componentes a crear
        failure_rates: Rango de tasas de fallo (min, max) para componentes
        latency_range: Rango de latencia (min, max) para componentes
        crash_component_indices: Índices de componentes que crashean
        network_latency: Rango de latencia de red (min, max)
        network_failure_rate: Tasa de fallos de red
    """
    # Crear coordinador
    coordinator = StressTestCoordinator(
        simulated_network_latency=network_latency,
        network_failure_rate=network_failure_rate
    )
    
    # Crear componentes
    for i in range(num_components):
        # Determinar tasa de fallo para este componente
        if failure_rates[0] == failure_rates[1]:
            failure_rate = failure_rates[0]
        else:
            failure_rate = random.uniform(failure_rates[0], failure_rates[1])
        
        # Determinar si este componente debe crashear
        crash_after = None
        if crash_component_indices and i in crash_component_indices:
            # Crashear después de 50-150 llamadas
            crash_after = random.randint(50, 150)
        
        # Crear componente con configuración
        component = StressTestComponent(
            f"comp_{i}",
            failure_rate=failure_rate,
            latency_range=latency_range,
            crash_after_n_calls=crash_after
        )
        
        # Registrar componente
        coordinator.register_component(f"comp_{i}", component)
        
        # Suscribir a eventos de forma aleatoria
        event_types = ["notification", "data_update", "status_change", 
                      "heartbeat", "large_payload", "high_priority"]
        
        # Cada componente se suscribe a 2-4 tipos de eventos
        num_subscriptions = random.randint(2, min(4, len(event_types)))
        subscribed_events = random.sample(event_types, num_subscriptions)
        
        coordinator.subscribe(f"comp_{i}", subscribed_events)
    
    # Iniciar sistema
    await coordinator.start()
    
    return coordinator

# Ejemplo de ejecución de prueba
async def run_example_test():
    """Ejecutar una prueba sencilla de ejemplo."""
    logger.info("Iniciando prueba de ejemplo...")
    
    # Configurar sistema pequeño para prueba
    system = await setup_stress_test_system(
        num_components=3,
        failure_rates=(0.01, 0.05),  # 1-5% de fallos
        latency_range=(0.01, 0.1),   # 10-100ms de latencia
        crash_component_indices=[1]  # El componente 1 crasheará
    )
    
    try:
        # Ejecutar prueba de volumen moderada
        results = await test_volume_scalability(
            system,
            max_events_per_second=20,
            duration_seconds=10,
            ramp_up_seconds=3
        )
        
        # Mostrar resultados
        logger.info("=== Resultados de la prueba ===")
        logger.info(f"Eventos emitidos: {results['test_specific']['total_events_emitted']}")
        logger.info(f"Tasa efectiva: {results['test_specific']['actual_rate']:.2f} eventos/s")
        logger.info(f"Tasa de éxito: {results['global']['success_rate']}")
        
        return results
    finally:
        # Asegurar que el sistema se detenga
        await system.stop()

# Punto de entrada
if __name__ == "__main__":
    # Ejecutar prueba de ejemplo
    asyncio.run(run_example_test())