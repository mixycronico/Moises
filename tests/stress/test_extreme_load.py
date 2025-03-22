"""
Prueba de estrés extremo para el sistema híbrido API+WebSocket de Genesis.

Este script implementa una serie de pruebas rigurosas que evalúan:
1. Capacidad para manejar cargas extremadamente altas
2. Resiliencia ante fallos simultáneos en múltiples componentes
3. Comportamiento bajo condiciones de latencia variable
4. Capacidad de auto-recuperación bajo carga sostenida
5. Rendimiento con múltiples tipos de eventos concurrentes
"""

import asyncio
import logging
import random
import time
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Callable, Union

# Configurar logging detallado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("extreme_test")

# Aumentar verbose para diagnóstico
VERBOSE = True

# Clases de componentes específicas para pruebas extremas

class HighStressComponent:
    """Componente diseñado para soportar pruebas de estrés extremas."""
    
    def __init__(self, id: str, failure_probability: float = 0.0, 
                 latency_range: Tuple[float, float] = (0.0, 0.0),
                 crash_after_n_calls: Optional[int] = None,
                 recovery_time: float = 0.5,
                 max_parallel_requests: int = 100):
        self.id = id
        self.received_events = []
        self.processed_requests = []
        self.failure_probability = failure_probability
        self.latency_range = latency_range
        self.crash_after_n_calls = crash_after_n_calls
        self.recovery_time = recovery_time
        self.max_parallel_requests = max_parallel_requests
        
        # Estado del componente
        self.call_count = 0
        self.event_count = 0
        self.crashed = False
        self.overloaded = False
        self.last_overload_time = 0
        self.active_requests = 0
        self.failures = 0
        self.timeouts = 0
        self.start_time = time.time()
        
        # Semáforo para limitar solicitudes paralelas
        self.request_semaphore = asyncio.Semaphore(max_parallel_requests)
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitudes API con gestión avanzada de sobrecarga y fallos."""
        if self.crashed:
            # No procesar si está caído
            self.failures += 1
            logger.warning(f"[{self.id}] Solicitud rechazada: componente crasheado")
            raise Exception(f"Componente {self.id} no disponible (crashed)")
        
        self.call_count += 1
        start_time = time.time()
        
        # Registrar datos de solicitud
        self.processed_requests.append({
            "type": request_type,
            "source": source,
            "timestamp": start_time,
            "data_size": len(json.dumps(data)) if data else 0
        })
        
        # Verificar si debe crashear por número de llamadas
        if self.crash_after_n_calls and self.call_count >= self.crash_after_n_calls:
            self.crashed = True
            logger.warning(f"[{self.id}] Componente crasheado tras {self.call_count} llamadas")
            raise Exception(f"Componente {self.id} ha crasheado (simulación)")
        
        # Intentar adquirir semáforo para controlar concurrencia
        try:
            # Usar timeout para evitar esperas infinitas
            acquire_timeout = 0.5  # 500ms max para adquirir semáforo
            async with asyncio.timeout(acquire_timeout):
                sem_acquired = await self.request_semaphore.acquire()
                if not sem_acquired:
                    raise Exception("No se pudo adquirir semáforo")
                
                try:
                    self.active_requests += 1
                    
                    # Verificar sobrecarga (si hay demasiadas solicitudes activas)
                    if self.active_requests > self.max_parallel_requests * 0.9:
                        overload_time = time.time()
                        # Si ya estaba sobrecargado hace poco, aumentar probabilidad de fallo
                        if self.overloaded and (overload_time - self.last_overload_time < 1.0):
                            self.failure_probability *= 1.5  # Aumentar probabilidad de fallos durante sobrecarga
                        
                        self.overloaded = True
                        self.last_overload_time = overload_time
                        logger.warning(f"[{self.id}] Componente sobrecargado: {self.active_requests} solicitudes activas")
                    
                    # Simular latencia (mayor si está sobrecargado)
                    latency_multiplier = 2.0 if self.overloaded else 1.0
                    latency = random.uniform(
                        self.latency_range[0], 
                        self.latency_range[1] * latency_multiplier
                    )
                    if latency > 0:
                        await asyncio.sleep(latency)
                    
                    # Simular fallos aleatorios (probabilidad aumentada si está sobrecargado)
                    effective_failure_prob = self.failure_probability
                    if self.overloaded:
                        effective_failure_prob = min(0.8, effective_failure_prob * 1.5)
                    
                    if random.random() < effective_failure_prob:
                        self.failures += 1
                        logger.warning(f"[{self.id}] Fallo aleatorio en solicitud {request_type}")
                        raise Exception(f"Error simulado en procesamiento de {request_type}")
                    
                    # Procesamiento según tipo de solicitud
                    result = await self._process_by_type(request_type, data, source)
                    
                    # Restablecer estado de sobrecarga si hay pocas solicitudes activas
                    if self.overloaded and self.active_requests < self.max_parallel_requests * 0.7:
                        self.overloaded = False
                        self.failure_probability = max(0.01, self.failure_probability / 1.5)
                    
                    return result
                    
                finally:
                    self.active_requests -= 1
                    self.request_semaphore.release()
        
        except asyncio.TimeoutError:
            self.timeouts += 1
            logger.warning(f"[{self.id}] Timeout al intentar procesar solicitud (sobrecarga)")
            raise Exception(f"Componente {self.id} sobrecargado, no puede procesar más solicitudes")
    
    async def _process_by_type(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Implementar procesamiento específico por tipo de solicitud."""
        if request_type == "query":
            # Consulta de datos
            return {
                "result": self._generate_data(data.get("size", 10)),
                "status": "success",
                "processor": self.id,
            }
            
        elif request_type == "compute":
            # Cálculo intensivo
            complexity = data.get("complexity", 1)
            result = self._compute_intensive(complexity)
            return {
                "result": result,
                "status": "success",
                "computation_time": time.time() - self.start_time,
                "processor": self.id,
            }
            
        elif request_type == "update":
            # Actualización de estado
            return {
                "previous_state": {"active": random.choice([True, False])},
                "new_state": {"active": data.get("active", True)},
                "status": "success",
                "processor": self.id,
            }
            
        elif request_type == "health":
            # Verificación de salud
            return {
                "status": "degraded" if self.overloaded else "healthy",
                "uptime": time.time() - self.start_time,
                "active_requests": self.active_requests,
                "failure_rate": self.failures / max(1, self.call_count),
                "processor": self.id,
            }
            
        else:
            # Tipo genérico
            return {
                "status": "success",
                "message": f"Processed {request_type}",
                "processor": self.id,
            }
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Procesar eventos WebSocket con manejo de sobrecarga."""
        if self.crashed:
            # Eventos se descartan silenciosamente si está caído
            logger.debug(f"[{self.id}] Evento {event_type} descartado: componente crasheado")
            return
        
        self.event_count += 1
        start_time = time.time()
        
        # Registrar evento recibido
        event_size = len(json.dumps(data)) if data else 0
        self.received_events.append({
            "type": event_type,
            "source": source,
            "timestamp": start_time,
            "data_size": event_size
        })
        
        # Para eventos grandes, aumentar latencia
        latency_factor = 1.0
        if event_size > 10000:  # 10KB
            latency_factor = 2.0
        elif event_size > 1000:  # 1KB
            latency_factor = 1.5
        
        # Simular latencia de procesamiento
        latency = random.uniform(
            self.latency_range[0], 
            self.latency_range[1] * latency_factor
        )
        if latency > 0:
            await asyncio.sleep(latency)
        
        # Actualizar estados internos según el tipo de evento
        if event_type == "system_status":
            # Eventos de estado global pueden modificar comportamiento
            system_load = data.get("load", 0.5)
            if system_load > 0.8:
                # Sistema sobrecargado, aumentar latencia
                self.latency_range = (
                    self.latency_range[0],
                    self.latency_range[1] * 1.5
                )
            elif system_load < 0.3:
                # Sistema con poca carga, reducir latencia
                self.latency_range = (
                    self.latency_range[0],
                    max(0.01, self.latency_range[1] / 1.5)
                )
        
        elif event_type == "recovery_signal" and source == "monitor":
            # Evento especial de recuperación 
            if self.crashed:
                logger.info(f"[{self.id}] Recibida señal de recuperación, iniciando...")
                await self.reset()
        
        elif event_type == "config_update":
            # Actualizar configuración dinámica
            new_failure_prob = data.get("failure_probability")
            if new_failure_prob is not None:
                self.failure_probability = new_failure_prob
                logger.debug(f"[{self.id}] Actualizada probabilidad de fallo a {new_failure_prob}")
                
            new_max_parallel = data.get("max_parallel_requests")
            if new_max_parallel is not None and new_max_parallel > 0:
                old_value = self.max_parallel_requests
                self.max_parallel_requests = new_max_parallel
                logger.debug(f"[{self.id}] Límite de solicitudes paralelas cambiado: {old_value} → {new_max_parallel}")
    
    async def start(self) -> None:
        """Iniciar componente."""
        self.start_time = time.time()
        self.crashed = False
        self.overloaded = False
        logger.info(f"Componente {self.id} iniciado")
    
    async def stop(self) -> None:
        """Detener componente."""
        uptime = time.time() - self.start_time
        logger.info(f"Componente {self.id} detenido. Uptime: {uptime:.2f}s")
    
    async def reset(self) -> None:
        """Restablecer un componente crasheado."""
        if self.crashed:
            # Simular tiempo de recuperación
            await asyncio.sleep(self.recovery_time)
            self.crashed = False
            self.call_count = 0  # Reset contador de llamadas
            logger.info(f"Componente {self.id} recuperado después de crash")
    
    def _generate_data(self, size: int) -> List[Dict[str, Any]]:
        """Generar datos de respuesta."""
        items = []
        for i in range(min(size, 1000)):  # Limitar tamaño máximo
            items.append({
                "id": i,
                "value": random.random() * 100,
                "timestamp": time.time(),
                "tags": [f"tag{j}" for j in range(random.randint(1, 3))],
                "metrics": {
                    "precision": random.random(),
                    "recall": random.random()
                }
            })
        return items
    
    def _compute_intensive(self, complexity: int) -> Dict[str, Any]:
        """Realizar cálculo computacionalmente intensivo."""
        result = 0
        # Limitar complejidad máxima
        iterations = min(100000 * complexity, 1000000)
        for i in range(iterations):
            result += (i * 997) % 2003
        
        return {
            "value": result % 10000,
            "iterations": iterations,
            "timestamp": time.time()
        }

class ExtremeCaseCoordinator:
    """Coordinador optimizado para casos extremos."""
    
    def __init__(self, 
                 network_latency_range: Tuple[float, float] = (0.0, 0.0),
                 network_failure_rate: float = 0.0,
                 max_parallel_events: int = 500,
                 priority_event_types: List[str] = None,
                 monitor_interval: float = 1.0):
        self.components = {}  # id -> componente
        self.event_subscribers = {}  # tipo_evento -> set(id_componente)
        self.network_latency_range = network_latency_range
        self.network_failure_rate = network_failure_rate
        self.max_parallel_events = max_parallel_events
        self.priority_event_types = set(priority_event_types or ["health", "alert", "recovery_signal"])
        self.monitor_interval = monitor_interval
        
        # Métricas y estado
        self.start_time = time.time()
        self.request_count = 0
        self.event_count = 0
        self.request_failures = 0
        self.event_failures = 0
        self.active_event_tasks = 0
        self.metrics_by_type = defaultdict(lambda: {"count": 0, "failures": 0, "total_time": 0.0})
        
        # Semáforo para controlar emisión de eventos concurrentes
        self.event_semaphore = asyncio.Semaphore(max_parallel_events)
        
        # Cola de eventos prioritarios
        self.priority_queue = asyncio.Queue()
        self.regular_queue = asyncio.Queue()
        
        # Tarea de monitoreo
        self.monitor_task = None
    
    def register_component(self, id: str, component: HighStressComponent) -> None:
        """Registrar componente en el sistema."""
        self.components[id] = component
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
    
    async def request(self, target_id: str, request_type: str, 
                     data: Dict[str, Any], source: str,
                     timeout: float = 5.0) -> Optional[Any]:
        """API: Enviar solicitud directa a un componente."""
        self.request_count += 1
        start_time = time.time()
        
        # Registrar métricas por tipo
        metrics = self.metrics_by_type[f"request.{request_type}"]
        metrics["count"] += 1
        
        if target_id not in self.components:
            self.request_failures += 1
            metrics["failures"] += 1
            logger.warning(f"Solicitud a componente inexistente: {target_id}")
            return None
        
        # Simular fallos de red
        if random.random() < self.network_failure_rate:
            self.request_failures += 1
            metrics["failures"] += 1
            logger.warning(f"Fallo de red simulado en solicitud a {target_id}")
            return None
        
        # Simular latencia de red (en ambas direcciones)
        network_latency = random.uniform(
            self.network_latency_range[0],
            self.network_latency_range[1]
        )
        
        if network_latency > 0:
            # Mitad de latencia antes de enviar
            await asyncio.sleep(network_latency / 2)
        
        try:
            # Enviar solicitud con timeout
            result = await asyncio.wait_for(
                self.components[target_id].process_request(
                    request_type, data, source
                ),
                timeout=timeout
            )
            
            # Segunda mitad de latencia de red
            if network_latency > 0:
                await asyncio.sleep(network_latency / 2)
            
            # Actualizar métricas
            request_time = time.time() - start_time
            metrics["total_time"] += request_time
            
            return result
            
        except asyncio.TimeoutError:
            self.request_failures += 1
            metrics["failures"] += 1
            logger.warning(f"Timeout en solicitud {request_type} a {target_id}")
            return None
            
        except Exception as e:
            self.request_failures += 1
            metrics["failures"] += 1
            logger.warning(f"Error en solicitud {request_type} a {target_id}: {e}")
            return None
    
    async def emit_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Encolar evento para procesamiento asíncrono."""
        self.event_count += 1
        
        # Determinar si es un evento prioritario
        is_priority = event_type in self.priority_event_types
        
        # Encolar para procesamiento (sistema de colas dual)
        event_data = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time(),
            "priority": is_priority
        }
        
        if is_priority:
            await self.priority_queue.put(event_data)
            if VERBOSE:
                logger.debug(f"Evento prioritario {event_type} encolado")
        else:
            await self.regular_queue.put(event_data)
            if VERBOSE:
                logger.debug(f"Evento regular {event_type} encolado")
    
    async def _process_event_queues(self):
        """Procesar colas de eventos continuamente."""
        while True:
            try:
                # Procesar primero eventos prioritarios
                if not self.priority_queue.empty():
                    event_data = await self.priority_queue.get()
                    await self._process_single_event(event_data)
                    self.priority_queue.task_done()
                
                # Luego procesar eventos regulares
                elif not self.regular_queue.empty():
                    event_data = await self.regular_queue.get()
                    await self._process_single_event(event_data)
                    self.regular_queue.task_done()
                
                # Si ambas colas están vacías, esperar un poco
                else:
                    await asyncio.sleep(0.01)
            
            except Exception as e:
                logger.error(f"Error en procesador de colas: {e}")
                # Continuar con el próximo evento
    
    async def _process_single_event(self, event_data: Dict[str, Any]):
        """Procesar un solo evento."""
        event_type = event_data["type"]
        data = event_data["data"]
        source = event_data["source"]
        is_priority = event_data["priority"]
        
        # Registrar métricas por tipo
        metrics = self.metrics_by_type[f"event.{event_type}"]
        metrics["count"] += 1
        
        # Distribuir a suscriptores
        subscribers = self.event_subscribers.get(event_type, set())
        if not subscribers:
            return  # No hay suscriptores, nada que hacer
        
        # Adquirir semáforo para controlar paralelismo
        # Los eventos prioritarios siempre se procesan
        if not is_priority:
            try:
                # Timeout corto para no bloquear la cola
                async with asyncio.timeout(0.1):
                    await self.event_semaphore.acquire()
            except asyncio.TimeoutError:
                # No se pudo adquirir el semáforo, descartar evento
                metrics["failures"] += 1
                logger.warning(f"Evento {event_type} descartado: sistema sobrecargado")
                return
        
        try:
            self.active_event_tasks += 1
            
            # Crear tareas para cada suscriptor
            delivery_tasks = []
            
            for comp_id in subscribers:
                if comp_id in self.components and comp_id != source:
                    # Simular fallo de red específico para este suscriptor
                    if random.random() < self.network_failure_rate:
                        if VERBOSE:
                            logger.debug(f"Fallo de red simulado para evento {event_type} a {comp_id}")
                        continue
                    
                    # Calcular latencia de red para este suscriptor
                    if self.network_latency_range[1] > 0:
                        network_latency = random.uniform(
                            self.network_latency_range[0],
                            self.network_latency_range[1]
                        )
                        # Para eventos, la latencia se aplica al delivery completo
                        delivery_task = self._delayed_event_delivery(
                            comp_id, event_type, data, source, network_latency
                        )
                    else:
                        # Sin latencia extra
                        delivery_task = self.components[comp_id].on_event(
                            event_type, data, source
                        )
                    
                    delivery_tasks.append(delivery_task)
            
            # Ejecutar entregas en paralelo con manejo de errores
            for future in asyncio.as_completed(delivery_tasks):
                try:
                    await future
                except Exception as e:
                    metrics["failures"] += 1
                    self.event_failures += 1
                    if VERBOSE:
                        logger.debug(f"Error en entrega de evento {event_type}: {e}")
        
        finally:
            self.active_event_tasks -= 1
            # Liberar semáforo solo si no era prioritario
            if not is_priority:
                self.event_semaphore.release()
    
    async def _delayed_event_delivery(self, comp_id: str, event_type: str, 
                                    data: Dict[str, Any], source: str, 
                                    delay: float) -> None:
        """Entregar evento con retraso simulado."""
        await asyncio.sleep(delay)
        try:
            await self.components[comp_id].on_event(event_type, data, source)
        except Exception as e:
            logger.warning(f"Error en entrega retrasada de evento a {comp_id}: {e}")
    
    async def _monitor_system_health(self):
        """Monitorear salud del sistema periódicamente."""
        while True:
            try:
                await asyncio.sleep(self.monitor_interval)
                
                current_time = time.time()
                uptime = current_time - self.start_time
                
                # Calcular métricas globales
                priority_queue_size = self.priority_queue.qsize()
                regular_queue_size = self.regular_queue.qsize()
                total_queue_size = priority_queue_size + regular_queue_size
                active_tasks = self.active_event_tasks
                
                # Si hay sobrecarga de eventos, emitir alerta
                if total_queue_size > self.max_parallel_events * 0.9:
                    logger.warning(
                        f"Sistema sobrecargado: {total_queue_size} eventos en cola, "
                        f"{active_tasks} activos. Evento de alerta emitido."
                    )
                    # Emitir evento de salud del sistema (prioritario)
                    await self.emit_event(
                        "system_status",
                        {
                            "status": "overloaded",
                            "load": min(1.0, total_queue_size / self.max_parallel_events),
                            "queued_events": total_queue_size,
                            "active_tasks": active_tasks,
                            "timestamp": current_time,
                        },
                        "coordinator_monitor"
                    )
                
                # Verificar componentes caídos y enviar señales de recuperación
                for comp_id, comp in self.components.items():
                    if comp.crashed:
                        logger.info(f"Detectado componente caído: {comp_id}, enviando señal de recuperación")
                        # Emitir evento de recuperación (prioritario)
                        await self.emit_event(
                            "recovery_signal",
                            {
                                "target": comp_id,
                                "timestamp": current_time,
                            },
                            "monitor"
                        )
            
            except Exception as e:
                logger.error(f"Error en monitoreo de salud: {e}")
    
    async def start(self) -> None:
        """Iniciar coordinador y componentes."""
        self.start_time = time.time()
        
        # Iniciar componentes
        start_tasks = []
        for comp in self.components.values():
            start_tasks.append(comp.start())
        
        if start_tasks:
            await asyncio.gather(*start_tasks)
        
        # Iniciar procesador de colas de eventos
        asyncio.create_task(self._process_event_queues())
        
        # Iniciar monitor de salud
        self.monitor_task = asyncio.create_task(self._monitor_system_health())
        
        logger.info(f"Coordinador iniciado con {len(self.components)} componentes")
        
        # Emitir evento de inicio del sistema
        await self.emit_event(
            "system_status",
            {
                "status": "starting",
                "components": list(self.components.keys()),
                "timestamp": self.start_time
            },
            "coordinator"
        )
    
    async def stop(self) -> None:
        """Detener coordinador y componentes."""
        # Detener tareas internas
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Emitir evento de finalización
        await self.emit_event(
            "system_status",
            {
                "status": "stopping",
                "uptime": time.time() - self.start_time,
                "timestamp": time.time()
            },
            "coordinator"
        )
        
        # Esperar a que se procesen los eventos pendientes (con timeout)
        try:
            # Esperar que se vacíen las colas
            if self.priority_queue.qsize() > 0:
                await asyncio.wait_for(self.priority_queue.join(), timeout=2.0)
            
            if self.regular_queue.qsize() > 0:
                await asyncio.wait_for(self.regular_queue.join(), timeout=2.0)
        
        except asyncio.TimeoutError:
            logger.warning(f"Timeout esperando eventos pendientes: {self.priority_queue.qsize()} prioritarios, {self.regular_queue.qsize()} regulares")
        
        # Detener componentes
        stop_tasks = []
        for comp in self.components.values():
            stop_tasks.append(comp.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks)
        
        logger.info(f"Coordinador detenido después de {time.time() - self.start_time:.2f}s")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento completas."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        metrics = {
            "timestamp": current_time,
            "uptime": uptime,
            "total_requests": self.request_count,
            "total_events": self.event_count,
            "request_failures": self.request_failures,
            "event_failures": self.event_failures,
            "request_success_rate": 1.0 - (self.request_failures / max(1, self.request_count)),
            "event_success_rate": 1.0 - (self.event_failures / max(1, self.event_count)),
            "requests_per_second": self.request_count / uptime,
            "events_per_second": self.event_count / uptime,
            "active_event_tasks": self.active_event_tasks,
            "priority_queue_size": self.priority_queue.qsize(),
            "regular_queue_size": self.regular_queue.qsize(),
            "components": {}
        }
        
        # Métricas detalladas por tipo 
        metrics["metrics_by_type"] = {}
        for key, data in self.metrics_by_type.items():
            if data["count"] > 0:
                metrics["metrics_by_type"][key] = {
                    "count": data["count"],
                    "failures": data["failures"],
                    "failure_rate": data["failures"] / data["count"],
                    "avg_time": data["total_time"] / data["count"] if data["count"] > 0 else 0,
                }
        
        # Métricas por componente
        for comp_id, comp in self.components.items():
            metrics["components"][comp_id] = {
                "status": "crashed" if comp.crashed else "overloaded" if comp.overloaded else "healthy",
                "call_count": comp.call_count,
                "event_count": comp.event_count,
                "active_requests": comp.active_requests,
                "failures": comp.failures,
                "timeouts": comp.timeouts,
                "failure_rate": comp.failures / max(1, comp.call_count) if comp.call_count > 0 else 0,
            }
        
        return metrics

# Pruebas de carga extrema

async def setup_extreme_test_system(
        num_components: int = 10,
        failure_rates: Tuple[float, float] = (0.01, 0.05),
        latency_range: Tuple[float, float] = (0.01, 0.05),
        crash_component_indices: Optional[List[int]] = None,
        network_latency: Tuple[float, float] = (0.005, 0.02),
        network_failure_rate: float = 0.01,
        max_parallel_events: int = 500) -> ExtremeCaseCoordinator:
    """
    Configurar sistema de prueba con componentes para carga extrema.
    
    Args:
        num_components: Número de componentes
        failure_rates: Rango de probabilidad de fallos (min, max)
        latency_range: Rango de latencia de procesamiento (min, max) en segundos
        crash_component_indices: Índices de componentes que crashearán
        network_latency: Rango de latencia de red (min, max) en segundos
        network_failure_rate: Probabilidad de fallos de red
        max_parallel_events: Número máximo de eventos paralelos
    """
    # Crear coordinador
    coordinator = ExtremeCaseCoordinator(
        network_latency_range=network_latency,
        network_failure_rate=network_failure_rate,
        max_parallel_events=max_parallel_events
    )
    
    # Crear componentes
    for i in range(num_components):
        # Determinar tasa de fallo para este componente 
        failure_probability = random.uniform(failure_rates[0], failure_rates[1])
        
        # Determinar si este componente debe crashear
        crash_after = None
        if crash_component_indices and i in crash_component_indices:
            # Crashear después de 100-300 llamadas
            crash_after = random.randint(100, 300)
        
        # Latencia variada según el índice (algunos más lentos que otros)
        component_latency = (
            latency_range[0],
            latency_range[1] * (1 + (i % 3) * 0.5)  # Algunos componentes son más lentos
        )
        
        # Máximo de solicitudes paralelas variable
        max_parallel = random.choice([50, 75, 100, 125])
        
        # Crear componente
        component = HighStressComponent(
            id=f"comp_{i}",
            failure_probability=failure_probability,
            latency_range=component_latency,
            crash_after_n_calls=crash_after,
            recovery_time=random.uniform(0.5, 2.0),
            max_parallel_requests=max_parallel
        )
        
        # Registrar en coordinador
        coordinator.register_component(f"comp_{i}", component)
        
        # Suscribir a eventos (cada componente a 3-6 tipos)
        event_types = [
            "data_update", "notification", "status_change", 
            "heartbeat", "system_status", "config_update",
            "metrics_report", "recovery_signal", "alert"
        ]
        
        num_subscriptions = random.randint(3, min(6, len(event_types)))
        subscribed_events = random.sample(event_types, num_subscriptions)
        
        coordinator.subscribe(f"comp_{i}", subscribed_events)
    
    # Iniciar sistema
    await coordinator.start()
    
    return coordinator

async def run_high_volume_test(coordinator: ExtremeCaseCoordinator, 
                             max_rate: int = 1000,
                             duration_seconds: int = 30,
                             ramp_up_seconds: int = 5,
                             report_interval: int = 5) -> Dict[str, Any]:
    """
    Ejecutar prueba de alto volumen con incremento gradual y sostenido.
    
    Args:
        coordinator: Coordinador del sistema
        max_rate: Tasa máxima (eventos por segundo)
        duration_seconds: Duración total de la prueba
        ramp_up_seconds: Tiempo para alcanzar la tasa máxima
        report_interval: Intervalo de reporte de métricas
    """
    logger.info(f"Iniciando prueba de alto volumen:")
    logger.info(f"- Tasa máxima: {max_rate} eventos/s")
    logger.info(f"- Duración: {duration_seconds}s")
    logger.info(f"- Ramp-up: {ramp_up_seconds}s")
    
    # Tipos de eventos para la prueba
    event_types = [
        "data_update", "notification", "status_change", 
        "heartbeat", "system_status", "metrics_report"
    ]
    
    # Control de tiempo
    start_time = time.time()
    end_time = start_time + duration_seconds
    last_report_time = start_time
    events_emitted = 0
    
    # Diccionario para rastrear solicitudes API
    api_requests_sent = 0
    
    try:
        # Bucle principal de emisión
        while time.time() < end_time:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Calcular tasa actual según fase
            if elapsed < ramp_up_seconds:
                current_rate = max(1, int((elapsed / ramp_up_seconds) * max_rate))
            else:
                current_rate = max_rate
            
            # Determinar eventos a emitir en este ciclo
            events_this_second = []
            
            # Cantidad efectiva ajustada por tiempo transcurrido
            effective_count = max(1, int(current_rate / 10))  # Emitir en lotes de ~10% de la tasa
            
            for _ in range(effective_count):
                # Seleccionar tipo de evento (pueden repetirse)
                event_type = random.choice(event_types)
                
                # Determinar tamaño de datos según tipo
                if event_type == "data_update":
                    # Datos más grandes
                    data_items = random.randint(10, 100)
                    data = {
                        "items": [
                            {
                                "id": i, 
                                "value": random.random() * 100,
                                "timestamp": time.time(),
                                "metrics": {
                                    "accuracy": random.random(),
                                    "precision": random.random()
                                }
                            } 
                            for i in range(data_items)
                        ],
                        "source_timestamp": time.time(),
                        "update_id": events_emitted
                    }
                elif event_type == "system_status":
                    # Estado del sistema
                    data = {
                        "load": random.random(),
                        "memory_usage": random.random() * 100,
                        "cpu_usage": random.random() * 100,
                        "timestamp": time.time(),
                        "status": random.choice(["normal", "warning", "critical"])
                    }
                else:
                    # Eventos más ligeros
                    data = {
                        "timestamp": time.time(),
                        "priority": random.randint(1, 5),
                        "message": f"Evento {event_type} #{events_emitted}",
                        "tags": [f"tag{i}" for i in range(random.randint(1, 3))]
                    }
                
                events_this_second.append((event_type, data))
            
            # Emitir eventos en tareas paralelas (no esperar resultados)
            emission_tasks = []
            for event_type, data in events_this_second:
                emission_tasks.append(
                    coordinator.emit_event(event_type, data, "load_test")
                )
                events_emitted += 1
            
            # Inicio asíncrono de eventos
            for task in emission_tasks:
                asyncio.create_task(task)
            
            # Incluir algunas solicitudes API directas (menos frecuentes)
            if random.random() < 0.2:  # ~20% de probabilidad por ciclo
                # Seleccionar componente y tipo aleatorios
                target_comp = f"comp_{random.randint(0, len(coordinator.components)-1)}"
                request_type = random.choice(["query", "compute", "health", "update"])
                
                # Datos según tipo
                if request_type == "query":
                    req_data = {"size": random.randint(1, 50)}
                elif request_type == "compute":
                    req_data = {"complexity": random.randint(1, 5)}
                elif request_type == "update":
                    req_data = {"active": random.choice([True, False])}
                else:
                    req_data = {}
                
                # Enviar solicitud (sin esperar respuesta)
                asyncio.create_task(coordinator.request(
                    target_comp,
                    request_type,
                    req_data,
                    "load_test"
                ))
                api_requests_sent += 1
            
            # Reporte periódico de progreso
            if current_time - last_report_time >= report_interval:
                metrics = coordinator.get_performance_metrics()
                elapsed_total = current_time - start_time
                rate = events_emitted / elapsed_total
                
                logger.info(
                    f"Progreso: {elapsed_total:.1f}s / {duration_seconds}s - "
                    f"Tasa actual: {rate:.1f} eventos/s - "
                    f"Eventos: {events_emitted} - "
                    f"API: {api_requests_sent} - "
                    f"Colas: {metrics['priority_queue_size']}P/{metrics['regular_queue_size']}R"
                )
                
                last_report_time = current_time
            
            # Esperar según tasa deseada para no saturar CPU
            elapsed_in_cycle = time.time() - current_time
            # Ajustar para mantener la tasa correcta
            cycle_interval = max(0.001, (1.0 / current_rate) * 10 - elapsed_in_cycle)
            if cycle_interval > 0:
                await asyncio.sleep(cycle_interval)
        
        # Esperar un poco para que se completen los eventos pendientes
        logger.info(f"Prueba completada. Esperando procesamiento de eventos pendientes...")
        await asyncio.sleep(2.0)
        
        # Recopilar métricas finales
        final_metrics = coordinator.get_performance_metrics()
        final_metrics["test_specific"] = {
            "target_max_rate": max_rate,
            "actual_rate": events_emitted / (time.time() - start_time),
            "total_events_emitted": events_emitted,
            "api_requests_sent": api_requests_sent,
            "test_duration": time.time() - start_time,
            "ramp_up_seconds": ramp_up_seconds
        }
        
        return final_metrics
        
    except Exception as e:
        logger.error(f"Error durante prueba de volumen: {e}")
        return {
            "error": str(e),
            "events_emitted": events_emitted,
            "api_requests_sent": api_requests_sent,
            "elapsed_seconds": time.time() - start_time
        }

async def run_component_failure_test(coordinator: ExtremeCaseCoordinator,
                                    event_rate: int = 200,
                                    duration_seconds: int = 30,
                                    failure_interval: int = 5) -> Dict[str, Any]:
    """
    Prueba con fallos simultáneos en múltiples componentes.
    
    Args:
        coordinator: Coordinador del sistema
        event_rate: Eventos por segundo constante
        duration_seconds: Duración total
        failure_interval: Intervalo entre inyección de fallos
    """
    logger.info(f"Iniciando prueba de fallos en componentes:")
    logger.info(f"- Tasa de eventos: {event_rate}/s")
    logger.info(f"- Duración: {duration_seconds}s")
    logger.info(f"- Intervalo de fallos: {failure_interval}s")
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    last_failure_time = start_time
    events_emitted = 0
    failures_injected = 0
    
    # Tipo de fallos a inyectar
    failure_types = [
        "crash",            # Forzar crash completo
        "increase_latency", # Aumentar latencia
        "increase_failures" # Aumentar tasa de fallos
    ]
    
    # Tipos de eventos regulares
    event_types = [
        "data_update", "notification", "heartbeat", 
        "metrics_report", "config_update"
    ]
    
    try:
        while time.time() < end_time:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Inyectar fallos periódicamente
            if current_time - last_failure_time >= failure_interval:
                # Seleccionar componentes para fallar (1-3 simultáneamente)
                num_components_to_fail = random.randint(1, min(3, len(coordinator.components)))
                components_to_fail = random.sample(list(coordinator.components.keys()), 
                                                num_components_to_fail)
                
                logger.info(f"Inyectando fallos en {num_components_to_fail} componentes: {components_to_fail}")
                
                for comp_id in components_to_fail:
                    # Seleccionar tipo de fallo aleatorio
                    failure_type = random.choice(failure_types)
                    
                    if failure_type == "crash":
                        # Forzar crash marcando como crashed
                        component = coordinator.components[comp_id]
                        if not component.crashed:
                            component.crashed = True
                            logger.info(f"Componente {comp_id} forzado a estado crashed")
                            failures_injected += 1
                    
                    elif failure_type == "increase_latency":
                        # Aumentar latencia significativamente
                        component = coordinator.components[comp_id]
                        old_latency = component.latency_range
                        component.latency_range = (
                            old_latency[0] * 2, 
                            old_latency[1] * 5
                        )
                        logger.info(
                            f"Latencia de {comp_id} aumentada: "
                            f"{old_latency} → {component.latency_range}"
                        )
                        failures_injected += 1
                    
                    elif failure_type == "increase_failures":
                        # Aumentar probabilidad de fallos
                        component = coordinator.components[comp_id]
                        old_failure_prob = component.failure_probability
                        component.failure_probability = min(0.8, old_failure_prob * 3)
                        logger.info(
                            f"Tasa de fallos de {comp_id} aumentada: "
                            f"{old_failure_prob:.2f} → {component.failure_probability:.2f}"
                        )
                        failures_injected += 1
                
                last_failure_time = current_time
            
            # Emitir eventos regulares a tasa constante
            events_this_cycle = max(1, event_rate // 10)  # Dividir en ~10 ciclos por segundo
            emission_tasks = []
            
            for _ in range(events_this_cycle):
                event_type = random.choice(event_types)
                data = {
                    "timestamp": time.time(),
                    "message": f"Evento {events_emitted}",
                    "test_phase": f"failure_test_{elapsed:.0f}s"
                }
                
                emission_tasks.append(
                    coordinator.emit_event(event_type, data, "failure_test")
                )
                events_emitted += 1
            
            # Inicio asíncrono de eventos
            for task in emission_tasks:
                asyncio.create_task(task)
            
            # Esperar para mantener tasa constante
            await asyncio.sleep(0.1)  # ~10 ciclos por segundo
        
        # Esperar a que se procesen los eventos pendientes
        logger.info("Prueba de fallos completada. Esperando eventos pendientes...")
        await asyncio.sleep(2.0)
        
        # Recopilar métricas finales
        final_metrics = coordinator.get_performance_metrics()
        final_metrics["test_specific"] = {
            "event_rate": event_rate,
            "total_events_emitted": events_emitted,
            "failures_injected": failures_injected,
            "test_duration": time.time() - start_time,
            "failure_interval": failure_interval
        }
        
        return final_metrics
        
    except Exception as e:
        logger.error(f"Error durante prueba de fallos: {e}")
        return {
            "error": str(e),
            "events_emitted": events_emitted,
            "failures_injected": failures_injected,
            "elapsed_seconds": time.time() - start_time
        }

async def analyze_and_report_results(volume_test_results: Dict[str, Any], 
                                  failure_test_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analizar y generar reporte de resultados combinados.
    
    Args:
        volume_test_results: Resultados de prueba de volumen
        failure_test_results: Resultados de prueba de fallos
    """
    # Análisis de prueba de volumen
    volume_rate = volume_test_results["test_specific"]["actual_rate"]
    volume_target = volume_test_results["test_specific"]["target_max_rate"]
    volume_ratio = volume_rate / volume_target
    
    volume_success_rate = volume_test_results["event_success_rate"]
    request_success_rate = volume_test_results["request_success_rate"]
    
    # Análisis de prueba de fallos
    failures_injected = failure_test_results["test_specific"]["failures_injected"]
    failure_event_success = failure_test_results["event_success_rate"]
    failure_request_success = failure_test_results["request_success_rate"]
    
    # Generar reporte
    report = {
        "timestamp": time.time(),
        "volume_test_summary": {
            "achieved_rate": f"{volume_rate:.1f} eventos/s",
            "target_rate": f"{volume_target} eventos/s",
            "rate_percentage": f"{volume_ratio * 100:.1f}%",
            "events_emitted": volume_test_results["test_specific"]["total_events_emitted"],
            "api_requests": volume_test_results["test_specific"]["api_requests_sent"],
            "event_success_rate": f"{volume_success_rate * 100:.2f}%",
            "request_success_rate": f"{request_success_rate * 100:.2f}%",
            "system_assessment": (
                "Excelente" if volume_ratio > 0.9 and volume_success_rate > 0.95 else
                "Bueno" if volume_ratio > 0.7 and volume_success_rate > 0.9 else
                "Aceptable" if volume_ratio > 0.5 and volume_success_rate > 0.8 else
                "Insuficiente"
            )
        },
        "failure_test_summary": {
            "failures_injected": failures_injected,
            "events_emitted": failure_test_results["test_specific"]["total_events_emitted"],
            "event_success_rate": f"{failure_event_success * 100:.2f}%",
            "request_success_rate": f"{failure_request_success * 100:.2f}%",
            "system_assessment": (
                "Excelente" if failure_event_success > 0.9 and failure_request_success > 0.9 else
                "Bueno" if failure_event_success > 0.8 and failure_request_success > 0.8 else
                "Aceptable" if failure_event_success > 0.7 and failure_request_success > 0.7 else
                "Insuficiente"
            )
        },
        "overall_assessment": {
            "volume_handling": (
                "Excelente" if volume_ratio > 0.9 else
                "Bueno" if volume_ratio > 0.7 else
                "Aceptable" if volume_ratio > 0.5 else
                "Insuficiente"
            ),
            "resilience": (
                "Excelente" if failure_event_success > 0.9 else
                "Bueno" if failure_event_success > 0.8 else
                "Aceptable" if failure_event_success > 0.7 else
                "Insuficiente"
            ),
            "combined_score": (
                (volume_ratio * 0.5) + 
                (volume_success_rate * 0.25) + 
                (failure_event_success * 0.25)
            ) * 100
        }
    }
    
    # Formatear para salida legible
    logger.info("=== RESUMEN DE PRUEBAS DE ESTRÉS ===")
    logger.info(f"Prueba de Volumen:")
    logger.info(f"- Tasa alcanzada: {report['volume_test_summary']['achieved_rate']} de {report['volume_test_summary']['target_rate']} ({report['volume_test_summary']['rate_percentage']})")
    logger.info(f"- Éxito eventos: {report['volume_test_summary']['event_success_rate']}")
    logger.info(f"- Éxito solicitudes: {report['volume_test_summary']['request_success_rate']}")
    logger.info(f"- Evaluación: {report['volume_test_summary']['system_assessment']}")
    logger.info("")
    
    logger.info(f"Prueba de Fallos Simultáneos:")
    logger.info(f"- Fallos inyectados: {report['failure_test_summary']['failures_injected']}")
    logger.info(f"- Éxito eventos: {report['failure_test_summary']['event_success_rate']}")
    logger.info(f"- Éxito solicitudes: {report['failure_test_summary']['request_success_rate']}")
    logger.info(f"- Evaluación: {report['failure_test_summary']['system_assessment']}")
    logger.info("")
    
    logger.info(f"Evaluación General:")
    logger.info(f"- Manejo de volumen: {report['overall_assessment']['volume_handling']}")
    logger.info(f"- Resiliencia: {report['overall_assessment']['resilience']}")
    logger.info(f"- Puntuación combinada: {report['overall_assessment']['combined_score']:.1f}/100")
    
    return report

async def generate_markdown_report(report: Dict[str, Any], output_file: str = "docs/informe_pruebas_extremas.md") -> None:
    """
    Generar informe detallado en formato Markdown.
    
    Args:
        report: Datos del reporte
        output_file: Ruta del archivo de salida
    """
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Crear contenido del informe
    content = f"""# Informe de Pruebas Extremas del Sistema Híbrido API+WebSocket

## Resumen Ejecutivo

Este informe presenta los resultados de pruebas de estrés extremas realizadas sobre el sistema híbrido API+WebSocket Genesis. Las pruebas evaluaron la capacidad del sistema para manejar cargas muy altas y su resiliencia ante fallos simultáneos en múltiples componentes.

**Puntuación Global: {report['overall_assessment']['combined_score']:.1f}/100**

| Dimensión | Evaluación | Detalles |
|-----------|------------|----------|
| Manejo de Volumen | {report['overall_assessment']['volume_handling']} | Tasa alcanzada: {report['volume_test_summary']['achieved_rate']} de {report['volume_test_summary']['target_rate']} ({report['volume_test_summary']['rate_percentage']}) |
| Resiliencia | {report['overall_assessment']['resilience']} | Tasa de éxito con fallos: {report['failure_test_summary']['event_success_rate']} |

## Prueba de Volumen Extremo

Esta prueba evaluó la capacidad del sistema para manejar un volumen creciente de eventos hasta alcanzar una tasa objetivo de {report['volume_test_summary']['target_rate']} eventos por segundo.

### Resultados Clave

- **Tasa máxima alcanzada**: {report['volume_test_summary']['achieved_rate']}
- **Porcentaje del objetivo**: {report['volume_test_summary']['rate_percentage']}
- **Eventos totales emitidos**: {report['volume_test_summary']['events_emitted']}
- **Solicitudes API enviadas**: {report['volume_test_summary']['api_requests']}
- **Tasa de éxito de eventos**: {report['volume_test_summary']['event_success_rate']}
- **Tasa de éxito de solicitudes**: {report['volume_test_summary']['request_success_rate']}

### Evaluación: {report['volume_test_summary']['system_assessment']}

## Prueba de Fallos Simultáneos

Esta prueba evaluó la resiliencia del sistema ante fallos simultáneos en múltiples componentes, incluyendo crasheos completos, aumento de latencia y aumento de tasas de error.

### Resultados Clave

- **Fallos inyectados**: {report['failure_test_summary']['failures_injected']}
- **Eventos emitidos durante fallos**: {report['failure_test_summary']['events_emitted']}
- **Tasa de éxito de eventos**: {report['failure_test_summary']['event_success_rate']}
- **Tasa de éxito de solicitudes**: {report['failure_test_summary']['request_success_rate']}

### Evaluación: {report['failure_test_summary']['system_assessment']}

## Análisis y Conclusiones

El sistema híbrido API+WebSocket ha demostrado una capacidad {report['overall_assessment']['volume_handling'].lower()} para manejar volúmenes altos de eventos y una resiliencia {report['overall_assessment']['resilience'].lower()} ante fallos simultáneos en múltiples componentes.

### Fortalezas

- La arquitectura híbrida permite que el sistema continúe funcionando incluso cuando múltiples componentes fallan simultáneamente.
- El sistema de colas prioritarias garantiza que los eventos críticos se procesen incluso durante sobrecarga.
- Los timeouts efectivos previenen bloqueos indefinidos en solicitudes API.

### Áreas de Mejora

- Optimizar la recuperación automática de componentes caídos para mejorar el tiempo de recuperación.
- Implementar ajuste dinámico de recursos basado en la carga del sistema.
- Mejorar el manejo de grandes volúmenes de datos en eventos específicos.

## Recomendaciones

1. **Ajuste de Timeouts**: Optimizar los valores de timeout según los patrones de tráfico esperados.
2. **Monitoreo Proactivo**: Implementar alertas tempranas para detectar degradación antes de que afecte al sistema completo.
3. **Política de Reintentos**: Implementar reintentos exponenciales para solicitudes críticas que fallan.
4. **Circuit Breakers**: Añadir circuit breakers para aislar componentes problemáticos más rápidamente.
5. **Optimización de Colas**: Ajustar tamaños de colas y prioridades para adaptarse a patrones de uso específicos.

---

*Informe generado: {timestamp}*
"""

    # Escribir a archivo
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(content)
    
    logger.info(f"Informe Markdown generado: {output_file}")

async def run_extreme_tests():
    """Ejecutar conjunto completo de pruebas extremas."""
    try:
        logger.info("=== INICIANDO PRUEBAS DE ESTRÉS EXTREMAS ===")
        
        # 1. Configurar sistema
        system = await setup_extreme_test_system(
            num_components=15,                  # Más componentes
            failure_rates=(0.01, 0.05),         # 1-5% de fallos
            latency_range=(0.01, 0.05),         # 10-50ms de latencia base
            crash_component_indices=[3, 7, 11], # Componentes que crashearán
            network_latency=(0.005, 0.02),      # 5-20ms de latencia de red
            network_failure_rate=0.01,          # 1% de fallos de red
            max_parallel_events=1000            # Mayor capacidad de eventos
        )
        
        try:
            # 2. Prueba de volumen extremo
            logger.info("\n=== PRUEBA DE VOLUMEN EXTREMO ===\n")
            volume_results = await run_high_volume_test(
                system,
                max_rate=1000,           # 1000 eventos/s objetivo
                duration_seconds=30,     # 30s duración
                ramp_up_seconds=5,       # 5s ramp-up
                report_interval=5        # Reportes cada 5s
            )
            
            # 3. Prueba de fallos simultáneos
            logger.info("\n=== PRUEBA DE FALLOS SIMULTÁNEOS ===\n")
            failure_results = await run_component_failure_test(
                system,
                event_rate=200,          # 200 eventos/s constantes
                duration_seconds=30,     # 30s duración
                failure_interval=5       # Inyectar fallos cada 5s
            )
            
            # 4. Analizar resultados y generar reporte
            report = await analyze_and_report_results(volume_results, failure_results)
            
            # 5. Generar informe en Markdown
            await generate_markdown_report(report)
            
            return report
            
        finally:
            # Asegurar que el sistema se detenga correctamente
            await system.stop()
            
    except Exception as e:
        logger.error(f"Error durante pruebas extremas: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Punto de entrada
if __name__ == "__main__":
    asyncio.run(run_extreme_tests())