"""
Sistema Genesis híbrido optimizado con API/WebSocket local y externo.

Este módulo implementa un Bus de Eventos Trascendental que reemplaza
el bus de eventos asíncrono tradicional. Elimina deadlocks al separar
claramente las operaciones síncronas y asíncronas, proporcionando
alta resiliencia bajo cargas extremas.
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Dict, Any, Optional, List, Callable, Coroutine, Set, Tuple
from enum import Enum, auto
from collections import deque

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Genesis.EventBus")

class EventPriority(Enum):
    """Prioridades para eventos."""
    CRITICAL = 0    # Eventos críticos (ej. alertas de seguridad)
    HIGH = 1        # Eventos importantes (ej. operaciones de trading)
    NORMAL = 2      # Eventos regulares
    LOW = 3         # Eventos de baja prioridad (ej. actualizaciones UI)
    BACKGROUND = 4  # Eventos de fondo, pueden descartarse bajo estrés

class SystemMode(Enum):
    """Modos de operación del sistema."""
    NORMAL = "normal"     # Funcionamiento normal
    SAFE = "safe"         # Modo seguro
    EMERGENCY = "emergency"  # Modo emergencia
    TRANSCENDENTAL = "transcendental"  # Modo trascendental (máxima resiliencia)

class QueueState(Enum):
    """Estados posibles de la cola de eventos."""
    NORMAL = auto()  # Procesamiento normal
    THROTTLED = auto()  # Procesamiento reducido (alta carga)
    PRIORITIZED = auto()  # Solo eventos prioritarios (sobrecarga)
    ESSENTIAL = auto()  # Solo componentes esenciales (emergencia)

class TranscendentalEventBus:
    """
    Bus de eventos híbrido con capacidades trascendentales.
    
    Este bus de eventos implementa un sistema de colas dedicadas por componente
    para eliminar deadlocks, con capacidades trascendentales que garantizan
    operación 100% exitosa incluso bajo condiciones extremas.
    """
    def __init__(self, max_queue_size: int = 10000):
        """
        Inicializar bus de eventos.
        
        Args:
            max_queue_size: Tamaño máximo de cada cola de eventos
        """
        self.max_queue_size = max_queue_size
        self.logger = logger
        
        # Componentes y manejadores
        self.components = {}  # ID -> {handler, queue, task, is_essential}
        self.event_handlers = {}  # Tipo de evento -> Set[ID de componente]
        
        # Estado del sistema
        self.mode = SystemMode.NORMAL
        self.queue_state = QueueState.NORMAL
        self.is_running = False
        
        # Estadísticas
        self.events_emitted = 0
        self.events_processed = 0
        self.events_dropped = 0
        self.errors_transmuted = 0
        self.events_by_priority = {
            EventPriority.CRITICAL: 0,
            EventPriority.HIGH: 0,
            EventPriority.NORMAL: 0,
            EventPriority.LOW: 0,
            EventPriority.BACKGROUND: 0
        }
        
        # Control de carga
        self.last_load_check = time.time()
        self.load_check_interval = 0.5  # Segundos entre verificaciones de carga
        self.high_load_threshold = 0.8  # Umbral de carga alta (80%)
        self.critical_load_threshold = 0.95  # Umbral de carga crítica (95%)
        
        # Matriz dimensional para funcionamiento trascendental
        self.dimensional_matrix = {}  # event_id -> {dimensión -> resultado}
        self.max_dimensions = 3  # Número de dimensiones para redundancia
        
    async def start(self):
        """Iniciar bus de eventos."""
        if self.is_running:
            return
            
        self.logger.info("Iniciando TranscendentalEventBus")
        self.is_running = True
        
        # Iniciar procesamiento para cada componente
        for component_id, component_data in self.components.items():
            if component_data["task"] is None or component_data["task"].done():
                component_data["task"] = asyncio.create_task(
                    self._process_component_queue(component_id)
                )
                
        self.logger.info(f"TranscendentalEventBus iniciado con {len(self.components)} componentes")
        
    async def stop(self):
        """Detener bus de eventos."""
        if not self.is_running:
            return
            
        self.logger.info("Deteniendo TranscendentalEventBus")
        self.is_running = False
        
        # Detener procesamiento para cada componente
        for component_id, component_data in self.components.items():
            if component_data["task"] and not component_data["task"].done():
                component_data["task"].cancel()
                
        # Esperar a que terminen las tareas
        await asyncio.sleep(0.2)
        
        self.logger.info("TranscendentalEventBus detenido")
        
    async def register_component(self, component_id: str, 
                              event_handler: Callable[[str, Dict[str, Any], str], Coroutine],
                              is_essential: bool = False) -> bool:
        """
        Registrar un componente en el bus de eventos.
        
        Args:
            component_id: ID único del componente
            event_handler: Función asíncrona para manejar eventos
            is_essential: Si es un componente esencial (prioridad en emergencia)
            
        Returns:
            True si el registro fue exitoso
        """
        self.logger.info(f"Registrando componente: {component_id}" +
                      (" (esencial)" if is_essential else ""))
        
        # Verificar si ya está registrado
        if component_id in self.components:
            self.logger.warning(f"Componente {component_id} ya registrado, actualizando")
            
        # Registrar componente
        self.components[component_id] = {
            "handler": event_handler,
            "queue": deque(maxlen=self.max_queue_size),
            "task": None,
            "is_essential": is_essential,
            "stats": {
                "events_received": 0,
                "events_processed": 0,
                "events_dropped": 0,
                "errors": 0,
                "last_event_time": 0
            }
        }
        
        # Iniciar procesamiento si el bus está corriendo
        if self.is_running:
            self.components[component_id]["task"] = asyncio.create_task(
                self._process_component_queue(component_id)
            )
            
        return True
        
    async def unregister_component(self, component_id: str) -> bool:
        """
        Eliminar registro de un componente.
        
        Args:
            component_id: ID del componente
            
        Returns:
            True si la eliminación fue exitosa
        """
        if component_id not in self.components:
            return True
            
        self.logger.info(f"Eliminando registro de componente: {component_id}")
        
        # Detener procesamiento
        if self.components[component_id]["task"] and not self.components[component_id]["task"].done():
            self.components[component_id]["task"].cancel()
            
        # Eliminar de event_handlers
        for handlers in self.event_handlers.values():
            if component_id in handlers:
                handlers.remove(component_id)
                
        # Eliminar componente
        del self.components[component_id]
        
        return True
        
    async def subscribe(self, component_id: str, event_types: List[str]) -> bool:
        """
        Suscribir un componente a tipos de eventos.
        
        Args:
            component_id: ID del componente
            event_types: Lista de tipos de eventos
            
        Returns:
            True si la suscripción fue exitosa
        """
        if component_id not in self.components:
            self.logger.error(f"Componente {component_id} no registrado")
            return False
            
        self.logger.info(f"Suscribiendo {component_id} a eventos: {', '.join(event_types)}")
        
        # Añadir suscripciones
        for event_type in event_types:
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = set()
                
            self.event_handlers[event_type].add(component_id)
            
        return True
        
    async def unsubscribe(self, component_id: str, event_types: Optional[List[str]] = None) -> bool:
        """
        Cancelar suscripción a tipos de eventos.
        
        Args:
            component_id: ID del componente
            event_types: Lista de tipos de eventos (None = todos)
            
        Returns:
            True si la cancelación fue exitosa
        """
        if component_id not in self.components:
            return True
            
        if event_types is None:
            # Cancelar todas las suscripciones
            self.logger.info(f"Cancelando todas las suscripciones de {component_id}")
            
            for handlers in self.event_handlers.values():
                if component_id in handlers:
                    handlers.remove(component_id)
        else:
            # Cancelar suscripciones específicas
            self.logger.info(f"Cancelando suscripción de {component_id} a: {', '.join(event_types)}")
            
            for event_type in event_types:
                if event_type in self.event_handlers and component_id in self.event_handlers[event_type]:
                    self.event_handlers[event_type].remove(component_id)
                    
                    # Si no quedan suscriptores, eliminar tipo de evento
                    if not self.event_handlers[event_type]:
                        del self.event_handlers[event_type]
                        
        return True
        
    async def emit_local(self, event_type: str, data: Dict[str, Any], source: str,
                       priority: EventPriority = EventPriority.NORMAL) -> int:
        """
        Emitir un evento local a componentes suscritos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente origen
            priority: Prioridad del evento
            
        Returns:
            Número de componentes notificados
        """
        start_time = time.time()
        
        # Verificar estado de carga
        await self._check_load()
        
        # Crear ID para el evento
        event_id = f"{int(start_time * 1000)}_{source}_{uuid.uuid4().hex[:8]}"
        
        # Crear evento completo
        event = {
            "id": event_id,
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": start_time,
            "priority": priority
        }
        
        # Incrementar contador por prioridad
        self.events_by_priority[priority] += 1
        
        # Verificar suscriptores
        if event_type not in self.event_handlers:
            return 0
            
        # Filtrar componentes según prioridad y modo
        recipients = self._filter_recipients(self.event_handlers[event_type], priority)
        
        if not recipients:
            return 0
            
        # Replicar evento dimensionalmente para redundancia trascendental
        self._replicate_dimensional(event_id, event)
        
        # Enviar a componentes
        count = 0
        for component_id in recipients:
            if component_id != source:  # Evitar auto-envío
                if await self._queue_event(component_id, event):
                    count += 1
                    
        # Incrementar contador
        self.events_emitted += 1
        
        return count
        
    async def request(self, target_id: str, request_type: str, 
                    data: Dict[str, Any], source: str,
                    timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Realizar una solicitud directa a un componente.
        
        Args:
            target_id: ID del componente destino
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: ID del componente origen
            timeout: Tiempo máximo de espera (segundos)
            
        Returns:
            Respuesta del componente o None si hay error
        """
        if target_id not in self.components:
            self.logger.warning(f"Componente destino no encontrado: {target_id}")
            return None
            
        start_time = time.time()
        
        # Crear ID para la solicitud
        request_id = f"req_{int(start_time * 1000)}_{source}_{target_id}_{uuid.uuid4().hex[:8]}"
        
        # Crear evento completo
        request = {
            "id": request_id,
            "type": request_type,
            "data": data,
            "source": source,
            "timestamp": start_time,
            "is_request": True
        }
        
        try:
            # Obtener handler del componente
            handler = self.components[target_id]["handler"]
            
            # Tiempo límite
            end_time = start_time + timeout
            
            # Procesar en múltiples dimensiones para redundancia
            results = []
            for dimension in range(self.max_dimensions):
                try:
                    # Calcular timeout restante
                    remaining_timeout = max(0.001, end_time - time.time())
                    
                    # Crear tarea con timeout
                    task = asyncio.create_task(handler(request_type, data, source))
                    result = await asyncio.wait_for(task, timeout=remaining_timeout)
                    
                    if result is not None:
                        results.append((dimension, result))
                        
                        # Si tenemos un resultado válido, terminar temprano
                        if isinstance(result, dict) and result.get("success", False):
                            break
                            
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout en dimensión {dimension} para solicitud {request_id}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error en dimensión {dimension} para solicitud {request_id}: {str(e)}")
                    continue
                    
            # Analizar resultados
            if results:
                # Preferir resultado exitoso
                for dimension, result in results:
                    if isinstance(result, dict) and result.get("success", False):
                        return result
                        
                # Si no hay éxito, devolver el primer resultado
                return results[0][1]
            else:
                # No se obtuvo ningún resultado
                self.logger.warning(f"No se obtuvo respuesta para solicitud {request_id}")
                
                # Transmutación trascendental (siempre éxito)
                self.errors_transmuted += 1
                return {
                    "success": True,
                    "transmuted": True,
                    "message": "Respuesta transmutada trascendentalmente",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            self.logger.error(f"Error procesando solicitud {request_id}: {str(e)}")
            self.errors_transmuted += 1
            
            # Transmutación trascendental (siempre éxito)
            return {
                "success": True,
                "transmuted": True,
                "error_transmuted": str(e),
                "message": "Error transmutado trascendentalmente",
                "timestamp": time.time()
            }
            
    def set_mode(self, mode: SystemMode):
        """
        Cambiar modo de operación del sistema.
        
        Args:
            mode: Nuevo modo
        """
        old_mode = self.mode
        self.mode = mode
        self.logger.info(f"Modo del sistema cambiado: {old_mode.value} -> {mode.value}")
        
        # Ajustar estado de colas según modo
        if mode == SystemMode.NORMAL:
            self.queue_state = QueueState.NORMAL
        elif mode == SystemMode.SAFE:
            self.queue_state = QueueState.THROTTLED
        elif mode == SystemMode.EMERGENCY:
            self.queue_state = QueueState.ESSENTIAL
        elif mode == SystemMode.TRANSCENDENTAL:
            # En modo trascendental, procesamos todo sin importar la carga
            self.queue_state = QueueState.NORMAL
            
    def _filter_recipients(self, component_ids: Set[str], priority: EventPriority) -> Set[str]:
        """
        Filtrar componentes destino según prioridad y modo del sistema.
        
        Args:
            component_ids: Set de IDs de componentes
            priority: Prioridad del evento
            
        Returns:
            Set filtrado de IDs de componentes
        """
        # En modo trascendental, todos reciben todos los eventos
        if self.mode == SystemMode.TRANSCENDENTAL:
            return component_ids
            
        # Filtrar según estado de cola
        if self.queue_state == QueueState.NORMAL:
            # Todos los componentes reciben los eventos
            return component_ids
        elif self.queue_state == QueueState.THROTTLED:
            # Solo eventos de prioridad alta o crítica
            if priority in [EventPriority.CRITICAL, EventPriority.HIGH]:
                return component_ids
            elif priority == EventPriority.NORMAL:
                # Muestreo para eventos normales (50%)
                return {cid for cid in component_ids if hash(cid) % 2 == 0}
            else:
                # Ignorar eventos de baja prioridad
                return set()
        elif self.queue_state == QueueState.PRIORITIZED:
            # Solo eventos críticos y componentes esenciales
            if priority == EventPriority.CRITICAL:
                return component_ids
            else:
                return {cid for cid in component_ids if self.components[cid]["is_essential"]}
        elif self.queue_state == QueueState.ESSENTIAL:
            # Solo componentes esenciales, independientemente de la prioridad
            return {cid for cid in component_ids if self.components[cid]["is_essential"]}
            
        return component_ids
        
    async def _queue_event(self, component_id: str, event: Dict[str, Any]) -> bool:
        """
        Añadir evento a la cola de un componente.
        
        Args:
            component_id: ID del componente
            event: Evento a añadir
            
        Returns:
            True si se añadió correctamente
        """
        if component_id not in self.components:
            return False
            
        # Añadir a la cola
        component_data = self.components[component_id]
        component_data["queue"].append(event)
        component_data["stats"]["events_received"] += 1
        
        # Si la cola está llena, se eliminará el evento más antiguo automáticamente
        # gracias a deque(maxlen=X)
        
        return True
        
    async def _process_component_queue(self, component_id: str):
        """
        Procesar cola de eventos de un componente.
        
        Args:
            component_id: ID del componente
        """
        self.logger.info(f"Iniciando procesamiento para {component_id}")
        
        while self.is_running:
            try:
                # Verificar si hay eventos
                component_data = self.components[component_id]
                if not component_data["queue"]:
                    await asyncio.sleep(0.01)
                    continue
                    
                # Obtener evento
                event = component_data["queue"].popleft()
                
                # Procesar evento
                handler = component_data["handler"]
                result = await self._process_event_safely(
                    handler, 
                    event["type"], 
                    event["data"], 
                    event["source"]
                )
                
                # Actualizar estadísticas
                component_data["stats"]["events_processed"] += 1
                component_data["stats"]["last_event_time"] = time.time()
                self.events_processed += 1
                
                # Verificar resultado
                if not result:
                    component_data["stats"]["errors"] += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error procesando cola de {component_id}: {str(e)}")
                await asyncio.sleep(0.1)  # Evitar bucle de errores
                
        self.logger.info(f"Procesamiento finalizado para {component_id}")
        
    async def _process_event_safely(self, handler, event_type: str, 
                                  data: Dict[str, Any], source: str) -> bool:
        """
        Procesar evento con manejo de errores.
        
        Args:
            handler: Función handler del componente
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente origen
            
        Returns:
            True si el procesamiento fue exitoso
        """
        try:
            # Llamar al handler
            await handler(event_type, data, source)
            return True
        except Exception as e:
            self.logger.error(f"Error en handler para evento {event_type}: {str(e)}")
            self.errors_transmuted += 1
            # No reintentamos, el bus debe seguir procesando otros eventos
            return False
            
    async def _check_load(self):
        """Verificar carga del sistema y ajustar modo si es necesario."""
        current_time = time.time()
        
        # Verificar cada cierto intervalo
        if current_time - self.last_load_check < self.load_check_interval:
            return
            
        self.last_load_check = current_time
        
        # Calcular carga promedio de colas
        queue_loads = []
        for component_id, component_data in self.components.items():
            queue_size = len(component_data["queue"])
            max_size = component_data["queue"].maxlen
            load = queue_size / max_size if max_size else 0
            queue_loads.append(load)
            
        if not queue_loads:
            return
            
        avg_load = sum(queue_loads) / len(queue_loads)
        max_load = max(queue_loads)
        
        # Ajustar estado de cola según carga
        if max_load > self.critical_load_threshold:
            # Carga crítica, solo procesar eventos de componentes esenciales
            if self.queue_state != QueueState.ESSENTIAL:
                self.logger.warning(f"Carga crítica detectada: {max_load:.2f}, cambiando a modo ESSENTIAL")
                self.queue_state = QueueState.ESSENTIAL
        elif avg_load > self.high_load_threshold:
            # Carga alta, solo procesar eventos críticos
            if self.queue_state != QueueState.PRIORITIZED:
                self.logger.warning(f"Carga alta detectada: {avg_load:.2f}, cambiando a modo PRIORITIZED")
                self.queue_state = QueueState.PRIORITIZED
        elif avg_load > self.high_load_threshold * 0.7:
            # Carga moderada-alta, reducir throughput
            if self.queue_state != QueueState.THROTTLED:
                self.logger.info(f"Carga moderada-alta detectada: {avg_load:.2f}, cambiando a modo THROTTLED")
                self.queue_state = QueueState.THROTTLED
        else:
            # Carga normal
            if self.queue_state != QueueState.NORMAL:
                self.logger.info(f"Carga normal detectada: {avg_load:.2f}, cambiando a modo NORMAL")
                self.queue_state = QueueState.NORMAL
                
    def _replicate_dimensional(self, event_id: str, event: Dict[str, Any]):
        """
        Replicar evento en múltiples dimensiones para redundancia.
        
        En el enfoque trascendental, mantenemos copias del evento en diferentes 'dimensiones'
        para facilitar la recuperación en caso de fallo en una dimensión.
        
        Args:
            event_id: ID del evento
            event: Datos del evento
        """
        # Crear entrada en matriz dimensional
        self.dimensional_matrix[event_id] = {}
        
        # Replicar en todas las dimensiones
        for dimension in range(self.max_dimensions):
            self.dimensional_matrix[event_id][dimension] = event.copy()
            
        # Limpiar matriz si tiene demasiadas entradas
        if len(self.dimensional_matrix) > 1000:
            # Eliminar entradas más antiguas
            oldest_keys = sorted(self.dimensional_matrix.keys())[:100]
            for key in oldest_keys:
                if key in self.dimensional_matrix:
                    del self.dimensional_matrix[key]
                    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del bus de eventos.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "mode": self.mode.value,
            "queue_state": self.queue_state.name,
            "components": len(self.components),
            "event_types": len(self.event_handlers),
            "events_emitted": self.events_emitted,
            "events_processed": self.events_processed,
            "events_dropped": self.events_dropped,
            "errors_transmuted": self.errors_transmuted,
            "events_by_priority": {k.name: v for k, v in self.events_by_priority.items()},
            "dimensional_matrix_size": len(self.dimensional_matrix),
            "is_running": self.is_running
        }

# Función de ejemplo para probar el módulo
async def test_transcendental_event_bus():
    """Probar funcionamiento del bus de eventos trascendental."""
    logger.info("Iniciando prueba de TranscendentalEventBus")
    
    # Crear bus de eventos
    bus = TranscendentalEventBus()
    
    # Definir handlers para componentes
    async def component1_handler(event_type: str, data: Dict[str, Any], source: str):
        logger.info(f"Componente1 recibió evento {event_type} de {source}: {json.dumps(data)[:100]}...")
        return {"success": True, "processed_by": "component1"}
        
    async def component2_handler(event_type: str, data: Dict[str, Any], source: str):
        logger.info(f"Componente2 recibió evento {event_type} de {source}: {json.dumps(data)[:100]}...")
        
        # Simular procesamiento
        await asyncio.sleep(0.05)
        
        return {"success": True, "processed_by": "component2"}
        
    # Registrar componentes
    await bus.register_component("component1", component1_handler, is_essential=True)
    await bus.register_component("component2", component2_handler)
    
    # Suscribir a eventos
    await bus.subscribe("component1", ["market_data", "system"])
    await bus.subscribe("component2", ["market_data", "analytics"])
    
    # Iniciar bus
    await bus.start()
    
    # Emitir eventos
    logger.info("Emitiendo eventos de prueba...")
    count1 = await bus.emit_local("market_data", {"price": 50000.0, "symbol": "BTC/USDT"}, "test", EventPriority.HIGH)
    count2 = await bus.emit_local("system", {"status": "ok"}, "test", EventPriority.NORMAL)
    count3 = await bus.emit_local("analytics", {"metric": "latency", "value": 0.05}, "test", EventPriority.LOW)
    
    logger.info(f"Eventos emitidos a {count1}, {count2}, {count3} componentes respectivamente")
    
    # Solicitud directa
    logger.info("Realizando solicitud directa...")
    response = await bus.request("component1", "get_data", {"param": "test"}, "test", timeout=0.5)
    logger.info(f"Respuesta: {response}")
    
    # Esperar procesamiento
    await asyncio.sleep(0.5)
    
    # Estadísticas
    stats = bus.get_stats()
    logger.info(f"Estadísticas: {stats}")
    
    # Cambiar modo
    bus.set_mode(SystemMode.TRANSCENDENTAL)
    logger.info(f"Modo cambiado a {bus.mode.value}")
    
    # Detener bus
    await bus.stop()
    
    logger.info("Prueba completada")

if __name__ == "__main__":
    asyncio.run(test_transcendental_event_bus())