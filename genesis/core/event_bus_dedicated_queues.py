"""
Bus de eventos mejorado con colas dedicadas para el sistema Genesis.

Este módulo implementa un bus de eventos que utiliza colas dedicadas por componente
y mecanismos de monitoreo para evitar deadlocks y timeouts en las interacciones asíncronas.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, Any, Set, List, Optional, Callable, Coroutine

# Configuración del logger
logger = logging.getLogger(__name__)

class DedicatedQueueEventBus:
    """
    Bus de eventos con colas dedicadas por componente para evitar bloqueos.
    
    Este bus implementa:
    1. Una cola de eventos dedicada para cada componente
    2. Supervisión de tareas para detectar y resolver bloqueos
    3. Timeouts agresivos en operaciones asíncronas
    4. Mecanismos de limpieza para evitar fugas de memoria
    """
    
    def __init__(self, test_mode: bool = False):
        """
        Inicializar el bus de eventos con colas dedicadas.
        
        Args:
            test_mode: Si es True, usa timeouts más agresivos para pruebas
        """
        # Colas dedicadas por componente_id -> List[eventos]
        self.component_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        
        # Componentes registrados
        self.components: Dict[str, Any] = {}
        
        # Tareas asíncronas activas
        self.active_tasks: Set[asyncio.Task] = set()
        
        # Estado de los componentes
        self.component_last_active: Dict[str, float] = {}
        self.component_health: Dict[str, bool] = {}
        
        # Callbacks para notificaciones
        self.status_callbacks: List[Callable[[str, bool], Coroutine]] = []
        
        # Modo test para timeouts más agresivos
        self.test_mode = test_mode
        self.default_timeout = 0.5 if test_mode else 2.0
        self.restart_threshold = 1.0 if test_mode else 5.0
        
        # Flag para indicar si el bus está activo
        self.running = True
        
        # Estadísticas
        self.events_published = 0
        self.events_delivered = 0
        self.events_timed_out = 0
        
    def attach_component(self, component_id: str, component: Any) -> None:
        """
        Registrar un componente en el bus.
        
        Args:
            component_id: ID único del componente
            component: Instancia del componente
        """
        logger.debug(f"Registrando componente {component_id} en el bus de eventos")
        self.components[component_id] = component
        self.component_last_active[component_id] = time.time()
        self.component_health[component_id] = True
        
        # Iniciar tarea de procesamiento para este componente
        self._create_processing_task(component_id)
        
    def _create_processing_task(self, component_id: str) -> None:
        """
        Crear tarea de procesamiento para un componente.
        
        Args:
            component_id: ID del componente
        """
        task = asyncio.create_task(self._process_component_events(component_id))
        self.active_tasks.add(task)
        # Eliminar la tarea cuando termine
        task.add_done_callback(lambda t: self.active_tasks.discard(t))
        
    async def _process_component_events(self, component_id: str) -> None:
        """
        Procesar eventos para un componente específico.
        
        Args:
            component_id: ID del componente
        """
        component = self.components.get(component_id)
        if not component:
            logger.warning(f"Componente {component_id} no encontrado para procesar eventos")
            return
            
        queue = self.component_queues[component_id]
        logger.debug(f"Iniciando procesamiento de eventos para {component_id}")
        
        while self.running and component_id in self.components:
            try:
                # Obtener evento con timeout para evitar bloqueos
                event_type, data, source = await asyncio.wait_for(
                    queue.get(), 
                    timeout=self.default_timeout
                )
                
                start_time = time.time()
                logger.debug(f"Componente {component_id} procesando evento {event_type} de {source}")
                
                # Llamar al manejador del componente con timeout
                try:
                    result = await asyncio.wait_for(
                        component.handle_event(event_type, data, source),
                        timeout=self.default_timeout
                    )
                    # Actualizar timestamp de actividad
                    self.component_last_active[component_id] = time.time()
                    self.events_delivered += 1
                    
                    # Marcar la tarea como completada
                    queue.task_done()
                    
                    # Si resultado es un evento de respuesta, publicarlo
                    if "response_to" in data and data.get("response_type"):
                        await self.emit(
                            data["response_type"],
                            {"response": result, "original_event": event_type},
                            component_id
                        )
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout en componente {component_id} para evento {event_type}")
                    self.events_timed_out += 1
                    # Marcar el componente como no saludable si demasiados timeouts
                    self._register_component_timeout(component_id)
                    # Consideramos el evento como procesado aunque haya timeout
                    queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Error en componente {component_id} para evento {event_type}: {e}")
                    # Marcar el evento como completado a pesar del error
                    queue.task_done()
                    
                # Verificar tiempo total de procesamiento
                process_time = time.time() - start_time
                if process_time > 0.5:  # Advertencia si toma más de 500ms
                    logger.warning(f"Componente {component_id} tardó {process_time:.2f}s en procesar {event_type}")
                    
            except asyncio.TimeoutError:
                # No hay eventos en la cola, continuar esperando
                continue
                
            except Exception as e:
                logger.error(f"Error en bucle de procesamiento para {component_id}: {e}")
                # Breve pausa para evitar bucles de error a alta velocidad
                await asyncio.sleep(0.1)
                
        logger.debug(f"Finalizando procesamiento de eventos para {component_id}")
        
    def _register_component_timeout(self, component_id: str) -> None:
        """
        Registrar un timeout para un componente y actualizar su estado.
        
        Args:
            component_id: ID del componente que experimentó timeout
        """
        # Si componente está experimentando muchos timeouts, considerarlo no saludable
        # Este es un mecanismo simple, podría implementarse un contador de timeouts
        prev_health = self.component_health.get(component_id, True)
        self.component_health[component_id] = False
        
        # Notificar cambio de estado si cambió
        if prev_health and not self.component_health[component_id]:
            logger.warning(f"Componente {component_id} marcado como no saludable")
            # Ejecutar callbacks de estado
            for callback in self.status_callbacks:
                asyncio.create_task(callback(component_id, False))
                
    async def emit(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[List[Dict[str, Any]]]:
        """
        Emitir un evento a todos los componentes registrados.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente que origina el evento
            
        Returns:
            Lista de respuestas si las hay, o None
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} cuando el bus está detenido")
            return None
            
        logger.debug(f"Emitiendo evento {event_type} desde {source}")
        self.events_published += 1
        
        # Si es evento de respuesta, dirigirlo solo al destinatario
        if event_type.startswith("response.") and "target" in data:
            target = data["target"]
            if target in self.component_queues:
                await self.component_queues[target].put((event_type, data, source))
                return None
        
        # Poner el evento en la cola de cada componente (excepto el emisor)
        tasks = []
        for component_id, queue in self.component_queues.items():
            if component_id != source:  # No enviar el evento al emisor
                tasks.append(queue.put((event_type, data, source)))
                
        # Esperar a que todos los eventos se pongan en las colas
        if tasks:
            await asyncio.gather(*tasks)
            
        return None
        
    async def emit_with_response(
        self, 
        event_type: str, 
        data: Dict[str, Any], 
        source: str,
        timeout: float = None
    ) -> List[Dict[str, Any]]:
        """
        Emitir un evento y esperar respuestas.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente que origina el evento
            timeout: Timeout en segundos (usa default_timeout si es None)
            
        Returns:
            Lista de respuestas de los componentes
        """
        if timeout is None:
            timeout = self.default_timeout
            
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} cuando el bus está detenido")
            return []
            
        logger.debug(f"Emitiendo evento {event_type} con espera de respuesta desde {source}")
        
        # Crear cola de respuestas específica para este evento
        response_queue = asyncio.Queue()
        response_id = f"resp_{time.time()}_{event_type}"
        
        # Modificar datos para incluir información de respuesta
        event_data = data.copy()
        event_data["response_to"] = source
        event_data["response_id"] = response_id
        event_data["response_type"] = f"response.{event_type}"
        
        # Crear tarea para recolectar respuestas
        responses: List[Dict[str, Any]] = []
        response_count = 0
        expected_responses = len(self.components) - 1  # Todos excepto el emisor
        
        # Función para recolectar respuestas
        async def collect_responses():
            nonlocal response_count
            try:
                while response_count < expected_responses:
                    try:
                        response = await asyncio.wait_for(response_queue.get(), timeout=timeout)
                        responses.append(response)
                        response_count += 1
                        response_queue.task_done()
                    except asyncio.TimeoutError:
                        logger.debug(f"Timeout esperando respuestas para {event_type}")
                        break
            except Exception as e:
                logger.error(f"Error recolectando respuestas: {e}")
                
        # Iniciar recolección de respuestas
        collector_task = asyncio.create_task(collect_responses())
        self.active_tasks.add(collector_task)
        collector_task.add_done_callback(lambda t: self.active_tasks.discard(t))
        
        # Enviar evento a todos los componentes
        await self.emit(event_type, event_data, source)
        
        # Esperar a que se recolecten las respuestas o se alcance el timeout
        try:
            await asyncio.wait_for(collector_task, timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug(f"Timeout general esperando respuestas para {event_type}")
            
        return responses
        
    async def start_monitoring(self) -> None:
        """Iniciar monitoreo de componentes para detectar bloqueos."""
        logger.info("Iniciando monitoreo de componentes en el bus de eventos")
        
        monitor_task = asyncio.create_task(self._monitor_components())
        self.active_tasks.add(monitor_task)
        monitor_task.add_done_callback(lambda t: self.active_tasks.discard(t))
        
    async def _monitor_components(self) -> None:
        """Monitorear componentes para detectar inactividad."""
        while self.running:
            try:
                current_time = time.time()
                
                # Verificar componentes inactivos
                for component_id, last_active in list(self.component_last_active.items()):
                    if component_id not in self.components:
                        continue
                        
                    inactive_time = current_time - last_active
                    # Si un componente ha estado inactivo por mucho tiempo
                    if inactive_time > self.restart_threshold:
                        logger.warning(f"Componente {component_id} inactivo por {inactive_time:.2f}s")
                        
                        # Marcar como no saludable
                        prev_health = self.component_health.get(component_id, True)
                        self.component_health[component_id] = False
                        
                        # Notificar cambio de estado
                        if prev_health:
                            logger.warning(f"Componente {component_id} marcado como no saludable por inactividad")
                            for callback in self.status_callbacks:
                                asyncio.create_task(callback(component_id, False))
                
                # Verificar estado del bus
                if len(self.active_tasks) > 100:  # Demasiadas tareas activas
                    logger.warning(f"El bus tiene {len(self.active_tasks)} tareas activas, posible fuga de memoria")
                    
                # Dormir antes de siguiente verificación
                await asyncio.sleep(1.0 if self.test_mode else 2.0)
                
            except asyncio.CancelledError:
                logger.info("Monitoreo de componentes cancelado")
                break
                
            except Exception as e:
                logger.error(f"Error en monitoreo de componentes: {e}")
                await asyncio.sleep(0.5)
                
    def register_status_callback(self, callback: Callable[[str, bool], Coroutine]) -> None:
        """
        Registrar callback para cambios de estado de componentes.
        
        Args:
            callback: Función a llamar cuando cambia el estado (component_id, healthy)
        """
        self.status_callbacks.append(callback)
        
    async def stop(self) -> None:
        """Detener el bus de eventos y limpiar recursos."""
        logger.info("Deteniendo bus de eventos")
        self.running = False
        
        # Cancelar todas las tareas activas
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
                
        # Esperar a que las tareas terminen
        if self.active_tasks:
            await asyncio.sleep(0.2)
            
        # Limpiar colas
        for queue in self.component_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
                    
        logger.info(f"Bus de eventos detenido. Eventos: publicados={self.events_published}, "
                   f"entregados={self.events_delivered}, timeouts={self.events_timed_out}")