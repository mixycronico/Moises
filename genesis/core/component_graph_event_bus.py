"""
Bus de eventos basado en grafo de dependencias para el sistema Genesis.

Este módulo implementa un bus de eventos que utiliza un grafo dirigido acíclico (DAG)
para gestionar explícitamente las dependencias entre componentes y eliminar deadlocks.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Dict, Set, List, Any, Optional, Callable, Coroutine, Tuple

# Configuración del logger
logger = logging.getLogger(__name__)

class CircularDependencyError(Exception):
    """Error que indica una dependencia circular entre componentes."""
    pass

class ComponentGraphEventBus:
    """
    Bus de eventos basado en grafo de dependencias explícitas.
    
    Este bus:
    1. Organiza componentes en un grafo dirigido acíclico (DAG).
    2. Procesa eventos siguiendo un orden topológico, eliminando deadlocks.
    3. Gestiona la concurrencia con colas dedicadas por componente.
    4. Proporciona detección y prevención de dependencias circulares.
    """
    
    def __init__(self, test_mode: bool = False, max_queue_size: int = 100):
        """
        Inicializar el bus de eventos basado en grafo.
        
        Args:
            test_mode: Usar timeouts más agresivos si es True
            max_queue_size: Tamaño máximo de cada cola de eventos
        """
        # Componentes registrados
        self.components: Dict[str, Any] = {}
        
        # Grafo de dependencias
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)  # componente -> dependencias
        self.reverse_deps: Dict[str, Set[str]] = defaultdict(set)  # componente -> dependientes
        
        # Colas de eventos para cada componente
        self.event_queues: Dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=max_queue_size)
        )
        
        # Tareas activas de procesamiento por componente
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Orden topológico actual de componentes
        self.topological_order: List[str] = []
        
        # Estado de los componentes
        self.component_health: Dict[str, bool] = {}  # component_id -> salud
        self.component_last_active: Dict[str, float] = {}  # component_id -> timestamp
        
        # Callbacks para notificaciones
        self.status_callbacks: List[Callable[[str, Dict[str, Any]], Coroutine]] = []
        
        # Modo test para timeouts más agresivos
        self.test_mode = test_mode
        self.default_timeout = 0.5 if test_mode else 2.0
        
        # Estado del bus
        self.running = True
        
        # Estadísticas
        self.events_published = 0
        self.events_delivered = 0
        self.events_timed_out = 0
        self.components_processing: Dict[str, int] = defaultdict(int)  # component_id -> eventos procesando
        self.start_time = time.time()
        
    def register_component(self, component_id: str, component: Any, 
                          depends_on: Optional[List[str]] = None) -> None:
        """
        Registrar un componente con sus dependencias explícitas.
        
        Args:
            component_id: ID único del componente
            component: Instancia del componente
            depends_on: Lista de IDs de componentes de los que depende
        
        Raises:
            CircularDependencyError: Si la dependencia crearía un ciclo
        """
        logger.debug(f"Registrando componente {component_id}")
        
        # Verificar si ya existe
        if component_id in self.components:
            logger.warning(f"Componente {component_id} ya registrado, actualizando")
        
        # Registrar componente
        self.components[component_id] = component
        self.component_health[component_id] = True
        self.component_last_active[component_id] = time.time()
        
        # Registrar dependencias
        if depends_on:
            # Verificar que todas las dependencias existen
            for dep_id in depends_on:
                if dep_id not in self.components:
                    logger.warning(f"Dependencia {dep_id} no está registrada como componente")
            
            # Añadir dependencias
            self.dependencies[component_id].update(depends_on)
            
            # Actualizar dependencias inversas
            for dep_id in depends_on:
                self.reverse_deps[dep_id].add(component_id)
            
            # Verificar ciclos
            try:
                self._update_topological_order()
            except CircularDependencyError as e:
                # Eliminar las dependencias que causaron el ciclo
                self.dependencies[component_id].clear()
                for dep_id in depends_on:
                    self.reverse_deps[dep_id].discard(component_id)
                raise CircularDependencyError(f"Error al registrar {component_id}: {e}")
        
        # Iniciar tarea de procesamiento
        self._start_processing_task(component_id)
        
    def _start_processing_task(self, component_id: str) -> None:
        """
        Iniciar tarea de procesamiento para un componente.
        
        Args:
            component_id: ID del componente
        """
        if component_id in self.active_tasks and not self.active_tasks[component_id].done():
            logger.debug(f"Cancelando tarea existente para {component_id}")
            self.active_tasks[component_id].cancel()
        
        logger.debug(f"Iniciando tarea de procesamiento para {component_id}")
        task = asyncio.create_task(self._process_component_events(component_id))
        self.active_tasks[component_id] = task
        task.add_done_callback(lambda t: self._handle_task_done(component_id, t))
        
    def _handle_task_done(self, component_id: str, task: asyncio.Task) -> None:
        """
        Manejar finalización de una tarea de procesamiento.
        
        Args:
            component_id: ID del componente
            task: Tarea finalizada
        """
        # Eliminar de tareas activas
        self.active_tasks.pop(component_id, None)
        
        # Verificar si terminó con error
        if task.done() and not task.cancelled():
            try:
                exception = task.exception()
                if exception:
                    logger.error(f"Tarea de {component_id} terminó con error: {exception}")
                    self.component_health[component_id] = False
            except asyncio.CancelledError:
                logger.debug(f"Tarea de {component_id} cancelada")
        
        # Reiniciar si el componente sigue registrado y el bus activo
        if component_id in self.components and self.running:
            logger.debug(f"Reiniciando tarea de procesamiento para {component_id}")
            self._start_processing_task(component_id)
            
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
            
        queue = self.event_queues[component_id]
        logger.debug(f"Iniciando procesamiento de eventos para {component_id}")
        
        while self.running and component_id in self.components:
            try:
                # Obtener evento con timeout
                event_type, data, source = await asyncio.wait_for(
                    queue.get(), 
                    timeout=self.default_timeout
                )
                
                start_time = time.time()
                self.components_processing[component_id] += 1
                
                logger.debug(f"Componente {component_id} procesando evento {event_type} de {source}")
                
                # Procesar evento con timeout
                try:
                    result = await asyncio.wait_for(
                        component.handle_event(event_type, data, source),
                        timeout=self.default_timeout
                    )
                    
                    # Actualizar timestamp y estadísticas
                    self.component_last_active[component_id] = time.time()
                    self.events_delivered += 1
                    queue.task_done()
                    
                    # Si hay respuesta y se espera una
                    if "response_to" in data and "response_type" in data:
                        # Emitir respuesta
                        target = data["response_to"]
                        response_type = data["response_type"]
                        await self.emit(
                            response_type,
                            {
                                "original_event": event_type,
                                "response": result,
                                "target": target
                            },
                            component_id
                        )
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout al procesar evento {event_type} en {component_id}")
                    queue.task_done()
                    self.events_timed_out += 1
                    self.component_health[component_id] = False
                    
                except Exception as e:
                    logger.error(f"Error al procesar evento {event_type} en {component_id}: {e}")
                    queue.task_done()
                    
                finally:
                    self.components_processing[component_id] -= 1
                    
                # Verificar tiempo de procesamiento
                process_time = time.time() - start_time
                if process_time > 0.5:
                    logger.warning(f"Componente {component_id} tardó {process_time:.2f}s en procesar {event_type}")
                
            except asyncio.TimeoutError:
                # No hay eventos en la cola
                continue
                
            except asyncio.CancelledError:
                logger.debug(f"Tarea de procesamiento de {component_id} cancelada")
                break
                
            except Exception as e:
                logger.error(f"Error inesperado en procesamiento de {component_id}: {e}")
                await asyncio.sleep(0.1)  # Evitar bucle de error a alta velocidad
                
    def _update_topological_order(self) -> None:
        """
        Actualizar el orden topológico de los componentes.
        
        Raises:
            CircularDependencyError: Si se detecta una dependencia circular
        """
        # Inicializar grados de entrada
        in_degree = {node: 0 for node in self.components}
        for node, deps in self.dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Nodos con grado 0 (sin dependencias)
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result = []
        
        # Algoritmo de Kahn para ordenamiento topológico
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Reducir grados de entrada de dependientes
            for dependent in self.reverse_deps.get(current, set()):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        # Verificar si hay ciclos
        if len(result) != len(self.components):
            # Encontrar el ciclo para un mensaje de error más útil
            remaining = [node for node, degree in in_degree.items() if degree > 0]
            cycle_components = ", ".join(remaining)
            raise CircularDependencyError(
                f"Dependencia circular detectada entre los componentes: {cycle_components}"
            )
        
        # Actualizar orden topológico
        self.topological_order = result
        logger.debug(f"Orden topológico actualizado: {' -> '.join(self.topological_order)}")
        
    async def emit(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Emitir un evento siguiendo el orden topológico de componentes.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente que origina el evento
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} cuando el bus está detenido")
            return
            
        logger.debug(f"Emitiendo evento {event_type} desde {source}")
        self.events_published += 1
        
        # Si no tenemos orden topológico, actualizarlo
        if not self.topological_order or len(self.topological_order) != len(self.components):
            try:
                self._update_topological_order()
            except CircularDependencyError as e:
                logger.error(f"Error al emitir evento {event_type}: {e}")
                return
        
        # Si es un evento de respuesta dirigido
        if (event_type.startswith("response.") or "target" in data) and "target" in data:
            target = data["target"]
            if target in self.event_queues:
                try:
                    await asyncio.wait_for(
                        self.event_queues[target].put((event_type, data, source)),
                        timeout=self.default_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout al encolar evento {event_type} para {target}")
                except asyncio.QueueFull:
                    logger.warning(f"Cola llena para {target}, evento {event_type} descartado")
            return
            
        # Para eventos normales, seguir orden topológico
        tasks = []
        for component_id in self.topological_order:
            # No enviar al emisor ni a componentes fuera del DAG
            if component_id != source and component_id in self.components:
                try:
                    # Agregar tarea para poner evento en cola
                    task = asyncio.create_task(
                        self.event_queues[component_id].put((event_type, data, source))
                    )
                    tasks.append(task)
                except asyncio.QueueFull:
                    logger.warning(f"Cola llena para {component_id}, evento {event_type} descartado")
        
        # Esperar a que se completen todas las tareas de encolado
        if tasks:
            try:
                await asyncio.wait(tasks, timeout=self.default_timeout)
            except Exception as e:
                logger.error(f"Error al encolar eventos para {event_type}: {e}")
                
    async def emit_with_response(
        self, event_type: str, data: Dict[str, Any], source: str,
        timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Emitir un evento y esperar respuestas.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente que origina el evento
            timeout: Timeout para esperar respuestas
            
        Returns:
            Lista de respuestas de los componentes
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} con respuesta cuando el bus está detenido")
            return []
            
        timeout = timeout or self.default_timeout
        logger.debug(f"Emitiendo evento {event_type} con espera de respuesta desde {source}")
        
        # Crear cola de respuestas
        response_queue = asyncio.Queue()
        
        # ID único para esta solicitud de respuesta
        response_id = f"resp_{time.time()}_{event_type}"
        
        # Preparar datos con información de respuesta
        event_data = data.copy()
        event_data["response_to"] = source
        event_data["response_id"] = response_id
        event_data["response_type"] = f"response.{event_type}"
        
        # Respuestas recolectadas
        responses: List[Dict[str, Any]] = []
        
        # Componentes esperados (todos excepto el emisor, siguiendo orden topológico)
        expected_components = [cid for cid in self.topological_order if cid != source]
        expected_count = len(expected_components)
        
        # Función para recolectar respuestas
        async def collect_responses():
            collected = 0
            start_time = time.time()
            
            while collected < expected_count and time.time() - start_time < timeout:
                try:
                    # Esperar respuesta con timeout reducido
                    response = await asyncio.wait_for(
                        response_queue.get(),
                        timeout=min(0.5, timeout / 2)
                    )
                    responses.append(response)
                    collected += 1
                    response_queue.task_done()
                except asyncio.TimeoutError:
                    # Verificar si ha pasado demasiado tiempo
                    if time.time() - start_time >= timeout:
                        logger.debug(f"Timeout global para respuestas de {event_type}")
                        break
                except Exception as e:
                    logger.error(f"Error recolectando respuestas para {event_type}: {e}")
        
        # Crear tarea para recolectar respuestas
        collector_task = asyncio.create_task(collect_responses())
        
        # Emitir evento normal (siguiendo orden topológico)
        await self.emit(event_type, event_data, source)
        
        # Esperar a que se recolecten las respuestas
        try:
            await asyncio.wait_for(collector_task, timeout=timeout * 1.1)
        except asyncio.TimeoutError:
            logger.debug(f"Timeout general esperando respuestas para {event_type}")
        
        return responses
        
    async def start_monitoring(self) -> None:
        """Iniciar monitoreo de estado de componentes."""
        logger.info("Iniciando monitoreo de componentes")
        monitor_task = asyncio.create_task(self._component_monitor_loop())
        monitor_task.add_done_callback(
            lambda t: logger.warning(f"Tarea de monitoreo terminó: {t.exception() if t.done() and not t.cancelled() else 'Cancelada'}")
        )
        
    async def _component_monitor_loop(self) -> None:
        """Monitorear estado de componentes periódicamente."""
        while self.running:
            try:
                current_time = time.time()
                
                # Verificar componentes inactivos
                for component_id, last_active in list(self.component_last_active.items()):
                    if component_id not in self.components:
                        continue
                        
                    # Verificar tiempo inactivo
                    inactive_time = current_time - last_active
                    queue_size = self.event_queues[component_id].qsize()
                    
                    # Si está inactivo por mucho tiempo o tiene muchos eventos pendientes
                    if inactive_time > 10.0 or queue_size > 50:
                        logger.warning(
                            f"Componente {component_id} inactivo por {inactive_time:.1f}s o "
                            f"con {queue_size} eventos pendientes"
                        )
                        
                        # Marcar como no saludable
                        if self.component_health.get(component_id, True):
                            self.component_health[component_id] = False
                            
                            # Notificar a callbacks
                            for callback in self.status_callbacks:
                                try:
                                    asyncio.create_task(callback(
                                        component_id, 
                                        {"healthy": False, "reason": "Inactivo"}
                                    ))
                                except Exception as e:
                                    logger.error(f"Error en callback para {component_id}: {e}")
                
                # Revisar estado del bus
                uptime = current_time - self.start_time
                total_events = self.events_published
                active_tasks = len(self.active_tasks)
                
                if uptime > 60 and uptime % 60 < 1:  # Cada minuto aproximadamente
                    logger.info(
                        f"Estado del bus: {total_events} eventos, {active_tasks} tareas activas, "
                        f"{len(self.components)} componentes"
                    )
                
                # Esperar antes de siguiente verificación
                await asyncio.sleep(2.0 if not self.test_mode else 0.5)
                
            except asyncio.CancelledError:
                logger.info("Bucle de monitoreo cancelado")
                break
            except Exception as e:
                logger.error(f"Error en bucle de monitoreo: {e}")
                await asyncio.sleep(1.0)
                
    def register_status_callback(self, callback: Callable[[str, Dict[str, Any]], Coroutine]) -> None:
        """
        Registrar un callback para cambios de estado de componentes.
        
        Args:
            callback: Función que recibe (component_id, metadata)
        """
        self.status_callbacks.append(callback)
        
    async def stop(self) -> None:
        """Detener el bus y liberar recursos."""
        logger.info("Deteniendo bus de eventos basado en grafo")
        self.running = False
        
        # Cancelar todas las tareas activas
        for component_id, task in list(self.active_tasks.items()):
            if not task.done():
                logger.debug(f"Cancelando tarea de {component_id}")
                task.cancel()
        
        # Esperar a que terminen
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        # Limpiar colas
        for component_id, queue in list(self.event_queues.items()):
            try:
                # Vaciar cola
                while not queue.empty():
                    queue.get_nowait()
                    queue.task_done()
            except Exception as e:
                logger.error(f"Error al limpiar cola de {component_id}: {e}")
        
        logger.info(f"Bus detenido. Eventos: publicados={self.events_published}, "
                   f"entregados={self.events_delivered}, timeouts={self.events_timed_out}")