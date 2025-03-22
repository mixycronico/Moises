"""
Sistema Genesis con bucle de actualización centralizado.

Este módulo implementa un sistema síncrono donde un bucle central actualiza todos los
componentes en orden, procesando eventos de manera determinista.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Tuple, Optional, Set, Callable
from collections import deque
import traceback

# Configuración del logger
logger = logging.getLogger(__name__)

class SynchronousEngine:
    """
    Sistema centralizado para gestionar componentes y eventos de forma síncrona.
    
    Características:
    1. Bucle de actualización único que procesa todos los componentes
    2. Buffer global de eventos para comunicación entre componentes
    3. Orden determinista de actualización basado en prioridades
    4. Fácil rastreo y análisis de fallos
    """
    
    def __init__(self, tick_rate: float = 0.01, max_events_per_tick: int = 100):
        """
        Inicializar el motor síncrono Genesis.
        
        Args:
            tick_rate: Tiempo en segundos entre iteraciones del bucle (default: 100 Hz).
            max_events_per_tick: Número máximo de eventos a procesar por tick para evitar bucles infinitos
        """
        # Componentes y orden
        self.components: Dict[str, Any] = {}  # component_id -> componente
        self.component_priorities: Dict[str, int] = {}  # component_id -> prioridad (menor = mayor prioridad)
        self.component_dependencies: Dict[str, Set[str]] = {}  # component_id -> dependencias
        
        # Buffers de eventos
        self.event_buffer: deque = deque()  # Buffer global de eventos
        self.response_buffers: Dict[str, List[Dict[str, Any]]] = {}  # request_id -> respuestas
        
        # Control del bucle
        self.running = False
        self.tick_rate = tick_rate
        self.max_events_per_tick = max_events_per_tick
        self.thread: Optional[threading.Thread] = None
        
        # Estado y monitoreo
        self.component_health: Dict[str, bool] = {}  # component_id -> estado de salud
        self.component_last_update: Dict[str, float] = {}  # component_id -> última actualización exitosa
        self.component_errors: Dict[str, int] = {}  # component_id -> contador de errores
        
        # Estadísticas
        self.events_processed = 0
        self.events_dropped = 0
        self.tick_count = 0
        self.start_time = 0
        self.slow_ticks = 0
        
        # Callbacks
        self.status_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
    def register_component(self, component_id: str, component: Any, 
                          priority: int = 100, depends_on: Optional[List[str]] = None) -> None:
        """
        Registrar un componente en el sistema.
        
        Args:
            component_id: ID único del componente
            component: Instancia del componente con método handle_event
            priority: Prioridad de actualización (menor = mayor prioridad)
            depends_on: Lista de IDs de componentes que deben procesarse antes que este
        """
        if component_id in self.components:
            logger.warning(f"Componente {component_id} ya registrado, reemplazando")
            
        logger.info(f"Registrando componente {component_id} con prioridad {priority}")
        self.components[component_id] = component
        self.component_priorities[component_id] = priority
        self.component_health[component_id] = True
        self.component_last_update[component_id] = time.time()
        self.component_errors[component_id] = 0
        
        # Registrar dependencias
        if depends_on:
            self.component_dependencies[component_id] = set(depends_on)
            # Verificar dependencias no registradas
            for dep_id in depends_on:
                if dep_id not in self.components:
                    logger.warning(f"Dependencia {dep_id} aún no registrada")
        else:
            self.component_dependencies[component_id] = set()
            
    def _get_component_order(self) -> List[str]:
        """
        Determinar el orden óptimo de procesamiento de componentes.
        
        Utiliza dependencias y prioridades para crear un orden topológico.
        
        Returns:
            Lista de IDs de componentes en orden de procesamiento
        """
        # Si no hay componentes, devolver lista vacía
        if not self.components:
            return []
            
        # Crear grafo de dependencias
        in_degree = {comp_id: 0 for comp_id in self.components}
        for comp_id, deps in self.component_dependencies.items():
            for dep_id in deps:
                if dep_id in in_degree:
                    in_degree[comp_id] += 1
                    
        # Componentes sin dependencias, ordenados por prioridad
        no_deps = sorted(
            [comp_id for comp_id, degree in in_degree.items() if degree == 0],
            key=lambda x: self.component_priorities.get(x, 100)
        )
        
        result = []
        while no_deps:
            # Tomar el siguiente componente
            current = no_deps.pop(0)
            result.append(current)
            
            # Actualizar dependencias
            for comp_id in self.components:
                if current in self.component_dependencies.get(comp_id, set()):
                    in_degree[comp_id] -= 1
                    if in_degree[comp_id] == 0:
                        # Insertar en la posición correcta según prioridad
                        priority = self.component_priorities.get(comp_id, 100)
                        insert_idx = 0
                        while (insert_idx < len(no_deps) and 
                               self.component_priorities.get(no_deps[insert_idx], 100) < priority):
                            insert_idx += 1
                        no_deps.insert(insert_idx, comp_id)
                        
        # Verificar ciclos
        if len(result) != len(self.components):
            missing = set(self.components.keys()) - set(result)
            logger.warning(f"Posible dependencia circular detectada: {missing}")
            # Añadir los componentes no incluidos al final, ordenados por prioridad
            remaining = sorted(
                list(missing),
                key=lambda x: self.component_priorities.get(x, 100)
            )
            result.extend(remaining)
            
        return result
        
    def emit(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Emitir un evento al buffer global.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente emisor
        """
        if not self.running and not event_type.startswith("system."):
            logger.warning(f"Sistema detenido, evento {event_type} ignorado")
            return
            
        logger.debug(f"Evento {event_type} emitido desde {source}")
        
        # Crear copia de datos para evitar modificaciones externas
        event_data = data.copy() if data else {}
        
        # Añadir timestamp
        event_data["_timestamp"] = time.time()
        
        # Añadir a buffer
        self.event_buffer.append((event_type, event_data, source))
        
    def emit_with_response(
        self, event_type: str, data: Dict[str, Any], source: str,
        timeout: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Emitir un evento y esperar respuestas.
        
        En esta implementación síncrona, procesa inmediatamente las respuestas.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente emisor
            timeout: Tiempo máximo de espera (ignorado en implementación síncrona)
            
        Returns:
            Lista de respuestas de los componentes
        """
        if not self.running:
            logger.warning(f"Sistema detenido, evento {event_type} con respuesta ignorado")
            return []
            
        logger.debug(f"Evento {event_type} con respuesta emitido desde {source}")
        
        # Crear copia de datos
        event_data = data.copy() if data else {}
        
        # Añadir timestamp y marcadores de respuesta
        event_data["_timestamp"] = time.time()
        request_id = f"req_{time.time()}_{event_type}"
        event_data["_request_id"] = request_id
        event_data["_response_to"] = source
        
        # Inicializar buffer de respuestas
        self.response_buffers[request_id] = []
        
        # Procesar inmediatamente en el hilo actual
        self._process_event_with_response(event_type, event_data, source, request_id)
        
        # Obtener respuestas
        responses = self.response_buffers.pop(request_id, [])
        
        return responses
        
    def _process_event_with_response(
        self, event_type: str, data: Dict[str, Any], source: str, request_id: str
    ) -> None:
        """
        Procesar un evento solicitando respuestas inmediatamente.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente emisor
            request_id: ID de la solicitud para agrupar respuestas
        """
        # Obtener orden de componentes
        component_order = self._get_component_order()
        
        # Procesar evento en cada componente excepto la fuente
        for comp_id in component_order:
            if comp_id != source and comp_id in self.components:
                component = self.components[comp_id]
                
                try:
                    # Verificar si el componente tiene método handle_event
                    if not hasattr(component, "handle_event"):
                        continue
                        
                    # Procesar evento y obtener respuesta
                    response = component.handle_event(event_type, data, source)
                    
                    # Si hay respuesta, almacenarla
                    if response is not None:
                        self.response_buffers[request_id].append({
                            "component": comp_id,
                            "response": response,
                            "timestamp": time.time()
                        })
                        
                except Exception as e:
                    logger.error(f"Error en {comp_id} procesando {event_type}: {e}")
                    self._record_component_error(comp_id, str(e))
                    
                    # Añadir respuesta de error
                    self.response_buffers[request_id].append({
                        "component": comp_id,
                        "error": str(e),
                        "timestamp": time.time()
                    })
                    
    def start(self, threaded: bool = True) -> None:
        """
        Iniciar el bucle de actualización.
        
        Args:
            threaded: Si es True, inicia en un hilo separado; si es False, bloquea el hilo actual
        """
        if self.running:
            logger.warning("Sistema ya está corriendo")
            return
            
        logger.info("Iniciando sistema Genesis Synchronous")
        self.running = True
        self.start_time = time.time()
        
        # Iniciar componentes
        self._start_components()
        
        if threaded:
            # Iniciar en hilo separado
            self.thread = threading.Thread(target=self._run_loop)
            self.thread.daemon = True
            self.thread.start()
        else:
            # Iniciar en hilo actual (bloqueante)
            self._run_loop()
            
    def _start_components(self) -> None:
        """Iniciar todos los componentes en orden apropiado."""
        component_order = self._get_component_order()
        logger.info(f"Iniciando {len(component_order)} componentes")
        
        for comp_id in component_order:
            component = self.components.get(comp_id)
            if component and hasattr(component, "start"):
                try:
                    logger.debug(f"Iniciando componente {comp_id}")
                    
                    # Si el método start es asíncrono, ejecutarlo de forma síncrona
                    # Algunos componentes pueden tener implementaciones asíncronas que necesitamos adaptar
                    result = component.start()
                    if hasattr(result, "__await__"):
                        logger.warning(f"Componente {comp_id} tiene start() asíncrono, adaptando")
                        # No podemos usar await directamente, ignoramos el resultado
                        
                    self.component_health[comp_id] = True
                    self.component_last_update[comp_id] = time.time()
                    
                except Exception as e:
                    logger.error(f"Error al iniciar componente {comp_id}: {e}")
                    self._record_component_error(comp_id, str(e))
                    
    def _run_loop(self) -> None:
        """Bucle central de actualización síncrona."""
        logger.info("Iniciando bucle de actualizaciones")
        
        while self.running:
            start_time = time.time()
            self.tick_count += 1
            
            try:
                # 1. Procesar eventos en el buffer (con límite)
                events_this_tick = 0
                while self.event_buffer and events_this_tick < self.max_events_per_tick:
                    events_this_tick += 1
                    self._process_next_event()
                    
                # Si quedan eventos después del límite, registrar
                if self.event_buffer and events_this_tick >= self.max_events_per_tick:
                    excess = len(self.event_buffer)
                    self.events_dropped += excess
                    logger.warning(f"Buffer de eventos lleno: {excess} eventos pendientes")
                    
                # 2. Actualizar todos los componentes
                self._update_components()
                
                # 3. Verificar estado de componentes
                self._check_component_health()
                
                # 4. Controlar la tasa de ticks
                elapsed = time.time() - start_time
                if elapsed > self.tick_rate:
                    self.slow_ticks += 1
                    if self.tick_count % 100 == 0 or elapsed > self.tick_rate * 2:
                        logger.warning(f"Tick {self.tick_count} lento: {elapsed:.4f}s > {self.tick_rate:.4f}s")
                        
                sleep_time = max(0, self.tick_rate - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error en bucle principal: {e}")
                logger.error(traceback.format_exc())
                # Continuar a pesar del error
                time.sleep(self.tick_rate)
                
        logger.info(f"Bucle detenido. Stats: ticks={self.tick_count}, "
                   f"eventos={self.events_processed}, drops={self.events_dropped}, "
                   f"ticks_lentos={self.slow_ticks}")
                   
        # Detener componentes
        self._stop_components()
        
    def _process_next_event(self) -> None:
        """Procesar el siguiente evento del buffer."""
        try:
            # Extraer evento
            event_type, data, source = self.event_buffer.popleft()
            self.events_processed += 1
            
            # Obtener orden de componentes
            component_order = self._get_component_order()
            
            # Procesar evento en cada componente excepto la fuente
            for comp_id in component_order:
                if comp_id != source and comp_id in self.components:
                    component = self.components[comp_id]
                    
                    # Verificar si el componente tiene método handle_event
                    if not hasattr(component, "handle_event"):
                        continue
                        
                    try:
                        # Verificar tiempo de procesamiento
                        start_time = time.time()
                        
                        # Manejar evento
                        component.handle_event(event_type, data, source)
                        
                        # Actualizar timestamp
                        self.component_last_update[comp_id] = time.time()
                        
                        # Verificar si el procesamiento fue lento
                        process_time = time.time() - start_time
                        if process_time > 0.1:  # > 100ms se considera lento
                            logger.warning(f"Componente {comp_id} lento procesando {event_type}: {process_time:.4f}s")
                            
                    except Exception as e:
                        logger.error(f"Error en {comp_id} procesando {event_type}: {e}")
                        self._record_component_error(comp_id, str(e))
                        
        except IndexError:
            # Buffer vacío
            pass
        except Exception as e:
            logger.error(f"Error procesando evento: {e}")
            
    def _update_components(self) -> None:
        """Actualizar todos los componentes que tengan método update."""
        component_order = self._get_component_order()
        
        for comp_id in component_order:
            component = self.components.get(comp_id)
            
            # Verificar si el componente tiene método update
            if component and hasattr(component, "update"):
                try:
                    # Verificar tiempo de actualización
                    start_time = time.time()
                    
                    # Actualizar componente
                    component.update()
                    
                    # Registrar última actualización exitosa
                    self.component_last_update[comp_id] = time.time()
                    
                    # Verificar si la actualización fue lenta
                    update_time = time.time() - start_time
                    if update_time > 0.1:  # > 100ms se considera lento
                        logger.warning(f"Actualización lenta en {comp_id}: {update_time:.4f}s")
                        
                except Exception as e:
                    logger.error(f"Error actualizando {comp_id}: {e}")
                    self._record_component_error(comp_id, str(e))
                    
    def _check_component_health(self) -> None:
        """Verificar estado de salud de los componentes."""
        current_time = time.time()
        
        # Verificar cada 10 ticks (para no consumir recursos)
        if self.tick_count % 10 != 0:
            return
            
        for comp_id, component in self.components.items():
            last_update = self.component_last_update.get(comp_id, 0)
            errors = self.component_errors.get(comp_id, 0)
            
            # Verificar inactividad prolongada (> 5 segundos sin actualización)
            if current_time - last_update > 5.0:
                logger.warning(f"Componente {comp_id} inactivo por {current_time - last_update:.1f}s")
                
                # Marcar como no saludable
                if self.component_health.get(comp_id, True):
                    self.component_health[comp_id] = False
                    self._notify_status_change(comp_id, False, "Inactividad prolongada")
                    
            # Verificar errores excesivos (> 5 errores)
            elif errors > 5:
                logger.warning(f"Componente {comp_id} con {errors} errores")
                
                # Marcar como no saludable
                if self.component_health.get(comp_id, True):
                    self.component_health[comp_id] = False
                    self._notify_status_change(comp_id, False, "Errores excesivos")
                    
            # Restaurar salud si estaba mal pero ahora parece bien
            elif not self.component_health.get(comp_id, True):
                # Si ha pasado tiempo sin errores
                if current_time - last_update < 2.0 and errors <= 5:
                    logger.info(f"Componente {comp_id} recuperado")
                    self.component_health[comp_id] = True
                    self._notify_status_change(comp_id, True, "Recuperado")
                    
                    # Reiniciar contador de errores
                    self.component_errors[comp_id] = 0
                    
    def _record_component_error(self, component_id: str, error_message: str) -> None:
        """
        Registrar un error de componente.
        
        Args:
            component_id: ID del componente
            error_message: Descripción del error
        """
        # Incrementar contador
        self.component_errors[component_id] = self.component_errors.get(component_id, 0) + 1
        
        # Si supera umbral, marcar como no saludable
        if self.component_errors[component_id] >= 3 and self.component_health.get(component_id, True):
            self.component_health[component_id] = False
            self._notify_status_change(component_id, False, f"Error: {error_message}")
            
    def _notify_status_change(self, component_id: str, healthy: bool, reason: str) -> None:
        """
        Notificar cambio de estado de un componente.
        
        Args:
            component_id: ID del componente
            healthy: Estado de salud
            reason: Razón del cambio
        """
        metadata = {
            "healthy": healthy,
            "reason": reason,
            "timestamp": time.time(),
            "error_count": self.component_errors.get(component_id, 0)
        }
        
        # Llamar callbacks registrados
        for callback in self.status_callbacks:
            try:
                callback(component_id, metadata)
            except Exception as e:
                logger.error(f"Error en callback de estado para {component_id}: {e}")
                
    def register_status_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Registrar callback para cambios de estado de componentes.
        
        Args:
            callback: Función que recibe (component_id, metadata)
        """
        self.status_callbacks.append(callback)
        
    def stop(self) -> None:
        """Detener el sistema."""
        if not self.running:
            logger.warning("Sistema ya está detenido")
            return
            
        logger.info("Deteniendo sistema Genesis Synchronous")
        self.running = False
        
        # Si estamos en modo hilo, esperar a que termine
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Timeout esperando a que termine el hilo principal")
                
    def _stop_components(self) -> None:
        """Detener todos los componentes en orden inverso."""
        # Obtener orden y revertirlo para detener primero los dependientes
        component_order = list(reversed(self._get_component_order()))
        logger.info(f"Deteniendo {len(component_order)} componentes")
        
        for comp_id in component_order:
            component = self.components.get(comp_id)
            if component and hasattr(component, "stop"):
                try:
                    logger.debug(f"Deteniendo componente {comp_id}")
                    
                    # Igual que con start, adaptamos si es asíncrono
                    result = component.stop()
                    if hasattr(result, "__await__"):
                        logger.warning(f"Componente {comp_id} tiene stop() asíncrono, adaptando")
                        
                except Exception as e:
                    logger.error(f"Error al detener componente {comp_id}: {e}")
                    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado completo del sistema.
        
        Returns:
            Diccionario con estado del sistema y componentes
        """
        current_time = time.time()
        uptime = current_time - self.start_time if self.start_time > 0 else 0
        
        return {
            "running": self.running,
            "uptime": uptime,
            "tick_count": self.tick_count,
            "events_processed": self.events_processed,
            "events_dropped": self.events_dropped,
            "events_pending": len(self.event_buffer),
            "slow_ticks": self.slow_ticks,
            "components": {
                comp_id: {
                    "healthy": self.component_health.get(comp_id, False),
                    "last_update": self.component_last_update.get(comp_id, 0),
                    "errors": self.component_errors.get(comp_id, 0),
                    "inactive_time": current_time - self.component_last_update.get(comp_id, current_time),
                    "priority": self.component_priorities.get(comp_id, 100),
                    "dependencies": list(self.component_dependencies.get(comp_id, set()))
                }
                for comp_id in self.components
            }
        }
        
    def restart_component(self, component_id: str) -> bool:
        """
        Reiniciar un componente específico.
        
        Args:
            component_id: ID del componente a reiniciar
            
        Returns:
            True si se reinició correctamente, False en caso contrario
        """
        if component_id not in self.components:
            logger.warning(f"Componente {component_id} no encontrado")
            return False
            
        logger.info(f"Reiniciando componente {component_id}")
        component = self.components[component_id]
        
        # Detener componente
        if hasattr(component, "stop"):
            try:
                component.stop()
            except Exception as e:
                logger.error(f"Error deteniendo {component_id}: {e}")
                
        # Iniciar componente
        if hasattr(component, "start"):
            try:
                component.start()
                self.component_health[component_id] = True
                self.component_last_update[component_id] = time.time()
                self.component_errors[component_id] = 0
                return True
            except Exception as e:
                logger.error(f"Error iniciando {component_id}: {e}")
                self._record_component_error(component_id, str(e))
                return False
                
        return True  # Si no tiene start/stop, se considera exitoso
        
    def clear_event_buffer(self) -> int:
        """
        Limpiar el buffer de eventos.
        
        Returns:
            Número de eventos eliminados
        """
        count = len(self.event_buffer)
        self.event_buffer.clear()
        logger.info(f"Buffer de eventos limpiado: {count} eventos eliminados")
        return count