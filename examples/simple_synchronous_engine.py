"""
Implementación simplificada del motor síncrono para fines de demostración.

Este módulo contiene una versión reducida del SynchronousEngine
que mantiene las funcionalidades esenciales sin complejidades adicionales.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Set
from collections import deque

# Configuración del logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sync_engine_demo")

class SimpleSynchronousEngine:
    """
    Motor síncrono simplificado para demostración.
    
    Características principales:
    1. Registro de componentes
    2. Dependencias entre componentes
    3. Bucle de actualización centralizado
    4. Procesamiento de eventos
    """
    
    def __init__(self, tick_rate: float = 0.01):
        """
        Inicializar el motor síncrono.
        
        Args:
            tick_rate: Tiempo en segundos entre actualizaciones
        """
        # Componentes
        self.components: Dict[str, Any] = {}
        self.component_dependencies: Dict[str, Set[str]] = {}
        
        # Eventos
        self.event_buffer = deque()
        self.response_buffers: Dict[str, List[Dict[str, Any]]] = {}
        
        # Control
        self.running = False
        self.tick_rate = tick_rate
        self.thread = None
        
        # Estadísticas
        self.events_processed = 0
        self.tick_count = 0
        
    def register_component(self, component_id: str, component: Any, 
                           depends_on: Optional[List[str]] = None) -> None:
        """
        Registrar un componente en el sistema.
        
        Args:
            component_id: ID único del componente
            component: Instancia del componente
            depends_on: Lista de IDs de componentes que deben procesarse antes
        """
        if component_id in self.components:
            logger.warning(f"Componente {component_id} ya registrado, reemplazando")
            
        logger.info(f"Registrando componente {component_id}")
        self.components[component_id] = component
        
        # Dependencias
        self.component_dependencies[component_id] = set(depends_on or [])
        
    def _get_component_order(self) -> List[str]:
        """
        Determinar orden de procesamiento de componentes basado en dependencias.
        
        Returns:
            Lista de IDs de componentes en orden de procesamiento
        """
        # Calcular grados de entrada
        in_degree = {comp_id: 0 for comp_id in self.components}
        for comp_id, deps in self.component_dependencies.items():
            for dep_id in deps:
                if dep_id in in_degree:
                    in_degree[comp_id] += 1
                    
        # Ordenación topológica
        no_deps = [comp_id for comp_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while no_deps:
            current = no_deps.pop(0)
            result.append(current)
            
            # Actualizar dependencias
            for comp_id in self.components:
                if current in self.component_dependencies.get(comp_id, set()):
                    in_degree[comp_id] -= 1
                    if in_degree[comp_id] == 0:
                        no_deps.append(comp_id)
                        
        # Verificar ciclos
        if len(result) != len(self.components):
            missing = set(self.components.keys()) - set(result)
            logger.warning(f"Posible dependencia circular: {missing}")
            result.extend(list(missing))
            
        return result
        
    def emit(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Emitir un evento al buffer global.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente emisor
        """
        if not self.running:
            logger.warning(f"Sistema detenido, evento {event_type} ignorado")
            return
            
        logger.debug(f"Evento {event_type} emitido desde {source}")
        
        # Clonar datos para evitar modificaciones externas
        event_data = data.copy() if data else {}
        event_data["_timestamp"] = time.time()
        
        # Añadir al buffer
        self.event_buffer.append((event_type, event_data, source))
        
    def emit_with_response(self, event_type: str, data: Dict[str, Any], 
                           source: str) -> List[Dict[str, Any]]:
        """
        Emitir un evento y esperar respuestas síncronamente.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente emisor
            
        Returns:
            Lista de respuestas de los componentes
        """
        if not self.running:
            logger.warning(f"Sistema detenido, evento {event_type} con respuesta ignorado")
            return []
            
        logger.debug(f"Evento {event_type} con respuesta emitido desde {source}")
        
        # Preparar datos
        event_data = data.copy() if data else {}
        event_data["_timestamp"] = time.time()
        request_id = f"{time.time()}_{event_type}"
        event_data["_request_id"] = request_id
        event_data["_response_to"] = source
        
        # Buffer de respuestas
        self.response_buffers[request_id] = []
        
        # Procesar inmediatamente
        component_order = self._get_component_order()
        for comp_id in component_order:
            if comp_id != source:
                component = self.components.get(comp_id)
                if not component or not hasattr(component, "handle_event"):
                    continue
                    
                try:
                    response = component.handle_event(event_type, event_data, source)
                    if response is not None:
                        self.response_buffers[request_id].append({
                            "component": comp_id,
                            "response": response,
                            "timestamp": time.time()
                        })
                except Exception as e:
                    logger.error(f"Error en {comp_id} procesando {event_type}: {e}")
                    
        # Obtener respuestas
        responses = self.response_buffers.pop(request_id, [])
        return responses
        
    def start(self, threaded: bool = True) -> None:
        """
        Iniciar el bucle de actualización.
        
        Args:
            threaded: Si es True, inicia en un hilo separado
        """
        if self.running:
            logger.warning("Sistema ya está corriendo")
            return
            
        logger.info("Iniciando sistema síncrono simplificado")
        self.running = True
        
        # Iniciar componentes
        self._start_components()
        
        if threaded:
            self.thread = threading.Thread(target=self._run_loop)
            self.thread.daemon = True
            self.thread.start()
        else:
            self._run_loop()
            
    def _start_components(self) -> None:
        """Iniciar todos los componentes."""
        component_order = self._get_component_order()
        
        for comp_id in component_order:
            component = self.components.get(comp_id)
            if component and hasattr(component, "start"):
                try:
                    component.start()
                    logger.debug(f"Componente {comp_id} iniciado")
                except Exception as e:
                    logger.error(f"Error al iniciar {comp_id}: {e}")
                    
    def _run_loop(self) -> None:
        """Bucle central de actualización."""
        logger.info("Iniciando bucle de actualizaciones")
        
        while self.running:
            start_time = time.time()
            self.tick_count += 1
            
            try:
                # 1. Procesar eventos (hasta 10 por tick)
                events_processed = 0
                while self.event_buffer and events_processed < 10:
                    self._process_next_event()
                    events_processed += 1
                    
                # 2. Actualizar componentes
                self._update_components()
                
                # 3. Controlar tasa de actualización
                elapsed = time.time() - start_time
                sleep_time = max(0, self.tick_rate - elapsed)
                
                if elapsed > self.tick_rate and self.tick_count % 100 == 0:
                    logger.warning(f"Tick lento: {elapsed:.4f}s > {self.tick_rate:.4f}s")
                    
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error en bucle principal: {e}")
                time.sleep(self.tick_rate)  # Recuperación
                
        logger.info(f"Bucle detenido. Stats: ticks={self.tick_count}, eventos={self.events_processed}")
        
        # Detener componentes
        self._stop_components()
        
    def _process_next_event(self) -> None:
        """Procesar el siguiente evento del buffer."""
        try:
            event_type, data, source = self.event_buffer.popleft()
            self.events_processed += 1
            
            # Procesar en cada componente excepto la fuente
            component_order = self._get_component_order()
            for comp_id in component_order:
                if comp_id != source:
                    component = self.components.get(comp_id)
                    if not component or not hasattr(component, "handle_event"):
                        continue
                        
                    try:
                        component.handle_event(event_type, data, source)
                    except Exception as e:
                        logger.error(f"Error en {comp_id} procesando {event_type}: {e}")
                        
        except Exception as e:
            logger.error(f"Error procesando evento: {e}")
            
    def _update_components(self) -> None:
        """Actualizar todos los componentes."""
        component_order = self._get_component_order()
        
        for comp_id in component_order:
            component = self.components.get(comp_id)
            if component and hasattr(component, "update"):
                try:
                    component.update()
                except Exception as e:
                    logger.error(f"Error actualizando {comp_id}: {e}")
                    
    def _stop_components(self) -> None:
        """Detener todos los componentes en orden inverso."""
        component_order = self._get_component_order()
        
        for comp_id in reversed(component_order):
            component = self.components.get(comp_id)
            if component and hasattr(component, "stop"):
                try:
                    component.stop()
                    logger.debug(f"Componente {comp_id} detenido")
                except Exception as e:
                    logger.error(f"Error al detener {comp_id}: {e}")
                    
    def stop(self) -> None:
        """Detener el sistema."""
        if not self.running:
            logger.warning("Sistema ya está detenido")
            return
            
        logger.info("Deteniendo sistema...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema.
        
        Returns:
            Diccionario con estado actual
        """
        return {
            "running": self.running,
            "components": len(self.components),
            "events_in_buffer": len(self.event_buffer),
            "events_processed": self.events_processed,
            "tick_count": self.tick_count,
            "uptime": time.time() - getattr(self, "start_time", time.time())
        }