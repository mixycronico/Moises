"""
Motor basado en grafo de dependencias para el sistema Genesis.

Este módulo implementa un motor que utiliza el ComponentGraphEventBus
para gestionar componentes con dependencias explícitas.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Set, Optional, Tuple, Callable, Coroutine

from genesis.core.component_graph_event_bus import ComponentGraphEventBus, CircularDependencyError
from genesis.core.component import Component

# Configuración del logger
logger = logging.getLogger(__name__)

class GraphBasedEngine:
    """
    Motor basado en grafo de dependencias para gestionar componentes.
    
    Este motor:
    1. Utiliza un ComponentGraphEventBus que organiza componentes en un grafo acíclico.
    2. Facilita la declaración explícita de dependencias para eliminar deadlocks.
    3. Inicia y detiene componentes en orden topológico correcto.
    4. Monitoriza la salud de los componentes y manejaa fallos.
    """
    
    def __init__(self, test_mode: bool = False, max_queue_size: int = 100):
        """
        Inicializar motor basado en grafo.
        
        Args:
            test_mode: Activar timeouts más agresivos para pruebas
            max_queue_size: Tamaño máximo de las colas de eventos
        """
        self.event_bus = ComponentGraphEventBus(test_mode=test_mode, max_queue_size=max_queue_size)
        self.components: Dict[str, Component] = {}
        self.running: bool = False
        self.started_at: Optional[float] = None
        self.shutdown_complete: Optional[asyncio.Event] = None
        self.test_mode: bool = test_mode
        
        # Mapa de dependencias (bidireccional)
        self.component_dependencies: Dict[str, Set[str]] = {}  # id -> depende de
        self.component_dependents: Dict[str, Set[str]] = {}  # id -> dependientes
        
        # Estado de los componentes
        self.component_health: Dict[str, bool] = {}  # id -> salud
        self.component_error_count: Dict[str, int] = {}  # id -> errores
        
        # Historial de errores para análisis
        self.error_history: List[Dict[str, Any]] = []
        
        # Callbacks para eventos de estado
        self.status_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
    def register_component(
        self, component: Component, depends_on: Optional[List[str]] = None
    ) -> None:
        """
        Registrar un componente con dependencias explícitas.
        
        Args:
            component: Instancia del componente a registrar
            depends_on: Lista de IDs de componentes de los que depende
            
        Raises:
            CircularDependencyError: Si la dependencia crearía un ciclo
            ValueError: Si el componente no tiene ID
        """
        component_id = getattr(component, "id", None)
        
        if not component_id:
            raise ValueError("El componente debe tener un atributo 'id'")
            
        logger.info(f"Registrando componente: {component_id}")
        
        # Verificar dependencias
        self.components[component_id] = component
        
        # Inicializar contadores y estado
        self.component_health[component_id] = True
        self.component_error_count[component_id] = 0
        
        # Registrar dependencias
        dependencies = depends_on or []
        self.component_dependencies[component_id] = set(dependencies)
        
        # Actualizar mapa inverso de dependencias
        for dep_id in dependencies:
            if dep_id not in self.component_dependents:
                self.component_dependents[dep_id] = set()
            self.component_dependents[dep_id].add(component_id)
            
        # Intentar registrar en el bus de eventos
        try:
            self.event_bus.register_component(component_id, component, depends_on)
        except CircularDependencyError as e:
            # Limpiar dependencias en caso de error
            self.component_dependencies.pop(component_id, None)
            for dep_id in dependencies:
                if dep_id in self.component_dependents:
                    self.component_dependents[dep_id].discard(component_id)
            raise CircularDependencyError(f"Error al registrar {component_id}: {e}")
            
    async def start(self) -> None:
        """
        Iniciar el motor y todos los componentes.
        
        Se inician según el orden topológico, garantizando que las dependencias
        se inician antes que los componentes que dependen de ellas.
        """
        if self.running:
            logger.warning("El motor ya está en ejecución")
            return
            
        self.running = True
        self.started_at = time.time()
        self.shutdown_complete = asyncio.Event()
        logger.info("Iniciando motor")
        
        # Iniciar monitoreo
        await self.event_bus.start_monitoring()
        
        # Registrar callback de estado
        self.event_bus.register_status_callback(self._handle_component_status)
        
        # Obtener orden topológico desde el bus
        topological_order = self.event_bus.topological_order
        if not topological_order:
            logger.warning("Sin orden topológico definido, iniciando en orden arbitrario")
            topological_order = list(self.components.keys())
            
        # Iniciar componentes en orden topológico
        for component_id in topological_order:
            if component_id in self.components:
                await self._start_component(component_id)
                
        logger.info(f"Motor iniciado con {len(self.components)} componentes")
        
    async def _start_component(self, component_id: str) -> None:
        """
        Iniciar un componente específico.
        
        Args:
            component_id: ID del componente a iniciar
        """
        if component_id not in self.components:
            logger.warning(f"Componente {component_id} no encontrado")
            return
            
        component = self.components[component_id]
        logger.debug(f"Iniciando componente: {component_id}")
        
        try:
            # Verificar dependencias antes de iniciar
            for dep_id in self.component_dependencies.get(component_id, set()):
                if dep_id not in self.components:
                    logger.warning(f"Dependencia {dep_id} no registrada para {component_id}")
                elif not self.component_health.get(dep_id, False):
                    logger.warning(f"Dependencia {dep_id} no saludable para {component_id}")
            
            # Iniciar componente con timeout
            timeout = 2.0 if not self.test_mode else 0.5
            await asyncio.wait_for(component.start(), timeout)
            self.component_health[component_id] = True
            
            logger.info(f"Componente iniciado: {component_id}")
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout al iniciar componente: {component_id}")
            self.component_health[component_id] = False
            self.component_error_count[component_id] += 1
            
            self._record_error(component_id, "start_timeout", "Timeout durante inicialización")
            
        except Exception as e:
            logger.error(f"Error al iniciar componente {component_id}: {e}")
            self.component_health[component_id] = False
            self.component_error_count[component_id] += 1
            
            self._record_error(component_id, "start_error", str(e))
            
    async def stop(self) -> None:
        """
        Detener el motor y todos los componentes.
        
        Se detienen en orden inverso al topológico, garantizando que los
        componentes se detienen antes que sus dependencias.
        """
        if not self.running:
            logger.warning("El motor ya está detenido")
            return
            
        self.running = False
        logger.info("Deteniendo motor")
        
        # Determinar orden de detención (inverso al topológico)
        stop_order = list(reversed(self.event_bus.topological_order))
        if not stop_order:
            logger.warning("Sin orden topológico definido, deteniendo en orden arbitrario")
            stop_order = list(self.components.keys())
            
        # Detener componentes en orden
        for component_id in stop_order:
            if component_id in self.components:
                await self._stop_component(component_id)
                
        # Detener bus de eventos
        await self.event_bus.stop()
        
        # Marcar como completado
        if self.shutdown_complete:
            self.shutdown_complete.set()
            
        logger.info("Motor detenido")
        
    async def _stop_component(self, component_id: str) -> None:
        """
        Detener un componente específico.
        
        Args:
            component_id: ID del componente a detener
        """
        if component_id not in self.components:
            logger.warning(f"Componente {component_id} no encontrado para detener")
            return
            
        component = self.components[component_id]
        logger.debug(f"Deteniendo componente: {component_id}")
        
        try:
            # Detener con timeout
            timeout = 3.0 if not self.test_mode else 0.5
            await asyncio.wait_for(component.stop(), timeout)
            logger.info(f"Componente detenido: {component_id}")
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout al detener componente: {component_id}")
            self._record_error(component_id, "stop_timeout", "Timeout durante detención")
            
        except Exception as e:
            logger.error(f"Error al detener componente {component_id}: {e}")
            self._record_error(component_id, "stop_error", str(e))
            
    async def restart_component(self, component_id: str) -> bool:
        """
        Reiniciar un componente específico.
        
        Args:
            component_id: ID del componente a reiniciar
            
        Returns:
            True si se reinició correctamente, False en caso contrario
        """
        if component_id not in self.components:
            logger.warning(f"Componente {component_id} no encontrado para reiniciar")
            return False
            
        logger.info(f"Reiniciando componente: {component_id}")
        
        # Detener componente
        await self._stop_component(component_id)
        
        # Reiniciar sus tareas en el bus
        # El bus se encargará automáticamente
        
        # Iniciar componente
        await self._start_component(component_id)
        
        return self.component_health.get(component_id, False)
        
    async def emit(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Emitir un evento a través del bus.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente que origina el evento
        """
        if not self.running:
            logger.warning("Intento de emitir evento con el motor detenido")
            return
            
        await self.event_bus.emit(event_type, data, source)
        
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
            logger.warning("Intento de emitir evento con respuesta con el motor detenido")
            return []
            
        return await self.event_bus.emit_with_response(event_type, data, source, timeout)
        
    def _record_error(self, component_id: str, error_type: str, message: str) -> None:
        """
        Registrar un error para análisis.
        
        Args:
            component_id: ID del componente que genera el error
            error_type: Tipo de error
            message: Mensaje descriptivo
        """
        error_entry = {
            "timestamp": time.time(),
            "component_id": component_id,
            "error_type": error_type,
            "message": message,
            "dependencies": list(self.component_dependencies.get(component_id, set())),
            "dependents": list(self.component_dependents.get(component_id, set()))
        }
        
        self.error_history.append(error_entry)
        
        # Limitar historial
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
            
    async def _handle_component_status(self, component_id: str, metadata: Dict[str, Any]) -> None:
        """
        Manejar cambios de estado de componentes.
        
        Args:
            component_id: ID del componente
            metadata: Metadatos del cambio de estado
        """
        is_healthy = metadata.get("healthy", False)
        reason = metadata.get("reason", "Desconocido")
        
        logger.debug(f"Cambio de estado en {component_id}: healthy={is_healthy}, reason={reason}")
        
        # Actualizar estado interno
        self.component_health[component_id] = is_healthy
        
        # Si no está saludable, incrementar contador de errores
        if not is_healthy:
            self.component_error_count[component_id] = self.component_error_count.get(component_id, 0) + 1
            self._record_error(component_id, "health_change", reason)
            
            # Verificar si afecta a dependientes
            affected = self.component_dependents.get(component_id, set())
            if affected:
                logger.warning(f"Componente no saludable {component_id} puede afectar a: {', '.join(affected)}")
                
        # Notificar a callbacks registrados
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
        
    def get_component_status(self) -> Dict[str, Any]:
        """
        Obtener estado completo de los componentes.
        
        Returns:
            Diccionario con estado de todos los componentes
        """
        return {
            "timestamp": time.time(),
            "uptime": time.time() - (self.started_at or time.time()),
            "running": self.running,
            "components": {
                component_id: {
                    "healthy": self.component_health.get(component_id, False),
                    "error_count": self.component_error_count.get(component_id, 0),
                    "dependencies": list(self.component_dependencies.get(component_id, set())),
                    "dependents": list(self.component_dependents.get(component_id, set()))
                }
                for component_id in self.components
            },
            "event_stats": {
                "published": self.event_bus.events_published,
                "delivered": self.event_bus.events_delivered,
                "timed_out": self.event_bus.events_timed_out
            }
        }
        
    def generate_component_graph(self) -> Dict[str, Any]:
        """
        Generar representación del grafo de componentes.
        
        Returns:
            Estructura de datos que representa el grafo de componentes
        """
        return {
            "nodes": [
                {
                    "id": component_id,
                    "healthy": self.component_health.get(component_id, False),
                    "error_count": self.component_error_count.get(component_id, 0)
                }
                for component_id in self.components
            ],
            "edges": [
                {
                    "source": dep_id,
                    "target": component_id
                }
                for component_id, deps in self.component_dependencies.items()
                for dep_id in deps
            ]
        }
        
    def get_error_history(self) -> List[Dict[str, Any]]:
        """
        Obtener historial de errores.
        
        Returns:
            Lista de errores registrados
        """
        return self.error_history