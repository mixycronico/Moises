"""
Implementación configurable del motor del sistema Genesis.

Esta versión extiende el EngineNonBlocking para permitir
timeouts configurables y mejor manejo de condiciones extremas.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Union, List

from genesis.core.engine_non_blocking import EngineNonBlocking
from genesis.core.component import Component
from genesis.core.event_bus import EventBus


class ConfigurableTimeoutEngine(EngineNonBlocking):
    """
    Motor con timeouts configurables para mejor adaptación a diferentes escenarios.
    
    Esta implementación permite ajustar los timeouts para diferentes operaciones
    y proporciona mejores mecanismos para manejar componentes lentos o con fallas.
    """
    
    def __init__(
        self,
        event_bus_or_name: Union[EventBus, str, None] = None,
        test_mode: bool = False,
        component_start_timeout: float = 0.5,
        component_stop_timeout: float = 0.5,
        event_timeout: float = 0.5,
        component_event_timeout: float = 0.5
    ):
        """
        Inicializar el motor configurable.
        
        Args:
            event_bus_or_name: Instancia de EventBus o nombre para crear uno nuevo
            test_mode: Si True, habilitará comportamiento optimizado para pruebas
            component_start_timeout: Timeout para inicio de componentes (en segundos)
            component_stop_timeout: Timeout para detención de componentes (en segundos)
            event_timeout: Timeout para emisión de eventos (en segundos)
            component_event_timeout: Timeout para manejadores de eventos (en segundos)
        """
        # Inicializar clase base
        super().__init__(event_bus_or_name, test_mode)
        
        # Configurar timeouts personalizados
        self.component_start_timeout = component_start_timeout
        self.component_stop_timeout = component_stop_timeout
        self.event_timeout = event_timeout
        self.component_event_timeout = component_event_timeout
        
        # Contadores para diagnóstico
        self.timeouts_occurred = {
            "component_start": 0,
            "component_stop": 0,
            "event_emission": 0,
            "component_event": 0
        }
        
        # Flag para modo de recuperación avanzada
        self.advanced_recovery = False
        
        # Registro de componentes con timeouts frecuentes
        self.problem_components = {}
        
        logging.getLogger(__name__).debug(
            f"ConfigurableTimeoutEngine inicializado con timeouts: "
            f"start={component_start_timeout}s, stop={component_stop_timeout}s, "
            f"event={event_timeout}s, component_event={component_event_timeout}s"
        )
    
    def enable_advanced_recovery(self, enabled: bool = True) -> None:
        """
        Habilitar o deshabilitar el modo de recuperación avanzada.
        
        En este modo, el motor intenta recuperar componentes que fallan 
        automáticamente y aplica estrategias adaptativas para timeouts.
        
        Args:
            enabled: True para habilitar, False para deshabilitar
        """
        self.advanced_recovery = enabled
        logging.getLogger(__name__).info(
            f"Modo de recuperación avanzada: {'habilitado' if enabled else 'deshabilitado'}"
        )
    
    async def _start_component(self, component: Component) -> None:
        """
        Iniciar un componente con timeout configurable.
        
        Args:
            component: Componente a iniciar
        """
        try:
            # Usar timeout configurable
            timeout_value = self.component_start_timeout
            await asyncio.wait_for(component.start(), timeout=timeout_value)
        except asyncio.TimeoutError:
            self.timeouts_occurred["component_start"] += 1
            logging.getLogger(__name__).warning(
                f"Timeout iniciando componente {component.name} "
                f"después de {timeout_value}s"
            )
            
            # Registrar componente problemático
            if component.name not in self.problem_components:
                self.problem_components[component.name] = {"start_timeouts": 0, "stop_timeouts": 0, "event_timeouts": 0}
            self.problem_components[component.name]["start_timeouts"] += 1
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error iniciando componente {component.name}: {e}")
    
    async def start_component(self, name: str) -> bool:
        """
        Iniciar un componente específico por nombre.
        
        Args:
            name: Nombre del componente a iniciar
            
        Returns:
            True si el componente se inició correctamente, False en caso contrario
        """
        if name not in self.components:
            logging.getLogger(__name__).warning(f"Componente {name} no encontrado, no se puede iniciar")
            return False
            
        component = self.components[name]
        try:
            # Usar timeout configurable
            await asyncio.wait_for(component.start(), timeout=self.component_start_timeout)
            return True
        except asyncio.TimeoutError:
            self.timeouts_occurred["component_start"] += 1
            logging.getLogger(__name__).warning(
                f"Timeout iniciando componente {name} después de {self.component_start_timeout}s"
            )
            
            # Registrar componente problemático
            if name not in self.problem_components:
                self.problem_components[name] = {"start_timeouts": 0, "stop_timeouts": 0, "event_timeouts": 0}
            self.problem_components[name]["start_timeouts"] += 1
            
            return False
        except Exception as e:
            logging.getLogger(__name__).error(f"Error iniciando componente {name}: {e}")
            return False
    
    async def stop_component(self, name: str) -> bool:
        """
        Detener un componente específico por nombre.
        
        Args:
            name: Nombre del componente a detener
            
        Returns:
            True si el componente se detuvo correctamente, False en caso contrario
        """
        if name not in self.components:
            logging.getLogger(__name__).warning(f"Componente {name} no encontrado, no se puede detener")
            return False
            
        component = self.components[name]
        try:
            # Usar timeout configurable
            await asyncio.wait_for(component.stop(), timeout=self.component_stop_timeout)
            return True
        except asyncio.TimeoutError:
            self.timeouts_occurred["component_stop"] += 1
            logging.getLogger(__name__).warning(
                f"Timeout deteniendo componente {name} después de {self.component_stop_timeout}s"
            )
            
            # Registrar componente problemático
            if name not in self.problem_components:
                self.problem_components[name] = {"start_timeouts": 0, "stop_timeouts": 0, "event_timeouts": 0}
            self.problem_components[name]["stop_timeouts"] += 1
            
            return False
        except Exception as e:
            logging.getLogger(__name__).error(f"Error deteniendo componente {name}: {e}")
            return False
    
    async def start(self) -> None:
        """
        Iniciar el motor y todos los componentes registrados.
        
        Esta implementación usa timeouts configurables para evitar bloqueos.
        """
        if self.running:
            logging.getLogger(__name__).warning("Motor ya en ejecución")
            return
            
        logging.getLogger(__name__).info("Iniciando motor configurable Genesis")
        
        # Asegurar que el event bus esté iniciado
        await self._ensure_event_bus_started()
            
        # Ordenar componentes por prioridad (mayor prioridad primero para iniciar)
        ordered_components = sorted(
            self.components.items(),
            key=lambda x: self.operation_priorities.get(x[0], 50),
            reverse=True  # Mayor prioridad primero
        )
        
        # Iniciar los componentes usando timeouts configurables
        for name, component in ordered_components:
            try:
                # Usar un timeout configurable
                await asyncio.wait_for(component.start(), timeout=self.component_start_timeout)
            except asyncio.TimeoutError:
                self.timeouts_occurred["component_start"] += 1
                logging.getLogger(__name__).warning(
                    f"Timeout al iniciar componente {name} después de {self.component_start_timeout}s"
                )
                
                # Registrar componente problemático
                if name not in self.problem_components:
                    self.problem_components[name] = {"start_timeouts": 0, "stop_timeouts": 0, "event_timeouts": 0}
                self.problem_components[name]["start_timeouts"] += 1
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Error iniciando componente {name}: {e}")
        
        # Marcar como iniciado 
        self.running = True
        self.started_at = time.time()
        logging.getLogger(__name__).info("Motor configurable Genesis iniciado")
        
        # Emitir evento con timeout configurable
        try:
            await asyncio.wait_for(
                self.event_bus.emit(
                    "system.started",
                    {"components": list(self.components.keys())},
                    "engine"
                ),
                timeout=self.event_timeout
            )
        except asyncio.TimeoutError:
            self.timeouts_occurred["event_emission"] += 1
            logging.getLogger(__name__).warning(
                f"Timeout al emitir evento system.started después de {self.event_timeout}s"
            )
        except Exception as e:
            logging.getLogger(__name__).error(f"Error al emitir evento system.started: {e}")
            
    async def stop(self, timeout: Optional[float] = None) -> None:
        """
        Detener el motor y todos los componentes registrados.
        
        Args:
            timeout: Tiempo máximo (en segundos) para esperar que los componentes se detengan
        """
        if not self.running:
            logging.getLogger(__name__).warning("Motor no está en ejecución")
            return
        
        logging.getLogger(__name__).info("Deteniendo motor configurable Genesis")
        
        # Usar timeout configurable o el proporcionado
        actual_timeout = timeout if timeout is not None else self.component_stop_timeout
            
        # Ordenar componentes por prioridad (menor prioridad primero para detener)
        ordered_components = sorted(
            self.components.items(),
            key=lambda x: self.operation_priorities.get(x[0], 50)  # Menor prioridad primero
        )
        
        # Detener componentes usando timeouts configurables
        for name, component in ordered_components:
            try:
                await asyncio.wait_for(component.stop(), timeout=actual_timeout)
            except asyncio.TimeoutError:
                self.timeouts_occurred["component_stop"] += 1
                logging.getLogger(__name__).warning(
                    f"Timeout al detener componente {name} después de {actual_timeout}s"
                )
                
                # Registrar componente problemático
                if name not in self.problem_components:
                    self.problem_components[name] = {"start_timeouts": 0, "stop_timeouts": 0, "event_timeouts": 0}
                self.problem_components[name]["stop_timeouts"] += 1
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Error al detener componente {name}: {e}")
        
        # Solo detener el EventBus si no estamos en modo prueba
        if not self.test_mode:
            try:
                await asyncio.wait_for(self.event_bus.stop(), timeout=self.event_timeout)
            except asyncio.TimeoutError:
                self.timeouts_occurred["event_emission"] += 1
                logging.getLogger(__name__).warning(
                    f"Timeout al detener event_bus después de {self.event_timeout}s"
                )
            except Exception as e:
                logging.getLogger(__name__).error(f"Error al detener event_bus: {e}")
        
        # Marcar como detenido
        self.running = False
        logging.getLogger(__name__).info("Motor configurable Genesis detenido")
            
    async def emit_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Emitir un evento a través del bus con timeout configurable.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Asegurar que el event bus está iniciado
        await self._ensure_event_bus_started()
        
        # Emitir evento con timeout configurable
        try:
            await asyncio.wait_for(
                self.event_bus.emit(event_type, data, source),
                timeout=self.event_timeout
            )
        except asyncio.TimeoutError:
            self.timeouts_occurred["event_emission"] += 1
            logging.getLogger(__name__).warning(
                f"Timeout al emitir evento {event_type} después de {self.event_timeout}s"
            )
        except Exception as e:
            logging.getLogger(__name__).error(f"Error al emitir evento {event_type}: {e}")
    
    async def emit_event_with_response(
        self, event_type: str, data: Dict[str, Any], source: str
    ) -> List[Any]:
        """
        Emitir un evento y esperar respuestas con timeout configurable.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
            
        Returns:
            Lista de respuestas de los manejadores
        """
        # Asegurar que el event bus está iniciado
        await self._ensure_event_bus_started()
        
        # Emitir evento con timeout configurable
        try:
            return await asyncio.wait_for(
                self.event_bus.emit_with_response(event_type, data, source),
                timeout=self.event_timeout
            )
        except asyncio.TimeoutError:
            self.timeouts_occurred["event_emission"] += 1
            logging.getLogger(__name__).warning(
                f"Timeout al esperar respuestas del evento {event_type} después de {self.event_timeout}s"
            )
            return []
        except Exception as e:
            logging.getLogger(__name__).error(f"Error al emitir evento con respuesta {event_type}: {e}")
            return []
    
    def get_timeout_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de timeouts.
        
        Returns:
            Diccionario con contadores de timeouts por categoría
        """
        return {
            "timeouts": dict(self.timeouts_occurred),
            "problem_components": self.problem_components
        }
    
    def adjust_timeouts_based_on_stats(self, factor: float = 1.5) -> None:
        """
        Ajustar timeouts automáticamente basado en estadísticas.
        
        Esta función aumenta los timeouts si se han producido demasiados timeouts.
        
        Args:
            factor: Factor de multiplicación para los timeouts actuales
        """
        if sum(self.timeouts_occurred.values()) > 5:
            logging.getLogger(__name__).info(
                f"Ajustando timeouts automáticamente (factor={factor})"
            )
            self.component_start_timeout *= factor
            self.component_stop_timeout *= factor
            self.event_timeout *= factor
            self.component_event_timeout *= factor