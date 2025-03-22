"""
Motor configurable con timeouts personalizables.

Este módulo implementa una versión mejorada del motor no bloqueante
que permite configurar los tiempos máximos para diferentes operaciones,
optimizando la ejecución de pruebas y mejorando la robustez del sistema.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Callable

from genesis.core.component import Component
from genesis.core.event_bus import EventBus
from genesis.core.engine_non_blocking import EngineNonBlocking

logger = logging.getLogger(__name__)

class ConfigurableTimeoutEngine(EngineNonBlocking):
    """
    Motor asíncrono con timeouts configurables para cada tipo de operación.
    
    Esta implementación extiende el motor no bloqueante básico añadiendo
    la capacidad de configurar tiempos máximos específicos para:
    - Inicio de componentes
    - Parada de componentes
    - Procesamiento de eventos
    - Emisión de eventos
    
    También registra estadísticas de timeouts para análisis y ajuste.
    """
    
    def __init__(self, 
                 component_start_timeout: float = 0.5,
                 component_stop_timeout: float = 0.5,
                 component_event_timeout: float = 0.3,
                 event_timeout: float = 1.0,
                 test_mode: bool = False):
        """
        Inicializar motor configurable.
        
        Args:
            component_start_timeout: Tiempo máximo en segundos para inicio de componente
            component_stop_timeout: Tiempo máximo en segundos para parada de componente
            component_event_timeout: Tiempo máximo en segundos para procesamiento de evento
            event_timeout: Tiempo máximo en segundos para emisión de evento
            test_mode: Modo de prueba (controles adicionales)
        """
        super().__init__()
        
        self._component_start_timeout = component_start_timeout
        self._component_stop_timeout = component_stop_timeout
        self._component_event_timeout = component_event_timeout
        self._event_timeout = event_timeout
        self._test_mode = test_mode
        
        # Estadísticas para análisis
        self._timeout_stats = {
            "timeouts": {
                "component_start": 0,
                "component_stop": 0,
                "component_event": 0,
                "event": 0
            },
            "successes": {
                "component_start": 0,
                "component_stop": 0,
                "component_event": 0,
                "event": 0
            }
        }
        
        # Métricas de rendimiento
        self._performance_metrics = {
            "component_start_times": [],
            "component_stop_times": [],
            "event_processing_times": []
        }
        
        logger.info(f"Motor configurable inicializado con timeouts: "
                   f"start={component_start_timeout}s, "
                   f"stop={component_stop_timeout}s, "
                   f"event={component_event_timeout}s, "
                   f"emit={event_timeout}s")
    
    async def _start_component(self, component: Component) -> None:
        """
        Iniciar componente con timeout configurable.
        
        Args:
            component: Componente a iniciar
            
        Returns:
            None
        """
        start_time = time.time()
        
        try:
            # Usar wait_for con timeout configurable
            await asyncio.wait_for(
                component.start(),
                timeout=self._component_start_timeout
            )
            
            # Registrar éxito
            self._timeout_stats["successes"]["component_start"] += 1
            elapsed = time.time() - start_time
            self._performance_metrics["component_start_times"].append(elapsed)
            
            logger.debug(f"Componente {component.name} iniciado en {elapsed:.3f}s")
            
        except asyncio.TimeoutError:
            # Registrar timeout
            self._timeout_stats["timeouts"]["component_start"] += 1
            elapsed = time.time() - start_time
            
            logger.warning(f"Timeout al iniciar componente {component.name} "
                          f"después de {elapsed:.3f}s (límite: {self._component_start_timeout}s)")
            
        except Exception as e:
            # Registrar error
            logger.error(f"Error al iniciar componente {component.name}: {str(e)}")
    
    async def _stop_component(self, component: Component) -> bool:
        """
        Detener componente con timeout configurable.
        
        Args:
            component: Componente a detener
            
        Returns:
            True si el componente se detuvo correctamente, False en caso contrario
        """
        start_time = time.time()
        
        try:
            # Usar wait_for con timeout configurable
            await asyncio.wait_for(
                component.stop(),
                timeout=self._component_stop_timeout
            )
            
            # Registrar éxito
            self._timeout_stats["successes"]["component_stop"] += 1
            elapsed = time.time() - start_time
            self._performance_metrics["component_stop_times"].append(elapsed)
            
            logger.debug(f"Componente {component.name} detenido en {elapsed:.3f}s")
            return True
            
        except asyncio.TimeoutError:
            # Registrar timeout
            self._timeout_stats["timeouts"]["component_stop"] += 1
            elapsed = time.time() - start_time
            
            logger.warning(f"Timeout al detener componente {component.name} "
                          f"después de {elapsed:.3f}s (límite: {self._component_stop_timeout}s)")
            return False
            
        except Exception as e:
            # Registrar error
            logger.error(f"Error al detener componente {component.name}: {str(e)}")
            return False
    
    async def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                         source: str = "system") -> None:
        """
        Emitir evento con timeout configurable.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento (opcional)
            source: Origen del evento (por defecto: "system")
            
        Returns:
            None
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} con motor detenido")
            return
            
        start_time = time.time()
        event_data = data or {}
        
        try:
            # Implementar versión simplificada de emisión de evento
            # Esto es más simple que la versión original y evita bloqueos
            for component in self._components.values():
                if not component:
                    continue
                    
                try:
                    # Usar gather con timeout para manejar múltiples componentes
                    # pero tratar cada uno individualmente para evitar que uno lento
                    # bloquee a los demás
                    await asyncio.wait_for(
                        component.handle_event(event_type, event_data, source),
                        timeout=self._component_event_timeout
                    )
                    
                    # Registrar éxito
                    self._timeout_stats["successes"]["component_event"] += 1
                    
                except asyncio.TimeoutError:
                    # Registrar timeout pero continuar con otros componentes
                    self._timeout_stats["timeouts"]["component_event"] += 1
                    logger.warning(f"Timeout al enviar evento {event_type} a {component.name} "
                                  f"(límite: {self._component_event_timeout}s)")
                    
                except Exception as e:
                    # Registrar error pero continuar con otros componentes
                    logger.error(f"Error al enviar evento {event_type} a {component.name}: {str(e)}")
            
            # Registrar éxito global
            self._timeout_stats["successes"]["event"] += 1
            elapsed = time.time() - start_time
            
            logger.debug(f"Evento {event_type} emitido en {elapsed:.3f}s")
            
        except Exception as e:
            # Registrar error global
            self._timeout_stats["timeouts"]["event"] += 1
            elapsed = time.time() - start_time
            
            logger.error(f"Error al emitir evento {event_type}: {str(e)} ({elapsed:.3f}s)")
    
    def get_timeout_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Obtener estadísticas de timeouts.
        
        Returns:
            Diccionario con estadísticas de timeouts y éxitos
        """
        return self._timeout_stats.copy()
    
    def get_performance_metrics(self) -> Dict[str, List[float]]:
        """
        Obtener métricas de rendimiento.
        
        Returns:
            Diccionario con tiempos de ejecución de diferentes operaciones
        """
        return self._performance_metrics.copy()
    
    def adjust_timeouts_based_on_stats(self, factor: float = 1.5, 
                                      min_failures: int = 3) -> None:
        """
        Ajustar timeouts basados en estadísticas recientes.
        
        Esta función aumenta los timeouts para operaciones que han experimentado
        múltiples fallos, mejorando la adaptabilidad del sistema.
        
        Args:
            factor: Factor de multiplicación para timeouts (default: 1.5)
            min_failures: Mínimo de fallos para aplicar ajuste (default: 3)
        """
        timeouts = self._timeout_stats["timeouts"]
        
        # Ajustar timeout de inicio de componente
        if timeouts["component_start"] >= min_failures:
            old_timeout = self._component_start_timeout
            self._component_start_timeout *= factor
            logger.info(f"Timeout de inicio ajustado: {old_timeout:.3f}s → {self._component_start_timeout:.3f}s")
        
        # Ajustar timeout de parada de componente
        if timeouts["component_stop"] >= min_failures:
            old_timeout = self._component_stop_timeout
            self._component_stop_timeout *= factor
            logger.info(f"Timeout de parada ajustado: {old_timeout:.3f}s → {self._component_stop_timeout:.3f}s")
        
        # Ajustar timeout de procesamiento de evento
        if timeouts["component_event"] >= min_failures:
            old_timeout = self._component_event_timeout
            self._component_event_timeout *= factor
            logger.info(f"Timeout de procesamiento ajustado: {old_timeout:.3f}s → {self._component_event_timeout:.3f}s")
        
        # Ajustar timeout de emisión de evento
        if timeouts["event"] >= min_failures:
            old_timeout = self._event_timeout
            self._event_timeout *= factor
            logger.info(f"Timeout de emisión ajustado: {old_timeout:.3f}s → {self._event_timeout:.3f}s")
        
        # Reiniciar contadores de timeouts
        for key in timeouts:
            timeouts[key] = 0