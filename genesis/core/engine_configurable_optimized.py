"""
Motor configurable optimizado para testing.

Este módulo implementa una versión optimizada del motor con timeouts
configurables, basada en la arquitectura simple que ha demostrado funcionar
correctamente en las pruebas.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set

from genesis.core.component import Component

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigurableTimeoutEngineOptimized:
    """
    Motor asíncrono con timeouts configurables optimizado para testing.
    
    Esta implementación combina la simplicidad del motor supersimple
    con la capacidad de configurar timeouts para diferentes operaciones.
    """
    
    def __init__(self, 
                 component_start_timeout: float = 1.0,
                 component_stop_timeout: float = 1.0,
                 component_event_timeout: float = 0.5,
                 event_timeout: float = 2.0):
        """
        Inicializar motor configurable optimizado.
        
        Args:
            component_start_timeout: Tiempo máximo en segundos para inicio de componente
            component_stop_timeout: Tiempo máximo en segundos para parada de componente
            component_event_timeout: Tiempo máximo en segundos para procesamiento de evento
            event_timeout: Tiempo máximo en segundos para emisión de evento
        """
        self._components = {}  # name -> component
        self.running = False
        
        # Configuración de timeouts
        self._component_start_timeout = component_start_timeout
        self._component_stop_timeout = component_stop_timeout
        self._component_event_timeout = component_event_timeout
        self._event_timeout = event_timeout
        
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
        
        logger.info(f"Motor configurable optimizado inicializado con timeouts: "
                   f"start={component_start_timeout}s, "
                   f"stop={component_stop_timeout}s, "
                   f"event={component_event_timeout}s, "
                   f"emit={event_timeout}s")
    
    def register_component(self, component: Component) -> None:
        """
        Registrar componente en el motor.
        
        Args:
            component: Componente a registrar
        """
        name = component.name
        if name in self._components:
            logger.warning(f"Componente {name} ya registrado, reemplazando")
        
        self._components[name] = component
        logger.info(f"Componente {name} registrado en el motor")
    
    def get_component(self, name: str) -> Optional[Component]:
        """
        Obtener componente por nombre.
        
        Args:
            name: Nombre del componente
            
        Returns:
            Componente o None si no existe
        """
        return self._components.get(name)
    
    def get_components(self) -> Dict[str, Component]:
        """
        Obtener todos los componentes.
        
        Returns:
            Diccionario con todos los componentes (nombre -> componente)
        """
        return self._components.copy()
    
    async def start(self) -> None:
        """Iniciar el motor y todos los componentes registrados."""
        if self.running:
            logger.warning("Motor ya en ejecución")
            return
        
        logger.info("Iniciando motor configurable optimizado")
        start_time = time.time()
        
        # Iniciar todos los componentes con manejo de timeouts
        success_count = 0
        timeout_count = 0
        
        for name, component in self._components.items():
            try:
                # Iniciar con timeout configurable
                await asyncio.wait_for(
                    component.start(),
                    timeout=self._component_start_timeout
                )
                self._timeout_stats["successes"]["component_start"] += 1
                success_count += 1
                logger.info(f"Componente {name} iniciado correctamente")
                
            except asyncio.TimeoutError:
                # Registrar timeout pero continuar con otros componentes
                self._timeout_stats["timeouts"]["component_start"] += 1
                timeout_count += 1
                logger.warning(f"Timeout al iniciar componente {name} "
                              f"(límite: {self._component_start_timeout}s)")
                
            except Exception as e:
                # Registrar error pero continuar con otros componentes
                logger.error(f"Error al iniciar componente {name}: {str(e)}")
        
        self.running = True
        elapsed = time.time() - start_time
        logger.info(f"Motor iniciado en {elapsed:.3f}s "
                   f"({success_count} éxitos, {timeout_count} timeouts)")
    
    async def stop(self) -> None:
        """Detener el motor y todos los componentes registrados."""
        if not self.running:
            logger.warning("Motor ya detenido")
            return
        
        logger.info("Deteniendo motor configurable optimizado")
        stop_time = time.time()
        
        # Detener todos los componentes con manejo de timeouts
        success_count = 0
        timeout_count = 0
        
        for name, component in self._components.items():
            try:
                # Detener con timeout configurable
                await asyncio.wait_for(
                    component.stop(),
                    timeout=self._component_stop_timeout
                )
                self._timeout_stats["successes"]["component_stop"] += 1
                success_count += 1
                logger.info(f"Componente {name} detenido correctamente")
                
            except asyncio.TimeoutError:
                # Registrar timeout pero continuar con otros componentes
                self._timeout_stats["timeouts"]["component_stop"] += 1
                timeout_count += 1
                logger.warning(f"Timeout al detener componente {name} "
                              f"(límite: {self._component_stop_timeout}s)")
                
            except Exception as e:
                # Registrar error pero continuar con otros componentes
                logger.error(f"Error al detener componente {name}: {str(e)}")
        
        self.running = False
        elapsed = time.time() - stop_time
        logger.info(f"Motor detenido en {elapsed:.3f}s "
                   f"({success_count} éxitos, {timeout_count} timeouts)")
    
    async def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                         source: str = "system") -> bool:
        """
        Emitir evento a todos los componentes registrados.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento (opcional)
            source: Origen del evento (por defecto: "system")
            
        Returns:
            True si el evento se emitió sin errores globales, False en caso contrario
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} con motor detenido")
            return False
        
        logger.info(f"Emitiendo evento {event_type} desde {source}")
        start_time = time.time()
        event_data = data or {}
        
        # Control de tiempo global para esta emisión
        try:
            async def _process_all_components():
                success_count = 0
                timeout_count = 0
                
                # Enviar evento a cada componente con manejo individual de timeouts
                for name, component in self._components.items():
                    try:
                        # Procesar evento con timeout configurable
                        await asyncio.wait_for(
                            component.handle_event(event_type, event_data, source),
                            timeout=self._component_event_timeout
                        )
                        self._timeout_stats["successes"]["component_event"] += 1
                        success_count += 1
                        logger.debug(f"Componente {name} procesó evento {event_type}")
                        
                    except asyncio.TimeoutError:
                        # Registrar timeout pero continuar con otros componentes
                        self._timeout_stats["timeouts"]["component_event"] += 1
                        timeout_count += 1
                        logger.warning(f"Timeout al procesar evento {event_type} en componente {name} "
                                      f"(límite: {self._component_event_timeout}s)")
                        
                    except Exception as e:
                        # Registrar error pero continuar con otros componentes
                        logger.error(f"Error al procesar evento {event_type} en componente {name}: {str(e)}")
                
                return success_count, timeout_count
            
            # Ejecutar procesamiento global con timeout
            await asyncio.wait_for(
                _process_all_components(),
                timeout=self._event_timeout
            )
            
            # Registrar éxito global
            self._timeout_stats["successes"]["event"] += 1
            elapsed = time.time() - start_time
            logger.info(f"Evento {event_type} emitido a todos los componentes en {elapsed:.3f}s")
            return True
            
        except asyncio.TimeoutError:
            # Registrar timeout global
            self._timeout_stats["timeouts"]["event"] += 1
            elapsed = time.time() - start_time
            logger.warning(f"Timeout global al emitir evento {event_type} "
                          f"después de {elapsed:.3f}s (límite: {self._event_timeout}s)")
            return False
            
        except Exception as e:
            # Registrar error global
            elapsed = time.time() - start_time
            logger.error(f"Error al emitir evento {event_type}: {str(e)} ({elapsed:.3f}s)")
            return False
    
    def get_timeout_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Obtener estadísticas de timeouts.
        
        Returns:
            Diccionario con estadísticas de timeouts y éxitos
        """
        return self._timeout_stats.copy()
    
    def adjust_timeouts(self, 
                       component_start_timeout: Optional[float] = None, 
                       component_stop_timeout: Optional[float] = None,
                       component_event_timeout: Optional[float] = None,
                       event_timeout: Optional[float] = None) -> None:
        """
        Ajustar manualmente los timeouts configurables.
        
        Args:
            component_start_timeout: Nuevo timeout para inicio de componentes (opcional)
            component_stop_timeout: Nuevo timeout para parada de componentes (opcional)
            component_event_timeout: Nuevo timeout para procesamiento de eventos (opcional)
            event_timeout: Nuevo timeout para emisión de eventos (opcional)
        """
        old_start = self._component_start_timeout
        old_stop = self._component_stop_timeout
        old_event = self._component_event_timeout
        old_emit = self._event_timeout
        
        # Actualizar solo los timeouts especificados
        if component_start_timeout is not None:
            self._component_start_timeout = component_start_timeout
            
        if component_stop_timeout is not None:
            self._component_stop_timeout = component_stop_timeout
            
        if component_event_timeout is not None:
            self._component_event_timeout = component_event_timeout
            
        if event_timeout is not None:
            self._event_timeout = event_timeout
        
        # Registrar cambios
        logger.info(f"Timeouts ajustados: "
                   f"start: {old_start}s → {self._component_start_timeout}s, "
                   f"stop: {old_stop}s → {self._component_stop_timeout}s, "
                   f"event: {old_event}s → {self._component_event_timeout}s, "
                   f"emit: {old_emit}s → {self._event_timeout}s")
    
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