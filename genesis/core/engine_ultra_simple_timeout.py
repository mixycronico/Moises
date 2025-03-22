"""
Motor ultra simple con timeouts.

Este módulo implementa un motor extremadamente simplificado
con capacidad de manejar timeouts de forma segura.

Versión corregida que evita posibles bloqueos.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

from genesis.core.component import Component

# Configurar logging
logger = logging.getLogger(__name__)

class UltraSimpleTimeoutEngine:
    """
    Motor asíncrono ultra simplificado con timeouts.
    
    Esta implementación combina la simplicidad probada
    con capacidad de manejar timeouts de forma segura.
    """
    
    def __init__(self, 
                component_timeout: float = 0.5,
                event_timeout: float = 0.5):
        """
        Inicializar motor ultra simple con timeouts.
        
        Args:
            component_timeout: Tiempo máximo para operaciones de componentes
            event_timeout: Tiempo máximo para emisión de eventos
        """
        self._components = {}  # name -> component
        self.running = False
        self._component_timeout = component_timeout
        self._event_timeout = event_timeout
        
        # Estadísticas simples
        self._timeouts = 0
        self._successes = 0
        
        logger.info(f"Motor ultra simple creado con timeouts: "
                   f"component={component_timeout}s, event={event_timeout}s")
    
    def register_component(self, component: Component) -> None:
        """
        Registrar componente en el motor.
        
        Args:
            component: Componente a registrar
        """
        self._components[component.name] = component
        logger.info(f"Componente {component.name} registrado")
    
    async def start(self) -> None:
        """Iniciar el motor y todos los componentes registrados."""
        if self.running:
            logger.warning("Motor ya en ejecución")
            return
            
        logger.info("Iniciando motor ultra simple")
        
        # Iniciar componentes secuencialmente con timeout
        for name, component in self._components.items():
            try:
                # Crear tarea con timeout
                task = asyncio.create_task(component.start())
                try:
                    await asyncio.wait_for(task, timeout=self._component_timeout)
                    self._successes += 1
                    logger.info(f"Componente {name} iniciado")
                except asyncio.TimeoutError:
                    self._timeouts += 1
                    logger.warning(f"Timeout al iniciar componente {name} - continuando")
            except Exception as e:
                logger.error(f"Error al iniciar componente {name}: {str(e)}")
        
        self.running = True
        logger.info("Motor ultra simple iniciado")
    
    async def stop(self) -> None:
        """Detener el motor y todos los componentes registrados."""
        if not self.running:
            logger.warning("Motor ya detenido")
            return
            
        logger.info("Deteniendo motor ultra simple")
        
        # Detener componentes secuencialmente con timeout
        for name, component in self._components.items():
            try:
                # Crear tarea con timeout
                task = asyncio.create_task(component.stop())
                try:
                    await asyncio.wait_for(task, timeout=self._component_timeout)
                    self._successes += 1
                    logger.info(f"Componente {name} detenido")
                except asyncio.TimeoutError:
                    self._timeouts += 1
                    logger.warning(f"Timeout al detener componente {name} - continuando")
            except Exception as e:
                logger.error(f"Error al detener componente {name}: {str(e)}")
        
        self.running = False
        logger.info("Motor ultra simple detenido")
    
    async def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                       source: str = "system") -> bool:
        """
        Emitir evento a todos los componentes registrados.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento (opcional)
            source: Origen del evento (por defecto: "system")
            
        Returns:
            True si la emisión fue exitosa, False en caso contrario
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} con motor detenido")
            return False
        
        logger.info(f"Emitiendo evento {event_type} desde {source}")
        event_data = data or {}
        
        # Enviar evento a cada componente secuencialmente con timeout individual
        for name, component in self._components.items():
            try:
                # Crear tarea con timeout
                task = asyncio.create_task(component.handle_event(event_type, event_data, source))
                try:
                    await asyncio.wait_for(task, timeout=self._component_timeout)
                    self._successes += 1
                    logger.debug(f"Componente {name} procesó evento {event_type}")
                except asyncio.TimeoutError:
                    self._timeouts += 1
                    logger.warning(f"Timeout al procesar evento {event_type} en componente {name} - continuando")
            except Exception as e:
                logger.error(f"Error al procesar evento {event_type} en componente {name}: {str(e)}")
        
        logger.info(f"Evento {event_type} emitido")
        return True
    
    def get_stats(self) -> Dict[str, int]:
        """
        Obtener estadísticas simples del motor.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "timeouts": self._timeouts,
            "successes": self._successes
        }