"""
Motor básico con timeouts.

Este módulo implementa un motor simplificado con timeouts,
centrándose únicamente en la funcionalidad principal.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List

from genesis.core.component import Component

# Configurar logging
logger = logging.getLogger(__name__)

class BasicTimeoutEngine:
    """
    Motor asíncrono básico con timeouts.
    
    Esta implementación se centra en la funcionalidad esencial
    con manejo adecuado de timeouts.
    """
    
    def __init__(self, timeout: float = 0.5):
        """
        Inicializar motor con timeout.
        
        Args:
            timeout: Tiempo máximo (en segundos) para operaciones
        """
        self.timeout = timeout
        self.components = {}
        self.running = False
        logger.info(f"Motor básico con timeout de {timeout}s creado")
    
    def register_component(self, component: Component) -> None:
        """
        Registrar componente.
        
        Args:
            component: Componente a registrar
        """
        self.components[component.name] = component
        logger.info(f"Componente {component.name} registrado")
    
    async def start(self) -> None:
        """Iniciar motor y componentes."""
        if self.running:
            logger.warning("Motor ya está en ejecución")
            return
        
        logger.info("Iniciando motor básico")
        self.running = True
        
        # Iniciar componentes con timeout
        for name, component in self.components.items():
            try:
                task = asyncio.create_task(component.start())
                try:
                    await asyncio.wait_for(task, timeout=self.timeout)
                    logger.info(f"Componente {name} iniciado")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout al iniciar componente {name}")
            except Exception as e:
                logger.error(f"Error al iniciar componente {name}: {str(e)}")
        
        logger.info("Motor básico iniciado")
    
    async def stop(self) -> None:
        """Detener motor y componentes."""
        if not self.running:
            logger.warning("Motor ya está detenido")
            return
        
        logger.info("Deteniendo motor básico")
        
        # Detener componentes con timeout
        for name, component in self.components.items():
            try:
                task = asyncio.create_task(component.stop())
                try:
                    await asyncio.wait_for(task, timeout=self.timeout)
                    logger.info(f"Componente {name} detenido")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout al detener componente {name}")
            except Exception as e:
                logger.error(f"Error al detener componente {name}: {str(e)}")
        
        self.running = False
        logger.info("Motor básico detenido")
    
    async def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                       source: str = "system") -> None:
        """
        Emitir evento a componentes registrados.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} con motor detenido")
            return
        
        logger.info(f"Emitiendo evento {event_type} desde {source}")
        event_data = data or {}
        
        # Procesamiento secuencial con timeout individual
        for name, component in self.components.items():
            try:
                task = asyncio.create_task(
                    component.handle_event(event_type, event_data, source)
                )
                try:
                    await asyncio.wait_for(task, timeout=self.timeout)
                    logger.debug(f"Componente {name} procesó evento {event_type}")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout al procesar evento {event_type} en componente {name}")
            except Exception as e:
                logger.error(f"Error en componente {name} al procesar evento {event_type}: {str(e)}")
        
        logger.info(f"Evento {event_type} emitido a todos los componentes")