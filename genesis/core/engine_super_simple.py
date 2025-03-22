"""
Motor ultra simplificado para identificar problemas de timeout.

Este módulo implementa un motor extremadamente simplificado
para identificar y solucionar problemas de timeout.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

from genesis.core.component import Component

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperSimpleEngine:
    """
    Motor asíncrono ultra simplificado.
    
    Esta versión elimina toda complejidad innecesaria para
    facilitar la identificación y solución de problemas.
    """
    
    def __init__(self):
        """Inicializar motor simplificado."""
        self._components = {}  # name -> component
        self.running = False
        logger.info("Motor ultra simplificado inicializado")
    
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
    
    async def start(self) -> None:
        """Iniciar el motor y todos los componentes."""
        if self.running:
            logger.warning("Motor ya en ejecución")
            return
        
        logger.info("Iniciando motor simplificado")
        
        # Iniciar todos los componentes de forma secuencial para simplificar
        start_time = time.time()
        for name, component in self._components.items():
            try:
                # Usar timeout corto pero suficiente
                await asyncio.wait_for(component.start(), timeout=1.0)
                logger.info(f"Componente {name} iniciado")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout al iniciar componente {name}")
            except Exception as e:
                logger.error(f"Error al iniciar componente {name}: {str(e)}")
        
        self.running = True
        elapsed = time.time() - start_time
        logger.info(f"Motor iniciado en {elapsed:.3f}s")
    
    async def stop(self) -> None:
        """Detener el motor y todos los componentes."""
        if not self.running:
            logger.warning("Motor ya detenido")
            return
        
        logger.info("Deteniendo motor simplificado")
        
        # Detener todos los componentes de forma secuencial para simplificar
        stop_time = time.time()
        for name, component in self._components.items():
            try:
                # Usar timeout corto pero suficiente
                await asyncio.wait_for(component.stop(), timeout=1.0)
                logger.info(f"Componente {name} detenido")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout al detener componente {name}")
            except Exception as e:
                logger.error(f"Error al detener componente {name}: {str(e)}")
        
        self.running = False
        elapsed = time.time() - stop_time
        logger.info(f"Motor detenido en {elapsed:.3f}s")
    
    async def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                        source: str = "system") -> None:
        """
        Emitir evento a todos los componentes.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento (opcional)
            source: Origen del evento (por defecto: "system")
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} con motor detenido")
            return
        
        logger.info(f"Emitiendo evento {event_type} desde {source}")
        event_data = data or {}
        
        # Enviar evento a cada componente de forma secuencial para simplificar
        for name, component in self._components.items():
            try:
                # Usar timeout corto pero suficiente
                await asyncio.wait_for(
                    component.handle_event(event_type, event_data, source),
                    timeout=0.5
                )
                logger.debug(f"Componente {name} procesó evento {event_type}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout al procesar evento {event_type} en componente {name}")
            except Exception as e:
                logger.error(f"Error al procesar evento {event_type} en componente {name}: {str(e)}")
        
        logger.info(f"Evento {event_type} emitido a todos los componentes")