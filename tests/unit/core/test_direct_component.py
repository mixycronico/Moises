"""
Prueba directa de componentes.

Esta prueba utiliza componentes aislados, sin interacción con el motor,
para verificar que los componentes funcionan correctamente y ayudar a
diagnosticar problemas con las pruebas más complejas.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, Optional

from genesis.core.component import Component

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestComponent(Component):
    """Componente simple para pruebas."""
    
    def __init__(self, name: str):
        """
        Inicializar componente.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.events = []
        self.started = False
        self.stopped = False
        logger.info(f"Componente {name} creado")
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Componente {self.name} iniciando")
        self.started = True
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Componente {self.name} deteniendo")
        self.stopped = True
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            
        Returns:
            Respuesta opcional
        """
        logger.info(f"Componente {self.name} recibiendo evento {event_type} de {source}")
        self.events.append({
            "type": event_type,
            "data": data.copy() if data else {},
            "source": source
        })
        logger.info(f"Componente {self.name} registró evento {event_type}")
        return None


@pytest.mark.asyncio
async def test_component_directly():
    """
    Prueba un componente directamente, sin depender del motor.
    
    Esta prueba verifica que el componente registra eventos correctamente
    cuando se llama a su método handle_event directamente.
    """
    # Crear componente
    comp = TestComponent("test_direct")
    
    # Iniciar componente directamente
    await comp.start()
    
    # Verificar que el componente está iniciado
    assert comp.started, "El componente debería estar iniciado"
    
    # Llamar directamente al manejador de eventos
    await comp.handle_event("test_event", {"data": "value"}, "test")
    
    # Verificar que el evento fue registrado
    assert len(comp.events) == 1, "El componente debería haber registrado el evento"
    assert comp.events[0]["type"] == "test_event", "El tipo de evento debería ser correcto"
    assert comp.events[0]["data"]["data"] == "value", "Los datos del evento deberían ser correctos"
    
    # Detener componente
    await comp.stop()
    
    # Verificar que el componente está detenido
    assert comp.stopped, "El componente debería estar detenido"