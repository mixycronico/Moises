"""
Prueba ultra-básica para el comportamiento de timeout.

Este módulo contiene una prueba muy simple para verificar
el comportamiento básico de timeout sin complejidades adicionales.
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, Optional

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicComponent(Component):
    """Componente básico que registra eventos."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.events = []
        self.started = False
        self.stopped = False
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Componente {self.name} iniciado")
        self.started = True
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Componente {self.name} detenido")
        self.stopped = True
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """Registrar evento recibido."""
        self.events.append({
            "type": event_type,
            "data": data.copy() if data else {},
            "source": source
        })
        logger.info(f"Componente {self.name} recibió evento {event_type}")
        return None


@pytest.mark.asyncio
async def test_basic_engine():
    """
    Prueba básica del motor sin bloqueo.
    
    Esta prueba verifica la funcionalidad fundamental sin comportamientos
    complejos que puedan causar timeouts en la prueba.
    """
    # Crear motor
    engine = EngineNonBlocking()
    
    # Crear componente simple
    comp = BasicComponent("test")
    
    # Registrar componente
    engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    
    # Verificar que el motor está iniciado
    assert engine.running, "El motor debería estar iniciado"
    
    # Verificar que el componente está iniciado
    assert comp.started, "El componente debería estar iniciado"
    
    # Emitir evento de prueba
    await engine.emit_event("test_event", {"data": "value"}, "test")
    
    # Pausa más larga para permitir procesamiento asíncrono
    await asyncio.sleep(0.5)
    
    # Depuración: imprimir todos los eventos recibidos
    logger.info(f"Eventos recibidos por {comp.name}: {comp.events}")
    
    # Verificar que el componente recibió eventos (incluyendo eventos del sistema)
    assert len(comp.events) > 0, "El componente debería haber recibido eventos"
    
    # Verificar que nuestro evento de prueba está entre los recibidos
    found_event = False
    for event in comp.events:
        # Examinar el tipo de evento y los datos
        if event["type"] == "test_event":
            found_event = True
            assert event["data"]["data"] == "value", "Los datos del evento deberían ser correctos"
            break
    
    # Verificar que se encontró el evento
    assert found_event, "El componente debería haber recibido el evento 'test_event'"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que el motor está detenido
    assert not engine.running, "El motor debería estar detenido"
    
    # Verificar que el componente está detenido
    assert comp.stopped, "El componente debería estar detenido"