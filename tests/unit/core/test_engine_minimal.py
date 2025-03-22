"""
Test minimal para detectar problemas de deadlock en Engine.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from genesis.core.component import Component
from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus


class SimpleComponent(Component):
    """Componente simple para pruebas."""
    
    def __init__(self, name):
        """Inicializar con nombre."""
        super().__init__(name)
        self.started = False
        self.stopped = False
        self.events = []
    
    async def start(self):
        """Marcar como iniciado."""
        self.started = True
    
    async def stop(self):
        """Marcar como detenido."""
        self.stopped = True
    
    async def handle_event(self, event_type, data, source):
        """Registrar evento recibido."""
        self.events.append((event_type, data, source))
        return None


@pytest.fixture
def event_bus():
    """Crear un bus de eventos para pruebas."""
    return EventBus(test_mode=True)


@pytest.fixture
def engine(event_bus):
    """Crear un motor con un componente simple."""
    engine = Engine(event_bus_or_name=event_bus, test_mode=True)
    component = SimpleComponent("test_component")
    engine.register_component(component)
    return engine


@pytest.mark.asyncio
async def test_engine_start_stop(engine):
    """Verificar que el motor puede iniciar y detener correctamente."""
    # Iniciar el motor
    await engine.start()
    assert engine.running is True
    
    # Comprobar que el componente se inició
    component = engine.components["test_component"]
    assert component.started is True
    
    # Detener el motor
    await engine.stop()
    assert engine.running is False
    
    # Comprobar que el componente se detuvo
    assert component.stopped is True


@pytest.mark.asyncio
async def test_engine_emit_event(engine):
    """Verificar que el motor puede emitir eventos correctamente."""
    # Iniciar el motor
    await engine.start()
    
    # Emitir un evento simple
    test_data = {"message": "test"}
    await engine.event_bus.emit("test.event", test_data, "test_source")
    
    # Verificar que el componente recibió el evento
    component = engine.components["test_component"]
    assert len(component.events) == 1
    event = component.events[0]
    assert event[0] == "test.event"
    assert event[1] == test_data
    assert event[2] == "test_source"
    
    # Detener el motor
    await engine.stop()