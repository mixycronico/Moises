"""
Test para la versión optimizada de Engine.
"""

import asyncio
import pytest
import logging
from unittest.mock import Mock, AsyncMock

from genesis.core.event_bus import EventBus
from genesis.core.component import Component
from genesis.core.engine_optimized import EngineOptimized


class SimpleTestComponent(Component):
    """Componente simple para pruebas."""
    
    def __init__(self, name):
        """Inicializar componente."""
        super().__init__(name)
        self.started = False
        self.stopped = False
        self.events = []
    
    async def start(self):
        """Iniciar componente."""
        self.started = True
    
    async def stop(self):
        """Detener componente."""
        self.stopped = True
    
    async def handle_event(self, event_type, data, source):
        """Manejar evento."""
        self.events.append((event_type, data, source))
        return None


@pytest.fixture
def event_bus():
    """Proporcionar event bus para pruebas."""
    return EventBus(test_mode=True)


@pytest.fixture
def engine(event_bus):
    """Proporcionar engine optimizado con componentes para pruebas."""
    engine = EngineOptimized(event_bus_or_name=event_bus, test_mode=True)
    
    # Registrar componentes
    component1 = SimpleTestComponent("component1")
    component2 = SimpleTestComponent("component2")
    
    engine.register_component(component1, priority=80)
    engine.register_component(component2, priority=60)
    
    return engine


@pytest.mark.asyncio
async def test_engine_optimized_start_stop(engine):
    """Verificar que el engine optimizado puede iniciar y detenerse correctamente."""
    # Iniciar el engine
    await engine.start()
    
    # Verificar que está en ejecución
    assert engine.running is True
    
    # Verificar que los componentes se iniciaron por orden de prioridad
    component1 = engine.components["component1"]
    component2 = engine.components["component2"]
    
    assert component1.started is True
    assert component2.started is True
    
    # Detener el engine
    await engine.stop()
    
    # Verificar que está detenido
    assert engine.running is False
    
    # Verificar que los componentes se detuvieron
    assert component1.stopped is True
    assert component2.stopped is True


@pytest.mark.asyncio
async def test_engine_optimized_emit_event(engine):
    """Verificar que el engine optimizado puede emitir eventos correctamente."""
    # Iniciar el engine
    await engine.start()
    
    # Emitir evento
    event_data = {"message": "Test message"}
    await engine.event_bus.emit("test.event", event_data, "test_source")
    
    # Verificar que todos los componentes recibieron el evento
    for component in engine.components.values():
        # Debe haber al menos 1 evento (puede haber system.started también)
        assert len(component.events) >= 1
        # Buscar el evento test.event en la lista de eventos recibidos
        test_event = next((e for e in component.events if e[0] == "test.event"), None)
        assert test_event is not None
        assert test_event[1] == event_data
        assert test_event[2] == "test_source"
    
    # Detener el engine
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_optimized_component_communication(engine):
    """Verificar la comunicación entre componentes en el engine optimizado."""
    # Iniciar el engine
    await engine.start()
    
    # Obtener componentes
    component1 = engine.components["component1"]
    component2 = engine.components["component2"]
    
    # Emitir evento desde component1
    event_data = {"message": "Hello from component1"}
    await engine.event_bus.emit("component.message", event_data, "component1")
    
    # Verificar que component2 recibió el evento
    assert len(component2.events) >= 1
    # Buscar el evento component.message en la lista de eventos recibidos
    component_event = next((e for e in component2.events if e[0] == "component.message"), None)
    assert component_event is not None
    assert component_event[1] == event_data
    assert component_event[2] == "component1"
    
    # Detener el engine
    await engine.stop()