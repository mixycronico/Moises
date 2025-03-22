"""
Test ultra minimal para identificar el problema exacto en Engine.
"""

import asyncio
import pytest
import logging
from unittest.mock import Mock, AsyncMock

from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus


class MinimalComponent:
    """Componente mínimo sin herencia de Component."""
    
    def __init__(self, name):
        self.name = name
        self.event_bus = None
        
    async def start(self):
        print(f"Starting {self.name}")
        return None
        
    async def stop(self):
        print(f"Stopping {self.name}")
        return None
        
    async def handle_event(self, event_type, data, source):
        print(f"Event received by {self.name}: {event_type}")
        return None
        
    def attach_event_bus(self, event_bus):
        self.event_bus = event_bus


@pytest.mark.asyncio
async def test_engine_creation():
    """Test que solo crea un Engine."""
    # Crear EventBus con test_mode=True
    event_bus = EventBus(test_mode=True)
    
    # Crear Engine con test_mode=True
    engine = Engine(event_bus_or_name=event_bus, test_mode=True)
    
    # No hacer nada más, solo confirmar que se creó
    assert engine is not None
    assert engine.event_bus is event_bus


@pytest.mark.asyncio
async def test_engine_with_component():
    """Test que registra un componente pero no inicia el Engine."""
    # Configurar logging para ver mensajes de depuración
    logging.basicConfig(level=logging.DEBUG)
    
    # Crear EventBus con test_mode=True
    event_bus = EventBus(test_mode=True)
    
    # Crear Engine con test_mode=True
    engine = Engine(event_bus_or_name=event_bus, test_mode=True)
    
    # Crear un componente mínimo
    component = MinimalComponent("test_component")
    
    # Registrar el componente en el Engine
    engine.register_component(component)
    
    # No iniciar el Engine, solo confirmar que el componente se registró
    assert "test_component" in engine.components


@pytest.mark.asyncio
async def test_engine_start_only():
    """Test que inicia el Engine pero no lo detiene."""
    # Configurar logging para ver mensajes de depuración
    logging.basicConfig(level=logging.DEBUG)
    
    # Crear EventBus con test_mode=True
    event_bus = EventBus(test_mode=True)
    print("EventBus creado con test_mode=True")
    
    # Crear Engine con test_mode=True
    engine = Engine(event_bus_or_name=event_bus, test_mode=True)
    print("Engine creado con test_mode=True")
    
    # Crear un componente mínimo con mocks para start y stop
    component = MinimalComponent("test_component")
    component.start = AsyncMock()
    component.stop = AsyncMock()
    component.handle_event = AsyncMock()
    
    # Registrar el componente en el Engine
    engine.register_component(component)
    print("Componente registrado")
    
    # Iniciar el Engine con un timeout bajo
    try:
        print("Iniciando Engine...")
        await asyncio.wait_for(engine.start(), timeout=0.5)
        print("Engine iniciado correctamente")
    except asyncio.TimeoutError:
        print("Timeout al iniciar Engine")
    
    # No detener el Engine, solo confirmar que se inició
    assert engine.running is True