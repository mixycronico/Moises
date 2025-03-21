"""
Tests básicos para los componentes core del sistema Genesis.

Este módulo prueba las funcionalidades básicas de los componentes core,
incluyendo la configuración, eventos, componentes y el motor principal.
"""

import pytest
import asyncio
import logging
import time
from unittest.mock import Mock, patch, AsyncMock

from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus
from genesis.core.component import Component
from genesis.core.config import Config
from genesis.core.logger import Logger


class TestComponent(Component):
    """Componente de prueba para testing."""
    
    def __init__(self, name="test_component"):
        """Inicializar componente con contadores para seguimiento."""
        super().__init__(name)
        self.start_count = 0
        self.stop_count = 0
        self.events_received = []
        self.is_started = False
    
    async def start(self):
        """Iniciar el componente y registrar la llamada."""
        self.start_count += 1
        self.is_started = True
        return True
    
    async def stop(self):
        """Detener el componente y registrar la llamada."""
        self.stop_count += 1
        self.is_started = False
        return True
    
    async def handle_event(self, event_type, data, source):
        """Manejar un evento y registrarlo."""
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source
        })


@pytest.fixture
def config():
    """Proporcionar una instancia de configuración para pruebas."""
    return Config()


@pytest.fixture
def event_bus():
    """Proporcionar un bus de eventos para pruebas."""
    return EventBus()


@pytest.fixture
def test_component():
    """Proporcionar un componente de prueba."""
    return TestComponent()


@pytest.fixture
def engine(event_bus):
    """Proporcionar un motor del sistema para pruebas."""
    return Engine(event_bus)


@pytest.mark.asyncio
async def test_config_basic_operations(config):
    """Probar operaciones básicas de configuración."""
    # Configurar un valor
    config.set("test_key", "test_value")
    
    # Verificar que se guardó correctamente
    assert config.get("test_key") == "test_value"
    
    # Obtener un valor con default
    assert config.get("non_existent_key", "default") == "default"
    
    # Verificar tipos diferentes
    config.set("int_key", 123)
    config.set("float_key", 3.14)
    config.set("bool_key", True)
    config.set("list_key", [1, 2, 3])
    config.set("dict_key", {"a": 1, "b": 2})
    
    assert config.get("int_key") == 123
    assert config.get("float_key") == 3.14
    assert config.get("bool_key") is True
    assert config.get("list_key") == [1, 2, 3]
    assert config.get("dict_key") == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_logger_setup():
    """Probar configuración básica del logger."""
    # Probar inicialización del logger
    logger = Logger.setup_logging(level=logging.DEBUG)
    
    # Verificar que el logger se configuró correctamente
    assert logger is not None
    
    # Verificar que los handlers se configuraron
    assert len(logger.handlers) > 0
    
    # Verificar nivel de log
    assert logger.level == logging.DEBUG


@pytest.mark.asyncio
async def test_event_bus_basic(event_bus):
    """Probar funcionalidades básicas del bus de eventos."""
    # Preparar un listener
    received_events = []
    
    async def test_listener(event_type, data, source):
        received_events.append({
            "type": event_type,
            "data": data,
            "source": source
        })
    
    # Registrar el listener
    event_bus.register_listener("test_event", test_listener)
    
    # Emitir un evento
    await event_bus.emit("test_event", {"message": "Hello"}, "test_source")
    
    # Verificar que se recibió el evento
    assert len(received_events) == 1
    assert received_events[0]["type"] == "test_event"
    assert received_events[0]["data"] == {"message": "Hello"}
    assert received_events[0]["source"] == "test_source"
    
    # Emitir otro tipo de evento (no debería ser capturado por el listener)
    await event_bus.emit("other_event", {"message": "Not captured"}, "test_source")
    
    # Verificar que no se capturó el segundo evento
    assert len(received_events) == 1
    
    # Desregistrar listener
    event_bus.unregister_listener("test_event", test_listener)
    
    # Emitir evento después de desregistrar
    await event_bus.emit("test_event", {"message": "After unregister"}, "test_source")
    
    # Verificar que no se recibió el evento
    assert len(received_events) == 1


@pytest.mark.asyncio
async def test_component_basic(test_component):
    """Probar funcionalidades básicas de un componente."""
    # Verificar el nombre
    assert test_component.name == "test_component"
    
    # Verificar que los contadores de start y stop comienzan en 0
    assert test_component.start_count == 0
    assert test_component.stop_count == 0
    
    # Iniciar el componente
    result = await test_component.start()
    assert result is True
    assert test_component.start_count == 1
    assert test_component.is_started is True
    
    # Detener el componente
    result = await test_component.stop()
    assert result is True
    assert test_component.stop_count == 1
    assert test_component.is_started is False
    
    # Verificar que no se han recibido eventos
    assert len(test_component.events_received) == 0


@pytest.mark.asyncio
async def test_component_handle_event(test_component):
    """Probar el manejo de eventos de un componente."""
    # Enviar un evento al componente
    await test_component.handle_event("test_event", {"message": "Test"}, "test_source")
    
    # Verificar que se registró el evento
    assert len(test_component.events_received) == 1
    assert test_component.events_received[0]["type"] == "test_event"
    assert test_component.events_received[0]["data"] == {"message": "Test"}
    assert test_component.events_received[0]["source"] == "test_source"
    
    # Enviar otro evento
    await test_component.handle_event("another_event", {"count": 123}, "another_source")
    
    # Verificar que se registró el segundo evento
    assert len(test_component.events_received) == 2
    assert test_component.events_received[1]["type"] == "another_event"
    assert test_component.events_received[1]["data"] == {"count": 123}
    assert test_component.events_received[1]["source"] == "another_source"


@pytest.mark.asyncio
async def test_engine_basic(engine, test_component):
    """Probar funcionamiento básico del motor del sistema."""
    # Registrar un componente
    engine.register_component(test_component)
    
    # Verificar que el componente se registró
    assert test_component.name in engine.components
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que el componente se inició
    assert test_component.start_count == 1
    assert test_component.is_started is True
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que el componente se detuvo
    assert test_component.stop_count == 1
    assert test_component.is_started is False


@pytest.mark.asyncio
async def test_engine_event_propagation(engine, test_component, event_bus):
    """Probar la propagación de eventos a través del motor."""
    # Registrar el componente
    engine.register_component(test_component)
    
    # Iniciar el motor
    await engine.start()
    
    # Emitir un evento
    await event_bus.emit("test_event", {"message": "Via engine"}, "test_source")
    
    # Verificar que el componente recibió el evento
    assert len(test_component.events_received) == 1
    assert test_component.events_received[0]["type"] == "test_event"
    assert test_component.events_received[0]["data"] == {"message": "Via engine"}
    assert test_component.events_received[0]["source"] == "test_source"
    
    # Detener el motor
    await engine.stop()
"""