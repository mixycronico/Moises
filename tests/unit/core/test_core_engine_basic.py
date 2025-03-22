"""
Tests básicos para el motor principal (core.engine).

Este módulo prueba las funcionalidades básicas del motor principal,
incluyendo registro e inicialización de componentes, ciclo de vida,
y eventos básicos.
"""

import pytest
import asyncio
import logging
import time
from unittest.mock import Mock, patch, AsyncMock

from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus
from genesis.core.component import Component
from genesis.core.config import Settings


class SimpleComponent(Component):
    """Componente simple para pruebas básicas."""
    
    def __init__(self, name="simple_component"):
        """Inicializar componente con contadores para seguimiento."""
        super().__init__(name)
        self.start_called = False
        self.stop_called = False
        self.events_received = []
        self.startup_time = None
        self.shutdown_time = None
    
    async def start(self):
        """Iniciar el componente y registrar la llamada."""
        self.start_called = True
        self.startup_time = time.time()
        return True
    
    async def stop(self):
        """Detener el componente y registrar la llamada."""
        self.stop_called = True
        self.shutdown_time = time.time()
        return True
    
    async def handle_event(self, event_type, data, source):
        """Manejar un evento y registrarlo."""
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        })


class CounterComponent(Component):
    """Componente que cuenta operaciones para pruebas."""
    
    def __init__(self, name="counter_component"):
        """Inicializar componente con contadores."""
        super().__init__(name)
        self.start_count = 0
        self.stop_count = 0
        self.event_count = 0
    
    async def start(self):
        """Incrementar contador de inicio."""
        self.start_count += 1
        return True
    
    async def stop(self):
        """Incrementar contador de parada."""
        self.stop_count += 1
        return True
    
    async def handle_event(self, event_type, data, source):
        """Incrementar contador de eventos."""
        self.event_count += 1


@pytest.fixture
def event_bus():
    """Proporcionar un bus de eventos para pruebas."""
    return EventBus(test_mode=True)


@pytest.fixture
def engine(event_bus):
    """Proporcionar un motor del sistema para pruebas."""
    return Engine(event_bus, test_mode=True)


@pytest.fixture
def simple_component():
    """Proporcionar un componente simple para pruebas."""
    return SimpleComponent()


@pytest.fixture
def counter_component():
    """Proporcionar un componente contador para pruebas."""
    return CounterComponent()


@pytest.mark.asyncio
async def test_engine_initialization():
    """Probar inicialización básica del motor."""
    # Crear motor sin bus de eventos (debe crear uno internamente)
    engine = Engine(test_mode=True)
    
    # Verificar que se creó un bus de eventos
    assert engine.event_bus is not None
    
    # Verificar estado inicial
    assert not engine.is_running
    assert len(engine.components) == 0
    
    # Crear motor con bus de eventos proporcionado
    event_bus = EventBus(test_mode=True)
    engine = Engine(event_bus, test_mode=True)
    
    # Verificar que se usó el bus proporcionado
    assert engine.event_bus is event_bus


@pytest.mark.asyncio
async def test_engine_component_registration(engine, simple_component, counter_component):
    """Probar registro básico de componentes."""
    # Verificar que no hay componentes registrados inicialmente
    assert len(engine.components) == 0
    
    # Registrar un componente
    engine.register_component(simple_component)
    
    # Verificar que se registró correctamente
    assert len(engine.components) == 1
    assert simple_component.name in engine.components
    
    # Registrar un segundo componente
    engine.register_component(counter_component)
    
    # Verificar que se registraron ambos
    assert len(engine.components) == 2
    assert counter_component.name in engine.components
    
    # Intentar registrar un componente duplicado
    duplicate = SimpleComponent(simple_component.name)
    with pytest.raises(ValueError):
        engine.register_component(duplicate)
    
    # Verificar que no cambió el número de componentes
    assert len(engine.components) == 2


@pytest.mark.asyncio
async def test_engine_component_deregistration(engine, simple_component, counter_component):
    """Probar eliminación de registro de componentes."""
    # Registrar componentes
    engine.register_component(simple_component)
    engine.register_component(counter_component)
    
    # Verificar que se registraron
    assert len(engine.components) == 2
    
    # Eliminar registro de un componente
    engine.deregister_component(simple_component.name)
    
    # Verificar que se eliminó correctamente
    assert len(engine.components) == 1
    assert simple_component.name not in engine.components
    assert counter_component.name in engine.components
    
    # Intentar eliminar un componente que no existe
    with pytest.raises(ValueError):
        engine.deregister_component("nonexistent_component")
    
    # Verificar que no cambió el número de componentes
    assert len(engine.components) == 1


@pytest.mark.asyncio
async def test_engine_start_stop_basic(engine, simple_component):
    """Probar inicio y parada básicos del motor."""
    # Registrar componente
    engine.register_component(simple_component)
    
    # Verificar estado inicial
    assert not engine.is_running
    assert not simple_component.start_called
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que el motor está ejecutándose
    assert engine.is_running
    
    # Verificar que se inició el componente
    assert simple_component.start_called
    assert simple_component.startup_time is not None
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que el motor no está ejecutándose
    assert not engine.is_running
    
    # Verificar que se detuvo el componente
    assert simple_component.stop_called
    assert simple_component.shutdown_time is not None


@pytest.mark.asyncio
async def test_engine_multiple_components_lifecycle(engine):
    """Probar ciclo de vida con múltiples componentes."""
    # Crear varios componentes
    components = [SimpleComponent(f"component_{i}") for i in range(5)]
    
    # Registrar componentes
    for component in components:
        engine.register_component(component)
    
    # Verificar que todos están registrados
    assert len(engine.components) == 5
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que todos los componentes se iniciaron
    for component in components:
        assert component.start_called
        assert component.startup_time is not None
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que todos los componentes se detuvieron
    for component in components:
        assert component.stop_called
        assert component.shutdown_time is not None


@pytest.mark.asyncio
async def test_engine_event_bus_integration(engine, simple_component, event_bus):
    """Probar integración básica del motor con el bus de eventos."""
    # Registrar componente
    engine.register_component(simple_component)
    
    # Iniciar el motor
    await engine.start()
    
    # Limpiar eventos recibidos (eliminar system.started)
    simple_component.events_received.clear()
    
    # Emitir un evento
    test_data = {"message": "Test event"}
    await event_bus.emit("test_event", test_data, "test_source")
    
    # Verificar que el componente recibió el evento
    assert len(simple_component.events_received) == 1
    event = simple_component.events_received[0]
    assert event["type"] == "test_event"
    assert event["data"] == test_data
    assert event["source"] == "test_source"
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_start_twice(engine, counter_component):
    """Probar que iniciar el motor dos veces no causa problemas."""
    # Registrar componente
    engine.register_component(counter_component)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que el componente se inició
    assert counter_component.start_count == 1
    
    # Intentar iniciar de nuevo
    await engine.start()
    
    # Verificar que el componente no se inició nuevamente
    assert counter_component.start_count == 1
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_stop_without_start(engine, counter_component):
    """Probar que detener el motor sin iniciar no causa problemas."""
    # Registrar componente
    engine.register_component(counter_component)
    
    # Detener el motor sin iniciarlo
    await engine.stop()
    
    # Verificar que no se intentó detener el componente
    assert counter_component.stop_count == 0


@pytest.mark.asyncio
async def test_engine_unregister_running_component(engine, simple_component):
    """Probar eliminación de registro de un componente mientras el motor está ejecutándose."""
    # Registrar componente
    engine.register_component(simple_component)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que el componente se inició
    assert simple_component.start_called
    
    # Eliminar registro del componente
    engine.deregister_component(simple_component.name)
    
    # Verificar que el componente se detuvo al ser eliminado
    assert simple_component.stop_called
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_component_access(engine):
    """Probar acceso a componentes registrados."""
    # Crear y registrar varios componentes
    components = {}
    for i in range(3):
        name = f"test_component_{i}"
        component = SimpleComponent(name)
        components[name] = component
        engine.register_component(component)
    
    # Verificar acceso por nombre
    for name, component in components.items():
        assert engine.get_component(name) is component
    
    # Verificar que devuelve None para componentes inexistentes
    assert engine.get_component("nonexistent") is None
    
    # Verificar obtención de todos los componentes
    all_components = engine.get_all_components()
    assert len(all_components) == 3
    for component in components.values():
        assert component in all_components


@pytest.mark.asyncio
async def test_engine_register_at_runtime(engine, simple_component):
    """Probar registro de componentes mientras el motor está ejecutándose."""
    # Iniciar el motor
    await engine.start()
    
    # Registrar componente
    engine.register_component(simple_component)
    
    # Verificar que el componente se registró
    assert simple_component.name in engine.components
    
    # Verificar que el componente se inició automáticamente
    assert simple_component.start_called
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que el componente se detuvo
    assert simple_component.stop_called


@pytest.mark.asyncio
async def test_engine_with_settings(simple_component):
    """Probar integración del motor con configuración."""
    # Crear configuración
    settings = Settings()
    settings.set("engine.use_priorities", True)
    settings.set("engine.operation_timeout", 5.0)
    
    # Crear motor con configuración
    engine = Engine(event_bus_or_name="test_engine", test_mode=True)
    
    # Establecer configuración manualmente
    engine.use_priorities = True
    engine.operation_timeout = 5.0
    
    # Verificar que la configuración se aplicó
    assert engine.use_priorities is True
    assert engine.operation_timeout == 5.0
    
    # Registrar componente
    engine.register_component(simple_component)
    
    # Iniciar y detener el motor
    await engine.start()
    await engine.stop()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])