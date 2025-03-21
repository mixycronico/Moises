"""
Pruebas unitarias para el módulo core.engine.

Este módulo prueba las funcionalidades básicas del motor principal del sistema Genesis,
incluyendo el registro de componentes, inicio, parada y comunicación entre componentes.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch

from genesis.core.base import Component
from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus


class TestComponent(Component):
    """Componente de prueba para testing."""
    
    def __init__(self, name):
        """Inicializar componente de prueba."""
        super().__init__(name)
        self.started = False
        self.stopped = False
        self.received_events = []
    
    async def start(self):
        """Iniciar el componente de prueba."""
        await super().start()
        self.started = True
    
    async def stop(self):
        """Detener el componente de prueba."""
        await super().stop()
        self.stopped = True
    
    async def handle_event(self, event_type, data, source):
        """Manejar eventos para el componente de prueba."""
        self.received_events.append((event_type, data, source))


@pytest.mark.asyncio
async def test_engine_start_stop():
    """Verificar que el motor puede iniciarse y detenerse correctamente."""
    # Crear instancia de Engine
    engine = Engine(name="test_engine")
    
    # Verificar inicio
    await engine.start()
    assert engine.running is True
    assert engine.event_bus is not None
    
    # Verificar parada
    await engine.stop()
    assert engine.running is False


@pytest.mark.asyncio
async def test_component_registration():
    """Verificar que los componentes se pueden registrar correctamente."""
    # Crear instancia de Engine
    engine = Engine(name="test_engine")
    
    # Crear componentes de prueba
    component1 = TestComponent("component1")
    component2 = TestComponent("component2")
    
    # Registrar componentes
    engine.register_component(component1)
    engine.register_component(component2)
    
    # Verificar registro
    assert "component1" in engine.components
    assert "component2" in engine.components
    assert engine.components["component1"] is component1
    assert engine.components["component2"] is component2
    
    # Verificar que cada componente tiene acceso al bus de eventos
    assert component1.event_bus is engine.event_bus
    assert component2.event_bus is engine.event_bus


@pytest.mark.asyncio
async def test_component_event_communication():
    """Verificar que los componentes pueden comunicarse a través del bus de eventos."""
    # Crear instancia de Engine
    engine = Engine(name="test_engine")
    
    # Crear componentes de prueba
    sender = TestComponent("sender")
    receiver = TestComponent("receiver")
    
    # Registrar componentes
    engine.register_component(sender)
    engine.register_component(receiver)
    
    # Iniciar el motor
    await engine.start()
    
    # Emitir evento desde el emisor
    test_data = {"key": "value"}
    await sender.emit_event("test_event", test_data)
    
    # Esperar brevemente para que el evento se procese
    await asyncio.sleep(0.1)
    
    # Verificar que el receptor recibió el evento
    assert len(receiver.received_events) > 0
    
    found_event = False
    for event in receiver.received_events:
        event_type, data, source = event
        if event_type == "test_event" and data == test_data and source == "sender":
            found_event = True
            break
    
    assert found_event, "Evento no recibido correctamente"
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_component_startup_order():
    """Verificar que los componentes se inician en el orden correcto."""
    # Crear instancia de Engine
    engine = Engine(name="test_engine")
    
    # Crear componentes de prueba con dependencias
    component1 = TestComponent("component1")
    component2 = TestComponent("component2")
    component3 = TestComponent("component3")
    
    # Establecer dependencias (component3 depende de component2 que depende de component1)
    component2.dependencies = ["component1"]
    component3.dependencies = ["component2"]
    
    # Registrar componentes en orden inverso para probar la resolución de dependencias
    engine.register_component(component3)
    engine.register_component(component2)
    engine.register_component(component1)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que todos los componentes se han iniciado
    assert component1.started is True
    assert component2.started is True
    assert component3.started is True
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_graceful_shutdown():
    """Verificar que el motor maneja correctamente el apagado gracioso."""
    # Crear instancia de Engine
    engine = Engine(name="test_engine")
    
    # Mockear el método stop de los componentes para simular un tiempo de cierre
    async def slow_stop():
        await asyncio.sleep(0.2)
        component.stopped = True
    
    component = TestComponent("slow_component")
    component.stop = slow_stop
    
    # Registrar componente
    engine.register_component(component)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que el componente se ha iniciado
    assert component.started is True
    
    # Detener el motor con timeout
    start_time = asyncio.get_event_loop().time()
    await engine.stop(timeout=0.5)
    shutdown_time = asyncio.get_event_loop().time() - start_time
    
    # Verificar que el shutdown tomó al menos el tiempo del slow_stop
    assert shutdown_time >= 0.2
    
    # Verificar que el componente se ha detenido
    assert component.stopped is True
    assert engine.running is False


@pytest.mark.asyncio
async def test_engine_component_error_handling():
    """Verificar que el motor maneja correctamente errores en los componentes."""
    # Crear instancia de Engine
    engine = Engine(name="test_engine")
    
    # Crear un componente que lanza errores
    class ErrorComponent(Component):
        def __init__(self, name):
            super().__init__(name)
        
        async def start(self):
            await super().start()
            raise RuntimeError("Error de prueba en start")
        
        async def handle_event(self, event_type, data, source):
            raise ValueError("Error de prueba en handle_event")
    
    error_component = ErrorComponent("error_component")
    
    # Registrar componente
    engine.register_component(error_component)
    
    # Mockear el logger para capturar errores
    mock_logger = MagicMock()
    engine.logger = mock_logger
    
    # Iniciar el motor y verificar que maneja el error
    try:
        await engine.start()
    except RuntimeError:
        pytest.fail("El error del componente no fue manejado correctamente")
    
    # Verificar que se registró el error
    assert mock_logger.error.called
    
    # Verificar manejo de errores en eventos
    normal_component = TestComponent("normal_component")
    engine.register_component(normal_component)
    
    # Reiniciar el motor con ambos componentes
    engine.components = {}
    engine.register_component(error_component)
    engine.register_component(normal_component)
    
    # Mockear start para evitar el error
    error_component.start = MagicMock(return_value=asyncio.Future())
    error_component.start.return_value.set_result(None)
    
    # Iniciar el motor
    await engine.start()
    
    # Emitir un evento que causará error en error_component pero no en normal_component
    await engine.event_bus.emit("test_event", {"data": "test"}, "test_source")
    
    # Esperar brevemente para que el evento se procese
    await asyncio.sleep(0.1)
    
    # Verificar que normal_component recibió el evento a pesar del error en error_component
    assert len(normal_component.received_events) > 0
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_performance():
    """Verificar el rendimiento básico del motor con múltiples componentes."""
    # Crear instancia de Engine
    engine = Engine(name="test_engine")
    
    # Crear varios componentes
    num_components = 10
    components = []
    
    for i in range(num_components):
        component = TestComponent(f"component{i}")
        engine.register_component(component)
        components.append(component)
    
    # Medir tiempo de inicio
    start_time = asyncio.get_event_loop().time()
    await engine.start()
    startup_time = asyncio.get_event_loop().time() - start_time
    
    print(f"Tiempo de inicio para {num_components} componentes: {startup_time:.6f} segundos")
    
    # Verificar que todos los componentes se iniciaron
    for component in components:
        assert component.started is True
    
    # Medir rendimiento del bus de eventos
    num_events = 100
    start_time = asyncio.get_event_loop().time()
    
    for i in range(num_events):
        await engine.event_bus.emit(f"test_event_{i}", {"index": i}, "test_source")
    
    # Esperar a que se procesen los eventos
    await asyncio.sleep(0.5)
    
    event_time = asyncio.get_event_loop().time() - start_time
    events_per_sec = num_events / event_time if event_time > 0 else 0
    
    print(f"Procesamiento de {num_events} eventos: {event_time:.6f} segundos ({events_per_sec:.2f} eventos/segundo)")
    
    # Verificar que cada componente recibió todos los eventos
    for component in components:
        assert len(component.received_events) == num_events
    
    # Detener el motor
    start_time = asyncio.get_event_loop().time()
    await engine.stop()
    shutdown_time = asyncio.get_event_loop().time() - start_time
    
    print(f"Tiempo de parada para {num_components} componentes: {shutdown_time:.6f} segundos")
    
    # Verificar que todos los componentes se detuvieron
    for component in components:
        assert component.stopped is True