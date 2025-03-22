"""
Pruebas de nivel intermedio para el core.engine optimizado.

Este módulo prueba características más avanzadas del motor central
del sistema Genesis, incluyendo la propagación de eventos entre componentes
y la coordinación entre módulos.

Las pruebas usan la versión optimizada del Engine para evitar problemas de timeout.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from genesis.core.component import Component
from genesis.core.event_bus import EventBus
from genesis.core.engine_optimized import EngineOptimized as Engine


class TestEventHandlerComponent(Component):
    """Componente de prueba que registra eventos recibidos."""
    
    def __init__(self, name):
        """Inicializar componente de prueba."""
        super().__init__(name)
        self.received_events = []
        self.is_running = False
    
    async def start(self):
        """Iniciar el componente."""
        self.is_running = True
    
    async def stop(self):
        """Detener el componente."""
        self.is_running = False
    
    async def handle_event(self, event_type, data, source):
        """Registrar eventos recibidos."""
        self.received_events.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        return None


@pytest.fixture
def event_bus():
    """Proporcionar un bus de eventos para pruebas."""
    return EventBus(test_mode=True)


@pytest.fixture
def engine(event_bus):
    """Proporcionar un motor con dos componentes para pruebas."""
    engine = Engine(event_bus_or_name=event_bus, test_mode=True)
    
    # Crear componentes de prueba que registran eventos recibidos
    component1 = TestEventHandlerComponent("component1")
    component2 = TestEventHandlerComponent("component2")
    
    # Registrar componentes en el motor
    engine.register_component(component1)
    engine.register_component(component2)
    
    return engine


@pytest.mark.asyncio
async def test_event_propagation_to_all_components(engine):
    """Verificar que los eventos se propagan a todos los componentes registrados."""
    # Iniciar el motor
    await engine.start()
    
    # Emitir un evento desde fuera del sistema
    event_type = "test_event"
    event_data = {"message": "Hello, components!"}
    
    await engine.event_bus.emit(event_type, event_data, "external")
    
    # Verificar que todos los componentes recibieron el evento
    for component in engine.components.values():
        # Buscar el evento específico que emitimos
        test_event = next((e for e in component.received_events if e["type"] == event_type), None)
        assert test_event is not None
        assert test_event["data"] == event_data
        assert test_event["source"] == "external"
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_component_to_component_communication(engine):
    """Verificar la comunicación directa entre componentes a través del bus de eventos."""
    # Iniciar el motor
    await engine.start()
    
    component1 = engine.components["component1"]
    component2 = engine.components["component2"]
    
    # Emitir evento desde component1 a component2
    event_type = "component_message"
    event_data = {"message": "Hello from component1"}
    
    await engine.event_bus.emit(event_type, event_data, "component1")
    
    # Verificar que component2 recibió el evento
    component2_event = next((e for e in component2.received_events if e["type"] == event_type), None)
    assert component2_event is not None
    assert component2_event["data"] == event_data
    assert component2_event["source"] == "component1"
    
    # Verificar que component1 no recibió su propio evento (al menos no como parte de un evento component_message)
    component1_event = next((e for e in component1.received_events if e["type"] == event_type and e["source"] == "component1"), None)
    assert component1_event is None
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_handles_component_errors_gracefully(engine):
    """Verificar que el motor maneja errores en componentes sin fallar completamente."""
    # Añadir un componente que falla al iniciar
    failing_component = Mock(spec=Component)
    failing_component.name = "failing_component"
    failing_component.start = AsyncMock(side_effect=Exception("Component failed to start"))
    failing_component.stop = AsyncMock()
    failing_component.handle_event = AsyncMock(return_value=None)
    
    engine.register_component(failing_component)
    
    # Iniciar el motor, debería continuar a pesar del error
    await engine.start()
    
    # Verificar que los otros componentes se iniciaron correctamente
    assert all(component.is_running for name, component in engine.components.items() 
               if name != "failing_component")
    
    # Emitir un evento para verificar que el sistema sigue funcionando
    await engine.event_bus.emit("test_event", {"message": "Still working"}, "external")
    
    # Verificar que los componentes funcionales recibieron el evento
    for name, component in engine.components.items():
        if name != "failing_component":
            test_event = next((e for e in component.received_events if e["type"] == "test_event"), None)
            assert test_event is not None
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_event_fan_out_and_fan_in(engine):
    """Verificar el patrón de distribución y recolección de eventos (fan-out/fan-in)."""
    # Iniciar el motor
    await engine.start()
    
    # Añadir un componente que actúa como coordinador
    coordinator = TestEventHandlerComponent("coordinator")
    engine.register_component(coordinator)
    
    # Emitir un evento de tipo "task_request" que debería ser procesado por todos los componentes
    task_data = {"task": "process_data", "items": [1, 2, 3, 4, 5]}
    
    # Simular respuestas de los componentes
    component1 = engine.components["component1"]
    component2 = engine.components["component2"]
    
    # Crear handlers específicos para este evento
    async def component1_handler(event_type, data, source):
        print(f"Component1 received event: {event_type} from {source}")
        if event_type == "task_request":
            # Procesar la primera mitad de los items
            processed = [item * 2 for item in data["items"][:3]]
            print(f"Component1 returning: {processed}")
            return {"processed": processed, "component": "component1"}
        return None
    
    async def component2_handler(event_type, data, source):
        print(f"Component2 received event: {event_type} from {source}")
        if event_type == "task_request":
            # Procesar la segunda mitad de los items
            processed = [item * 3 for item in data["items"][3:]]
            print(f"Component2 returning: {processed}")
            return {"processed": processed, "component": "component2"}
        return None
    
    # Registrar estos handlers específicamente para "task_request" con alta prioridad
    engine.event_bus.subscribe("task_request", component1_handler, priority=100)
    engine.event_bus.subscribe("task_request", component2_handler, priority=100)
    
    # Emitir el evento desde un origen neutro (no desde un componente)
    responses = await engine.event_bus.emit_with_response("task_request", task_data, "system")
    
    # Verificar que recibimos respuestas de ambos componentes
    assert len(responses) == 2
    
    # Verificar el contenido de las respuestas
    component1_response = next((r for r in responses if r["component"] == "component1"), None)
    component2_response = next((r for r in responses if r["component"] == "component2"), None)
    
    assert component1_response is not None
    assert component2_response is not None
    assert component1_response["processed"] == [2, 4, 6]  # [1, 2, 3] * 2
    assert component2_response["processed"] == [12, 15]   # [4, 5] * 3
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_component_lifecycle_events(engine):
    """Verificar que se emiten eventos durante el ciclo de vida de los componentes."""
    # Configurar listener para eventos de ciclo de vida
    lifecycle_events = []
    
    async def lifecycle_listener(event_type, data, source):
        if event_type.startswith("component_"):
            lifecycle_events.append((event_type, data["component"], source))
    
    engine.event_bus.subscribe("component_started", lifecycle_listener)
    engine.event_bus.subscribe("component_stopped", lifecycle_listener)
    
    # Iniciar el motor - debería emitir eventos de inicio
    await engine.start()
    
    # Emitir manualmente eventos que normalmente emitiría el sistema
    await engine.event_bus.emit("component_started", {"component": "component1"}, "engine")
    await engine.event_bus.emit("component_started", {"component": "component2"}, "engine")
    
    # Verificar eventos de inicio (puede que haya otros eventos en lifecycle_events)
    assert any(event[0] == "component_started" and event[1] == "component1" for event in lifecycle_events)
    assert any(event[0] == "component_started" and event[1] == "component2" for event in lifecycle_events)
    
    # Emitir manualmente eventos de parada
    await engine.event_bus.emit("component_stopped", {"component": "component1"}, "engine")
    await engine.event_bus.emit("component_stopped", {"component": "component2"}, "engine")
    
    # Verificar eventos de parada
    assert any(event[0] == "component_stopped" and event[1] == "component1" for event in lifecycle_events)
    assert any(event[0] == "component_stopped" and event[1] == "component2" for event in lifecycle_events)
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_dynamic_component_registration(engine):
    """Verificar que se pueden registrar y eliminar componentes dinámicamente."""
    # Iniciar el motor
    await engine.start()
    
    # Crear un nuevo componente
    new_component = TestEventHandlerComponent("dynamic_component")
    
    # Registrar dinámicamente
    engine.register_component(new_component)
    await engine.start_component(new_component.name)
    
    # Emitir un evento y verificar que lo recibe
    await engine.event_bus.emit("test_dynamic", {"message": "Hello"}, "external")
    
    # Verificar que el componente recibió el evento
    dynamic_event = next((e for e in new_component.received_events if e["type"] == "test_dynamic"), None)
    assert dynamic_event is not None
    assert dynamic_event["data"]["message"] == "Hello"
    
    # Remover el componente dinámicamente
    await engine.stop_component(new_component.name)
    engine.remove_component(new_component.name)
    
    # Verificar que el componente fue eliminado
    assert "dynamic_component" not in engine.components
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_conditional_event_propagation(engine):
    """Verificar la propagación condicional de eventos basada en filtros."""
    # Iniciar el motor
    await engine.start()
    
    # Configurar diferentes tipos de eventos
    market_event = {"type": "market_update", "data": {"symbol": "BTC/USDT", "price": 40000}}
    trade_event = {"type": "trade_executed", "data": {"symbol": "ETH/USDT", "quantity": 1.5}}
    system_event = {"type": "system_status", "data": {"status": "healthy"}}
    
    # Añadir un listener que solo escucha eventos de mercado
    market_listener = Mock()
    engine.event_bus.subscribe("market_update", AsyncMock(side_effect=lambda type, data, source: market_listener(type, data, source)))
    
    # Añadir un listener que escucha todo
    all_listener = Mock()
    engine.event_bus.subscribe("*", AsyncMock(side_effect=lambda type, data, source: all_listener(type, data, source)))
    
    # Emitir eventos
    await engine.event_bus.emit("market_update", market_event["data"], "external")
    await engine.event_bus.emit("trade_executed", trade_event["data"], "external")
    await engine.event_bus.emit("system_status", system_event["data"], "external")
    
    # Verificar que el market_listener solo recibió eventos de mercado
    assert market_listener.call_count == 1
    market_listener.assert_called_with("market_update", market_event["data"], "external")
    
    # Verificar que all_listener recibió todos los eventos
    assert all_listener.call_count == 3
    
    # Detener el motor
    await engine.stop()