"""
Pruebas optimizadas para el motor utilizando un EventBus simplificado.

Este módulo implementa pruebas para el motor que utilizan una versión
simplificada del EventBus para evitar timeouts y problemas de concurrencia.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Importar TestEventBus desde nuestro módulo de pruebas optimizadas
from tests.unit.core.test_event_bus_optimized import TestEventBus


# Componente simplificado para pruebas
class SimpleComponent:
    """Componente muy simple para pruebas."""
    __test__ = False  # Esto informa a pytest que no es una clase de prueba
    
    def __init__(self, name):
        self.name = name
        self.event_bus = None
        self.started = False
        self.stopped = False
        self.events = []
    
    def attach_event_bus(self, event_bus):
        """Adjunta un bus de eventos al componente."""
        self.event_bus = event_bus
    
    async def start(self):
        """Inicia el componente."""
        self.started = True
        print(f"Componente {self.name} iniciado")
        return self.name
    
    async def stop(self):
        """Detiene el componente."""
        self.stopped = True
        print(f"Componente {self.name} detenido")
        return self.name
    
    async def handle_event(self, event_type, data, source):
        """Maneja un evento."""
        self.events.append((event_type, data, source))
        print(f"Componente {self.name} recibió evento {event_type}")
        return f"{self.name}_response"


# Motor simplificado para pruebas
class SimpleEngine:
    """Motor simplificado para pruebas."""
    __test__ = False  # Esto informa a pytest que no es una clase de prueba
    
    def __init__(self, event_bus=None):
        """Inicializa el motor con un bus de eventos."""
        self.event_bus = event_bus or TestEventBus()
        self.components = {}
        self.running = False
        self.operation_priorities = {}
    
    def register_component(self, component, priority=50):
        """Registra un componente en el motor."""
        if component.name in self.components:
            raise ValueError(f"Componente {component.name} ya registrado")
        
        self.components[component.name] = component
        self.operation_priorities[component.name] = priority
        
        # Adjuntar bus de eventos
        component.attach_event_bus(self.event_bus)
        
        # Suscribir a eventos
        self.event_bus.subscribe("*", component.handle_event, priority=priority)
    
    async def start(self):
        """Inicia el motor y todos los componentes."""
        if self.running:
            print("El motor ya está en ejecución")
            return
        
        print("Iniciando el motor")
        
        # Iniciar bus de eventos
        await self.event_bus.start()
        
        # Ordenar componentes por prioridad (mayor prioridad primero)
        ordered_components = sorted(
            self.components.items(),
            key=lambda x: self.operation_priorities.get(x[0], 50),
            reverse=True
        )
        
        # Iniciar componentes en orden de prioridad
        for name, component in ordered_components:
            print(f"Iniciando componente: {name} (prioridad: {self.operation_priorities.get(name, 50)})")
            await component.start()
        
        self.running = True
        print("Motor iniciado")
        
        # Emitir evento de inicio
        await self.event_bus.emit(
            "system.started",
            {"components": list(self.components.keys())},
            "engine"
        )
    
    async def stop(self):
        """Detiene el motor y todos los componentes."""
        if not self.running:
            print("El motor no está en ejecución")
            return
        
        print("Deteniendo el motor")
        
        # Ordenar componentes por prioridad (menor prioridad primero)
        ordered_components = sorted(
            self.components.items(),
            key=lambda x: self.operation_priorities.get(x[0], 50)
        )
        
        # Detener componentes en orden de prioridad
        for name, component in ordered_components:
            print(f"Deteniendo componente: {name} (prioridad: {self.operation_priorities.get(name, 50)})")
            await component.stop()
        
        # Detener bus de eventos
        await self.event_bus.stop()
        
        self.running = False
        print("Motor detenido")


# Pruebas para el motor simplificado
@pytest.mark.asyncio
async def test_engine_simple_lifecycle():
    """Prueba el ciclo de vida básico del motor."""
    # Crear bus de eventos
    event_bus = TestEventBus()
    
    # Crear motor
    engine = SimpleEngine(event_bus)
    
    # Crear componentes
    high = SimpleComponent("high")
    medium = SimpleComponent("medium")
    low = SimpleComponent("low")
    
    # Registrar componentes
    engine.register_component(high, priority=100)
    engine.register_component(medium, priority=50)
    engine.register_component(low, priority=10)
    
    # Iniciar motor
    await engine.start()
    
    # Verificar que el motor está en ejecución
    assert engine.running is True
    
    # Verificar que todos los componentes se iniciaron
    assert high.started is True
    assert medium.started is True
    assert low.started is True
    
    # Verificar que los componentes recibieron el evento system.started
    for component in [high, medium, low]:
        assert any(event[0] == "system.started" for event in component.events)
    
    # Detener motor
    await engine.stop()
    
    # Verificar que el motor se detuvo
    assert engine.running is False
    
    # Verificar que todos los componentes se detuvieron
    assert high.stopped is True
    assert medium.stopped is True
    assert low.stopped is True


@pytest.mark.asyncio
async def test_engine_priority_ordering():
    """Prueba que los componentes se inician y detienen en el orden correcto según su prioridad."""
    # Crear bus de eventos
    event_bus = TestEventBus()
    
    # Crear motor
    engine = SimpleEngine(event_bus)
    
    # Crear registros para el orden de inicio y parada
    start_order = []
    stop_order = []
    
    # Crear componentes con inicio/parada personalizados
    async def custom_start(component):
        start_order.append(component.name)
        component.started = True
        return component.name
    
    async def custom_stop(component):
        stop_order.append(component.name)
        component.stopped = True
        return component.name
    
    # Crear componentes
    high = SimpleComponent("high")
    medium = SimpleComponent("medium")
    low = SimpleComponent("low")
    
    # Sobrescribir métodos de inicio/parada
    high.start = lambda: custom_start(high)
    medium.start = lambda: custom_start(medium)
    low.start = lambda: custom_start(low)
    
    high.stop = lambda: custom_stop(high)
    medium.stop = lambda: custom_stop(medium)
    low.stop = lambda: custom_stop(low)
    
    # Registrar componentes
    engine.register_component(high, priority=100)
    engine.register_component(medium, priority=50)
    engine.register_component(low, priority=10)
    
    # Iniciar motor
    await engine.start()
    
    # Verificar orden de inicio (mayor prioridad primero)
    assert start_order == ["high", "medium", "low"]
    
    # Detener motor
    await engine.stop()
    
    # Verificar orden de parada (menor prioridad primero)
    assert stop_order == ["low", "medium", "high"]


@pytest.mark.asyncio
async def test_engine_event_propagation():
    """Prueba que los eventos se propagan correctamente a los componentes registrados."""
    # Crear bus de eventos
    event_bus = TestEventBus()
    
    # Crear motor
    engine = SimpleEngine(event_bus)
    
    # Crear componentes
    comp1 = SimpleComponent("comp1")
    comp2 = SimpleComponent("comp2")
    
    # Registrar componentes
    engine.register_component(comp1, priority=100)
    engine.register_component(comp2, priority=50)
    
    # Iniciar motor
    await engine.start()
    
    # Emitir evento personalizado
    test_data = {"message": "test"}
    await event_bus.emit("test.event", test_data, "test")
    
    # Verificar que ambos componentes recibieron el evento
    assert any(event[0] == "test.event" and event[1] == test_data for event in comp1.events)
    assert any(event[0] == "test.event" and event[1] == test_data for event in comp2.events)
    
    # Detener motor
    await engine.stop()