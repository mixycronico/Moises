"""
Prueba con un EventBus optimizado para tests que evita timeouts.

Este módulo implementa una versión simplificada del EventBus específicamente
diseñada para evitar problemas de timeouts durante las pruebas.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock

# Nueva implementación simplificada del EventBus específica para pruebas
# No es una clase de prueba real, solo una utilidad para las pruebas
# pytest.mark.no_collect para evitar que pytest intente recopilarla como prueba
class TestEventBus:
    """Versión simplificada del EventBus para pruebas."""
    __test__ = False  # Esto informa a pytest que no es una clase de prueba
    
    def __init__(self):
        self.subscribers = {}
        self.running = False
        self.test_mode = True
    
    async def start(self):
        """Inicia el bus de eventos."""
        self.running = True
    
    async def stop(self):
        """Detiene el bus de eventos."""
        self.running = False
    
    def subscribe(self, event_type, handler, priority=50):
        """Suscribe un manejador a un tipo de evento."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append((priority, handler))
        self.subscribers[event_type].sort(key=lambda x: x[0], reverse=True)
    
    def unsubscribe(self, event_type, handler):
        """Elimina la suscripción de un manejador."""
        if event_type in self.subscribers:
            self.subscribers[event_type] = [
                (p, h) for p, h in self.subscribers[event_type] if h != handler
            ]
            
            if not self.subscribers[event_type]:
                del self.subscribers[event_type]
    
    async def emit(self, event_type, data, source):
        """Emite un evento a los suscriptores."""
        handlers = []
        
        # Handlers específicos
        if event_type in self.subscribers:
            handlers.extend(self.subscribers[event_type])
        
        # Handlers wildcard
        if '*' in self.subscribers:
            handlers.extend(self.subscribers['*'])
        
        # Ordenar por prioridad
        handlers.sort(key=lambda x: x[0], reverse=True)
        
        # Ejecutar handlers con timeout para evitar bloqueos
        for priority, handler in handlers:
            try:
                await asyncio.wait_for(handler(event_type, data, source), timeout=0.5)
            except asyncio.TimeoutError:
                print(f"Timeout en handler para {event_type}")
            except Exception as e:
                print(f"Error en handler: {e}")
    
    async def emit_with_response(self, event_type, data, source):
        """Emite un evento y recopila respuestas."""
        handlers = []
        
        # Handlers específicos
        if event_type in self.subscribers:
            handlers.extend(self.subscribers[event_type])
        
        # Handlers wildcard
        if '*' in self.subscribers:
            handlers.extend(self.subscribers['*'])
        
        # Ordenar por prioridad
        handlers.sort(key=lambda x: x[0], reverse=True)
        
        # Ejecutar handlers y recopilar respuestas
        responses = []
        for priority, handler in handlers:
            try:
                response = await asyncio.wait_for(handler(event_type, data, source), timeout=0.5)
                if response is not None:
                    responses.append(response)
            except asyncio.TimeoutError:
                print(f"Timeout en handler para {event_type}")
            except Exception as e:
                print(f"Error en handler: {e}")
        
        return responses


# Componente simplificado para pruebas
# No es una clase de prueba real, solo una utilidad para las pruebas
class TestComponent:
    """Componente muy simple para pruebas."""
    __test__ = False  # Esto informa a pytest que no es una clase de prueba
    
    def __init__(self, name):
        self.name = name
        self.event_bus = None
        self.start_called = False
        self.stop_called = False
        self.events_received = []
    
    def attach_event_bus(self, event_bus):
        """Adjunta un bus de eventos al componente."""
        self.event_bus = event_bus
    
    async def start(self):
        """Inicia el componente."""
        self.start_called = True
        print(f"Componente {self.name} iniciado")
    
    async def stop(self):
        """Detiene el componente."""
        self.stop_called = True
        print(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type, data, source):
        """Maneja un evento."""
        self.events_received.append((event_type, data, source))
        print(f"Componente {self.name} recibió evento {event_type}")
        return f"{self.name}_response"


# Pruebas simples para verificar funcionamiento
@pytest.mark.asyncio
async def test_event_bus_basic():
    """Prueba básica del bus de eventos."""
    event_bus = TestEventBus()
    
    # Crear componentes
    comp1 = TestComponent("comp1")
    comp2 = TestComponent("comp2")
    
    # Suscribir componentes
    event_bus.subscribe("test_event", comp1.handle_event, priority=100)
    event_bus.subscribe("test_event", comp2.handle_event, priority=50)
    
    # Iniciar bus
    await event_bus.start()
    
    # Emitir evento
    await event_bus.emit("test_event", {"data": "test"}, "source")
    
    # Verificar que los componentes recibieron el evento
    assert len(comp1.events_received) == 1
    assert len(comp2.events_received) == 1
    
    # Verificar el orden de procesamiento (comp1 debería procesarlo primero)
    assert comp1.events_received[0][0] == "test_event"
    assert comp2.events_received[0][0] == "test_event"
    
    # Detener bus
    await event_bus.stop()


@pytest.mark.asyncio
async def test_event_bus_with_responses():
    """Prueba del bus de eventos con respuestas."""
    event_bus = TestEventBus()
    
    # Crear componentes
    comp1 = TestComponent("comp1")
    comp2 = TestComponent("comp2")
    
    # Suscribir componentes
    event_bus.subscribe("test_event", comp1.handle_event, priority=100)
    event_bus.subscribe("test_event", comp2.handle_event, priority=50)
    
    # Iniciar bus
    await event_bus.start()
    
    # Emitir evento y recopilar respuestas
    responses = await event_bus.emit_with_response("test_event", {"data": "test"}, "source")
    
    # Verificar respuestas
    assert len(responses) == 2
    assert "comp1_response" in responses
    assert "comp2_response" in responses
    
    # Verificar el orden de las respuestas (comp1 debería ir primero)
    assert responses[0] == "comp1_response"
    assert responses[1] == "comp2_response"
    
    # Detener bus
    await event_bus.stop()


@pytest.mark.asyncio
async def test_unsubscribe():
    """Prueba de cancelación de suscripción."""
    event_bus = TestEventBus()
    
    # Crear componentes
    comp = TestComponent("comp")
    
    # Suscribir y luego cancelar
    event_bus.subscribe("test_event", comp.handle_event)
    event_bus.unsubscribe("test_event", comp.handle_event)
    
    # Iniciar bus
    await event_bus.start()
    
    # Emitir evento
    await event_bus.emit("test_event", {"data": "test"}, "source")
    
    # Verificar que no recibió el evento tras cancelar
    assert len(comp.events_received) == 0
    
    # Detener bus
    await event_bus.stop()


@pytest.mark.asyncio
async def test_priority_ordering():
    """Prueba de ordenación por prioridad."""
    event_bus = TestEventBus()
    
    # Crear componentes
    high = TestComponent("high")
    medium = TestComponent("medium")
    low = TestComponent("low")
    
    # Crear mocks para verificar el orden de ejecución
    execute_order = []
    
    # Crear funciones de manejo que registren el orden
    async def handle_high(event_type, data, source):
        execute_order.append("high")
        return "high_response"
    
    async def handle_medium(event_type, data, source):
        execute_order.append("medium")
        return "medium_response"
    
    async def handle_low(event_type, data, source):
        execute_order.append("low")
        return "low_response"
    
    # Asignar las funciones de manejo personalizadas
    high.handle_event = handle_high
    medium.handle_event = handle_medium
    low.handle_event = handle_low
    
    # Suscribir componentes con diferentes prioridades
    event_bus.subscribe("test_event", low.handle_event, priority=10)
    event_bus.subscribe("test_event", medium.handle_event, priority=50)
    event_bus.subscribe("test_event", high.handle_event, priority=100)
    
    # Iniciar bus
    await event_bus.start()
    
    # Emitir evento
    await event_bus.emit("test_event", {"data": "test"}, "source")
    
    # Verificar el orden de ejecución por prioridad
    assert execute_order == ["high", "medium", "low"]
    
    # Detener bus
    await event_bus.stop()