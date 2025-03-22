"""
Pruebas para el EventBusMinimal.

Este módulo prueba la versión minimalista del EventBus diseñada para evitar
los problemas de timeout en las pruebas.
"""

import pytest
import asyncio
import logging

from genesis.core.event_bus_minimal import EventBusMinimal
from genesis.core.component import Component

# Silenciar logs durante las pruebas
logging.basicConfig(level=logging.CRITICAL)

class TestComponent(Component):
    """Componente minimalista para pruebas."""
    
    def __init__(self, name, should_fail=False):
        """
        Inicializar componente de prueba.
        
        Args:
            name: Nombre del componente
            should_fail: Si True, el manejador de eventos lanzará una excepción
        """
        super().__init__(name)
        self.events_received = []
        self.should_fail = should_fail
    
    async def start(self):
        """Iniciar el componente."""
        pass
    
    async def stop(self):
        """Detener el componente."""
        pass
    
    async def handle_event(self, event_type, data, source):
        """
        Manejar un evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        # Guardar evento recibido
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        
        # Simular un error si está configurado para fallar
        if self.should_fail:
            raise Exception(f"Error simulado en el manejador de {self.name}")

@pytest.mark.asyncio
async def test_event_bus_minimal_basic():
    """Probar funcionalidad básica del EventBusMinimal."""
    # Crear bus de eventos
    bus = EventBusMinimal()
    
    # Crear componentes
    component1 = TestComponent("comp1")
    component2 = TestComponent("comp2")
    
    # Suscribir componentes
    bus.subscribe("test_event", component1.handle_event)
    bus.subscribe("test_event", component2.handle_event)
    
    # Iniciar bus
    await bus.start()
    
    # Emitir un evento
    test_data = {"message": "Hola mundo", "value": 42}
    await bus.emit("test_event", test_data, "test_source")
    
    # Verificar que los componentes recibieron el evento
    assert len(component1.events_received) == 1
    assert component1.events_received[0]["type"] == "test_event"
    assert component1.events_received[0]["data"] == test_data
    assert component1.events_received[0]["source"] == "test_source"
    
    assert len(component2.events_received) == 1
    assert component2.events_received[0]["type"] == "test_event"
    assert component2.events_received[0]["data"] == test_data
    assert component2.events_received[0]["source"] == "test_source"
    
    # Detener bus
    await bus.stop()

@pytest.mark.asyncio
async def test_event_bus_minimal_error_handling():
    """Probar manejo de errores en el EventBusMinimal."""
    # Crear bus de eventos
    bus = EventBusMinimal()
    
    # Crear componentes (uno que falla)
    component1 = TestComponent("comp1")
    component2 = TestComponent("comp2", should_fail=True)  # Este componente fallará
    component3 = TestComponent("comp3")
    
    # Suscribir componentes
    bus.subscribe("test_event", component1.handle_event)
    bus.subscribe("test_event", component2.handle_event)
    bus.subscribe("test_event", component3.handle_event)
    
    # Iniciar bus
    await bus.start()
    
    # Emitir un evento
    test_data = {"message": "Test de error", "value": 123}
    await bus.emit("test_event", test_data, "test_source")
    
    # Verificar que todos los componentes recibieron el evento
    # incluso después de que uno falló
    assert len(component1.events_received) == 1
    assert len(component2.events_received) == 1  # Recibió evento pero falló al procesarlo
    assert len(component3.events_received) == 1  # Debería recibir el evento a pesar del fallo en component2
    
    # Verificar contenido del evento
    assert component1.events_received[0]["data"] == test_data
    assert component2.events_received[0]["data"] == test_data
    assert component3.events_received[0]["data"] == test_data
    
    # Detener bus
    await bus.stop()

if __name__ == "__main__":
    asyncio.run(test_event_bus_minimal_basic())
    asyncio.run(test_event_bus_minimal_error_handling())