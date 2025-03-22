"""
Prueba minimalista para verificar el manejo de errores del motor.

Esta versión simplificada se enfoca exclusivamente en probar que el motor 
pueda manejar errores durante el procesamiento de eventos sin bloquear
el sistema.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock

from genesis.core.event_bus_minimal import EventBusMinimal
from genesis.core.component import Component
from genesis.core.engine_optimized import EngineOptimized

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
        self.started = False
        self.stopped = False
    
    async def start(self):
        """Iniciar el componente."""
        self.started = True
    
    async def stop(self):
        """Detener el componente."""
        self.stopped = True
    
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
        if self.should_fail and event_type == "test_event":
            raise Exception(f"Error simulado en el manejador de {self.name}")
        
        return None

@pytest.fixture
def event_bus():
    """Proporcionar un EventBusMinimal para pruebas."""
    return EventBusMinimal()

@pytest.fixture
def engine(event_bus):
    """Proporcionar un motor con EventBusMinimal para pruebas."""
    return EngineOptimized(event_bus_or_name=event_bus, test_mode=True)

@pytest.mark.asyncio
async def test_engine_error_handling_minimal(engine, event_bus):
    """
    Probar que el motor maneje correctamente errores durante el procesamiento de eventos.
    
    Esta versión minimalista verifica solo la funcionalidad básica para evitar timeouts.
    """
    # Crear componentes
    component1 = TestComponent("comp1")
    component2 = TestComponent("comp2", should_fail=True)  # Este componente fallará
    component3 = TestComponent("comp3")
    
    # Registrar componentes
    engine.register_component(component1)
    engine.register_component(component2)
    engine.register_component(component3)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que todos los componentes se iniciaron
    assert component1.started
    assert component2.started
    assert component3.started
    
    # Emitir un evento (usando el bus directamente para tener más control)
    await event_bus.emit("test_event", {"message": "Test"}, "test_source")
    
    # Verificar que los componentes sin error recibieron el evento
    assert len(component1.events_received) == 1
    assert component1.events_received[0]["type"] == "test_event"
    
    # Verificar que el componente que falla recibió el evento (aunque haya fallado al procesarlo)
    assert len(component2.events_received) == 1
    assert component2.events_received[0]["type"] == "test_event"
    
    # Verificar que el componente después del que falla también recibió el evento
    assert len(component3.events_received) == 1
    assert component3.events_received[0]["type"] == "test_event"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que todos los componentes se detuvieron
    assert component1.stopped
    assert component2.stopped
    assert component3.stopped