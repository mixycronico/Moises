"""
Tests simplificados para el sistema de prioridad del Engine.

Este módulo contiene pruebas altamente simplificadas para verificar
la funcionalidad de prioridad de componentes en el motor.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock

from genesis.core.engine import Engine
from genesis.core.component import Component
from genesis.core.event_bus import EventBus


class SimpleComponent(Component):
    """Componente sencillo para pruebas."""
    
    def __init__(self, name):
        """Inicializar componente básico."""
        super().__init__(name)
        self.start_called = False
        self.stop_called = False
        
    async def start(self):
        """Iniciar componente con registro simple."""
        self.start_called = True
        
    async def stop(self):
        """Detener componente con registro simple."""
        self.stop_called = True
        
    async def handle_event(self, event_type, data, source):
        """Simplemente devolver el nombre del componente."""
        return {"component": self.name}


@pytest.mark.asyncio
async def test_priority_component_ordering():
    """Prueba básica de ordenamiento por prioridad."""
    # Crear motor de prueba con modo test para evitar tareas en background
    event_bus = EventBus(test_mode=True)
    engine = Engine(event_bus, test_mode=True)
    
    # Crear componentes con mocks para facilitar el seguimiento
    start_order = []
    
    class TrackingComponent(Component):
        def __init__(self, name):
            super().__init__(name)
            
        async def start(self):
            start_order.append(self.name)
            
        async def stop(self):
            pass
            
        async def handle_event(self, event_type, data, source):
            return {"component": self.name}
    
    # Crear componentes en orden inverso para asegurar que la prioridad es la que determina
    high = TrackingComponent("high")
    medium = TrackingComponent("medium")
    low = TrackingComponent("low")
    
    # Registrar componentes con diferentes prioridades
    engine.register_component(low, priority=10)
    engine.register_component(medium, priority=50)
    engine.register_component(high, priority=100)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar el orden de inicio
    assert len(start_order) == 3, f"Esperaba 3 componentes iniciados, obtuvo {len(start_order)}: {start_order}"
    assert start_order[0] == "high", f"Esperaba high primero, obtuvo {start_order}"
    assert start_order[1] == "medium", f"Esperaba medium segundo, obtuvo {start_order}"
    assert start_order[2] == "low", f"Esperaba low tercero, obtuvo {start_order}"
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_priority_direct_validation():
    """Prueba directa de la funcionalidad interna de prioridades."""
    # Crear motor de prueba
    event_bus = EventBus(test_mode=True)
    engine = Engine(event_bus, test_mode=True)
    
    # Crear componentes
    high = SimpleComponent("high")
    medium = SimpleComponent("medium")
    low = SimpleComponent("low")
    
    # Registrar componentes con diferentes prioridades
    engine.register_component(low, priority=10)
    engine.register_component(medium, priority=50)
    engine.register_component(high, priority=100)
    
    # Verificar que las prioridades se guardaron correctamente
    priorities = getattr(engine, 'operation_priorities', {})
    assert priorities["high"] == 100
    assert priorities["medium"] == 50
    assert priorities["low"] == 10
    
    # Verificar la ordenación directamente
    ordered_components = sorted(
        engine.components.items(),
        key=lambda x: priorities.get(x[0], 50),
        reverse=True  # Mayor prioridad primero
    )
    
    assert ordered_components[0][0] == "high"
    assert ordered_components[1][0] == "medium"
    assert ordered_components[2][0] == "low"