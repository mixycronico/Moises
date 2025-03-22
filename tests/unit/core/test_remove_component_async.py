"""
Test mínimo para verificar que los métodos de eliminación de componentes funcionan correctamente.

Este test simple verifica que remove_component, deregister_component y unregister_component
son métodos asíncronos y funcionan correctamente.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, Optional, List

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestComponent(Component):
    """Componente simple para pruebas."""
    
    def __init__(self, name):
        super().__init__(name)
        self.started = False
        self.stopped = False
        self.events_received = []
        
    async def start(self):
        self.started = True
        
    async def stop(self):
        self.stopped = True
        
    async def handle_event(self, event_type, data, source):
        self.events_received.append((event_type, data, source))
        return {"status": "ok", "from": self.name}


@pytest.mark.asyncio
async def test_remove_component_async():
    """Verificar que remove_component es asíncrono y funciona correctamente."""
    # Crear motor y componentes
    engine = EngineNonBlocking(test_mode=True)
    component1 = TestComponent("component1")
    component2 = TestComponent("component2")
    
    # Registrar componentes
    engine.register_component(component1)
    engine.register_component(component2)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que los componentes están registrados
    assert "component1" in engine.components
    assert "component2" in engine.components
    
    # Eliminar component1 con remove_component (debe ser asíncrono)
    await engine.remove_component("component1")
    
    # Verificar que component1 fue eliminado
    assert "component1" not in engine.components
    assert "component2" in engine.components
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_unregister_component_async():
    """Verificar que unregister_component es asíncrono y funciona correctamente."""
    # Crear motor y componentes
    engine = EngineNonBlocking(test_mode=True)
    component1 = TestComponent("component1")
    component2 = TestComponent("component2")
    
    # Registrar componentes
    engine.register_component(component1)
    engine.register_component(component2)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que los componentes están registrados
    assert "component1" in engine.components
    assert "component2" in engine.components
    
    # Eliminar component1 con unregister_component
    await engine.unregister_component("component1")
    
    # Verificar que component1 fue eliminado
    assert "component1" not in engine.components
    assert "component2" in engine.components
    
    # Detener el motor
    await engine.stop()


@pytest.mark.asyncio
async def test_deregister_component_async():
    """Verificar que deregister_component es asíncrono y funciona correctamente."""
    # Crear motor y componentes
    engine = EngineNonBlocking(test_mode=True)
    component1 = TestComponent("component1")
    component2 = TestComponent("component2")
    
    # Registrar componentes
    engine.register_component(component1)
    engine.register_component(component2)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que los componentes están registrados
    assert "component1" in engine.components
    assert "component2" in engine.components
    
    # Eliminar component1 con deregister_component
    await engine.deregister_component(component1)
    
    # Verificar que component1 fue eliminado
    assert "component1" not in engine.components
    assert "component2" in engine.components
    
    # Detener el motor
    await engine.stop()