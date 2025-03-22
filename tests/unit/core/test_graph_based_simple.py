"""
Pruebas simplificadas para el motor basado en grafo de dependencias.

Este módulo contiene un conjunto mínimo de pruebas para verificar 
el funcionamiento básico del motor GraphBasedEngine.
"""

import pytest
import asyncio
from typing import Dict, Any, Optional

from genesis.core.component import Component
from genesis.core.component_graph_event_bus import CircularDependencyError
from genesis.core.engine_graph_based import GraphBasedEngine

# Componente de prueba simplificado
class SimpleComponent(Component):
    """Componente simple para pruebas básicas."""
    
    def __init__(self, component_id: str):
        """Inicializar componente con ID."""
        self.id = component_id
        self.events = []
        self.started = False
        self.stopped = False
        
    async def start(self):
        """Iniciar componente."""
        self.started = True
        
    async def stop(self):
        """Detener componente."""
        self.stopped = True
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """Manejar evento, simplemente registrándolo."""
        self.events.append((event_type, data, source))
        return None
        
@pytest.fixture
def engine():
    """Fixture para crear motor de prueba."""
    return GraphBasedEngine(test_mode=True)
    
async def test_engine_initialization():
    """Probar inicialización básica del motor."""
    engine = GraphBasedEngine(test_mode=True)
    assert not engine.running
    assert len(engine.components) == 0
    
async def test_component_registration():
    """Probar registro de componentes."""
    engine = GraphBasedEngine(test_mode=True)
    
    # Registrar componentes
    a = SimpleComponent("a")
    b = SimpleComponent("b")
    
    engine.register_component(a)
    engine.register_component(b)
    
    # Verificar registro
    assert len(engine.components) == 2
    assert "a" in engine.components
    assert "b" in engine.components
    
async def test_dependency_registration():
    """Probar registro con dependencias."""
    engine = GraphBasedEngine(test_mode=True)
    
    # Registrar componentes con dependencia
    a = SimpleComponent("a")
    b = SimpleComponent("b")
    
    engine.register_component(a)
    engine.register_component(b, ["a"])
    
    # Verificar dependencias
    assert "a" in engine.component_dependencies["b"]
    assert "b" in engine.component_dependents["a"]
    
async def test_circular_dependency_detection():
    """Probar detección de dependencias circulares."""
    engine = GraphBasedEngine(test_mode=True)
    
    # Registrar componentes que formarían un ciclo
    a = SimpleComponent("a")
    b = SimpleComponent("b")
    c = SimpleComponent("c")
    
    engine.register_component(a)
    engine.register_component(b, ["a"])
    
    # Esto debería provocar error al crear dependencia circular
    with pytest.raises(CircularDependencyError):
        engine.register_component(c, ["b"])
        engine.register_component(a, ["c"])  # Cierra el ciclo
        
async def test_basic_start_stop():
    """Probar inicio y detención básicos."""
    engine = GraphBasedEngine(test_mode=True)
    
    # Registrar componente
    a = SimpleComponent("a")
    engine.register_component(a)
    
    # Iniciar y verificar
    await engine.start()
    assert engine.running
    assert a.started
    
    # Detener y verificar
    await engine.stop()
    assert not engine.running
    assert a.stopped
    
async def test_simple_event_emission():
    """Probar emisión de eventos simple."""
    engine = GraphBasedEngine(test_mode=True)
    
    # Registrar componentes
    a = SimpleComponent("a")
    b = SimpleComponent("b")
    
    engine.register_component(a)
    engine.register_component(b)
    
    # Iniciar motor
    await engine.start()
    
    # Emitir evento simple
    data = {"message": "prueba"}
    await engine.emit("test_event", data, "test_source")
    
    # Dar tiempo para procesamiento
    await asyncio.sleep(0.1)
    
    # Verificar recepción
    assert len(a.events) == 1
    assert len(b.events) == 1
    assert a.events[0][0] == "test_event"
    assert a.events[0][1]["message"] == "prueba"
    assert a.events[0][2] == "test_source"
    
    # Detener motor
    await engine.stop()