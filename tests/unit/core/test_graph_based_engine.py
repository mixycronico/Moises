"""
Pruebas unitarias para el motor basado en grafo de dependencias.

Este módulo prueba el funcionamiento básico del motor GraphBasedEngine,
que utiliza un grafo dirigido acíclico para gestionar las dependencias
entre componentes y prevenir deadlocks.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List, Optional

from genesis.core.component import Component
from genesis.core.component_graph_event_bus import CircularDependencyError
from genesis.core.engine_graph_based import GraphBasedEngine

# Clase para componentes de prueba
class TestComponent(Component):
    """Componente simple para pruebas."""
    
    def __init__(self, component_id: str):
        """
        Inicializar componente de prueba.
        
        Args:
            component_id: ID único del componente
        """
        self.id = component_id
        self.events_received = []
        self.started = False
        self.stopped = False
        self.fail_on_events = []
        self.slow_events = []
        self.response_events = {}  # event_type -> data a responder
        
    async def start(self) -> None:
        """Iniciar el componente."""
        self.started = True
        
    async def stop(self) -> None:
        """Detener el componente."""
        self.stopped = True
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar un evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Datos de respuesta o None
            
        Raises:
            Exception: Si el evento está en la lista de eventos que fallan
        """
        # Registrar evento
        self.events_received.append((event_type, data, source))
        
        # Simular fallo si corresponde
        if event_type in self.fail_on_events:
            raise Exception(f"Fallo simulado para evento {event_type}")
            
        # Simular lentitud si corresponde
        if event_type in self.slow_events:
            await asyncio.sleep(0.3)
            
        # Responder si es un evento con respuesta
        if event_type in self.response_events:
            return self.response_events[event_type]
            
        return None
        
# Tests para el motor basado en grafo

@pytest.fixture
def engine():
    """Fixture para crear una instancia limpia del motor."""
    return GraphBasedEngine(test_mode=True)
    
@pytest.fixture
def components():
    """Fixture para crear componentes de prueba."""
    return {
        "a": TestComponent("a"),
        "b": TestComponent("b"),
        "c": TestComponent("c"),
        "d": TestComponent("d")
    }
    
async def test_engine_basic_initialization(engine):
    """Probar inicialización básica del motor."""
    assert not engine.running
    assert engine.event_bus is not None
    assert len(engine.components) == 0
    assert len(engine.component_dependencies) == 0
    assert len(engine.component_dependents) == 0
    
async def test_engine_component_registration(engine, components):
    """Probar registro de componentes sin dependencias."""
    # Registrar componentes
    engine.register_component(components["a"])
    engine.register_component(components["b"])
    
    # Verificar registro
    assert len(engine.components) == 2
    assert "a" in engine.components
    assert "b" in engine.components
    assert engine.components["a"] == components["a"]
    assert engine.components["b"] == components["b"]
    
    # Verificar estado inicial
    assert engine.component_health["a"]
    assert engine.component_health["b"]
    assert engine.component_error_count["a"] == 0
    assert engine.component_error_count["b"] == 0
    
async def test_engine_component_with_dependencies(engine, components):
    """Probar registro de componentes con dependencias."""
    # Registrar componentes con dependencias
    engine.register_component(components["a"])
    engine.register_component(components["b"], ["a"])
    engine.register_component(components["c"], ["a", "b"])
    
    # Verificar dependencias
    assert "a" in engine.component_dependencies["b"]
    assert "a" in engine.component_dependencies["c"]
    assert "b" in engine.component_dependencies["c"]
    
    # Verificar dependencias inversas
    assert "b" in engine.component_dependents["a"]
    assert "c" in engine.component_dependents["a"]
    assert "c" in engine.component_dependents["b"]
    
async def test_engine_circular_dependency_detection(engine, components):
    """Probar detección de dependencias circulares."""
    # Registrar componentes que formarían un ciclo
    engine.register_component(components["a"])
    engine.register_component(components["b"], ["a"])
    
    # Intentar crear ciclo
    with pytest.raises(CircularDependencyError):
        engine.register_component(components["c"], ["b"])
        engine.register_component(components["a"], ["c"])
        
async def test_engine_start_stop(engine, components):
    """Probar inicio y detención de componentes."""
    # Registrar componentes
    engine.register_component(components["a"])
    engine.register_component(components["b"], ["a"])
    
    # Iniciar motor
    await engine.start()
    
    # Verificar que los componentes se iniciaron
    assert engine.running
    assert components["a"].started
    assert components["b"].started
    
    # Detener motor
    await engine.stop()
    
    # Verificar que los componentes se detuvieron
    assert not engine.running
    assert components["a"].stopped
    assert components["b"].stopped
    
async def test_engine_event_basic(engine, components):
    """Probar emisión básica de eventos."""
    # Registrar componentes
    engine.register_component(components["a"])
    engine.register_component(components["b"])
    
    # Iniciar motor
    await engine.start()
    
    # Emitir evento
    await engine.emit("test_event", {"value": 42}, "test_source")
    
    # Esperar propagación
    await asyncio.sleep(0.1)
    
    # Verificar recepción
    assert len(components["a"].events_received) == 1
    assert len(components["b"].events_received) == 1
    
    assert components["a"].events_received[0][0] == "test_event"
    assert components["a"].events_received[0][1]["value"] == 42
    assert components["a"].events_received[0][2] == "test_source"
    
    # Detener motor
    await engine.stop()
    
async def test_engine_event_respects_source(engine, components):
    """Probar que los eventos no se devuelven a la fuente."""
    # Registrar componentes
    engine.register_component(components["a"])
    engine.register_component(components["b"])
    
    # Iniciar motor
    await engine.start()
    
    # Emitir evento desde componente a
    await engine.emit("test_event", {"value": 42}, "a")
    
    # Esperar propagación
    await asyncio.sleep(0.1)
    
    # Verificar que a no recibe, pero b sí
    assert len(components["a"].events_received) == 0
    assert len(components["b"].events_received) == 1
    
    # Detener motor
    await engine.stop()
    
async def test_engine_topological_order(engine, components):
    """Probar que los eventos siguen el orden topológico."""
    # Registrar componentes con dependencias
    engine.register_component(components["a"])
    engine.register_component(components["b"], ["a"])
    engine.register_component(components["c"], ["b"])
    
    # Iniciar motor
    await engine.start()
    
    # Simular eventos lentos en a y b
    components["a"].slow_events = ["order_test"]
    components["b"].slow_events = ["order_test"]
    
    # Verificar componentes en orden topológico
    topo_order = engine.event_bus.topological_order
    assert topo_order.index("a") < topo_order.index("b")
    assert topo_order.index("b") < topo_order.index("c")
    
    # Emitir evento
    await engine.emit("order_test", {"timestamp": time.time()}, "test")
    
    # Esperar propagación
    await asyncio.sleep(0.7)
    
    # Verificar recepción
    assert len(components["a"].events_received) == 1
    assert len(components["b"].events_received) == 1
    assert len(components["c"].events_received) == 1
    
    # Detener motor
    await engine.stop()
    
async def test_engine_component_failure_isolation(engine, components):
    """Probar aislamiento de fallos en componentes."""
    # Registrar componentes
    engine.register_component(components["a"])
    engine.register_component(components["b"])
    
    # Configurar fallo en un componente
    components["a"].fail_on_events = ["fail_test"]
    
    # Iniciar motor
    await engine.start()
    
    # Emitir evento que fallará en a
    await engine.emit("fail_test", {"value": 42}, "test")
    
    # Esperar procesamiento
    await asyncio.sleep(0.1)
    
    # Verificar que el evento llegó a ambos
    assert len(components["a"].events_received) == 1
    assert len(components["b"].events_received) == 1
    
    # Emitir otro evento normal
    await engine.emit("normal_event", {"value": 100}, "test")
    
    # Esperar procesamiento
    await asyncio.sleep(0.1)
    
    # Verificar que sigue llegando a ambos (a no fue aislado)
    assert len(components["a"].events_received) == 2
    assert len(components["b"].events_received) == 2
    
    # Detener motor
    await engine.stop()
    
async def test_engine_emit_with_response(engine, components):
    """Probar emisión de eventos con respuesta."""
    # Configurar respuestas
    components["a"].response_events = {
        "query_event": {"result": "response_from_a"}
    }
    components["b"].response_events = {
        "query_event": {"result": "response_from_b"}
    }
    
    # Registrar componentes
    engine.register_component(components["a"])
    engine.register_component(components["b"])
    
    # Iniciar motor
    await engine.start()
    
    # Emitir evento con respuesta
    responses = await engine.emit_with_response("query_event", {}, "test", timeout=0.5)
    
    # Verificar respuestas
    assert len(responses) == 2
    
    # Extraer resultados
    results = [r.get("response", {}).get("result") for r in responses]
    assert "response_from_a" in results
    assert "response_from_b" in results
    
    # Detener motor
    await engine.stop()
    
async def test_engine_restart_component(engine, components):
    """Probar reinicio de componentes."""
    # Registrar componentes
    engine.register_component(components["a"])
    
    # Iniciar motor
    await engine.start()
    
    # Verificar estado inicial
    assert components["a"].started
    assert not components["a"].stopped
    
    # Reiniciar componente
    success = await engine.restart_component("a")
    
    # Verificar reinicio
    assert success
    assert components["a"].stopped  # Se llamó a stop
    assert components["a"].started  # Se llamó a start nuevamente
    
    # Detener motor
    await engine.stop()
    
async def test_engine_component_graph(engine, components):
    """Probar generación del grafo de componentes."""
    # Registrar componentes con dependencias
    engine.register_component(components["a"])
    engine.register_component(components["b"], ["a"])
    engine.register_component(components["c"], ["b"])
    
    # Generar grafo
    graph = engine.generate_component_graph()
    
    # Verificar nodos
    assert len(graph["nodes"]) == 3
    node_ids = [node["id"] for node in graph["nodes"]]
    assert "a" in node_ids
    assert "b" in node_ids
    assert "c" in node_ids
    
    # Verificar aristas
    assert len(graph["edges"]) == 2
    
    # Buscar aristas específicas
    edge_pairs = [(edge["source"], edge["target"]) for edge in graph["edges"]]
    assert ("a", "b") in edge_pairs
    assert ("b", "c") in edge_pairs
    
async def test_engine_error_history(engine, components):
    """Probar registro de historial de errores."""
    # Registrar componente problemático
    engine.register_component(components["a"])
    
    # Configurar fallo
    components["a"].fail_on_events = ["fail_event"]
    
    # Iniciar motor
    await engine.start()
    
    # Emitir evento que fallará
    await engine.emit("fail_event", {}, "test")
    
    # Esperar procesamiento
    await asyncio.sleep(0.1)
    
    # Verificar historial de errores
    errors = engine.get_error_history()
    assert len(errors) > 0
    
    # Detener motor
    await engine.stop()
    
async def test_complex_dependency_chain(engine, components):
    """Probar cadena compleja de dependencias."""
    # Registrar componentes con cadena de dependencias
    engine.register_component(components["a"])
    engine.register_component(components["b"], ["a"])
    engine.register_component(components["c"], ["b"])
    engine.register_component(components["d"], ["a", "c"])
    
    # Iniciar motor
    await engine.start()
    
    # Verificar orden topológico
    order = engine.event_bus.topological_order
    
    # 'a' debe estar antes que sus dependientes
    a_index = order.index("a")
    b_index = order.index("b")
    c_index = order.index("c")
    d_index = order.index("d")
    
    assert a_index < b_index
    assert b_index < c_index
    assert a_index < d_index
    assert c_index < d_index
    
    # Emitir evento
    await engine.emit("test_chain", {}, "test")
    
    # Esperar propagación
    await asyncio.sleep(0.1)
    
    # Verificar que todos recibieron
    for component in components.values():
        assert len(component.events_received) == 1
        
    # Detener motor
    await engine.stop()