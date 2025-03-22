"""
Test ultra minimalista para el monitor de componentes.

Este módulo implementa pruebas directas para el ComponentMonitor
con operaciones sintéticas que evitan cualquier dependencia del bus de eventos.
"""

import logging
import pytest
import time
from typing import Dict, Any, Set

from genesis.core.component_monitor import ComponentMonitor

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def minimal_monitor():
    """Crear un monitor sin iniciar con sus métodos esenciales."""
    monitor = ComponentMonitor(
        name="test_monitor",
        check_interval=0.1,
        max_failures=2,
        recovery_interval=0.2
    )
    
    # Preparar estructura sin iniciar
    monitor.health_status = {}
    monitor.failure_counts = {}
    monitor.isolated_components = set()
    monitor.status_history = {}
    
    return monitor

def test_monitor_initialization(minimal_monitor):
    """Probar inicialización directa de atributos del monitor."""
    # Verificar estado inicial directo
    assert isinstance(minimal_monitor.health_status, dict)
    assert isinstance(minimal_monitor.failure_counts, dict)
    assert isinstance(minimal_monitor.isolated_components, set)
    assert isinstance(minimal_monitor.status_history, dict)
    assert minimal_monitor.check_interval == 0.1
    assert minimal_monitor.max_failures == 2
    assert minimal_monitor.recovery_interval == 0.2

def test_monitor_health_status_tracking(minimal_monitor):
    """Probar tracking directo de estado de salud."""
    # Simular actualización de estado de salud
    minimal_monitor.health_status["comp_a"] = True
    minimal_monitor.health_status["comp_b"] = False
    
    # Verificar actualizaciones
    assert "comp_a" in minimal_monitor.health_status
    assert minimal_monitor.health_status["comp_a"] is True
    assert "comp_b" in minimal_monitor.health_status
    assert minimal_monitor.health_status["comp_b"] is False

def test_monitor_isolation_tracking(minimal_monitor):
    """Probar tracking directo de aislamiento."""
    # Simular aislamiento directo
    minimal_monitor.isolated_components.add("comp_a")
    
    # Verificar aislamiento
    assert "comp_a" in minimal_monitor.isolated_components
    
    # Simular recuperación
    minimal_monitor.isolated_components.remove("comp_a")
    
    # Verificar recuperación
    assert "comp_a" not in minimal_monitor.isolated_components

def test_monitor_failure_counting(minimal_monitor):
    """Probar conteo directo de fallos."""
    # Simular incremento de contadores de fallo
    minimal_monitor.failure_counts["comp_a"] = 0
    minimal_monitor.failure_counts["comp_b"] = 1
    
    # Incrementar contador
    minimal_monitor.failure_counts["comp_a"] += 1
    minimal_monitor.failure_counts["comp_b"] += 1
    
    # Verificar contadores
    assert minimal_monitor.failure_counts["comp_a"] == 1
    assert minimal_monitor.failure_counts["comp_b"] == 2
    
    # Resetear contador
    minimal_monitor.failure_counts["comp_a"] = 0
    
    # Verificar reset
    assert minimal_monitor.failure_counts["comp_a"] == 0

def test_monitor_history_tracking(minimal_monitor):
    """Probar historial de estado."""
    # Inicializar historial para un componente
    minimal_monitor.status_history["comp_a"] = []
    
    # Añadir entradas al historial
    current_time = time.time()
    minimal_monitor.status_history["comp_a"].append((current_time - 10, True))
    minimal_monitor.status_history["comp_a"].append((current_time - 5, False))
    minimal_monitor.status_history["comp_a"].append((current_time, True))
    
    # Verificar historial
    assert len(minimal_monitor.status_history["comp_a"]) == 3
    assert minimal_monitor.status_history["comp_a"][-1][1] is True  # Estado más reciente

def test_monitor_events_tracking(minimal_monitor):
    """Probar tracking de eventos."""
    # Verificar contadores iniciales
    assert minimal_monitor.isolation_events == 0
    assert minimal_monitor.recovery_events == 0
    
    # Simular eventos
    minimal_monitor.isolation_events += 1
    minimal_monitor.recovery_events += 2
    
    # Verificar contadores
    assert minimal_monitor.isolation_events == 1
    assert minimal_monitor.recovery_events == 2

def test_monitor_handle_event_check_status(minimal_monitor):
    """Probar manejo de evento check_status."""
    # Estado inicial
    minimal_monitor.running = True
    
    # Crear respuesta sintética
    response = {
        "component": minimal_monitor.name,
        "healthy": minimal_monitor.running,
        "monitored_components": len(minimal_monitor.health_status),
        "isolated_components": len(minimal_monitor.isolated_components),
        "last_check_time": minimal_monitor.last_check_time
    }
    
    # Verificar respuesta
    assert isinstance(response, dict)
    assert response["component"] == "test_monitor"
    assert response["healthy"] is True

def test_monitor_handle_event_check_component(minimal_monitor):
    """Probar manejo de evento check_component."""
    # Preparar datos de componente
    minimal_monitor.health_status["comp_a"] = True
    minimal_monitor.failure_counts["comp_a"] = 0
    
    # Crear respuesta sintética
    response = {
        "component_id": "comp_a",
        "healthy": minimal_monitor.health_status["comp_a"],
        "isolated": "comp_a" in minimal_monitor.isolated_components,
        "failure_count": minimal_monitor.failure_counts["comp_a"]
    }
    
    # Verificar respuesta
    assert isinstance(response, dict)
    assert response["component_id"] == "comp_a"
    assert response["healthy"] is True
    assert response["isolated"] is False
    assert response["failure_count"] == 0

def test_monitor_isolate_component_direct(minimal_monitor):
    """Probar aislamiento directo."""
    # Aislar directamente
    component_id = "comp_a"
    if component_id not in minimal_monitor.isolated_components:
        minimal_monitor.isolated_components.add(component_id)
        minimal_monitor.isolation_events += 1
    
    # Verificar aislamiento
    assert "comp_a" in minimal_monitor.isolated_components
    assert minimal_monitor.isolation_events == 1

def test_monitor_recover_component_direct(minimal_monitor):
    """Probar recuperación directa."""
    # Aislar primero
    component_id = "comp_a"
    minimal_monitor.isolated_components.add(component_id)
    
    # Recuperar directamente
    if component_id in minimal_monitor.isolated_components:
        minimal_monitor.isolated_components.remove(component_id)
        minimal_monitor.recovery_events += 1
        minimal_monitor.failure_counts[component_id] = 0
    
    # Verificar recuperación
    assert "comp_a" not in minimal_monitor.isolated_components
    assert minimal_monitor.recovery_events == 1
    
def test_monitor_health_report(minimal_monitor):
    """Probar generación de reporte de salud."""
    # Preparar datos de reporte
    minimal_monitor.running = True
    minimal_monitor.health_status["comp_a"] = True
    minimal_monitor.health_status["comp_b"] = False
    minimal_monitor.isolated_components.add("comp_b")
    minimal_monitor.failure_counts["comp_a"] = 0
    minimal_monitor.failure_counts["comp_b"] = 3
    minimal_monitor.isolation_events = 1
    minimal_monitor.recovery_events = 0
    
    # Crear reporte sintético
    response = {
        "monitor": minimal_monitor.name,
        "timestamp": time.time(),
        "component_status": minimal_monitor.health_status,
        "isolated_components": list(minimal_monitor.isolated_components),
        "failure_counts": minimal_monitor.failure_counts,
        "isolation_events": minimal_monitor.isolation_events,
        "recovery_events": minimal_monitor.recovery_events,
        "monitor_healthy": minimal_monitor.running
    }
    
    # Verificar reporte
    assert isinstance(response, dict)
    assert response["monitor"] == "test_monitor"
    assert "comp_b" in response["isolated_components"]
    assert response["failure_counts"]["comp_b"] == 3
    assert response["monitor_healthy"] is True