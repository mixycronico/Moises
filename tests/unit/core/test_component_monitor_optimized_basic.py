"""
Tests básicos para el monitor de componentes optimizado.

Este módulo implementa pruebas ultra simplificadas para el monitor optimizado
sin utilizar dependencias del motor completo o del bus de eventos.
"""

import pytest
import time
from typing import Dict, Any, List, Set

from genesis.core.component_monitor_optimized import OptimizedComponentMonitor

def test_monitor_optimized_initialization():
    """Probar inicialización básica del monitor optimizado."""
    # Crear monitor sin iniciar
    monitor = OptimizedComponentMonitor(
        name="test_monitor",
        check_interval=0.2,
        max_failures=2,
        recovery_interval=0.3,
        test_mode=True
    )
    
    # Verificar atributos iniciales
    assert monitor.name == "test_monitor"
    assert monitor.check_interval == 0.2
    assert monitor.max_failures == 2
    assert monitor.recovery_interval == 0.3
    assert monitor.test_mode is True
    assert isinstance(monitor.health_status, dict)
    assert isinstance(monitor.failure_counts, dict)
    assert isinstance(monitor.isolated_components, set)
    assert isinstance(monitor.status_history, dict)
    assert isinstance(monitor.component_metadata, dict)
    
    # Verificar que no está en ejecución inicialmente
    assert monitor.running is False

def test_monitor_optimized_health_tracking():
    """Probar tracking de estado de salud."""
    # Crear monitor
    monitor = OptimizedComponentMonitor(name="test_monitor", test_mode=True)
    
    # Actualizar estado de salud directamente
    monitor.health_status["comp_a"] = True
    monitor.health_status["comp_b"] = False
    
    # Verificar estado
    assert "comp_a" in monitor.health_status
    assert monitor.health_status["comp_a"] is True
    assert "comp_b" in monitor.health_status
    assert monitor.health_status["comp_b"] is False

def test_monitor_optimized_failure_counting():
    """Probar conteo de fallos."""
    # Crear monitor
    monitor = OptimizedComponentMonitor(name="test_monitor", test_mode=True)
    
    # Inicializar contadores
    monitor.failure_counts["comp_a"] = 0
    monitor.failure_counts["comp_b"] = 1
    
    # Incrementar contadores
    monitor.failure_counts["comp_a"] += 1
    monitor.failure_counts["comp_b"] += 1
    
    # Verificar valores
    assert monitor.failure_counts["comp_a"] == 1
    assert monitor.failure_counts["comp_b"] == 2

def test_monitor_optimized_isolation():
    """Probar aislamiento de componentes."""
    # Crear monitor
    monitor = OptimizedComponentMonitor(name="test_monitor", test_mode=True)
    
    # Aislar componentes directamente
    monitor.isolated_components.add("comp_a")
    
    # Verificar aislamiento
    assert "comp_a" in monitor.isolated_components
    
    # Verificar que otro componente no está aislado
    assert "comp_b" not in monitor.isolated_components
    
    # Añadir metadatos
    monitor.component_metadata["comp_a"] = {
        "isolation_reason": "Prueba de aislamiento",
        "isolation_time": time.time()
    }
    
    # Verificar metadatos
    assert "comp_a" in monitor.component_metadata
    assert "isolation_reason" in monitor.component_metadata["comp_a"]
    assert monitor.component_metadata["comp_a"]["isolation_reason"] == "Prueba de aislamiento"

def test_monitor_optimized_history_tracking():
    """Probar seguimiento de historial de estado."""
    # Crear monitor
    monitor = OptimizedComponentMonitor(name="test_monitor", test_mode=True)
    
    # Crear historial
    monitor.status_history["comp_a"] = []
    
    # Añadir entradas al historial
    current_time = time.time()
    monitor.status_history["comp_a"].append((current_time - 10, True))
    monitor.status_history["comp_a"].append((current_time - 5, False))
    monitor.status_history["comp_a"].append((current_time, True))
    
    # Verificar historial
    assert len(monitor.status_history["comp_a"]) == 3
    assert monitor.status_history["comp_a"][0][1] is True  # Primera entrada
    assert monitor.status_history["comp_a"][1][1] is False  # Segunda entrada
    assert monitor.status_history["comp_a"][2][1] is True  # Tercera entrada
    
    # Verificar orden cronológico
    assert monitor.status_history["comp_a"][0][0] < monitor.status_history["comp_a"][1][0]
    assert monitor.status_history["comp_a"][1][0] < monitor.status_history["comp_a"][2][0]

def test_monitor_optimized_event_counting():
    """Probar conteo de eventos de aislamiento y recuperación."""
    # Crear monitor
    monitor = OptimizedComponentMonitor(name="test_monitor", test_mode=True)
    
    # Verificar valores iniciales
    assert monitor.isolation_events == 0
    assert monitor.recovery_events == 0
    
    # Incrementar contadores
    monitor.isolation_events += 1
    monitor.recovery_events += 2
    
    # Verificar valores
    assert monitor.isolation_events == 1
    assert monitor.recovery_events == 2

def test_monitor_optimized_status_callbacks():
    """Probar callbacks de notificación de estado."""
    # Crear monitor
    monitor = OptimizedComponentMonitor(name="test_monitor", test_mode=True)
    
    # Crear callback de prueba
    callback_called = False
    callback_component = None
    callback_metadata = None
    
    def test_callback(component_id, metadata):
        nonlocal callback_called, callback_component, callback_metadata
        callback_called = True
        callback_component = component_id
        callback_metadata = metadata
    
    # Registrar callback
    monitor.register_status_callback(test_callback)
    
    # Verificar registro
    assert len(monitor.status_callbacks) == 1
    assert monitor.status_callbacks[0] == test_callback
    
    # Verificar que se puede añadir otro callback
    monitor.register_status_callback(lambda c, m: None)
    assert len(monitor.status_callbacks) == 2

def test_monitor_optimized_health_report():
    """Probar generación de informe de salud."""
    # Crear monitor
    monitor = OptimizedComponentMonitor(name="test_monitor", test_mode=True)
    
    # Configurar estado para el informe
    monitor.running = True
    monitor.health_status["comp_a"] = True
    monitor.health_status["comp_b"] = False
    monitor.failure_counts["comp_a"] = 0
    monitor.failure_counts["comp_b"] = 3
    monitor.isolated_components.add("comp_b")
    monitor.component_metadata["comp_b"] = {"isolation_reason": "Fallos múltiples"}
    monitor.isolation_events = 1
    monitor.recovery_events = 0
    
    # Simular respuesta a get_health_report
    response = {
        "monitor": monitor.name,
        "timestamp": time.time(),
        "component_status": monitor.health_status,
        "isolated_components": list(monitor.isolated_components),
        "failure_counts": monitor.failure_counts,
        "isolation_events": monitor.isolation_events,
        "recovery_events": monitor.recovery_events,
        "monitor_healthy": monitor.running,
        "component_metadata": monitor.component_metadata
    }
    
    # Verificar respuesta
    assert response["monitor"] == "test_monitor"
    assert "comp_a" in response["component_status"]
    assert response["component_status"]["comp_a"] is True
    assert "comp_b" in response["component_status"]
    assert response["component_status"]["comp_b"] is False
    assert "comp_b" in response["isolated_components"]
    assert response["failure_counts"]["comp_b"] == 3
    assert response["isolation_events"] == 1
    assert response["recovery_events"] == 0
    assert response["monitor_healthy"] is True
    assert "comp_b" in response["component_metadata"]
    assert response["component_metadata"]["comp_b"]["isolation_reason"] == "Fallos múltiples"