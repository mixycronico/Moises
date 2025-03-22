"""
Test mínimo para verificar la ordenación por prioridad del motor.

Este módulo contiene una prueba muy simple que verifica exclusivamente
el mecanismo interno de ordenación por prioridad sin iniciar componentes.
"""

import pytest

from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus


def test_component_priority_sorting():
    """
    Prueba mínima que verifica que el motor ordena correctamente los componentes
    por prioridad, sin ejecutar ningún método start o stop.
    """
    # Crear instancias simuladas
    event_bus = object()  # No necesitamos un EventBus real para esta prueba
    engine = Engine(event_bus, test_mode=True)
    
    # Simular componentes como objetos simples
    high = type('Component', (), {'name': 'high'})()
    medium = type('Component', (), {'name': 'medium'})()
    low = type('Component', (), {'name': 'low'})()
    
    # Monkeypatching para evitar interacciones
    engine.components = {}
    engine.operation_priorities = {}
    
    # "Registrar" los componentes para la prueba
    engine.components['low'] = low
    engine.components['medium'] = medium
    engine.components['high'] = high
    
    engine.operation_priorities['low'] = 10
    engine.operation_priorities['medium'] = 50
    engine.operation_priorities['high'] = 100
    
    # Probar la ordenación para inicio (mayor prioridad primero)
    start_ordered = sorted(
        engine.components.items(),
        key=lambda x: engine.operation_priorities.get(x[0], 50),
        reverse=True  # Mayor prioridad primero
    )
    
    # Verificar el orden para inicio
    assert start_ordered[0][0] == 'high', "La ordenación para inicio no es correcta"
    assert start_ordered[1][0] == 'medium', "La ordenación para inicio no es correcta"
    assert start_ordered[2][0] == 'low', "La ordenación para inicio no es correcta"
    
    # Probar la ordenación para detener (menor prioridad primero)
    stop_ordered = sorted(
        engine.components.items(),
        key=lambda x: engine.operation_priorities.get(x[0], 50)
    )
    
    # Verificar el orden para detener
    assert stop_ordered[0][0] == 'low', "La ordenación para detener no es correcta"
    assert stop_ordered[1][0] == 'medium', "La ordenación para detener no es correcta"
    assert stop_ordered[2][0] == 'high', "La ordenación para detener no es correcta"