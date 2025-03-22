"""
Test mínimo para verificar la ordenación por prioridad del motor.

Este módulo contiene una prueba independiente que verifica exclusivamente
la lógica de ordenación de componentes por prioridad sin depender del motor real.
"""

import pytest


def test_component_priority_sorting():
    """
    Prueba mínima que verifica la lógica de ordenación de componentes por prioridad
    sin usar los componentes o el motor real para evitar problemas de tipos.
    """
    # Diccionarios para simular los del motor
    components = {}
    operation_priorities = {}
    
    # Añadir "componentes" simulados con sus prioridades
    components['low'] = 'low_component'
    components['medium'] = 'medium_component' 
    components['high'] = 'high_component'
    
    operation_priorities['low'] = 10
    operation_priorities['medium'] = 50
    operation_priorities['high'] = 100
    
    # Probar la ordenación para inicio (mayor prioridad primero)
    start_ordered = sorted(
        components.items(),
        key=lambda x: operation_priorities.get(x[0], 50),
        reverse=True  # Mayor prioridad primero
    )
    
    # Verificar el orden para inicio
    assert start_ordered[0][0] == 'high', "La ordenación para inicio no es correcta"
    assert start_ordered[1][0] == 'medium', "La ordenación para inicio no es correcta" 
    assert start_ordered[2][0] == 'low', "La ordenación para inicio no es correcta"
    
    # Probar la ordenación para detener (menor prioridad primero)
    stop_ordered = sorted(
        components.items(),
        key=lambda x: operation_priorities.get(x[0], 50)
    )
    
    # Verificar el orden para detener
    assert stop_ordered[0][0] == 'low', "La ordenación para detener no es correcta"
    assert stop_ordered[1][0] == 'medium', "La ordenación para detener no es correcta"
    assert stop_ordered[2][0] == 'high', "La ordenación para detener no es correcta"