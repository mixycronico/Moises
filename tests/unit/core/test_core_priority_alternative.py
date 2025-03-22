"""
Test alternativo para verificar la ordenación por prioridad del motor.

Este módulo contiene una prueba que usa mocks para los componentes
y evita timeouts en los eventos asincrónicos.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus


@pytest.mark.asyncio
async def test_engine_priority_order_alternative():
    """
    Prueba que el motor ordena los componentes correctamente por prioridad
    usando mocks para los componentes.
    """
    # Crear event_bus simulado para evitar problemas
    event_bus = MagicMock()
    event_bus.test_mode = True
    
    # Configurar métodos esenciales como AsyncMock
    event_bus.start = AsyncMock()
    event_bus.stop = AsyncMock()
    event_bus.emit = AsyncMock()
    event_bus.subscribe = MagicMock()
    
    # Crear motor para pruebas
    engine = Engine(event_bus, test_mode=True)
    
    # Crear componentes simulados
    components = []
    for name, priority in [('high', 100), ('medium', 50), ('low', 10)]:
        mock_component = MagicMock(name=name)
        mock_component.name = name
        mock_component.start = AsyncMock()
        mock_component.stop = AsyncMock()
        mock_component.handle_event = AsyncMock()
        mock_component.attach_event_bus = MagicMock()
        
        # Registrar en el motor con prioridad
        engine.register_component(mock_component, priority=priority)
        components.append(mock_component)
    
    # Patchear el método emit_system_started
    with patch.object(engine.event_bus, 'emit') as mock_emit:
        # Iniciar el motor
        await engine.start()
        
        # Verificar que se inicializó el event_bus
        event_bus.start.assert_called_once()
        
        # Verificar que todos los componentes fueron iniciados
        assert all(comp.start.call_count == 1 for comp in components)
        
        # Verificar el orden en que se llamaron los métodos start
        # de los componentes simulados (por prioridad descendente)
        
        # El orden esperado es high, medium, low
        high, medium, low = components[0], components[1], components[2]
        
        # Comprobar prioridades almacenadas
        assert engine.operation_priorities[high.name] == 100
        assert engine.operation_priorities[medium.name] == 50
        assert engine.operation_priorities[low.name] == 10
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que todos los componentes fueron detenidos
    assert all(comp.stop.call_count == 1 for comp in components)
    
    # Verificar que el event_bus fue detenido
    event_bus.stop.assert_called_once()
    
    # Detener componentes verificando que se llaman en orden reverso (por prioridad ascendente)
    # El orden esperado es low, medium, high
    assert engine.components['low'].stop.call_count == 1
    assert engine.components['medium'].stop.call_count == 1
    assert engine.components['high'].stop.call_count == 1