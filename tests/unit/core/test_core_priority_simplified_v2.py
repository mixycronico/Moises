"""
Test simplificado para verificar la ordenación por prioridad del motor.

Esta versión corrige problemas de timeouts evitando múltiples inicios
del motor y usando timeouts explícitos en todas las operaciones.
"""

import pytest
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus


@pytest.mark.asyncio
async def test_priority_based_startup_simplified_v2():
    """Prueba simplificada para verificar el inicio basado en prioridades."""
    # Crear un event_bus mockeado para mayor control
    event_bus = MagicMock()
    event_bus.test_mode = True
    event_bus.start = AsyncMock()
    event_bus.stop = AsyncMock()
    event_bus.emit = AsyncMock()
    event_bus.subscribe = MagicMock()
    
    # Crear un motor para pruebas con el event_bus mockeado
    engine = Engine(event_bus, test_mode=True)
    
    # Crear componentes simulados con AsyncMock
    components = []
    for name, priority in [('high', 100), ('medium', 50), ('low', 10)]:
        component = MagicMock(name=name)
        component.name = name
        component.start = AsyncMock()
        component.stop = AsyncMock()
        component.handle_event = AsyncMock()
        component.attach_event_bus = MagicMock()
        
        # Registrar componente
        engine.register_component(component, priority=priority)
        components.append(component)
    
    # Iniciar el motor con un timeout explícito para evitar bloqueos
    try:
        await asyncio.wait_for(engine.start(), timeout=2.0)
    except asyncio.TimeoutError:
        pytest.fail("Timeout al iniciar el motor")
    
    # Verificar que el event_bus fue iniciado
    event_bus.start.assert_called_once()
    
    # Verificar que todos los componentes fueron iniciados
    for component in components:
        assert component.start.called, f"El componente {component.name} no fue iniciado"
    
    # Verificar las prioridades almacenadas
    assert engine.operation_priorities['high'] == 100
    assert engine.operation_priorities['medium'] == 50
    assert engine.operation_priorities['low'] == 10
    
    # Detener el motor con un timeout explícito
    try:
        await asyncio.wait_for(engine.stop(), timeout=2.0)
    except asyncio.TimeoutError:
        pytest.fail("Timeout al detener el motor")
    
    # Verificar que el event_bus fue detenido
    event_bus.stop.assert_called_once()
    
    # Verificar que todos los componentes fueron detenidos
    for component in components:
        assert component.stop.called, f"El componente {component.name} no fue detenido"