"""
Test mínimo para verificar la ordenación por prioridad del motor.

Este módulo contiene una prueba muy simple que se enfoca exclusivamente
en comprobar que el motor inicia los componentes en orden de prioridad.
"""

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from genesis.core.engine import Engine


@pytest.mark.asyncio
async def test_priority_based_startup_minimal():
    """Prueba mínima para verificar el inicio basado en prioridades."""
    # Crear un motor para pruebas
    engine = Engine(test_mode=True)
    
    # Crear componentes simulados con AsyncMock
    high = MagicMock(name="high")
    high.name = "high"
    high.start = AsyncMock()
    high.stop = AsyncMock()
    high.handle_event = AsyncMock()
    high.attach_event_bus = MagicMock()
    
    medium = MagicMock(name="medium")
    medium.name = "medium"
    medium.start = AsyncMock()
    medium.stop = AsyncMock()
    medium.handle_event = AsyncMock()
    medium.attach_event_bus = MagicMock()
    
    low = MagicMock(name="low")
    low.name = "low"
    low.start = AsyncMock()
    low.stop = AsyncMock()
    low.handle_event = AsyncMock()
    low.attach_event_bus = MagicMock()
    
    # Registrar componentes simulados con diferentes prioridades
    engine.register_component(low, priority=10)
    engine.register_component(medium, priority=50)
    engine.register_component(high, priority=100)
    
    # Iniciar el motor para disparar las llamadas a start
    await engine.start()
    
    # Verificar que todos los componentes fueron iniciados
    high.start.assert_called_once()
    medium.start.assert_called_once()
    low.start.assert_called_once()
    
    # Verificar el orden de las llamadas mediante el mock_calls registry
    # Nota: Esto es una aproximación simplificada del orden real, ya que
    # las llamadas asíncronas podrían tener un orden ligeramente diferente,
    # pero con la implementación actual de prioridades deberían seguir el orden.
    call_order = []
    
    # Simular el registro del orden de llamadas
    with patch.object(engine, '_start_component', side_effect=engine._start_component) as mock_start:
        # Reiniciar el motor para usar el mock
        await engine.stop()
        await engine.start()
        
        # Verificar que _start_component fue llamado 3 veces
        assert mock_start.call_count == 3
        
        # Primer argumento (componente) de cada llamada
        components_started = [call.args[0] for call in mock_start.call_args_list]
        
        # Verificar que el orden es por prioridad (high, medium, low)
        component_names = [comp.name for comp in components_started]
        assert component_names == ['high', 'medium', 'low'], f"El orden de inicio es incorrecto: {component_names}"
    
    # Detener el motor
    await engine.stop()