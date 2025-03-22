"""
Prueba simplificada para verificar los métodos start y stop del motor.

Este módulo prueba específicamente la funcionalidad de inicio y parada
del motor sin depender de componentes complejos o interacciones asíncronas.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus


class SimpleComponent:
    """Componente muy simple para pruebas."""
    __test__ = False  # Esto informa a pytest que no es una clase de prueba
    
    def __init__(self, name):
        self.name = name
        self.started = False
        self.stopped = False
        self.handle_event = AsyncMock()
        self.attach_event_bus = MagicMock()
    
    async def start(self):
        self.started = True
        return self.name
    
    async def stop(self):
        self.stopped = True
        return self.name


@pytest.mark.asyncio
async def test_engine_start_stop_basic():
    """Prueba básica de inicio y parada del motor."""
    # Crear mocks en lugar de objetos reales para evitar problemas de timeout
    event_bus = MagicMock()
    event_bus.start = AsyncMock()
    event_bus.stop = AsyncMock()
    event_bus.emit = AsyncMock()
    event_bus.subscribe = MagicMock()
    
    # Crear un motor con el event_bus simulado
    engine = Engine(event_bus, test_mode=True)
    
    # Crear componentes simples con prioridades diferentes
    components = [
        SimpleComponent("high"),
        SimpleComponent("medium"),
        SimpleComponent("low")
    ]
    
    # Registrar componentes con diferentes prioridades
    engine.register_component(components[0], priority=100)
    engine.register_component(components[1], priority=50)
    engine.register_component(components[2], priority=10)
    
    # Iniciar el motor - no necesitamos timeout porque todo está mockeado
    await engine.start()
    
    # Verificar que el motor está corriendo
    assert engine.running is True
    
    # Verificar que todos los componentes fueron iniciados
    for component in components:
        assert component.started is True, f"El componente {component.name} no fue iniciado"
    
    # Verificar prioridades almacenadas
    assert engine.operation_priorities["high"] == 100
    assert engine.operation_priorities["medium"] == 50
    assert engine.operation_priorities["low"] == 10
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que el motor está detenido
    assert engine.running is False
    
    # Verificar que todos los componentes fueron detenidos
    for component in components:
        assert component.stopped is True, f"El componente {component.name} no fue detenido"