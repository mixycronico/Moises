"""
Test ultra simplificado para el motor basado en grafo.

Este módulo contiene una prueba extremadamente simplificada
que solo verifica la inicialización básica del motor.
"""

import pytest
import asyncio
from typing import Dict, Any, Optional

from genesis.core.component import Component
from genesis.core.engine_graph_based import GraphBasedEngine

# Prueba única ultra simple
async def test_graph_engine_simple_init():
    """Probar solo la inicialización básica sin registro de componentes."""
    engine = GraphBasedEngine(test_mode=True)
    assert not engine.running
    assert len(engine.components) == 0
    assert engine.event_bus is not None