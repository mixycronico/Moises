"""
Configuración compartida para pruebas del motor core.

Este módulo contiene fixtures compartidos entre diferentes pruebas
del motor core, facilitando la configuración y limpieza de recursos.
"""

import asyncio
import logging
import pytest
from typing import Dict, Any, List, Optional

# Configuración de logging para pruebas
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Función interna para limpieza de engine
async def cleanup_engine(engine):
    """
    Limpieza completa del motor y tareas pendientes.
    
    Args:
        engine: Instancia del motor de eventos a limpiar
    """
    # Detener el motor
    if hasattr(engine, 'stop'):
        await engine.stop()
    
    # Cancelar tareas pendientes
    pending = [t for t in asyncio.all_tasks() 
              if not t.done() and t != asyncio.current_task()]
    
    if pending:
        logger.warning(f"Cancelando {len(pending)} tareas pendientes")
        for task in pending:
            task.cancel()
        
        try:
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Error durante la cancelación de tareas: {str(e)}")

# Importamos las clases necesarias
from genesis.core.engine_non_blocking import EngineNonBlocking
from genesis.core.engine_configurable import ConfigurableTimeoutEngine
from genesis.core.engine_priority_blocks import PriorityBlockEngine
from genesis.core.engine_dynamic_blocks import DynamicExpansionEngine


@pytest.fixture
async def non_blocking_engine():
    """
    Fixture optimizado para proporcionar un EngineNonBlocking limpio.
    
    Este fixture garantiza que los recursos se limpien correctamente
    entre pruebas, evitando problemas de recursos colgados o condiciones de carrera.
    """
    engine = EngineNonBlocking()
    yield engine
    await cleanup_engine(engine)


@pytest.fixture
async def configurable_engine():
    """
    Fixture optimizado para proporcionar un ConfigurableTimeoutEngine limpio.
    
    Este fixture proporciona un motor con timeouts configurables y
    garantiza que los recursos se limpien correctamente entre pruebas.
    """
    engine = ConfigurableTimeoutEngine(
        default_timeout=2.0,
        handler_timeout=1.0,
        recovery_timeout=1.5
    )
    yield engine
    await cleanup_engine(engine)


@pytest.fixture
async def priority_engine():
    """
    Fixture optimizado para proporcionar un PriorityBlockEngine limpio.
    
    Este fixture proporciona un motor con bloques de prioridad y
    garantiza que los recursos se limpien correctamente entre pruebas.
    """
    engine = PriorityBlockEngine()
    yield engine
    await cleanup_engine(engine)


@pytest.fixture
async def dynamic_engine():
    """
    Fixture optimizado para proporcionar un DynamicExpansionEngine limpio.
    
    Este fixture proporciona un motor con expansión dinámica y
    garantiza que los recursos se limpien correctamente entre pruebas.
    """
    engine = DynamicExpansionEngine(
        min_blocks=2,
        max_blocks=8,
        scaling_threshold=0.7,
        cooldown_period=0.1  # Período de enfriamiento reducido para pruebas
    )
    yield engine
    await cleanup_engine(engine)