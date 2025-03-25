"""
Inicializador de Seraphim para el Sistema Genesis

Este módulo proporciona las funciones necesarias para inicializar
la estrategia Seraphim y su integración con el motor de comportamiento
Gabriel en el Sistema Genesis.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List

from genesis.strategies.seraphim_strategy_integrator import SeraphimStrategyIntegrator
from genesis.accounting.capital_scaling import CapitalScalingManager

logger = logging.getLogger(__name__)

# Almacenamiento global para la instancia de la estrategia
_strategy_instance = None

async def initialize_seraphim_strategy(
    capital_base: float = 10000.0,
    symbols: Optional[List[str]] = None,
    archetype: str = "COLLECTIVE",
    capital_manager: Optional[CapitalScalingManager] = None
) -> SeraphimStrategyIntegrator:
    """
    Inicializar estrategia Seraphim en el sistema.
    
    Args:
        capital_base: Capital base para la estrategia
        symbols: Lista de símbolos a operar
        archetype: Arquetipo de comportamiento para Gabriel
        capital_manager: Gestor de capital externo (opcional)
        
    Returns:
        Instancia de SeraphimStrategyIntegrator
    """
    global _strategy_instance
    
    logger.info(f"Inicializando estrategia Seraphim con arquetipo {archetype}")
    
    if not symbols:
        symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT"]
    
    # Crear e inicializar instancia
    strategy = SeraphimStrategyIntegrator(
        capital_base=capital_base,
        symbols=symbols,
        archetype=archetype
    )
    
    success = await strategy.initialize(capital_manager)
    
    if success:
        logger.info(f"Estrategia Seraphim inicializada correctamente con {len(symbols)} símbolos")
        _strategy_instance = strategy
    else:
        logger.error("Error al inicializar estrategia Seraphim")
    
    return strategy

def get_seraphim_strategy() -> Optional[SeraphimStrategyIntegrator]:
    """
    Obtener instancia actual de la estrategia Seraphim.
    
    Returns:
        Instancia de SeraphimStrategyIntegrator o None si no está inicializada
    """
    global _strategy_instance
    return _strategy_instance

async def start_trading_cycle() -> Dict[str, Any]:
    """
    Iniciar ciclo de trading con la estrategia Seraphim.
    
    Returns:
        Información del ciclo iniciado
    """
    strategy = get_seraphim_strategy()
    
    if not strategy:
        logger.error("Estrategia Seraphim no inicializada")
        return {"status": "error", "message": "Estrategia no inicializada"}
    
    return await strategy.start_trading_cycle()

async def stop_trading_cycle() -> Dict[str, Any]:
    """
    Detener ciclo de trading actual.
    
    Returns:
        Resultados del ciclo detenido
    """
    strategy = get_seraphim_strategy()
    
    if not strategy:
        logger.error("Estrategia Seraphim no inicializada")
        return {"status": "error", "message": "Estrategia no inicializada"}
    
    return await strategy.stop_trading_cycle()

async def get_system_status() -> Dict[str, Any]:
    """
    Obtener estado completo del sistema Seraphim.
    
    Returns:
        Estado del sistema
    """
    strategy = get_seraphim_strategy()
    
    if not strategy:
        return {
            "status": "not_initialized",
            "is_running": False
        }
    
    return await strategy.get_system_status()

async def cleanup_seraphim() -> None:
    """Liberar recursos de la estrategia Seraphim."""
    global _strategy_instance
    
    if _strategy_instance:
        await _strategy_instance.cleanup()
        _strategy_instance = None
        logger.info("Recursos de estrategia Seraphim liberados")

# Función para integrar Seraphim en el sistema principal
async def integrate_with_genesis(
    scaling_manager: Optional[CapitalScalingManager] = None,
    capital_base: float = 10000.0,
    symbols: Optional[List[str]] = None
) -> bool:
    """
    Integrar la estrategia Seraphim en el sistema Genesis.
    
    Args:
        scaling_manager: Gestor de escalabilidad del sistema principal
        capital_base: Capital base para la estrategia
        symbols: Lista de símbolos a operar
        
    Returns:
        True si la integración fue exitosa
    """
    try:
        # Inicializar estrategia con el gestor de capital del sistema
        strategy = await initialize_seraphim_strategy(
            capital_base=capital_base,
            symbols=symbols,
            capital_manager=scaling_manager
        )
        
        # Verificar inicialización
        if not strategy or not strategy.is_initialized:
            logger.error("Falló la inicialización de la estrategia Seraphim")
            return False
        
        logger.info("Estrategia Seraphim integrada correctamente en el sistema Genesis")
        return True
        
    except Exception as e:
        logger.error(f"Error al integrar Seraphim en Genesis: {str(e)}")
        return False