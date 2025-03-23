"""
Módulo de estrategias avanzadas para el Sistema Genesis.

Este módulo contiene estrategias de trading de alta complejidad que integran múltiples
componentes como Reinforcement Learning, indicadores avanzados, análisis de sentimiento,
datos on-chain y simulaciones de Monte Carlo.
"""
from typing import Dict, Any, List, Type

from genesis.strategies.base import Strategy
from genesis.strategies.advanced.reinforcement_ensemble import ReinforcementEnsembleStrategy

# Diccionario de estrategias avanzadas disponibles
ADVANCED_STRATEGIES = {
    'reinforcement_ensemble': ReinforcementEnsembleStrategy
}

def get_advanced_strategy(name: str, config: Dict[str, Any]) -> Strategy:
    """
    Obtener una instancia de estrategia avanzada por nombre.
    
    Args:
        name: Nombre de la estrategia
        config: Configuración para la estrategia
        
    Returns:
        Instancia de la estrategia
        
    Raises:
        ValueError: Si la estrategia no existe
    """
    if name not in ADVANCED_STRATEGIES:
        raise ValueError(f"Estrategia avanzada '{name}' no encontrada. " 
                         f"Opciones disponibles: {list(ADVANCED_STRATEGIES.keys())}")
    
    strategy_class = ADVANCED_STRATEGIES[name]
    return strategy_class(config)

def list_advanced_strategies() -> List[str]:
    """
    Listar todas las estrategias avanzadas disponibles.
    
    Returns:
        Lista de nombres de estrategias
    """
    return list(ADVANCED_STRATEGIES.keys())

def get_all_advanced_strategy_classes() -> Dict[str, Type[Strategy]]:
    """
    Obtener todas las clases de estrategias avanzadas.
    
    Returns:
        Diccionario de nombres y clases
    """
    return ADVANCED_STRATEGIES