"""
Módulo de estrategias avanzadas para el Sistema Genesis.

Este paquete contiene implementaciones de estrategias de trading avanzadas
que utilizan técnicas como aprendizaje por refuerzo, ensemble learning,
análisis de sentimiento, y capacidades de DeepSeek para toma de decisiones.
"""

from typing import Dict, Any, List, Optional, Union

# Importar estrategias avanzadas
try:
    from genesis.strategies.advanced.reinforcement_ensemble import ReinforcementEnsembleStrategy
except ImportError:
    from genesis.strategies.advanced.reinforcement_ensemble_simple import ReinforcementEnsembleStrategy

# Diccionario con todas las estrategias avanzadas disponibles
ADVANCED_STRATEGIES = {
    "reinforcement_ensemble": ReinforcementEnsembleStrategy,
    # Otras estrategias avanzadas aquí
}

def get_advanced_strategy(strategy_name: str, config: Dict[str, Any]) -> Optional[Any]:
    """
    Obtener una instancia de una estrategia avanzada por nombre.
    
    Args:
        strategy_name: Nombre de la estrategia
        config: Configuración para la estrategia
        
    Returns:
        Instancia de la estrategia o None si no existe
    """
    strategy_class = ADVANCED_STRATEGIES.get(strategy_name)
    if strategy_class:
        return strategy_class(config)
    return None

def list_advanced_strategies() -> List[str]:
    """
    Listar todas las estrategias avanzadas disponibles.
    
    Returns:
        Lista con los nombres de las estrategias
    """
    return list(ADVANCED_STRATEGIES.keys())

__all__ = [
    'ReinforcementEnsembleStrategy',
    'get_advanced_strategy',
    'list_advanced_strategies',
]