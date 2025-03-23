"""
Strategies module for trading algorithms.

Este módulo contiene diversas estrategias de trading que pueden usarse
para la toma de decisiones automatizada en el sistema Genesis.
Incluye estrategias básicas así como estrategias avanzadas basadas en
aprendizaje por refuerzo, indicadores técnicos avanzados, análisis de sentimiento, 
datos on-chain y simulaciones de Monte Carlo.
"""

from genesis.strategies.base import Strategy, SignalType
from genesis.strategies.mean_reversion import RSIStrategy, BollingerBandsStrategy
from genesis.strategies.trend_following import MovingAverageCrossover, MACDStrategy
from genesis.strategies.sentiment_based import SentimentStrategy, SocialVolumeStrategy
from genesis.strategies.orchestrator import StrategyOrchestrator
from genesis.strategies.adaptive_scaling_strategy import AdaptiveScalingStrategy, CapitalScalingManager

# Importar estrategias avanzadas
from genesis.strategies.advanced import (
    ReinforcementEnsembleStrategy,
    get_advanced_strategy,
    list_advanced_strategies
)

__all__ = [
    # Estrategias básicas
    "Strategy",
    "SignalType",
    "RSIStrategy",
    "BollingerBandsStrategy",
    "MovingAverageCrossover",
    "MACDStrategy",
    "SentimentStrategy",
    "SocialVolumeStrategy",
    "StrategyOrchestrator",
    "AdaptiveScalingStrategy",
    "CapitalScalingManager",
    
    # Estrategias avanzadas
    "ReinforcementEnsembleStrategy",
    "get_advanced_strategy",
    "list_advanced_strategies"
]
