"""
Strategies module for trading algorithms.

This module contains various trading strategies that can be used
for automated trading decision-making.
"""

from genesis.strategies.base import Strategy, SignalType
from genesis.strategies.mean_reversion import RSIStrategy, BollingerBandsStrategy
from genesis.strategies.trend_following import MovingAverageCrossover, MACDStrategy
from genesis.strategies.sentiment_based import SentimentStrategy, SocialVolumeStrategy
from genesis.strategies.orchestrator import StrategyOrchestrator
from genesis.strategies.adaptive_scaling_strategy import AdaptiveScalingStrategy, CapitalScalingManager

__all__ = [
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
    "CapitalScalingManager"
]
