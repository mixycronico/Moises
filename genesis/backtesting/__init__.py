"""
Módulo de backtesting para el sistema Genesis.

Este módulo proporciona herramientas para realizar pruebas retrospectivas
de estrategias de trading usando datos históricos.
"""

from genesis.backtesting.engine import BacktestEngine
from genesis.backtesting.api import BacktestAPI

__all__ = ["BacktestEngine", "BacktestAPI"]