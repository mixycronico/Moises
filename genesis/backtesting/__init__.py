"""
Módulo de backtesting para el sistema Genesis.

Este módulo proporciona herramientas para realizar backtesting
de estrategias de trading, incluyendo simulación de operaciones,
optimización de parámetros y análisis de resultados.
"""

from genesis.backtesting.engine import BacktestEngine
from genesis.backtesting.api import BacktestAPI

__all__ = ["BacktestEngine", "BacktestAPI"]