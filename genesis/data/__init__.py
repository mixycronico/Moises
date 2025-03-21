"""
Módulo de datos para el sistema Genesis.

Este módulo proporciona funcionalidades para la obtención, procesamiento
y análisis de datos de mercado para el sistema de trading.
"""

from genesis.data.market_data import MarketDataManager
from genesis.data.indicators import calculate_indicators
from genesis.data.analyzer import MarketAnalyzer

__all__ = ["MarketDataManager", "calculate_indicators", "MarketAnalyzer"]