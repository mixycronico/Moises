"""
Módulo de datos para el sistema Genesis.

Este módulo proporciona funcionalidades para obtener, procesar y analizar
datos del mercado, incluyendo precios históricos, indicadores técnicos,
y patrones de precio.
"""

from genesis.data.market_data import MarketDataManager
from genesis.data.indicators import calculate_indicators
from genesis.data.analyzer import MarketAnalyzer

__all__ = ["MarketDataManager", "calculate_indicators", "MarketAnalyzer"]