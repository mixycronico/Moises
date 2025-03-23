"""
Módulo de integración de APIs externas para el Sistema Genesis.

Este módulo proporciona una interfaz unificada para interactuar con 
diversas APIs externas como Alpha Vantage, NewsAPI, CoinMarketCap, 
Reddit, y DeepSeek, entre otras.
"""

from .api_manager import api_manager, initialize, close, test_apis

__all__ = ['api_manager', 'initialize', 'close', 'test_apis']