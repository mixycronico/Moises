"""
Módulo de base de datos para el sistema Genesis.

Este módulo proporciona la capa de acceso a datos y modelos
para la persistencia de información del sistema de trading.
"""

from genesis.db.models import Base, User, Trade, Candle, BacktestResult, SystemLog, ApiKey, Alert, PerformanceMetric
from genesis.db.repository import DatabaseManager, BaseRepository

__all__ = [
    "Base", "User", "Trade", "Candle", "BacktestResult", 
    "SystemLog", "ApiKey", "Alert", "PerformanceMetric",
    "DatabaseManager", "BaseRepository"
]