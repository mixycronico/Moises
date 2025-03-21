"""
Módulo de base de datos para el sistema Genesis.

Este módulo proporciona la capa de acceso a datos y modelos
para la persistencia de información del sistema de trading.
"""

# Importar modelos principales si existen
try:
    from genesis.db.models import Base, User, Trade, Candle, BacktestResult, SystemLog, ApiKey, Alert, PerformanceMetric
except ImportError:
    # Si los modelos principales no existen, crear variables vacías para evitar errores
    Base = None
    User = Trade = Candle = BacktestResult = SystemLog = ApiKey = Alert = PerformanceMetric = None

# Importar el repositorio
try:
    from genesis.db.repository import Repository
except ImportError:
    Repository = None

# Importar modelos de paper trading
from genesis.db.paper_trading_models import (
    Base as PaperTradingBase,
    PaperTradingAccount, 
    PaperTradingBalance, 
    PaperTradingOrder, 
    PaperTradingTrade, 
    MarketData,
    PaperTradingSettings
)

__all__ = [
    "Base", "User", "Trade", "Candle", "BacktestResult", 
    "SystemLog", "ApiKey", "Alert", "PerformanceMetric",
    "Repository", 
    "PaperTradingBase", "PaperTradingAccount", "PaperTradingBalance", 
    "PaperTradingOrder", "PaperTradingTrade", "MarketData", "PaperTradingSettings"
]