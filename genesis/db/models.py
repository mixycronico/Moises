"""
Modelos de base de datos para el sistema Genesis.

Este módulo define los modelos ORM para la persistencia
de datos en el sistema de trading.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    """Modelo para usuarios del sistema."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False)
    last_login = Column(DateTime)
    
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    
class ApiKey(Base):
    """Modelo para almacenar las claves API de intercambios."""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    exchange = Column(String(50), nullable=False)
    description = Column(String(200))
    api_key = Column(String(256), nullable=False)
    api_secret = Column(String(512), nullable=False)  # Encriptado
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False)
    
    user = relationship("User", back_populates="api_keys")
    
class Trade(Base):
    """Modelo para registrar operaciones de trading."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    trade_id = Column(String(100), unique=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # 'buy' o 'sell'
    type = Column(String(20), nullable=False)  # 'market', 'limit', etc.
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float)
    fee_currency = Column(String(10))
    total = Column(Float)
    status = Column(String(20), nullable=False)  # 'open', 'closed', 'canceled'
    strategy = Column(String(50))
    execution_time = Column(Float)  # milisegundos
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    trade_metadata = Column(JSON)
    
    user = relationship("User", back_populates="trades")
    
class Candle(Base):
    """Modelo para almacenar datos de velas OHLCV."""
    __tablename__ = 'candles'
    
    id = Column(Integer, primary_key=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)  # '1m', '5m', '1h', etc.
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    num_trades = Column(Integer)
    
    __table_args__ = (
        # Índice compuesto para búsquedas rápidas
        {'sqlite_autoincrement': True},
    )
    
class Alert(Base):
    """Modelo para alertas de precio o condiciones de mercado."""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    condition_type = Column(String(50), nullable=False)  # 'price', 'indicator', etc.
    parameters = Column(JSON, nullable=False)
    message_template = Column(String(500))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False)
    last_triggered = Column(DateTime)
    trigger_count = Column(Integer, default=0)
    
    user = relationship("User", back_populates="alerts")
    
class BacktestResult(Base):
    """Modelo para almacenar resultados de backtesting."""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    profit_loss = Column(Float, nullable=False)
    profit_loss_pct = Column(Float, nullable=False)
    max_drawdown = Column(Float)
    max_drawdown_pct = Column(Float)
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    parameters = Column(JSON)
    summary = Column(Text)
    created_at = Column(DateTime, nullable=False)
    
class PerformanceMetric(Base):
    """Modelo para métricas de rendimiento del sistema."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_type = Column(String(50), nullable=False)  # 'daily', 'weekly', 'monthly', etc.
    metric_date = Column(DateTime, nullable=False)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    cumulative_return = Column(Float)
    drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    volatility = Column(Float)
    data = Column(JSON)
    
class SystemLog(Base):
    """Modelo para logs del sistema."""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    level = Column(String(20), nullable=False)  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component = Column(String(50), nullable=False)
    source = Column(String(50))
    correlation_id = Column(String(100))
    user_id = Column(Integer)
    message = Column(Text, nullable=False)
    log_metadata = Column(JSON)