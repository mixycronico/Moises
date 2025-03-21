"""
Modelos de base de datos modulares para el sistema Genesis.

Este módulo define modelos ORM optimizados para un sistema de base de datos modular,
diseñado para evitar cuellos de botella en entornos de trading de alto rendimiento.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Text, Table, MetaData
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

# Esquema para trading y datos de mercado - Alta frecuencia
class Exchange(Base):
    """Modelo para intercambios de criptomonedas."""
    __tablename__ = 'exchanges'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    symbols = relationship("Symbol", back_populates="exchange", cascade="all, delete-orphan")

class Symbol(Base):
    """Modelo para pares de trading."""
    __tablename__ = 'symbols'
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False, index=True)
    name = Column(String(20), nullable=False)
    base_asset = Column(String(10), nullable=False)
    quote_asset = Column(String(10), nullable=False)
    price_precision = Column(Integer)
    quantity_precision = Column(Integer)
    min_quantity = Column(Float)
    max_quantity = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    exchange = relationship("Exchange", back_populates="symbols")
    
    __table_args__ = (
        {'postgresql_partition_by': 'LIST (exchange_id)'}
    )

# Modelos particionados para datos de mercado (particionados por timeframe)
class Candle1m(Base):
    """Modelo para velas de 1 minuto."""
    __tablename__ = 'candles_1m'
    
    id = Column(Integer, primary_key=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    num_trades = Column(Integer)
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'}
    )

class Candle5m(Base):
    """Modelo para velas de 5 minutos."""
    __tablename__ = 'candles_5m'
    
    id = Column(Integer, primary_key=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    num_trades = Column(Integer)
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'}
    )

class Candle15m(Base):
    """Modelo para velas de 15 minutos."""
    __tablename__ = 'candles_15m'
    
    id = Column(Integer, primary_key=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    num_trades = Column(Integer)
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'}
    )

class Candle1h(Base):
    """Modelo para velas de 1 hora."""
    __tablename__ = 'candles_1h'
    
    id = Column(Integer, primary_key=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    num_trades = Column(Integer)
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'}
    )

class Candle4h(Base):
    """Modelo para velas de 4 horas."""
    __tablename__ = 'candles_4h'
    
    id = Column(Integer, primary_key=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    num_trades = Column(Integer)
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'}
    )

class Candle1d(Base):
    """Modelo para velas diarias."""
    __tablename__ = 'candles_1d'
    
    id = Column(Integer, primary_key=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    num_trades = Column(Integer)
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'}
    )

# Esquema para operaciones de trading - particionado por fecha
class Trade(Base):
    """Modelo para registrar operaciones de trading."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    trade_id = Column(String(100), unique=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' o 'sell'
    type = Column(String(20), nullable=False)  # 'market', 'limit', etc.
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float)
    fee_currency = Column(String(10))
    total = Column(Float)
    status = Column(String(20), nullable=False, index=True)  # 'open', 'closed', 'canceled'
    strategy = Column(String(50), index=True)
    execution_time = Column(Float)  # milisegundos
    entry_time = Column(DateTime, nullable=False, index=True)
    exit_time = Column(DateTime, index=True)
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    trade_metadata = Column(JSON)
    
    user = relationship("User", back_populates="trades")
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (entry_time)'}
    )

# Esquema para balances - particionado por exchange y usuario
class Balance(Base):
    """Modelo para saldos de criptomonedas."""
    __tablename__ = 'balances'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    asset = Column(String(20), nullable=False, index=True)
    free = Column(Float, nullable=False)
    used = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        {'postgresql_partition_by': 'LIST (exchange_id)'}
    )

# Esquema para usuarios y autenticación
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
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime)
    
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")

class ApiKey(Base):
    """Modelo para almacenar las claves API de intercambios."""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    exchange = Column(String(50), nullable=False)
    description = Column(String(200))
    api_key = Column(String(256), nullable=False)
    api_secret = Column(String(512), nullable=False)  # Encriptado
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    user = relationship("User", back_populates="api_keys")

# Esquema para estrategias de trading
class Strategy(Base):
    """Modelo para estrategias de trading."""
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    parameters = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Esquema para señales de trading - particionado por estrategia
class Signal(Base):
    """Modelo para señales de trading."""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False, index=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)  # 'buy', 'sell', 'hold'
    price = Column(Float, nullable=False)
    strength = Column(Float, nullable=False)  # 0.0 - 1.0
    signal_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        {'postgresql_partition_by': 'LIST (strategy_id)'}
    )

# Esquema para alertas
class Alert(Base):
    """Modelo para alertas de precio o condiciones de mercado."""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    condition_type = Column(String(50), nullable=False)  # 'price', 'indicator', etc.
    parameters = Column(JSON, nullable=False)
    message_template = Column(String(500))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_triggered = Column(DateTime)
    trigger_count = Column(Integer, default=0)
    
    user = relationship("User", back_populates="alerts")

# Esquema para resultados de backtesting
class BacktestResult(Base):
    """Modelo para almacenar resultados de backtesting."""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
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
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        {'postgresql_partition_by': 'LIST (strategy_name)'}
    )

# Esquema para métricas de rendimiento
class PerformanceMetric(Base):
    """Modelo para métricas de rendimiento del sistema."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_type = Column(String(50), nullable=False, index=True)  # 'daily', 'weekly', 'monthly', etc.
    metric_date = Column(DateTime, nullable=False, index=True)
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
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (metric_date)'}
    )

# Esquema para logs del sistema
class SystemLog(Base):
    """Modelo para logs del sistema."""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    level = Column(String(20), nullable=False, index=True)  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component = Column(String(50), nullable=False, index=True)
    source = Column(String(50), index=True)
    correlation_id = Column(String(100), index=True)
    user_id = Column(Integer, index=True)
    message = Column(Text, nullable=False)
    log_metadata = Column(JSON)
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'}
    )

# Para transacciones auditables
class AuditLog(Base):
    """Modelo para registro de auditoría de transacciones críticas."""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    action_type = Column(String(50), nullable=False, index=True)  # 'login', 'trade', 'api_key', etc.
    entity_type = Column(String(50), nullable=False)  # 'user', 'trade', 'api_key', etc.
    entity_id = Column(String(100), nullable=False)
    old_values = Column(JSON)
    new_values = Column(JSON)
    ip_address = Column(String(50))
    user_agent = Column(String(256))
    notes = Column(Text)
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'}
    )

# Tabla de configuración del sistema
class SystemConfig(Base):
    """Modelo para configuración del sistema."""
    __tablename__ = 'system_config'
    
    id = Column(Integer, primary_key=True)
    section = Column(String(50), nullable=False, index=True)
    key = Column(String(100), nullable=False, index=True)
    value = Column(Text, nullable=False)
    data_type = Column(String(20), nullable=False)  # 'string', 'integer', 'float', 'boolean', 'json'
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(Integer, ForeignKey('users.id'))