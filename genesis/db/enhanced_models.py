"""
Modelos de base de datos modulares optimizados para el sistema Genesis.

Este módulo define modelos ORM avanzados para un sistema de base de datos modular,
diseñado para evitar cuellos de botella en entornos de trading de alta frecuencia
con optimizaciones para rendimiento extremo, seguridad y escalabilidad.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Text
from sqlalchemy import Table, MetaData, Index, BigInteger, Enum, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid
import enum

Base = declarative_base()

# Clase enumerada para roles de usuario
class UserRole(enum.Enum):
    ADMIN = "admin"
    TRADER = "trader" 
    ANALYST = "analyst"
    VIEWER = "viewer"
    API = "api"

# Tabla de relación muchos a muchos para roles de usuario
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)

# Clase enumerada para timeframes
class Timeframe(enum.Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

# Esquema de seguridad y control de acceso
class Role(Base):
    """Modelo para roles y permisos del sistema."""
    __tablename__ = 'roles'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    permissions = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    users = relationship("User", secondary=user_roles, back_populates="roles")

# Esquema para usuarios y autenticación
class User(Base):
    """Modelo para usuarios del sistema."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, index=True, nullable=True)  # Soporte multi-tenant
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    last_login = Column(TIMESTAMP(timezone=True))
    auth_factors = Column(JSONB)  # MFA configs
    settings = Column(JSONB)  # Configuración del usuario
    
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_users_tenant_username', 'tenant_id', 'username'),
    )

class ApiKey(Base):
    """Modelo para almacenar las claves API de intercambios."""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False)
    description = Column(String(200))
    api_key = Column(String(256), nullable=False)
    api_secret = Column(String(512), nullable=False)  # Encriptado con AES-256
    api_passphrase = Column(String(512))  # Para exchanges que requieren passphrase
    permissions = Column(JSONB)  # Restricciones de la API (lecture/trade/withdraw)
    ip_whitelist = Column(JSONB)  # Lista de IPs permitidas 
    is_active = Column(Boolean, default=True)
    last_used = Column(TIMESTAMP(timezone=True))
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    data_hash = Column(String(64))  # Integridad de datos
    
    user = relationship("User", back_populates="api_keys")
    exchange = relationship("Exchange")
    
    __table_args__ = (
        Index('ix_api_keys_user_exchange', 'user_id', 'exchange_id'),
    )

# Esquema para intercambios de criptomonedas - Alta frecuencia
class Exchange(Base):
    """Modelo para intercambios de criptomonedas."""
    __tablename__ = 'exchanges'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    api_base_url = Column(String(256))
    websocket_url = Column(String(256))
    capabilities = Column(JSONB)  # Features soportados 
    rate_limits = Column(JSONB)  # Límites de API
    fees = Column(JSONB)  # Estructura de comisiones
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
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
    min_notional = Column(Float)  # Valor mínimo de la orden
    is_active = Column(Boolean, default=True)
    is_spot = Column(Boolean, default=True)
    is_margin = Column(Boolean, default=False)
    is_futures = Column(Boolean, default=False)
    contract_size = Column(Float)  # Para futuros/derivados
    tick_size = Column(Float)  # Incremento mínimo de precio
    lot_size = Column(Float)  # Incremento mínimo de cantidad
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    exchange = relationship("Exchange", back_populates="symbols")
    
    __table_args__ = (
        {'postgresql_partition_by': 'LIST (exchange_id)'},
        Index('ix_symbols_exchange_name', 'exchange_id', 'name'),
    )

# Modelo unificado para datos de velas con particionamiento por timeframe y rango de tiempo
class Candle(Base):
    """Modelo unificado para velas OHLCV."""
    __tablename__ = 'candles'
    
    id = Column(BigInteger, primary_key=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    exchange = Column(String(50), nullable=False)  # Redundante para evitar joins
    symbol = Column(String(20), nullable=False)    # Redundante para evitar joins
    timeframe = Column(String(10), nullable=False, index=True)  # '1m', '5m', '1h', etc.
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    num_trades = Column(Integer)
    vwap = Column(Float)  # Precio promedio ponderado por volumen
    is_complete = Column(Boolean, default=True)  # Si la vela está completa
    source = Column(String(20))  # Origen de los datos (API, WS, calculado)
    data_hash = Column(String(64))  # Integridad de datos
    
    __table_args__ = (
        {'postgresql_partition_by': 'LIST (timeframe), RANGE (timestamp)'},
        Index('ix_candles_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
        Index('ix_candles_exchange_symbol_timeframe', 'exchange', 'symbol', 'timeframe'),
    )

# Esquema para operaciones de trading - particionado por fecha y usuario
class Trade(Base):
    """Modelo para registrar operaciones de trading."""
    __tablename__ = 'trades'
    
    id = Column(BigInteger, primary_key=True)
    tenant_id = Column(Integer, index=True, nullable=True)  # Soporte multi-tenant
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    trade_id = Column(String(100), unique=True)
    correlation_id = Column(UUID(as_uuid=True), default=uuid.uuid4)  # Para agregar eventos relacionados
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False, index=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False, index=True)
    exchange = Column(String(50), nullable=False)  # Redundante para evitar joins
    symbol = Column(String(20), nullable=False, index=True)  # Redundante para evitar joins
    side = Column(String(10), nullable=False)  # 'buy' o 'sell'
    type = Column(String(20), nullable=False)  # 'market', 'limit', etc.
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float)
    fee_currency = Column(String(10))
    total = Column(Float)
    status = Column(String(20), nullable=False, index=True)  # 'open', 'closed', 'canceled'
    strategy_id = Column(Integer, ForeignKey('strategies.id'), index=True)
    execution_time = Column(Float)  # milisegundos
    entry_time = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    exit_time = Column(TIMESTAMP(timezone=True), index=True)
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    execution_latency = Column(Integer)  # ms desde señal hasta ejecución
    slippage = Column(Float)  # diferencia entre precio esperado y ejecutado
    data_hash = Column(String(64))  # Integridad de datos con SHA-256
    trade_metadata = Column(JSONB)  # Para mejor rendimiento en consultas JSON
    
    user = relationship("User", back_populates="trades")
    exchange_rel = relationship("Exchange")
    symbol_rel = relationship("Symbol")
    strategy = relationship("Strategy")
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (entry_time)'},
        Index('ix_trades_user_status_entry', 'user_id', 'status', 'entry_time'),
        Index('ix_trades_strategy_symbol_entry', 'strategy_id', 'symbol', 'entry_time'),
        Index('ix_trades_exchange_symbol_entry', 'exchange', 'symbol', 'entry_time'),
    )

# Eventos de trade para Event Sourcing
class TradeEvent(Base):
    """Modelo para almacenar eventos de trading (Event Sourcing)."""
    __tablename__ = 'trade_events'
    
    id = Column(BigInteger, primary_key=True)
    trade_id = Column(String(100), index=True, nullable=False)
    correlation_id = Column(UUID(as_uuid=True), index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)  # 'order_created', 'order_filled', etc.
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    data = Column(JSONB, nullable=False)
    sequence = Column(Integer, nullable=False)  # Para ordenar eventos
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'},
        Index('ix_trade_events_trade_id_sequence', 'trade_id', 'sequence'),
    )

# Esquema para balances - particionado por exchange, usuario y fecha
class Balance(Base):
    """Modelo para saldos de criptomonedas."""
    __tablename__ = 'balances'
    
    id = Column(BigInteger, primary_key=True)
    tenant_id = Column(Integer, index=True, nullable=True)  # Soporte multi-tenant
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False, index=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    asset = Column(String(20), nullable=False, index=True)
    free = Column(Float, nullable=False)
    used = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    equivalent_usd = Column(Float)  # Valor en USD
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    data_hash = Column(String(64))  # Integridad de datos
    
    user = relationship("User")
    exchange = relationship("Exchange")
    
    __table_args__ = (
        {'postgresql_partition_by': 'LIST (exchange_id), RANGE (timestamp)'},
        Index('ix_balances_user_exchange_asset', 'user_id', 'exchange_id', 'asset'),
        Index('ix_balances_user_timestamp', 'user_id', 'timestamp'),
    )

# Esquema para estrategias de trading
class Strategy(Base):
    """Modelo para estrategias de trading."""
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    type = Column(String(50), index=True)  # 'trend_following', 'mean_reversion', etc.
    parameters = Column(JSONB)
    risk_profile = Column(String(20), index=True)  # 'conservative', 'aggressive', etc.
    version = Column(String(20))  # Versión de la estrategia
    author = Column(String(100))
    is_active = Column(Boolean, default=True)
    backtest_id = Column(Integer, ForeignKey('backtest_results.id'))  # Mejor backtest
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'))
    
    performances = relationship("StrategyPerformance", back_populates="strategy", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_strategies_type_is_active', 'type', 'is_active'),
    )

# Rendimiento de estrategias
class StrategyPerformance(Base):
    """Modelo para rendimiento de estrategias."""
    __tablename__ = 'strategy_performances'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(TIMESTAMP(timezone=True), nullable=False)
    end_date = Column(TIMESTAMP(timezone=True), nullable=False)
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    max_drawdown = Column(Float)
    max_drawdown_pct = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    volatility = Column(Float)
    metrics = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    
    strategy = relationship("Strategy", back_populates="performances")
    
    __table_args__ = (
        Index('ix_strategy_performances_strategy_symbol', 'strategy_id', 'symbol'),
    )

# Esquema para señales de trading - particionado por estrategia
class Signal(Base):
    """Modelo para señales de trading."""
    __tablename__ = 'signals'
    
    id = Column(BigInteger, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False, index=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False, index=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False, index=True)  # 'buy', 'sell', 'hold'
    timeframe = Column(String(10), nullable=False, index=True)
    price = Column(Float, nullable=False)
    strength = Column(Float, nullable=False)  # 0.0 - 1.0
    confidence = Column(Float)  # 0.0 - 1.0, basado en backtest
    expiration = Column(TIMESTAMP(timezone=True))  # Cuando expira la señal
    ttl = Column(Integer)  # Time to Live en segundos
    is_executed = Column(Boolean, default=False, index=True)
    execution_id = Column(String(100), ForeignKey('trades.trade_id'))
    execution_time = Column(TIMESTAMP(timezone=True))
    execution_price = Column(Float)
    signal_metadata = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    
    strategy = relationship("Strategy")
    symbol = relationship("Symbol")
    
    __table_args__ = (
        {'postgresql_partition_by': 'LIST (strategy_id), RANGE (timestamp)'},
        Index('ix_signals_strategy_symbol_timestamp', 'strategy_id', 'symbol_id', 'timestamp'),
        Index('ix_signals_timestamp_type', 'timestamp', 'signal_type'),
    )

# Esquema para alertas
class Alert(Base):
    """Modelo para alertas de precio o condiciones de mercado."""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    condition_type = Column(String(50), nullable=False, index=True)  # 'price', 'indicator', etc.
    parameters = Column(JSONB, nullable=False)
    message_template = Column(String(500))
    notification_channels = Column(JSONB)  # 'email', 'sms', 'webhook', etc.
    is_active = Column(Boolean, default=True, index=True)
    is_recurring = Column(Boolean, default=False)  # Si puede activarse múltiples veces
    priority = Column(Integer, default=1)  # Prioridad de la alerta
    cooldown = Column(Integer)  # Segundos entre activaciones
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    last_triggered = Column(TIMESTAMP(timezone=True))
    trigger_count = Column(Integer, default=0)
    model_id = Column(Integer, ForeignKey('ml_models.id'))  # Para alertas basadas en ML
    
    user = relationship("User", back_populates="alerts")
    model = relationship("MLModel")
    
    __table_args__ = (
        Index('ix_alerts_user_symbol', 'user_id', 'symbol'),
        Index('ix_alerts_condition_is_active', 'condition_type', 'is_active'),
    )

# Modelo para ML
class MLModel(Base):
    """Modelo para almacenar configuración y metadatos de modelos ML."""
    __tablename__ = 'ml_models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    model_type = Column(String(50), nullable=False, index=True)  # 'regression', 'classification', etc.
    features = Column(JSONB)  # Lista de características usadas
    hyperparameters = Column(JSONB)  # Hiperparámetros
    metrics = Column(JSONB)  # Métricas de evaluación
    version = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    trained_at = Column(TIMESTAMP(timezone=True))
    created_by = Column(Integer, ForeignKey('users.id'))
    
    __table_args__ = (
        Index('ix_ml_models_type_is_active', 'model_type', 'is_active'),
    )

# Esquema para resultados de backtesting
class BacktestResult(Base):
    """Modelo para almacenar resultados de backtesting."""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(TIMESTAMP(timezone=True), nullable=False)
    end_date = Column(TIMESTAMP(timezone=True), nullable=False)
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
    calmar_ratio = Column(Float)
    expectancy = Column(Float)  # Expectativa matemática por operación
    avg_profit = Column(Float)  # Ganancia media
    avg_loss = Column(Float)  # Pérdida media
    avg_bars_win = Column(Float)  # Duración media de operaciones ganadoras
    avg_bars_loss = Column(Float)  # Duración media de operaciones perdedoras
    parameters = Column(JSONB)
    optimization_metric = Column(String(50))  # Métrica usada para optimización
    trades = Column(JSONB)  # Lista detallada de operaciones
    equity_curve = Column(JSONB)  # Datos de la curva de capital
    user_id = Column(Integer, ForeignKey('users.id'))
    summary = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        {'postgresql_partition_by': 'LIST (strategy_name)'},
        Index('ix_backtest_results_strategy_symbol', 'strategy_name', 'symbol'),
        Index('ix_backtest_results_user_created', 'user_id', 'created_at'),
    )

# Esquema para métricas de rendimiento en tiempo real
class RealtimeMetric(Base):
    """Modelo para métricas en tiempo real."""
    __tablename__ = 'realtime_metrics'
    
    id = Column(BigInteger, primary_key=True)
    metric_type = Column(String(50), nullable=False, index=True)  # 'latency', 'cpu', 'memory', etc.
    component = Column(String(50), nullable=False, index=True)
    timestamp = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String(20))  # 'ms', '%', 'MB', etc.
    context = Column(JSONB)
    
    __table_args__ = (
        {'postgresql_partition_by': 'LIST (metric_type), RANGE (timestamp)'},
        Index('ix_realtime_metrics_component_timestamp', 'component', 'timestamp'),
    )

# Esquema para métricas de rendimiento histórico
class PerformanceMetric(Base):
    """Modelo para métricas de rendimiento del sistema."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    metric_type = Column(String(50), nullable=False, index=True)  # 'daily', 'weekly', 'monthly', etc.
    metric_date = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    cumulative_return = Column(Float)
    drawdown = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    volatility = Column(Float)
    exposure = Column(Float)  # Porcentaje de tiempo en mercado
    strategy_allocation = Column(JSONB)  # Distribución por estrategia
    asset_allocation = Column(JSONB)  # Distribución por activo
    data = Column(JSONB)
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (metric_date)'},
        Index('ix_performance_metrics_user_type_date', 'user_id', 'metric_type', 'metric_date'),
    )

# Esquema para logs del sistema
class SystemLog(Base):
    """Modelo para logs del sistema."""
    __tablename__ = 'system_logs'
    
    id = Column(BigInteger, primary_key=True)
    timestamp = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    level = Column(String(20), nullable=False, index=True)  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component = Column(String(50), nullable=False, index=True)
    source = Column(String(50), index=True)
    correlation_id = Column(String(100), index=True)
    user_id = Column(Integer, index=True)
    trace_id = Column(String(100), index=True)  # Para seguimiento distribuido
    message = Column(Text, nullable=False)
    exception = Column(Text)  # Para stack traces
    duration = Column(Integer)  # Duración en ms
    status_code = Column(Integer)  # Código de estado HTTP/error
    endpoint = Column(String(256))  # Para logs de API
    method = Column(String(10))  # Para logs de API (GET, POST, etc.)
    ip_address = Column(String(50))
    user_agent = Column(String(256))
    log_metadata = Column(JSONB)
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'},
        Index('ix_system_logs_level_component', 'level', 'component'),
        Index('ix_system_logs_correlation', 'correlation_id'),
    )

# Para transacciones auditables
class AuditLog(Base):
    """Modelo para registro de auditoría de transacciones críticas."""
    __tablename__ = 'audit_logs'
    
    id = Column(BigInteger, primary_key=True)
    timestamp = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    action_type = Column(String(50), nullable=False, index=True)  # 'login', 'trade', 'api_key', etc.
    entity_type = Column(String(50), nullable=False, index=True)  # 'user', 'trade', 'api_key', etc.
    entity_id = Column(String(100), nullable=False, index=True)
    transaction_id = Column(String(100), index=True)  # ID de transacción en DB
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    changes = Column(JSONB)  # Específicamente qué cambió
    ip_address = Column(String(50))
    user_agent = Column(String(256))
    status = Column(String(20), index=True)  # 'success', 'failed', 'pending'
    notes = Column(Text)
    signature = Column(String(512))  # Firma digital HMAC
    
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'},
        Index('ix_audit_logs_entity', 'entity_type', 'entity_id'),
        Index('ix_audit_logs_user_action', 'user_id', 'action_type'),
    )

# Tabla de configuración del sistema
class SystemConfig(Base):
    """Modelo para configuración del sistema."""
    __tablename__ = 'system_config'
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, index=True, nullable=True)  # Soporte multi-tenant
    section = Column(String(50), nullable=False, index=True)
    key = Column(String(100), nullable=False, index=True)
    value = Column(Text, nullable=False)
    data_type = Column(String(20), nullable=False)  # 'string', 'integer', 'float', 'boolean', 'json'
    description = Column(Text)
    is_editable = Column(Boolean, default=True)
    is_sensitive = Column(Boolean, default=False)  # Si contiene información sensible
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(Integer, ForeignKey('users.id'))
    version = Column(Integer, default=1)  # Versión de la configuración
    
    __table_args__ = (
        Index('ix_system_config_section_key', 'section', 'key'),
        Index('ix_system_config_tenant', 'tenant_id'),
    )

# Historial de configuración
class ConfigHistory(Base):
    """Modelo para histórico de cambios en configuración."""
    __tablename__ = 'config_history'
    
    id = Column(Integer, primary_key=True)
    config_id = Column(Integer, ForeignKey('system_config.id'), nullable=False)
    value = Column(Text, nullable=False)
    changed_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)
    changed_by = Column(Integer, ForeignKey('users.id'))
    version = Column(Integer, nullable=False)
    change_reason = Column(Text)
    
    __table_args__ = (
        Index('ix_config_history_config_version', 'config_id', 'version'),
    )

# Buffer para datos de alta frecuencia (alternativa a Redis/Kafka)
class DataBuffer(Base):
    """Modelo para almacenamiento temporal de datos de alta frecuencia."""
    __tablename__ = 'data_buffer'
    
    id = Column(BigInteger, primary_key=True)
    data_type = Column(String(50), nullable=False, index=True)  # 'candle', 'trade', 'signal', etc.
    timestamp = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    data = Column(JSONB, nullable=False)
    processed = Column(Boolean, default=False, index=True)
    ttl = Column(Integer)  # Tiempo de vida en segundos
    
    __table_args__ = (
        {'postgresql_partition_by': 'LIST (data_type), RANGE (timestamp)'},
        {'postgresql_with': 'unlogged'},  # Tabla sin log para máximo rendimiento
        Index('ix_data_buffer_exchange_symbol', 'exchange', 'symbol'),
        Index('ix_data_buffer_processed', 'processed'),
    )