"""
Database models for the Genesis trading system.

This module defines SQLAlchemy models for database tables.
"""

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, ForeignKey, Text, 
    UniqueConstraint, Index, JSON, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class TradeStatus(enum.Enum):
    """Enum for trade status."""
    OPEN = 'open'
    CLOSED = 'closed'
    CANCELLED = 'cancelled'


class Exchange(Base):
    """Exchange model."""
    __tablename__ = 'exchanges'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    symbols = relationship("Symbol", back_populates="exchange")
    candles = relationship("Candle", back_populates="exchange")
    trades = relationship("Trade", back_populates="exchange")


class Symbol(Base):
    """Trading symbol model."""
    __tablename__ = 'symbols'
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False)
    name = Column(String(20), nullable=False)
    base_asset = Column(String(10), nullable=False)
    quote_asset = Column(String(10), nullable=False)
    price_precision = Column(Integer, default=8)
    quantity_precision = Column(Integer, default=8)
    min_quantity = Column(Float, nullable=True)
    max_quantity = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    exchange = relationship("Exchange", back_populates="symbols")
    candles = relationship("Candle", back_populates="symbol")
    trades = relationship("Trade", back_populates="symbol")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('exchange_id', 'name', name='uq_exchange_symbol'),
    )


class Candle(Base):
    """OHLCV candle model."""
    __tablename__ = 'candles'
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(String(10), nullable=False)  # e.g., '1m', '1h', '1d'
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    exchange = relationship("Exchange", back_populates="candles")
    symbol = relationship("Symbol", back_populates="candles")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('exchange_id', 'symbol_id', 'timestamp', 'timeframe', name='uq_candle'),
        Index('ix_candles_timestamp', 'timestamp'),
        Index('ix_candles_symbol_timestamp', 'symbol_id', 'timestamp'),
    )


class Strategy(Base):
    """Trading strategy model."""
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    parameters = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    signals = relationship("Signal", back_populates="strategy")
    trades = relationship("Trade", back_populates="strategy")


class Signal(Base):
    """Trading signal model."""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    signal_type = Column(String(20), nullable=False)  # e.g., 'buy', 'sell', 'exit'
    price = Column(Float, nullable=True)
    strength = Column(Float, nullable=True)
    signal_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="signals")
    symbol = relationship("Symbol")
    
    # Indexes
    __table_args__ = (
        Index('ix_signals_strategy_symbol', 'strategy_id', 'symbol_id'),
        Index('ix_signals_timestamp', 'timestamp'),
    )


class Trade(Base):
    """Trade model."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(50), nullable=False, unique=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False)
    
    side = Column(String(10), nullable=False)  # 'long' or 'short'
    status = Column(Enum(TradeStatus), nullable=False, default=TradeStatus.OPEN)
    
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    
    quantity = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    leverage = Column(Float, default=1.0)
    
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    
    realized_pnl = Column(Float, nullable=True)
    fees = Column(Float, default=0.0)
    
    trade_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    exchange = relationship("Exchange", back_populates="trades")
    symbol = relationship("Symbol", back_populates="trades")
    strategy = relationship("Strategy", back_populates="trades")
    
    # Indexes
    __table_args__ = (
        Index('ix_trades_trade_id', 'trade_id'),
        Index('ix_trades_symbol_status', 'symbol_id', 'status'),
        Index('ix_trades_entry_time', 'entry_time'),
    )


class Balance(Base):
    """Account balance model."""
    __tablename__ = 'balances'
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    asset = Column(String(10), nullable=False)
    free = Column(Float, nullable=False)
    used = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    exchange = relationship("Exchange")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('exchange_id', 'timestamp', 'asset', name='uq_balance'),
        Index('ix_balances_timestamp', 'timestamp'),
    )


class PerformanceMetric(Base):
    """Performance metrics model."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    equity = Column(Float, nullable=False)
    balance = Column(Float, nullable=False)
    open_trades = Column(Integer, nullable=False)
    daily_profit = Column(Float, nullable=True)
    daily_return = Column(Float, nullable=True)
    total_profit = Column(Float, nullable=True)
    total_return = Column(Float, nullable=True)
    drawdown = Column(Float, nullable=True)
    performance_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('ix_performance_metrics_timestamp', 'timestamp'),
    )

