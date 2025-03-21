"""
Modelos de base de datos para el modo Paper Trading.

Este módulo define las tablas y relaciones necesarias para el sistema
de paper trading, que permite simular operaciones sin usar fondos reales.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    ForeignKey, Enum, Text, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class PaperTradingAccount(Base):
    """Cuenta de paper trading."""
    __tablename__ = 'paper_trading_accounts'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    initial_balance_usd = Column(Float, nullable=False, default=10000.0)
    is_active = Column(Boolean, default=True)
    
    # Relaciones
    balances = relationship("PaperTradingBalance", back_populates="account", cascade="all, delete-orphan")
    orders = relationship("PaperTradingOrder", back_populates="account", cascade="all, delete-orphan")
    trades = relationship("PaperTradingTrade", back_populates="account", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<PaperTradingAccount(id={self.id}, name='{self.name}', user_id={self.user_id})>"


class PaperTradingBalance(Base):
    """Saldo de activos en una cuenta de paper trading."""
    __tablename__ = 'paper_trading_balances'
    
    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey('paper_trading_accounts.id'), nullable=False)
    asset = Column(String(20), nullable=False)  # Ej: BTC, ETH, USDT
    free = Column(Float, nullable=False, default=0.0)  # Saldo disponible
    locked = Column(Float, nullable=False, default=0.0)  # Saldo bloqueado en órdenes
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    account = relationship("PaperTradingAccount", back_populates="balances")
    
    # Restricciones
    __table_args__ = (
        UniqueConstraint('account_id', 'asset', name='uix_account_asset'),
    )
    
    def __repr__(self):
        return f"<PaperTradingBalance(account_id={self.account_id}, asset='{self.asset}', free={self.free}, locked={self.locked})>"


class PaperTradingOrder(Base):
    """Orden en el sistema de paper trading."""
    __tablename__ = 'paper_trading_orders'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    account_id = Column(Integer, ForeignKey('paper_trading_accounts.id'), nullable=False)
    symbol = Column(String(20), nullable=False)  # Ej: BTC/USDT
    side = Column(Enum('buy', 'sell', name='order_side'), nullable=False)
    type = Column(Enum('limit', 'market', 'stop_loss', 'take_profit', name='order_type'), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=True)  # Null para órdenes market
    status = Column(Enum('pending', 'open', 'closed', 'canceled', name='order_status'), nullable=False, default='open')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    filled_quantity = Column(Float, nullable=False, default=0.0)
    average_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)  # Para órdenes stop
    is_test = Column(Boolean, default=False)  # Indica si es una orden de prueba
    
    # Relaciones
    account = relationship("PaperTradingAccount", back_populates="orders")
    trades = relationship("PaperTradingTrade", back_populates="order", cascade="all, delete-orphan")
    
    # Índices
    __table_args__ = (
        Index('idx_orders_account_symbol', 'account_id', 'symbol'),
        Index('idx_orders_status', 'status'),
    )
    
    def __repr__(self):
        return f"<PaperTradingOrder(order_id='{self.order_id}', symbol='{self.symbol}', side='{self.side}', status='{self.status}')>"


class PaperTradingTrade(Base):
    """Operación ejecutada en el sistema de paper trading."""
    __tablename__ = 'paper_trading_trades'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    order_id = Column(String(36), ForeignKey('paper_trading_orders.order_id'), nullable=False)
    account_id = Column(Integer, ForeignKey('paper_trading_accounts.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(Enum('buy', 'sell', name='trade_side'), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, nullable=False, default=0.0)
    commission_asset = Column(String(20), nullable=False, default='USDT')
    timestamp = Column(DateTime, default=datetime.utcnow)
    strategy_id = Column(Integer, nullable=True)  # ID de la estrategia que generó la operación
    
    # Relaciones
    account = relationship("PaperTradingAccount", back_populates="trades")
    order = relationship("PaperTradingOrder", back_populates="trades")
    
    # Índices
    __table_args__ = (
        Index('idx_trades_account_symbol', 'account_id', 'symbol'),
        Index('idx_trades_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<PaperTradingTrade(trade_id='{self.trade_id}', symbol='{self.symbol}', side='{self.side}', price={self.price})>"


class MarketData(Base):
    """Datos de mercado históricos para paper trading."""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)  # Ej: 1m, 5m, 1h, 1d
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    source = Column(String(20), nullable=False, default='binance')  # Fuente de los datos (ej: 'binance', 'testnet')
    
    # Índices y restricciones
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'timestamp', 'source', name='uix_market_data'),
        Index('idx_market_data_symbol_timeframe', 'symbol', 'timeframe'),
        Index('idx_market_data_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timeframe='{self.timeframe}', timestamp='{self.timestamp}')>"


class PaperTradingSettings(Base):
    """Configuración para el sistema de paper trading."""
    __tablename__ = 'paper_trading_settings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    account_id = Column(Integer, ForeignKey('paper_trading_accounts.id'), nullable=False)
    commission_rate = Column(Float, nullable=False, default=0.001)  # 0.1% por defecto
    slippage_factor = Column(Float, nullable=False, default=0.0005)  # 0.05% por defecto
    simulate_latency = Column(Boolean, default=False)
    latency_ms = Column(Integer, default=200)  # Latencia simulada en milisegundos
    price_data_source = Column(String(20), nullable=False, default='testnet')  # 'testnet', 'live', 'historical'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<PaperTradingSettings(account_id={self.account_id}, commission_rate={self.commission_rate})>"