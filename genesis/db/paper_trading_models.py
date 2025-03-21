"""
Modelos de base de datos para Paper Trading.

Este módulo define los modelos utilizados para el modo Paper Trading
del sistema Genesis, permitiendo simular operaciones con datos reales
sin ejecutar órdenes reales en los exchanges.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from genesis.db.models import Base

class PaperTradingAccount(Base):
    """
    Cuenta de Paper Trading para simular operaciones.
    
    Esta tabla almacena las cuentas virtuales para paper trading,
    incluyendo el balance inicial y la configuración.
    """
    __tablename__ = 'paper_trading_accounts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)  # Puede ser nulo para cuentas de sistema/test
    name = Column(String(100), nullable=False)
    description = Column(String(255), nullable=True)
    initial_balance_usd = Column(Float, nullable=False, default=10000.0)
    current_balance_usd = Column(Float, nullable=False, default=10000.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=True)
    config = Column(JSON, nullable=True)  # Configuración adicional como fees, slippage, etc.
    
    # Relaciones
    balances = relationship("PaperAssetBalance", back_populates="account", cascade="all, delete-orphan")
    orders = relationship("PaperOrder", back_populates="account", cascade="all, delete-orphan")
    trades = relationship("PaperTrade", back_populates="account", cascade="all, delete-orphan")


class PaperAssetBalance(Base):
    """
    Balance de un activo específico en una cuenta de Paper Trading.
    
    Esta tabla almacena los balances de cada activo (BTC, ETH, etc.)
    para cada cuenta de paper trading.
    """
    __tablename__ = 'paper_asset_balances'
    
    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey('paper_trading_accounts.id'), nullable=False)
    asset = Column(String(20), nullable=False)  # BTC, ETH, USDT, etc.
    total = Column(Float, nullable=False, default=0.0)
    available = Column(Float, nullable=False, default=0.0)
    locked = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    account = relationship("PaperTradingAccount", back_populates="balances")
    
    __table_args__ = (
        # Índice compuesto para búsquedas rápidas por cuenta y activo
        {'sqlite_autoincrement': True},
    )


class PaperOrder(Base):
    """
    Orden en el sistema de Paper Trading.
    
    Esta tabla almacena las órdenes simuladas, con su estado y detalles.
    """
    __tablename__ = 'paper_orders'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), nullable=False, unique=True)  # ID único de la orden
    account_id = Column(Integer, ForeignKey('paper_trading_accounts.id'), nullable=False)
    symbol = Column(String(20), nullable=False)  # BTC/USDT, ETH/USDT, etc.
    order_type = Column(String(20), nullable=False)  # limit, market, stop_loss, etc.
    side = Column(String(10), nullable=False)  # buy, sell
    price = Column(Float, nullable=True)  # Null para órdenes market
    amount = Column(Float, nullable=False)
    filled = Column(Float, nullable=False, default=0.0)
    remaining = Column(Float, nullable=False)
    status = Column(String(20), nullable=False)  # open, closed, canceled, expired
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    extra = Column(JSON, nullable=True)  # Información adicional como stop price, condiciones, etc.
    
    # Relaciones
    account = relationship("PaperTradingAccount", back_populates="orders")
    trades = relationship("PaperTrade", back_populates="order", cascade="all, delete-orphan")
    
    __table_args__ = (
        # Índices para búsquedas comunes
        {'sqlite_autoincrement': True},
    )


class PaperTrade(Base):
    """
    Operación ejecutada en el sistema de Paper Trading.
    
    Esta tabla almacena las operaciones (trades) que resultan
    de la ejecución (total o parcial) de órdenes simuladas.
    """
    __tablename__ = 'paper_trades'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(50), nullable=False, unique=True)  # ID único del trade
    account_id = Column(Integer, ForeignKey('paper_trading_accounts.id'), nullable=False)
    order_id = Column(String(50), ForeignKey('paper_orders.order_id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # buy, sell
    price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)  # price * amount
    fee = Column(Float, nullable=False, default=0.0)
    fee_currency = Column(String(10), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    strategy_id = Column(String(50), nullable=True)  # ID de la estrategia que generó el trade
    extra = Column(JSON, nullable=True)  # Información adicional
    
    # Relaciones
    account = relationship("PaperTradingAccount", back_populates="trades")
    order = relationship("PaperOrder", back_populates="trades")
    
    __table_args__ = (
        # Índices para búsquedas comunes
        {'sqlite_autoincrement': True},
    )


class PaperBalanceSnapshot(Base):
    """
    Instantánea periódica del balance de una cuenta de Paper Trading.
    
    Esta tabla almacena instantáneas del balance total en diferentes 
    momentos para análisis de rendimiento y seguimiento.
    """
    __tablename__ = 'paper_balance_snapshots'
    
    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey('paper_trading_accounts.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    balance_usd = Column(Float, nullable=False)  # Balance total en USD
    assets = Column(JSON, nullable=False)  # Desglose de activos en ese momento
    daily_pnl = Column(Float, nullable=True)  # Ganancia/pérdida diaria si está disponible
    total_pnl = Column(Float, nullable=True)  # Ganancia/pérdida total desde inicio
    total_pnl_pct = Column(Float, nullable=True)  # Porcentaje de ganancia/pérdida total
    
    __table_args__ = (
        # Índice para búsquedas por cuenta y fecha
        {'sqlite_autoincrement': True},
    )