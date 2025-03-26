"""
Modelos de base de datos para el Proyecto Genesis.

Este módulo define los modelos de datos utilizados por la aplicación web.
"""

from datetime import datetime
from app import db
from flask_login import UserMixin
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text, Enum
from sqlalchemy.orm import relationship
import enum

class User(UserMixin, db.Model):
    """Modelo para usuarios del sistema."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(20), default='investor')  # investor, admin, super_admin
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relaciones
    investments = relationship('Investment', back_populates='user')
    transactions = relationship('Transaction', back_populates='user')
    
    def __repr__(self):
        return f'<User {self.username}>'

class Investment(db.Model):
    """Modelo para inversiones de usuarios."""
    __tablename__ = 'investments'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    asset_name = Column(String(100), nullable=False)
    asset_symbol = Column(String(20), nullable=False)
    amount = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=False)  # Precio promedio de compra
    current_price = Column(Float)  # Precio actual (actualizado periódicamente)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    user = relationship('User', back_populates='investments')
    
    def __repr__(self):
        return f'<Investment {self.asset_symbol} {self.amount}>'

class TransactionType(enum.Enum):
    """Tipos de transacciones disponibles."""
    BUY = 'BUY'
    SELL = 'SELL'
    DEPOSIT = 'DEPOSIT'
    WITHDRAW = 'WITHDRAW'

class TransactionStatus(enum.Enum):
    """Estados posibles para una transacción."""
    PENDING = 'PENDING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'

class Transaction(db.Model):
    """Modelo para transacciones de usuarios."""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    type = Column(Enum(TransactionType), nullable=False)
    asset_name = Column(String(100))  # Nullable para depósitos/retiros
    asset_symbol = Column(String(20))  # Nullable para depósitos/retiros
    amount = Column(Float, nullable=False)
    price = Column(Float)  # Precio unitario (nullable para depósitos/retiros)
    total = Column(Float, nullable=False)  # Total en USD
    status = Column(Enum(TransactionStatus), default=TransactionStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)
    
    # Relaciones
    user = relationship('User', back_populates='transactions')
    
    def __repr__(self):
        return f'<Transaction {self.type.value} {self.asset_symbol} {self.amount}>'

class SystemState(db.Model):
    """Modelo para almacenar el estado del sistema."""
    __tablename__ = 'system_state'
    
    id = Column(Integer, primary_key=True)
    component_name = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False, default='active')
    last_update = Column(DateTime, default=datetime.utcnow)
    performance_metric = Column(Float)
    details = Column(Text)
    
    def __repr__(self):
        return f'<SystemState {self.component_name} {self.status}>'

class AetherionState(db.Model):
    """Modelo para almacenar el estado de conciencia de Aetherion."""
    __tablename__ = 'aetherion_state'
    
    id = Column(Integer, primary_key=True)
    consciousness_level = Column(Integer, default=1)
    energy_level = Column(Float, default=1.0)
    dominant_emotion = Column(String(50), default='neutral')
    adaptation_count = Column(Integer, default=0)
    last_interaction = Column(DateTime, default=datetime.utcnow)
    state_description = Column(Text)
    
    def __repr__(self):
        return f'<AetherionState level={self.consciousness_level} emotion={self.dominant_emotion}>'

class ConsciousnessLog(db.Model):
    """Modelo para registrar eventos importantes de la conciencia de Aetherion."""
    __tablename__ = 'consciousness_logs'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    importance = Column(Integer, default=1)  # 1-10, donde 10 es máxima importancia
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ConsciousnessLog {self.event_type} importance={self.importance}>'