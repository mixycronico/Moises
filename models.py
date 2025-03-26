"""
Modelos de base de datos para el sistema Genesis con integración de préstamos.

Este archivo define los modelos ORM SQLAlchemy para el sistema de préstamos
e inversionistas.
"""

import logging
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship, declarative_base

from flask_sqlalchemy import SQLAlchemy

# Configurar logging
logger = logging.getLogger('genesis_website')

# Inicializar SQLAlchemy
db = SQLAlchemy()

# Modelo de usuario
class User(db.Model):
    """Modelo de usuarios del sistema Genesis."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    role = Column(String(20), default='inversionista')  # inversionista, admin, super_admin
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    investors = relationship('Investor', back_populates='user')
    
    def __repr__(self):
        return f"<User {self.username} ({self.role})>"

# Modelo de inversionista
class Investor(db.Model):
    """Modelo de inversionista para el sistema Genesis."""
    __tablename__ = 'investors'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    balance = Column(Float, default=0.0)
    capital = Column(Float, default=0.0)  # Capital invertido inicialmente
    earnings = Column(Float, default=0.0)  # Ganancias acumuladas
    risk_level = Column(String(20), default='moderate')  # low, moderate, high
    category = Column(String(20), default='bronze')  # bronze, silver, gold, platinum
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    user = relationship('User', back_populates='investors')
    transactions = relationship('Transaction', back_populates='investor')
    loans = relationship('InvestorLoan', back_populates='investor')
    
    def __repr__(self):
        return f"<Investor {self.id} (User {self.user_id}) - Balance: ${self.balance:.2f}>"

# Modelo de transacción
class Transaction(db.Model):
    """Modelo de transacción para inversionistas."""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    investor_id = Column(Integer, ForeignKey('investors.id'), nullable=False)
    type = Column(String(20), nullable=False)  # deposit, withdrawal, profit, transfer
    amount = Column(Float, nullable=False)
    description = Column(String(200))
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='completed')  # pending, completed, failed
    reference_id = Column(String(100))  # ID de referencia externa
    
    # Relaciones
    investor = relationship('Investor', back_populates='transactions')
    
    def __repr__(self):
        return f"<Transaction {self.id} - {self.type} ${self.amount:.2f}>"

# Modelo de préstamo para inversionistas importado desde investor_loans
from investor_loans import InvestorLoan, LoanPayment