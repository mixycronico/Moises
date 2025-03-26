"""
Modelos de datos para el Proyecto Genesis.

Este módulo define los modelos de SQLAlchemy utilizados en el Proyecto Genesis.
"""

from datetime import datetime
from flask_login import UserMixin
from website.version2.app import db

class User(UserMixin, db.Model):
    """Modelo de usuario del sistema."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='investor')  # 'investor', 'admin', 'super_admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default='active')  # 'active', 'inactive', 'suspended'
    
    # Relaciones
    investments = db.relationship('Investment', back_populates='user', lazy='dynamic')
    transactions = db.relationship('Transaction', back_populates='user', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.username}>'

class Investment(db.Model):
    """Modelo de inversión realizada por un usuario."""
    __tablename__ = 'investments'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    asset_name = db.Column(db.String(100), nullable=False)
    asset_symbol = db.Column(db.String(20), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    current_value_usd = db.Column(db.Float, nullable=False)
    purchase_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='active')  # 'active', 'sold', 'pending'
    
    # Relaciones
    user = db.relationship('User', back_populates='investments')
    
    def __repr__(self):
        return f'<Investment {self.asset_symbol} {self.amount}>'

class Transaction(db.Model):
    """Modelo de transacción realizada por un usuario."""
    __tablename__ = 'transactions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)  # 'BUY', 'SELL', 'DEPOSIT', 'WITHDRAW'
    asset_name = db.Column(db.String(100), nullable=True)  # Puede ser nulo para depósitos/retiros
    asset_symbol = db.Column(db.String(20), nullable=True)  # Puede ser nulo para depósitos/retiros
    amount = db.Column(db.Float, nullable=False)
    price_usd = db.Column(db.Float, nullable=True)  # Precio unitario, nulo para depósitos/retiros
    total_usd = db.Column(db.Float, nullable=False)  # Monto total de la transacción
    date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='COMPLETED')  # 'COMPLETED', 'PENDING', 'FAILED', 'CANCELLED'
    
    # Relaciones
    user = db.relationship('User', back_populates='transactions')
    
    def __repr__(self):
        return f'<Transaction {self.transaction_type} {self.asset_symbol} {self.amount}>'

class SystemState(db.Model):
    """Modelo para almacenar el estado actual del sistema."""
    __tablename__ = 'system_state'
    
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(50), default='operational')
    memory_usage = db.Column(db.Float, default=0.0)
    cpu_usage = db.Column(db.Float, default=0.0)
    uptime_seconds = db.Column(db.Integer, default=0)
    active_users = db.Column(db.Integer, default=0)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SystemState {self.status}>'

class AIComponentState(db.Model):
    """Modelo para almacenar el estado de los componentes de IA."""
    __tablename__ = 'ai_component_state'
    
    id = db.Column(db.Integer, primary_key=True)
    component_name = db.Column(db.String(50), nullable=False)  # 'aetherion', 'deepseek', 'buddha', 'gabriel'
    status = db.Column(db.String(50), default='active')
    level = db.Column(db.Integer, default=1)
    efficiency = db.Column(db.Float, default=80.0)
    api_calls_today = db.Column(db.Integer, default=0)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<AIComponentState {self.component_name} {self.status}>'

class ConsciousnessState(db.Model):
    """Modelo para almacenar los estados de consciencia de Aetherion."""
    __tablename__ = 'consciousness_states'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    level = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text)
    associated_emotion = db.Column(db.String(50), nullable=True)
    activation_count = db.Column(db.Integer, default=0)
    
    def __repr__(self):
        return f'<ConsciousnessState {self.name} (Level {self.level})>'

class Memory(db.Model):
    """Modelo para almacenar las memorias del sistema Aetherion."""
    __tablename__ = 'memories'
    
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(20), default='short_term')  # short_term, long_term
    importance = db.Column(db.Float, default=0.5)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    context = db.Column(db.Text)
    
    def __repr__(self):
        return f'<Memory {self.id} ({self.type}, Importance: {self.importance})>'