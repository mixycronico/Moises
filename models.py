from extensions import db
from datetime import datetime
from flask_login import UserMixin
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)

    @property
    def name(self):
        return f"{self.first_name or ''} {self.last_name or ''}".strip() or self.username

    @property
    def role(self):
        if self.username == 'mixycronico':
            return 'super_admin'
        elif self.is_admin:
            return 'admin'
        return 'inversionista'

    # Relaciones
    investors = relationship('Investor', back_populates='user', cascade="all, delete-orphan")

    def __repr__(self):
        return f'<User {self.username}>'

class Investor(db.Model):
    __tablename__ = 'investors'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    balance = Column(Float, default=0.0)
    capital = Column(Float, default=0.0)
    earnings = Column(Float, default=0.0)
    risk_level = Column(String(20), default='moderate')
    category = Column(String(20), default='bronze')
    created_at = Column(DateTime, default=datetime.utcnow)
    last_category_change = Column(DateTime)

    user = relationship('User', back_populates='investors')
    transactions = relationship('Transaction', back_populates='investor', cascade="all, delete-orphan")
    loans = relationship('Loan', back_populates='investor', cascade="all, delete-orphan")
    bonuses = relationship('Bonus', back_populates='investor', cascade="all, delete-orphan")
    commissions = relationship('Commission', back_populates='investor', cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Investor {self.id} (User: {self.user_id})>'

class Transaction(db.Model):
    __tablename__ = 'transactions'

    id = Column(Integer, primary_key=True)
    investor_id = Column(Integer, ForeignKey('investors.id'), nullable=False)
    type = Column(String(20), nullable=False)
    amount = Column(Float, nullable=False)
    description = Column(String(200))
    timestamp = Column(DateTime, default=datetime.utcnow)

    investor = relationship('Investor', back_populates='transactions')

    def __repr__(self):
        return f'<Transaction {self.id} ({self.type}: {self.amount})>'

class Loan(db.Model):
    __tablename__ = 'loans'

    id = Column(Integer, primary_key=True)
    investor_id = Column(Integer, ForeignKey('investors.id'), nullable=False)
    amount = Column(Float, nullable=False)
    remaining_amount = Column(Float, nullable=False)
    loan_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_payment_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    interest_rate = Column(Float, default=0.0)

    investor = relationship('Investor', back_populates='loans')
    payments = relationship('LoanPayment', back_populates='loan', cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Loan {self.id} (Amount: {self.amount}, Remaining: {self.remaining_amount})>'

class LoanPayment(db.Model):
    __tablename__ = 'loan_payments'

    id = Column(Integer, primary_key=True)
    loan_id = Column(Integer, ForeignKey('loans.id'), nullable=False)
    amount = Column(Float, nullable=False)
    payment_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    loan = relationship('Loan', back_populates='payments')

    def __repr__(self):
        return f'<LoanPayment {self.id} (Loan: {self.loan_id}, Amount: {self.amount})>'

class Bonus(db.Model):
    __tablename__ = 'bonuses'

    id = Column(Integer, primary_key=True)
    investor_id = Column(Integer, ForeignKey('investors.id'), nullable=False)
    amount = Column(Float, nullable=False)
    reason = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_claimed = Column(Boolean, default=False)
    claimed_at = Column(DateTime)

    investor = relationship('Investor', back_populates='bonuses')

    def __repr__(self):
        return f'<Bonus {self.id} (Investor: {self.investor_id}, Amount: {self.amount})>'

class Commission(db.Model):
    __tablename__ = 'commissions'

    id = Column(Integer, primary_key=True)
    investor_id = Column(Integer, ForeignKey('investors.id'), nullable=False)
    admin_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    amount = Column(Float, nullable=False)
    description = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='pending')
    processed_at = Column(DateTime)

    investor = relationship('Investor', back_populates='commissions')
    admin = relationship('User')

    def __repr__(self):
        return f'<Commission {self.id} (Investor: {self.investor_id}, Amount: {self.amount})>'

class IAMetric(db.Model):
    __tablename__ = 'ia_metrics'

    id = Column(Integer, primary_key=True)
    ia_name = Column(String(50), nullable=False)
    metric_type = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<IAMetric {self.id} ({self.ia_name}, {self.metric_type}: {self.value})>'

class IAInteraction(db.Model):
    __tablename__ = 'ia_interactions'

    id = Column(Integer, primary_key=True)
    ia_name = Column(String(50), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    emotion = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship('User')

    def __repr__(self):
        return f'<IAInteraction {self.id} ({self.ia_name}, User: {self.user_id})>'

class IADiaryEntry(db.Model):
    __tablename__ = 'ia_diary_entries'

    id = Column(Integer, primary_key=True)
    ia_name = Column(String(50), nullable=False)
    entry_date = Column(DateTime, default=datetime.utcnow)
    content = Column(Text, nullable=False)

    def __repr__(self):
        return f'<IADiaryEntry {self.id} ({self.ia_name}, Date: {self.entry_date})>'
