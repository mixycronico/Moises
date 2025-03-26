"""
Sistema de préstamos para inversionistas.

Este módulo implementa la lógica para el sistema de préstamos de Proto Genesis,
permitiendo a los inversionistas con más de 3 meses obtener préstamos por
el 40% de su capital, con pagos automáticos del 30% de las ganancias diarias.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from models import db

# Configurar logging
logger = logging.getLogger('genesis_website')

# Modelo para préstamos
class InvestorLoan(db.Model):
    """Modelo de préstamo para inversionistas."""
    __tablename__ = 'investor_loans'
    
    id = Column(Integer, primary_key=True)
    investor_id = Column(Integer, ForeignKey('investors.id'), nullable=False)
    amount = Column(Float, nullable=False)
    remaining_amount = Column(Float, nullable=False)
    loan_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_payment_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relación con el inversionista
    investor = relationship('Investor', back_populates='loans')
    
    def __repr__(self):
        return f"<InvestorLoan id={self.id}, investor_id={self.investor_id}, amount={self.amount}, remaining={self.remaining_amount}>"

# Registro de pagos de préstamos
class LoanPayment(db.Model):
    """Modelo para registrar pagos de préstamos."""
    __tablename__ = 'loan_payments'
    
    id = Column(Integer, primary_key=True)
    loan_id = Column(Integer, ForeignKey('investor_loans.id'), nullable=False)
    amount = Column(Float, nullable=False)
    payment_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relación con el préstamo
    loan = relationship('InvestorLoan')
    
    def __repr__(self):
        return f"<LoanPayment id={self.id}, loan_id={self.loan_id}, amount={self.amount}>"

# Funciones para gestión de préstamos
def check_loan_eligibility(investor_id: int) -> Tuple[bool, str]:
    """
    Verificar si un inversionista es elegible para un préstamo.
    
    Args:
        investor_id: ID del inversionista
        
    Returns:
        Tupla (elegible, mensaje)
    """
    from models import Investor
    
    # Obtener datos del inversionista
    investor = Investor.query.get(investor_id)
    if not investor:
        return False, "Inversionista no encontrado"
    
    # Verificar si tiene préstamos activos
    active_loans = InvestorLoan.query.filter_by(investor_id=investor_id, is_active=True).first()
    if active_loans:
        return False, "Ya tiene un préstamo activo"
    
    # Verificar antigüedad mínima (3 meses)
    three_months_ago = datetime.utcnow() - timedelta(days=90)
    if investor.created_at > three_months_ago:
        days_remaining = (three_months_ago - investor.created_at).days
        return False, f"Debe tener al menos 3 meses como inversionista. Faltan {abs(days_remaining)} días."
    
    return True, "Elegible para préstamo"

def calculate_max_loan_amount(investor_id: int) -> float:
    """
    Calcular el monto máximo de préstamo (40% del capital).
    
    Args:
        investor_id: ID del inversionista
        
    Returns:
        Monto máximo de préstamo
    """
    from models import Investor
    
    investor = Investor.query.get(investor_id)
    if not investor:
        return 0.0
    
    # El 40% del capital invertido
    return investor.capital * 0.4

def create_loan(investor_id: int, amount: float) -> Tuple[bool, str, Optional[InvestorLoan]]:
    """
    Crear un nuevo préstamo para un inversionista.
    
    Args:
        investor_id: ID del inversionista
        amount: Monto solicitado
        
    Returns:
        Tupla (éxito, mensaje, préstamo)
    """
    from models import Investor
    
    # Verificar elegibilidad
    eligible, message = check_loan_eligibility(investor_id)
    if not eligible:
        return False, message, None
    
    # Verificar que el monto no exceda el máximo
    max_amount = calculate_max_loan_amount(investor_id)
    if amount > max_amount:
        return False, f"El monto solicitado excede el máximo permitido (40% del capital = ${max_amount:.2f})", None
    
    try:
        # Crear el préstamo
        loan = InvestorLoan(
            investor_id=investor_id,
            amount=amount,
            remaining_amount=amount
        )
        
        # Actualizar balance del inversionista
        investor = Investor.query.get(investor_id)
        investor.balance += amount
        
        # Guardar en la base de datos
        db.session.add(loan)
        db.session.commit()
        
        logger.info(f"Préstamo creado: ID={loan.id}, Inversionista={investor_id}, Monto=${amount:.2f}")
        return True, "Préstamo aprobado y depositado", loan
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error al crear préstamo: {str(e)}")
        return False, f"Error al procesar préstamo: {str(e)}", None

def process_loan_payment(loan_id: int, amount: float) -> Tuple[bool, str]:
    """
    Procesar un pago de préstamo.
    
    Args:
        loan_id: ID del préstamo
        amount: Monto del pago
        
    Returns:
        Tupla (éxito, mensaje)
    """
    try:
        # Obtener el préstamo
        loan = InvestorLoan.query.get(loan_id)
        if not loan:
            return False, "Préstamo no encontrado"
        
        if not loan.is_active:
            return False, "Este préstamo ya ha sido pagado completamente"
        
        # Limitar el pago al saldo restante
        if amount > loan.remaining_amount:
            amount = loan.remaining_amount
        
        # Registrar el pago
        payment = LoanPayment(
            loan_id=loan_id,
            amount=amount
        )
        
        # Actualizar saldo restante
        loan.remaining_amount -= amount
        loan.last_payment_date = datetime.utcnow()
        
        # Si el saldo llega a cero, marcar como pagado
        if loan.remaining_amount <= 0:
            loan.is_active = False
            loan.remaining_amount = 0
        
        # Guardar cambios
        db.session.add(payment)
        db.session.commit()
        
        logger.info(f"Pago procesado: Préstamo={loan_id}, Monto=${amount:.2f}, Saldo=${loan.remaining_amount:.2f}")
        
        if not loan.is_active:
            return True, "Préstamo pagado completamente"
        return True, f"Pago procesado. Saldo restante: ${loan.remaining_amount:.2f}"
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error al procesar pago: {str(e)}")
        return False, f"Error al procesar pago: {str(e)}"

def process_daily_loan_payments() -> Dict[str, Any]:
    """
    Procesar pagos diarios del 30% de las ganancias para todos los préstamos activos.
    Esta función debe ejecutarse diariamente mediante un trabajo programado.
    
    Returns:
        Diccionario con resultados del procesamiento
    """
    from models import Investor, Transaction
    
    results = {
        "total_processed": 0,
        "successful_payments": 0,
        "failed_payments": 0,
        "total_amount_paid": 0.0,
        "details": []
    }
    
    try:
        # Obtener todos los préstamos activos
        active_loans = InvestorLoan.query.filter_by(is_active=True).all()
        results["total_processed"] = len(active_loans)
        
        for loan in active_loans:
            try:
                # Obtener el inversionista
                investor = Investor.query.get(loan.investor_id)
                if not investor:
                    continue
                
                # Calcular ganancias del día
                today = datetime.utcnow().date()
                yesterday = today - timedelta(days=1)
                
                # Contar solo transacciones de ganancia
                daily_profit = Transaction.query.filter(
                    Transaction.investor_id == investor.id,
                    Transaction.type == 'profit',
                    Transaction.timestamp >= yesterday,
                    Transaction.timestamp < today
                ).with_entities(db.func.sum(Transaction.amount)).scalar() or 0
                
                # Calcular pago (30% de las ganancias diarias)
                payment_amount = daily_profit * 0.3
                
                # Si hay ganancias, procesar pago
                if payment_amount > 0:
                    # Limitar al saldo restante
                    if payment_amount > loan.remaining_amount:
                        payment_amount = loan.remaining_amount
                    
                    # Procesar pago
                    success, message = process_loan_payment(loan.id, payment_amount)
                    
                    if success:
                        results["successful_payments"] += 1
                        results["total_amount_paid"] += payment_amount
                    else:
                        results["failed_payments"] += 1
                    
                    results["details"].append({
                        "loan_id": loan.id,
                        "investor_id": loan.investor_id,
                        "payment_amount": payment_amount,
                        "success": success,
                        "message": message
                    })
            
            except Exception as e:
                results["failed_payments"] += 1
                results["details"].append({
                    "loan_id": loan.id,
                    "investor_id": loan.investor_id,
                    "error": str(e)
                })
                logger.error(f"Error procesando pago para préstamo {loan.id}: {str(e)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error en proceso de pagos diarios: {str(e)}")
        return {
            "error": str(e),
            "total_processed": 0,
            "successful_payments": 0,
            "failed_payments": 0
        }

def get_investor_loan_status(investor_id: int) -> Dict[str, Any]:
    """
    Obtener estado de préstamos de un inversionista.
    
    Args:
        investor_id: ID del inversionista
        
    Returns:
        Diccionario con estado de préstamos
    """
    from models import Investor
    
    try:
        # Verificar si el inversionista existe
        investor = Investor.query.get(investor_id)
        if not investor:
            return {"error": "Inversionista no encontrado"}
        
        # Verificar elegibilidad
        eligible, message = check_loan_eligibility(investor_id)
        max_amount = calculate_max_loan_amount(investor_id) if eligible else 0
        
        # Obtener préstamos activos
        active_loans = InvestorLoan.query.filter_by(
            investor_id=investor_id, 
            is_active=True
        ).all()
        
        # Obtener historial de préstamos
        loan_history = InvestorLoan.query.filter_by(
            investor_id=investor_id, 
            is_active=False
        ).all()
        
        # Formatear préstamos activos
        active_loan_data = []
        for loan in active_loans:
            payments = LoanPayment.query.filter_by(loan_id=loan.id).all()
            active_loan_data.append({
                "id": loan.id,
                "amount": loan.amount,
                "remaining": loan.remaining_amount,
                "date": loan.loan_date.strftime('%Y-%m-%d'),
                "last_payment": loan.last_payment_date.strftime('%Y-%m-%d'),
                "paid": loan.amount - loan.remaining_amount,
                "payments": [
                    {
                        "id": p.id,
                        "amount": p.amount,
                        "date": p.payment_date.strftime('%Y-%m-%d')
                    } for p in payments
                ]
            })
        
        # Formatear historial de préstamos
        history_data = []
        for loan in loan_history:
            payments = LoanPayment.query.filter_by(loan_id=loan.id).all()
            history_data.append({
                "id": loan.id,
                "amount": loan.amount,
                "date": loan.loan_date.strftime('%Y-%m-%d'),
                "paid_date": loan.last_payment_date.strftime('%Y-%m-%d'),
                "days_to_payoff": (loan.last_payment_date - loan.loan_date).days,
                "payments": len(payments)
            })
        
        # Determinar estado
        loan_state = "no_active_loan"
        if not eligible and active_loans:
            loan_state = "has_active_loan"
        elif not eligible and not active_loans:
            loan_state = "not_eligible"
        elif eligible:
            loan_state = "eligible"
        
        # Construir respuesta
        return {
            "eligible": eligible,
            "message": message,
            "max_amount": max_amount if eligible else 0,
            "state": loan_state,
            "active_loan": active_loan_data[0] if active_loan_data else None,
            "loan_history": history_data,
            "total_active_loans": len(active_loans),
            "total_completed_loans": len(loan_history),
            "investor_since": investor.created_at.strftime('%Y-%m-%d'),
            "days_as_investor": (datetime.utcnow() - investor.created_at).days
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estado de préstamos: {str(e)}")
        return {"error": f"Error al obtener estado de préstamos: {str(e)}"}