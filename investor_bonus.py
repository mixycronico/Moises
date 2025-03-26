"""
Sistema de bonos para inversionistas.

Este módulo implementa la lógica para el sistema de bonos de Proto Genesis,
otorgando a los inversionistas con más de 3 meses bonos del 5-10% según
su categoría en días de rendimiento "excelente".
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List
import random

from models import db, Investor, Transaction

# Configurar logging
logger = logging.getLogger('genesis_website')

# Tasas de bonos por categoría (porcentaje anualizado)
BONUS_RATES = {
    'platinum': 0.07,  # 7% anual
    'gold': 0.05,      # 5% anual
    'silver': 0.03,    # 3% anual
    'bronze': 0.02     # 2% anual
}

# Umbral de rendimiento para considerar un día como "excelente"
EXCELLENT_PERFORMANCE_THRESHOLD = 0.015  # 1.5% diario

def calculate_monthly_bonus_rate(category: str) -> float:
    """
    Calcular la tasa de bono mensual según la categoría del inversionista.
    
    Args:
        category: Categoría del inversionista (platinum, gold, silver, bronze)
        
    Returns:
        Tasa de bono mensual (porcentaje)
    """
    annual_rate = BONUS_RATES.get(category.lower(), BONUS_RATES['bronze'])
    monthly_rate = annual_rate / 12
    return monthly_rate

def is_eligible_for_bonus(investor_id: int) -> Tuple[bool, str]:
    """
    Verificar si un inversionista es elegible para recibir bonos.
    
    Args:
        investor_id: ID del inversionista
        
    Returns:
        Tupla (elegible, mensaje)
    """
    # Obtener datos del inversionista
    investor = Investor.query.get(investor_id)
    if not investor:
        return False, "Inversionista no encontrado"
    
    # Verificar antigüedad mínima (3 meses)
    three_months_ago = datetime.utcnow() - timedelta(days=90)
    if investor.created_at > three_months_ago:
        days_remaining = (three_months_ago - investor.created_at).days
        return False, f"Debe tener al menos 3 meses como inversionista. Faltan {abs(days_remaining)} días."
    
    return True, "Elegible para bonos"

def check_daily_performance() -> bool:
    """
    Verificar si el rendimiento del día actual es "excelente".
    
    En una implementación real, esto consultaría los datos de rendimiento
    del portafolio global o específico. Para esta demo, usamos valores simulados.
    
    Returns:
        True si el rendimiento es excelente, False en caso contrario
    """
    # Simulación de rendimiento para la demo
    # En una implementación real, esto consultaría datos reales
    daily_return = random.uniform(-0.02, 0.03)
    is_excellent = daily_return >= EXCELLENT_PERFORMANCE_THRESHOLD
    
    logger.info(f"Rendimiento diario: {daily_return:.2%}, ¿Es excelente? {is_excellent}")
    return is_excellent

def apply_daily_bonus_excellent_performance() -> Dict[str, Any]:
    """
    Aplicar bonos diarios a inversionistas en días de rendimiento excelente.
    
    Returns:
        Diccionario con resultados del procesamiento
    """
    results = {
        "is_excellent_day": False,
        "bonuses_applied": 0,
        "total_bonus_amount": 0.0,
        "details": []
    }
    
    # Verificar si es un día de rendimiento excelente
    is_excellent = check_daily_performance()
    results["is_excellent_day"] = is_excellent
    
    if not is_excellent:
        logger.info("No es un día de rendimiento excelente. No se aplican bonos.")
        return results
    
    try:
        # Obtener todos los inversionistas elegibles (más de 3 meses)
        three_months_ago = datetime.utcnow() - timedelta(days=90)
        eligible_investors = Investor.query.filter(Investor.created_at <= three_months_ago).all()
        
        for investor in eligible_investors:
            try:
                # Obtener tasa de bono según categoría
                bonus_rate = BONUS_RATES.get(investor.category.lower(), BONUS_RATES['bronze'])
                
                # Calcular bono (entre 5-10% del rendimiento diario)
                # El porcentaje exacto se determina según la categoría y un factor aleatorio
                # para simular variabilidad en los días excelentes
                bonus_percentage = random.uniform(0.05, 0.10)
                bonus_rate_adjusted = bonus_rate * bonus_percentage * 365  # Ajustar a equivalente diario
                
                # Calcular monto del bono
                bonus_amount = investor.capital * bonus_rate_adjusted
                
                # Crear transacción de bono
                bonus_transaction = Transaction(
                    investor_id=investor.id,
                    type='bonus',
                    amount=bonus_amount,
                    description=f"Bono por rendimiento excelente ({bonus_percentage:.1%})",
                    status='completed'
                )
                
                # Actualizar balance
                investor.balance += bonus_amount
                investor.earnings += bonus_amount
                
                # Guardar cambios
                db.session.add(bonus_transaction)
                
                # Registrar resultado
                results["bonuses_applied"] += 1
                results["total_bonus_amount"] += bonus_amount
                results["details"].append({
                    "investor_id": investor.id,
                    "category": investor.category,
                    "bonus_amount": bonus_amount,
                    "bonus_rate": bonus_rate_adjusted
                })
                
                logger.info(f"Bono aplicado: Inversionista={investor.id}, Categoría={investor.category}, Monto=${bonus_amount:.2f}")
                
            except Exception as e:
                logger.error(f"Error al aplicar bono para inversionista {investor.id}: {str(e)}")
        
        # Confirmar cambios en la base de datos
        db.session.commit()
        return results
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error en proceso de bonos diarios: {str(e)}")
        return {
            "error": str(e),
            "is_excellent_day": False,
            "bonuses_applied": 0,
            "total_bonus_amount": 0
        }

def apply_monthly_bonus() -> Dict[str, Any]:
    """
    Aplicar bonos mensuales a todos los inversionistas elegibles.
    Esta función debe ejecutarse una vez al mes mediante un trabajo programado.
    
    Returns:
        Diccionario con resultados del procesamiento
    """
    results = {
        "bonuses_applied": 0,
        "total_bonus_amount": 0.0,
        "details": []
    }
    
    try:
        # Obtener todos los inversionistas elegibles (más de 3 meses)
        three_months_ago = datetime.utcnow() - timedelta(days=90)
        eligible_investors = Investor.query.filter(Investor.created_at <= three_months_ago).all()
        
        for investor in eligible_investors:
            try:
                # Calcular tasa de bono mensual según categoría
                monthly_rate = calculate_monthly_bonus_rate(investor.category)
                
                # Calcular monto del bono
                bonus_amount = investor.capital * monthly_rate
                
                # Crear transacción de bono
                bonus_transaction = Transaction(
                    investor_id=investor.id,
                    type='bonus',
                    amount=bonus_amount,
                    description=f"Bono mensual ({investor.category.capitalize()})",
                    status='completed'
                )
                
                # Actualizar balance
                investor.balance += bonus_amount
                investor.earnings += bonus_amount
                
                # Guardar cambios
                db.session.add(bonus_transaction)
                
                # Registrar resultado
                results["bonuses_applied"] += 1
                results["total_bonus_amount"] += bonus_amount
                results["details"].append({
                    "investor_id": investor.id,
                    "category": investor.category,
                    "bonus_amount": bonus_amount,
                    "monthly_rate": monthly_rate
                })
                
                logger.info(f"Bono mensual aplicado: Inversionista={investor.id}, Categoría={investor.category}, Monto=${bonus_amount:.2f}")
                
            except Exception as e:
                logger.error(f"Error al aplicar bono mensual para inversionista {investor.id}: {str(e)}")
        
        # Confirmar cambios en la base de datos
        db.session.commit()
        return results
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error en proceso de bonos mensuales: {str(e)}")
        return {
            "error": str(e),
            "bonuses_applied": 0,
            "total_bonus_amount": 0
        }

def get_investor_bonus_history(investor_id: int) -> Dict[str, Any]:
    """
    Obtener historial de bonos de un inversionista.
    
    Args:
        investor_id: ID del inversionista
        
    Returns:
        Diccionario con historial de bonos
    """
    try:
        # Verificar si el inversionista existe
        investor = Investor.query.get(investor_id)
        if not investor:
            return {"error": "Inversionista no encontrado"}
        
        # Verificar elegibilidad
        eligible, message = is_eligible_for_bonus(investor_id)
        
        # Obtener transacciones de tipo 'bonus'
        bonus_transactions = Transaction.query.filter_by(
            investor_id=investor_id,
            type='bonus'
        ).order_by(Transaction.timestamp.desc()).all()
        
        # Formatear transacciones
        transactions_data = []
        for tx in bonus_transactions:
            transactions_data.append({
                "id": tx.id,
                "amount": tx.amount,
                "description": tx.description,
                "date": tx.timestamp.strftime('%Y-%m-%d'),
                "status": tx.status
            })
        
        # Calcular estadísticas
        total_bonus = sum(tx.amount for tx in bonus_transactions)
        monthly_rate = calculate_monthly_bonus_rate(investor.category)
        expected_monthly = investor.capital * monthly_rate
        
        # Construir respuesta
        return {
            "eligible": eligible,
            "message": message,
            "investor_category": investor.category,
            "annual_rate": BONUS_RATES.get(investor.category.lower(), BONUS_RATES['bronze']),
            "monthly_rate": monthly_rate,
            "bonus_history": transactions_data,
            "total_bonus_received": total_bonus,
            "expected_monthly_bonus": expected_monthly,
            "next_bonus_date": (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1).strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        logger.error(f"Error al obtener historial de bonos: {str(e)}")
        return {"error": f"Error al obtener historial de bonos: {str(e)}"}