"""
Sistema de bonos para inversionistas.

Este módulo implementa la lógica para el sistema de bonos de Proto Genesis,
otorgando a los inversionistas con más de 3 meses bonos del 5-10% según
su categoría en días de rendimiento "excelente".
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
import json
import random

from flask import current_app
from sqlalchemy import func, desc
from models import db, Investor, Transaction, User

# Configurar logging
logger = logging.getLogger('genesis_website')

# Tasas de bonos por categoría (porcentaje)
BONUS_RATES = {
    'platinum': 10.0,
    'gold': 7.0,
    'silver': 5.0,
    'bronze': 3.0
}

# Nombre del tipo de transacción para bonos
BONUS_TRANSACTION_TYPE = 'bonus'

# Tipos de bonos
BONUS_TYPES = {
    'daily_excellent': 'Bono por rendimiento excelente diario',
    'monthly': 'Bono mensual fidelidad',
    'special': 'Bono especial Proto Genesis'
}

def calculate_monthly_bonus_rate(category: str) -> float:
    """
    Calcular la tasa de bono mensual según la categoría del inversionista.
    
    Args:
        category: Categoría del inversionista (platinum, gold, silver, bronze)
        
    Returns:
        Tasa de bono mensual (porcentaje)
    """
    category = category.lower()
    return BONUS_RATES.get(category, BONUS_RATES['bronze'])

def is_eligible_for_bonus(investor_id: int) -> Tuple[bool, str]:
    """
    Verificar si un inversionista es elegible para recibir bonos.
    
    Args:
        investor_id: ID del inversionista
        
    Returns:
        Tupla (elegible, mensaje)
    """
    try:
        # Obtener el inversionista
        investor = Investor.query.get(investor_id)
        
        if not investor:
            return False, "Inversionista no encontrado"
        
        # Verificar antigüedad (mínimo 3 meses)
        min_tenure = timedelta(days=90)  # 3 meses
        tenure = datetime.utcnow() - investor.created_at
        
        if tenure < min_tenure:
            days_remaining = (min_tenure - tenure).days
            return False, f"El inversionista necesita {days_remaining} días más para ser elegible para bonos"
        
        # Todo en orden, inversionista elegible
        return True, "Inversionista elegible para bonos"
        
    except Exception as e:
        logger.error(f"Error al verificar elegibilidad para bonos: {str(e)}")
        return False, f"Error al verificar elegibilidad: {str(e)}"

def check_daily_performance() -> bool:
    """
    Verificar si el rendimiento del día actual es "excelente".
    
    En una implementación real, esto consultaría los datos de rendimiento
    del portafolio global o específico. Para esta demo, usamos valores simulados.
    
    Returns:
        True si el rendimiento es excelente, False en caso contrario
    """
    try:
        # En una implementación real, aquí consultaríamos datos del portafolio
        # y aplicaríamos algoritmos para determinar si el rendimiento es excelente
        
        # Para demo, simulamos que 1 de cada 5 días tiene rendimiento excelente
        # En producción, esto se reemplazaría con análisis real de datos
        
        # Obtener resultados del último análisis de mercado (simulado)
        # Simulamos resultados con random pero en el sistema real usaríamos
        # DeepSeek y Buddha para análisis real
        
        # Análisis simulado del mercado
        market_conditions = {
            'volatility': random.uniform(0.5, 3.0),
            'trend_strength': random.uniform(0.1, 1.0),
            'market_sentiment': random.uniform(-1.0, 1.0)
        }
        
        # Simulación de análisis del portafolio
        portfolio_metrics = {
            'daily_return': random.uniform(-0.5, 2.5),
            'alpha': random.uniform(-0.2, 0.5),
            'sharpe_ratio': random.uniform(0.2, 1.8)
        }
        
        # Determinar si el rendimiento es excelente basado en métricas
        # En el mundo real, esto sería un algoritmo sofisticado
        is_excellent = (
            portfolio_metrics['daily_return'] > 1.2 and
            portfolio_metrics['alpha'] > 0.2 and
            portfolio_metrics['sharpe_ratio'] > 1.0
        )
        
        # Registrar la decisión para auditoría
        logger.info(
            f"Análisis rendimiento diario: excelente={is_excellent}, "
            f"return={portfolio_metrics['daily_return']:.2f}%, "
            f"alpha={portfolio_metrics['alpha']:.2f}, "
            f"sharpe={portfolio_metrics['sharpe_ratio']:.2f}"
        )
        
        return is_excellent
        
    except Exception as e:
        logger.error(f"Error al verificar rendimiento diario: {str(e)}")
        return False

def apply_daily_bonus_excellent_performance() -> Dict[str, Any]:
    """
    Aplicar bonos diarios a inversionistas en días de rendimiento excelente.
    
    Returns:
        Diccionario con resultados del procesamiento
    """
    results = {
        'fecha': datetime.utcnow().isoformat(),
        'rendimiento_excelente': False,
        'inversionistas_procesados': 0,
        'bonos_aplicados': 0,
        'total_bonos': 0.0,
        'errores': 0,
        'detalle': []
    }
    
    try:
        # Verificar si hoy es día de rendimiento excelente
        is_excellent_day = check_daily_performance()
        results['rendimiento_excelente'] = is_excellent_day
        
        if not is_excellent_day:
            return results
        
        # Si es día excelente, procesar bonos
        investors = Investor.query.all()
        
        for investor in investors:
            try:
                # Verificar elegibilidad
                is_eligible, message = is_eligible_for_bonus(investor.id)
                
                if not is_eligible:
                    results['detalle'].append({
                        'investor_id': investor.id,
                        'elegible': False,
                        'mensaje': message
                    })
                    continue
                
                # Calcular bono según categoría
                bonus_rate = calculate_monthly_bonus_rate(investor.category)
                bonus_amount = investor.capital * (bonus_rate / 100.0)
                
                # Registrar transacción de bono
                transaction = Transaction(
                    investor_id=investor.id,
                    type=BONUS_TRANSACTION_TYPE,
                    amount=bonus_amount,
                    description=BONUS_TYPES['daily_excellent'],
                    status='completed'
                )
                
                # Actualizar balance y ganancias
                investor.balance += bonus_amount
                investor.earnings += bonus_amount
                
                # Guardar cambios
                db.session.add(transaction)
                db.session.commit()
                
                # Actualizar resultados
                results['bonos_aplicados'] += 1
                results['total_bonos'] += bonus_amount
                
                results['detalle'].append({
                    'investor_id': investor.id,
                    'elegible': True,
                    'categoria': investor.category,
                    'tasa_bono': bonus_rate,
                    'monto_bono': bonus_amount,
                    'mensaje': 'Bono aplicado correctamente'
                })
                
                logger.info(f"Bono diario aplicado: Inversionista={investor.id}, Categoría={investor.category}, Monto={bonus_amount:.2f}")
                
            except Exception as e:
                results['errores'] += 1
                logger.error(f"Error al procesar bono para inversionista {investor.id}: {str(e)}")
                results['detalle'].append({
                    'investor_id': investor.id,
                    'error': str(e)
                })
            
            finally:
                results['inversionistas_procesados'] += 1
        
        return results
        
    except Exception as e:
        logger.error(f"Error en proceso de bonos diarios: {str(e)}")
        return {
            'error': str(e),
            'inversionistas_procesados': results['inversionistas_procesados'],
            'bonos_aplicados': results['bonos_aplicados']
        }

def apply_monthly_bonus() -> Dict[str, Any]:
    """
    Aplicar bonos mensuales a todos los inversionistas elegibles.
    Esta función debe ejecutarse una vez al mes mediante un trabajo programado.
    
    Returns:
        Diccionario con resultados del procesamiento
    """
    results = {
        'fecha': datetime.utcnow().isoformat(),
        'inversionistas_procesados': 0,
        'bonos_aplicados': 0,
        'total_bonos': 0.0,
        'errores': 0,
        'detalle': []
    }
    
    try:
        # Obtener todos los inversionistas
        investors = Investor.query.all()
        
        for investor in investors:
            try:
                # Verificar elegibilidad
                is_eligible, message = is_eligible_for_bonus(investor.id)
                
                if not is_eligible:
                    results['detalle'].append({
                        'investor_id': investor.id,
                        'elegible': False,
                        'mensaje': message
                    })
                    continue
                
                # Calcular bono según categoría (tasa mensual es 1/4 de la tasa diaria)
                bonus_rate = calculate_monthly_bonus_rate(investor.category) / 4
                bonus_amount = investor.capital * (bonus_rate / 100.0)
                
                # Registrar transacción de bono
                transaction = Transaction(
                    investor_id=investor.id,
                    type=BONUS_TRANSACTION_TYPE,
                    amount=bonus_amount,
                    description=BONUS_TYPES['monthly'],
                    status='completed'
                )
                
                # Actualizar balance y ganancias
                investor.balance += bonus_amount
                investor.earnings += bonus_amount
                
                # Guardar cambios
                db.session.add(transaction)
                db.session.commit()
                
                # Actualizar resultados
                results['bonos_aplicados'] += 1
                results['total_bonos'] += bonus_amount
                
                results['detalle'].append({
                    'investor_id': investor.id,
                    'elegible': True,
                    'categoria': investor.category,
                    'tasa_bono': bonus_rate,
                    'monto_bono': bonus_amount,
                    'mensaje': 'Bono mensual aplicado correctamente'
                })
                
                logger.info(f"Bono mensual aplicado: Inversionista={investor.id}, Categoría={investor.category}, Monto={bonus_amount:.2f}")
                
            except Exception as e:
                results['errores'] += 1
                logger.error(f"Error al procesar bono mensual para inversionista {investor.id}: {str(e)}")
                results['detalle'].append({
                    'investor_id': investor.id,
                    'error': str(e)
                })
            
            finally:
                results['inversionistas_procesados'] += 1
        
        return results
        
    except Exception as e:
        logger.error(f"Error en proceso de bonos mensuales: {str(e)}")
        return {
            'error': str(e),
            'inversionistas_procesados': results['inversionistas_procesados'],
            'bonos_aplicados': results['bonos_aplicados']
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
        # Verificar que el inversionista existe
        investor = Investor.query.get(investor_id)
        
        if not investor:
            return {
                'success': False,
                'message': 'Inversionista no encontrado',
                'data': None
            }
        
        # Obtener transacciones de tipo bono
        bonus_transactions = Transaction.query.filter_by(
            investor_id=investor_id,
            type=BONUS_TRANSACTION_TYPE
        ).order_by(Transaction.timestamp.desc()).all()
        
        # Formatear resultados
        bonuses = []
        total_bonus = 0.0
        
        for tx in bonus_transactions:
            bonuses.append({
                'id': tx.id,
                'fecha': tx.timestamp.isoformat(),
                'monto': tx.amount,
                'descripcion': tx.description,
                'estado': tx.status
            })
            total_bonus += tx.amount
        
        # Agrupar por tipo de bono
        bonus_by_type = {}
        
        for bonus_type, description in BONUS_TYPES.items():
            type_transactions = [tx for tx in bonus_transactions if tx.description == description]
            type_total = sum(tx.amount for tx in type_transactions)
            
            bonus_by_type[bonus_type] = {
                'descripcion': description,
                'total': type_total,
                'cantidad': len(type_transactions)
            }
        
        return {
            'success': True,
            'message': 'Historial de bonos obtenido correctamente',
            'data': {
                'inversionista': {
                    'id': investor.id,
                    'categoria': investor.category,
                    'tasa_actual': calculate_monthly_bonus_rate(investor.category)
                },
                'bonos': bonuses,
                'resumen': {
                    'total_bonos': total_bonus,
                    'cantidad_bonos': len(bonuses),
                    'por_tipo': bonus_by_type
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error al obtener historial de bonos: {str(e)}")
        return {
            'success': False,
            'message': f"Error al obtener historial de bonos: {str(e)}",
            'data': None
        }