"""
Rutas API para el sistema de bonos de Proto Genesis.

Este módulo implementa los endpoints necesarios para administrar bonos
a inversionistas después de 3 meses en el sistema, con reportes para
administradores y inversionistas.
"""

import logging
from flask import Blueprint, request, jsonify, session, current_app
from datetime import datetime

from models import db, Investor
from investor_bonus import (
    apply_daily_bonus_excellent_performance,
    apply_monthly_bonus,
    get_investor_bonus_history,
    calculate_monthly_bonus_rate
)

# Configurar logging
logger = logging.getLogger('genesis_website')

# Crear Blueprint para las rutas de bonos
bonus_bp = Blueprint('bonus', __name__)

@bonus_bp.route('/api/bonus/status', methods=['GET'])
def bonus_status():
    """Obtener estado de bonos del inversionista actual."""
    # Verificar si hay usuario y inversionista en sesión
    if 'user_id' not in session or 'investor_id' not in session:
        return jsonify({
            'success': False,
            'message': 'No hay una sesión activa'
        }), 401
    
    investor_id = session['investor_id']
    result = get_investor_bonus_history(investor_id)
    
    return jsonify(result)

@bonus_bp.route('/api/bonus/simulate', methods=['GET'])
def simulate_bonus():
    """Simular cálculo de bonos para el inversionista actual."""
    # Verificar si hay usuario y inversionista en sesión
    if 'user_id' not in session or 'investor_id' not in session:
        return jsonify({
            'success': False,
            'message': 'No hay una sesión activa'
        }), 401
    
    investor_id = session['investor_id']
    
    try:
        # Obtener el inversionista
        investor = Investor.query.get(investor_id)
        
        if not investor:
            return jsonify({
                'success': False,
                'message': 'Inversionista no encontrado'
            }), 404
        
        # Calcular tasas de bono según categoría
        daily_rate = calculate_monthly_bonus_rate(investor.category)
        monthly_rate = daily_rate / 4
        
        # Calcular montos potenciales
        daily_amount = investor.capital * (daily_rate / 100.0)
        monthly_amount = investor.capital * (monthly_rate / 100.0)
        
        # Determinar elegibilidad
        eligible = False
        days_to_eligible = 0
        
        if investor.created_at:
            days_since_creation = (datetime.utcnow() - investor.created_at).days
            if days_since_creation >= 90:  # 3 meses
                eligible = True
            else:
                days_to_eligible = 90 - days_since_creation
        
        return jsonify({
            'success': True,
            'data': {
                'inversionista': {
                    'id': investor.id,
                    'categoria': investor.category,
                    'capital': investor.capital,
                    'creado': investor.created_at.isoformat() if investor.created_at else None
                },
                'elegibilidad': {
                    'es_elegible': eligible,
                    'dias_para_elegibilidad': days_to_eligible if not eligible else 0
                },
                'bonos_potenciales': {
                    'diario': {
                        'tasa': daily_rate,
                        'monto_estimado': round(daily_amount, 2)
                    },
                    'mensual': {
                        'tasa': monthly_rate,
                        'monto_estimado': round(monthly_amount, 2)
                    }
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error al simular bonos: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error al simular bonos: {str(e)}"
        }), 500

@bonus_bp.route('/api/admin/bonus/run-daily', methods=['POST'])
def run_daily_bonus():
    """Ejecutar proceso de bonos diarios (solo admin)."""
    # Verificar si hay usuario en sesión y es admin
    if 'user_id' not in session or 'role' not in session:
        return jsonify({
            'success': False,
            'message': 'No hay una sesión activa'
        }), 401
    
    if session['role'] not in ['admin', 'super_admin']:
        return jsonify({
            'success': False,
            'message': 'No tiene permisos para ejecutar esta acción'
        }), 403
    
    # Ejecutar proceso de bonos diarios
    results = apply_daily_bonus_excellent_performance()
    
    return jsonify({
        'success': True,
        'message': 'Proceso de bonos diarios ejecutado correctamente',
        'data': results
    })

@bonus_bp.route('/api/admin/bonus/run-monthly', methods=['POST'])
def run_monthly_bonus():
    """Ejecutar proceso de bonos mensuales (solo admin)."""
    # Verificar si hay usuario en sesión y es admin
    if 'user_id' not in session or 'role' not in session:
        return jsonify({
            'success': False,
            'message': 'No hay una sesión activa'
        }), 401
    
    if session['role'] not in ['admin', 'super_admin']:
        return jsonify({
            'success': False,
            'message': 'No tiene permisos para ejecutar esta acción'
        }), 403
    
    # Ejecutar proceso de bonos mensuales
    results = apply_monthly_bonus()
    
    return jsonify({
        'success': True,
        'message': 'Proceso de bonos mensuales ejecutado correctamente',
        'data': results
    })

@bonus_bp.route('/api/admin/bonus/status/<int:investor_id>', methods=['GET'])
def admin_bonus_status(investor_id):
    """Obtener estado de bonos de un inversionista específico (solo admin)."""
    # Verificar si hay usuario en sesión y es admin
    if 'user_id' not in session or 'role' not in session:
        return jsonify({
            'success': False,
            'message': 'No hay una sesión activa'
        }), 401
    
    if session['role'] not in ['admin', 'super_admin']:
        return jsonify({
            'success': False,
            'message': 'No tiene permisos para ejecutar esta acción'
        }), 403
    
    result = get_investor_bonus_history(investor_id)
    
    return jsonify(result)

@bonus_bp.route('/api/admin/bonus/summary', methods=['GET'])
def admin_bonus_summary():
    """Obtener resumen de bonos de todos los inversionistas (solo admin)."""
    # Verificar si hay usuario en sesión y es admin
    if 'user_id' not in session or 'role' not in session:
        return jsonify({
            'success': False,
            'message': 'No hay una sesión activa'
        }), 401
    
    if session['role'] not in ['admin', 'super_admin']:
        return jsonify({
            'success': False,
            'message': 'No tiene permisos para ejecutar esta acción'
        }), 403
    
    try:
        # Obtener todos los inversionistas
        investors = Investor.query.all()
        
        # Recopilar estadísticas de bonos
        summary = {
            'total_inversionistas': len(investors),
            'total_capital': sum(investor.capital for investor in investors if investor.capital),
            'bonos_potenciales_diarios': 0.0,
            'bonos_potenciales_mensuales': 0.0,
            'por_categoria': {
                'platinum': {'inversionistas': 0, 'capital': 0.0, 'bonos_potenciales': 0.0},
                'gold': {'inversionistas': 0, 'capital': 0.0, 'bonos_potenciales': 0.0},
                'silver': {'inversionistas': 0, 'capital': 0.0, 'bonos_potenciales': 0.0},
                'bronze': {'inversionistas': 0, 'capital': 0.0, 'bonos_potenciales': 0.0}
            }
        }
        
        # Calcular estadísticas por inversionista
        for investor in investors:
            category = investor.category.lower() if investor.category else 'bronze'
            capital = investor.capital or 0.0
            
            # Calcular bonos potenciales
            daily_rate = calculate_monthly_bonus_rate(category)
            monthly_rate = daily_rate / 4
            daily_bonus = capital * (daily_rate / 100.0)
            monthly_bonus = capital * (monthly_rate / 100.0)
            
            # Actualizar estadísticas generales
            summary['bonos_potenciales_diarios'] += daily_bonus
            summary['bonos_potenciales_mensuales'] += monthly_bonus
            
            # Actualizar estadísticas por categoría
            if category in summary['por_categoria']:
                summary['por_categoria'][category]['inversionistas'] += 1
                summary['por_categoria'][category]['capital'] += capital
                summary['por_categoria'][category]['bonos_potenciales'] += daily_bonus
        
        return jsonify({
            'success': True,
            'data': summary
        })
        
    except Exception as e:
        logger.error(f"Error al obtener resumen de bonos: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error al obtener resumen de bonos: {str(e)}"
        }), 500

def register_bonus_routes(app):
    """Registrar rutas de bonos en la aplicación Flask."""
    app.register_blueprint(bonus_bp)