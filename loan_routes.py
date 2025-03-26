"""
Rutas API para el sistema de préstamos de Genesis.

Este módulo implementa los endpoints necesarios para la gestión de préstamos
a inversionistas después de 3 meses en el sistema.
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from models import db
from investor_loans import (
    check_loan_eligibility, 
    calculate_max_loan_amount, 
    create_loan, 
    process_loan_payment, 
    get_investor_loan_status
)

# Configurar logger
logger = logging.getLogger('genesis_website')

# Crear Blueprint
loan_bp = Blueprint('loans', __name__)

@loan_bp.route('/api/investor/loan/status', methods=['GET'])
def loan_status():
    """Obtener estado de préstamos del inversionista actual."""
    # Verificar autenticación
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    # Obtener ID del inversionista
    from models import Investor, User
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
    
    # Buscar el inversionista asociado al usuario
    investor = Investor.query.filter_by(user_id=user.id).first()
    if not investor:
        return jsonify({'success': False, 'message': 'Perfil de inversionista no encontrado'}), 404
    
    # Obtener estado de préstamos
    loan_status = get_investor_loan_status(investor.id)
    
    if 'error' in loan_status:
        return jsonify({'success': False, 'message': loan_status['error']}), 400
    
    return jsonify({
        'success': True,
        'status': loan_status
    })

@loan_bp.route('/api/investor/loan/request', methods=['POST'])
def request_loan():
    """Solicitar un nuevo préstamo."""
    # Verificar autenticación
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    # Obtener ID del inversionista
    from models import Investor, User
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
    
    # Buscar el inversionista asociado al usuario
    investor = Investor.query.filter_by(user_id=user.id).first()
    if not investor:
        return jsonify({'success': False, 'message': 'Perfil de inversionista no encontrado'}), 404
    
    # Obtener monto solicitado
    data = request.get_json()
    if not data or 'amount' not in data:
        return jsonify({'success': False, 'message': 'Monto no especificado'}), 400
    
    try:
        # Convertir a float
        amount = float(data['amount'])
        if amount <= 0:
            return jsonify({'success': False, 'message': 'El monto debe ser mayor que cero'}), 400
        
        # Crear préstamo
        success, message, loan = create_loan(investor.id, amount)
        
        if not success:
            return jsonify({'success': False, 'message': message}), 400
        
        # Respuesta exitosa
        return jsonify({
            'success': True,
            'message': message,
            'loan': {
                'id': loan.id,
                'amount': loan.amount,
                'date': loan.loan_date.strftime('%Y-%m-%d'),
                'remaining': loan.remaining_amount
            }
        })
        
    except ValueError:
        return jsonify({'success': False, 'message': 'Monto inválido'}), 400
    except Exception as e:
        logger.error(f"Error al procesar solicitud de préstamo: {str(e)}")
        return jsonify({'success': False, 'message': f'Error al procesar solicitud: {str(e)}'}), 500

@loan_bp.route('/api/investor/loan/pay/<int:loan_id>', methods=['POST'])
def make_payment(loan_id):
    """Realizar un pago manual a un préstamo."""
    # Verificar autenticación
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    # Obtener ID del inversionista
    from models import Investor, User, InvestorLoan
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
    
    # Buscar el inversionista asociado al usuario
    investor = Investor.query.filter_by(user_id=user.id).first()
    if not investor:
        return jsonify({'success': False, 'message': 'Perfil de inversionista no encontrado'}), 404
    
    # Verificar que el préstamo exista y pertenezca al inversionista
    loan = InvestorLoan.query.get(loan_id)
    if not loan:
        return jsonify({'success': False, 'message': 'Préstamo no encontrado'}), 404
    
    if loan.investor_id != investor.id:
        return jsonify({'success': False, 'message': 'No autorizado para este préstamo'}), 403
    
    # Obtener monto a pagar
    data = request.get_json()
    if not data or 'amount' not in data:
        return jsonify({'success': False, 'message': 'Monto no especificado'}), 400
    
    try:
        # Convertir a float
        amount = float(data['amount'])
        if amount <= 0:
            return jsonify({'success': False, 'message': 'El monto debe ser mayor que cero'}), 400
        
        # Verificar que el inversionista tenga suficiente balance
        if amount > investor.balance:
            return jsonify({'success': False, 'message': 'Balance insuficiente para este pago'}), 400
        
        # Restar del balance
        investor.balance -= amount
        
        # Procesar pago
        success, message = process_loan_payment(loan_id, amount)
        
        if success:
            # Commit a la base de datos
            db.session.commit()
        else:
            # Rollback
            db.session.rollback()
            return jsonify({'success': False, 'message': message}), 400
        
        # Respuesta exitosa
        return jsonify({
            'success': True,
            'message': message,
            'new_balance': investor.balance
        })
        
    except ValueError:
        return jsonify({'success': False, 'message': 'Monto inválido'}), 400
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error al procesar pago: {str(e)}")
        return jsonify({'success': False, 'message': f'Error al procesar pago: {str(e)}'}), 500

@loan_bp.route('/api/admin/loans/status', methods=['GET'])
def admin_loan_status():
    """Obtener estado de todos los préstamos (solo admin)."""
    # Verificar autenticación y rol
    if 'user_id' not in session or session.get('role') not in ['admin', 'super_admin']:
        return jsonify({'success': False, 'message': 'No autorizado'}), 403
    
    try:
        from models import InvestorLoan, Investor, User
        
        # Obtener todos los préstamos activos
        active_loans = InvestorLoan.query.filter_by(is_active=True).all()
        
        # Formatear datos
        loans_data = []
        for loan in active_loans:
            investor = Investor.query.get(loan.investor_id)
            user = User.query.get(investor.user_id) if investor else None
            
            loans_data.append({
                'id': loan.id,
                'investor_id': loan.investor_id,
                'investor_name': user.name if user else 'Unknown',
                'amount': loan.amount,
                'remaining': loan.remaining_amount,
                'date': loan.loan_date.strftime('%Y-%m-%d'),
                'last_payment': loan.last_payment_date.strftime('%Y-%m-%d'),
                'days_since_creation': (datetime.utcnow() - loan.loan_date).days,
                'days_since_payment': (datetime.utcnow() - loan.last_payment_date).days
            })
        
        # Estadísticas generales
        total_active = len(active_loans)
        total_amount = sum(loan.amount for loan in active_loans)
        total_remaining = sum(loan.remaining_amount for loan in active_loans)
        
        return jsonify({
            'success': True,
            'active_loans': loans_data,
            'stats': {
                'total_active': total_active,
                'total_amount': total_amount,
                'total_remaining': total_remaining,
                'percent_paid': 0 if total_amount == 0 else ((total_amount - total_remaining) / total_amount) * 100
            }
        })
        
    except Exception as e:
        logger.error(f"Error al obtener estado de préstamos: {str(e)}")
        return jsonify({'success': False, 'message': f'Error al obtener préstamos: {str(e)}'}), 500

def register_loan_routes(app):
    """Registrar rutas de préstamos en la aplicación Flask."""
    app.register_blueprint(loan_bp)
    logger.info("Rutas de préstamos registradas")