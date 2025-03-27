"""
Rutas API para el sistema de comisiones de Proto Genesis.

Este módulo implementa los endpoints necesarios para la gestión de comisiones
para los administradores del sistema.
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from models import db, Commission, Investor, User

# Configurar logger
logger = logging.getLogger('genesis_website')

# Crear Blueprint
commission_bp = Blueprint('commissions', __name__)

@commission_bp.route('/api/commissions', methods=['GET'])
def get_all_commissions():
    """Obtener todas las comisiones (solo admin)."""
    # Verificar autenticación
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    # Verificar rol
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
    
    # Solo admin y super_admin pueden ver todas las comisiones
    if not user.is_admin:
        return jsonify({'success': False, 'message': 'Acceso denegado'}), 403
    
    # Obtener todas las comisiones
    commissions = Commission.query.all()
    
    # Convertir a diccionario para JSON
    commissions_data = [{
        'id': c.id,
        'investor_id': c.investor_id,
        'admin_id': c.admin_id,
        'amount': c.amount,
        'description': c.description,
        'status': c.status,
        'created_at': c.created_at.isoformat(),
        'processed_at': c.processed_at.isoformat() if c.processed_at else None
    } for c in commissions]
    
    return jsonify({
        'success': True,
        'commissions': commissions_data
    })

@commission_bp.route('/api/commissions/<int:commission_id>', methods=['GET'])
def get_commission(commission_id):
    """Obtener detalles de una comisión específica."""
    # Verificar autenticación
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    # Verificar rol
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
    
    # Solo admin o el inversionista asociado pueden ver la comisión
    commission = Commission.query.get(commission_id)
    if not commission:
        return jsonify({'success': False, 'message': 'Comisión no encontrada'}), 404
    
    # Verificar si es admin o el inversionista asociado
    investor = Investor.query.filter_by(user_id=user.id).first()
    if not user.is_admin and (not investor or investor.id != commission.investor_id):
        return jsonify({'success': False, 'message': 'Acceso denegado'}), 403
    
    # Cargar datos relacionados
    investor_data = {
        'id': commission.investor.id,
        'user': {
            'id': commission.investor.user.id,
            'username': commission.investor.user.username
        },
        'category': commission.investor.category
    }
    
    admin_data = {
        'id': commission.admin.id,
        'username': commission.admin.username
    }
    
    # Convertir a diccionario para JSON
    commission_data = {
        'id': commission.id,
        'investor': investor_data,
        'admin': admin_data,
        'amount': commission.amount,
        'description': commission.description,
        'status': commission.status,
        'created_at': commission.created_at.isoformat(),
        'processed_at': commission.processed_at.isoformat() if commission.processed_at else None
    }
    
    return jsonify({
        'success': True,
        'commission': commission_data
    })

@commission_bp.route('/api/commissions/create', methods=['POST'])
def create_commission():
    """Crear una nueva comisión (solo admin)."""
    # Verificar autenticación
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    # Verificar rol
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
    
    # Solo admin y super_admin pueden crear comisiones
    if not user.is_admin:
        return jsonify({'success': False, 'message': 'Acceso denegado'}), 403
    
    # Obtener datos del formulario
    try:
        investor_id = request.form.get('investor_id')
        amount = request.form.get('amount')
        description = request.form.get('description', '')
        
        if not investor_id or not amount:
            return jsonify({'success': False, 'message': 'Faltan campos requeridos'}), 400
        
        # Convertir valores
        investor_id = int(investor_id)
        amount = float(amount)
        
        # Verificar que el inversionista exista
        investor = Investor.query.get(investor_id)
        if not investor:
            return jsonify({'success': False, 'message': 'Inversionista no encontrado'}), 404
        
        # Crear comisión
        commission = Commission(
            investor_id=investor_id,
            admin_id=user.id,
            amount=amount,
            description=description,
            status='pending'
        )
        
        db.session.add(commission)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Comisión creada exitosamente',
            'commission_id': commission.id
        })
        
    except ValueError:
        return jsonify({'success': False, 'message': 'Valor inválido'}), 400
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error al crear comisión: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@commission_bp.route('/api/commissions/<int:commission_id>/approve', methods=['POST'])
def approve_commission(commission_id):
    """Aprobar una comisión (solo super_admin o mixycronico)."""
    # Verificar autenticación
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    # Verificar que sea mixycronico o super_admin
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
    
    is_creator = user.username.lower() == 'mixycronico'
    if not is_creator and user.role != 'super_admin':
        return jsonify({'success': False, 'message': 'Acceso denegado, solo el creador puede aprobar comisiones'}), 403
    
    # Buscar la comisión
    commission = Commission.query.get(commission_id)
    if not commission:
        return jsonify({'success': False, 'message': 'Comisión no encontrada'}), 404
    
    # Verificar que esté pendiente
    if commission.status != 'pending':
        return jsonify({'success': False, 'message': 'Esta comisión ya ha sido procesada'}), 400
    
    try:
        # Cambiar estado y agregar fecha de procesamiento
        commission.status = 'approved'
        commission.processed_at = datetime.utcnow()
        
        # Transferir el monto al admin
        admin = User.query.get(commission.admin_id)
        if admin and admin.is_admin:
            admin_investor = Investor.query.filter_by(user_id=admin.id).first()
            if admin_investor:
                admin_investor.balance += commission.amount
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Comisión aprobada exitosamente'
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error al aprobar comisión: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@commission_bp.route('/api/commissions/<int:commission_id>/reject', methods=['POST'])
def reject_commission(commission_id):
    """Rechazar una comisión (solo super_admin o mixycronico)."""
    # Verificar autenticación
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    # Verificar que sea mixycronico o super_admin
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
    
    is_creator = user.username.lower() == 'mixycronico'
    if not is_creator and user.role != 'super_admin':
        return jsonify({'success': False, 'message': 'Acceso denegado, solo el creador puede rechazar comisiones'}), 403
    
    # Buscar la comisión
    commission = Commission.query.get(commission_id)
    if not commission:
        return jsonify({'success': False, 'message': 'Comisión no encontrada'}), 404
    
    # Verificar que esté pendiente
    if commission.status != 'pending':
        return jsonify({'success': False, 'message': 'Esta comisión ya ha sido procesada'}), 400
    
    try:
        # Cambiar estado y agregar fecha de procesamiento
        commission.status = 'rejected'
        commission.processed_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Comisión rechazada exitosamente'
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error al rechazar comisión: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@commission_bp.route('/api/investors', methods=['GET'])
def get_all_investors():
    """Obtener todos los inversionistas (solo admin)."""
    # Verificar autenticación
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    # Verificar rol
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
    
    # Solo admin puede ver todos los inversionistas
    if not user.is_admin:
        return jsonify({'success': False, 'message': 'Acceso denegado'}), 403
    
    # Obtener todos los inversionistas
    investors = Investor.query.all()
    
    # Convertir a diccionario para JSON
    investors_data = []
    for investor in investors:
        # Cargar usuario relacionado
        investor_user = User.query.get(investor.user_id)
        if investor_user:
            investors_data.append({
                'id': investor.id,
                'user': {
                    'id': investor_user.id,
                    'username': investor_user.username,
                    'email': investor_user.email
                },
                'balance': investor.balance,
                'capital': investor.capital,
                'category': investor.category
            })
    
    return jsonify({
        'success': True,
        'investors': investors_data
    })

@commission_bp.route('/api/investors/<int:investor_id>/commissions', methods=['GET'])
def get_investor_commissions(investor_id):
    """Obtener comisiones de un inversionista específico."""
    # Verificar autenticación
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
    
    # Verificar acceso (admin o el inversionista mismo)
    investor = Investor.query.get(investor_id)
    if not investor:
        return jsonify({'success': False, 'message': 'Inversionista no encontrado'}), 404
    
    if not user.is_admin and user.id != investor.user_id:
        return jsonify({'success': False, 'message': 'Acceso denegado'}), 403
    
    # Obtener comisiones del inversionista
    commissions = Commission.query.filter_by(investor_id=investor_id).all()
    
    # Convertir a diccionario para JSON
    commissions_data = [{
        'id': c.id,
        'admin_id': c.admin_id,
        'admin_username': c.admin.username,
        'amount': c.amount,
        'description': c.description,
        'status': c.status,
        'created_at': c.created_at.isoformat(),
        'processed_at': c.processed_at.isoformat() if c.processed_at else None
    } for c in commissions]
    
    return jsonify({
        'success': True,
        'commissions': commissions_data
    })

def register_commission_routes(app):
    """Registrar rutas de comisiones en la aplicación Flask."""
    app.register_blueprint(commission_bp)