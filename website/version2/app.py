"""
Aplicación principal de Flask para el Proyecto Genesis.

Este módulo proporciona las rutas API y la integración con el frontend.
"""

import os
import logging
from flask import Flask, jsonify, request, render_template, session, redirect, url_for
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('genesis_website')
logger.info('Logging inicializado para página web de Genesis')

# Crear base de datos
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Crear aplicación Flask
app = Flask(__name__, 
            static_folder='frontend/dist',
            template_folder='frontend/dist')

# Configurar secreto para sesiones
app.secret_key = os.environ.get("SESSION_SECRET", "genesis_dev_secret")

# Configurar base de datos
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Habilitar CORS para el desarrollo
CORS(app, supports_credentials=True)

# Inicializar base de datos
db.init_app(app)

# Importar modelos después de inicializar db
from website.version2.models import User, Investment, Transaction

# Asegurar que las tablas existan
with app.app_context():
    db.create_all()
    
    # Crear usuarios de prueba si no existen
    if not User.query.filter_by(username='super_admin').first():
        super_admin = User(
            username='super_admin',
            email='super_admin@genesis.com',
            password_hash=generate_password_hash('super_admin_password'),
            role='super_admin'
        )
        db.session.add(super_admin)
        
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@genesis.com',
            password_hash=generate_password_hash('admin_password'),
            role='admin'
        )
        db.session.add(admin)
        
    if not User.query.filter_by(username='investor').first():
        investor = User(
            username='investor',
            email='investor@genesis.com',
            password_hash=generate_password_hash('investor_password'),
            role='investor'
        )
        db.session.add(investor)
        
    db.session.commit()
    logger.info('Base de datos inicializada')

# Rutas para la interfaz web
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Capturar todas las rutas y redirigir al frontend."""
    return render_template('index.html')

# API para autenticación
@app.route('/api/login', methods=['POST'])
def login():
    """API para iniciar sesión."""
    data = request.json
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'success': False, 'message': 'Datos incorrectos'}), 400
    
    user = User.query.filter_by(username=data['username']).first()
    
    if not user or not check_password_hash(user.password_hash, data['password']):
        return jsonify({'success': False, 'message': 'Usuario o contraseña incorrectos'}), 401
    
    # Guardar información de sesión
    session['user_id'] = user.id
    session['username'] = user.username
    session['role'] = user.role
    
    return jsonify({
        'success': True, 
        'message': 'Inicio de sesión exitoso',
        'user': {
            'id': user.id,
            'username': user.username,
            'role': user.role
        }
    })

@app.route('/api/logout', methods=['POST'])
def logout():
    """API para cerrar sesión."""
    session.clear()
    return jsonify({'success': True, 'message': 'Sesión cerrada'})

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    """API para verificar autenticación."""
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            return jsonify({
                'authenticated': True,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'role': user.role
                }
            })
    
    return jsonify({'authenticated': False})

# API para datos de inversor
@app.route('/api/investor/portfolio', methods=['GET'])
def investor_portfolio():
    """API para obtener portafolio del inversor."""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    # En una implementación real, obtendríamos datos reales de la base de datos
    # Por ahora, devolvemos datos de muestra
    return jsonify({
        'balance': 10000,
        'invested': 7500,
        'available': 2500,
        'performance': {
            'daily': 1.5,
            'weekly': 3.2,
            'monthly': 7.8,
            'yearly': 22.4
        },
        'investments': [
            {
                'id': 1,
                'name': 'Bitcoin',
                'symbol': 'BTC',
                'amount': 0.5,
                'value_usd': 3000,
                'change_24h': 2.3
            },
            {
                'id': 2,
                'name': 'Ethereum',
                'symbol': 'ETH',
                'amount': 5,
                'value_usd': 2500,
                'change_24h': -1.2
            },
            {
                'id': 3,
                'name': 'Cardano',
                'symbol': 'ADA',
                'amount': 2000,
                'value_usd': 2000,
                'change_24h': 5.7
            }
        ]
    })

@app.route('/api/investor/transactions', methods=['GET'])
def investor_transactions():
    """API para obtener transacciones del inversor."""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'No autenticado'}), 401
    
    # En una implementación real, obtendríamos datos reales de la base de datos
    # Por ahora, devolvemos datos de muestra
    return jsonify([
        {
            'id': 1,
            'type': 'BUY',
            'asset': 'Bitcoin',
            'symbol': 'BTC',
            'amount': 0.2,
            'price_usd': 60000,
            'total_usd': 12000,
            'date': '2023-04-10T14:30:00Z',
            'status': 'COMPLETED'
        },
        {
            'id': 2,
            'type': 'BUY',
            'asset': 'Ethereum',
            'symbol': 'ETH',
            'amount': 3,
            'price_usd': 3000,
            'total_usd': 9000,
            'date': '2023-04-05T10:15:00Z',
            'status': 'COMPLETED'
        },
        {
            'id': 3,
            'type': 'SELL',
            'asset': 'Cardano',
            'symbol': 'ADA',
            'amount': 500,
            'price_usd': 1.2,
            'total_usd': 600,
            'date': '2023-04-01T09:45:00Z',
            'status': 'COMPLETED'
        }
    ])

# API para datos de administrador
@app.route('/api/admin/investors', methods=['GET'])
def admin_investors():
    """API para obtener lista de inversores (solo admin)."""
    if 'user_id' not in session or session['role'] not in ['admin', 'super_admin']:
        return jsonify({'success': False, 'message': 'No autorizado'}), 403
    
    # En una implementación real, obtendríamos datos reales de la base de datos
    # Por ahora, devolvemos datos de muestra
    return jsonify([
        {
            'id': 3,
            'username': 'investor',
            'email': 'investor@genesis.com',
            'balance': 10000,
            'invested': 7500,
            'performance': 15.3,
            'status': 'active'
        },
        {
            'id': 4,
            'username': 'investor2',
            'email': 'investor2@genesis.com',
            'balance': 25000,
            'invested': 20000,
            'performance': 8.7,
            'status': 'active'
        },
        {
            'id': 5,
            'username': 'investor3',
            'email': 'investor3@genesis.com',
            'balance': 5000,
            'invested': 3000,
            'performance': 22.1,
            'status': 'inactive'
        }
    ])

@app.route('/api/admin/system-status', methods=['GET'])
def admin_system_status():
    """API para obtener estado del sistema (solo admin)."""
    if 'user_id' not in session or session['role'] not in ['admin', 'super_admin']:
        return jsonify({'success': False, 'message': 'No autorizado'}), 403
    
    # En una implementación real, obtendríamos datos reales del sistema
    # Por ahora, devolvemos datos de muestra
    return jsonify({
        'system': {
            'status': 'operational',
            'memory_usage': 45.2,
            'cpu_usage': 22.7,
            'uptime': '7d 12h 45m',
            'active_users': 15
        },
        'market': {
            'status': 'open',
            'connected_exchanges': 3,
            'active_pairs': 25,
            'data_latency': 0.8
        },
        'operations': {
            'transactions_today': 128,
            'pending_operations': 3,
            'success_rate': 99.7
        }
    })

# API para datos de super administrador
@app.route('/api/super-admin/admins', methods=['GET'])
def super_admin_admins():
    """API para obtener lista de administradores (solo super admin)."""
    if 'user_id' not in session or session['role'] != 'super_admin':
        return jsonify({'success': False, 'message': 'No autorizado'}), 403
    
    # En una implementación real, obtendríamos datos reales de la base de datos
    # Por ahora, devolvemos datos de muestra
    return jsonify([
        {
            'id': 2,
            'username': 'admin',
            'email': 'admin@genesis.com',
            'last_login': '2023-04-15T08:30:00Z',
            'status': 'active',
            'permissions': ['manage_users', 'view_reports']
        },
        {
            'id': 6,
            'username': 'admin2',
            'email': 'admin2@genesis.com',
            'last_login': '2023-04-14T14:20:00Z',
            'status': 'active',
            'permissions': ['manage_users']
        }
    ])

@app.route('/api/super-admin/system-stats', methods=['GET'])
def super_admin_system_stats():
    """API para obtener estadísticas avanzadas del sistema (solo super admin)."""
    if 'user_id' not in session or session['role'] != 'super_admin':
        return jsonify({'success': False, 'message': 'No autorizado'}), 403
    
    # En una implementación real, obtendríamos datos reales del sistema
    # Por ahora, devolvemos datos de muestra
    return jsonify({
        'system': {
            'status': 'operational',
            'memory_usage': 45.2,
            'cpu_usage': 22.7,
            'uptime': '7d 12h 45m',
            'active_users': 15,
            'server_load': [12.3, 14.5, 11.8, 13.2, 15.6],
            'database_size': 1245.8,
            'error_rate': 0.03
        },
        'ai_components': {
            'aetherion': {
                'status': 'active',
                'consciousness_level': 3,
                'adaptation_count': 157,
                'efficiency': 98.3
            },
            'deepseek': {
                'status': 'active',
                'api_calls_today': 532,
                'success_rate': 99.7
            },
            'buddha': {
                'status': 'active',
                'predictions_accuracy': 86.5,
                'learning_cycles': 1245
            },
            'gabriel': {
                'status': 'active',
                'current_state': 'SERENE',
                'behavior_adaptations': 75
            }
        },
        'security': {
            'threat_level': 'low',
            'blocked_attempts': 23,
            'last_update': '2023-04-15T06:00:00Z',
            'encryption_status': 'active'
        }
    })