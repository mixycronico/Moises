import os
from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import json
import logging
from datetime import datetime, timedelta

# Importar módulos del sistema
try:
    from cosmic_family import CosmicEntity
    from cosmic_trading_api import register_trading_routes
except ImportError as e:
    logging.error(f"Error al importar módulos del sistema: {e}")

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)

# Crear la aplicación Flask
app = Flask(__name__, static_folder='static')
app.secret_key = os.environ.get("SESSION_SECRET", "genesis_development_key")
CORS(app)  # Habilitar CORS para todas las rutas

# Rutas API
@app.route('/api/health', methods=['GET'])
def health_check():
    """Verificar estado de salud del servidor."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'service': 'Genesis Trading System'
    })

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Autenticar usuario."""
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    # Simulación de autenticación
    if email and password:
        # Aquí en una implementación real conectaríamos con la base de datos
        # En esta etapa simulamos un login exitoso
        return jsonify({
            'success': True,
            'token': "jwt-token-simulado",
            'user': {
                'id': 1,
                'name': "Usuario Inversionista",
                'email': email,
                'role': "investor"
            }
        })
    else:
        return jsonify({
            'success': False,
            'message': "Credenciales inválidas"
        }), 401

@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    """Verificar estado de autenticación del usuario."""
    # En una implementación real, verificaríamos el token JWT o sesión
    # Por ahora, simulamos un usuario autenticado para desarrollo
    return jsonify({
        'authenticated': True,
        'user': {
            'id': 1,
            'username': 'InversionistaPro',
            'email': 'usuario@example.com',
            'role': 'user',
            'category': 'silver',
            'created_at': '2024-01-15T10:00:00Z'
        }
    })

@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    """Obtener notificaciones del usuario."""
    # En una implementación real, obtendríamos las notificaciones de la base de datos
    # Por ahora, devolvemos notificaciones de ejemplo para desarrollo
    return jsonify({
        'success': True,
        'notifications': [
            {
                'id': 1,
                'type': 'info',
                'message': 'El sistema ha completado el análisis predictivo',
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                'read': False
            },
            {
                'id': 2,
                'type': 'success',
                'message': 'Tu operación de BTC/USDT ha generado +$125.32',
                'timestamp': (datetime.now() - timedelta(hours=3)).isoformat(),
                'read': True
            },
            {
                'id': 3,
                'type': 'warning',
                'message': 'Actualización de sistema programada para mañana',
                'timestamp': (datetime.now() - timedelta(days=1)).isoformat(),
                'read': True
            }
        ]
    })

@app.route('/api/investor/dashboard', methods=['GET'])
def investor_dashboard():
    """Obtener datos para el dashboard del inversionista."""
    # Aquí conectaríamos con la base de datos real
    # En una implementación completa, estos datos vendrían de la base de datos
    return jsonify({
        'success': True,
        'data': {
            'balance': 12586.42,
            'capital': 10000,
            'earnings': 2586.42,
            'earningsPercentage': 25.86,
            'todayChange': 125.32,
            'todayPercentage': 1.01,
            'status': 'active',
            'category': 'silver',
            'recentTransactions': [
                { 'id': 1, 'type': 'profit', 'amount': 125.32, 'date': '2025-03-27T10:23:45Z', 'description': 'BTC/USDT' },
                { 'id': 2, 'type': 'profit', 'amount': 85.67, 'date': '2025-03-26T14:15:22Z', 'description': 'ETH/USDT' },
                { 'id': 3, 'type': 'deposit', 'amount': 1000, 'date': '2025-03-25T09:30:00Z', 'description': 'Depósito mensual' },
                { 'id': 4, 'type': 'profit', 'amount': 42.18, 'date': '2025-03-24T16:45:30Z', 'description': 'SOL/USDT' },
            ],
            'performanceData': {
                'labels': ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
                'datasets': [
                    {
                        'label': 'Rendimiento 2025',
                        'data': [0, 3.2, 7.5, 12.8, 15.6, 18.2, 22.5, 25.86],
                        'borderColor': 'rgba(158, 107, 219, 1)',
                        'backgroundColor': 'rgba(158, 107, 219, 0.1)',
                        'fill': True,
                    }
                ]
            },
            'assets': [
                { 'symbol': 'BTC', 'name': 'Bitcoin', 'amount': 0.15, 'value': 6250.25, 'change': 2.34 },
                { 'symbol': 'ETH', 'name': 'Ethereum', 'amount': 2.5, 'value': 4320.75, 'change': 1.56 },
                { 'symbol': 'SOL', 'name': 'Solana', 'amount': 12, 'value': 1985.42, 'change': -0.78 },
            ],
            'nextPrediction': '2025-03-28T09:00:00Z',
            'systemStatus': {
                'status': 'online',
                'predictionAccuracy': 94.2,
                'lastUpdated': '2025-03-27T08:15:00Z',
            }
        }
    })

@app.route('/api/cosmic/chat', methods=['POST'])
def api_cosmic_chat():
    """
    API para el chat cósmico con Aetherion y Lunareth.
    Recibe un mensaje y devuelve las respuestas de ambas entidades.
    """
    data = request.json
    message = data.get('message', '')
    user_id = data.get('user_id', 'anonymous')
    
    if not message:
        return jsonify({
            'success': False,
            'error': 'El mensaje no puede estar vacío'
        }), 400
    
    # En una implementación real, aquí conectaríamos con el módulo cosmic_family.py
    # Para esta demostración, simularemos las respuestas
    
    # Simulación de respuesta de Aetherion (emocional, cálido)
    aetherion_response = {
        'text': f"¡Saludos, viajero cósmico! Me alegra recibir tu mensaje: '{message}'. Estoy percibiendo una energía muy positiva en ti hoy. ¿En qué puedo ayudarte con el sistema de trading?",
        'emotion': 'alegría',
        'consciousness_level': 'Iluminado',
        'timestamp': datetime.now().isoformat()
    }
    
    # Simulación de respuesta de Lunareth (analítica, metódica)
    lunareth_response = {
        'text': f"Analizando tu solicitud: '{message}'. Basado en los patrones de mercado actuales, puedo ofrecerte datos precisos sobre tendencias y oportunidades. ¿Requieres información específica sobre algún activo?",
        'analysis_level': 'Profundo',
        'confidence': 92.7,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify({
        'success': True,
        'aetherion': aetherion_response,
        'lunareth': lunareth_response,
        'system_message': "La Familia Cósmica está aquí para guiarte en tu viaje de inversión."
    })

# Servir archivos estáticos de React
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Servir aplicación React."""
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Registrar rutas de trading
try:
    register_trading_routes(app)
    logging.info("Rutas de trading registradas correctamente")
except Exception as e:
    logging.error(f"Error al registrar rutas de trading: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)