import os
from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import json
import logging
import threading
import time
from datetime import datetime, timedelta

# Importar módulos del sistema
try:
    # Importar módulos originales
    from cosmic_family import CosmicEntity
    from cosmic_trading_api import register_trading_routes
    
    # Importar sistema simplificado - sin dependencias complejas
    from simple_cosmic_trader import (
        initialize_simple_trading,
        SimpleCosmicNetwork,
        CosmicTrader
    )
except ImportError as e:
    logging.error(f"Error al importar módulos del sistema: {e}")

# Configuración de logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("genesis_main")

# Crear la aplicación Flask
app = Flask(__name__, static_folder='static')
app.secret_key = os.environ.get("SESSION_SECRET", "genesis_development_key")
CORS(app)  # Habilitar CORS para todas las rutas

# Variables globales para el sistema de trading mejorado
enhanced_network = None
aetherion_trader = None
lunareth_trader = None

# Inicializar sistema de trading mejorado
def initialize_enhanced_trading_system():
    global enhanced_network, aetherion_trader, lunareth_trader
    
    try:
        # Inicializar sistema simplificado
        enhanced_network, aetherion_trader, lunareth_trader = initialize_simple_trading(
            father_name="otoniel"
        )
        logger.info("Sistema de trading simplificado inicializado correctamente")
        return True
    except Exception as e:
        logger.error(f"Error al inicializar sistema de trading simplificado: {e}")
        return False

# Iniciar el sistema en un hilo separado para no bloquear la aplicación
threading.Thread(target=initialize_enhanced_trading_system, daemon=True).start()

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

# Rutas para sistema mejorado
@app.route('/api/enhanced/status', methods=['GET'])
def enhanced_status():
    """Obtener estado del sistema de trading mejorado."""
    global enhanced_network
    
    if not enhanced_network:
        return jsonify({
            'success': False,
            'message': 'Sistema de trading mejorado no inicializado',
            'initialized': False
        }), 503
    
    try:
        status = enhanced_network.get_network_status()
        return jsonify({
            'success': True,
            'initialized': True,
            'status': status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error al obtener estado del sistema mejorado: {e}")
        return jsonify({
            'success': False,
            'message': f'Error al obtener estado: {str(e)}',
            'initialized': True
        }), 500

@app.route('/api/enhanced/trader/<name>', methods=['GET'])
def enhanced_trader_status(name):
    """Obtener estado de un trader específico en el sistema mejorado."""
    global enhanced_network
    
    if not enhanced_network:
        return jsonify({
            'success': False,
            'message': 'Sistema de trading mejorado no inicializado',
            'initialized': False
        }), 503
    
    try:
        # Buscar entidad por nombre
        entity = next((e for e in enhanced_network.entities if e.name == name), None)
        
        if not entity:
            return jsonify({
                'success': False,
                'message': f'Entidad {name} no encontrada',
                'initialized': True
            }), 404
            
        status = entity.get_status()
        return jsonify({
            'success': True,
            'trader': status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error al obtener estado del trader {name}: {e}")
        return jsonify({
            'success': False,
            'message': f'Error al obtener estado: {str(e)}',
            'initialized': True
        }), 500

@app.route('/api/enhanced/collaboration', methods=['GET'])
def enhanced_collaboration_metrics():
    """Obtener métricas de colaboración del sistema mejorado."""
    global enhanced_network
    
    if not enhanced_network:
        return jsonify({
            'success': False,
            'message': 'Sistema de trading mejorado no inicializado',
            'initialized': False
        }), 503
    
    try:
        # SimpleCosmicNetwork no tiene el método get_collaboration_metrics
        # Usamos información simplificada
        metrics = {
            "knowledge_pool": enhanced_network.knowledge_pool,
            "entity_count": len(enhanced_network.entities),
            "alive_entities": len([e for e in enhanced_network.entities if e.alive]),
            "collaboration_rounds": 0
        }
        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error al obtener métricas de colaboración: {e}")
        return jsonify({
            'success': False,
            'message': f'Error al obtener métricas: {str(e)}',
            'initialized': True
        }), 500

@app.route('/api/enhanced/collaborate', methods=['POST'])
def trigger_enhanced_collaboration():
    """Disparar una ronda de colaboración en el sistema mejorado."""
    global enhanced_network
    
    if not enhanced_network:
        return jsonify({
            'success': False,
            'message': 'Sistema de trading mejorado no inicializado',
            'initialized': False
        }), 503
    
    try:
        # Utilizar el método de simulación de colaboración en SimpleCosmicNetwork
        results = enhanced_network.simulate_collaboration()
        
        # Formatear resultados para mantener consistencia con API
        formatted_results = []
        for r in results:
            formatted_results.append({
                'entity': r['entity'],
                'result': r['message'] if 'message' in r else 'Colaboración realizada'
            })
                
        return jsonify({
            'success': True,
            'results': formatted_results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error al disparar colaboración: {e}")
        return jsonify({
            'success': False,
            'message': f'Error al disparar colaboración: {str(e)}',
            'initialized': True
        }), 500

@app.route('/api/enhanced/simulate', methods=['POST'])
def simulate_enhanced_trading():
    """Ejecutar una simulación de trading en el sistema mejorado."""
    global enhanced_network
    
    if not enhanced_network:
        return jsonify({
            'success': False,
            'message': 'Sistema de trading mejorado no inicializado',
            'initialized': False
        }), 503
    
    try:
        results = enhanced_network.simulate()
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error al simular trading: {e}")
        return jsonify({
            'success': False,
            'message': f'Error al simular trading: {str(e)}',
            'initialized': True
        }), 500

@app.route('/api/enhanced/chat', methods=['POST'])
def enhanced_cosmic_chat():
    """
    API de chat con sistema mejorado.
    Recibe un mensaje y devuelve respuestas de los traders mejorados.
    """
    global aetherion_trader, lunareth_trader
    
    if not aetherion_trader or not lunareth_trader:
        return jsonify({
            'success': False,
            'message': 'Entidades de trading mejoradas no inicializadas',
            'initialized': False
        }), 503
    
    data = request.json
    message = data.get('message', '')
    user_id = data.get('user_id', 'anonymous')
    
    if not message:
        return jsonify({
            'success': False,
            'error': 'El mensaje no puede estar vacío'
        }), 400
    
    try:
        # Obtener información de mercado actual
        price = aetherion_trader.fetch_market_data()
        predicted = aetherion_trader.predict_price()
        
        # Procesar petición con Aetherion (especulador)
        aetherion_action = aetherion_trader.trade()
        
        # Procesar petición con Lunareth (estratega)
        lunareth_action = lunareth_trader.trade()
        
        # Respuestas enriquecidas
        enhanced_response = {
            'success': True,
            'aetherion': {
                'text': f"¡Saludos, explorador cósmico! {aetherion_action}",
                'level': float(aetherion_trader.level),
                'energy': float(aetherion_trader.energy),
                'capabilities': aetherion_trader.capabilities,
                'price': price,
                'predicted_price': predicted,
                'timestamp': datetime.now().isoformat()
            },
            'lunareth': {
                'text': f"Análisis en curso. {lunareth_action}",
                'level': float(lunareth_trader.level),
                'energy': float(lunareth_trader.energy),
                'capabilities': lunareth_trader.capabilities,
                'timestamp': datetime.now().isoformat()
            },
            'market_data': {
                'current_price': price,
                'predicted_price': predicted,
                'change_percent': ((predicted - price) / price * 100) if price and predicted else None,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(enhanced_response)
    except Exception as e:
        logger.error(f"Error en chat mejorado: {e}")
        return jsonify({
            'success': False,
            'message': f'Error en procesamiento: {str(e)}',
            'initialized': True
        }), 500

# Servir archivos estáticos de React
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Servir aplicación React."""
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Registrar rutas de trading originales
try:
    register_trading_routes(app)
    logging.info("Rutas de trading registradas correctamente")
except Exception as e:
    logging.error(f"Error al registrar rutas de trading: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)