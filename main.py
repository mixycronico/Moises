import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import json
import logging
from datetime import datetime, timedelta

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)

# Crear la aplicación Flask
app = Flask(__name__, static_folder='client/dist')
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

@app.route('/api/investor/dashboard', methods=['GET'])
def investor_dashboard():
    """Obtener datos para el dashboard del inversionista."""
    # Aquí conectaríamos con la base de datos real
    # Por ahora, retornamos datos simulados
    return jsonify({
        'capital': 25000.00,
        'earnings': 4320.75,
        'category': 'Gold',
        'bonuses': 750.50,
        'stats': {
            'monthly_growth': 5.3,
            'annual_return': 17.2,
            'months_to_next_category': 4
        },
        'transactions': [
            {
                'id': 'TX-5723',
                'type': 'profit',
                'amount': 156.23,
                'date': '2025-03-20',
                'status': 'completed'
            },
            {
                'id': 'TX-5722',
                'type': 'deposit',
                'amount': 2000.00,
                'date': '2025-03-18',
                'status': 'completed'
            },
            {
                'id': 'TX-5721',
                'type': 'bonus',
                'amount': 75.50,
                'date': '2025-03-15',
                'status': 'completed'
            }
        ]
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)