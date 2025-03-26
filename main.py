"""
Punto de entrada principal para la aplicación web del Sistema Genesis.

Este módulo proporciona la integración entre la aplicación web y
los componentes backend del Sistema Genesis. Sirve la aplicación React
desde el directorio 'website/frontend/dist' y los endpoints de API.
"""

import os
import sys
import logging
from flask import Flask, send_from_directory, send_file, jsonify

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('genesis_website')

# Crear app Flask
app = Flask(__name__, 
           static_folder='website/frontend/dist',
           static_url_path='')

# Configurar secreto para sesiones
app.secret_key = os.environ.get("SESSION_SECRET", "genesis_dev_key")

# Rutas para la aplicación React
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path == 'login':
        return send_file(os.path.join(app.static_folder, 'login.html'))
    elif path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_file(os.path.join(app.static_folder, 'index.html'))

# Agregar proto_genesis al path de Python para importar endpoints de API
sys.path.append(os.path.abspath('proto_genesis'))

# Importar rutas de autenticación
from auth_routes import register_auth_routes

# Registrar rutas de autenticación
register_auth_routes(app)

# Importar endpoints de API de proto_genesis
try:
    # Importar sólo los endpoints de API, no la app completa
    from proto_genesis.app import status, logs, evolution_chart, emotion_chart, update_metrics
    
    # Registrar endpoints de API
    app.add_url_rule('/api/status', 'status', status)
    app.add_url_rule('/api/logs', 'logs', logs)
    app.add_url_rule('/api/charts/evolution', 'evolution_chart', evolution_chart)
    app.add_url_rule('/api/charts/emotions', 'emotion_chart', emotion_chart)
    app.add_url_rule('/api/update', 'update_metrics', update_metrics, methods=['POST'])
    
    # Endpoint de verificación para comprobar que la API está funcionando
    @app.route('/api/check')
    def api_check():
        return jsonify({"status": "operational", "message": "API de Genesis funcionando correctamente"})
    
except ImportError as e:
    logger.error(f"Error al importar endpoints de API: {e}")
    
    # Endpoint de verificación alternativo
    @app.route('/api/check')
    def api_check():
        return jsonify({"status": "limited", "message": "API de Genesis en modo mantenimiento"})

# Punto de entrada para Gunicorn
# No eliminar - esta línea es requerida para que el servidor funcione correctamente

if __name__ == "__main__":
    logger.info("Iniciando servidor web del Sistema Genesis...")
    # Verificar si existe la carpeta dist
    if not os.path.exists('website/frontend/dist'):
        logger.warning("No se encontró la carpeta dist de React. La aplicación web no estará disponible.")
    app.run(host="0.0.0.0", port=5000, debug=True)