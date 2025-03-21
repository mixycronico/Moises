"""
Aplicación web para el sistema de trading Genesis.

Este módulo proporciona la interfaz web para el sistema de trading,
permitiendo la visualización de datos, configuración y monitoreo.
También expone la API REST para integración con sistemas externos.
"""

import os
import logging
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from genesis.db.models import Base
from genesis.utils.logger import setup_logging
from genesis.api import init_api, init_swagger
from genesis.api.logger import APILogger

# Configurar el logger
logger = setup_logging('webapp')
logging.basicConfig(level=logging.DEBUG)

# Crear la aplicación Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "genesis_dev_key")

# Configurar la base de datos
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/genesis")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Inicializar SQLAlchemy
db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Inicializar API REST
init_api(app)

# Inicializar documentación Swagger
init_swagger(app)

# Inicializar logger API
api_logger = APILogger(app)

# Rutas básicas de la aplicación web
@app.route('/')
def index():
    """Página principal."""
    return jsonify({
        "status": "ok",
        "message": "Genesis Trading System API",
        "version": "1.0.0",
        "documentation": "/api/docs",
        "endpoints": [
            "/",
            "/health",
            "/status",
            "/api/v1/...",
            "/api/docs",
            "/logs"
        ]
    })

@app.route('/health')
def health():
    """Punto de verificación de salud del sistema."""
    return jsonify({
        "status": "ok",
        "database": "connected" if check_database_connection() else "disconnected",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/status')
def status():
    """Estado del sistema de trading."""
    return jsonify({
        "status": "running",
        "message": "Sistema operativo",
        "components": {
            "api": "active",
            "rest_api": "active",
            "swagger": "active",
            "api_logger": "active",
            "database": "connected" if check_database_connection() else "disconnected",
        }
    })

@app.route('/logs')
def view_logs():
    """Ver logs recientes de la API."""
    try:
        # Parámetros opcionales
        log_type = request.args.get('type')
        count = min(int(request.args.get('count', 50)), 1000)  # Limitar a 1000 máximo
        
        # Obtener logs
        logs = api_logger.get_recent_logs(log_type, count)
        
        # Formatear respuesta en base al Accept header
        if request.headers.get('Accept') == 'text/html':
            # HTML response
            html_content = "<html><head><title>Genesis API Logs</title>"
            html_content += "<style>body { font-family: monospace; padding: 20px; background: #f5f5f5; }"
            html_content += "h1 { color: #333; } .log { margin-bottom: 10px; padding: 10px; background: white; border-radius: 5px; }"
            html_content += ".request { border-left: 5px solid #4CAF50; } .response { border-left: 5px solid #2196F3; }"
            html_content += ".exception { border-left: 5px solid #F44336; }</style></head><body>"
            html_content += f"<h1>Genesis API Logs ({len(logs)} entries)</h1>"
            
            for log in logs:
                log_class = log['type']
                html_content += f"<div class='log {log_class}'>"
                html_content += f"<strong>Type:</strong> {log_class} | "
                html_content += f"<strong>Time:</strong> {log['data'].get('timestamp', '')}"
                html_content += "<pre>" + json.dumps(log['data'], indent=2) + "</pre>"
                html_content += "</div>"
            
            html_content += "</body></html>"
            return Response(html_content, mimetype='text/html')
        else:
            # Default to JSON
            return jsonify({
                "success": True,
                "data": {
                    "logs": logs,
                    "count": len(logs)
                }
            })
    except Exception as e:
        logger.error(f"Error al obtener logs: {e}")
        return jsonify({
            "success": False,
            "error": "Error al obtener logs",
            "message": str(e)
        }), 500

def check_database_connection():
    """Verificar la conexión a la base de datos."""
    try:
        db.session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Error de conexión a la base de datos: {e}")
        return False

# Crear las tablas de la base de datos si no existen
with app.app_context():
    db.create_all()
    logger.info("Tablas de base de datos inicializadas")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)