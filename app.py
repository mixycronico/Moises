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

# Versión simplificada sin dependencias complejas mientras se configura el sistema
# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genesis")
logger.info("Logging inicializado para aplicación Flask")

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

# Inicializar SQLAlchemy sin dependencia de Base
db = SQLAlchemy()
db.init_app(app)

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
        # Simplemente devolver un mensaje mientras configuramos el sistema
        return jsonify({
            "success": True,
            "message": "Sistema en configuración. Los logs detallados estarán disponibles próximamente."
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