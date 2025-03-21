"""
Aplicación web para el sistema de trading Genesis.

Este módulo proporciona la interfaz web para el sistema de trading,
permitiendo la visualización de datos, configuración y monitoreo.
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from genesis.db.models import Base
from genesis.utils.logger import setup_logging

# Configurar el logger
logger = setup_logging('webapp')

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

# Rutas básicas de la aplicación web
@app.route('/')
def index():
    """Página principal."""
    return jsonify({
        "status": "ok",
        "message": "Genesis Trading System API",
        "version": "0.1.0",
        "endpoints": [
            "/",
            "/health",
            "/status"
        ]
    })

@app.route('/health')
def health():
    """Punto de verificación de salud del sistema."""
    return jsonify({
        "status": "ok",
        "database": "connected" if check_database_connection() else "disconnected",
        "timestamp": str(db.func.now())
    })

@app.route('/status')
def status():
    """Estado del sistema de trading."""
    return jsonify({
        "status": "running",
        "message": "Sistema operativo",
        "components": {
            "api": "active",
            "database": "connected" if check_database_connection() else "disconnected",
        }
    })

def check_database_connection():
    """Verificar la conexión a la base de datos."""
    try:
        db.session.execute("SELECT 1")
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