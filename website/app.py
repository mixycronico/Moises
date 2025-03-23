"""
Aplicación web para el Sistema Genesis.

Este módulo proporciona la interfaz web para el sistema de trading Genesis,
permitiendo la visualización de datos, configuración y monitoreo.
"""

import os
import json
import logging
from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
from flask_cors import CORS

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("genesis_website")
logger.info("Logging inicializado para página web de Genesis")

# Crear la aplicación Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "genesis_transcendental_key")

# Habilitar CORS para las API
CORS(app)

# Datos de ejemplo para el frontend
SAMPLE_DATA = {
    "performance": {
        "roi": 90,
        "drawdown": 65,
        "successRate": 76,
        "volatility": 84,
        "sharpeRatio": 1.5,
        "winLossRatio": 2.3,
    },
    "hot_cryptos": [
        {"symbol": "BTC", "price": 55340.42, "change_24h": 2.5, "signal": "strong_buy", "confidence": 98},
        {"symbol": "ETH", "price": 3150.75, "change_24h": 1.8, "signal": "buy", "confidence": 92},
        {"symbol": "SOL", "price": 132.68, "change_24h": 5.2, "signal": "strong_buy", "confidence": 96},
        {"symbol": "ADA", "price": 0.58, "change_24h": -0.7, "signal": "hold", "confidence": 75},
        {"symbol": "AVAX", "price": 36.25, "change_24h": 4.1, "signal": "buy", "confidence": 89},
    ],
    "system_status": {
        "components_active": 5,
        "success_rate": 100,
        "uptime_hours": 267,
        "last_error": None,
        "mode": "SINGULARIDAD_V4",
        "intensity": 1000.0
    }
}

# Rutas de la aplicación web
@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html', title="Genesis - Sistema Trascendental")

@app.route('/dashboard')
def dashboard():
    """Dashboard principal del sistema."""
    return render_template('dashboard.html', 
                         title="Dashboard - Genesis", 
                         metrics=SAMPLE_DATA["performance"],
                         system_status=SAMPLE_DATA["system_status"])

@app.route('/trading')
def trading_process():
    """Página de proceso de trading."""
    return render_template('trading.html', 
                         title="Proceso de Trading - Genesis")

@app.route('/cryptos')
def cryptos():
    """Página de criptomonedas hot."""
    return render_template('cryptos.html', 
                         title="Criptomonedas Hot - Genesis",
                         hot_cryptos=SAMPLE_DATA["hot_cryptos"])

@app.route('/api/metrics')
def get_metrics():
    """API para obtener métricas de rendimiento."""
    return jsonify(SAMPLE_DATA["performance"])

@app.route('/api/hot-cryptos')
def get_hot_cryptos():
    """API para obtener criptomonedas hot."""
    return jsonify(SAMPLE_DATA["hot_cryptos"])

@app.route('/api/system-status')
def get_system_status():
    """API para obtener estado del sistema."""
    return jsonify(SAMPLE_DATA["system_status"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)