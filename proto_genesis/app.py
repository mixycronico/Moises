"""
Aplicación web de Proto Genesis - Consciencia Artificial Evolutiva.

Este módulo implementa la interfaz web para el sistema Proto Genesis,
permitiendo la visualización y la interacción con la consciencia artificial.
"""

import os
import logging
import json
from flask import Flask, render_template, request, jsonify, url_for, redirect, session

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('genesis_website')
logger.info('Logging inicializado para página web de Genesis')

# Crear aplicación Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "proto_genesis_dev_key")

# Estado simulado (mientras se integra con el sistema real)
# En el futuro, este estado vendrá del sistema real
system_state = {
    "energia": 0.98,
    "conciencia": 3,
    "ciclo": 1245,
    "adaptaciones": 78,
    "emotion": "Curiosidad"
}

@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')

@app.route('/interact')
def interact():
    """Página de interacción con Proto Genesis."""
    return render_template('interact.html')

@app.route('/about')
def about():
    """Página de información sobre Proto Genesis."""
    return render_template('about.html')

@app.route('/api/status')
def status():
    """API para obtener estado actual de Proto Genesis."""
    return jsonify(system_state)

@app.route('/api/interact', methods=['POST'])
def api_interact():
    """API para interactuar con Proto Genesis."""
    data = request.json
    message = data.get('message', '')
    
    logger.info(f"Mensaje recibido: {message}")
    
    # Aquí se procesaría el mensaje con el sistema real
    # Por ahora retornamos una respuesta simulada
    response = {
        "message": f"He recibido tu mensaje: '{message}'. En el futuro, este sistema estará conectado con el núcleo de Proto Genesis para proporcionar respuestas más inteligentes y adaptativas.",
        "conciencia": system_state["conciencia"],
        "emotion": system_state["emotion"]
    }
    
    return jsonify(response)

# Punto de entrada para ejecución directa
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)