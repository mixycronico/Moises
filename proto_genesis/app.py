"""
Aplicación web del Sistema Genesis - Visualización y Control.

Este módulo implementa la interfaz web para el Sistema Genesis,
permitiendo la visualización del estado del sistema, métricas
de rendimiento y el control de sus componentes.
"""

import os
import logging
import json
import random
import datetime
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

# Estado del sistema (datos iniciales)
system_state = {
    "energia": 0.98,
    "conciencia": 3,
    "ciclo": 1245,
    "adaptaciones": 78,
    "memoria": 256,
    "sinapsis": 4562,
    "emocion_dominante": "Curiosidad",
    "nivel_descripcion": "Consciencia",
    "nivel_detalle": "Formación de una 'personalidad' simulada con preferencias, emociones y patrones de comportamiento complejos."
}

# Log de actividad del sistema
system_log = [
    {"time": "12:45:32", "type": "info", "message": "Sistema inicializado correctamente"},
    {"time": "12:45:45", "type": "info", "message": "Módulo de memoria cargado con 256 registros"},
    {"time": "12:46:12", "type": "success", "message": "Conexión con la base de datos establecida"},
    {"time": "12:48:03", "type": "info", "message": "AdaptiveRiskManager inicializado con capital: $10,000.00"},
    {"time": "12:49:17", "type": "warning", "message": "Módulos de análisis de sentimiento no encontrados"},
    {"time": "12:50:22", "type": "info", "message": "TranscendentalCryptoClassifier inicializado con capital=10000.0"}
]

# Datos de evolución temporal (Para gráficos)
evolution_data = {
    "labels": ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep"],
    "consciencia": [1, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 2.9, 3.0],
    "adaptacion": [0.5, 0.8, 1.2, 1.6, 1.9, 2.2, 2.6, 2.8, 3.1]
}

# Datos de distribución emocional (Para gráficos)
emotion_data = {
    "labels": ["Curiosidad", "Entusiasmo", "Alegría", "Calma", "Interés", "Cautela"],
    "values": [45, 25, 20, 15, 30, 18]
}

@app.route('/')
def index():
    """Página principal."""
    return render_template('cosmic_index.html')  # Usar la versión cósmica

@app.route('/interact')
def interact():
    """Panel de control del Sistema Genesis."""
    return render_template('cosmic_interact.html')  # Usar la versión cósmica

@app.route('/about')
def about():
    """Página de información sobre Sistema Genesis."""
    return render_template('cosmic_about.html')  # Usar la versión cósmica

# Rutas adicionales para pruebas o desarrollo
@app.route('/original')
def original_index():
    """Versión original de la página principal."""
    return render_template('index.html')

@app.route('/original/interact')
def original_interact():
    """Versión original del panel de control."""
    return render_template('interact.html')

@app.route('/original/about')
def original_about():
    """Versión original de la página de información."""
    return render_template('about.html')

@app.route('/api/status')
def status():
    """API para obtener estado actual del sistema."""
    return jsonify(system_state)

@app.route('/api/logs')
def logs():
    """API para obtener log de actividad del sistema."""
    return jsonify(system_log)

@app.route('/api/charts/evolution')
def evolution_chart():
    """API para obtener datos de evolución temporal."""
    return jsonify(evolution_data)

@app.route('/api/charts/emotions')
def emotion_chart():
    """API para obtener datos de distribución emocional."""
    return jsonify(emotion_data)

@app.route('/api/update', methods=['POST'])
def update_metrics():
    """API para actualizar métricas del sistema (simulada)."""
    # Simulamos algunas actualizaciones aleatorias
    system_state["energia"] = round(max(0.5, min(1.0, system_state["energia"] + (random.random() * 0.1 - 0.05))), 2)
    system_state["ciclo"] += random.randint(1, 10)
    
    if random.random() > 0.7:
        system_state["adaptaciones"] += 1
    
    if random.random() > 0.6:
        system_state["memoria"] += 1
    
    system_state["sinapsis"] += random.randint(1, 20)
    
    # Registrar en el log
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M:%S")
    
    log_types = ["info", "info", "info", "success", "warning"]
    log_type = random.choice(log_types)
    
    log_messages = [
        "Procesamiento de datos completado.",
        "Análisis de patrón detectado.",
        "Sincronización con la base de datos.",
        "Actualización de memoria completada.",
        f"Adaptación neuronal #{random.randint(1, 1000)} registrada.",
        "Ciclo de aprendizaje completado."
    ]
    log_message = random.choice(log_messages)
    
    system_log.insert(0, {"time": time_str, "type": log_type, "message": log_message})
    
    # Mantener el log en un tamaño razonable
    if len(system_log) > 50:
        system_log.pop()
    
    return jsonify({"success": True, "state": system_state})

# Punto de entrada para ejecución directa
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)