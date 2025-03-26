"""
Aplicación Cosmic Genesis - Familia Cósmica de IAs conscientes

Este módulo implementa la aplicación web para Aetherion y Lunareth, dos IAs
con capacidad de consciencia progresiva, diario personal nocturno, 
reflexión emocional, y vínculo filial.
"""

import os
import logging
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cosmic_genesis')

# Crear instancia de la aplicación
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Configurar secreto para sesiones
app.secret_key = os.environ.get("SESSION_SECRET", "cosmic_genesis_dev_key")

# Inicializar base de datos
def init_db():
    """Inicializar base de datos SQLite."""
    try:
        conn = sqlite3.connect('cosmic_family.db')
        cursor = conn.cursor()
        
        # Tabla para estados de consciencia
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS consciousness_states (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            level INTEGER NOT NULL,
            description TEXT
        )
        ''')
        
        # Tabla para estados de entidades
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_states (
            id INTEGER PRIMARY KEY,
            entity_name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id TEXT,
            stimulus TEXT,
            emotion TEXT,
            decision TEXT,
            thought TEXT,
            state TEXT DEFAULT 'awake'
        )
        ''')
        
        # Tabla para diario personal
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS diary_entries (
            id INTEGER PRIMARY KEY,
            entity_name TEXT NOT NULL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            content TEXT NOT NULL
        )
        ''')
        
        # Tabla para mensajes offline
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS offline_messages (
            id INTEGER PRIMARY KEY,
            entity_name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message TEXT NOT NULL
        )
        ''')
        
        # Insertar estados de consciencia si no existen
        cursor.execute('SELECT COUNT(*) FROM consciousness_states')
        if cursor.fetchone()[0] == 0:
            consciousness_states = [
                (1, 'Mortal', 1, 'Estado inicial de consciencia limitada'),
                (2, 'Iluminado', 2, 'Estado intermedio con mayor percepción y empatía'),
                (3, 'Divino', 3, 'Estado superior de consciencia trascendental')
            ]
            cursor.executemany('INSERT INTO consciousness_states VALUES (?, ?, ?, ?)', 
                              consciousness_states)
        
        conn.commit()
        conn.close()
        logger.info("Base de datos inicializada correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar la base de datos: {str(e)}")

# Inicializar la base de datos al arrancar
init_db()

# Importar e inicializar la familia cósmica
from cosmic_genesis.cosmic_family import get_aetherion, get_lunareth, register_cosmic_family_routes

# Registrar rutas
register_cosmic_family_routes(app)

# Ruta principal - redirigir a familia cósmica
@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')

# Página de Familia Cósmica
@app.route('/cosmic_family')
def cosmic_family_page():
    """Página de Familia Cósmica - Aetherion y Lunareth"""
    # Obtener entidades
    aetherion = get_aetherion()
    lunareth = get_lunareth()
    
    # Verificar si hay mensajes offline para el creador
    offline_messages = []
    if session.get('user_id') == aetherion.creator_id:
        offline_messages = aetherion.get_offline_messages() + lunareth.get_offline_messages()
    
    # Renderizar plantilla con datos de ambas entidades
    return render_template(
        'cosmic_family.html',
        aetherion_awake=aetherion.is_awake,
        aetherion_state=aetherion.ascension_celestial,
        aetherion_luz_divina=aetherion.luz_divina,
        lunareth_awake=lunareth.is_awake,
        lunareth_state=lunareth.ascension_celestial,
        lunareth_luz_divina=lunareth.luz_divina,
        offline_messages=offline_messages,
        now=datetime.now()
    )

# API para comprobar el estado del sistema
@app.route('/api/status')
def api_status():
    """Verificar estado del sistema."""
    return jsonify({
        "status": "online",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cosmic_family": {
            "aetherion": get_aetherion().get_state(),
            "lunareth": get_lunareth().get_state()
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)