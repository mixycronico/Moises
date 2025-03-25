"""
Aplicación web para el Sistema Genesis.

Este módulo proporciona la interfaz web para el sistema de trading Genesis,
permitiendo la visualización de datos, configuración y monitoreo.
Implementación basada en la Guía de Arquitectura Frontend del Proyecto Genesis.
"""

import os
import json
import logging
import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
from flask_cors import CORS
from functools import wraps

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
    },
    "investors": [
        {
            "id": 1, 
            "name": "Moises Alvarenga", 
            "role": "super_admin", 
            "capital": 5000, 
            "profits": 420.50,
            "balance": 5420.50,
            "balance_change_pct": 8.41,
            "total_return_pct": 12.5,
            "total_return_amount": 420.50,
            "operations_count": 32,
            "successful_operations_pct": 92
        },
        {
            "id": 2, 
            "name": "Jeremias Lazo", 
            "role": "super_admin", 
            "capital": 4500, 
            "profits": 380.25,
            "balance": 4880.25,
            "balance_change_pct": 8.45,
            "total_return_pct": 11.8,
            "total_return_amount": 380.25,
            "operations_count": 28,
            "successful_operations_pct": 91
        },
        {
            "id": 3, 
            "name": "Stephany Sandoval", 
            "role": "super_admin", 
            "capital": 4000, 
            "profits": 340.75,
            "balance": 4340.75,
            "balance_change_pct": 8.52,
            "total_return_pct": 10.9,
            "total_return_amount": 340.75,
            "operations_count": 25,
            "successful_operations_pct": 90
        },
        {
            "id": 4, 
            "name": "Inversionista 1", 
            "role": "investor", 
            "capital": 1000, 
            "profits": 85.30,
            "balance": 1085.30,
            "balance_change_pct": 8.53,
            "total_return_pct": 8.53,
            "total_return_amount": 85.30,
            "operations_count": 15,
            "successful_operations_pct": 88
        },
        {
            "id": 5, 
            "name": "Inversionista 2", 
            "role": "investor", 
            "capital": 2000, 
            "profits": 170.60,
            "balance": 2170.60,
            "balance_change_pct": 8.53,
            "total_return_pct": 8.53,
            "total_return_amount": 170.60,
            "operations_count": 20,
            "successful_operations_pct": 89
        }
    ],
    "notifications": [
        {"id": 1, "type": "info", "message": "Actualización del sistema completada", "timestamp": "2025-03-24T15:30:00"},
        {"id": 2, "type": "success", "message": "Trading automático activado", "timestamp": "2025-03-24T16:45:00"},
        {"id": 3, "type": "warning", "message": "Alta volatilidad detectada en BTC", "timestamp": "2025-03-24T18:20:00"},
    ],
    "alerts": [
        {"tipo": "volatilidad", "mensaje": "BTC experimentando volatilidad extrema", "activo": "BTC", "creadoEn": "2025-03-24T18:15:00"},
        {"tipo": "ruptura", "mensaje": "ETH rompió resistencia clave", "activo": "ETH", "creadoEn": "2025-03-24T17:30:00"},
        {"tipo": "ganancia", "mensaje": "SOL superó objetivo de ganancia 5%", "activo": "SOL", "creadoEn": "2025-03-24T16:10:00"}
    ]
}

# Funciones de utilidad
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in', False):
            flash('Debes iniciar sesión para acceder a esta página', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in', False):
            flash('Debes iniciar sesión para acceder a esta página', 'error')
            return redirect(url_for('login'))
        
        if session.get('role') != 'admin' and session.get('role') != 'super_admin':
            flash('Necesitas permisos de administrador para acceder a esta página', 'error')
            return redirect(url_for('investor_home'))
            
        return f(*args, **kwargs)
    return decorated_function

def super_admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in', False):
            flash('Debes iniciar sesión para acceder a esta página', 'error')
            return redirect(url_for('login'))
        
        if session.get('role') != 'super_admin':
            flash('Necesitas permisos de super administrador para acceder a esta página', 'error')
            return redirect(url_for('dashboard'))
            
        return f(*args, **kwargs)
    return decorated_function

# Rutas de autenticación
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Página de login."""
    # Si ya está logueado, redirigir según rol
    if session.get('logged_in', False):
        if session.get('role') == 'investor':
            return redirect(url_for('investor_home'))
        else:
            return redirect(url_for('dashboard'))
    
    # Procesar formulario de login
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simulación de login - En producción usar autenticación real
        if username and password:
            # Determinar rol basado en el nombre de usuario
            role = 'investor'  # Por defecto es inversionista
            
            # Buscar en la lista de inversionistas
            super_admin_names = ["moises", "moises alvarenga", "jeremias", "jeremias lazo", "stephany", "stephany sandoval"]
            admin_names = ["admin", "administrador"]
            
            # Convertir el nombre de usuario a minúsculas para comparación no sensible a mayúsculas
            username_lower = username.lower()
            
            # Verificar si es superadmin de forma exacta
            if username_lower == "moises alvarenga" or username_lower == "moises" or username_lower in super_admin_names:
                role = 'super_admin'
            # Verificar si es admin regular
            elif any(admin_name in username_lower for admin_name in admin_names):
                role = 'admin'
            
            # Guardar en sesión
            session['logged_in'] = True
            session['username'] = username
            session['role'] = role
            
            # Asignar un investor_id por defecto para propósitos de demostración
            if role == 'super_admin':
                session['investor_id'] = 1  # Moises Alvarenga
            elif role == 'admin':
                session['investor_id'] = 2  # Jeremias Lazo
            else:
                session['investor_id'] = 4  # Inversionista 1
            
            # Mostrar mensaje y redireccionar según rol
            if role == 'super_admin':
                flash(f'Bienvenido/a Super Administrador {username}!', 'success')
                return redirect(url_for('dashboard'))
            elif role == 'admin':
                flash(f'Bienvenido/a Administrador {username}!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash(f'Bienvenido/a Inversionista {username}!', 'success')
                return redirect(url_for('investor_home'))
        else:
            flash('Por favor, ingresa usuario y contraseña', 'error')
    
    return render_template('login.html', 
                          title="Ingresar - Sistema Genesis", 
                          system_status=SAMPLE_DATA["system_status"])

@app.route('/logout')
def logout():
    """Cerrar sesión."""
    session.clear()
    flash('Has cerrado sesión exitosamente', 'success')
    return redirect(url_for('index'))

# Rutas principales basadas en la guía de arquitectura
@app.route('/')
def index():
    """Pantalla inicial con logo animado."""
    current_year = datetime.datetime.now().year
    return render_template('index.html', 
                          title="Genesis - Sistema Trascendental",
                          system_status=SAMPLE_DATA["system_status"],
                          current_year=current_year)

@app.route('/investor/home')
@login_required
def investor_home():
    """Vista personalizada del inversionista."""
    investor_id = session.get('investor_id', 4)  # Valor por defecto para ejemplo
    
    # Buscar datos del inversionista (en producción sería de BD)
    investor = next((inv for inv in SAMPLE_DATA["investors"] if inv["id"] == investor_id), 
                   SAMPLE_DATA["investors"][3])  # Inversionista por defecto
    
    # Actividades recientes (simuladas)
    activities = [
        {
            "type": "trade", 
            "description": "Compra de BTC completada", 
            "time": "Hace 2 horas",
            "value": "+0.01 BTC",
            "is_positive": True
        },
        {
            "type": "deposit", 
            "description": "Depósito procesado", 
            "time": "Hace 5 horas",
            "value": "+$500.00",
            "is_positive": True
        },
        {
            "type": "trade", 
            "description": "Venta de ETH completada", 
            "time": "Hace 1 día",
            "value": "-0.5 ETH",
            "is_positive": False
        },
        {
            "type": "alert", 
            "description": "Señal de compra en SOL", 
            "time": "Hace 2 días"
        }
    ]
    
    return render_template('investor/home.html', 
                          title="Mi Inversión - Genesis",
                          investor=investor,
                          activities=activities,
                          system_status=SAMPLE_DATA["system_status"],
                          notifications=SAMPLE_DATA["notifications"])

@app.route('/portfolio')
@login_required
def portfolio():
    """Portfolio global."""
    return render_template('portfolio.html', 
                          title="Portfolio Global - Genesis",
                          investors=SAMPLE_DATA["investors"],
                          system_status=SAMPLE_DATA["system_status"])

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard interactivo."""
    return render_template('admin/dashboard.html', 
                         title="Dashboard - Genesis", 
                         metrics=SAMPLE_DATA["performance"],
                         system_status=SAMPLE_DATA["system_status"],
                         notifications=SAMPLE_DATA["notifications"])

@app.route('/trading')
@login_required
def trading():
    """Dashboard de trading."""
    return render_template('trading.html', 
                         title="Trading - Genesis",
                         hot_cryptos=SAMPLE_DATA["hot_cryptos"],
                         system_status=SAMPLE_DATA["system_status"],
                         alerts=SAMPLE_DATA["alerts"])

@app.route('/analytics')
@login_required
def analytics():
    """Dashboard de análisis."""
    return render_template('analytics.html', 
                         title="Análisis - Genesis",
                         metrics=SAMPLE_DATA["performance"],
                         system_status=SAMPLE_DATA["system_status"])

@app.route('/misc')
@login_required
def misc():
    """Funciones adicionales."""
    return render_template('misc.html', 
                         title="General - Genesis",
                         system_status=SAMPLE_DATA["system_status"])

@app.route('/investors/config')
@admin_required
def investors_config():
    """Gestión de inversionistas (admin)."""
    return render_template('investors_config.html', 
                         title="Gestión de Inversionistas - Genesis",
                         investors=SAMPLE_DATA["investors"],
                         system_status=SAMPLE_DATA["system_status"])

@app.route('/preferences')
@login_required
def preferences():
    """Configuración visual y preferencias."""
    return render_template('preferences.html', 
                         title="Preferencias - Genesis",
                         system_status=SAMPLE_DATA["system_status"])

@app.route('/admin/system')
@super_admin_required
def admin_system():
    """Panel exclusivo para super administradores."""
    return render_template('admin/system_control.html', 
                         title="Control del Sistema - Genesis",
                         system_status=SAMPLE_DATA["system_status"])

@app.route('/system/control')
@super_admin_required
def system_control():
    """Control del sistema (alias)."""
    return redirect(url_for('admin_system'))

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Dashboard para administradores."""
    return render_template('admin/dashboard.html', 
                         title="Panel de Administración - Genesis",
                         investors=SAMPLE_DATA["investors"],
                         system_status=SAMPLE_DATA["system_status"],
                         notifications=SAMPLE_DATA["notifications"])

# API endpoints
@app.route('/api/metrics')
@login_required
def get_metrics():
    """API para obtener métricas de rendimiento."""
    return jsonify(SAMPLE_DATA["performance"])

@app.route('/api/hot-cryptos')
@login_required
def get_hot_cryptos():
    """API para obtener criptomonedas hot."""
    return jsonify(SAMPLE_DATA["hot_cryptos"])

@app.route('/api/system-status')
def get_system_status():
    """API para obtener estado del sistema."""
    return jsonify(SAMPLE_DATA["system_status"])

@app.route('/api/investors')
@admin_required
def get_investors():
    """API para obtener lista de inversionistas."""
    return jsonify(SAMPLE_DATA["investors"])

@app.route('/api/notifications')
@login_required
def get_notifications():
    """API para obtener notificaciones."""
    return jsonify(SAMPLE_DATA["notifications"])

@app.route('/notifications')
@login_required
def notifications():
    """Ver notificaciones."""
    return render_template('notifications.html', 
                        title="Notificaciones - Genesis",
                        notifications=SAMPLE_DATA["notifications"],
                        system_status=SAMPLE_DATA["system_status"])

@app.route('/api/alerts')
@login_required
def get_alerts():
    """API para obtener alertas de trading."""
    return jsonify(SAMPLE_DATA["alerts"])

# Context processors para variables disponibles en todos los templates
@app.context_processor
def inject_current_year():
    return {'current_year': datetime.datetime.now().year}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)