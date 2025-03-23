"""
Aplicación web para el Sistema Genesis con integración de estrategia adaptativa.

Este módulo proporciona la interfaz web para el sistema de trading Genesis,
permitiendo la visualización de datos, configuración y monitoreo.
También expone la API REST para integración con sistemas externos.

Integra el clasificador transcendental de criptomonedas con capacidades adaptativas
que mantienen la eficiencia del sistema incluso cuando el capital crece significativamente.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
import threading
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask_cors import CORS

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("genesis")
logger.info("Logging inicializado para aplicación Flask")

# Crear la aplicación Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "genesis_transcendental_key")

# Habilitar CORS para las API
CORS(app)

# Configurar la base de datos
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/genesis")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["GENESIS_CONFIG"] = os.path.join(os.path.dirname(__file__), "genesis_config.json")

# Inicializar SQLAlchemy sin dependencia de Base
db_sql = SQLAlchemy()
db_sql.init_app(app)

# Estado de inicialización y cachés
genesis_initialized = False
genesis_init_results = {}
crypto_hot_cache = []
last_classification_time = None
last_performance_update = None

# Variables para importación diferida
db = None
crypto_classifier = None
risk_manager = None
performance_tracker = None

def init_genesis_components():
    """Inicializar componentes de Genesis tras el arranque."""
    global db, crypto_classifier, risk_manager, performance_tracker
    
    # Importar módulos del sistema Genesis con importación diferida
    # para evitar problemas de dependencias circulares
    try:
        from genesis.db.transcendental_database import db as genesis_db
        from genesis.analysis.transcendental_crypto_classifier import crypto_classifier as genesis_classifier
        from genesis.risk.adaptive_risk_manager import risk_manager as genesis_risk
        from genesis.analytics.transcendental_performance_tracker import performance_tracker as genesis_tracker
        
        db = genesis_db
        crypto_classifier = genesis_classifier
        risk_manager = genesis_risk
        performance_tracker = genesis_tracker
        
        logger.info("Componentes de Genesis importados correctamente")
        return True
    except Exception as e:
        logger.error(f"Error al importar componentes de Genesis: {e}")
        return False

# Inicializar componentes
init_components_success = init_genesis_components()

def run_async_function(coro):
    """Ejecutar una función asincrónica en un hilo separado."""
    def wrapper():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(coro)
        except Exception as e:
            logger.error(f"Error en ejecución asincrónica: {e}")
            result = None
        finally:
            loop.close()
            
        return result
    
    thread = threading.Thread(target=wrapper)
    thread.start()
    return thread

def get_genesis_initialization_status():
    """Obtener estado actual de inicialización del sistema."""
    return {
        "initialized": genesis_initialized,
        "results": genesis_init_results,
        "components_imported": init_components_success,
        "hot_cryptos_count": len(crypto_hot_cache),
        "last_classification": last_classification_time.isoformat() if last_classification_time else None,
        "last_performance_update": last_performance_update.isoformat() if last_performance_update else None
    }

async def initialize_system():
    """Inicializar el Sistema Genesis por completo."""
    global genesis_initialized, genesis_init_results, db, crypto_classifier
    
    if not init_components_success:
        logger.error("No se pueden inicializar los componentes, importación fallida")
        return False
    
    try:
        # Inicializar sistema principal
        from genesis.init import initialize_genesis_system
        results = await initialize_genesis_system(app.config["GENESIS_CONFIG"])
        genesis_init_results = results
        
        if results["overall_status"] == "success":
            genesis_initialized = True
            logger.info("Sistema Genesis inicializado correctamente")
            
            # Ejecutar clasificación inicial en otro hilo
            run_async_function(refresh_classification())
            
            return True
        else:
            logger.error(f"Error al inicializar Sistema Genesis: {results}")
            return False
    except Exception as e:
        logger.error(f"Error crítico en inicialización del sistema: {e}")
        genesis_init_results = {"overall_status": "error", "message": str(e)}
        return False

async def refresh_classification():
    """Actualizar clasificación de criptomonedas."""
    global crypto_hot_cache, last_classification_time
    
    if not genesis_initialized or not crypto_classifier:
        logger.warning("No se puede actualizar clasificación, sistema no inicializado")
        return False
    
    try:
        # Lista de criptomonedas principales para clasificar
        top_symbols = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", "MATIC", "LINK",
                     "UNI", "ATOM", "LTC", "ALGO", "FIL", "AAVE", "SNX", "CRV", "YFI", "COMP"]
        
        # Ejecutar clasificación
        results = await crypto_classifier.clasificar_cryptos(top_symbols, force_update=True)
        
        # Actualizar caché
        crypto_hot_cache = results["hot_cryptos"]
        last_classification_time = datetime.now()
        
        logger.info(f"Clasificación actualizada. {len(crypto_hot_cache)} cryptos calientes identificadas.")
        return True
    except Exception as e:
        logger.error(f"Error al actualizar clasificación: {e}")
        return False

async def update_performance_data():
    """Actualizar datos de rendimiento con simulación."""
    global last_performance_update
    
    if not genesis_initialized or not performance_tracker:
        logger.warning("No se puede actualizar rendimiento, sistema no inicializado")
        return False
    
    try:
        # En un sistema real, obtendríamos datos reales de rendimiento
        # Para esta demo, simulamos algunas actualizaciones
        
        # Simular cambio en capital
        current_capital = performance_tracker.metricas["capital_actual"]
        
        # Cambio aleatorio entre -0.5% y +1.5%
        import random
        change_percent = random.uniform(-0.005, 0.015)
        new_capital = current_capital * (1 + change_percent)
        
        # Actualizar capital
        await performance_tracker.actualizar_capital(new_capital)
        
        # Si hay cambio positivo, simular una operación exitosa
        if change_percent > 0:
            random_symbol = random.choice(["BTC", "ETH", "SOL", "ADA", "BNB"])
            
            operacion = {
                "symbol": random_symbol,
                "tipo": "LONG",
                "entrada_precio": 100.0,
                "salida_precio": 105.0,
                "unidades": 0.1,
                "resultado_usd": 0.5,
                "resultado_porcentual": 0.05,
                "ganadora": True,
                "estrategia": "adaptativa"
            }
            
            await performance_tracker.registrar_operacion(operacion)
        
        last_performance_update = datetime.now()
        logger.info(f"Datos de rendimiento actualizados. Nuevo capital: ${new_capital:.2f}")
        return True
    except Exception as e:
        logger.error(f"Error al actualizar datos de rendimiento: {e}")
        return False

# Inicializar sistema en un hilo separado al iniciar la aplicación
def start_system_initialization():
    run_async_function(initialize_system())

# Iniciar inicialización tras un retraso mínimo
timer = threading.Timer(1.0, start_system_initialization)
timer.daemon = True
timer.start()

# Rutas de la aplicación web
@app.route('/')
def index():
    """Página principal."""
    return jsonify({
        "status": "ok",
        "message": "Genesis Trading System API - Estrategia Adaptativa",
        "version": "2.0.0",
        "sistema_inicializado": genesis_initialized,
        "documentation": "/api/docs",
        "dashboard": "/dashboard",
        "endpoints": [
            "/",
            "/dashboard",
            "/health",
            "/status",
            "/api/hot_cryptos",
            "/api/performance",
            "/api/check_database",
            "/logs"
        ]
    })

@app.route('/dashboard')
def dashboard():
    """Dashboard principal del sistema."""
    # En un sistema real, renderizaríamos una plantilla
    # Para esta API, devolveremos datos en JSON
    
    if not genesis_initialized:
        return jsonify({
            "status": "initializing",
            "message": "Sistema Genesis en proceso de inicialización",
            "init_results": genesis_init_results
        }), 202
    
    try:
        return jsonify({
            "status": "ok",
            "classifier_state": crypto_classifier.get_estado_actual() if crypto_classifier else {},
            "risk_state": risk_manager.get_estado_actual() if risk_manager else {},
            "performance_state": performance_tracker.get_estado_actual() if performance_tracker else {}
        })
    except Exception as e:
        logger.error(f"Error al obtener datos para dashboard: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error al generar dashboard: {str(e)}"
        }), 500

@app.route('/health')
def health():
    """Punto de verificación de salud del sistema."""
    db_connected = check_database_connection()
    
    system_status = {
        "status": "ok" if db_connected and genesis_initialized else "degraded",
        "database": "connected" if db_connected else "disconnected",
        "genesis_system": "initialized" if genesis_initialized else "initializing",
        "timestamp": datetime.now().isoformat()
    }
    
    if genesis_initialized:
        system_status["components"] = {
            "database": "ok" if db else "error",
            "classifier": "ok" if crypto_classifier else "error",
            "risk_manager": "ok" if risk_manager else "error",
            "performance_tracker": "ok" if performance_tracker else "error"
        }
    
    return jsonify(system_status)

@app.route('/status')
def status():
    """Estado del sistema de trading."""
    return jsonify({
        "status": "running",
        "message": "Sistema Genesis operativo",
        "genesis_initialized": genesis_initialized,
        "initialization_results": genesis_init_results,
        "components": {
            "api": "active",
            "rest_api": "active",
            "database": "connected" if check_database_connection() else "disconnected",
            "classifier": "active" if crypto_classifier else "inactive",
            "risk_manager": "active" if risk_manager else "inactive",
            "performance_tracker": "active" if performance_tracker else "inactive"
        },
        "hot_cryptos_count": len(crypto_hot_cache),
        "last_classification": last_classification_time.isoformat() if last_classification_time else None,
        "last_performance_update": last_performance_update.isoformat() if last_performance_update else None
    })

@app.route('/api/hot_cryptos')
def hot_cryptos():
    """Obtener criptomonedas calientes actuales."""
    if not genesis_initialized:
        return jsonify({
            "status": "initializing",
            "message": "Sistema Genesis en proceso de inicialización"
        }), 202
    
    # Actualizar clasificación si han pasado más de 15 minutos
    if (not last_classification_time or 
        datetime.now() - last_classification_time > timedelta(minutes=15)):
        
        run_async_function(refresh_classification())
        
        # Devolver datos actuales mientras se actualiza
        if not crypto_hot_cache:
            return jsonify({"status": "updating", "hot_cryptos": []}), 202
    
    return jsonify({
        "status": "success",
        "hot_cryptos": crypto_hot_cache,
        "last_update": last_classification_time.isoformat() if last_classification_time else None
    })

@app.route('/api/performance')
def performance_data():
    """Obtener datos de rendimiento actuales."""
    if not genesis_initialized:
        return jsonify({
            "status": "initializing",
            "message": "Sistema Genesis en proceso de inicialización"
        }), 202
    
    # Actualizar datos si han pasado más de 5 minutos
    if (not last_performance_update or 
        datetime.now() - last_performance_update > timedelta(minutes=5)):
        
        run_async_function(update_performance_data())
    
    return jsonify({
        "status": "success",
        "performance": performance_tracker.get_estado_actual() if performance_tracker else {},
        "last_update": last_performance_update.isoformat() if last_performance_update else None
    })

@app.route('/logs')
def view_logs():
    """Ver logs recientes de la API."""
    try:
        # En un sistema real, obtendríamos logs de una tabla o archivo
        # Para esta demo, mostramos datos simulados o basic_logs si existen
        
        logs = [
            {
                "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                "level": "INFO",
                "component": comp,
                "message": f"Actividad simulada del componente {comp}"
            }
            for i, comp in enumerate([
                "classifier", "risk_manager", "database", "api", 
                "performance_tracker", "transcendental_core"
            ])
        ]
        
        return jsonify({
            "success": True,
            "logs": logs,
            "genesis_status": get_genesis_initialization_status()
        })
    except Exception as e:
        logger.error(f"Error al obtener logs: {e}")
        return jsonify({
            "success": False,
            "error": "Error al obtener logs",
            "message": str(e)
        }), 500

@app.route('/api/check_database')
def check_database_status():
    """Verificar el estado de la base de datos."""
    sql_db_connected = check_database_connection()
    
    genesis_db_status = "not_initialized"
    if genesis_initialized and db:
        try:
            system_status = db.get_system_status()
            genesis_db_status = system_status["system_state"]
        except Exception as e:
            logger.error(f"Error al verificar estado de Genesis DB: {e}")
            genesis_db_status = "error"
    
    return jsonify({
        "status": "ok" if sql_db_connected else "error",
        "sqlalchemy_connected": sql_db_connected,
        "genesis_db_status": genesis_db_status,
        "timestamp": datetime.now().isoformat()
    })

def check_database_connection():
    """Verificar la conexión a la base de datos SQLAlchemy."""
    try:
        db_sql.session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Error de conexión a la base de datos SQLAlchemy: {e}")
        return False

# Crear las tablas de la base de datos si no existen
with app.app_context():
    db_sql.create_all()
    logger.info("Tablas de base de datos inicializadas")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)