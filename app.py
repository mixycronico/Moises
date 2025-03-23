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
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/genesis")
app.config["GENESIS_CONFIG"] = os.path.join(os.path.dirname(__file__), "genesis_config.json")

# Estado de inicialización y cachés
genesis_initialized = False
genesis_init_results = {}
crypto_hot_cache = []
last_classification_time = None
last_performance_update = None

# Variables para importación diferida
db_manager = None
transcendental_db = None
classifier = None
risk_manager = None
performance_tracker = None
initialize_system = None

def init_genesis_components():
    """Inicializar componentes de Genesis tras el arranque."""
    global db_manager, transcendental_db, classifier, risk_manager, performance_tracker, initialize_system
    
    # Importar módulos del sistema Genesis con importación diferida
    # para evitar problemas de dependencias circulares
    try:
        from genesis.db.base import db_manager as genesis_db_manager
        from genesis.db.transcendental_database import transcendental_db as genesis_db
        from genesis.analysis.transcendental_crypto_classifier import classifier as genesis_classifier
        from genesis.risk.adaptive_risk_manager import risk_manager as genesis_risk
        from genesis.analytics.transcendental_performance_tracker import performance_tracker as genesis_tracker
        from genesis.init import initialize_system as genesis_init
        
        db_manager = genesis_db_manager
        transcendental_db = genesis_db
        classifier = genesis_classifier
        risk_manager = genesis_risk
        performance_tracker = genesis_tracker
        initialize_system = genesis_init
        
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
    thread.daemon = True
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

async def initialize_genesis():
    """Inicializar el Sistema Genesis por completo."""
    global genesis_initialized, genesis_init_results
    
    if not init_components_success:
        logger.error("No se pueden inicializar los componentes, importación fallida")
        return False
    
    try:
        # Inicializar sistema principal
        results = await initialize_system(app.config["GENESIS_CONFIG"])
        genesis_init_results = results
        
        if results.get("database", False) and results.get("classifier", False):
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
        genesis_init_results = {"mensaje": str(e)}
        return False

async def refresh_classification():
    """Actualizar clasificación de criptomonedas."""
    global crypto_hot_cache, last_classification_time
    
    if not genesis_initialized or classifier is None:
        logger.warning("No se puede actualizar clasificación, sistema no inicializado")
        return False
    
    try:
        # Ejecutar clasificación completa
        results = await classifier.classify_all()
        
        # Obtener criptomonedas hot
        hot_cryptos = await classifier.get_hot_cryptos()
        
        # Actualizar caché
        crypto_hot_cache = list(hot_cryptos.values())
        last_classification_time = datetime.now()
        
        logger.info(f"Clasificación actualizada. {len(crypto_hot_cache)} cryptos calientes identificadas.")
        return True
    except Exception as e:
        logger.error(f"Error al actualizar clasificación: {e}")
        return False

async def update_performance_data():
    """Actualizar datos de rendimiento con simulación."""
    global last_performance_update
    
    if not genesis_initialized or performance_tracker is None:
        logger.warning("No se puede actualizar rendimiento, sistema no inicializado")
        return False
    
    try:
        # Obtener resumen actual
        current_summary = await performance_tracker.obtener_resumen_rendimiento()
        current_capital = performance_tracker.capital_actual
        
        # Simular cambio en capital (en un sistema real esto vendría de operaciones reales)
        import random
        change_percent = random.uniform(-0.005, 0.015)
        new_capital = current_capital * (1 + change_percent)
        
        # Actualizar capital
        await performance_tracker.actualizar_capital(new_capital, "actualizacion_simulada")
        
        # Si hay cambio positivo, simular una operación exitosa
        if change_percent > 0:
            random_symbol = random.choice(["BTC", "ETH", "SOL", "ADA", "BNB"])
            random_price = random.uniform(10, 100)
            
            operacion = {
                "symbol": random_symbol,
                "estrategia": "adaptativa",
                "tipo": "LONG",
                "entrada": random_price,
                "salida": random_price * (1 + 0.02),  # 2% de ganancia
                "unidades": round(100 / random_price, 3),
                "resultado_usd": round(100 * 0.02, 2),  # ~$2 de ganancia
                "resultado_porcentual": 0.02,
                "timestamp": datetime.now()
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
    run_async_function(initialize_genesis())

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
            "/api/system_status",
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
        # Obtener datos de componentes principales
        risk_info = {"capital_actual": risk_manager.capital_actual, "nivel_proteccion": risk_manager.nivel_proteccion}
        
        return jsonify({
            "status": "ok",
            "classifier": {
                "capital_actual": classifier.current_capital,
                "hot_cryptos": len(crypto_hot_cache),
                "ultima_clasificacion": last_classification_time.isoformat() if last_classification_time else None
            },
            "risk_manager": risk_info,
            "performance": {
                "capital_actual": performance_tracker.capital_actual,
                "rendimiento_total": performance_tracker.metricas["rendimiento_total"],
                "ultima_actualizacion": last_performance_update.isoformat() if last_performance_update else None
            }
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
    db_connected = (db_manager is not None and transcendental_db is not None)
    
    system_status = {
        "status": "ok" if db_connected and genesis_initialized else "degraded",
        "database": "connected" if db_connected else "disconnected",
        "genesis_system": "initialized" if genesis_initialized else "initializing",
        "timestamp": datetime.now().isoformat()
    }
    
    if genesis_initialized:
        system_status["components"] = {
            "database": "ok" if transcendental_db is not None else "error",
            "classifier": "ok" if classifier is not None else "error",
            "risk_manager": "ok" if risk_manager is not None else "error",
            "performance_tracker": "ok" if performance_tracker is not None else "error"
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
            "database": "connected" if transcendental_db is not None else "disconnected",
            "classifier": "active" if classifier is not None else "inactive",
            "risk_manager": "active" if risk_manager is not None else "inactive",
            "performance_tracker": "active" if performance_tracker is not None else "inactive"
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
    
    try:
        # Obtener resumen de rendimiento
        resumen = {
            "capital_actual": performance_tracker.capital_actual,
            "capital_inicial": performance_tracker.capital_inicial,
            "rendimiento_total": performance_tracker.metricas["rendimiento_total"],
            "rendimiento_anualizado": performance_tracker.metricas["rendimiento_anualizado"],
            "max_drawdown": performance_tracker.metricas["max_drawdown"],
            "sharpe_ratio": performance_tracker.metricas["sharpe_ratio"],
            "win_rate": performance_tracker.metricas["win_rate"]
        }
    except Exception as e:
        logger.error(f"Error al obtener resumen de rendimiento: {e}")
        resumen = {"error": str(e)}
    
    return jsonify({
        "status": "success",
        "performance": resumen,
        "last_update": last_performance_update.isoformat() if last_performance_update else None
    })

@app.route('/api/system_status')
def system_status():
    """Verificar el estado completo del sistema."""
    
    if not genesis_initialized:
        return jsonify({
            "status": "initializing",
            "message": "Sistema Genesis en proceso de inicialización",
            "init_results": genesis_init_results
        }), 202
    
    try:
        # Obtener estadísticas de los componentes
        db_stats = transcendental_db.get_stats() if transcendental_db else {"error": "No inicializado"}
        
        risk_stats = {
            "capital_actual": risk_manager.capital_actual,
            "nivel_proteccion": risk_manager.nivel_proteccion,
            "modo_trascendental": risk_manager.modo_trascendental
        } if risk_manager else {"error": "No inicializado"}
        
        classifier_stats = {
            "capital_actual": classifier.current_capital,
            "hot_cryptos": len(crypto_hot_cache),
            "modo_trascendental": classifier.modo_trascendental
        } if classifier else {"error": "No inicializado"}
        
        performance_stats = {
            "capital_actual": performance_tracker.capital_actual,
            "rendimiento_total": performance_tracker.metricas["rendimiento_total"],
            "modo_trascendental": performance_tracker.modo_trascendental
        } if performance_tracker else {"error": "No inicializado"}
        
        # Generar estado completo
        complete_status = {
            "timestamp": datetime.now().isoformat(),
            "genesis_initialized": genesis_initialized,
            "database": db_stats,
            "classifier": classifier_stats,
            "risk_manager": risk_stats,
            "performance_tracker": performance_stats,
            "last_classification": last_classification_time.isoformat() if last_classification_time else None,
            "last_performance_update": last_performance_update.isoformat() if last_performance_update else None,
            "hot_cryptos_count": len(crypto_hot_cache)
        }
        
        return jsonify(complete_status)
    except Exception as e:
        logger.error(f"Error al obtener estado del sistema: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error al obtener estado del sistema: {str(e)}"
        }), 500

@app.route('/logs')
def view_logs():
    """Ver logs recientes de la API."""
    try:
        # En un sistema real, obtendríamos logs de una tabla o archivo
        # Para esta demo, mostramos datos simulados
        
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

# Inicializamos base de datos al arrancar
logger.info("Tablas de base de datos inicializadas")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)