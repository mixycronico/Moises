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
        # Cargar configuración si existe
        config = {}
        if os.path.exists(app.config["GENESIS_CONFIG"]):
            try:
                with open(app.config["GENESIS_CONFIG"], 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Error al cargar configuración: {e}")
        
        # Añadir configuración para sistema de escalabilidad adaptativa
        if "scaling_config" not in config:
            config["scaling_config"] = {
                "initial_capital": 10000.0,
                "min_efficiency": 0.5,
                "default_model_type": "polynomial"
            }
            
        # Inicializar sistema con la nueva función
        success, components = await initialize_system(transcendental_db, config)
        
        # Almacenar resultados
        genesis_init_results = {
            "success": success,
            "components": list(components.keys()) if components else []
        }
        
        if success:
            genesis_initialized = True
            logger.info("Sistema Genesis inicializado correctamente")
            
            # Ejecutar clasificación inicial en otro hilo
            run_async_function(refresh_classification())
            
            return True
        else:
            logger.error(f"Error al inicializar Sistema Genesis: {genesis_init_results}")
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
        
        # Obtener estadísticas del sistema de escalabilidad si está disponible
        scaling_stats = {}
        if "engine" in genesis_init_results:
            try:
                predictive_engine = genesis_init_results["engine"]
                scaling_stats = {
                    "saturation_points": predictive_engine.get_saturation_points(),
                    "model_count": len(predictive_engine.models),
                    "models": {symbol: model.get_stats() for symbol, model in predictive_engine.models.items()}
                }
            except Exception as e:
                scaling_stats = {"error": f"Error al obtener estadísticas de escalabilidad: {str(e)}"}
        
        # Generar estado completo
        complete_status = {
            "timestamp": datetime.now().isoformat(),
            "genesis_initialized": genesis_initialized,
            "database": db_stats,
            "classifier": classifier_stats,
            "risk_manager": risk_stats,
            "performance_tracker": performance_stats,
            "scaling_system": scaling_stats,
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

@app.route('/api/adaptive_scaling', methods=['GET'])
def adaptive_scaling_status():
    """Obtener información sobre el sistema de escalabilidad adaptativa."""
    if not genesis_initialized:
        return jsonify({
            "status": "initializing",
            "message": "Sistema Genesis en proceso de inicialización"
        }), 202
    
    try:
        # Buscar componentes de escalabilidad en los resultados de inicialización
        components = genesis_init_results.get("components", [])
        
        if "engine" not in components or "scaling_manager" not in components:
            return jsonify({
                "status": "not_available",
                "message": "Sistema de escalabilidad adaptativa no disponible",
                "components_available": components
            }), 404
        
        # Obtener estadísticas del motor predictivo
        engine = components["engine"]
        scaling_manager = components["scaling_manager"]
        
        stats = {
            "engine_stats": engine.get_stats(),
            "current_capital": scaling_manager.get_current_capital(),
            "saturation_points": engine.get_saturation_points()
        }
        
        # Obtener modelos disponibles
        model_info = {}
        for symbol, model in engine.models.items():
            if model.is_trained:
                model_info[symbol] = {
                    "model_type": model.model_type,
                    "r_squared": model.r_squared,
                    "samples_count": model.samples_count,
                    "parameters": model.parameters,
                    "saturation_point": model.saturation_point
                }
        
        stats["models"] = model_info
        
        return jsonify({
            "status": "success",
            "scaling_system": stats
        })
    except Exception as e:
        logger.error(f"Error al obtener información de escalabilidad: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error al obtener información de escalabilidad: {str(e)}"
        }), 500

@app.route('/api/adaptive_scaling/optimize', methods=['POST'])
def optimize_allocation():
    """Optimizar asignación de capital entre instrumentos."""
    if not genesis_initialized:
        return jsonify({
            "status": "initializing",
            "message": "Sistema Genesis en proceso de inicialización"
        }), 202
    
    try:
        # Obtener parámetros del request
        data = request.get_json() or {}
        symbols = data.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"])
        total_capital = float(data.get("total_capital", 10000.0))
        min_efficiency = float(data.get("min_efficiency", 0.5))
        
        # Validar parámetros
        if not symbols or total_capital <= 0 or min_efficiency < 0 or min_efficiency > 1:
            return jsonify({
                "status": "error",
                "message": "Parámetros inválidos",
                "errors": {
                    "symbols": "Debe proporcionar al menos un símbolo" if not symbols else None,
                    "total_capital": "El capital debe ser positivo" if total_capital <= 0 else None,
                    "min_efficiency": "La eficiencia mínima debe estar entre 0 y 1" if min_efficiency < 0 or min_efficiency > 1 else None
                }
            }), 400
        
        # Buscar componentes de escalabilidad en los resultados de inicialización
        components = genesis_init_results.get("components", [])
        
        if "engine" not in components:
            return jsonify({
                "status": "not_available",
                "message": "Sistema de escalabilidad adaptativa no disponible",
                "components_available": components
            }), 404
        
        # Obtener motor predictivo
        engine = components["engine"]
        
        # Ejecutar optimización en otro hilo
        def wrapper():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Ejecutar optimización
                allocations = loop.run_until_complete(engine.optimize_allocation(
                    symbols=symbols,
                    total_capital=total_capital,
                    min_efficiency=min_efficiency
                ))
                
                result = {
                    "status": "success",
                    "allocations": allocations,
                    "total_allocated": sum(allocations.values()),
                    "symbols_used": len([a for a in allocations.values() if a > 0]),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Convertir decimales a flotantes para serialización JSON
                for symbol, amount in result["allocations"].items():
                    result["allocations"][symbol] = float(amount)
                
                return result
            except Exception as e:
                logger.error(f"Error en optimización: {e}")
                return {
                    "status": "error",
                    "message": f"Error en optimización: {str(e)}"
                }
            finally:
                loop.close()
        
        # Ejecutar en otro hilo para no bloquear
        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()
        thread.join(timeout=5.0)  # Esperar hasta 5 segundos por resultado
        
        if thread.is_alive():
            # Si aún está ejecutando, devolver respuesta inmediata
            return jsonify({
                "status": "processing",
                "message": "La optimización está en proceso",
                "parameters": {
                    "symbols": symbols,
                    "total_capital": total_capital,
                    "min_efficiency": min_efficiency
                }
            }), 202
        
        # Si terminó, devolver resultado
        result = wrapper()
        
        if result["status"] == "error":
            return jsonify(result), 500
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error en endpoint de optimización: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error en endpoint de optimización: {str(e)}"
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