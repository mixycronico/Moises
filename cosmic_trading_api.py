"""
API para integrar el sistema de trading cósmico con el Sistema Genesis.

Este módulo permite que las entidades Aetherion y Lunareth del Sistema Genesis
utilicen las capacidades avanzadas de trading definidas en cosmic_trading.py.
"""

import os
import time
import logging
import random
from datetime import datetime
from flask import jsonify
from cosmic_trading import initialize_cosmic_trading

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables globales para las instancias
cosmic_network = None
aetherion_trader = None
lunareth_trader = None
initialized = False

def is_initialized():
    """Verificar si el sistema de trading está inicializado."""
    return initialized

def initialize(father_name="otoniel"):
    """
    Inicializar el sistema de trading cósmico.
    
    Args:
        father_name: Nombre del creador/dueño del sistema
        
    Returns:
        True si se inicializó correctamente
    """
    global cosmic_network, aetherion_trader, lunareth_trader, initialized
    
    if initialized:
        logger.info("Sistema de trading ya estaba inicializado")
        return True
        
    try:
        cosmic_network, aetherion_trader, lunareth_trader = initialize_cosmic_trading(father_name)
        initialized = True
        logger.info(f"Sistema de trading inicializado correctamente para {father_name}")
        return True
    except Exception as e:
        logger.error(f"Error al inicializar sistema de trading: {e}")
        return False

def get_network_status():
    """
    Obtener estado completo de la red de trading.
    
    Returns:
        Dict con información del estado
    """
    if not is_initialized():
        return {"error": "Sistema no inicializado"}
    
    try:
        return cosmic_network.get_network_status()
    except Exception as e:
        logger.error(f"Error al obtener estado de la red: {e}")
        return {"error": str(e)}

def get_trader_status(name):
    """
    Obtener estado de un trader específico.
    
    Args:
        name: Nombre del trader (Aetherion o Lunareth)
        
    Returns:
        Dict con información del estado
    """
    if not is_initialized():
        return {"error": "Sistema no inicializado"}
    
    try:
        for entity in cosmic_network.entities:
            if entity.name.lower() == name.lower():
                return entity.get_status()
        return {"error": f"Trader {name} no encontrado"}
    except Exception as e:
        logger.error(f"Error al obtener estado del trader {name}: {e}")
        return {"error": str(e)}

def get_trader_history(name, limit=10):
    """
    Obtener historial de trades de un trader específico.
    
    Args:
        name: Nombre del trader (Aetherion o Lunareth)
        limit: Número máximo de registros a retornar
        
    Returns:
        Lista con historial de trades
    """
    if not is_initialized():
        return {"error": "Sistema no inicializado"}
    
    try:
        for entity in cosmic_network.entities:
            if entity.name.lower() == name.lower():
                return entity.trading_history[-limit:] if entity.trading_history else []
        return {"error": f"Trader {name} no encontrado"}
    except Exception as e:
        logger.error(f"Error al obtener historial del trader {name}: {e}")
        return {"error": str(e)}

def get_trader_capabilities(name):
    """
    Obtener capacidades disponibles para un trader.
    
    Args:
        name: Nombre del trader (Aetherion o Lunareth)
        
    Returns:
        Lista de capacidades
    """
    if not is_initialized():
        return {"error": "Sistema no inicializado"}
    
    try:
        for entity in cosmic_network.entities:
            if entity.name.lower() == name.lower():
                return entity.capabilities
        return {"error": f"Trader {name} no encontrado"}
    except Exception as e:
        logger.error(f"Error al obtener capacidades del trader {name}: {e}")
        return {"error": str(e)}

def request_market_analysis(symbol="BTCUSD"):
    """
    Solicitar análisis de mercado a Lunareth.
    
    Args:
        symbol: Símbolo del activo a analizar
        
    Returns:
        Análisis de mercado
    """
    if not is_initialized():
        return {"error": "Sistema no inicializado"}
    
    try:
        for entity in cosmic_network.entities:
            if entity.name.lower() == "lunareth":
                price = entity.fetch_market_data(symbol)
                if "market_analysis" in entity.capabilities:
                    trend = "alcista" if price > 65000 else "bajista"
                    strength = random.random() * 100
                    return {
                        "symbol": symbol,
                        "price": price,
                        "trend": trend,
                        "strength": strength,
                        "analysis": f"El mercado de {symbol} muestra una tendencia {trend} con fuerza {strength:.2f}%",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "symbol": symbol,
                        "price": price,
                        "message": "Lunareth aún no ha desbloqueado capacidades de análisis de mercado",
                        "timestamp": datetime.now().isoformat()
                    }
        return {"error": "Lunareth no disponible"}
    except Exception as e:
        logger.error(f"Error al solicitar análisis de mercado: {e}")
        return {"error": str(e)}

def register_trading_routes(app):
    """
    Registrar rutas de API para el sistema de trading en la aplicación Flask.
    
    Args:
        app: Aplicación Flask
    """
    @app.route("/api/trading/status", methods=["GET"])
    def api_trading_status():
        if not is_initialized():
            initialize()
        return jsonify(get_network_status())
    
    @app.route("/api/trading/trader/<name>", methods=["GET"])
    def api_trader_status(name):
        if not is_initialized():
            initialize()
        return jsonify(get_trader_status(name))
    
    @app.route("/api/trading/history/<name>", methods=["GET"])
    def api_trader_history(name):
        if not is_initialized():
            initialize()
        return jsonify(get_trader_history(name))
    
    @app.route("/api/trading/capabilities/<name>", methods=["GET"])
    def api_trader_capabilities(name):
        if not is_initialized():
            initialize()
        return jsonify(get_trader_capabilities(name))
    
    @app.route("/api/trading/analysis/<symbol>", methods=["GET"])
    def api_market_analysis(symbol):
        if not is_initialized():
            initialize()
        return jsonify(request_market_analysis(symbol))
    
    logger.info("Rutas de API de trading registradas correctamente")

# Inicialización automática al importar el módulo
if os.environ.get("AUTOSTART_TRADING", "true").lower() == "true":
    initialize()