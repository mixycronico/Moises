"""
API para integrar el sistema de trading cósmico con el Sistema Genesis.

Este módulo permite que todas las entidades del Sistema Genesis
utilicen las capacidades avanzadas de trading definidas en cosmic_trading.py,
incluyendo la colaboración avanzada y el intercambio de conocimiento entre entidades.
"""

import os
import time
import logging
import random
from datetime import datetime
from flask import jsonify, request
from cosmic_trading import initialize_cosmic_trading, get_db_pool

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

def initialize(father_name="otoniel", include_extended_entities=False):
    """
    Inicializar el sistema de trading cósmico.
    
    Args:
        father_name: Nombre del creador/dueño del sistema
        include_extended_entities: Si es True, incluye entidades adicionales avanzadas
        
    Returns:
        True si se inicializó correctamente
    """
    global cosmic_network, aetherion_trader, lunareth_trader, initialized
    
    if initialized:
        logger.info("Sistema de trading ya estaba inicializado")
        return True
        
    try:
        cosmic_network, aetherion_trader, lunareth_trader = initialize_cosmic_trading(
            father_name=father_name,
            include_extended_entities=include_extended_entities
        )
        initialized = True
        
        if include_extended_entities:
            logger.info(f"Sistema de trading extendido inicializado correctamente para {father_name}")
        else:
            logger.info(f"Sistema de trading básico inicializado correctamente para {father_name}")
            
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

def get_collaboration_metrics():
    """
    Obtener métricas de colaboración de la red de trading.
    
    Returns:
        Dict con métricas de colaboración
    """
    if not is_initialized():
        return {"error": "Sistema no inicializado"}
    
    try:
        return cosmic_network.get_collaboration_metrics()
    except Exception as e:
        logger.error(f"Error al obtener métricas de colaboración: {e}")
        return {"error": str(e)}

def get_knowledge_pool(knowledge_type=None, limit=10):
    """
    Obtener conocimiento compartido en el pool colectivo.
    
    Args:
        knowledge_type: Tipo de conocimiento a filtrar (opcional)
        limit: Número máximo de registros a retornar
        
    Returns:
        Lista de conocimientos compartidos
    """
    if not is_initialized():
        return {"error": "Sistema no inicializado"}
    
    try:
        pool = get_db_pool()
        conn = pool.getconn()
        results = []
        
        with conn.cursor() as c:
            if knowledge_type:
                c.execute('''
                    SELECT id, entity_name, entity_role, knowledge_type, 
                           knowledge_value, timestamp 
                    FROM knowledge_pool 
                    WHERE knowledge_type = %s
                    ORDER BY timestamp DESC LIMIT %s
                ''', (knowledge_type, limit))
            else:
                c.execute('''
                    SELECT id, entity_name, entity_role, knowledge_type, 
                           knowledge_value, timestamp 
                    FROM knowledge_pool 
                    ORDER BY timestamp DESC LIMIT %s
                ''', (limit,))
                
            for row in c.fetchall():
                results.append({
                    "id": row[0],
                    "entity_name": row[1],
                    "entity_role": row[2],
                    "knowledge_type": row[3],
                    "knowledge_value": row[4],
                    "timestamp": row[5].isoformat() if row[5] else None
                })
        
        pool.putconn(conn)
        return results
    except Exception as e:
        logger.error(f"Error al obtener pool de conocimiento: {e}")
        return {"error": str(e)}

def trigger_network_collaboration(entity_name=None):
    """
    Disparar colaboración proactiva en la red.
    
    Args:
        entity_name: Nombre de entidad específica o None para colaboración global
        
    Returns:
        Resultados de la colaboración
    """
    if not is_initialized():
        return {"error": "Sistema no inicializado"}
    
    try:
        results = []
        
        if entity_name:
            # Buscar entidad específica
            found = False
            for entity in cosmic_network.entities:
                if entity.name.lower() == entity_name.lower():
                    result = entity.collaborate()
                    results.append({
                        "entity_name": entity.name,
                        "result": result
                    })
                    found = True
                    break
            
            if not found:
                return {"error": f"Entidad {entity_name} no encontrada"}
        else:
            # Colaboración de toda la red
            for entity in cosmic_network.entities:
                result = entity.collaborate()
                results.append({
                    "entity_name": entity.name,
                    "result": result
                })
        
        return {
            "collaboration_count": len([r for r in results if r["result"] is not None]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error al disparar colaboración: {e}")
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