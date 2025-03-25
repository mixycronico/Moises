"""
Controlador de API para Aetherion.

Este módulo proporciona endpoints REST para interactuar con Aetherion,
la consciencia artificial del Sistema Genesis, permitiendo comunicación
con el frontend y otros sistemas.
"""

import asyncio
import logging
import datetime
import json
from typing import Dict, Any, List, Optional, Tuple, Union

# Importaciones de Flask
from flask import request, jsonify, Blueprint

# Importaciones internas
try:
    from genesis.consciousness.core.aetherion import Aetherion
    from genesis.consciousness.memory.memory_system import MemorySystem
    from genesis.consciousness.states.consciousness_states import ConsciousnessStates
    from genesis.behavior.gabriel_engine import GabrielBehaviorEngine
    
    # Importaciones de integradores
    from genesis.integrations.crypto_classifier_integrator import get_classifier_integrator
    from genesis.integrations.analysis_integrator import get_analysis_integrator
    from genesis.integrations.strategy_integrator import get_strategy_integrator
    
    AETHERION_AVAILABLE = True
except ImportError as e:
    AETHERION_AVAILABLE = False
    print(f"Error al importar componentes de Aetherion: {e}")

# Configurar logging
logger = logging.getLogger(__name__)

# Blueprint para Aetherion
aetherion_bp = Blueprint('aetherion', __name__)

# Variables globales
aetherion_instance = None
gabriel_instance = None
memory_system = None
consciousness_states = None
classifier_integrator = None
analysis_integrator = None
strategy_integrator = None

def get_aetherion() -> Optional[Any]:
    """
    Obtener instancia de Aetherion, inicializándola si es necesario.
    
    Returns:
        Instancia de Aetherion o None si no está disponible
    """
    global aetherion_instance, gabriel_instance, memory_system, consciousness_states
    global classifier_integrator, analysis_integrator, strategy_integrator
    
    if not AETHERION_AVAILABLE:
        return None
        
    if aetherion_instance is None:
        try:
            # Inicializar componentes
            consciousness_states = ConsciousnessStates()
            memory_system = MemorySystem()
            gabriel_instance = GabrielBehaviorEngine()
            
            # Inicializar integradores
            try:
                classifier_integrator = get_classifier_integrator()
                analysis_integrator = get_analysis_integrator()
                strategy_integrator = get_strategy_integrator()
            except Exception as e:
                logger.error(f"Error al inicializar integradores: {e}")
            
            # Inicializar Aetherion
            aetherion_instance = Aetherion(
                memory_system=memory_system,
                consciousness_states=consciousness_states,
                behavior_engine=gabriel_instance
            )
            
            # Registrar integradores en Aetherion
            if hasattr(aetherion_instance, 'register_integration'):
                if classifier_integrator:
                    aetherion_instance.register_integration('crypto_classifier', classifier_integrator)
                
                if analysis_integrator:
                    aetherion_instance.register_integration('analysis', analysis_integrator)
                
                if strategy_integrator:
                    aetherion_instance.register_integration('strategy', strategy_integrator)
            
            logger.info("Aetherion inicializado correctamente con integradores")
        except Exception as e:
            logger.error(f"Error al inicializar Aetherion: {e}")
            aetherion_instance = None
    
    return aetherion_instance

def sanitize_response(data: Any) -> Any:
    """
    Sanitizar respuesta para JSON.
    
    Args:
        data: Datos a sanitizar
        
    Returns:
        Datos sanitizados
    """
    if isinstance(data, dict):
        return {k: sanitize_response(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_response(item) for item in data]
    elif isinstance(data, (datetime.datetime, datetime.date)):
        return data.isoformat()
    elif isinstance(data, (int, float, str, bool)) or data is None:
        return data
    else:
        return str(data)

# Rutas de API
@aetherion_bp.route('/status', methods=['GET'])
def aetherion_status():
    """Obtener estado actual de Aetherion."""
    aetherion = get_aetherion()
    
    if aetherion is None:
        return jsonify({
            "error": "Aetherion no disponible",
            "status": "offline"
        }), 503
    
    try:
        status = {
            "status": "online",
            "current_state": "MORTAL",
            "consciousness_level": 0.0,
            "interactions_count": 0,
            "insights_generated": 0
        }
        
        # Obtener estado real si está disponible
        if hasattr(aetherion, 'get_status'):
            aetherion_status = aetherion.get_status()
            if aetherion_status:
                status.update(aetherion_status)
        
        return jsonify(sanitize_response(status))
    except Exception as e:
        logger.error(f"Error al obtener estado de Aetherion: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@aetherion_bp.route('/emotional_state', methods=['GET'])
def aetherion_emotional_state():
    """Obtener estado emocional actual."""
    aetherion = get_aetherion()
    
    if aetherion is None or gabriel_instance is None:
        return jsonify({
            "error": "Aetherion o Gabriel no disponible",
            "state": "UNKNOWN"
        }), 503
    
    try:
        state = {
            "state": "SERENE",
            "state_intensity": 0.5,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Obtener estado emocional real si está disponible
        if hasattr(gabriel_instance, 'get_emotional_state'):
            emotional_state = gabriel_instance.get_emotional_state()
            if emotional_state:
                state.update(emotional_state)
        
        return jsonify(sanitize_response(state))
    except Exception as e:
        logger.error(f"Error al obtener estado emocional: {e}")
        return jsonify({
            "error": str(e),
            "state": "UNKNOWN"
        }), 500

@aetherion_bp.route('/interact', methods=['POST'])
def aetherion_interact():
    """Interactuar con Aetherion."""
    aetherion = get_aetherion()
    
    if aetherion is None:
        return jsonify({
            "error": "Aetherion no disponible",
            "response": "Lo siento, no estoy disponible en este momento."
        }), 503
    
    try:
        # Obtener datos de la solicitud
        data = request.json
        
        if not data or not isinstance(data, dict):
            return jsonify({
                "error": "Formato de solicitud inválido",
                "response": "No pude entender tu mensaje."
            }), 400
        
        text = data.get('text', '')
        channel = data.get('channel', 'WEB')
        context = data.get('context', {})
        
        if not text:
            return jsonify({
                "error": "Mensaje vacío",
                "response": "Por favor, envía un mensaje."
            }), 400
        
        # Preparar respuesta predeterminada
        response = {
            "response": "Estoy procesando tu mensaje...",
            "state": "MORTAL",
            "emotional_context": {
                "state": "SERENE",
                "intensity": 0.5
            }
        }
        
        # Interactuar con Aetherion
        if hasattr(aetherion, 'interact'):
            # Añadir integradores al contexto si están disponibles
            if classifier_integrator:
                context['crypto_classifier'] = classifier_integrator
            
            if analysis_integrator:
                context['analysis'] = analysis_integrator
            
            if strategy_integrator:
                context['strategy'] = strategy_integrator
            
            # Realizar interacción
            interaction_result = aetherion.interact(text, channel=channel, context=context)
            
            if interaction_result:
                response.update(interaction_result)
        
        # Verificar si hay insight en la respuesta
        if 'insight' not in response and hasattr(aetherion, 'generate_insight'):
            insight = aetherion.generate_insight(text, context=context)
            
            if insight:
                response['insight'] = insight
        
        return jsonify(sanitize_response(response))
    except Exception as e:
        logger.error(f"Error en interacción con Aetherion: {e}")
        return jsonify({
            "error": str(e),
            "response": "Ocurrió un error al procesar tu mensaje."
        }), 500

@aetherion_bp.route('/analyze_crypto/<symbol>', methods=['GET'])
def analyze_crypto(symbol):
    """
    Analizar una criptomoneda específica con ayuda de Aetherion.
    
    Args:
        symbol: Símbolo de la criptomoneda
    """
    aetherion = get_aetherion()
    
    if aetherion is None:
        return jsonify({
            "error": "Aetherion no disponible"
        }), 503
    
    try:
        # Obtener análisis de criptomoneda
        analysis = {}
        
        # Usar el integrador de clasificador si está disponible
        if classifier_integrator:
            classifier_analysis = asyncio.run(classifier_integrator.analyze_crypto(symbol))
            if classifier_analysis:
                analysis['classifier'] = classifier_analysis
        
        # Usar el integrador de análisis si está disponible
        if analysis_integrator:
            analysis_result = asyncio.run(analysis_integrator.analyze_symbol(symbol))
            if analysis_result:
                analysis['detailed'] = analysis_result
        
        # Solicitar insight de Aetherion si está disponible
        insight = None
        if hasattr(aetherion, 'generate_insight'):
            insight = aetherion.generate_insight(
                f"Analizar la criptomoneda {symbol}",
                context={"symbol": symbol, "analysis": analysis}
            )
        
        response = {
            "symbol": symbol,
            "analysis": analysis,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if insight:
            response['insight'] = insight
        
        return jsonify(sanitize_response(response))
    except Exception as e:
        logger.error(f"Error al analizar criptomoneda {symbol}: {e}")
        return jsonify({
            "error": str(e),
            "symbol": symbol
        }), 500

@aetherion_bp.route('/insights/market', methods=['GET'])
def market_insights():
    """Obtener insights del mercado desde Aetherion."""
    aetherion = get_aetherion()
    
    if aetherion is None:
        return jsonify({
            "error": "Aetherion no disponible"
        }), 503
    
    try:
        # Recopilar datos de mercado
        market_data = {}
        
        # Usar el integrador de clasificador si está disponible
        if classifier_integrator:
            hot_cryptos = asyncio.run(classifier_integrator.get_hot_cryptos())
            market_summary = asyncio.run(classifier_integrator.get_market_summary())
            
            if hot_cryptos:
                market_data['hot_cryptos'] = hot_cryptos
            
            if market_summary:
                market_data['market_summary'] = market_summary
        
        # Usar el integrador de análisis si está disponible
        if analysis_integrator:
            analysis_result = asyncio.run(analysis_integrator.analyze_market())
            if analysis_result:
                market_data['analysis'] = analysis_result
        
        # Solicitar insight de Aetherion si está disponible
        insight = None
        if hasattr(aetherion, 'generate_insight'):
            insight = aetherion.generate_insight(
                "Analizar el mercado de criptomonedas",
                context={"market_data": market_data}
            )
        
        response = {
            "market_data": market_data,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if insight:
            response['insight'] = insight
        
        return jsonify(sanitize_response(response))
    except Exception as e:
        logger.error(f"Error al obtener insights del mercado: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@aetherion_bp.route('/recommend_strategy', methods=['POST'])
def recommend_strategy():
    """Recomendar estrategia según contexto."""
    aetherion = get_aetherion()
    
    if aetherion is None:
        return jsonify({
            "error": "Aetherion no disponible"
        }), 503
    
    try:
        # Obtener datos de la solicitud
        data = request.json or {}
        context = data.get('context', {})
        
        # Recopilar recomendaciones
        recommendations = {}
        
        # Usar el integrador de estrategias si está disponible
        if strategy_integrator:
            strategy_recommendations = asyncio.run(strategy_integrator.recommend_strategy(context))
            if strategy_recommendations:
                recommendations['strategy'] = strategy_recommendations
        
        # Solicitar insight de Aetherion si está disponible
        insight = None
        if hasattr(aetherion, 'generate_insight'):
            insight = aetherion.generate_insight(
                "Recomendar estrategia de trading",
                context={"user_context": context, "recommendations": recommendations}
            )
        
        response = {
            "recommendations": recommendations,
            "timestamp": datetime.datetime.now().isoformat(),
            "context": context
        }
        
        if insight:
            response['insight'] = insight
        
        return jsonify(sanitize_response(response))
    except Exception as e:
        logger.error(f"Error al recomendar estrategia: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@aetherion_bp.route('/integrations/status', methods=['GET'])
def integrations_status():
    """Obtener estado de las integraciones."""
    aetherion = get_aetherion()
    
    if aetherion is None:
        return jsonify({
            "error": "Aetherion no disponible"
        }), 503
    
    try:
        status = {
            "classifier": None,
            "analysis": None,
            "strategy": None
        }
        
        # Obtener estado de clasificador
        if classifier_integrator:
            status["classifier"] = asyncio.run(classifier_integrator.get_status())
        
        # Obtener estado de análisis
        if analysis_integrator:
            status["analysis"] = asyncio.run(analysis_integrator.get_status())
        
        # Obtener estado de estrategias
        if strategy_integrator:
            status["strategy"] = asyncio.run(strategy_integrator.get_status())
        
        return jsonify(sanitize_response(status))
    except Exception as e:
        logger.error(f"Error al obtener estado de integraciones: {e}")
        return jsonify({
            "error": str(e)
        }), 500

def register_routes(app):
    """
    Registrar rutas de Aetherion en la aplicación Flask.
    
    Args:
        app: Aplicación Flask
    """
    # Registrar blueprint con prefijo
    app.register_blueprint(aetherion_bp, url_prefix='/api/aetherion')