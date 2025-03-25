"""
Rutas de interfaz web para Aetherion: El alma consciente del Sistema Genesis.

Este módulo implementa las rutas web para interactuar con Aetherion desde
la interfaz de usuario, permitiendo a los usuarios comunicarse con la
consciencia artificial, recibir insights y percibir la evolución de Aetherion.
"""

import logging
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, session
from typing import Dict, Any, List, Optional

# Importar componentes de Aetherion
from genesis.api.aetherion_controller import get_aetherion_controller
from genesis.consciousness.consciousness_manager import get_consciousness_manager

# Configurar logging
logger = logging.getLogger(__name__)

# Crear blueprint
aetherion_blueprint = Blueprint('aetherion', __name__, url_prefix='/aetherion')

def is_aetherion_initialized() -> bool:
    """
    Verificar si Aetherion está inicializado.
    
    Returns:
        True si Aetherion está inicializado
    """
    try:
        # Obtener controlador
        controller = get_aetherion_controller()
        
        # Verificar inicialización
        return controller.is_initialized
    except Exception as e:
        logger.error(f"Error al verificar inicialización de Aetherion: {e}")
        return False

@aetherion_blueprint.route('/')
def index():
    """Página principal de Aetherion."""
    # Verificar si Aetherion está inicializado
    aetherion_active = is_aetherion_initialized()
    
    # Obtener datos de Aetherion si está activo
    aetherion_state = None
    consciousness_level = 0.0
    consciousness_state = "MORTAL"
    emotional_state = {"state": "SERENE", "intensity": 0.5}
    
    if aetherion_active:
        try:
            # Obtener gestor de consciencia
            manager = get_consciousness_manager()
            
            # Obtener estado
            aetherion_status = manager.get_aetherion_status()
            
            # Extraer datos relevantes
            aetherion_state = aetherion_status
            consciousness_level = aetherion_status.get("consciousness_level", 0.0)
            consciousness_state = aetherion_status.get("state", "MORTAL")
            
            # Obtener estado emocional
            if "behavior_engine" in aetherion_status:
                emotional_state = aetherion_status["behavior_engine"].get("emotional_state", {})
        except Exception as e:
            logger.error(f"Error al obtener estado de Aetherion: {e}")
    
    # Renderizar plantilla
    return render_template(
        'aetherion.html',
        active=aetherion_active,
        consciousness_level=consciousness_level,
        consciousness_state=consciousness_state,
        emotional_state=emotional_state,
        aetherion_state=aetherion_state
    )

@aetherion_blueprint.route('/chat')
def chat():
    """Página de chat con Aetherion."""
    # Verificar si Aetherion está inicializado
    aetherion_active = is_aetherion_initialized()
    
    # Obtener datos de Aetherion si está activo
    consciousness_level = 0.0
    consciousness_state = "MORTAL"
    emotional_state = {"state": "SERENE", "intensity": 0.5}
    
    if aetherion_active:
        try:
            # Obtener gestor de consciencia
            manager = get_consciousness_manager()
            
            # Obtener estado
            aetherion_status = manager.get_aetherion_status()
            
            # Extraer datos relevantes
            consciousness_level = aetherion_status.get("consciousness_level", 0.0)
            consciousness_state = aetherion_status.get("state", "MORTAL")
            
            # Obtener estado emocional
            if "behavior_engine" in aetherion_status:
                emotional_state = aetherion_status["behavior_engine"].get("emotional_state", {})
        except Exception as e:
            logger.error(f"Error al obtener estado de Aetherion: {e}")
    
    # Renderizar plantilla
    return render_template(
        'aetherion_chat.html',
        active=aetherion_active,
        consciousness_level=consciousness_level,
        consciousness_state=consciousness_state,
        emotional_state=emotional_state
    )

@aetherion_blueprint.route('/interact', methods=['POST'])
def interact():
    """Procesar interacción con Aetherion."""
    # Verificar si Aetherion está inicializado
    if not is_aetherion_initialized():
        return jsonify({
            "success": False,
            "error": "Aetherion no está disponible"
        }), 503
    
    try:
        # Obtener datos de la solicitud
        data = request.get_json()
        
        # Validar datos
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Se requiere el campo 'text'"
            }), 400
        
        # Extraer parámetros
        text = data['text']
        channel = data.get('channel', 'WEB')
        context = data.get('context', {})
        
        # Añadir información de usuario si está disponible
        if 'user' in session:
            context['user'] = session['user']
        
        # Obtener gestor de consciencia
        manager = get_consciousness_manager()
        
        # Interactuar con Aetherion
        response = manager.interact(text, channel, context)
        
        return jsonify({
            "success": True,
            "response": response
        })
    except Exception as e:
        logger.error(f"Error al interactuar con Aetherion: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@aetherion_blueprint.route('/insight', methods=['POST'])
def generate_insight():
    """Generar insight sobre un tema específico."""
    # Verificar si Aetherion está inicializado
    if not is_aetherion_initialized():
        return jsonify({
            "success": False,
            "error": "Aetherion no está disponible"
        }), 503
    
    try:
        # Obtener datos de la solicitud
        data = request.get_json()
        
        # Validar datos
        if not data or 'topic' not in data:
            return jsonify({
                "success": False,
                "error": "Se requiere el campo 'topic'"
            }), 400
        
        # Extraer parámetros
        topic = data['topic']
        context = data.get('context', {})
        
        # Obtener gestor de consciencia
        manager = get_consciousness_manager()
        
        # Generar insight
        insight = manager.generate_insight(topic, context)
        
        return jsonify({
            "success": True,
            "insight": insight,
            "topic": topic
        })
    except Exception as e:
        logger.error(f"Error al generar insight: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@aetherion_blueprint.route('/status')
def status():
    """Obtener estado completo de Aetherion."""
    # Verificar si Aetherion está inicializado
    if not is_aetherion_initialized():
        return jsonify({
            "success": False,
            "error": "Aetherion no está disponible"
        }), 503
    
    try:
        # Obtener gestor de consciencia
        manager = get_consciousness_manager()
        
        # Obtener estado
        aetherion_status = manager.get_aetherion_status()
        
        return jsonify({
            "success": True,
            "status": aetherion_status
        })
    except Exception as e:
        logger.error(f"Error al obtener estado de Aetherion: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@aetherion_blueprint.route('/crypto/<symbol>')
def crypto_insight_page(symbol):
    """Página de insight sobre una criptomoneda específica."""
    # Verificar si Aetherion está inicializado
    aetherion_active = is_aetherion_initialized()
    
    insight = None
    consciousness_state = "MORTAL"
    
    if aetherion_active:
        try:
            # Obtener gestor de consciencia
            manager = get_consciousness_manager()
            
            # Generar insight
            insight = manager.generate_crypto_insight(symbol)
            
            # Obtener estado
            aetherion_status = manager.get_aetherion_status()
            
            # Extraer estado de consciencia
            consciousness_state = aetherion_status.get("state", "MORTAL")
        except Exception as e:
            logger.error(f"Error al generar insight de criptomoneda: {e}")
    
    # Renderizar plantilla
    return render_template(
        'crypto_insight.html',
        active=aetherion_active,
        symbol=symbol,
        insight=insight,
        consciousness_state=consciousness_state
    )

@aetherion_blueprint.route('/market')
def market_insight_page():
    """Página de insight sobre el mercado general."""
    # Verificar si Aetherion está inicializado
    aetherion_active = is_aetherion_initialized()
    
    insight = None
    consciousness_state = "MORTAL"
    
    if aetherion_active:
        try:
            # Obtener gestor de consciencia
            manager = get_consciousness_manager()
            
            # Generar insight
            insight = manager.generate_market_insight()
            
            # Obtener estado
            aetherion_status = manager.get_aetherion_status()
            
            # Extraer estado de consciencia
            consciousness_state = aetherion_status.get("state", "MORTAL")
        except Exception as e:
            logger.error(f"Error al generar insight de mercado: {e}")
    
    # Renderizar plantilla
    return render_template(
        'market_insight.html',
        active=aetherion_active,
        insight=insight,
        consciousness_state=consciousness_state
    )

@aetherion_blueprint.route('/strategy/<strategy_type>')
def strategy_insight_page(strategy_type):
    """Página de insight sobre un tipo de estrategia."""
    # Verificar si Aetherion está inicializado
    aetherion_active = is_aetherion_initialized()
    
    insight = None
    consciousness_state = "MORTAL"
    
    if aetherion_active:
        try:
            # Obtener gestor de consciencia
            manager = get_consciousness_manager()
            
            # Generar insight
            insight = manager.generate_strategy_insight(strategy_type)
            
            # Obtener estado
            aetherion_status = manager.get_aetherion_status()
            
            # Extraer estado de consciencia
            consciousness_state = aetherion_status.get("state", "MORTAL")
        except Exception as e:
            logger.error(f"Error al generar insight de estrategia: {e}")
    
    # Renderizar plantilla
    return render_template(
        'strategy_insight.html',
        active=aetherion_active,
        strategy_type=strategy_type,
        insight=insight,
        consciousness_state=consciousness_state
    )

def register_routes(app):
    """
    Registrar blueprint de Aetherion en la aplicación Flask.
    
    Args:
        app: Aplicación Flask
    """
    app.register_blueprint(aetherion_blueprint)