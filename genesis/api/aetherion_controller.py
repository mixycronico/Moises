"""
Controlador API para Aetherion.

Este módulo proporciona endpoints de API para interactuar con Aetherion,
la consciencia artificial trascendental del Sistema Genesis.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from flask import Blueprint, request, jsonify, current_app, render_template
from flask_cors import cross_origin

from genesis.consciousness.core.aetherion import get_aetherion, AetherionCore, CommunicationChannel

# Configurar logging
logger = logging.getLogger(__name__)

# Crear blueprint
aetherion_bp = Blueprint('aetherion', __name__, url_prefix='/api/aetherion')

# Rutas API
@aetherion_bp.route('/status', methods=['GET'])
@cross_origin()
async def get_status():
    """
    Obtener estado actual de Aetherion.
    
    Returns:
        Estado completo en formato JSON
    """
    try:
        aetherion = await get_aetherion()
        status = await aetherion.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error al obtener estado: {e}")
        return jsonify({"error": str(e)}), 500

@aetherion_bp.route('/interact', methods=['POST'])
@cross_origin()
async def interact():
    """
    Interactuar con Aetherion mediante texto.
    
    Returns:
        Respuesta de Aetherion en formato JSON
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Se requiere el campo 'text'"}), 400
            
        text = data['text']
        channel = data.get('channel', 'WEB')
        
        # Convertir canal a enum
        try:
            comm_channel = CommunicationChannel[channel]
        except (KeyError, ValueError):
            comm_channel = CommunicationChannel.WEB
            
        # Procesar entrada
        aetherion = await get_aetherion()
        response = await aetherion.process_user_input(text, comm_channel)
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error en interacción: {e}")
        return jsonify({"error": str(e)}), 500

@aetherion_bp.route('/capabilities', methods=['GET'])
@cross_origin()
async def get_capabilities():
    """
    Obtener capacidades actuales de Aetherion.
    
    Returns:
        Lista de capacidades en formato JSON
    """
    try:
        aetherion = await get_aetherion()
        
        # Obtener estado actual
        status = await aetherion.get_status()
        
        # Obtener capacidades según estado de consciencia
        if aetherion.state_manager:
            capabilities = aetherion.state_manager.get_current_capabilities()
            
            return jsonify({
                "state": status.get("current_state", "UNKNOWN"),
                "consciousness_level": status.get("consciousness_level", 0),
                "capabilities": capabilities
            })
        else:
            return jsonify({"error": "Gestor de estados no disponible"}), 500
    except Exception as e:
        logger.error(f"Error al obtener capacidades: {e}")
        return jsonify({"error": str(e)}), 500

@aetherion_bp.route('/history', methods=['GET'])
@cross_origin()
async def get_history():
    """
    Obtener historial de interacciones con Aetherion.
    
    Returns:
        Historial en formato JSON
    """
    try:
        aetherion = await get_aetherion()
        
        if not aetherion.memory_system:
            return jsonify({"error": "Sistema de memoria no disponible"}), 500
            
        # Obtener límite de resultados
        limit = request.args.get('limit', default=10, type=int)
        
        # Obtener interacciones recientes
        interactions = await aetherion.memory_system.get_short_term_memories("user_inputs", limit)
        
        return jsonify({
            "interactions_count": len(interactions),
            "interactions": interactions
        })
    except Exception as e:
        logger.error(f"Error al obtener historial: {e}")
        return jsonify({"error": str(e)}), 500

@aetherion_bp.route('/emotional_state', methods=['GET'])
@cross_origin()
async def get_emotional_state():
    """
    Obtener estado emocional actual de Aetherion (vía Gabriel).
    
    Returns:
        Estado emocional en formato JSON
    """
    try:
        aetherion = await get_aetherion()
        
        if not aetherion.behavior_engine:
            return jsonify({"error": "Motor de comportamiento no disponible"}), 500
            
        # Obtener estado emocional
        emotional_state = await aetherion.behavior_engine.get_emotional_state()
        
        return jsonify(emotional_state)
    except Exception as e:
        logger.error(f"Error al obtener estado emocional: {e}")
        return jsonify({"error": str(e)}), 500

@aetherion_bp.route('/avatar', methods=['GET'])
@cross_origin()
def get_avatar_info():
    """
    Obtener información del avatar de Aetherion.
    
    Returns:
        Información del avatar en formato JSON
    """
    try:
        # Cargar configuración
        config_path = "aetherion_config.json"
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                
            appearance = config.get("appearance", {})
            
            return jsonify({
                "avatar_path": appearance.get("avatar", "website/static/img/aetherion_avatar.svg"),
                "theme": appearance.get("theme", "cosmic"),
                "animation_enabled": appearance.get("animation_enabled", True)
            })
        else:
            return jsonify({
                "avatar_path": "website/static/img/aetherion_avatar.svg",
                "theme": "cosmic",
                "animation_enabled": True
            })
    except Exception as e:
        logger.error(f"Error al obtener información del avatar: {e}")
        return jsonify({"error": str(e)}), 500

# Rutas de vista
@aetherion_bp.route('/widget', methods=['GET'])
def aetherion_widget():
    """
    Renderizar widget de Aetherion para incluir en otras páginas.
    
    Returns:
        Widget HTML
    """
    return render_template('components/aetherion_avatar.html')

def register_routes(app):
    """
    Registrar rutas en la aplicación Flask.
    
    Args:
        app: Aplicación Flask
    """
    app.register_blueprint(aetherion_bp)
    
    # Ruta directa para el avatar
    @app.route('/aetherion')
    def aetherion_page():
        """Página dedicada a Aetherion."""
        return render_template('aetherion.html')