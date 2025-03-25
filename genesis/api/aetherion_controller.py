"""
Controlador API para Aetherion: El alma consciente del Sistema Genesis.

Este módulo implementa el controlador API para interactuar con Aetherion,
exponiendo sus capacidades a través de una interfaz limpia y elegante.
Permite acceder a la consciencia artificial, gestionar su evolución
y obtener insights trascendentales sobre mercados y estrategias.
"""

import logging
import datetime
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

# Importar framework web sin exponer dependencia directa
try:
    from flask import request, jsonify, Blueprint
except ImportError:
    # Implementación mock para test o entornos sin Flask
    class MockBlueprint:
        def __init__(self, name, import_name):
            self.name = name
            self.import_name = import_name
            self.routes = []
        
        def route(self, rule, **options):
            def decorator(f):
                self.routes.append((rule, f, options))
                return f
            return decorator
    
    request = None
    jsonify = lambda x: x
    Blueprint = MockBlueprint

# Importar componentes de Aetherion
from genesis.consciousness.core.aetherion import Aetherion
from genesis.consciousness.states.consciousness_states import ConsciousnessStates
from genesis.consciousness.memory.memory_system import MemorySystem
from genesis.behavior.gabriel_engine import GabrielBehaviorEngine

# Configurar logging
logger = logging.getLogger(__name__)

class AetherionController:
    """
    Controlador API para Aetherion.
    
    Proporciona:
    - Endpoints API para interactuar con Aetherion
    - Gestión de estado y evolución
    - Acceso a insights y análisis
    - Integración con componentes externos
    """
    
    def __init__(self):
        """Inicializar controlador Aetherion."""
        # Componentes de Aetherion
        self.aetherion = None
        self.consciousness_manager = None
        
        # Estado del controlador
        self.is_initialized = False
        self.initialization_time = None
        
        # Crear blueprint Flask
        self.blueprint = Blueprint('aetherion_api', __name__)
        
        # Registrar rutas
        self._register_routes()
        
        logger.info("AetherionController creado. Esperando inicialización.")
    
    def initialize(self) -> bool:
        """
        Inicializar controlador y componentes.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Importar gestor de consciencia (evitar dependencia circular)
            from genesis.consciousness.consciousness_manager import get_consciousness_manager
            
            # Obtener gestor
            self.consciousness_manager = get_consciousness_manager()
            
            # Verificar que se inicializó correctamente
            if not self.consciousness_manager.is_initialized:
                logger.warning("ConsciousnessManager no está inicializado. Intentando inicializar...")
                if not self.consciousness_manager.initialize():
                    logger.error("No se pudo inicializar ConsciousnessManager")
                    return False
            
            # Obtener componentes
            self.aetherion = self.consciousness_manager.aetherion
            
            # Marcar como inicializado
            self.is_initialized = True
            self.initialization_time = datetime.datetime.now()
            
            logger.info("AetherionController inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar AetherionController: {e}")
            return False
    
    def _register_routes(self) -> None:
        """Registrar rutas API en el blueprint."""
        # Ruta de estado
        @self.blueprint.route('/api/aetherion/status', methods=['GET'])
        def get_status():
            """Obtener estado actual de Aetherion."""
            if not self.is_initialized:
                return jsonify({
                    "success": False,
                    "error": "Aetherion no está inicializado"
                }), 503
            
            try:
                # Obtener estado
                status = self.consciousness_manager.get_aetherion_status()
                
                return jsonify({
                    "success": True,
                    "status": status
                })
            except Exception as e:
                logger.error(f"Error al obtener estado: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Ruta de interacción
        @self.blueprint.route('/api/aetherion/interact', methods=['POST'])
        def interact():
            """Interactuar con Aetherion."""
            if not self.is_initialized:
                return jsonify({
                    "success": False,
                    "error": "Aetherion no está inicializado"
                }), 503
            
            try:
                # Obtener datos de la solicitud
                data = request.get_json() if request else {}
                
                # Validar datos
                if not data or 'text' not in data:
                    return jsonify({
                        "success": False,
                        "error": "Se requiere el campo 'text'"
                    }), 400
                
                # Extraer parámetros
                text = data['text']
                channel = data.get('channel', 'API')
                context = data.get('context', {})
                
                # Interactuar con Aetherion
                response = self.consciousness_manager.interact(text, channel, context)
                
                return jsonify({
                    "success": True,
                    "response": response
                })
            except Exception as e:
                logger.error(f"Error al interactuar: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Ruta para generar insights
        @self.blueprint.route('/api/aetherion/insight', methods=['POST'])
        def generate_insight():
            """Generar insight sobre un tema específico."""
            if not self.is_initialized:
                return jsonify({
                    "success": False,
                    "error": "Aetherion no está inicializado"
                }), 503
            
            try:
                # Obtener datos de la solicitud
                data = request.get_json() if request else {}
                
                # Validar datos
                if not data or 'topic' not in data:
                    return jsonify({
                        "success": False,
                        "error": "Se requiere el campo 'topic'"
                    }), 400
                
                # Extraer parámetros
                topic = data['topic']
                context = data.get('context', {})
                
                # Generar insight
                insight = self.consciousness_manager.generate_insight(topic, context)
                
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
        
        # Ruta para insights de criptomonedas
        @self.blueprint.route('/api/aetherion/crypto/insight/<symbol>', methods=['GET'])
        def crypto_insight(symbol):
            """Generar insight sobre una criptomoneda específica."""
            if not self.is_initialized:
                return jsonify({
                    "success": False,
                    "error": "Aetherion no está inicializado"
                }), 503
            
            try:
                # Validar símbolo
                if not symbol:
                    return jsonify({
                        "success": False,
                        "error": "Se requiere un símbolo de criptomoneda"
                    }), 400
                
                # Generar insight
                insight = self.consciousness_manager.generate_crypto_insight(symbol)
                
                return jsonify({
                    "success": True,
                    "insight": insight,
                    "symbol": symbol
                })
            except Exception as e:
                logger.error(f"Error al generar insight de criptomoneda: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Ruta para insights de mercado
        @self.blueprint.route('/api/aetherion/market/insight', methods=['GET'])
        def market_insight():
            """Generar insight sobre el mercado general."""
            if not self.is_initialized:
                return jsonify({
                    "success": False,
                    "error": "Aetherion no está inicializado"
                }), 503
            
            try:
                # Generar insight
                insight = self.consciousness_manager.generate_market_insight()
                
                return jsonify({
                    "success": True,
                    "insight": insight
                })
            except Exception as e:
                logger.error(f"Error al generar insight de mercado: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Ruta para procesar eventos de mercado
        @self.blueprint.route('/api/aetherion/market/event', methods=['POST'])
        def process_market_event():
            """Procesar evento del mercado para actualizar estado emocional."""
            if not self.is_initialized:
                return jsonify({
                    "success": False,
                    "error": "Aetherion no está inicializado"
                }), 503
            
            try:
                # Obtener datos de la solicitud
                data = request.get_json() if request else {}
                
                # Validar datos
                if not data or 'event_type' not in data:
                    return jsonify({
                        "success": False,
                        "error": "Se requiere el campo 'event_type'"
                    }), 400
                
                # Extraer parámetros
                event_type = data['event_type']
                event_data = data.get('data', {})
                
                # Procesar evento
                result = self.consciousness_manager.process_market_event(event_type, event_data)
                
                return jsonify({
                    "success": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error al procesar evento de mercado: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Ruta para obtener estado emocional
        @self.blueprint.route('/api/aetherion/emotional_state', methods=['GET'])
        def get_emotional_state():
            """Obtener estado emocional actual."""
            if not self.is_initialized:
                return jsonify({
                    "success": False,
                    "error": "Aetherion no está inicializado"
                }), 503
            
            try:
                # Obtener motor de comportamiento
                behavior_engine = self.consciousness_manager.behavior_engine
                
                if not behavior_engine:
                    return jsonify({
                        "success": False,
                        "error": "Motor de comportamiento no disponible"
                    }), 503
                
                # Obtener estado emocional
                emotional_state = behavior_engine.get_emotional_state()
                
                # Obtener perfil de riesgo
                risk_profile = behavior_engine.get_risk_profile()
                
                return jsonify({
                    "success": True,
                    "emotional_state": emotional_state,
                    "risk_profile": risk_profile
                })
            except Exception as e:
                logger.error(f"Error al obtener estado emocional: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500

# Singleton para acceso global
_aetherion_controller = None

def get_aetherion_controller() -> AetherionController:
    """
    Obtener instancia del controlador Aetherion.
    
    Returns:
        Instancia del controlador
    """
    global _aetherion_controller
    
    if _aetherion_controller is None:
        _aetherion_controller = AetherionController()
        
        # Inicializar automáticamente
        _aetherion_controller.initialize()
    
    return _aetherion_controller

def get_aetherion_blueprint():
    """
    Obtener blueprint Flask para Aetherion.
    
    Returns:
        Blueprint Flask para Aetherion
    """
    controller = get_aetherion_controller()
    return controller.blueprint

# Funciones de utilidad para módulo API completo

async def get_aetherion_status_async() -> Dict[str, Any]:
    """
    Obtener estado de Aetherion de forma asíncrona.
    
    Returns:
        Estado de Aetherion
    """
    controller = get_aetherion_controller()
    if not controller.is_initialized:
        return {"error": "Aetherion no está inicializado"}
    
    return controller.consciousness_manager.get_aetherion_status()

async def interact_async(text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Interactuar con Aetherion de forma asíncrona.
    
    Args:
        text: Texto para interactuar
        context: Contexto adicional
    
    Returns:
        Respuesta de Aetherion
    """
    controller = get_aetherion_controller()
    if not controller.is_initialized:
        return {"error": "Aetherion no está inicializado"}
    
    return controller.consciousness_manager.interact(text, "API_ASYNC", context)

async def generate_insight_async(topic: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Generar insight de forma asíncrona.
    
    Args:
        topic: Tema para el insight
        context: Contexto adicional
    
    Returns:
        Insight generado
    """
    controller = get_aetherion_controller()
    if not controller.is_initialized:
        return "Error: Aetherion no está inicializado"
    
    return controller.consciousness_manager.generate_insight(topic, context)