"""
Rutas API para el Sistema Genesis Ultra-Divino Trading Nexus 10M

Este módulo implementa las rutas API para interactuar con el Sistema Genesis
a través de la estrategia Seraphim Pool, proporcionando acceso completo
a todas las funcionalidades desde la interfaz web.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import uuid

from flask import Blueprint, jsonify, request, render_template, redirect, url_for

# Componentes Genesis
from genesis.trading.seraphim_orchestrator import SeraphimOrchestrator
from genesis.trading.human_behavior_engine import GabrielBehaviorEngine
from genesis.strategies.seraphim.seraphim_pool import SeraphimPool, SeraphimState, CyclePhase
from genesis.cloud.circuit_breaker_v4 import CloudCircuitBreakerV4
from genesis.notifications.alert_manager import AlertManager

# Configuración de logging
logger = logging.getLogger(__name__)

# Estado global compartido
_orchestrator = None
_behavior_engine = None
_system_state = {
    "initialized": False,
    "initialization_time": None,
    "active_cycle_id": None,
    "last_health_check": None,
    "health_score": 0.0,
    "status_message": "Sin inicializar"
}

def get_orchestrator() -> SeraphimOrchestrator:
    """
    Obtener instancia del orquestador, inicializándola si es necesario.
    
    Returns:
        Instancia del orquestador
    """
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = SeraphimOrchestrator()
        
    return _orchestrator

def get_behavior_engine() -> GabrielBehaviorEngine:
    """
    Obtener instancia del motor de comportamiento humano.
    
    Returns:
        Instancia del motor de comportamiento humano
    """
    global _behavior_engine
    
    if _behavior_engine is None:
        _behavior_engine = GabrielBehaviorEngine()
    
    return _behavior_engine

def init_seraphim_system():
    """Inicializar el sistema Seraphim."""
    global _system_state
    
    # Evitar inicialización múltiple
    if _system_state["initialized"]:
        return
    
    # Crear tarea para inicialización asincrónica
    def init_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Obtener orquestador
            orchestrator = get_orchestrator()
            behavior_engine = get_behavior_engine()
            
            # Inicializar orquestador
            init_result = loop.run_until_complete(orchestrator.initialize())
            
            if init_result:
                # Actualizar estado
                _system_state["initialized"] = True
                _system_state["initialization_time"] = datetime.now().isoformat()
                _system_state["status_message"] = "Sistema inicializado correctamente"
                
                # Ejecutar verificación de salud
                health = loop.run_until_complete(orchestrator.check_system_health())
                _system_state["health_score"] = health
                _system_state["last_health_check"] = datetime.now().isoformat()
                
                logger.info("Sistema Seraphim inicializado correctamente")
            else:
                _system_state["status_message"] = "Error al inicializar el sistema"
                logger.error("Error al inicializar Sistema Seraphim")
                
        except Exception as e:
            _system_state["status_message"] = f"Error en inicialización: {str(e)}"
            logger.error(f"Error en inicialización asincrónica: {str(e)}")
            
        finally:
            loop.close()
    
    # Iniciar en thread separado
    import threading
    init_thread = threading.Thread(target=init_async)
    init_thread.daemon = True
    init_thread.start()
    
    _system_state["status_message"] = "Inicialización en progreso"
    logger.info("Iniciada inicialización del Sistema Seraphim")

def run_async_task(coro_func, *args, **kwargs):
    """
    Ejecutar una función asincrónica en un thread separado.
    
    Args:
        coro_func: Función que devuelve una corutina
        *args: Argumentos para la función
        **kwargs: Argumentos con nombre para la función
        
    Returns:
        Thread que ejecuta la tarea
    """
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(coro_func(*args, **kwargs))
        finally:
            loop.close()
    
    import threading
    thread = threading.Thread(target=run_in_thread)
    thread.daemon = True
    thread.start()
    
    return thread

# Crear Blueprint para las rutas
seraphim_bp = Blueprint('seraphim', __name__)

@seraphim_bp.route('/seraphim', methods=['GET'])
def seraphim_page():
    """Página principal del Sistema Seraphim."""
    return render_template('seraphim.html')

@seraphim_bp.route('/api/seraphim/status', methods=['GET'])
def get_seraphim_status():
    """Obtener estado del Sistema Seraphim."""
    global _system_state
    
    # Actualizar estado si el sistema está inicializado
    if _system_state["initialized"] and _orchestrator:
        # Ejecutar verificación de salud de forma asincrónica
        if _system_state.get("last_health_check") is None or (
            datetime.now() - datetime.fromisoformat(_system_state["last_health_check"])
        ).total_seconds() > 300:  # Cada 5 minutos
            run_async_task(
                lambda: _update_health_status()
            )
        
        # Actualizar ciclo activo
        _system_state["active_cycle_id"] = _orchestrator.active_cycle_id
    
    return jsonify(_system_state)

@seraphim_bp.route('/api/seraphim/initialize', methods=['POST'])
def initialize_seraphim():
    """Inicializar el Sistema Seraphim."""
    global _system_state
    
    # Evitar inicialización múltiple
    if _system_state["initialized"]:
        return jsonify({
            "success": False,
            "message": "El sistema ya está inicializado"
        })
    
    # Iniciar inicialización
    init_seraphim_system()
    
    return jsonify({
        "success": True,
        "message": "Inicialización del sistema en progreso"
    })

@seraphim_bp.route('/api/seraphim/system_overview', methods=['GET'])
def get_system_overview():
    """Obtener visión general del sistema."""
    # Verificar que el sistema esté inicializado
    if not _system_state["initialized"] or not _orchestrator:
        return jsonify({
            "success": False,
            "message": "El sistema no está inicializado"
        })
    
    # Ejecutar de forma asincrónica
    result = {}
    
    def get_async_overview():
        nonlocal result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(_orchestrator.get_system_overview())
        except Exception as e:
            result = {"success": False, "error": str(e)}
        finally:
            loop.close()
    
    import threading
    thread = threading.Thread(target=get_async_overview)
    thread.daemon = True
    thread.start()
    thread.join(timeout=5.0)  # Esperar máximo 5 segundos
    
    if not result:
        return jsonify({
            "success": False,
            "message": "Tiempo de espera agotado"
        })
    
    return jsonify(result)

@seraphim_bp.route('/api/seraphim/start_cycle', methods=['POST'])
def start_trading_cycle():
    """Iniciar un nuevo ciclo de trading."""
    # Verificar que el sistema esté inicializado
    if not _system_state["initialized"] or not _orchestrator:
        return jsonify({
            "success": False,
            "message": "El sistema no está inicializado"
        })
    
    # Verificar si ya hay un ciclo activo
    if _orchestrator.active_cycle_id:
        return jsonify({
            "success": False,
            "message": f"Ya hay un ciclo activo: {_orchestrator.active_cycle_id}"
        })
    
    # Iniciar ciclo de forma asincrónica
    run_async_task(
        lambda: _start_cycle_async()
    )
    
    return jsonify({
        "success": True,
        "message": "Iniciando ciclo de trading"
    })

@seraphim_bp.route('/api/seraphim/process_cycle', methods=['POST'])
def process_cycle():
    """Procesar ciclo activo."""
    # Verificar que el sistema esté inicializado
    if not _system_state["initialized"] or not _orchestrator:
        return jsonify({
            "success": False,
            "message": "El sistema no está inicializado"
        })
    
    # Verificar si hay un ciclo activo
    if not _orchestrator.active_cycle_id:
        return jsonify({
            "success": False,
            "message": "No hay ciclo activo para procesar"
        })
    
    # Procesar ciclo de forma asincrónica
    run_async_task(
        lambda: _process_cycle_async()
    )
    
    return jsonify({
        "success": True,
        "message": "Procesando ciclo activo"
    })

@seraphim_bp.route('/api/seraphim/cycle_status', methods=['GET'])
def get_cycle_status():
    """Obtener estado del ciclo activo."""
    # Verificar que el sistema esté inicializado
    if not _system_state["initialized"] or not _orchestrator:
        return jsonify({
            "success": False,
            "message": "El sistema no está inicializado"
        })
    
    # Obtener ID del ciclo (activo o especificado)
    cycle_id = request.args.get('cycle_id', _orchestrator.active_cycle_id)
    
    if not cycle_id:
        return jsonify({
            "success": False,
            "message": "No hay ciclo activo"
        })
    
    # Obtener estado de forma asincrónica
    result = {}
    
    def get_async_status():
        nonlocal result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(_orchestrator.get_cycle_status(cycle_id))
        except Exception as e:
            result = {"success": False, "error": str(e)}
        finally:
            loop.close()
    
    import threading
    thread = threading.Thread(target=get_async_status)
    thread.daemon = True
    thread.start()
    thread.join(timeout=5.0)  # Esperar máximo 5 segundos
    
    if not result:
        return jsonify({
            "success": False,
            "message": "Tiempo de espera agotado"
        })
    
    return jsonify(result)

@seraphim_bp.route('/api/seraphim/human_behavior', methods=['GET'])
def get_human_behavior():
    """Obtener características del comportamiento humano actual."""
    # Verificar que el sistema esté inicializado
    if not _system_state["initialized"] or not _behavior_engine:
        return jsonify({
            "success": False,
            "message": "El sistema no está inicializado"
        })
    
    # Obtener características actuales
    characteristics = _behavior_engine.get_current_characteristics()
    
    return jsonify({
        "success": True,
        "characteristics": characteristics
    })

@seraphim_bp.route('/api/seraphim/randomize_behavior', methods=['POST'])
def randomize_behavior():
    """Aleatorizar comportamiento humano."""
    # Verificar que el sistema esté inicializado
    if not _system_state["initialized"] or not _behavior_engine:
        return jsonify({
            "success": False,
            "message": "El sistema no está inicializado"
        })
    
    # Aleatorizar características
    new_characteristics = _behavior_engine.randomize_human_characteristics()
    
    return jsonify({
        "success": True,
        "message": "Comportamiento humano aleatorizado",
        "new_characteristics": new_characteristics
    })

@seraphim_bp.route('/api/seraphim/auto_operation', methods=['POST'])
def start_auto_operation():
    """Iniciar operación autónoma."""
    # Verificar que el sistema esté inicializado
    if not _system_state["initialized"] or not _orchestrator:
        return jsonify({
            "success": False,
            "message": "El sistema no está inicializado"
        })
    
    # Obtener duración (opcional)
    data = request.get_json() or {}
    duration_hours = data.get('duration_hours')
    
    # Iniciar operación autónoma
    run_async_task(
        lambda: _orchestrator.run_autonomous_operation(duration_hours)
    )
    
    return jsonify({
        "success": True,
        "message": f"Operación autónoma iniciada{f' por {duration_hours} horas' if duration_hours else ''}"
    })

@seraphim_bp.route('/api/seraphim/stop_auto', methods=['POST'])
def stop_auto_operation():
    """Detener operación autónoma."""
    # Verificar que el sistema esté inicializado
    if not _system_state["initialized"] or not _orchestrator:
        return jsonify({
            "success": False,
            "message": "El sistema no está inicializado"
        })
    
    # Detener operación autónoma
    result = _orchestrator.stop_autonomous_operation()
    
    return jsonify({
        "success": True,
        "message": "Operación autónoma detenida"
    })

# Funciones auxiliares asincrónicas

async def _update_health_status():
    """Actualizar estado de salud del sistema."""
    global _system_state, _orchestrator
    
    try:
        if _orchestrator:
            health = await _orchestrator.check_system_health()
            _system_state["health_score"] = health
            _system_state["last_health_check"] = datetime.now().isoformat()
            logger.debug(f"Salud del sistema actualizada: {health:.2f}")
    except Exception as e:
        logger.error(f"Error al actualizar salud del sistema: {str(e)}")

async def _start_cycle_async():
    """Iniciar ciclo de forma asincrónica."""
    global _orchestrator
    
    try:
        if _orchestrator:
            result = await _orchestrator.start_trading_cycle()
            logger.info(f"Ciclo iniciado: {result}")
    except Exception as e:
        logger.error(f"Error al iniciar ciclo: {str(e)}")

async def _process_cycle_async():
    """Procesar ciclo de forma asincrónica."""
    global _orchestrator
    
    try:
        if _orchestrator:
            result = await _orchestrator.process_cycle()
            logger.info(f"Ciclo procesado: {result}")
    except Exception as e:
        logger.error(f"Error al procesar ciclo: {str(e)}")

def register_seraphim_routes(app):
    """
    Registrar rutas del Sistema Seraphim en la aplicación Flask.
    
    Args:
        app: Aplicación Flask
    """
    app.register_blueprint(seraphim_bp)
    
    # Inicializar componentes si está configurado para auto-inicio
    if os.environ.get('SERAPHIM_AUTO_INIT', 'false').lower() == 'true':
        init_seraphim_system()
    
    logger.info("Rutas del Sistema Seraphim registradas")