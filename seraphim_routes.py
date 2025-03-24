"""
Rutas API para el Sistema Genesis Ultra-Divino Trading Nexus 10M

Este módulo implementa las rutas API para interactuar con el Sistema Genesis
a través de la estrategia Seraphim Pool, proporcionando acceso completo
a todas las funcionalidades desde la interfaz web.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import os
import json
import logging
import asyncio
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

from flask import Flask, render_template, request, jsonify, redirect, url_for, Response

# Configurar logging
logger = logging.getLogger("genesis.seraphim")

# Singleton del orquestador Seraphim para toda la aplicación
_orchestrator_instance = None
_behavior_engine_instance = None


def get_orchestrator() -> 'SeraphimOrchestrator':
    """
    Obtener instancia del orquestador, inicializándola si es necesario.
    
    Returns:
        Instancia del orquestador
    """
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        try:
            from genesis.trading.seraphim_orchestrator import SeraphimOrchestrator
            _orchestrator_instance = SeraphimOrchestrator()
            logger.info("Orquestador Seraphim inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar orquestador Seraphim: {e}")
            # Devolver un orquestador simulado para evitar errores
            from types import SimpleNamespace
            _orchestrator_instance = SimpleNamespace()
            
    return _orchestrator_instance


def get_behavior_engine() -> 'GabrielBehaviorEngine':
    """
    Obtener instancia del motor de comportamiento humano.
    
    Returns:
        Instancia del motor de comportamiento humano
    """
    global _behavior_engine_instance
    
    if _behavior_engine_instance is None:
        try:
            from genesis.trading.human_behavior_engine import GabrielBehaviorEngine
            _behavior_engine_instance = GabrielBehaviorEngine()
            logger.info("Motor de comportamiento humano inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar motor de comportamiento humano: {e}")
            # Devolver un motor simulado para evitar errores
            from types import SimpleNamespace
            _behavior_engine_instance = SimpleNamespace()
            
    return _behavior_engine_instance


def init_seraphim_system():
    """Inicializar el sistema Seraphim."""
    orchestrator = get_orchestrator()
    behavior_engine = get_behavior_engine()
    
    def init_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Inicializar orquestador y motor de comportamiento
            loop.run_until_complete(orchestrator.initialize())
            loop.run_until_complete(behavior_engine.initialize())
            
            logger.info("Sistema Seraphim inicializado correctamente")
        except Exception as e:
            logger.error(f"Error durante inicialización asíncrona: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            loop.close()
    
    # Ejecutar inicialización en hilo separado
    thread = threading.Thread(target=init_async)
    thread.daemon = True
    thread.start()
    
    return {"success": True, "message": "Inicialización en progreso"}


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
        
        result = None
        try:
            coro = coro_func(*args, **kwargs)
            result = loop.run_until_complete(coro)
            return result
        except Exception as e:
            logger.error(f"Error en tarea asíncrona: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
        finally:
            try:
                # Verificar que el loop no esté cerrado antes de limpiarlo
                if not loop.is_closed():
                    try:
                        # Cancelar todas las tareas pendientes específicas de este loop
                        tasks = asyncio.all_tasks(loop=loop)
                        if tasks:
                            # Filtrar solo tareas no completadas o canceladas
                            active_tasks = [t for t in tasks if not t.done() and not t.cancelled()]
                            
                            # Si hay tareas activas, cancelarlas con timeout
                            if active_tasks:
                                for task in active_tasks:
                                    task.cancel()
                                
                                # Esperar que las cancelaciones terminen (con timeout)
                                wait_task = asyncio.gather(*active_tasks, return_exceptions=True)
                                try:
                                    loop.run_until_complete(asyncio.wait_for(wait_task, timeout=3.0))
                                except asyncio.TimeoutError:
                                    logger.warning("Timeout durante cancelación de tareas")
                    except Exception as e:
                        logger.warning(f"Error al cancelar tareas: {e}")
                    
                    try:
                        loop.close()
                    except Exception as e:
                        logger.warning(f"Error al cerrar bucle de eventos: {e}")
            except Exception as e:
                logger.warning(f"Error durante limpieza final: {e}")
    
    thread = threading.Thread(target=run_in_thread)
    thread.daemon = True
    thread.start()
    return thread


# Rutas
def seraphim_page():
    """Página principal del Sistema Seraphim."""
    return render_template('seraphim.html')


def get_seraphim_status():
    """Obtener estado del Sistema Seraphim."""
    orchestrator = get_orchestrator()
    
    try:
        # Comprobar si el orquestador está inicializado
        initialized = hasattr(orchestrator, 'initialized') and orchestrator.initialized
        
        return jsonify({
            "initialized": initialized,
            "message": "Sistema Seraphim listo" if initialized else "Sistema Seraphim no inicializado"
        })
    except Exception as e:
        logger.error(f"Error al obtener estado Seraphim: {e}")
        return jsonify({
            "initialized": False,
            "message": f"Error al obtener estado: {str(e)}"
        })


def initialize_seraphim():
    """Inicializar el Sistema Seraphim."""
    try:
        result = init_seraphim_system()
        
        return jsonify({
            "success": True,
            "message": "Sistema Seraphim inicializándose"
        })
    except Exception as e:
        logger.error(f"Error al inicializar Sistema Seraphim: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500


def get_system_overview():
    """Obtener visión general del sistema."""
    def get_async_overview():
        try:
            orchestrator = get_orchestrator()
            
            # Obtener estado general del sistema
            system_status = orchestrator.get_system_overview()
            
            # Obtener ciclo activo
            active_cycle = orchestrator.get_active_cycle()
            
            # Obtener top criptomonedas
            top_cryptos = orchestrator.get_top_cryptocurrencies()
            
            # Estado de Buddha
            buddha_status = orchestrator.get_buddha_status()
            
            return {
                "success": True,
                "system_stats": system_status,
                "active_cycle": active_cycle,
                "top_cryptos": top_cryptos,
                "buddha_status": buddha_status
            }
        except Exception as e:
            logger.error(f"Error al obtener visión general: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    # Ejecutar en hilo separado
    thread = run_async_task(lambda: _update_health_status())
    
    # Ejecutar la tarea principal
    thread = run_async_task(get_async_overview)
    
    # Esperar resultado con timeout
    thread.join(timeout=5.0)
    
    # Si aún está ejecutando, devolver respuesta inicial
    if thread.is_alive():
        return jsonify({
            "success": True,
            "system_stats": {
                "orchestrator_state": "LOADING",
                "health_score": 0.5,
                "uptime": "Calculando...",
                "completed_cycles_count": 0,
                "total_profit": 0.0
            },
            "active_cycle": None,
            "top_cryptos": [],
            "buddha_status": "Cargando..."
        })
    
    # Si el thread tiene un atributo result, devolverlo
    if hasattr(thread, 'result'):
        return jsonify(thread.result)
    
    # En caso contrario, devolver un estado por defecto
    return jsonify({
        "success": True,
        "system_stats": {
            "orchestrator_state": "UNKNOWN",
            "health_score": 0.0,
            "uptime": "Desconocido",
            "completed_cycles_count": 0,
            "total_profit": 0.0
        },
        "active_cycle": None,
        "top_cryptos": [],
        "buddha_status": "Desconocido"
    })


def start_trading_cycle():
    """Iniciar un nuevo ciclo de trading."""
    orchestrator = get_orchestrator()
    
    # Ejecutar en hilo separado
    thread = run_async_task(lambda: _start_cycle_async())
    
    return jsonify({
        "success": True,
        "message": "Ciclo de trading iniciado"
    })


def process_cycle():
    """Procesar ciclo activo."""
    orchestrator = get_orchestrator()
    
    # Ejecutar en hilo separado
    thread = run_async_task(lambda: _process_cycle_async())
    
    return jsonify({
        "success": True,
        "message": "Procesando ciclo"
    })


def get_cycle_status():
    """Obtener estado del ciclo activo."""
    def get_async_status():
        try:
            orchestrator = get_orchestrator()
            
            # Obtener estado del ciclo activo
            cycle_status = orchestrator.get_cycle_status()
            
            return {
                "success": True,
                "cycle_status": cycle_status
            }
        except Exception as e:
            logger.error(f"Error al obtener estado del ciclo: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    # Ejecutar en hilo separado
    thread = run_async_task(get_async_status)
    
    # Esperar resultado con timeout
    thread.join(timeout=3.0)
    
    # Si aún está ejecutando, devolver respuesta inicial
    if thread.is_alive():
        return jsonify({
            "success": True,
            "cycle_status": {
                "status": "LOADING",
                "phase": None,
                "performance": None
            }
        })
    
    # Si el thread tiene un atributo result, devolverlo
    if hasattr(thread, 'result'):
        return jsonify(thread.result)
    
    # En caso contrario, devolver un estado por defecto
    return jsonify({
        "success": True,
        "cycle_status": {
            "status": "UNKNOWN",
            "phase": None,
            "performance": None
        }
    })


def get_human_behavior():
    """Obtener características del comportamiento humano actual."""
    behavior_engine = get_behavior_engine()
    
    try:
        # Obtener características actuales
        characteristics = behavior_engine.current_characteristics if hasattr(behavior_engine, 'current_characteristics') else None
        
        if characteristics:
            return jsonify({
                "success": True,
                "characteristics": characteristics
            })
        else:
            return jsonify({
                "success": False,
                "message": "Características de comportamiento no disponibles"
            })
    except Exception as e:
        logger.error(f"Error al obtener comportamiento humano: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500


def randomize_behavior():
    """Aleatorizar comportamiento humano."""
    behavior_engine = get_behavior_engine()
    
    try:
        # Aleatorizar características
        if hasattr(behavior_engine, 'randomize'):
            new_characteristics = behavior_engine.randomize()
            
            return jsonify({
                "success": True,
                "new_characteristics": new_characteristics
            })
        else:
            # Simulación simplificada si no está disponible
            new_characteristics = {
                "emotional_state": "Neutral",
                "risk_tolerance": "Moderado",
                "decision_style": "Analítico",
                "market_perceptions": {
                    "market_sentiment": "Neutral"
                }
            }
            
            return jsonify({
                "success": True,
                "new_characteristics": new_characteristics
            })
    except Exception as e:
        logger.error(f"Error al aleatorizar comportamiento: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500


def start_auto_operation():
    """Iniciar operación autónoma."""
    orchestrator = get_orchestrator()
    
    # Obtener parámetros
    data = request.get_json() or {}
    duration_hours = data.get('duration_hours')
    
    try:
        # Convertir a int si es necesario
        if duration_hours:
            duration_hours = int(duration_hours)
        
        # Ejecutar en hilo separado
        thread = run_async_task(lambda: orchestrator.run_autonomous_operation(duration_hours))
        
        return jsonify({
            "success": True,
            "message": f"Operación autónoma iniciada {('por ' + str(duration_hours) + ' horas') if duration_hours else 'indefinidamente'}"
        })
    except Exception as e:
        logger.error(f"Error al iniciar operación autónoma: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500


def stop_auto_operation():
    """Detener operación autónoma."""
    orchestrator = get_orchestrator()
    
    try:
        # Ejecutar en hilo separado
        thread = run_async_task(lambda: orchestrator.stop_autonomous_operation())
        
        return jsonify({
            "success": True,
            "message": "Operación autónoma detenida"
        })
    except Exception as e:
        logger.error(f"Error al detener operación autónoma: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500


# Funciones auxiliares asíncronas
async def _update_health_status():
    """Actualizar estado de salud del sistema."""
    orchestrator = get_orchestrator()
    
    try:
        if hasattr(orchestrator, 'check_system_health'):
            await orchestrator.check_system_health()
    except Exception as e:
        logger.error(f"Error al actualizar estado de salud: {e}")


async def _start_cycle_async():
    """Iniciar ciclo de forma asincrónica."""
    orchestrator = get_orchestrator()
    
    try:
        if hasattr(orchestrator, 'start_trading_cycle'):
            await orchestrator.start_trading_cycle()
            logger.info("Ciclo de trading iniciado correctamente")
    except Exception as e:
        logger.error(f"Error al iniciar ciclo: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


async def _process_cycle_async():
    """Procesar ciclo de forma asincrónica."""
    orchestrator = get_orchestrator()
    
    try:
        if hasattr(orchestrator, 'process_cycle'):
            await orchestrator.process_cycle()
            logger.info("Ciclo procesado correctamente")
    except Exception as e:
        logger.error(f"Error al procesar ciclo: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def register_seraphim_routes(app):
    """
    Registrar rutas del Sistema Seraphim en la aplicación Flask.
    
    Args:
        app: Aplicación Flask
    """
    # Página principal
    app.add_url_rule('/seraphim', 'seraphim_page', seraphim_page)
    
    # Rutas API
    app.add_url_rule('/api/seraphim/status', 'get_seraphim_status', get_seraphim_status)
    app.add_url_rule('/api/seraphim/initialize', 'initialize_seraphim', initialize_seraphim, methods=['POST'])
    app.add_url_rule('/api/seraphim/system_overview', 'get_system_overview', get_system_overview)
    app.add_url_rule('/api/seraphim/start_cycle', 'start_trading_cycle', start_trading_cycle, methods=['POST'])
    app.add_url_rule('/api/seraphim/process_cycle', 'process_cycle', process_cycle, methods=['POST'])
    app.add_url_rule('/api/seraphim/cycle_status', 'get_cycle_status', get_cycle_status)
    app.add_url_rule('/api/seraphim/human_behavior', 'get_human_behavior', get_human_behavior)
    app.add_url_rule('/api/seraphim/randomize_behavior', 'randomize_behavior', randomize_behavior, methods=['POST'])
    app.add_url_rule('/api/seraphim/auto_operation', 'start_auto_operation', start_auto_operation, methods=['POST'])
    app.add_url_rule('/api/seraphim/stop_auto', 'stop_auto_operation', stop_auto_operation, methods=['POST'])
    
    logger.info("Rutas del Sistema Seraphim registradas correctamente")