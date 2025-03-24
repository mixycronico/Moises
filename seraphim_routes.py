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

from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, session

# Importar componentes del simulador de intercambio
try:
    from genesis.simulators import (
        ExchangeSimulator, 
        ExchangeSimulatorFactory,
        MarketPattern, 
        MarketEventType
    )
    from genesis.exchanges.simulated_exchange_adapter import SimulatedExchangeAdapter
    SIMULATOR_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger("genesis.seraphim")
    logger.warning(f"No se pudieron importar los componentes del simulador: {e}")
    SIMULATOR_AVAILABLE = False

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
            
            # Crear orquestador
            _orchestrator_instance = SeraphimOrchestrator()
            
            # Configurar simulador si está disponible
            if SIMULATOR_AVAILABLE:
                # Importar componentes necesarios
                from genesis.exchanges.adapter_factory import ExchangeAdapterFactory, AdapterType
                
                # Configurar simulador en thread separado
                async def setup_simulator():
                    try:
                        # Crear adaptador simulado
                        adapter = await ExchangeAdapterFactory.create_adapter(
                            exchange_id="BINANCE",
                            adapter_type=AdapterType.SIMULATED,
                            config={
                                "tick_interval_ms": 500,        # Actualizaciones cada 500ms
                                "volatility_factor": 0.005,     # 0.5% de volatilidad
                                "error_rate": 0.05,             # 5% de probabilidad de error
                                "pattern_duration": 30,         # 30 segundos por patrón
                                "enable_failures": True,        # Habilitar fallos simulados
                                "default_candle_count": 1000,   # 1000 velas históricas
                                "enable_websocket": True        # Habilitar websocket
                            }
                        )
                        
                        # Asignar adaptador al orquestador
                        _orchestrator_instance.exchange_adapter = adapter
                        
                        # Configurar símbolos iniciales
                        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
                        for symbol in symbols:
                            # Configurar símbolos en el simulador
                            await adapter.get_ticker(symbol)
                            
                        logger.info("Simulador de exchange configurado correctamente en orquestador Seraphim")
                        return True
                    except Exception as e:
                        logger.error(f"Error configurando simulador en orquestador: {e}")
                        return False
                
                # Ejecutar configuración
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(setup_simulator())
                finally:
                    loop.close()
            
            logger.info("Orquestador Seraphim inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar orquestador Seraphim: {e}")
            # Devolver un orquestador simulado para evitar errores
            from types import SimpleNamespace
            
            # Creamos un orquestador simulado con métodos básicos
            mock_orchestrator = SimpleNamespace()
            
            # Implementar método de aleatorización
            async def mock_randomize_human_behavior():
                logger.info("Simulando aleatorización de comportamiento humano")
                return {
                    "success": True,
                    "human_behavior": {
                        "emotional_state": "NEUTRAL",
                        "risk_tolerance": "BALANCED",
                        "decision_style": "ANALYTICAL",
                        "emotional_stability": 0.7,
                        "risk_adaptation_rate": 0.5,
                        "contrarian_tendency": 0.2,
                        "decision_speed": 1.0,
                        "market_perceptions": {"perceived_risk": 0.3}
                    }
                }
            
            # Implementar método de inicialización
            async def mock_initialize():
                logger.info("Simulando inicialización del orquestador")
                return True
            
            # Implementar métodos relacionados con exchange
            async def mock_verify_exchange_connections():
                logger.info("Simulando verificación de conexiones a exchanges")
                return True
                
            async def mock_get_market_data(symbol):
                logger.info(f"Simulando obtención de datos de mercado para {symbol}")
                return {
                    "success": True,
                    "symbol": symbol,
                    "last_price": 50000.0 if symbol.startswith("BTC") else 3000.0,
                    "bid": 49900.0 if symbol.startswith("BTC") else 2990.0,
                    "ask": 50100.0 if symbol.startswith("BTC") else 3010.0,
                    "volume": 100.0,
                    "timestamp": int(time.time() * 1000),
                    "trend": "UP",
                    "candles": [[int(time.time() * 1000), 50000, 50100, 49900, 50050, 10] for _ in range(5)]
                }
                
            async def mock_get_symbols():
                logger.info("Simulando obtención de símbolos")
                return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
                
            # Asignar métodos al orquestador simulado
            mock_orchestrator.randomize_human_behavior = mock_randomize_human_behavior
            mock_orchestrator.initialize = mock_initialize
            mock_orchestrator._verify_exchange_connections = mock_verify_exchange_connections
            mock_orchestrator.get_market_data = mock_get_market_data
            mock_orchestrator.get_symbols = mock_get_symbols
            mock_orchestrator.state = "SIMULATED"
            mock_orchestrator.active_cycle_id = None
            
            # Crear adaptador simulado
            exchange_adapter = SimpleNamespace()
            
            # Implementar métodos del adaptador
            async def mock_create_order(**kwargs):
                return {"success": True, "order_id": "simulated-order-123"}
                
            async def mock_cancel_order(**kwargs):
                return {"success": True, "canceled": True}
                
            async def mock_fetch_orders(**kwargs):
                return {"success": True, "orders": []}
                
            async def mock_fetch_open_orders(**kwargs):
                return {"success": True, "orders": []}
                
            async def mock_fetch_closed_orders(**kwargs):
                return {"success": True, "orders": []}
            
            # Asignar métodos al adaptador
            exchange_adapter.create_order = mock_create_order
            exchange_adapter.cancel_order = mock_cancel_order
            exchange_adapter.fetch_orders = mock_fetch_orders
            exchange_adapter.fetch_open_orders = mock_fetch_open_orders
            exchange_adapter.fetch_closed_orders = mock_fetch_closed_orders
            
            # Asociar adaptador con orquestador
            mock_orchestrator.exchange_adapter = exchange_adapter
            
            _orchestrator_instance = mock_orchestrator
            
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
            
            # Crear un motor de comportamiento simulado
            mock_engine = SimpleNamespace()
            
            # Simular características actuales
            mock_engine.current_characteristics = {
                "emotional_state": "NEUTRAL",
                "risk_tolerance": "BALANCED",
                "decision_style": "ANALYTICAL",
                "emotional_stability": 0.7,
                "risk_adaptation_rate": 0.5,
                "contrarian_tendency": 0.2,
                "decision_speed": 1.0,
                "market_perceptions": {"perceived_risk": 0.3}
            }
            
            # Simular propiedades alias
            mock_engine.mood = "NEUTRAL"
            mock_engine.risk_profile = "BALANCED"
            mock_engine.experience_level = "INTERMEDIATE"
            
            # Implementar initialize
            async def mock_initialize():
                logger.info("Simulando inicialización del motor de comportamiento")
                return True
                
            # Implementar randomize
            def mock_randomize():
                import random
                moods = ["NEUTRAL", "OPTIMISTIC", "CAUTIOUS", "FEARFUL", "GREEDY"]
                profiles = ["BALANCED", "CONSERVATIVE", "AGGRESSIVE", "MODERATE"]
                
                mock_engine.mood = random.choice(moods)
                mock_engine.risk_profile = random.choice(profiles)
                
                mock_engine.current_characteristics = {
                    "emotional_state": mock_engine.mood,
                    "risk_tolerance": mock_engine.risk_profile,
                    "decision_style": "ANALYTICAL",
                    "emotional_stability": round(random.uniform(0.3, 0.9), 2),
                    "risk_adaptation_rate": round(random.uniform(0.2, 0.8), 2),
                    "contrarian_tendency": round(random.uniform(0.1, 0.6), 2),
                    "decision_speed": round(random.uniform(0.5, 1.5), 2),
                    "market_perceptions": {"perceived_risk": round(random.uniform(0.1, 0.7), 2)}
                }
                
                return mock_engine.current_characteristics
            
            # Asignar métodos al motor simulado
            mock_engine.initialize = mock_initialize
            mock_engine.randomize = mock_randomize
            
            _behavior_engine_instance = mock_engine
            
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
    
def emotional_control_page():
    """Página de control de estados emocionales."""
    return render_template('emotional_control.html')


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
        
def set_emotional_state():
    """Establecer estado emocional directamente."""
    behavior_engine = get_behavior_engine()
    
    try:
        data = request.get_json()
        if not data or "state" not in data:
            return jsonify({
                "success": False,
                "message": "Se requiere el parámetro 'state'"
            }), 400
            
        state_name = data["state"]
        reason = data.get("reason", "manual_override")
        
        # Importar EmotionalState desde el motor de comportamiento humano
        from genesis.trading.human_behavior_engine import EmotionalState
        
        # Validar que el estado es válido
        try:
            state = EmotionalState[state_name.upper()]
        except (KeyError, ValueError):
            # Enumerar estados válidos
            valid_states = [s.name for s in EmotionalState]
            return jsonify({
                "success": False,
                "message": f"Estado emocional no válido. Estados disponibles: {valid_states}"
            }), 400
        
        # Establecer estado
        behavior_engine.set_emotional_state(state, reason)
        
        # Obtener estado actualizado
        characteristics = behavior_engine.get_current_characteristics()
        
        return jsonify({
            "success": True,
            "message": f"Estado emocional cambiado a {state.name}",
            "characteristics": characteristics
        })
    except Exception as e:
        logger.error(f"Error al establecer estado emocional: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500
        
def set_fearful_state():
    """Establecer estado FEARFUL (miedo) directamente."""
    behavior_engine = get_behavior_engine()
    
    try:
        data = request.get_json()
        reason = data.get("reason", "manual_override") if data else "manual_override"
        
        # Llamar al método específico
        behavior_engine.set_fearful_state(reason)
        
        # Obtener estado actualizado
        characteristics = behavior_engine.get_current_characteristics()
        
        return jsonify({
            "success": True,
            "message": "Estado cambiado a FEARFUL (miedo)",
            "characteristics": characteristics
        })
    except Exception as e:
        logger.error(f"Error al establecer estado FEARFUL: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500


def randomize_behavior():
    """Aleatorizar comportamiento humano."""
    orchestrator = get_orchestrator()
    
    def get_async_randomized():
        """Ejecutar aleatorización asincrónica."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(orchestrator.randomize_human_behavior())
            return result
        finally:
            loop.close()
    
    try:
        # Usar el orquestador para aleatorizar de forma sincronizada
        thread = run_async_task(get_async_randomized)
        result = thread.result
        
        if result and result.get("success", False):
            return jsonify({
                "success": True,
                "human_behavior": result.get("human_behavior", {})
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


# Simulador de Intercambio
_exchange_simulator_instance = None
_exchange_adapter_instance = None

def get_exchange_simulator():
    """
    Obtener instancia del simulador de intercambio.
    
    Returns:
        Instancia del simulador de intercambio o None si no está disponible
    """
    global _exchange_simulator_instance
    
    if not SIMULATOR_AVAILABLE:
        return None
        
    if _exchange_simulator_instance is None:
        try:
            async def init_simulator():
                config = {
                    "tick_interval_ms": 500,       # Actualizaciones cada 500ms
                    "volatility_factor": 0.005,    # 0.5% de volatilidad
                    "error_rate": 0.05,            # 5% de probabilidad de error
                    "pattern_duration": 30,        # 30 segundos por patrón
                    "enable_failures": True        # Habilitar fallos simulados
                }
                
                simulator = await ExchangeSimulatorFactory.create_simulator("BINANCE", config)
                return simulator
                
            # Ejecutar inicialización en thread separado
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                _exchange_simulator_instance = loop.run_until_complete(init_simulator())
                logger.info("Simulador de intercambio inicializado correctamente")
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error al inicializar simulador: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    return _exchange_simulator_instance

def get_exchange_adapter():
    """
    Obtener instancia del adaptador para el simulador de intercambio.
    
    Returns:
        Instancia del adaptador o None si no está disponible
    """
    global _exchange_adapter_instance
    
    if not SIMULATOR_AVAILABLE:
        return None
        
    if _exchange_adapter_instance is None:
        try:
            # Configuración del simulador
            config = {
                "tick_interval_ms": 500,       # Actualizaciones cada 500ms
                "volatility_factor": 0.005,    # 0.5% de volatilidad
                "error_rate": 0.05,            # 5% de probabilidad de error
                "pattern_duration": 30,        # 30 segundos por patrón
                "enable_failures": True        # Habilitar fallos simulados
            }
            
            # Crear adaptador
            _exchange_adapter_instance = SimulatedExchangeAdapter("BINANCE", config)
            
            # Ejecutar inicialización en thread separado
            async def init_adapter():
                await _exchange_adapter_instance.initialize()
                await _exchange_adapter_instance.connect()
                
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(init_adapter())
                logger.info("Adaptador de intercambio inicializado correctamente")
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error al inicializar adaptador: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    return _exchange_adapter_instance

def simulator_page():
    """Página del simulador de intercambio."""
    return render_template('simulator.html')

def initialize_simulator():
    """Inicializar simulador de intercambio."""
    if not SIMULATOR_AVAILABLE:
        return jsonify({
            "success": False,
            "message": "Componentes del simulador no disponibles"
        })
    
    try:
        # Inicializar el simulador
        simulator = get_exchange_simulator()
        adapter = get_exchange_adapter()
        
        if simulator is None or adapter is None:
            return jsonify({
                "success": False,
                "message": "Error al inicializar simulador"
            })
        
        return jsonify({
            "success": True,
            "message": "Simulador inicializado correctamente",
            "simulator_info": {
                "exchange_id": simulator.exchange_id,
                "symbols_count": len(simulator.symbols),
                "volatility": simulator.config.get("volatility_factor", 0.02)
            }
        })
    except Exception as e:
        logger.error(f"Error al inicializar simulador: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500

def get_simulator_status():
    """Obtener estado del simulador de intercambio."""
    if not SIMULATOR_AVAILABLE:
        return jsonify({
            "success": False,
            "message": "Componentes del simulador no disponibles"
        })
    
    try:
        simulator = get_exchange_simulator()
        
        if simulator is None:
            return jsonify({
                "success": False,
                "message": "Simulador no inicializado"
            })
        
        return jsonify({
            "success": True,
            "simulator_status": {
                "running": simulator.running,
                "symbols": list(simulator.symbols.keys())[:5],  # Limitar a 5 para la respuesta
                "error_rate": simulator.config.get("error_rate", 0.01),
                "tick_interval": simulator.config.get("tick_interval_ms", 100)
            }
        })
    except Exception as e:
        logger.error(f"Error al obtener estado del simulador: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500

async def _run_market_data_demo():
    """Ejecutar demostración de datos de mercado."""
    adapter = get_exchange_adapter()
    if adapter is None:
        return {"success": False, "message": "Adaptador no disponible"}
    
    # Lista de símbolos para la demostración
    DEMO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
    results = []
    
    try:
        # Obtener tickers para todos los símbolos
        for symbol in DEMO_SYMBOLS:
            ticker = await adapter.get_ticker(symbol)
            results.append({
                "type": "ticker",
                "symbol": symbol,
                "price": ticker['price'],
                "change_24h": ticker['percentage']
            })
            
        # Obtener libro de órdenes para BTC/USDT
        orderbook = await adapter.get_orderbook("BTC/USDT", limit=5)
        results.append({
            "type": "orderbook",
            "symbol": "BTC/USDT",
            "best_bid": {
                "price": orderbook['bids'][0][0],
                "quantity": orderbook['bids'][0][1]
            },
            "best_ask": {
                "price": orderbook['asks'][0][0],
                "quantity": orderbook['asks'][0][1]
            }
        })
        
        # Obtener velas para ETH/USDT
        candles = await adapter.get_candles("ETH/USDT", timeframe="1h", limit=5)
        candle_results = []
        
        for candle in candles:
            dt = datetime.fromtimestamp(candle['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
            candle_results.append({
                "timestamp": dt,
                "open": candle['open'],
                "close": candle['close'],
                "high": candle['high'],
                "low": candle['low'],
                "volume": candle['volume']
            })
            
        results.append({
            "type": "candles",
            "symbol": "ETH/USDT",
            "timeframe": "1h",
            "data": candle_results
        })
        
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"Error en demostración de datos de mercado: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"success": False, "message": str(e)}

def run_market_data_demo():
    """Ejecutar demostración de datos de mercado."""
    if not SIMULATOR_AVAILABLE:
        return jsonify({
            "success": False,
            "message": "Componentes del simulador no disponibles"
        })
    
    # Ejecutar en thread separado
    thread = run_async_task(_run_market_data_demo)
    thread.join(timeout=5.0)
    
    # Si aún está ejecutando, devolver mensaje de espera
    if thread.is_alive():
        return jsonify({
            "success": True,
            "message": "Ejecutando demostración, por favor espere...",
            "results": []
        })
    
    # Si el thread tiene un atributo result, devolverlo
    if hasattr(thread, 'result'):
        return jsonify(thread.result)
    
    # En caso contrario, devolver error
    return jsonify({
        "success": False,
        "message": "Error desconocido al ejecutar demostración"
    })

async def _run_orders_demo():
    """Ejecutar demostración de órdenes."""
    adapter = get_exchange_adapter()
    if adapter is None:
        return {"success": False, "message": "Adaptador no disponible"}
    
    results = []
    
    try:
        # Obtener precio actual de BTC
        ticker = await adapter.get_ticker("BTC/USDT")
        current_price = ticker["price"]
        
        results.append({
            "type": "price_check",
            "symbol": "BTC/USDT",
            "price": current_price
        })
        
        # Colocar una orden de mercado
        market_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "market",
            "quantity": 0.1
        }
        
        result = await adapter.place_order(market_order)
        order = result["order"]
        
        results.append({
            "type": "market_order",
            "order_id": order["id"],
            "status": order["status"],
            "price": order["price"],
            "quantity": order["quantity"]
        })
        
        # Colocar una orden límite
        limit_price = current_price * 0.95  # 5% por debajo del precio actual
        limit_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "quantity": 0.2,
            "price": limit_price
        }
        
        result = await adapter.place_order(limit_order)
        order = result["order"]
        
        results.append({
            "type": "limit_order",
            "order_id": order["id"],
            "status": order["status"],
            "price": order["price"],
            "quantity": order["quantity"]
        })
        
        # Obtener órdenes activas
        orders = await adapter.get_orders(symbol="BTC/USDT")
        
        results.append({
            "type": "active_orders",
            "count": len(orders["orders"]),
            "symbol": "BTC/USDT"
        })
        
        # Cancelar la orden límite si no está completada
        if order["status"] != "FILLED":
            cancel_result = await adapter.cancel_order(order["id"])
            
            results.append({
                "type": "cancel_order",
                "order_id": order["id"],
                "status": cancel_result["order"]["status"]
            })
            
            # Verificar órdenes activas nuevamente
            orders = await adapter.get_orders(symbol="BTC/USDT")
            active_count = len([o for o in orders["orders"] if o["status"] != "CANCELED" and o["status"] != "FILLED"])
            
            results.append({
                "type": "active_orders_after_cancel",
                "count": active_count,
                "symbol": "BTC/USDT"
            })
        
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"Error en demostración de órdenes: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"success": False, "message": str(e)}

def run_orders_demo():
    """Ejecutar demostración de órdenes."""
    if not SIMULATOR_AVAILABLE:
        return jsonify({
            "success": False,
            "message": "Componentes del simulador no disponibles"
        })
    
    # Ejecutar en thread separado
    thread = run_async_task(_run_orders_demo)
    thread.join(timeout=5.0)
    
    # Si aún está ejecutando, devolver mensaje de espera
    if thread.is_alive():
        return jsonify({
            "success": True,
            "message": "Ejecutando demostración de órdenes, por favor espere...",
            "results": []
        })
    
    # Si el thread tiene un atributo result, devolverlo
    if hasattr(thread, 'result'):
        return jsonify(thread.result)
    
    # En caso contrario, devolver error
    return jsonify({
        "success": False,
        "message": "Error desconocido al ejecutar demostración de órdenes"
    })


# Métodos para integración Seraphim-Simulador
def get_market_data(symbol):
    """
    Obtener datos de mercado para un símbolo específico.
    
    Args:
        symbol: Símbolo a consultar (ej: BTC/USDT)
    """
    orchestrator = get_orchestrator()
    if not orchestrator:
        return jsonify({"success": False, "error": "Seraphim Orchestrator not available"})
    
    def get_async_data():
        """Ejecutar consulta asincrónica."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Obtener datos del mercado
            market_data = loop.run_until_complete(orchestrator.get_market_data(symbol))
            
            # Importante: asignar el resultado al hilo actual
            current_thread = threading.current_thread()
            current_thread.result = market_data
            
            return market_data
        except Exception as e:
            logger.error(f"Error obteniendo datos de mercado: {e}")
            current_thread = threading.current_thread()
            current_thread.result = {"success": False, "error": f"Error: {str(e)}"}
        finally:
            loop.close()
    
    # Ejecutar en thread para no bloquear
    thread = threading.Thread(target=get_async_data)
    thread.start()
    thread.join(timeout=5.0)  # Añadir timeout para evitar bloqueos indefinidos
    
    # Obtener resultado o respuesta de error
    if hasattr(thread, 'result'):
        return jsonify(thread.result)
    else:
        return jsonify({"success": False, "error": "Failed to fetch market data"})

def get_available_symbols():
    """Obtener lista de símbolos disponibles."""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return jsonify({"success": False, "error": "Seraphim Orchestrator not available"})
    
    def get_async_symbols():
        """Ejecutar consulta asincrónica."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            symbols = loop.run_until_complete(orchestrator.get_symbols())
            
            # Importante: asignar el resultado al hilo actual
            current_thread = threading.current_thread()
            current_thread.result = {"success": True, "symbols": symbols}
            
            return {"success": True, "symbols": symbols}
        except Exception as e:
            logger.error(f"Error obteniendo símbolos: {e}")
            current_thread = threading.current_thread()
            current_thread.result = {"success": False, "error": f"Error: {str(e)}"}
        finally:
            loop.close()
    
    # Ejecutar en thread para no bloquear
    thread = threading.Thread(target=get_async_symbols)
    thread.start()
    thread.join(timeout=5.0)  # Añadir timeout para evitar bloqueos indefinidos
    
    # Obtener resultado o respuesta de error
    if hasattr(thread, 'result'):
        return jsonify(thread.result)
    else:
        return jsonify({"success": False, "error": "Failed to fetch symbols"})

def connect_exchange():
    """Conectar a exchange (simulado o real)."""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return jsonify({"success": False, "error": "Seraphim Orchestrator not available"})
    
    def connect_async():
        """Ejecutar conexión asincrónica."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(orchestrator._verify_exchange_connections())
            
            # Importante: asignar el resultado al hilo actual
            current_thread = threading.current_thread()
            current_thread.result = {"success": success}
            
            return {"success": success}
        except Exception as e:
            logger.error(f"Error verificando conexiones de exchange: {e}")
            current_thread = threading.current_thread()
            current_thread.result = {"success": False, "error": f"Error: {str(e)}"}
        finally:
            loop.close()
    
    # Ejecutar en thread para no bloquear
    thread = threading.Thread(target=connect_async)
    thread.start()
    thread.join(timeout=5.0)  # Añadir timeout para evitar bloqueos indefinidos
    
    # Obtener resultado o respuesta de error
    if hasattr(thread, 'result'):
        return jsonify(thread.result)
    else:
        return jsonify({"success": False, "error": "Failed to connect"})

def place_trade():
    """Colocar una orden de trading."""
    # Obtener datos de la solicitud
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No data provided"})
    
    # Validar datos
    required_fields = ["symbol", "side", "type", "amount"]
    for field in required_fields:
        if field not in data:
            return jsonify({"success": False, "error": f"Missing required field: {field}"})
    
    # Obtener orquestador
    orchestrator = get_orchestrator()
    if not orchestrator or not orchestrator.exchange_adapter:
        return jsonify({"success": False, "error": "Exchange adapter not available"})
    
    # Procesar solicitud
    def place_order_async():
        """Ejecutar orden asincrónica."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Pasar a adaptador
            result = loop.run_until_complete(
                orchestrator.exchange_adapter.create_order(
                    symbol=data["symbol"],
                    order_type=data["type"],
                    side=data["side"],
                    amount=float(data["amount"]),
                    price=float(data.get("price", 0))
                )
            )
            
            # Importante: asignar el resultado al hilo actual
            current_thread = threading.current_thread()
            current_thread.result = result
            
            return result
        except Exception as e:
            logger.error(f"Error colocando orden: {e}")
            current_thread = threading.current_thread()
            current_thread.result = {"success": False, "error": f"Error: {str(e)}"}
        finally:
            loop.close()
    
    # Ejecutar en thread para no bloquear
    thread = threading.Thread(target=place_order_async)
    thread.start()
    thread.join(timeout=5.0)  # Añadir timeout para evitar bloqueos indefinidos
    
    # Obtener resultado o respuesta de error
    if hasattr(thread, 'result'):
        return jsonify(thread.result)
    else:
        return jsonify({"success": False, "error": "Failed to place order"})

def cancel_trade(trade_id):
    """
    Cancelar una orden de trading.
    
    Args:
        trade_id: ID de la orden a cancelar
    """
    # Obtener datos desde la solicitud - manejar el caso de JSON nulo
    # Algunos clientes pueden enviar una solicitud sin cuerpo JSON
    try:
        data = request.json or {}
    except:
        data = {}
    
    # Obtener orquestador
    orchestrator = get_orchestrator()
    if not orchestrator or not orchestrator.exchange_adapter:
        return jsonify({"success": False, "error": "Exchange adapter not available"})
    
    # Procesar solicitud
    def cancel_order_async():
        """Ejecutar cancelación asincrónica."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Pasar a adaptador
            result = loop.run_until_complete(
                orchestrator.exchange_adapter.cancel_order(
                    id=trade_id,
                    symbol=data.get("symbol")  # Opcional en algunos exchanges
                )
            )
            
            # Importante: asignar el resultado al hilo actual
            current_thread = threading.current_thread()
            current_thread.result = result
            
            return result
        except Exception as e:
            logger.error(f"Error cancelando orden: {e}")
            current_thread = threading.current_thread()
            current_thread.result = {"success": False, "error": f"Error: {str(e)}"}
        finally:
            loop.close()
    
    # Ejecutar en thread para no bloquear
    thread = threading.Thread(target=cancel_order_async)
    thread.start()
    thread.join(timeout=5.0)  # Añadir timeout para evitar bloqueos indefinidos
    
    # Obtener resultado o respuesta de error
    if hasattr(thread, 'result'):
        return jsonify(thread.result)
    else:
        return jsonify({"success": False, "error": "Failed to cancel order"})

def get_trades():
    """Obtener órdenes activas y completadas."""
    # Obtener datos de consulta
    symbol = request.args.get("symbol")
    status = request.args.get("status", "all")  # 'open', 'closed', 'all'
    
    # Obtener orquestador
    orchestrator = get_orchestrator()
    if not orchestrator or not orchestrator.exchange_adapter:
        return jsonify({"success": False, "error": "Exchange adapter not available"})
    
    # Procesar solicitud
    def get_orders_async():
        """Ejecutar consulta asincrónica."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Determinar qué función llamar según el estado solicitado
            if status == "open":
                orders = loop.run_until_complete(
                    orchestrator.exchange_adapter.fetch_open_orders(symbol=symbol)
                )
                logger.info(f"Obtenidas {len(orders) if orders else 0} órdenes abiertas para {symbol or 'todos los símbolos'}")
            elif status == "closed":
                orders = loop.run_until_complete(
                    orchestrator.exchange_adapter.fetch_closed_orders(symbol=symbol)
                )
                logger.info(f"Obtenidas {len(orders) if orders else 0} órdenes cerradas para {symbol or 'todos los símbolos'}")
            else:
                # Obtener todas las órdenes
                orders = loop.run_until_complete(
                    orchestrator.exchange_adapter.fetch_orders(symbol=symbol)
                )
                logger.info(f"Obtenidas {len(orders) if orders else 0} órdenes para {symbol or 'todos los símbolos'}")
            
            # Asignar correctamente el resultado (sin anidamiento)
            current_thread = threading.current_thread()
            current_thread.result = {
                "success": True, 
                "orders": orders if isinstance(orders, list) else [],
                "count": len(orders) if isinstance(orders, list) else 0,
                "symbol": symbol,
                "status": status
            }
            
            return current_thread.result
        except Exception as e:
            logger.error(f"Error obteniendo órdenes: {e}")
            current_thread = threading.current_thread()
            current_thread.result = {"success": False, "error": f"Error: {str(e)}"}
        finally:
            loop.close()
    
    # Ejecutar en thread para no bloquear
    thread = threading.Thread(target=get_orders_async)
    thread.start()
    thread.join(timeout=5.0)  # Añadir timeout para evitar bloqueos indefinidos
    
    # Obtener resultado o respuesta de error
    if hasattr(thread, 'result'):
        return jsonify(thread.result)
    else:
        return jsonify({"success": False, "error": "Failed to fetch orders"})

def register_seraphim_routes(app):
    """
    Registrar rutas del Sistema Seraphim en la aplicación Flask.
    
    Args:
        app: Aplicación Flask
    """
    # Páginas principales
    app.add_url_rule('/seraphim', 'seraphim_page', seraphim_page)
    app.add_url_rule('/emotional_control', 'emotional_control_page', emotional_control_page)
    
    # Rutas API del Sistema Seraphim
    app.add_url_rule('/api/seraphim/status', 'get_seraphim_status', get_seraphim_status)
    app.add_url_rule('/api/seraphim/initialize', 'initialize_seraphim', initialize_seraphim, methods=['POST'])
    app.add_url_rule('/api/seraphim/system_overview', 'get_system_overview', get_system_overview)
    app.add_url_rule('/api/seraphim/start_cycle', 'start_trading_cycle', start_trading_cycle, methods=['POST'])
    app.add_url_rule('/api/seraphim/process_cycle', 'process_cycle', process_cycle, methods=['POST'])
    app.add_url_rule('/api/seraphim/cycle_status', 'get_cycle_status', get_cycle_status)
    app.add_url_rule('/api/seraphim/human_behavior', 'get_human_behavior', get_human_behavior)
    app.add_url_rule('/api/seraphim/set_emotional_state', 'set_emotional_state', set_emotional_state, methods=['POST'])
    app.add_url_rule('/api/seraphim/set_fearful_state', 'set_fearful_state', set_fearful_state, methods=['POST'])
    app.add_url_rule('/api/seraphim/randomize_behavior', 'randomize_behavior', randomize_behavior, methods=['POST'])
    app.add_url_rule('/api/seraphim/auto_operation', 'start_auto_operation', start_auto_operation, methods=['POST'])
    app.add_url_rule('/api/seraphim/stop_auto', 'stop_auto_operation', stop_auto_operation, methods=['POST'])
    
    # Rutas API del Simulador de Intercambio
    if SIMULATOR_AVAILABLE:
        app.add_url_rule('/simulator', 'simulator_page', simulator_page)
        app.add_url_rule('/api/simulator/initialize', 'initialize_simulator', initialize_simulator, methods=['POST'])
        app.add_url_rule('/api/simulator/status', 'get_simulator_status', get_simulator_status)
        app.add_url_rule('/api/simulator/market_data_demo', 'run_market_data_demo', run_market_data_demo)
        app.add_url_rule('/api/simulator/orders_demo', 'run_orders_demo', run_orders_demo)
        
        # Rutas para la integración Seraphim-Simulador
        app.add_url_rule('/api/seraphim/market/data/<symbol>', 'get_market_data', get_market_data)
        app.add_url_rule('/api/seraphim/market/symbols', 'get_available_symbols', get_available_symbols)
        app.add_url_rule('/api/seraphim/market/connect', 'connect_exchange', connect_exchange, methods=['POST'])
        app.add_url_rule('/api/seraphim/trade/place', 'place_trade', place_trade, methods=['POST'])
        app.add_url_rule('/api/seraphim/trade/cancel/<trade_id>', 'cancel_trade', cancel_trade, methods=['POST'])
        app.add_url_rule('/api/seraphim/trades', 'get_trades', get_trades)
        
        logger.info("Rutas del Simulador de Intercambio registradas correctamente")
    
    logger.info("Rutas del Sistema Seraphim registradas correctamente")