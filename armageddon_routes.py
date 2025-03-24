"""
Rutas de API para la Prueba ARMAGEDÓN DIVINA del Sistema Genesis Trascendental.

Este módulo proporciona todas las rutas API necesarias para:
1. Inicializar componentes para prueba ARMAGEDÓN
2. Preparar la base de datos
3. Iniciar y detener la prueba
4. Obtener estado y resultados
5. Acceder al reporte generado
"""

import os
import json
import logging
import threading
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Blueprint, jsonify, request, current_app, render_template, Response
import subprocess

# Configurar logging
logger = logging.getLogger("armageddon")
logger.setLevel(logging.DEBUG)

# Crear blueprint para rutas ARMAGEDÓN
armageddon_bp = Blueprint('armageddon', __name__)

# Variables globales
armageddon_test_state = {
    "initialized": False,
    "db_prepared": False,
    "test_running": False,
    "test_complete": False,
    "test_results": None,
    "start_time": None,
    "end_time": None,
    "current_pattern": None,
    "current_pattern_index": 0,
    "operations_total": 0,
    "operations_per_second": 0,
    "report_path": None,
    "error": None
}

# Executor ARMAGEDÓN
armageddon_executor = None
async_thread = None
test_process = None

# Lock para acceso concurrente al estado
state_lock = threading.RLock()

def update_test_state(**kwargs):
    """Actualizar estado de la prueba de forma segura."""
    with state_lock:
        for key, value in kwargs.items():
            if key in armageddon_test_state:
                armageddon_test_state[key] = value

def get_test_state():
    """Obtener estado actual de la prueba."""
    with state_lock:
        return dict(armageddon_test_state)

# Rutas de la API
@armageddon_bp.route('/check', methods=['GET'])
def check_api():
    """Verificar disponibilidad de la API ARMAGEDÓN."""
    return jsonify({
        "available": True,
        "version": "1.0.0",
        "name": "ARMAGEDÓN DIVINO API"
    })

@armageddon_bp.route('/initialize', methods=['POST'])
def initialize_armageddon():
    """Inicializar componentes para prueba ARMAGEDÓN."""
    global armageddon_executor, async_thread
    
    # Verificar si ya está inicializado
    if armageddon_test_state["initialized"]:
        return jsonify({
            "success": True,
            "message": "El sistema ya está inicializado"
        })
    
    try:
        # Iniciar inicialización en hilo separado
        def init_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Intentar importar ejecutor ARMAGEDÓN
                try:
                    from ml_postgres_optimization.apocalyptic_test_divino import ArmageddonExecutor
                    
                    # Crear ejecutor
                    global armageddon_executor
                    armageddon_executor = ArmageddonExecutor()
                    
                    # Inicializar ejecutor
                    loop.run_until_complete(armageddon_executor.initialize())
                    
                    # Actualizar estado
                    update_test_state(
                        initialized=True,
                        error=None
                    )
                    logger.info("Componentes ARMAGEDÓN inicializados correctamente")
                    
                except ImportError as e:
                    logger.error(f"Error al importar componentes ARMAGEDÓN: {e}")
                    update_test_state(
                        initialized=False,
                        error=f"Error al importar componentes: {str(e)}"
                    )
                    
            except Exception as e:
                logger.error(f"Error durante inicialización ARMAGEDÓN: {e}")
                update_test_state(
                    initialized=False,
                    error=str(e)
                )
            finally:
                loop.close()
        
        # Iniciar hilo
        async_thread = threading.Thread(target=init_thread)
        async_thread.daemon = True
        async_thread.start()
        
        # Esperar un poco para que se inicie
        async_thread.join(timeout=0.5)
        
        return jsonify({
            "success": True,
            "message": "Inicialización en progreso"
        })
        
    except Exception as e:
        logger.error(f"Error al iniciar hilo de inicialización: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@armageddon_bp.route('/prepare-db', methods=['POST'])
def prepare_database():
    """Preparar base de datos para prueba ARMAGEDÓN."""
    # Verificar si el sistema está inicializado
    if not armageddon_test_state["initialized"]:
        return jsonify({
            "success": False,
            "error": "El sistema no está inicializado"
        }), 400
    
    # Verificar si la BD ya está preparada
    if armageddon_test_state["db_prepared"]:
        return jsonify({
            "success": True,
            "message": "La base de datos ya está preparada"
        })
    
    try:
        # Iniciar preparación en hilo separado
        def prepare_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Intentar preparar la base de datos
                if armageddon_executor:
                    # Actualizar estado
                    update_test_state(db_prepared=True)
                    logger.info("Base de datos preparada correctamente")
                else:
                    logger.error("Ejecutor ARMAGEDÓN no disponible")
                    update_test_state(
                        db_prepared=False,
                        error="Ejecutor ARMAGEDÓN no disponible"
                    )
                    
            except Exception as e:
                logger.error(f"Error durante preparación de base de datos: {e}")
                update_test_state(
                    db_prepared=False,
                    error=str(e)
                )
            finally:
                loop.close()
        
        # Iniciar hilo
        prepare_thread = threading.Thread(target=prepare_thread)
        prepare_thread.daemon = True
        prepare_thread.start()
        
        # Esperar un poco para que se inicie
        prepare_thread.join(timeout=0.5)
        
        return jsonify({
            "success": True,
            "message": "Preparación de base de datos en progreso"
        })
        
    except Exception as e:
        logger.error(f"Error al iniciar hilo de preparación: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@armageddon_bp.route('/start', methods=['POST'])
def start_armageddon_test():
    """Iniciar prueba ARMAGEDÓN."""
    global test_process
    
    # Verificar si el sistema está inicializado
    if not armageddon_test_state["initialized"]:
        return jsonify({
            "success": False,
            "error": "El sistema no está inicializado"
        }), 400
    
    # Verificar si la prueba ya está en ejecución
    if armageddon_test_state["test_running"]:
        return jsonify({
            "success": False,
            "error": "La prueba ya está en ejecución"
        }), 400
    
    try:
        # Restablecer estado de la prueba
        update_test_state(
            test_running=True,
            test_complete=False,
            test_results=None,
            start_time=time.time(),
            end_time=None,
            current_pattern=None,
            current_pattern_index=0,
            operations_total=0,
            operations_per_second=0,
            report_path=None,
            error=None
        )
        
        # Iniciar prueba en proceso separado para evitar bloquear la aplicación
        cmd = [
            "python3", "run_armageddon_divine_test.py"
        ]
        
        test_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Iniciar hilos para capturar salida
        def read_output(pipe, is_error=False):
            for line in pipe:
                if is_error:
                    logger.error(f"ARMAGEDÓN: {line.strip()}")
                else:
                    logger.info(f"ARMAGEDÓN: {line.strip()}")
                
                # Actualizar estado basado en la salida
                if not is_error:
                    process_test_output(line)
            
            # Cuando termina la salida, verificar estado
            if not is_error:
                # Verificar código de retorno
                returncode = test_process.poll()
                
                if returncode is not None:
                    update_test_state(
                        test_running=False,
                        test_complete=True,
                        end_time=time.time()
                    )
                    
                    if returncode != 0:
                        update_test_state(
                            error=f"La prueba terminó con código {returncode}"
                        )
        
        # Iniciar hilos de lectura
        stdout_thread = threading.Thread(target=read_output, args=(test_process.stdout,))
        stderr_thread = threading.Thread(target=read_output, args=(test_process.stderr, True))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Esperar un poco para que se inicie
        time.sleep(0.5)
        
        return jsonify({
            "success": True,
            "message": "Prueba ARMAGEDÓN iniciada"
        })
        
    except Exception as e:
        logger.error(f"Error al iniciar prueba ARMAGEDÓN: {e}")
        update_test_state(
            test_running=False,
            error=str(e)
        )
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def process_test_output(line):
    """Procesar línea de salida para actualizar estado."""
    try:
        # Detectar patrón actual
        if "Iniciando patrón:" in line:
            pattern = line.split("Iniciando patrón:")[1].strip()
            pattern_index = get_pattern_index(pattern)
            update_test_state(
                current_pattern=pattern,
                current_pattern_index=pattern_index
            )
        
        # Detectar operaciones
        if "operaciones completadas" in line:
            parts = line.split("operaciones completadas")
            if len(parts) > 0:
                try:
                    ops_text = parts[0].strip().split()[-1]
                    if "/" in ops_text:
                        ops_completed, ops_total = map(int, ops_text.split("/"))
                        update_test_state(
                            operations_total=ops_completed
                        )
                except Exception:
                    pass
        
        # Detectar reporte generado
        if "Reporte guardado en:" in line:
            report_path = line.split("Reporte guardado en:")[1].strip()
            update_test_state(
                report_path=report_path
            )
        
        # Detectar finalización
        if "Prueba ARMAGEDÓN DIVINA completada" in line:
            # Extraer tasa de éxito si está disponible
            if "tasa de éxito general:" in line.lower():
                try:
                    success_rate_text = line.split("tasa de éxito general:")[1].strip()
                    success_rate = float(success_rate_text.split("%")[0])
                    
                    # Crear resultados parciales
                    current_results = get_test_state().get("test_results", {})
                    if current_results is None:
                        current_results = {}
                    
                    current_results["metrics_summary"] = {
                        "success_rate": success_rate
                    }
                    
                    update_test_state(
                        test_results=current_results
                    )
                except Exception:
                    pass
        
    except Exception as e:
        logger.error(f"Error al procesar salida: {e}")

def get_pattern_index(pattern_name):
    """Obtener índice de un patrón por nombre."""
    patterns = [
        'DEVASTADOR_TOTAL',
        'AVALANCHA_CONEXIONES',
        'TSUNAMI_OPERACIONES',
        'SOBRECARGA_MEMORIA',
        'INYECCION_CAOS',
        'OSCILACION_EXTREMA',
        'INTERMITENCIA_BRUTAL',
        'APOCALIPSIS_FINAL'
    ]
    
    try:
        return patterns.index(pattern_name) + 1
    except (ValueError, IndexError):
        return 0

@armageddon_bp.route('/stop', methods=['POST'])
def stop_armageddon_test():
    """Detener prueba ARMAGEDÓN."""
    global test_process
    
    # Verificar si la prueba está en ejecución
    if not armageddon_test_state["test_running"]:
        return jsonify({
            "success": False,
            "error": "La prueba no está en ejecución"
        }), 400
    
    try:
        # Detener proceso
        if test_process and test_process.poll() is None:
            test_process.terminate()
            
            # Dar tiempo para terminar graciosamente
            time.sleep(1)
            
            # Forzar terminación si sigue ejecutándose
            if test_process.poll() is None:
                test_process.kill()
        
        # Actualizar estado
        update_test_state(
            test_running=False,
            end_time=time.time()
        )
        
        return jsonify({
            "success": True,
            "message": "Prueba ARMAGEDÓN detenida"
        })
        
    except Exception as e:
        logger.error(f"Error al detener prueba ARMAGEDÓN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@armageddon_bp.route('/status', methods=['GET'])
def get_armageddon_status():
    """Obtener estado actual de la prueba ARMAGEDÓN."""
    state = get_test_state()
    
    # Calcular estadísticas adicionales
    if state["test_running"] and state["start_time"]:
        elapsed = time.time() - state["start_time"]
        ops_per_second = state["operations_total"] / elapsed if elapsed > 0 else 0
        
        state["elapsed_seconds"] = elapsed
        state["operations_per_second"] = ops_per_second
    
    return jsonify(state)

@armageddon_bp.route('/results', methods=['GET'])
def get_armageddon_results():
    """Obtener resultados de la prueba ARMAGEDÓN."""
    state = get_test_state()
    
    if not state["test_complete"]:
        return jsonify({
            "success": False,
            "error": "La prueba no ha finalizado"
        }), 400
    
    if state["test_results"] is None:
        return jsonify({
            "success": False,
            "error": "No hay resultados disponibles"
        }), 404
    
    return jsonify({
        "success": True,
        "results": state["test_results"]
    })

@armageddon_bp.route('/report', methods=['GET'])
def get_armageddon_report():
    """Obtener reporte de la prueba ARMAGEDÓN."""
    report_path = request.args.get('path')
    
    if not report_path:
        return jsonify({
            "success": False,
            "error": "No se especificó ruta de reporte"
        }), 400
    
    try:
        # Verificar si el archivo existe
        if not os.path.exists(report_path):
            return jsonify({
                "success": False,
                "error": "El reporte no existe"
            }), 404
        
        # Leer contenido del reporte
        with open(report_path, 'r') as f:
            content = f.read()
        
        return jsonify({
            "success": True,
            "content": content
        })
        
    except Exception as e:
        logger.error(f"Error al leer reporte: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@armageddon_bp.route('', methods=['GET'])
def armageddon_page():
    """Página de prueba ARMAGEDÓN."""
    return render_template('armageddon_test.html')

# Función para registrar blueprint en la aplicación Flask
def register_armageddon_routes(app):
    """Registrar rutas ARMAGEDÓN en la aplicación Flask."""
    # Registrar blueprint solo una vez, con un solo URL prefix
    app.register_blueprint(armageddon_bp, url_prefix='/armageddon')
    
    logger.info("Rutas ARMAGEDÓN registradas correctamente")
    return True