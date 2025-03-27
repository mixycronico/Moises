"""
API REST para pruebas ARMAGEDÓN del Sistema Genesis Trascendental.

Este módulo implementa los endpoints REST para configurar, ejecutar y monitorear
las pruebas de resistencia extrema del Sistema Genesis.
"""

import os
import json
import logging
import subprocess
import threading
import time
import sqlite3
from flask import Blueprint, jsonify, request, send_file, render_template
from werkzeug.utils import secure_filename

# Configuración
ARMAGEDDON_LOG_FILE = "armageddon_ultra_direct_report.log"
DB_PATH = "cosmic_trading.db"
TEST_STATUS = {
    "running": False,
    "preparing": False,
    "initialized": False,
    "completed": False,
    "entity_count": 0,
    "operations": 0,
    "ops_per_second": 0,
    "failures": 0,
    "recoveries": 0,
    "events": [],
    "start_time": None,
    "progress": 0,
    "test_process": None
}

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("armageddon_api")

# Crear blueprint
armageddon_bp = Blueprint('armageddon', __name__)

# Funciones de utilidad
def update_test_status(**kwargs):
    """Actualizar estado de prueba de forma segura."""
    global TEST_STATUS
    for key, value in kwargs.items():
        if key in TEST_STATUS:
            TEST_STATUS[key] = value
    
    # Limitar número de eventos en memoria
    if 'events' in TEST_STATUS and len(TEST_STATUS['events']) > 100:
        TEST_STATUS['events'] = TEST_STATUS['events'][-100:]

def add_test_event(message, event_type="info"):
    """Añadir evento al registro de eventos."""
    if 'events' not in TEST_STATUS:
        TEST_STATUS['events'] = []
    
    TEST_STATUS['events'].append({
        "timestamp": time.time(),
        "message": message,
        "type": event_type
    })
    logger.info(f"[{event_type}] {message}")

def get_test_config_from_request():
    """Extraer configuración de prueba de la solicitud."""
    try:
        config = request.json
        # Validar y aplicar límites
        entity_count = max(5, min(100, int(config.get('entity_count', 20))))
        duration = max(10, min(300, int(config.get('duration', 60))))
        volatility = max(1, min(50, int(config.get('volatility_factor', 10))))
        energy = max(1, min(20, int(config.get('energy_factor', 5))))
        communication = max(1, min(20, int(config.get('communication_factor', 8))))
        db_stress = max(1, min(30, int(config.get('db_stress_factor', 10))))
        fail_prob = max(0, min(0.5, float(config.get('fail_probability', 0.15))))
        test_type = config.get('test_type', 'ultra')
        
        return {
            "entity_count": entity_count,
            "duration": duration,
            "volatility_factor": volatility,
            "energy_factor": energy,
            "communication_factor": communication,
            "db_stress_factor": db_stress,
            "fail_probability": fail_prob,
            "test_type": test_type
        }
    except (ValueError, TypeError) as e:
        logger.error(f"Error al procesar configuración: {str(e)}")
        return None

def modify_test_script(config):
    """Modificar script de prueba con la configuración proporcionada."""
    try:
        script_path = "run_armageddon_ultra_direct.py"
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Modificar variables de configuración
        replacements = [
            (r"MAX_ENTITIES = \d+", f"MAX_ENTITIES = {config['entity_count']}"),
            (r"TEST_DURATION = \d+", f"TEST_DURATION = {config['duration']}"),
            (r"VOLATILITY_FACTOR = \d+", f"VOLATILITY_FACTOR = {config['volatility_factor']}"),
            (r"ENERGY_DRAIN_FACTOR = \d+", f"ENERGY_DRAIN_FACTOR = {config['energy_factor']}"),
            (r"COMMUNICATION_FACTOR = \d+", f"COMMUNICATION_FACTOR = {config['communication_factor']}"),
            (r"DATABASE_STRESS_FACTOR = \d+", f"DATABASE_STRESS_FACTOR = {config['db_stress_factor']}"),
            (r"FAILURE_PROBABILITY = 0\.\d+", f"FAILURE_PROBABILITY = {config['fail_probability']}")
        ]
        
        import re
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        with open(script_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Script modificado con configuración: {config}")
        return True
    except Exception as e:
        logger.error(f"Error al modificar script: {str(e)}")
        return False

def execute_test(config):
    """Ejecutar prueba ARMAGEDÓN con la configuración dada."""
    global TEST_STATUS
    
    try:
        # Modificar script con configuración
        if not modify_test_script(config):
            add_test_event("Error al configurar parámetros de prueba", "error")
            update_test_status(running=False, preparing=False)
            return
        
        # Iniciar proceso
        cmd = ["python", "run_armageddon_ultra_direct.py"]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        update_test_status(
            test_process=process,
            running=True,
            start_time=time.time(),
            entity_count=config['entity_count']
        )
        
        # Iniciar hilos para capturar salida
        def read_output(pipe, is_error=False):
            for line in iter(pipe.readline, ''):
                line = line.strip()
                if not line:
                    continue
                    
                # Procesar línea para actualizar estadísticas
                process_output_line(line)
                
                # Registrar como evento en la consola web
                event_type = "error" if is_error else "info"
                if "ERROR" in line or "CRITICAL" in line:
                    event_type = "error"
                elif "WARNING" in line:
                    event_type = "warning"
                
                # Solo añadir líneas importantes como eventos
                if "[ERROR]" in line or "[WARNING]" in line or "CHAOS:" in line or "INFO - Rendimiento:" in line:
                    add_test_event(line, event_type)
        
        threading.Thread(target=read_output, args=(process.stdout,), daemon=True).start()
        threading.Thread(target=read_output, args=(process.stderr, True), daemon=True).start()
        
        # Hilo para monitorear finalización del proceso
        def monitor_process():
            process.wait()
            update_test_status(
                running=False,
                completed=True,
                progress=100
            )
            add_test_event("Prueba ARMAGEDÓN completada")
            
        threading.Thread(target=monitor_process, daemon=True).start()
        
        add_test_event(f"Prueba ARMAGEDÓN iniciada con {config['entity_count']} entidades")
        return True
        
    except Exception as e:
        logger.error(f"Error al ejecutar prueba: {str(e)}")
        update_test_status(running=False, preparing=False)
        add_test_event(f"Error al ejecutar prueba: {str(e)}", "error")
        return False

def process_output_line(line):
    """Procesar línea de salida para actualizar estadísticas."""
    try:
        # Extraer información de rendimiento
        if "INFO - Rendimiento:" in line:
            parts = line.split("Rendimiento:")[1].split("|")
            
            # Extraer ops/s
            ops_s = float(parts[0].strip().split()[0])
            update_test_status(ops_per_second=ops_s)
            
            # Extraer total de operaciones
            if "Total:" in parts[1]:
                ops_total = int(parts[1].strip().split()[1])
                update_test_status(operations=ops_total)
            
            # Extraer fallos
            if "Fallos:" in parts[2]:
                failures = int(parts[2].strip().split()[1])
                update_test_status(failures=failures)
            
            # Extraer recuperaciones
            if "Recuperaciones:" in parts[3]:
                recoveries = int(parts[3].strip().split()[1])
                update_test_status(recoveries=recoveries)
    except Exception as e:
        logger.error(f"Error procesando línea: {str(e)}")

def stop_running_test():
    """Detener prueba en ejecución."""
    global TEST_STATUS
    
    if TEST_STATUS.get('test_process'):
        try:
            TEST_STATUS['test_process'].terminate()
            
            # Dar tiempo para terminar
            time.sleep(2)
            
            # Forzar terminación si aún está en ejecución
            if TEST_STATUS['test_process'].poll() is None:
                TEST_STATUS['test_process'].kill()
            
            add_test_event("Prueba detenida manualmente", "warning")
            update_test_status(running=False, completed=True)
            return True
        except Exception as e:
            logger.error(f"Error al detener prueba: {str(e)}")
            return False
    return True

def prepare_database():
    """Preparar base de datos para prueba ARMAGEDÓN."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verificar tablas necesarias
        tables = [
            "cosmic_entities", "entity_logs", "trade_history", 
            "entity_messages", "collective_knowledge"
        ]
        
        for table in tables:
            try:
                # Verificar si existe la tabla
                cursor.execute(f"SELECT 1 FROM {table} LIMIT 1")
            except sqlite3.OperationalError:
                # Crear tabla si no existe
                if table == "cosmic_entities":
                    cursor.execute("""
                        CREATE TABLE cosmic_entities (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            role TEXT NOT NULL,
                            father TEXT NOT NULL,
                            level REAL DEFAULT 1.0,
                            energy REAL DEFAULT 100.0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                elif table == "entity_logs":
                    cursor.execute("""
                        CREATE TABLE entity_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            entity_id INTEGER,
                            log_type TEXT,
                            message TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                elif table == "trade_history":
                    cursor.execute("""
                        CREATE TABLE trade_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            entity_id INTEGER,
                            operation_type TEXT,
                            amount REAL,
                            price REAL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                elif table == "entity_messages":
                    cursor.execute("""
                        CREATE TABLE entity_messages (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            sender TEXT,
                            receiver TEXT,
                            message TEXT,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                elif table == "collective_knowledge":
                    cursor.execute("""
                        CREATE TABLE collective_knowledge (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            knowledge_type TEXT,
                            content TEXT,
                            source TEXT,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
        
        # Limpiar datos de pruebas anteriores
        for table in tables:
            cursor.execute(f"DELETE FROM {table} WHERE id IN (SELECT id FROM {table} ORDER BY id DESC LIMIT 1000 OFFSET 100)")
        
        conn.commit()
        conn.close()
        
        logger.info("Base de datos preparada para prueba ARMAGEDÓN")
        return True
    except Exception as e:
        logger.error(f"Error al preparar base de datos: {str(e)}")
        return False

# Rutas API
@armageddon_bp.route('/api/status')
def get_status():
    """Obtener estado actual de la prueba."""
    global TEST_STATUS
    
    # Calcular progreso si está en ejecución
    if TEST_STATUS['running'] and TEST_STATUS['start_time']:
        elapsed = time.time() - TEST_STATUS['start_time']
        config_duration = TEST_STATUS.get('duration', 60)
        TEST_STATUS['progress'] = min(100, (elapsed / config_duration) * 100)
    
    # Crear copia del estado sin el proceso (no serializable)
    status_copy = dict(TEST_STATUS)
    if 'test_process' in status_copy:
        del status_copy['test_process']
    
    return jsonify(status_copy)

@armageddon_bp.route('/api/initialize', methods=['POST'])
def initialize_components():
    """Inicializar componentes para prueba ARMAGEDÓN."""
    if TEST_STATUS['running']:
        return jsonify({
            "status": "error",
            "message": "No se puede inicializar mientras hay una prueba en ejecución"
        })
    
    update_test_status(
        initialized=True,
        preparing=True,
        running=False,
        completed=False,
        entity_count=0,
        operations=0,
        ops_per_second=0,
        failures=0,
        recoveries=0,
        events=[],
        start_time=None,
        progress=0
    )
    
    add_test_event("Componentes inicializados correctamente")
    
    return jsonify({
        "status": "success",
        "message": "Componentes inicializados correctamente"
    })

@armageddon_bp.route('/api/prepare_database', methods=['POST'])
def prepare_db_endpoint():
    """Preparar base de datos para prueba ARMAGEDÓN."""
    if TEST_STATUS['running']:
        return jsonify({
            "status": "error",
            "message": "No se puede preparar la base de datos mientras hay una prueba en ejecución"
        })
    
    if not TEST_STATUS['initialized']:
        return jsonify({
            "status": "error",
            "message": "Debe inicializar los componentes primero"
        })
    
    success = prepare_database()
    
    if success:
        add_test_event("Base de datos preparada correctamente")
        return jsonify({
            "status": "success",
            "message": "Base de datos preparada correctamente"
        })
    else:
        add_test_event("Error al preparar la base de datos", "error")
        return jsonify({
            "status": "error",
            "message": "Error al preparar la base de datos"
        })

@armageddon_bp.route('/api/start', methods=['POST'])
def start_test():
    """Iniciar prueba ARMAGEDÓN."""
    if TEST_STATUS['running']:
        return jsonify({
            "status": "error",
            "message": "Ya hay una prueba en ejecución"
        })
    
    if not TEST_STATUS['initialized']:
        return jsonify({
            "status": "error",
            "message": "Debe inicializar los componentes primero"
        })
    
    # Obtener configuración
    config = get_test_config_from_request()
    if not config:
        return jsonify({
            "status": "error",
            "message": "Configuración de prueba inválida"
        })
    
    # Guardar duración para cálculos de progreso
    update_test_status(duration=config['duration'])
    
    # Ejecutar prueba
    success = execute_test(config)
    
    if success:
        return jsonify({
            "status": "success",
            "message": "Prueba ARMAGEDÓN iniciada correctamente"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Error al iniciar la prueba ARMAGEDÓN"
        })

@armageddon_bp.route('/api/stop', methods=['POST'])
def stop_test():
    """Detener prueba ARMAGEDÓN."""
    if not TEST_STATUS['running']:
        return jsonify({
            "status": "warning",
            "message": "No hay ninguna prueba en ejecución"
        })
    
    success = stop_running_test()
    
    if success:
        return jsonify({
            "status": "success",
            "message": "Prueba detenida correctamente"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Error al detener la prueba"
        })

@armageddon_bp.route('/api/results')
def get_results():
    """Obtener resultados de la prueba."""
    if TEST_STATUS['running']:
        return jsonify({
            "status": "error",
            "message": "La prueba aún está en ejecución"
        })
    
    # Métricas adicionales simuladas (en una implementación real, se calcularían de forma adecuada)
    metrics = {
        "max_cpu_usage": round(70 + 20 * TEST_STATUS.get('ops_per_second', 0) / 1000, 1),
        "max_memory_usage": round(200 + 150 * TEST_STATUS.get('entity_count', 0) / 50, 1),
        "max_disk_usage": round(5 + 15 * TEST_STATUS.get('operations', 0) / 10000, 1),
        "peak_operations": round(TEST_STATUS.get('ops_per_second', 0) * 1.5, 1)
    }
    
    # Crear copia del estado sin el proceso (no serializable)
    status_copy = dict(TEST_STATUS)
    if 'test_process' in status_copy:
        del status_copy['test_process']
    
    return jsonify({
        "status": "success",
        "message": "Resultados obtenidos correctamente",
        "data": status_copy,
        "metrics": metrics
    })

@armageddon_bp.route('/api/report')
def get_report():
    """Obtener reporte de la prueba."""
    if os.path.exists(ARMAGEDDON_LOG_FILE):
        return send_file(
            ARMAGEDDON_LOG_FILE,
            as_attachment=True,
            download_name="armageddon_report.log"
        )
    else:
        return jsonify({
            "status": "error",
            "message": "El archivo de reporte no existe"
        })

@armageddon_bp.route('/')
def armageddon_page():
    """Página principal de ARMAGEDÓN."""
    return render_template('armageddon.html')

def register_armageddon_routes(app):
    """Registrar rutas de ARMAGEDÓN en la aplicación Flask."""
    app.register_blueprint(armageddon_bp, url_prefix='/armageddon_test')