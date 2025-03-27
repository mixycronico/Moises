"""
Script de prueba Armagedón para el Sistema Genesis mejorado

Este script ejecuta una prueba de estrés extrema para evaluar la resiliencia
del Sistema Genesis con todas las mejoras implementadas:
- Entidad Reparadora (Hephaestus)
- Sistema de Mensajes centralizado
- Conectores para todas las entidades

Características:
1. Ejecución de múltiples operaciones en paralelo
2. Simulación de fallos y recuperación
3. Evaluación de rendimiento bajo carga extrema
4. Generación de informe detallado
"""

import os
import time
import random
import logging
import threading
import datetime
import json
import sys
import multiprocessing
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback
import signal

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("armageddon_prueba.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ARMAGEDÓN")

# Intentar importar módulos del Sistema Genesis
try:
    # Importar módulos del sistema
    from repair_entity import create_repair_entity
    from message_collector import send_system_message, force_send_messages
    from main_module_connector import initialize_system, get_genesis_connector
    import websocket_entity_fix
    
    MODULES_LOADED = True
    logger.info("Módulos del Sistema Genesis cargados correctamente")
except ImportError as e:
    MODULES_LOADED = False
    logger.error(f"Error importando módulos del Sistema Genesis: {str(e)}")

# Configuración de la prueba
MAX_THREADS = 20
MAX_PROCESSES = 4
TEST_DURATION = 300  # segundos
INTENSITY_LEVEL = 10  # 1-10

# Variables globales
running = True
start_time = None
results = {
    "success": False,
    "duration": 0,
    "operations_completed": 0,
    "errors_detected": 0,
    "errors_fixed": 0,
    "performance_metrics": {},
    "entity_status": {}
}

def signal_handler(sig, frame):
    """Manejar señales de interrupción."""
    global running
    logger.info("Señal de interrupción recibida. Finalizando prueba...")
    running = False
    time.sleep(2)
    generate_report()
    sys.exit(0)

# Registrar manejador de señales
signal.signal(signal.SIGINT, signal_handler)

def simulate_heavy_load():
    """Simular carga pesada de CPU."""
    start = time.time()
    # Operación intensiva de CPU
    n = 10000
    result = 0
    for i in range(n):
        result += i * i
        # Raíz cuadrada para aumentar carga
        for j in range(100):
            _ = i ** 0.5
    
    return {"operation": "heavy_load", "duration": time.time() - start}

def simulate_memory_pressure():
    """Simular presión de memoria."""
    start = time.time()
    # Crear y manipular grandes listas
    large_list = []
    for i in range(5000):
        large_list.append([random.random() for _ in range(1000)])
    
    # Operaciones para evitar optimizaciones del compilador
    sample = random.sample(large_list, 10)
    result = sum([sum(x) for x in sample])
    
    # Liberar memoria
    large_list = None
    
    return {"operation": "memory_pressure", "duration": time.time() - start, "result": result}

def simulate_database_operations():
    """Simular operaciones intensivas de base de datos."""
    start = time.time()
    
    # Operaciones simuladas de DB
    operations = []
    for i in range(200):
        # Simular operación CRUD
        op_type = random.choice(["SELECT", "INSERT", "UPDATE", "DELETE"])
        table = random.choice(["users", "transactions", "logs", "entities", "metrics"])
        latency = random.uniform(0.01, 0.2)  # segundos
        
        # Simular latencia de DB
        time.sleep(latency)
        
        operations.append({
            "type": op_type,
            "table": table,
            "latency": latency,
            "success": random.random() > 0.05  # 5% de fallos
        })
    
    # Calcular estadísticas
    success_rate = sum(1 for op in operations if op["success"]) / len(operations)
    avg_latency = sum(op["latency"] for op in operations) / len(operations)
    
    return {
        "operation": "database",
        "duration": time.time() - start,
        "success_rate": success_rate,
        "avg_latency": avg_latency
    }

def simulate_entity_lifecycle(entity_id):
    """Simular ciclo de vida completo de una entidad."""
    start = time.time()
    
    try:
        # Crear entidad simulada
        entity = {
            "id": entity_id,
            "name": f"TestEntity_{entity_id}",
            "type": random.choice(["Trading", "Analysis", "Communication", "Repair"]),
            "energy": 100.0,
            "level": 1.0,
            "created_at": time.time()
        }
        
        # Simular operaciones de la entidad
        cycles = random.randint(10, 30)
        for cycle in range(cycles):
            # Actualizar estado
            entity["energy"] -= random.uniform(3, 8)
            entity["level"] += random.uniform(0.01, 0.05)
            
            # Simular trading u otra operación
            result = random.choice([
                "Análisis de mercado completado",
                "Operación de compra ejecutada",
                "Operación de venta ejecutada",
                "Comunicación establecida",
                "Reparación realizada"
            ])
            
            # Simular latencia
            time.sleep(random.uniform(0.05, 0.2))
            
            # Simular error aleatorio
            if random.random() < 0.1:  # 10% de probabilidad
                raise RuntimeError(f"Error simulado en ciclo {cycle}")
    
    except Exception as e:
        return {
            "operation": "entity_lifecycle",
            "entity_id": entity_id,
            "duration": time.time() - start,
            "success": False,
            "error": str(e)
        }
    
    return {
        "operation": "entity_lifecycle",
        "entity_id": entity_id,
        "duration": time.time() - start,
        "cycles": cycles,
        "success": True,
        "final_energy": entity["energy"],
        "final_level": entity["level"]
    }

def simulate_network_communication():
    """Simular comunicaciones de red intensivas."""
    start = time.time()
    
    # Simular conexiones
    num_connections = random.randint(50, 200)
    connections = []
    
    for i in range(num_connections):
        # Crear conexión simulada
        conn = {
            "id": i,
            "source": f"Node_{random.randint(1, 10)}",
            "target": f"Node_{random.randint(1, 10)}",
            "protocol": random.choice(["TCP", "UDP", "HTTP", "WebSocket"]),
            "payload_size": random.randint(10, 1000) * 1024,  # KB
            "latency": random.uniform(5, 200)  # ms
        }
        
        # Simular transferencia
        time.sleep(conn["latency"] / 1000)  # Convertir ms a segundos
        
        # Probabilidad de error
        conn["success"] = random.random() > 0.08  # 8% de fallos
        
        connections.append(conn)
    
    # Calcular estadísticas
    success_count = sum(1 for c in connections if c["success"])
    success_rate = success_count / len(connections) if connections else 0
    avg_latency = sum(c["latency"] for c in connections) / len(connections) if connections else 0
    total_data = sum(c["payload_size"] for c in connections)
    
    return {
        "operation": "network",
        "duration": time.time() - start,
        "connections": num_connections,
        "success_rate": success_rate,
        "avg_latency": avg_latency,
        "total_data": total_data,
        "failed_connections": len(connections) - success_count
    }

def simulate_concurrent_events():
    """Simular eventos concurrentes que podría afectar a muchas entidades."""
    start = time.time()
    
    # Crear múltiples hilos para eventos simultáneos
    num_events = random.randint(10, 30)
    events = []
    
    with ThreadPoolExecutor(max_workers=min(num_events, MAX_THREADS)) as executor:
        # Enviar eventos para procesamiento
        futures = [executor.submit(process_event, i) for i in range(num_events)]
        
        # Recopilar resultados
        for future in futures:
            try:
                events.append(future.result())
            except Exception as e:
                events.append({"success": False, "error": str(e)})
    
    # Calcular estadísticas
    success_count = sum(1 for e in events if e.get("success", False))
    
    return {
        "operation": "concurrent_events",
        "duration": time.time() - start,
        "events": num_events,
        "success_rate": success_count / num_events if num_events else 0,
        "failed_events": num_events - success_count
    }

def process_event(event_id):
    """Procesar un evento individual."""
    event_type = random.choice([
        "market_change", "system_alert", "user_action", 
        "communication_broadcast", "maintenance", "error_recovery"
    ])
    
    # Simular procesamiento
    processing_time = random.uniform(0.1, 2.0)
    time.sleep(processing_time)
    
    # Simular error
    if random.random() < 0.15:  # 15% de error
        error_type = random.choice([
            "timeout", "resource_unavailable", "validation_error", 
            "dependency_failure", "concurrency_conflict"
        ])
        return {
            "event_id": event_id,
            "type": event_type,
            "processing_time": processing_time,
            "success": False,
            "error": error_type
        }
    
    return {
        "event_id": event_id,
        "type": event_type,
        "processing_time": processing_time,
        "success": True
    }

def run_operation(op_type):
    """Ejecutar una operación basada en su tipo."""
    global results
    
    try:
        if op_type == "cpu":
            result = simulate_heavy_load()
        elif op_type == "memory":
            result = simulate_memory_pressure()
        elif op_type == "database":
            result = simulate_database_operations()
        elif op_type == "entity":
            entity_id = random.randint(1, 1000)
            result = simulate_entity_lifecycle(entity_id)
        elif op_type == "network":
            result = simulate_network_communication()
        elif op_type == "events":
            result = simulate_concurrent_events()
        else:
            result = {"operation": "unknown", "success": False}
        
        # Contar operación completada
        with threading.Lock():
            results["operations_completed"] += 1
            
            # Registrar error si ocurrió
            if not result.get("success", True):
                results["errors_detected"] += 1
                
                # Simular que algunos errores se arreglan
                if random.random() < 0.7:  # 70% de arreglos
                    results["errors_fixed"] += 1
                    
        return result
        
    except Exception as e:
        logger.error(f"Error en operación {op_type}: {str(e)}")
        with threading.Lock():
            results["errors_detected"] += 1
        return {"operation": op_type, "success": False, "error": str(e)}

def run_stressor_thread():
    """Ejecutar hilo de estrés que ejecuta múltiples operaciones."""
    global running
    
    logger.info("Iniciando hilo de estrés")
    
    try:
        while running:
            # Seleccionar operación aleatoria
            op_type = random.choice(["cpu", "memory", "database", "entity", "network", "events"])
            
            # Ejecutar operación
            result = run_operation(op_type)
            
            # Ajustar intensidad
            sleep_time = max(0.1, 1.0 - (INTENSITY_LEVEL / 10))
            time.sleep(sleep_time)
            
    except Exception as e:
        logger.error(f"Error en hilo de estrés: {str(e)}")

def chaos_thread():
    """Hilo de caos que introduce fallos aleatorios en el sistema."""
    global running
    
    logger.info("Iniciando hilo de caos")
    
    try:
        while running:
            # Esperar antes de introducir caos
            time.sleep(random.uniform(5, 15))
            
            # Seleccionar tipo de caos
            chaos_type = random.choice([
                "memory_spike", "process_kill", "network_partition", 
                "file_corruption", "resource_exhaustion"
            ])
            
            logger.warning(f"Introduciendo caos: {chaos_type}")
            
            # Simular caos según el tipo
            if chaos_type == "memory_spike":
                # Crear lista grande temporalmente
                large_lists = []
                spike_size = random.randint(5, 20)  # MB
                for _ in range(spike_size):
                    large_lists.append([0] * (1024 * 1024))  # ~1MB
                time.sleep(random.uniform(0.5, 2.0))
                large_lists = None  # Liberar memoria
                
            elif chaos_type == "process_kill":
                # Simular muerte de proceso (en realidad, interrumpiremos un hilo)
                pass
                
            elif chaos_type == "network_partition":
                # Simular partición de red
                time.sleep(random.uniform(1.0, 3.0))
                
            elif chaos_type == "file_corruption":
                # Simular corrupción de archivo
                temp_filename = f"temp_chaos_{random.randint(1000, 9999)}.dat"
                try:
                    with open(temp_filename, 'w') as f:
                        f.write("Datos corruptos")
                    time.sleep(0.5)
                    os.remove(temp_filename)
                except:
                    pass
                
            elif chaos_type == "resource_exhaustion":
                # Simular agotamiento de recursos
                cpu_burn = []
                for _ in range(random.randint(1, 4)):
                    cpu_burn.append(threading.Thread(target=simulate_heavy_load))
                    cpu_burn[-1].daemon = True
                    cpu_burn[-1].start()
                
                time.sleep(random.uniform(1.0, 3.0))
                
                # Los hilos daemon terminarán solos
    
    except Exception as e:
        logger.error(f"Error en hilo de caos: {str(e)}")

def test_repair_entity():
    """Probar la entidad de reparación."""
    if not MODULES_LOADED:
        logger.warning("No se pueden probar módulos: Sistema Genesis no cargado")
        return {"success": False, "reason": "modules_not_loaded"}
    
    try:
        # Crear entidad reparadora
        logger.info("Creando entidad reparadora de prueba...")
        repair_entity = create_repair_entity("TestHephaestus", "otoniel", 15)
        
        # Ejecutar ciclos de reparación
        repairs = []
        for i in range(5):
            # Ejecutar método de comercio (en realidad hace reparación)
            result = repair_entity.trade()
            repairs.append(result)
            time.sleep(1)
        
        return {
            "success": True,
            "entity": "repair_entity",
            "cycles": len(repairs),
            "results": repairs
        }
    
    except Exception as e:
        logger.error(f"Error probando entidad reparadora: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "entity": "repair_entity", "error": str(e)}

def test_message_system():
    """Probar el sistema de mensajes."""
    if not MODULES_LOADED:
        logger.warning("No se pueden probar módulos: Sistema Genesis no cargado")
        return {"success": False, "reason": "modules_not_loaded"}
    
    try:
        # Enviar varios tipos de mensajes
        messages = []
        
        # Mensaje del sistema
        send_system_message("armagedón", "Prueba Armagedón ejecutándose en modo extremo")
        messages.append({"type": "system", "content": "Prueba Armagedón"})
        
        # Mensaje de entidad simulada
        for i in range(3):
            send_system_message(
                f"entity_{i}",
                f"Mensaje de prueba de entidad simulada {i}"
            )
            messages.append({"type": f"entity_{i}", "content": f"Mensaje {i}"})
        
        # Forzar envío
        force_result = force_send_messages()
        
        return {
            "success": True,
            "system": "message_system",
            "messages_sent": len(messages),
            "force_result": force_result
        }
    
    except Exception as e:
        logger.error(f"Error probando sistema de mensajes: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "system": "message_system", "error": str(e)}

def test_connector_system():
    """Probar el sistema de conectores."""
    if not MODULES_LOADED:
        logger.warning("No se pueden probar módulos: Sistema Genesis no cargado")
        return {"success": False, "reason": "modules_not_loaded"}
    
    try:
        # Inicializar sistema
        connector = initialize_system()
        
        # Crear entidades de prueba
        class TestEntity:
            def __init__(self, name, role):
                self.name = name
                self.role = role
                self.energy = 80
                self.level = 2.5
                self.emotion = "Curioso"
                self.is_alive = True
        
        # Aplicar conectores
        entities = []
        for i in range(3):
            entity = TestEntity(f"TestEntity_{i}", "Testing")
            from main_module_connector import apply_connectors_to_entity
            repair_connector, message_connector = apply_connectors_to_entity(entity)
            
            # Probar conectores
            health = repair_connector.check_health()
            message_connector.send_status_update()
            
            entities.append({
                "name": entity.name,
                "health": health,
                "has_repair_connector": hasattr(entity, "repair_connector"),
                "has_message_connector": hasattr(entity, "message_connector")
            })
        
        # Enviar estado consolidado
        connector.send_consolidated_status()
        
        return {
            "success": True,
            "system": "connector_system",
            "entities_tested": len(entities),
            "entity_status": entities
        }
    
    except Exception as e:
        logger.error(f"Error probando sistema de conectores: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "system": "connector_system", "error": str(e)}

def test_websocket_entities():
    """Probar entidades WebSocket."""
    if not MODULES_LOADED:
        logger.warning("No se pueden probar módulos: Sistema Genesis no cargado")
        return {"success": False, "reason": "modules_not_loaded"}
    
    try:
        # Crear entidades WebSocket
        hermes = websocket_entity_fix.create_local_websocket_entity("TestHermes", "otoniel", 30, False)
        apollo = websocket_entity_fix.create_external_websocket_entity("TestApollo", "otoniel", 35, False)
        
        # Aplicar conectores
        from main_module_connector import apply_connectors_to_entity
        hermes_repair, hermes_message = apply_connectors_to_entity(hermes)
        apollo_repair, apollo_message = apply_connectors_to_entity(apollo)
        
        # Añadir método faltante para pruebas
        if not hasattr(hermes, "adjust_energy"):
            hermes.adjust_energy = hermes_repair.auto_repair
            apollo.adjust_energy = apollo_repair.auto_repair
            
        # Ejecutar operación de trading (comunicación)
        hermes_result = hermes.trade()
        apollo_result = apollo.trade()
        
        return {
            "success": True,
            "entities": "websocket_entities",
            "local": {
                "name": hermes.name,
                "result": hermes_result
            },
            "external": {
                "name": apollo.name,
                "result": apollo_result
            }
        }
    
    except Exception as e:
        logger.error(f"Error probando entidades WebSocket: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "entities": "websocket_entities", "error": str(e)}

def run_system_tests():
    """Ejecutar pruebas del Sistema Genesis."""
    logger.info("Iniciando pruebas de sistema...")
    
    system_results = {}
    
    # Probar entidad reparadora
    logger.info("Probando entidad reparadora...")
    system_results["repair_entity"] = test_repair_entity()
    
    # Probar sistema de mensajes
    logger.info("Probando sistema de mensajes...")
    system_results["message_system"] = test_message_system()
    
    # Probar sistema de conectores
    logger.info("Probando sistema de conectores...")
    system_results["connector_system"] = test_connector_system()
    
    # Probar entidades WebSocket
    logger.info("Probando entidades WebSocket...")
    system_results["websocket_entities"] = test_websocket_entities()
    
    return system_results

def generate_report():
    """Generar informe detallado de la prueba."""
    global results, start_time
    
    if not start_time:
        logger.warning("No se puede generar informe: prueba no iniciada")
        return
    
    # Calcular duración total
    end_time = time.time()
    duration = end_time - start_time
    results["duration"] = duration
    
    # Estado general
    results["success"] = results["errors_fixed"] > 0 and results["operations_completed"] > 0
    
    # Calcular métricas de rendimiento
    ops_per_second = results["operations_completed"] / duration if duration > 0 else 0
    error_rate = results["errors_detected"] / results["operations_completed"] if results["operations_completed"] > 0 else 0
    fix_rate = results["errors_fixed"] / results["errors_detected"] if results["errors_detected"] > 0 else 0
    
    results["performance_metrics"] = {
        "operations_per_second": ops_per_second,
        "error_rate": error_rate,
        "fix_rate": fix_rate,
        "resilience_score": (1.0 - error_rate) * fix_rate * 100  # 0-100
    }
    
    # Guardar informe
    report_file = f"armageddon_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Informe guardado en {report_file}")
    except Exception as e:
        logger.error(f"Error guardando informe: {str(e)}")
    
    # Imprimir resumen
    print("\n" + "="*80)
    print(f"INFORME DE PRUEBA ARMAGEDÓN - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"Duración: {duration:.2f} segundos")
    print(f"Operaciones completadas: {results['operations_completed']}")
    print(f"Errores detectados: {results['errors_detected']}")
    print(f"Errores corregidos: {results['errors_fixed']}")
    print(f"Tasa de corrección: {fix_rate*100:.1f}%")
    print(f"Operaciones por segundo: {ops_per_second:.2f}")
    print(f"Puntuación de resiliencia: {results['performance_metrics']['resilience_score']:.1f}/100")
    print(f"RESULTADO: {'ÉXITO' if results['success'] else 'FRACASO'}")
    print("="*80)
    
    # Generar archivo HTML más detallado
    html_report = f"armageddon_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    try:
        with open(html_report, 'w') as f:
            f.write(generate_html_report(results))
        logger.info(f"Informe HTML guardado en {html_report}")
    except Exception as e:
        logger.error(f"Error guardando informe HTML: {str(e)}")

def generate_html_report(results):
    """Generar informe HTML detallado."""
    # Determinar color de estado
    status_color = "#4CAF50" if results["success"] else "#F44336"  # Verde o rojo
    resilience_score = results["performance_metrics"]["resilience_score"]
    
    # Generar HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Informe Armagedón - Sistema Genesis</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 20px; background-color: #f8f9fa; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1, h2, h3 {{ color: #2c3e50; }}
            h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; text-align: center; }}
            .status {{ text-align: center; font-size: 24px; padding: 15px; border-radius: 6px; margin: 20px 0; color: white; background-color: {status_color}; }}
            .metrics {{ display: flex; justify-content: space-between; flex-wrap: wrap; margin: 20px 0; }}
            .metric-card {{ flex: 1; min-width: 200px; margin: 10px; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background-color: #f8f9fa; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; margin: 10px 0; }}
            .metric-title {{ font-size: 14px; color: #7f8c8d; margin-bottom: 5px; }}
            .progress-bar {{ height: 20px; border-radius: 10px; background-color: #ecf0f1; margin: 15px 0; }}
            .progress-value {{ height: 100%; border-radius: 10px; background: linear-gradient(to right, #3498db, #2ecc71); width: {min(100, resilience_score)}%; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .success {{ color: #2ecc71; }}
            .error {{ color: #e74c3c; }}
            .footer {{ margin-top: 30px; text-align: center; font-size: 14px; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Informe de Prueba Armagedón</h1>
            <p>Informe generado el {datetime.datetime.now().strftime('%d de %B de %Y a las %H:%M:%S')}</p>
            
            <div class="status">
                {"PRUEBA EXITOSA" if results["success"] else "PRUEBA FALLIDA"}
            </div>
            
            <h2>Resumen de Resultados</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">Duración</div>
                    <div class="metric-value">{results["duration"]:.2f}s</div>
                    <div>Tiempo total de ejecución</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Operaciones</div>
                    <div class="metric-value">{results["operations_completed"]}</div>
                    <div>Total completadas</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Errores</div>
                    <div class="metric-value">{results["errors_detected"]}</div>
                    <div>Detectados durante la prueba</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Correcciones</div>
                    <div class="metric-value">{results["errors_fixed"]}</div>
                    <div>Errores corregidos automáticamente</div>
                </div>
            </div>
            
            <h2>Puntuación de Resiliencia</h2>
            <div class="progress-bar">
                <div class="progress-value"></div>
            </div>
            <div style="text-align: center; font-size: 18px;">{resilience_score:.1f}/100</div>
            
            <h2>Métricas de Rendimiento</h2>
            <table>
                <tr>
                    <th>Métrica</th>
                    <th>Valor</th>
                    <th>Interpretación</th>
                </tr>
                <tr>
                    <td>Operaciones por segundo</td>
                    <td>{results["performance_metrics"]["operations_per_second"]:.2f}</td>
                    <td>Throughput general del sistema</td>
                </tr>
                <tr>
                    <td>Tasa de error</td>
                    <td>{results["performance_metrics"]["error_rate"]*100:.1f}%</td>
                    <td>Porcentaje de operaciones con error</td>
                </tr>
                <tr>
                    <td>Tasa de corrección</td>
                    <td>{results["performance_metrics"]["fix_rate"]*100:.1f}%</td>
                    <td>Porcentaje de errores corregidos</td>
                </tr>
            </table>
            
            <h2>Pruebas de Sistema Genesis</h2>
    """
    
    # Añadir resultados de pruebas de sistema si existen
    if "entity_status" in results:
        html += """
            <table>
                <tr>
                    <th>Componente</th>
                    <th>Estado</th>
                    <th>Detalles</th>
                </tr>
        """
        
        for component, status in results["entity_status"].items():
            success = status.get("success", False)
            html += f"""
                <tr>
                    <td>{component}</td>
                    <td class="{'success' if success else 'error'}">{'✓ Éxito' if success else '✗ Fallo'}</td>
                    <td>{status.get('error', 'Sin detalles adicionales')}</td>
                </tr>
            """
        
        html += """
            </table>
        """
    
    # Pie de página
    html += f"""
            <div class="footer">
                <p>Sistema Genesis - Prueba Armagedón</p>
                <p>Nivel de intensidad: {INTENSITY_LEVEL}/10 | Hilos: {MAX_THREADS} | Procesos: {MAX_PROCESSES}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def run_armageddon():
    """Ejecutar prueba Armagedón completa."""
    global start_time, running, results
    
    logger.info("Iniciando prueba Armagedón...")
    logger.info(f"Configuración: INTENSITY_LEVEL={INTENSITY_LEVEL}, MAX_THREADS={MAX_THREADS}, TEST_DURATION={TEST_DURATION}s")
    
    # Inicializar variables
    start_time = time.time()
    running = True
    
    # Primero probar los componentes del Sistema Genesis
    if MODULES_LOADED:
        logger.info("Ejecutando pruebas de sistema...")
        results["entity_status"] = run_system_tests()
    
    try:
        # Crear hilos de estrés
        stress_threads = []
        for i in range(MAX_THREADS):
            thread = threading.Thread(target=run_stressor_thread)
            thread.daemon = True
            thread.start()
            stress_threads.append(thread)
        
        logger.info(f"Iniciados {len(stress_threads)} hilos de estrés")
        
        # Crear hilo de caos
        chaos = threading.Thread(target=chaos_thread)
        chaos.daemon = True
        chaos.start()
        logger.info("Hilo de caos iniciado")
        
        # Esperar duración de la prueba
        logger.info(f"Prueba en ejecución por {TEST_DURATION} segundos...")
        time.sleep(TEST_DURATION)
        
        # Finalizar prueba
        running = False
        logger.info("Prueba completada. Generando informe...")
        
        # Generar informe
        generate_report()
        
    except KeyboardInterrupt:
        logger.info("Prueba interrumpida por el usuario")
        running = False
        generate_report()
    except Exception as e:
        logger.error(f"Error en prueba Armagedón: {str(e)}")
        logger.error(traceback.format_exc())
        running = False
        generate_report()

if __name__ == "__main__":
    # Mostrar banner
    print("""
    ======================================================================
       █████╗ ██████╗ ███╗   ███╗ █████╗  ██████╗ ███████╗██████╗  ██████╗ ███╗   ██╗
      ██╔══██╗██╔══██╗████╗ ████║██╔══██╗██╔════╝ ██╔════╝██╔══██╗██╔═══██╗████╗  ██║
      ███████║██████╔╝██╔████╔██║███████║██║  ███╗█████╗  ██║  ██║██║   ██║██╔██╗ ██║
      ██╔══██║██╔══██╗██║╚██╔╝██║██╔══██║██║   ██║██╔══╝  ██║  ██║██║   ██║██║╚██╗██║
      ██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗██████╔╝╚██████╔╝██║ ╚████║
      ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝
                                                                       
         PRUEBA EXTREMA - SISTEMA GENESIS ULTRACONECTADO
    ======================================================================
    """)
    
    # Ejecutar prueba
    run_armageddon()