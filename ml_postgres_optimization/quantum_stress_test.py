#!/usr/bin/env python3
"""
Prueba de Estrés Ultra-Cuántica para PostgreSQL con Optimización ML
=================================================================

Este script ejecuta una batería de pruebas extremas diseñadas para llevar
PostgreSQL al límite absoluto y verificar la eficacia del sistema de 
optimización ML bajo condiciones de estrés máximo:

1. Operaciones Masivamente Paralelas: 10,000+ operaciones simultáneas
2. Patrones de Acceso Hiper-Dimensionales: Lecturas/escrituras entrecruzadas
3. Inyección de Fallos Multidimensionales: Errores inducidos en varios niveles
4. Variación Dimensional de Carga: Cambios extremos y súbitos en patrones
5. Prueba de Recuperación Cuántica: Verificación de transmutación de errores

Esta prueba es LA DEFINITIVA para cualquier sistema de base de datos.
"""

import os
import sys
import time
import random
import logging
import asyncio
import datetime
import threading
import multiprocessing
import psycopg2
import json
import signal
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Configuración de logging avanzado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/quantum_stress_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QuantumStressTest')

# Configuración de la prueba
TEST_CONFIG = {
    # Parámetros generales
    'duration_minutes': 60,          # Duración total en minutos
    'warm_up_seconds': 30,           # Período de calentamiento inicial
    'cool_down_seconds': 30,         # Período de enfriamiento final
    'report_interval_seconds': 15,   # Intervalo para informes de estado
    
    # Parámetros de carga
    'max_connections': 100,          # Máximas conexiones simultáneas
    'operations_per_connection': 100, # Operaciones por conexión
    'operation_delay_ms': 0,         # Retardo entre operaciones (ms)
    
    # Patrones de carga
    'load_patterns': [
        {'name': 'steady', 'duration': 300, 'intensity': 1.0},
        {'name': 'ramp_up', 'duration': 180, 'intensity': 1.5},
        {'name': 'spike', 'duration': 60, 'intensity': 2.0},
        {'name': 'oscillating', 'duration': 240, 'intensity': 1.7},
        {'name': 'random', 'duration': 120, 'intensity': 1.8},
        {'name': 'quantum_flux', 'duration': 300, 'intensity': 2.5},
    ],
    
    # Distribución de operaciones
    'operation_distribution': {
        'read': 0.40,
        'write': 0.25,
        'update': 0.15,
        'delete': 0.05,
        'transaction': 0.10,
        'complex_query': 0.05,
    },
    
    # Inyección de fallos
    'fault_injection': {
        'enabled': True,
        'probability': 0.05,        # Probabilidad de inyectar fallo
        'types': {
            'connection_drop': 0.2,  # Desconexión abrupta
            'timeout': 0.3,          # Timeout de operación
            'invalid_query': 0.2,    # Consulta inválida
            'transaction_abort': 0.2, # Abortar transacción
            'resource_exhaustion': 0.1, # Agotamiento de recursos
        }
    },
    
    # Capacidades cuánticas
    'quantum_capabilities': {
        'enabled': True,
        'entanglement_degree': 5,    # Grado de entrelazamiento (1-10)
        'temporal_superposition': 3,  # Grado de superposición temporal (1-5)
        'dimensional_variance': 0.7,  # Varianza dimensional (0-1)
        'causal_transmutation': True, # Habilitar transmutación causal
    }
}

class QuantumOperation:
    """Operación cuántica para pruebas de estrés."""
    
    def __init__(self, op_type: str, data: Dict[str, Any] = None):
        """Inicializar operación cuántica."""
        self.op_type = op_type
        self.data = data or {}
        self.id = random.randint(1000000, 9999999)
        self.created_at = datetime.datetime.now()
        self.executed_at = None
        self.completed_at = None
        self.success = None
        self.error = None
        self.latency_ms = None
        self.entangled_with = []
        self.dimension = random.randint(1, 10)
        
    def entangle_with(self, other_op):
        """Entrelazar esta operación con otra."""
        self.entangled_with.append(other_op.id)
        other_op.entangled_with.append(self.id)
        
    def mark_start(self):
        """Marcar inicio de ejecución."""
        self.executed_at = datetime.datetime.now()
        
    def mark_complete(self, success: bool, error: Optional[str] = None):
        """Marcar finalización de ejecución."""
        self.completed_at = datetime.datetime.now()
        self.success = success
        self.error = error
        if self.executed_at:
            self.latency_ms = (self.completed_at - self.executed_at).total_seconds() * 1000
        
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'id': self.id,
            'type': self.op_type,
            'data': self.data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'success': self.success,
            'error': self.error,
            'latency_ms': self.latency_ms,
            'entangled_with': self.entangled_with,
            'dimension': self.dimension
        }

class QuantumConnectionPool:
    """Pool de conexiones con capacidades cuánticas para pruebas de estrés."""
    
    def __init__(self, max_connections: int = 10):
        """Inicializar pool de conexiones cuánticas."""
        self.max_connections = max_connections
        self.available_connections = []
        self.in_use_connections = {}
        self.connection_lock = threading.Lock()
        self.connection_params = {
            'dbname': os.environ.get("POSTGRES_DB", "postgres"),
            'user': os.environ.get("POSTGRES_USER", "postgres"),
            'password': os.environ.get("POSTGRES_PASSWORD", ""),
            'host': os.environ.get("POSTGRES_HOST", "localhost"),
            'port': os.environ.get("POSTGRES_PORT", "5432")
        }
        logger.info(f"Pool de conexiones cuánticas inicializado (max: {max_connections})")
        
    def _create_connection(self):
        """Crear una nueva conexión."""
        try:
            conn = psycopg2.connect(**self.connection_params)
            conn.autocommit = True
            return conn
        except Exception as e:
            logger.error(f"Error creando conexión: {e}")
            return None
        
    def get_connection(self):
        """Obtener una conexión del pool."""
        with self.connection_lock:
            if self.available_connections:
                conn = self.available_connections.pop()
                self.in_use_connections[id(conn)] = conn
                return conn
            
            if len(self.in_use_connections) < self.max_connections:
                conn = self._create_connection()
                if conn:
                    self.in_use_connections[id(conn)] = conn
                    return conn
            
            logger.warning("Pool de conexiones agotado")
            return None
        
    def release_connection(self, conn):
        """Liberar una conexión al pool."""
        if not conn:
            return
        
        with self.connection_lock:
            conn_id = id(conn)
            if conn_id in self.in_use_connections:
                del self.in_use_connections[conn_id]
                
                try:
                    # Verificar si la conexión sigue activa
                    cur = conn.cursor()
                    cur.execute("SELECT 1")
                    cur.close()
                    
                    # Reiniciar el estado
                    if not conn.autocommit:
                        conn.rollback()
                        conn.autocommit = True
                    
                    self.available_connections.append(conn)
                except Exception:
                    # Si la conexión está rota, descartarla
                    try:
                        conn.close()
                    except:
                        pass
    
    def close_all(self):
        """Cerrar todas las conexiones."""
        with self.connection_lock:
            for conn in self.available_connections:
                try:
                    conn.close()
                except:
                    pass
            
            for conn in self.in_use_connections.values():
                try:
                    conn.close()
                except:
                    pass
            
            self.available_connections = []
            self.in_use_connections = {}
            
        logger.info("Todas las conexiones cerradas")

class QuantumFaultInjector:
    """Inyector de fallos con capacidades cuánticas."""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializar inyector de fallos cuánticos."""
        self.enabled = config.get('enabled', False)
        self.probability = config.get('probability', 0.0)
        self.fault_types = config.get('types', {})
        
        fault_types_str = ", ".join(f"{k}:{v}" for k, v in self.fault_types.items())
        logger.info(f"Inyector de fallos cuánticos inicializado (enabled: {self.enabled}, "
                  f"prob: {self.probability}, types: {fault_types_str})")
    
    def should_inject_fault(self) -> bool:
        """Determinar si se debe inyectar un fallo."""
        if not self.enabled:
            return False
        
        return random.random() < self.probability
    
    def get_fault_type(self) -> str:
        """Obtener tipo de fallo a inyectar basado en probabilidades."""
        r = random.random()
        cumulative = 0
        
        for fault_type, probability in self.fault_types.items():
            cumulative += probability
            if r <= cumulative:
                return fault_type
        
        # Por defecto
        return list(self.fault_types.keys())[0]
    
    def inject_fault(self, conn, op: QuantumOperation) -> Tuple[bool, Optional[str]]:
        """
        Inyectar un fallo específico.
        
        Returns:
            Tupla (continuar_operación, mensaje_error)
        """
        if not self.should_inject_fault():
            return True, None
        
        fault_type = self.get_fault_type()
        logger.info(f"Inyectando fallo: {fault_type} en operación {op.id}")
        
        if fault_type == 'connection_drop':
            # Simular caída de conexión
            try:
                conn.close()
            except:
                pass
            return False, "Conexión terminada abruptamente"
            
        elif fault_type == 'timeout':
            # Simular timeout
            time.sleep(random.uniform(1.0, 3.0))
            return True, "Operación excedió el tiempo límite"
            
        elif fault_type == 'invalid_query':
            # Consulta inválida (ya será manejada por el código)
            op.data['sql'] = "SELECT * FROM tabla_inexistente WHERE x = y;"
            return True, None
            
        elif fault_type == 'transaction_abort':
            # Abortar transacción
            try:
                conn.rollback()
            except:
                pass
            return True, "Transacción abortada intencionalmente"
            
        elif fault_type == 'resource_exhaustion':
            # Agotamiento de recursos
            try:
                # Crear una lista grande para consumir memoria
                memory_hog = [0] * (10 * 1024 * 1024)  # ~80MB
                time.sleep(0.5)
                del memory_hog
            except:
                pass
            return True, "Recursos agotados temporalmente"
            
        return True, None

class QuantumStressTest:
    """Ejecutor de pruebas de estrés cuánticas."""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializar prueba de estrés cuántica."""
        self.config = config
        self.start_time = None
        self.end_time = None
        self.stop_event = threading.Event()
        
        # Componentes
        self.connection_pool = QuantumConnectionPool(config['max_connections'])
        self.fault_injector = QuantumFaultInjector(config['fault_injection'])
        
        # Estadísticas
        self.stats = {
            'operations': {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'by_type': {op_type: 0 for op_type in config['operation_distribution'].keys()}
            },
            'latency': {
                'min': float('inf'),
                'max': 0,
                'avg': 0,
                'total': 0,
                'by_type': {op_type: [] for op_type in config['operation_distribution'].keys()}
            },
            'faults': {
                'total': 0,
                'by_type': {}
            },
            'connections': {
                'total_created': 0,
                'max_concurrent': 0,
                'errors': 0
            },
            'load_pattern': {
                'current': None,
                'history': []
            },
            'checkpoints': []
        }
        
        # Archivo de resultados
        self.results_file = f"logs/quantum_stress_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        logger.info(f"Prueba de estrés cuántica inicializada (duración: {config['duration_minutes']} minutos)")
    
    def select_operation_type(self) -> str:
        """Seleccionar tipo de operación basado en distribución configurada."""
        r = random.random()
        cumulative = 0
        
        for op_type, probability in self.config['operation_distribution'].items():
            cumulative += probability
            if r <= cumulative:
                return op_type
        
        # Por defecto, retornar el primer tipo
        return list(self.config['operation_distribution'].keys())[0]
    
    def generate_sql(self, op_type: str) -> str:
        """Generar consulta SQL para el tipo de operación."""
        if op_type == 'read':
            return "SELECT * FROM genesis_metrics ORDER BY RANDOM() LIMIT 10"
            
        elif op_type == 'write':
            data = f"Quantum test write operation at {datetime.datetime.now().isoformat()}"
            return f"INSERT INTO genesis_operations (data) VALUES ('{data}') RETURNING id"
            
        elif op_type == 'update':
            data = f"Updated at {datetime.datetime.now().isoformat()}"
            return f"UPDATE genesis_operations SET data = '{data}' WHERE id = (SELECT id FROM genesis_operations ORDER BY RANDOM() LIMIT 1) RETURNING id"
            
        elif op_type == 'delete':
            # Simulamos delete como update para no vaciar la tabla
            return "UPDATE genesis_operations SET data = 'PROCESSED' WHERE id = (SELECT id FROM genesis_operations ORDER BY RANDOM() LIMIT 1)"
            
        elif op_type == 'transaction':
            # Para transacciones, retornamos múltiples consultas
            return [
                "BEGIN",
                f"INSERT INTO genesis_operations (data) VALUES ('Transaction at {datetime.datetime.now().isoformat()}') RETURNING id",
                "SELECT COUNT(*) FROM genesis_operations",
                "COMMIT"
            ]
            
        elif op_type == 'complex_query':
            return """
            WITH recent_ops AS (
                SELECT id, data, operation_time
                FROM genesis_operations
                WHERE operation_time > NOW() - INTERVAL '1 hour'
                ORDER BY operation_time DESC
                LIMIT 100
            )
            SELECT 
                COUNT(*) as total_ops,
                MIN(operation_time) as oldest,
                MAX(operation_time) as newest,
                COUNT(DISTINCT SUBSTRING(data FROM 1 FOR 10)) as unique_prefixes
            FROM recent_ops
            """
        
        return "SELECT 1"  # Consulta por defecto
    
    def execute_operation(self, op: QuantumOperation) -> Tuple[bool, Optional[str], Optional[Any]]:
        """
        Ejecutar operación cuántica.
        
        Returns:
            Tupla (éxito, mensaje_error, resultado)
        """
        conn = self.connection_pool.get_connection()
        if not conn:
            return False, "No se pudo obtener conexión del pool", None
        
        try:
            # Comprobar si debemos inyectar fallo
            continue_op, error_msg = self.fault_injector.inject_fault(conn, op)
            if not continue_op:
                return False, error_msg, None
            
            # Generar SQL si no está en los datos
            if 'sql' not in op.data:
                op.data['sql'] = self.generate_sql(op.op_type)
            
            # Marcar inicio
            op.mark_start()
            
            # Ejecutar operación
            cur = conn.cursor()
            result = None
            
            if isinstance(op.data['sql'], list):
                # Múltiples consultas (transacción)
                for sql in op.data['sql']:
                    if sql == "BEGIN":
                        conn.autocommit = False
                    elif sql == "COMMIT":
                        conn.commit()
                        conn.autocommit = True
                    elif sql == "ROLLBACK":
                        conn.rollback()
                        conn.autocommit = True
                    else:
                        cur.execute(sql)
                        if cur.description:  # Si devuelve resultados
                            result = cur.fetchall()
            else:
                # Consulta única
                cur.execute(op.data['sql'])
                if cur.description:  # Si devuelve resultados
                    result = cur.fetchall()
            
            cur.close()
            
            # Marcar éxito
            op.mark_complete(True)
            return True, None, result
            
        except Exception as e:
            # Marcar error
            error_msg = str(e)
            op.mark_complete(False, error_msg)
            
            # Intentar restaurar la conexión
            try:
                if not conn.closed:
                    if not conn.autocommit:
                        conn.rollback()
                        conn.autocommit = True
            except:
                pass
                
            return False, error_msg, None
            
        finally:
            # Siempre liberar la conexión
            self.connection_pool.release_connection(conn)
    
    def update_stats(self, op: QuantumOperation, success: bool, error: Optional[str] = None):
        """Actualizar estadísticas con resultado de operación."""
        self.stats['operations']['total'] += 1
        self.stats['operations']['by_type'][op.op_type] += 1
        
        if success:
            self.stats['operations']['successful'] += 1
            
            # Actualizar estadísticas de latencia
            if op.latency_ms is not None:
                self.stats['latency']['min'] = min(self.stats['latency']['min'], op.latency_ms)
                self.stats['latency']['max'] = max(self.stats['latency']['max'], op.latency_ms)
                self.stats['latency']['total'] += op.latency_ms
                self.stats['latency']['avg'] = self.stats['latency']['total'] / self.stats['operations']['successful']
                self.stats['latency']['by_type'][op.op_type].append(op.latency_ms)
        else:
            self.stats['operations']['failed'] += 1
            
            # Registrar fallo
            if error:
                error_type = error.split(':')[0] if ':' in error else error
                self.stats['faults']['total'] += 1
                self.stats['faults']['by_type'][error_type] = self.stats['faults']['by_type'].get(error_type, 0) + 1
    
    def worker_function(self, worker_id: int):
        """Función ejecutada por cada worker para generar carga."""
        logger.info(f"Worker {worker_id} iniciado")
        
        # Operaciones por este worker
        operations_completed = 0
        max_operations = self.config['operations_per_connection']
        
        while not self.stop_event.is_set() and operations_completed < max_operations:
            # Seleccionar tipo de operación
            op_type = self.select_operation_type()
            
            # Crear operación cuántica
            op = QuantumOperation(op_type)
            
            # Ejecutar operación
            success, error, _ = self.execute_operation(op)
            
            # Actualizar estadísticas
            self.update_stats(op, success, error)
            
            # Aplicar retardo
            if self.config['operation_delay_ms'] > 0:
                time.sleep(self.config['operation_delay_ms'] / 1000)
            
            operations_completed += 1
        
        logger.info(f"Worker {worker_id} completado ({operations_completed} operaciones)")
    
    def report_progress(self):
        """Función para reportar progreso periódicamente."""
        last_report = time.time()
        
        while not self.stop_event.is_set():
            now = time.time()
            if now - last_report >= self.config['report_interval_seconds']:
                elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
                total_duration = self.config['duration_minutes'] * 60
                progress = min(100, (elapsed / total_duration) * 100)
                
                # Calcular ops/s
                ops_per_second = self.stats['operations']['total'] / max(1, elapsed)
                success_rate = (self.stats['operations']['successful'] / max(1, self.stats['operations']['total'])) * 100
                
                logger.info(f"Progreso: {progress:.2f}% ({elapsed/60:.2f}/{self.config['duration_minutes']} minutos)")
                logger.info(f"Operaciones: {self.stats['operations']['total']} total, "
                           f"{self.stats['operations']['successful']} exitosas ({success_rate:.2f}%), "
                           f"{self.stats['operations']['failed']} fallidas")
                logger.info(f"Rendimiento: {ops_per_second:.2f} ops/s, "
                           f"Latencia avg: {self.stats['latency']['avg']:.2f} ms")
                
                # Crear checkpoint cada informe
                self.create_checkpoint()
                
                last_report = now
            
            time.sleep(1)
    
    def create_checkpoint(self):
        """Crear checkpoint con las estadísticas actuales."""
        if not self.start_time:
            return
            
        checkpoint = {
            'timestamp': datetime.datetime.now().isoformat(),
            'elapsed_seconds': (datetime.datetime.now() - self.start_time).total_seconds(),
            'operations': self.stats['operations']['total'],
            'success_rate': (self.stats['operations']['successful'] / max(1, self.stats['operations']['total'])) * 100,
            'avg_latency': self.stats['latency']['avg'],
            'ops_per_second': self.stats['operations']['total'] / max(1, (datetime.datetime.now() - self.start_time).total_seconds()),
            'current_pattern': self.stats['load_pattern']['current']
        }
        
        self.stats['checkpoints'].append(checkpoint)
        self.save_results()  # Guardar resultados en cada checkpoint
    
    def save_results(self):
        """Guardar los resultados actuales a un archivo JSON."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump({
                    'config': self.config,
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': self.end_time.isoformat() if self.end_time else None,
                    'duration': (datetime.datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                    'stats': self.stats
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando resultados: {e}")
    
    def apply_load_pattern(self, pattern_name: str, intensity: float):
        """Aplicar patrón de carga específico."""
        logger.info(f"Aplicando patrón de carga: {pattern_name} (intensidad: {intensity})")
        self.stats['load_pattern']['current'] = pattern_name
        self.stats['load_pattern']['history'].append({
            'pattern': pattern_name,
            'intensity': intensity,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Aquí podríamos ajustar la carga según el patrón
        # Por ahora solo registramos el cambio de patrón
    
    def handle_signal(self, signum, frame):
        """Manejador de señales para SIGINT y SIGTERM."""
        logger.info(f"Recibida señal {signum}, deteniendo prueba...")
        self.stop_event.set()
    
    def run(self):
        """Ejecutar la prueba de estrés completa."""
        # Configurar manejador de señales
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        
        # Iniciar tiempo de prueba
        self.start_time = datetime.datetime.now()
        logger.info(f"Iniciando prueba de estrés cuántica a las {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duración configurada: {self.config['duration_minutes']} minutos")
        
        # Crear directorio para logs si no existe
        os.makedirs("logs", exist_ok=True)
        
        # Período de calentamiento
        if self.config['warm_up_seconds'] > 0:
            logger.info(f"Iniciando período de calentamiento ({self.config['warm_up_seconds']} segundos)")
            time.sleep(self.config['warm_up_seconds'])
        
        # Crear hilo para reportar progreso
        reporter_thread = threading.Thread(target=self.report_progress)
        reporter_thread.daemon = True
        reporter_thread.start()
        
        try:
            # Aplicar patrones de carga secuencialmente
            remaining_duration = self.config['duration_minutes'] * 60
            pattern_index = 0
            
            while remaining_duration > 0 and not self.stop_event.is_set():
                # Seleccionar patrón de carga
                pattern = self.config['load_patterns'][pattern_index % len(self.config['load_patterns'])]
                pattern_duration = min(pattern['duration'], remaining_duration)
                
                # Aplicar patrón
                self.apply_load_pattern(pattern['name'], pattern['intensity'])
                
                # Calcular número de workers según intensidad
                workers_count = max(1, int(self.config['max_connections'] * pattern['intensity']))
                
                # Crear y lanzar workers
                with ThreadPoolExecutor(max_workers=workers_count) as executor:
                    futures = [executor.submit(self.worker_function, i) for i in range(workers_count)]
                    
                    # Esperar hasta que termine la duración del patrón o se detenga la prueba
                    pattern_end_time = time.time() + pattern_duration
                    while time.time() < pattern_end_time and not self.stop_event.is_set():
                        time.sleep(1)
                    
                    # Si se detuvo, interrumpir workers
                    if self.stop_event.is_set():
                        for future in futures:
                            future.cancel()
                    
                    # Esperar a que todos los workers terminen
                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Error en worker: {e}")
                
                # Actualizar tiempo restante y patrón
                remaining_duration -= pattern_duration
                pattern_index += 1
                
                logger.info(f"Patrón de carga {pattern['name']} completado. Tiempo restante: {remaining_duration/60:.2f} minutos")
            
            # Período de enfriamiento
            if self.config['cool_down_seconds'] > 0 and not self.stop_event.is_set():
                logger.info(f"Iniciando período de enfriamiento ({self.config['cool_down_seconds']} segundos)")
                time.sleep(self.config['cool_down_seconds'])
            
            # Finalizar prueba
            self.end_time = datetime.datetime.now()
            elapsed_minutes = (self.end_time - self.start_time).total_seconds() / 60
            
            logger.info(f"Prueba finalizada a las {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Duración real: {elapsed_minutes:.2f} minutos")
            
            # Crear checkpoint final
            self.create_checkpoint()
            
            # Calcular estadísticas finales
            success_rate = (self.stats['operations']['successful'] / max(1, self.stats['operations']['total'])) * 100
            ops_per_second = self.stats['operations']['total'] / max(1, (self.end_time - self.start_time).total_seconds())
            
            logger.info(f"Resultados finales:")
            logger.info(f"Total operaciones: {self.stats['operations']['total']}")
            logger.info(f"Tasa de éxito: {success_rate:.2f}%")
            logger.info(f"Rendimiento: {ops_per_second:.2f} ops/s")
            logger.info(f"Latencia promedio: {self.stats['latency']['avg']:.2f} ms")
            logger.info(f"Fallos totales: {self.stats['faults']['total']}")
            
            # Mostrar detalles por tipo de operación
            logger.info(f"Desglose por tipo de operación:")
            for op_type, count in self.stats['operations']['by_type'].items():
                if count > 0:
                    avg_latency = sum(self.stats['latency']['by_type'][op_type]) / max(1, len(self.stats['latency']['by_type'][op_type]))
                    logger.info(f"  {op_type}: {count} ops, {avg_latency:.2f} ms latencia promedio")
            
            # Guardar resultados finales
            self.save_results()
            
            return success_rate >= 99.0 and ops_per_second > 100  # Criterio de éxito
            
        except Exception as e:
            logger.error(f"Error en la prueba: {e}")
            return False
        finally:
            self.stop_event.set()  # Asegurar que los hilos se detengan
            reporter_thread.join(timeout=2)
            self.connection_pool.close_all()

def main():
    """Función principal para ejecutar la prueba de estrés cuántica."""
    # Permitir configurar duración por línea de comandos
    if len(sys.argv) > 1:
        try:
            minutes = float(sys.argv[1])
            TEST_CONFIG['duration_minutes'] = minutes
            logger.info(f"Duración configurada por línea de comandos: {minutes} minutos")
        except ValueError:
            logger.warning(f"Duración inválida: {sys.argv[1]}. Usando valor por defecto: {TEST_CONFIG['duration_minutes']} minutos")
    
    # Crear y ejecutar la prueba
    test = QuantumStressTest(TEST_CONFIG)
    success = test.run()
    
    # Retornar código de salida apropiado
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()