#!/usr/bin/env python3
"""
TEST APOCALÍPTICO PARA POSTGRESQL - SIN PIEDAD
===================================================

Esta prueba está diseñada sin piedad ni remordimiento para llevar cualquier 
base de datos a sus límites absolutos y más allá. No se trata solo de un test 
de estrés - es un evento de extinción diseñado para probar la verdadera 
resiliencia del sistema bajo condiciones que violan todas las buenas prácticas.

ADVERTENCIA EXTREMA: ESTE SCRIPT PUEDE PROVOCAR:
- Uso de CPU al 100% en todos los núcleos
- Consumo de memoria hasta agotar swap
- Bloqueos completos del servidor
- Corrupción potencial de datos
- Sobrecalentamiento de hardware

UTILIZAR BAJO SU PROPIO RIESGO Y NUNCA EN PRODUCCIÓN.

La única misericordia que tiene es un temporizador que lo detiene eventualmente.
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
import concurrent.futures
import psycopg2
import json
import signal
import uuid
import gc
from typing import Dict, List, Any, Tuple, Optional, Union

# Configuración de logging brutal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [APOCALIPSIS]',
    handlers=[
        logging.FileHandler("logs/apocalipsis_postgresql.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('APOCALIPSIS')

# Configuración apocalíptica
APOCALYPSE_CONFIG = {
    # Parámetros temporales
    'duration_minutes': 2,                   # Duración antes del fin del mundo (reducido para test)
    
    # Capacidad de ataque
    'max_pool_connections': 50,              # Conexiones en el pool (reducido para evitar sobrecargar Replit)
    'active_connections': 20,                # Conexiones activas simultáneas (reducido para evitar sobrecargar Replit)
    'max_operations_per_second': 500,        # Operaciones por segundo (reducido para evitar sobrecargar Replit)
    'max_threads': multiprocessing.cpu_count() * 2,   # Hilos 2x núcleos (reducido para evitar sobrecargar Replit)
    'max_processes': multiprocessing.cpu_count(),     # Procesos = núcleos (reducido para evitar sobrecargar Replit)

    # Patrones de ataque
    'attack_patterns': [
        # Nombre                    % de conexiones  % de operaciones  Duración (s)  Probabilidad de fallos
        ('DEVASTADOR_TOTAL',        1.0,             1.0,              10,           0.0),  # Ataque total sin fallos
        ('AVALANCHA_CONEXIONES',    1.0,             0.3,              10,           0.0),  # Todas las conexiones, operaciones moderadas
        ('TSUNAMI_OPERACIONES',     0.3,             1.0,              10,           0.0),  # Pocas conexiones, todas las operaciones
        ('SOBRECARGA_MEMORIA',      0.5,             0.5,              5,            0.0),  # Consumo de memoria intensivo
        ('INYECCION_CAOS',          0.7,             0.7,              10,           0.3),  # Inyección de fallos intensiva
        ('OSCILACION_EXTREMA',      0.1,             1.0,              5,            0.0),  # Cambios bruscos de 10% a 100%
        ('INTERMITENCIA_BRUTAL',    0.0,             0.0,              3,            0.0),  # Pausa para dar falsa sensación de alivio
        ('APOCALIPSIS_FINAL',       1.0,             1.0,              20,           0.1)   # Golpe final con todo (reducido para test)
    ],
    
    # Tamaños de operaciones
    'operation_sizes': {
        'small': 0.2,    # 20% operaciones pequeñas
        'medium': 0.3,   # 30% operaciones medianas
        'large': 0.3,    # 30% operaciones grandes
        'massive': 0.2   # 20% operaciones masivas
    },
    
    # Configuración de conexión a PostgreSQL
    'db_config': {
        'dbname': os.environ.get("POSTGRES_DB", "neondb"),
        'user': os.environ.get("POSTGRES_USER", "neondb_owner"),
        'password': os.environ.get("POSTGRES_PASSWORD", ""),
        'host': os.environ.get("POSTGRES_HOST", "ep-shy-bush-a5z5mx7m.us-east-2.aws.neon.tech"),
        'port': os.environ.get("POSTGRES_PORT", "5432"),
        'sslmode': os.environ.get("PGSSLMODE", "require")
    }
}

class MemoryBomb:
    """Clase para consumir memoria de forma agresiva."""
    
    def __init__(self, size_mb=20):
        """Inicializar bomba de memoria (tamaño reducido para Replit)."""
        self.size_mb = size_mb
        self.data = []
        
    def explode(self):
        """Explotar y consumir memoria."""
        try:
            # Consumir aproximadamente X MB de memoria
            for _ in range(self.size_mb):
                # Aproximadamente 1MB por cada elemento (varía según la plataforma)
                self.data.append(bytearray(1024 * 1024))
            return True
        except MemoryError:
            return False
            
    def defuse(self):
        """Liberar memoria."""
        self.data = []
        gc.collect()  # Forzar recolección de basura

class PostgresqlApocalypse:
    """Clase principal que ejecuta el apocalipsis en PostgreSQL."""
    
    def __init__(self, config):
        """Inicializar apocalipsis."""
        self.config = config
        self.stop_event = threading.Event()
        self.connections_pool = []
        self.connection_lock = threading.Lock()
        self.stats = {
            'start_time': None,
            'end_time': None,
            'operations_attempted': 0,
            'operations_succeeded': 0,
            'operations_failed': 0,
            'max_concurrent_operations': 0,
            'max_memory_used_mb': 0,
            'connection_errors': 0,
            'current_pattern': None,
            'phases_completed': [],
        }
        self.memory_bombs = []
        
        # Crear directorio de logs
        os.makedirs("logs", exist_ok=True)
        
        logger.info("=== INICIALIZANDO APOCALIPSIS POSTGRESQL ===")
        logger.info(f"Configuración cargada: {len(self.config['attack_patterns'])} patrones de ataque")
        logger.info(f"Duración planificada: {self.config['duration_minutes']} minutos")
        logger.warning("ADVERTENCIA: ESTE TEST PUEDE CAUSAR INESTABILIDAD SEVERA EN EL SISTEMA")
        
    def _create_connection(self):
        """Crear una nueva conexión a PostgreSQL."""
        try:
            # Mostrar información de debug para la primera conexión
            if self.stats['connection_errors'] == 0:
                logger.info(f"Intentando conectar a PostgreSQL con: {self.config['db_config']}")
                
            conn = psycopg2.connect(**self.config['db_config'])
            conn.autocommit = True
            
            # Log para la primera conexión exitosa
            if self.stats['connection_errors'] == 0:
                logger.info("¡Conexión exitosa a PostgreSQL!")
                
            return conn
        except Exception as e:
            with self.connection_lock:
                self.stats['connection_errors'] += 1
            logger.error(f"Error creando conexión: {e}")
            return None
            
    def initialize_connection_pool(self):
        """Inicializar pool de conexiones."""
        logger.info(f"Inicializando pool con {self.config['max_pool_connections']} conexiones...")
        
        successful = 0
        threads = []
        
        # Crear conexiones en paralelo para acelerar la inicialización
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(self._create_connection) for _ in range(self.config['max_pool_connections'])]
            
            for future in concurrent.futures.as_completed(futures):
                conn = future.result()
                if conn:
                    self.connections_pool.append(conn)
                    successful += 1
                    if successful % 100 == 0:
                        logger.info(f"Conexiones creadas: {successful}/{self.config['max_pool_connections']}")
        
        logger.info(f"Pool inicializado con {len(self.connections_pool)} conexiones válidas")
        return len(self.connections_pool) > 0
        
    def get_connection(self):
        """Obtener una conexión del pool."""
        with self.connection_lock:
            if not self.connections_pool:
                return self._create_connection()
            return self.connections_pool.pop()
            
    def return_connection(self, conn):
        """Devolver una conexión al pool."""
        if conn:
            try:
                # Verificar que la conexión siga siendo válida
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
                
                with self.connection_lock:
                    self.connections_pool.append(conn)
            except:
                # Conexión inválida, intentar cerrarla
                try:
                    conn.close()
                except:
                    pass
                    
    def close_all_connections(self):
        """Cerrar todas las conexiones del pool."""
        logger.info("Cerrando todas las conexiones...")
        with self.connection_lock:
            for conn in self.connections_pool:
                try:
                    conn.close()
                except:
                    pass
            self.connections_pool = []
            
    def generate_random_data(self, size='medium'):
        """Generar datos aleatorios para operaciones."""
        if size == 'small':
            # Texto de 1-2KB
            length = random.randint(1000, 2000)
        elif size == 'medium':
            # Texto de 10-20KB
            length = random.randint(10000, 20000)
        elif size == 'large':
            # Texto de 100-200KB
            length = random.randint(100000, 200000)
        else:  # massive
            # Texto de 1-2MB
            length = random.randint(1000000, 2000000)
            
        # Generar texto aleatorio
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        return ''.join(random.choice(chars) for _ in range(length))
        
    def select_operation_size(self):
        """Seleccionar tamaño de operación según distribución configurada."""
        r = random.random()
        cumulative = 0
        for size, probability in self.config['operation_sizes'].items():
            cumulative += probability
            if r <= cumulative:
                return size
        return 'medium'  # Default
        
    def execute_destructive_operation(self, operation_type, conn=None):
        """Ejecutar una operación destructiva en PostgreSQL."""
        close_conn = False
        if not conn:
            conn = self.get_connection()
            close_conn = True
            
        if not conn:
            with self.connection_lock:
                self.stats['operations_failed'] += 1
            return False
            
        success = False
        try:
            cur = conn.cursor()
            
            if operation_type == 'massive_insert':
                # Insertar gran cantidad de datos
                size = self.select_operation_size()
                data = self.generate_random_data(size)
                operation_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO genesis_operations (data) VALUES (%s) RETURNING id",
                    (f"APOCALYPSE-{operation_id}-{data}",)
                )
                success = True
                
            elif operation_type == 'massive_update':
                # Actualizar registros aleatoriamente
                size = self.select_operation_size()
                data = self.generate_random_data(size)
                cur.execute(
                    "UPDATE genesis_operations SET data = %s WHERE id IN (SELECT id FROM genesis_operations ORDER BY RANDOM() LIMIT 5)",
                    (f"APOCALYPSE-UPDATE-{data}",)
                )
                success = True
                
            elif operation_type == 'complex_query':
                # Consulta compleja
                cur.execute("""
                WITH RECURSIVE t(n) AS (
                    SELECT 1
                    UNION ALL
                    SELECT n+1 FROM t WHERE n < 1000
                )
                SELECT COUNT(*), AVG(n), STDDEV(n), MIN(n), MAX(n) FROM t;
                """)
                success = True
                
            elif operation_type == 'table_scan':
                # Escaneo completo de tabla
                cur.execute("SELECT COUNT(*) FROM genesis_operations WHERE data LIKE '%X%'")
                success = True
                
            elif operation_type == 'transaction_batch':
                # Lote de operaciones en una transacción
                conn.autocommit = False
                try:
                    cur.execute("BEGIN")
                    for _ in range(20):
                        size = self.select_operation_size()
                        data = self.generate_random_data(size)
                        operation_id = str(uuid.uuid4())
                        cur.execute(
                            "INSERT INTO genesis_operations (data) VALUES (%s)",
                            (f"APOCALYPSE-BATCH-{operation_id}-{data}",)
                        )
                    conn.commit()
                    success = True
                except:
                    conn.rollback()
                    raise
                finally:
                    conn.autocommit = True
                    
            elif operation_type == 'large_sort':
                # Ordenamiento masivo
                cur.execute("SELECT * FROM genesis_operations ORDER BY data LIMIT 1000")
                result = cur.fetchall()
                success = True
                
            elif operation_type == 'many_small_queries':
                # Muchas consultas pequeñas
                for _ in range(50):
                    cur.execute("SELECT 1")
                    cur.fetchall()
                success = True
                
            elif operation_type == 'long_running_query':
                # Consulta de larga duración
                cur.execute("SELECT pg_sleep(2)")
                success = True
                
            elif operation_type == 'vacuum':
                # Operación VACUUM (administrativa)
                try:
                    old_isolation_level = conn.isolation_level
                    conn.set_isolation_level(0)  # ISOLATION_LEVEL_AUTOCOMMIT
                    cur.execute("VACUUM ANALYZE genesis_operations")
                    conn.set_isolation_level(old_isolation_level)
                    success = True
                except:
                    success = False
                    
            elif operation_type == 'memory_intensive':
                # Operación que consume mucha memoria
                try:
                    # Consulta que consume mucha memoria
                    cur.execute("""
                    WITH RECURSIVE t(n) AS (
                        SELECT 1
                        UNION ALL
                        SELECT n+1 FROM t WHERE n < 10000
                    )
                    SELECT n, MD5(n::text) FROM t;
                    """)
                    result = cur.fetchall()  # Esto carga todo en memoria
                    success = True
                except:
                    success = False
                    
            elif operation_type == 'random_bomb':
                # Operación aleatoria extrema
                op_type = random.choice([
                    'massive_insert', 'massive_update', 'complex_query', 
                    'table_scan', 'transaction_batch', 'large_sort',
                    'many_small_queries', 'memory_intensive'
                ])
                return self.execute_destructive_operation(op_type, conn)
                
            cur.close()
            
        except Exception as e:
            logger.debug(f"Error en operación {operation_type}: {e}")
            success = False
            
        finally:
            if close_conn:
                self.return_connection(conn)
                
        with self.connection_lock:
            self.stats['operations_attempted'] += 1
            if success:
                self.stats['operations_succeeded'] += 1
            else:
                self.stats['operations_failed'] += 1
                
        return success
        
    def worker_function(self, worker_id, attack_percentage, inject_failures):
        """Función ejecutada por cada worker para realizar operaciones."""
        logger.debug(f"Worker {worker_id} iniciado")
        
        # Lista de operaciones disponibles
        operations = [
            'massive_insert', 'massive_update', 'complex_query', 
            'table_scan', 'transaction_batch', 'large_sort',
            'many_small_queries', 'long_running_query', 'memory_intensive',
            'random_bomb'
        ]
        
        # Obtener conexión dedicada para este worker
        conn = self.get_connection()
        if not conn:
            logger.warning(f"Worker {worker_id} no pudo obtener conexión")
            return
            
        try:
            # Ejecutar operaciones mientras no se detenga el test
            while not self.stop_event.is_set():
                # Determinar si ejecutamos en esta iteración (basado en porcentaje de ataque)
                if random.random() > attack_percentage:
                    time.sleep(0.01)  # Pequeña pausa
                    continue
                    
                # Determinar si inyectamos un fallo
                if inject_failures and random.random() < inject_failures:
                    # Simular error cerrando la conexión y creando una nueva
                    try:
                        conn.close()
                    except:
                        pass
                    conn = self.get_connection()
                    if not conn:
                        logger.warning(f"Worker {worker_id} no pudo obtener nueva conexión tras fallo")
                        return
                    continue
                    
                # Seleccionar operación aleatoria
                op_type = random.choice(operations)
                
                # Ejecutar operación
                self.execute_destructive_operation(op_type, conn)
                
        finally:
            # Devolver conexión al pool
            self.return_connection(conn)
            logger.debug(f"Worker {worker_id} finalizado")
            
    def memory_bomb_thread(self):
        """Hilo que lanza bombas de memoria periódicamente."""
        logger.info("Iniciando hilo de bombas de memoria")
        
        while not self.stop_event.is_set():
            # Decidir si lanzamos bomba de memoria
            if random.random() < 0.3:  # 30% de probabilidad
                # Decidir tamaño de bomba (reducido para Replit)
                size_mb = random.randint(5, 20)
                
                logger.debug(f"Lanzando bomba de memoria de {size_mb}MB")
                bomb = MemoryBomb(size_mb)
                if bomb.explode():
                    self.memory_bombs.append(bomb)
                    
                    # Actualizar estadística de uso máximo de memoria
                    total_mb = sum(b.size_mb for b in self.memory_bombs)
                    with self.connection_lock:
                        self.stats['max_memory_used_mb'] = max(self.stats['max_memory_used_mb'], total_mb)
                    
                    # 50% de probabilidad de liberar una bomba aleatoria
                    if random.random() < 0.5 and self.memory_bombs:
                        random_bomb = random.choice(self.memory_bombs)
                        self.memory_bombs.remove(random_bomb)
                        random_bomb.defuse()
            
            # Esperar antes de la siguiente bomba
            time.sleep(random.uniform(1, 5))
            
    def execute_attack_pattern(self, name, conn_percentage, op_percentage, duration, failure_prob):
        """Ejecutar un patrón de ataque específico."""
        logger.info(f"Iniciando patrón de ataque: {name}")
        with self.connection_lock:
            self.stats['current_pattern'] = name
            
        # Calcular número de workers
        active_connections = int(self.config['active_connections'] * conn_percentage)
        active_connections = max(1, min(active_connections, self.config['max_threads']))
        
        logger.info(f"Patrón {name}: {active_connections} conexiones activas, {op_percentage*100:.1f}% intensidad, {duration}s duración")
        
        # Crear y lanzar workers
        workers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=active_connections) as executor:
            # Lanzar workers
            futures = []
            for i in range(active_connections):
                future = executor.submit(
                    self.worker_function, 
                    i, 
                    op_percentage,
                    failure_prob
                )
                futures.append(future)
                
            # Esperar hasta fin de duración o detención
            end_time = time.time() + duration
            while time.time() < end_time and not self.stop_event.is_set():
                time.sleep(0.1)
                
                # Actualizar estadística de operaciones concurrentes
                active_count = sum(1 for f in futures if not f.done())
                with self.connection_lock:
                    self.stats['max_concurrent_operations'] = max(
                        self.stats['max_concurrent_operations'], 
                        active_count
                    )
                    
            # Registrar fase completada
            self.stats['phases_completed'].append({
                'name': name,
                'start_time': (datetime.datetime.now() - (datetime.timedelta(seconds=duration))).isoformat(),
                'end_time': datetime.datetime.now().isoformat(),
                'conn_percentage': conn_percentage,
                'op_percentage': op_percentage,
                'operations_attempted': self.stats['operations_attempted'],
                'operations_succeeded': self.stats['operations_succeeded'],
                'operations_failed': self.stats['operations_failed']
            })
            
            logger.info(f"Patrón de ataque {name} completado")
            
    def stop_gracefully(self):
        """Detener el test de forma controlada."""
        logger.info("Deteniendo test apocalíptico...")
        self.stop_event.set()
        
        # Liberar memoria
        logger.info(f"Liberando {len(self.memory_bombs)} bombas de memoria...")
        for bomb in self.memory_bombs:
            bomb.defuse()
        self.memory_bombs = []
        gc.collect()
        
        # Cerrar conexiones
        self.close_all_connections()
        
        # Registrar fin
        self.stats['end_time'] = datetime.datetime.now().isoformat()
        logger.info(f"Test apocalíptico detenido a las {self.stats['end_time']}")
        
    def handle_signal(self, signum, frame):
        """Manejador de señales para SIGINT y SIGTERM."""
        logger.warning(f"Recibida señal {signum}, deteniendo apocalipsis...")
        self.stop_gracefully()
        
    def save_results(self):
        """Guardar resultados del test."""
        results_file = f"logs/apocalipsis_resultados_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    'config': {k: v for k, v in self.config.items() if k != 'db_config'},  # No guardar credenciales
                    'stats': self.stats,
                    'summary': {
                        'total_operations': self.stats['operations_attempted'],
                        'success_rate': (self.stats['operations_succeeded'] / max(1, self.stats['operations_attempted'])) * 100,
                        'max_concurrency': self.stats['max_concurrent_operations'],
                        'max_memory_mb': self.stats['max_memory_used_mb'],
                        'patterns_completed': len(self.stats['phases_completed']),
                        'total_duration_seconds': (
                            datetime.datetime.fromisoformat(self.stats['end_time']) - 
                            datetime.datetime.fromisoformat(self.stats['start_time'])
                        ).total_seconds() if self.stats['end_time'] else 0
                    }
                }, f, indent=2)
            logger.info(f"Resultados guardados en {results_file}")
        except Exception as e:
            logger.error(f"Error guardando resultados: {e}")
            
    def run(self):
        """Ejecutar el test apocalíptico completo."""
        # Configurar manejador de señales
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        
        # Registrar inicio
        self.stats['start_time'] = datetime.datetime.now().isoformat()
        logger.info(f"Iniciando apocalipsis a las {self.stats['start_time']}")
        
        try:
            # Inicializar pool de conexiones
            if not self.initialize_connection_pool():
                logger.error("No se pudo inicializar el pool de conexiones. Abortando.")
                return False
                
            # Iniciar hilo de bombas de memoria
            memory_thread = threading.Thread(target=self.memory_bomb_thread)
            memory_thread.daemon = True
            memory_thread.start()
            
            # Ejecutar patrones de ataque secuencialmente
            try:
                for pattern in self.config['attack_patterns']:
                    if self.stop_event.is_set():
                        break
                        
                    name, conn_percent, op_percent, duration, failure_prob = pattern
                    self.execute_attack_pattern(name, conn_percent, op_percent, duration, failure_prob)
                    
            except Exception as e:
                logger.error(f"Error ejecutando patrones de ataque: {e}")
                
            # Detener test controladamente
            self.stop_gracefully()
            
            # Mostrar resultados
            ops_attempted = self.stats['operations_attempted']
            ops_succeeded = self.stats['operations_succeeded']
            success_rate = (ops_succeeded / max(1, ops_attempted)) * 100
            
            logger.info(f"RESULTADOS DEL APOCALIPSIS:")
            logger.info(f"- Operaciones intentadas: {ops_attempted}")
            logger.info(f"- Operaciones exitosas: {ops_succeeded}")
            logger.info(f"- Operaciones fallidas: {self.stats['operations_failed']}")
            logger.info(f"- Tasa de éxito: {success_rate:.2f}%")
            logger.info(f"- Concurrencia máxima: {self.stats['max_concurrent_operations']}")
            logger.info(f"- Memoria máxima utilizada: {self.stats['max_memory_used_mb']}MB")
            logger.info(f"- Fases completadas: {len(self.stats['phases_completed'])}")
            
            # Guardar resultados
            self.save_results()
            
            return success_rate > 50  # Criterio de éxito mínimo
            
        except Exception as e:
            logger.error(f"Error catastrófico en el apocalipsis: {e}")
            self.stop_gracefully()
            return False

def main():
    """Función principal."""
    # Permitir configurar duración por línea de comandos
    if len(sys.argv) > 1:
        try:
            minutes = float(sys.argv[1])
            APOCALYPSE_CONFIG['duration_minutes'] = minutes
            logger.info(f"Duración configurada por línea de comandos: {minutes} minutos")
        except ValueError:
            logger.warning(f"Duración inválida: {sys.argv[1]}. Usando valor por defecto: {APOCALYPSE_CONFIG['duration_minutes']} minutos")
    
    # Mensaje de advertencia final
    logger.warning("=" * 80)
    logger.warning("ADVERTENCIA FINAL: ESTE TEST ES EXTREMADAMENTE AGRESIVO")
    logger.warning("PUEDE CAUSAR INESTABILIDAD SEVERA EN EL SISTEMA")
    logger.warning("PRESIONE CTRL+C DENTRO DE 5 SEGUNDOS SI DESEA CANCELAR")
    logger.warning("=" * 80)
    
    try:
        time.sleep(5)  # Dar tiempo para cancelar
    except KeyboardInterrupt:
        logger.info("Test apocalíptico cancelado por el usuario")
        return
    
    # Iniciar el apocalipsis
    apocalypse = PostgresqlApocalypse(APOCALYPSE_CONFIG)
    result = apocalypse.run()
    
    if result:
        logger.info("PostgreSQL sobrevivió al apocalipsis. ¡IMPRESIONANTE!")
    else:
        logger.warning("PostgreSQL no superó todas las pruebas apocalípticas")
    
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()