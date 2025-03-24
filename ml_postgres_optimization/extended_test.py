#!/usr/bin/env python3
"""
Prueba Extendida (72 horas) - Optimizador ML para PostgreSQL

Este script ejecuta pruebas de duración extendida para validar la estabilidad
y rendimiento del sistema de base de datos con el optimizador de ML.
"""
import psycopg2
import random
import time
import datetime
import logging
import os
import json
import signal
import sys
from threading import Thread, Event

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extended_test_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ExtendedTest')

# Configuración de la prueba
TEST_CONFIG = {
    'duration_hours': 72,              # Duración total en horas
    'operations_per_batch': 100,       # Operaciones por lote
    'operations_interval_ms': 1000,    # Intervalo entre lotes (ms)
    'checkpoint_interval_min': 30,     # Intervalo para checkpoints (minutos)
    'result_save_interval_min': 5,     # Intervalo para guardar resultados (minutos)
    'report_interval_min': 1,          # Intervalo para informes de progreso (minutos)
    'distribution': {                  # Distribución de tipos de operaciones
        'read': 0.45,                  # 45% lecturas
        'write': 0.30,                 # 30% escrituras
        'update': 0.15,                # 15% actualizaciones
        'delete': 0.05,                # 5% eliminaciones
        'transaction': 0.05            # 5% transacciones
    }
}

class ExtendedTestRunner:
    """Ejecutor de pruebas extendidas para el sistema de base de datos."""
    
    def __init__(self, config=None):
        """Inicializa el ejecutor de pruebas con la configuración especificada."""
        self.config = config or TEST_CONFIG
        self.stop_event = Event()
        self.conn = None
        self.cur = None
        self.start_time = None
        self.end_time = None
        
        # Estadísticas
        self.stats = {
            'operations': {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'by_type': {op_type: 0 for op_type in self.config['distribution'].keys()}
            },
            'latency': {
                'min': float('inf'),
                'max': 0,
                'avg': 0,
                'total': 0
            },
            'checkpoints': [],
            'errors': []
        }
        
        # Inicializar archivo de resultados
        self.results_file = f"extended_test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        logger.info(f"Test extendido inicializado con duración de {self.config['duration_hours']} horas")
    
    def connect_db(self):
        """Establece conexión a PostgreSQL."""
        try:
            self.conn = psycopg2.connect(
                dbname=os.environ.get("POSTGRES_DB", "postgres"),
                user=os.environ.get("POSTGRES_USER", "postgres"),
                password=os.environ.get("POSTGRES_PASSWORD", ""),
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                port=os.environ.get("POSTGRES_PORT", "5432")
            )
            self.cur = self.conn.cursor()
            logger.info("Conexión establecida con PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Error conectando a PostgreSQL: {e}")
            return False
    
    def close_db(self):
        """Cierra la conexión a PostgreSQL."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
            logger.info("Conexión cerrada con PostgreSQL")
    
    def save_results(self):
        """Guarda los resultados actuales a un archivo JSON."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump({
                    'config': self.config,
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': self.end_time.isoformat() if self.end_time else None,
                    'duration': (datetime.datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                    'stats': self.stats
                }, f, indent=2)
            logger.debug(f"Resultados guardados en {self.results_file}")
        except Exception as e:
            logger.error(f"Error guardando resultados: {e}")
    
    def create_checkpoint(self):
        """Crea un checkpoint con las estadísticas actuales."""
        checkpoint = {
            'timestamp': datetime.datetime.now().isoformat(),
            'elapsed_hours': (datetime.datetime.now() - self.start_time).total_seconds() / 3600,
            'operations': self.stats['operations']['total'],
            'success_rate': (self.stats['operations']['successful'] / max(1, self.stats['operations']['total'])) * 100,
            'avg_latency': self.stats['latency']['avg'],
            'error_count': len(self.stats['errors'])
        }
        self.stats['checkpoints'].append(checkpoint)
        logger.info(f"Checkpoint creado: {checkpoint['elapsed_hours']:.2f} horas, "
                   f"{checkpoint['success_rate']:.2f}% éxito, "
                   f"{checkpoint['avg_latency']:.2f} ms latencia")
    
    def execute_operation(self, op_type):
        """Ejecuta una operación del tipo especificado."""
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            if op_type == 'read':
                # Operación de lectura
                self.cur.execute("SELECT * FROM genesis_metrics ORDER BY RANDOM() LIMIT 10")
                rows = self.cur.fetchall()
                success = True
            
            elif op_type == 'write':
                # Operación de escritura
                data = f"Extended test write operation at {datetime.datetime.now().isoformat()}"
                self.cur.execute("INSERT INTO genesis_operations (data) VALUES (%s) RETURNING id", (data,))
                new_id = self.cur.fetchone()[0]
                self.conn.commit()
                success = True
            
            elif op_type == 'update':
                # Operación de actualización
                self.cur.execute("UPDATE genesis_operations SET data = %s WHERE id = (SELECT id FROM genesis_operations ORDER BY RANDOM() LIMIT 1) RETURNING id",
                              (f"Updated at {datetime.datetime.now().isoformat()}",))
                updated_id = self.cur.fetchone()
                self.conn.commit()
                success = updated_id is not None
            
            elif op_type == 'delete':
                # Operación de eliminación (simulada, no elimina realmente para no vaciar la tabla)
                self.cur.execute("SELECT id FROM genesis_operations ORDER BY RANDOM() LIMIT 1")
                row = self.cur.fetchone()
                if row:
                    # En vez de eliminar, marcamos como procesado 
                    self.cur.execute("UPDATE genesis_operations SET data = 'PROCESSED' WHERE id = %s", (row[0],))
                    self.conn.commit()
                    success = True
            
            elif op_type == 'transaction':
                # Operación de transacción compuesta
                self.cur.execute("BEGIN")
                # Insertar
                self.cur.execute("INSERT INTO genesis_operations (data) VALUES (%s) RETURNING id",
                              (f"Transaction insert at {datetime.datetime.now().isoformat()}",))
                new_id = self.cur.fetchone()[0]
                # Actualizar
                self.cur.execute("UPDATE genesis_operations SET data = %s WHERE id = (SELECT id FROM genesis_operations WHERE id != %s ORDER BY RANDOM() LIMIT 1)",
                              (f"Transaction update at {datetime.datetime.now().isoformat()}", new_id))
                # Leer
                self.cur.execute("SELECT COUNT(*) FROM genesis_operations")
                count = self.cur.fetchone()[0]
                # Confirmar
                self.conn.commit()
                success = True
            
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            error_msg = str(e)
            logger.warning(f"Error en operación {op_type}: {e}")
            success = False
        
        # Calcular latencia
        latency = (time.time() - start_time) * 1000  # en milisegundos
        
        # Actualizar estadísticas
        self.stats['operations']['total'] += 1
        self.stats['operations']['by_type'][op_type] += 1
        
        if success:
            self.stats['operations']['successful'] += 1
        else:
            self.stats['operations']['failed'] += 1
            self.stats['errors'].append({
                'timestamp': datetime.datetime.now().isoformat(),
                'operation': op_type,
                'message': error_msg
            })
        
        # Actualizar estadísticas de latencia
        self.stats['latency']['min'] = min(self.stats['latency']['min'], latency)
        self.stats['latency']['max'] = max(self.stats['latency']['max'], latency)
        self.stats['latency']['total'] += latency
        self.stats['latency']['avg'] = self.stats['latency']['total'] / self.stats['operations']['total']
        
        return success, latency
    
    def select_operation_type(self):
        """Selecciona un tipo de operación según la distribución configurada."""
        r = random.random()
        cumulative = 0
        for op_type, probability in self.config['distribution'].items():
            cumulative += probability
            if r <= cumulative:
                return op_type
        return list(self.config['distribution'].keys())[0]  # Por defecto, el primero
    
    def execute_batch(self):
        """Ejecuta un lote de operaciones."""
        operations = []
        
        for _ in range(self.config['operations_per_batch']):
            if self.stop_event.is_set():
                break
                
            op_type = self.select_operation_type()
            success, latency = self.execute_operation(op_type)
            operations.append({
                'type': op_type,
                'success': success,
                'latency': latency
            })
        
        return operations
    
    def progress_reporter(self):
        """Función para reportar progreso periódicamente."""
        last_report = time.time()
        
        while not self.stop_event.is_set():
            now = time.time()
            if now - last_report >= self.config['report_interval_min'] * 60:
                elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
                total_duration = self.config['duration_hours'] * 3600
                progress = min(100, (elapsed / total_duration) * 100)
                
                logger.info(f"Progreso: {progress:.2f}% ({elapsed/3600:.2f}/{self.config['duration_hours']} horas)")
                logger.info(f"Operaciones: {self.stats['operations']['total']} total, "
                           f"{self.stats['operations']['successful']} exitosas, "
                           f"{self.stats['operations']['failed']} fallidas")
                logger.info(f"Latencia (ms): min={self.stats['latency']['min']:.2f}, "
                           f"max={self.stats['latency']['max']:.2f}, "
                           f"avg={self.stats['latency']['avg']:.2f}")
                
                last_report = now
            
            time.sleep(1)
    
    def results_saver(self):
        """Función para guardar resultados periódicamente."""
        last_save = time.time()
        
        while not self.stop_event.is_set():
            now = time.time()
            if now - last_save >= self.config['result_save_interval_min'] * 60:
                self.save_results()
                last_save = now
            
            time.sleep(5)
    
    def checkpoint_creator(self):
        """Función para crear checkpoints periódicamente."""
        last_checkpoint = time.time()
        
        while not self.stop_event.is_set():
            now = time.time()
            if now - last_checkpoint >= self.config['checkpoint_interval_min'] * 60:
                self.create_checkpoint()
                last_checkpoint = now
            
            time.sleep(5)
    
    def handle_signal(self, signum, frame):
        """Manejador de señales para SIGINT y SIGTERM."""
        logger.info(f"Recibida señal {signum}, deteniendo prueba...")
        self.stop_event.set()
    
    def run(self):
        """Ejecuta la prueba extendida completa."""
        if not self.connect_db():
            logger.error("No se pudo conectar a la base de datos. Abortando prueba.")
            return False
        
        # Configurar manejador de señales
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        
        # Iniciar tiempo de prueba
        self.start_time = datetime.datetime.now()
        logger.info(f"Iniciando prueba extendida a las {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duración configurada: {self.config['duration_hours']} horas")
        
        # Crear hilos auxiliares
        reporter_thread = Thread(target=self.progress_reporter)
        saver_thread = Thread(target=self.results_saver)
        checkpoint_thread = Thread(target=self.checkpoint_creator)
        
        try:
            # Iniciar hilos
            reporter_thread.daemon = True
            saver_thread.daemon = True
            checkpoint_thread.daemon = True
            
            reporter_thread.start()
            saver_thread.start()
            checkpoint_thread.start()
            
            # Bucle principal de la prueba
            end_time = time.time() + (self.config['duration_hours'] * 3600)
            
            while time.time() < end_time and not self.stop_event.is_set():
                # Ejecutar un lote de operaciones
                self.execute_batch()
                
                # Esperar el intervalo configurado
                time.sleep(self.config['operations_interval_ms'] / 1000)
            
            # Finalizar prueba
            self.end_time = datetime.datetime.now()
            elapsed_hours = (self.end_time - self.start_time).total_seconds() / 3600
            
            logger.info(f"Prueba finalizada a las {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Duración real: {elapsed_hours:.2f} horas")
            
            # Crear checkpoint final
            self.create_checkpoint()
            
            # Calcular estadísticas finales
            success_rate = (self.stats['operations']['successful'] / max(1, self.stats['operations']['total'])) * 100
            
            logger.info(f"Resultados finales:")
            logger.info(f"Total operaciones: {self.stats['operations']['total']}")
            logger.info(f"Tasa de éxito: {success_rate:.2f}%")
            logger.info(f"Latencia promedio: {self.stats['latency']['avg']:.2f} ms")
            logger.info(f"Errores totales: {len(self.stats['errors'])}")
            
            # Guardar resultados finales
            self.save_results()
            
            return success_rate >= 99.5 and self.stats['latency']['avg'] < 5  # Criterio de éxito
            
        except Exception as e:
            logger.error(f"Error en la prueba: {e}")
            return False
        finally:
            self.stop_event.set()  # Asegurar que los hilos se detengan
            reporter_thread.join(timeout=2)
            saver_thread.join(timeout=2)
            checkpoint_thread.join(timeout=2)
            self.close_db()

def main():
    """Función principal para ejecutar la prueba extendida."""
    # Configuración con una duración más corta para pruebas iniciales (se puede cambiar)
    test_config = TEST_CONFIG.copy()
    
    # Permitir modificar la duración por línea de comandos
    if len(sys.argv) > 1:
        try:
            hours = float(sys.argv[1])
            test_config['duration_hours'] = hours
            logger.info(f"Duración configurada por línea de comandos: {hours} horas")
        except ValueError:
            logger.warning(f"Duración inválida: {sys.argv[1]}. Usando valor por defecto: {test_config['duration_hours']} horas")
    
    # Crear y ejecutar la prueba
    runner = ExtendedTestRunner(test_config)
    success = runner.run()
    
    # Retornar código de salida apropiado
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()