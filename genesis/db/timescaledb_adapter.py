"""
Adaptador de base de datos TimescaleDB para el Sistema Genesis.

Este módulo proporciona un adaptador optimizado para TimescaleDB (extensión
de PostgreSQL para series temporales) que opera en un hilo separado
para evitar conflictos con los event loops asíncronos de WebSockets y RL.
"""

import psycopg2
import psycopg2.extras
from concurrent.futures import ThreadPoolExecutor
import queue
import logging
import time
import threading
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta

class TimescaleDBManager:
    """
    Gestor de conexiones a TimescaleDB que ejecuta operaciones en un hilo separado.
    
    Evita los problemas de event loop al separar completamente el acceso a base
    de datos del event loop principal de asyncio. Proporciona métodos para
    operaciones síncronas y asíncronas.
    """
    
    def __init__(self, 
                dsn: str = None,
                max_workers: int = 2,
                connection_timeout: int = 5,
                execute_timeout: float = 10.0,
                retry_attempts: int = 3,
                retry_delay: float = 1.0,
                enable_checkpoints: bool = True,
                checkpoint_interval: int = 3600):  # 1 hora
        """
        Inicializar gestor de TimescaleDB.
        
        Args:
            dsn: Cadena de conexión a PostgreSQL
            max_workers: Número máximo de hilos
            connection_timeout: Timeout para conexiones en segundos
            execute_timeout: Timeout para ejecuciones en segundos
            retry_attempts: Intentos para operaciones fallidas
            retry_delay: Retardo entre reintentos en segundos
            enable_checkpoints: Si es True, habilita checkpoints automáticos
            checkpoint_interval: Intervalo para checkpoints automáticos en segundos
        """
        self.logger = logging.getLogger(__name__)
        self.dsn = dsn or os.environ.get('DATABASE_URL')
        
        if not self.dsn:
            self.logger.warning("No se proporcionó cadena de conexión (DSN) a TimescaleDB")
        
        self.max_workers = max_workers
        self.connection_timeout = connection_timeout
        self.execute_timeout = execute_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Executor para operaciones en hilos separados
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Cola para operaciones en orden
        self.operation_queue = queue.Queue()
        
        # Hilo de procesamiento de operaciones
        self.processing_thread = None
        self.processing_active = False
        
        # Estadísticas
        self.stats = {
            'successful_operations': 0,
            'failed_operations': 0,
            'retried_operations': 0,
            'last_checkpoint': None,
            'last_error': None,
            'connections_created': 0,
            'connections_closed': 0
        }
        
        # Checkpointing
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_thread = None
        self.checkpoint_active = False
        
        # Lock para operaciones thread-safe
        self.lock = threading.RLock()
        
        self.logger.info("TimescaleDBManager inicializado")
    
    def start(self) -> bool:
        """
        Iniciar hilos de procesamiento y checkpointing.
        
        Returns:
            True si se inició correctamente
        """
        with self.lock:
            # Iniciar hilo de procesamiento
            if not self.processing_active:
                self.processing_active = True
                self.processing_thread = threading.Thread(target=self._operation_loop)
                self.processing_thread.daemon = True
                self.processing_thread.start()
                self.logger.info("Hilo de procesamiento de operaciones iniciado")
            
            # Iniciar hilo de checkpointing
            if self.enable_checkpoints and not self.checkpoint_active:
                self.checkpoint_active = True
                self.checkpoint_thread = threading.Thread(target=self._checkpoint_loop)
                self.checkpoint_thread.daemon = True
                self.checkpoint_thread.start()
                self.logger.info("Hilo de checkpointing iniciado")
            
            return True
    
    def stop(self) -> bool:
        """
        Detener hilos de procesamiento y checkpointing.
        
        Returns:
            True si se detuvo correctamente
        """
        with self.lock:
            # Detener hilo de procesamiento
            if self.processing_active:
                self.processing_active = False
                if self.processing_thread and self.processing_thread.is_alive():
                    self.processing_thread.join(timeout=5.0)
                self.logger.info("Hilo de procesamiento de operaciones detenido")
            
            # Detener hilo de checkpointing
            if self.checkpoint_active:
                self.checkpoint_active = False
                if self.checkpoint_thread and self.checkpoint_thread.is_alive():
                    self.checkpoint_thread.join(timeout=5.0)
                self.logger.info("Hilo de checkpointing detenido")
            
            # Detener executor
            self.executor.shutdown(wait=True)
            
            return True
    
    def connect(self) -> Optional[psycopg2.extensions.connection]:
        """
        Establecer conexión a TimescaleDB.
        
        Returns:
            Conexión o None si falla
        """
        try:
            conn = psycopg2.connect(
                self.dsn,
                connect_timeout=self.connection_timeout
            )
            with self.lock:
                self.stats['connections_created'] += 1
            self.logger.debug("Conexión a TimescaleDB establecida")
            return conn
        except Exception as e:
            self.logger.error(f"Error al conectar a TimescaleDB: {str(e)}")
            with self.lock:
                self.stats['last_error'] = str(e)
            return None
    
    def _operation_loop(self) -> None:
        """Loop principal para procesar operaciones de la cola."""
        self.logger.info("Iniciando loop de procesamiento de operaciones")
        
        while self.processing_active:
            try:
                # Obtener operación de la cola
                try:
                    operation = self.operation_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Procesar operación
                try:
                    operation_type = operation.get('type', 'unknown')
                    
                    if operation_type == 'execute':
                        self._execute_operation(operation)
                    elif operation_type == 'fetch':
                        self._fetch_operation(operation)
                    elif operation_type == 'transaction':
                        self._transaction_operation(operation)
                    else:
                        self.logger.warning(f"Tipo de operación desconocido: {operation_type}")
                
                except Exception as e:
                    self.logger.error(f"Error procesando operación: {str(e)}")
                    with self.lock:
                        self.stats['failed_operations'] += 1
                        self.stats['last_error'] = str(e)
                
                finally:
                    # Marcar operación como completada
                    self.operation_queue.task_done()
            
            except Exception as e:
                self.logger.error(f"Error en loop de procesamiento: {str(e)}")
        
        self.logger.info("Loop de procesamiento de operaciones finalizado")
    
    def _checkpoint_loop(self) -> None:
        """Loop para crear checkpoints periódicos."""
        self.logger.info("Iniciando loop de checkpointing")
        
        last_checkpoint_time = time.time()
        
        while self.checkpoint_active:
            try:
                # Verificar si es hora de hacer checkpoint
                current_time = time.time()
                if current_time - last_checkpoint_time >= self.checkpoint_interval:
                    # Crear checkpoint
                    checkpoint_id = self.create_checkpoint()
                    
                    if checkpoint_id:
                        self.logger.info(f"Checkpoint automático creado: {checkpoint_id}")
                        last_checkpoint_time = current_time
                    else:
                        self.logger.warning("Error al crear checkpoint automático")
                
                # Esperar
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error en loop de checkpointing: {str(e)}")
                time.sleep(30)  # Esperar más tiempo en caso de error
        
        self.logger.info("Loop de checkpointing finalizado")
    
    def _execute_operation(self, operation: Dict[str, Any]) -> None:
        """
        Ejecutar operación de tipo execute.
        
        Args:
            operation: Diccionario con la operación
        """
        query = operation.get('query', '')
        params = operation.get('params', None)
        callback = operation.get('callback', None)
        error_callback = operation.get('error_callback', None)
        
        # Crear conexión
        conn = self.connect()
        if not conn:
            if error_callback:
                error_callback("Error de conexión a la base de datos")
            return
        
        try:
            # Realizar operación
            with conn.cursor() as cur:
                cur.execute(query, params)
            
            # Confirmar cambios
            conn.commit()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['successful_operations'] += 1
            
            # Llamar callback
            if callback:
                callback(True)
                
        except Exception as e:
            # Rollback
            conn.rollback()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['failed_operations'] += 1
                self.stats['last_error'] = str(e)
            
            # Llamar error_callback
            if error_callback:
                error_callback(str(e))
                
            self.logger.error(f"Error ejecutando query: {str(e)}")
            
        finally:
            # Cerrar conexión
            conn.close()
            with self.lock:
                self.stats['connections_closed'] += 1
    
    def _fetch_operation(self, operation: Dict[str, Any]) -> None:
        """
        Ejecutar operación de tipo fetch.
        
        Args:
            operation: Diccionario con la operación
        """
        query = operation.get('query', '')
        params = operation.get('params', None)
        callback = operation.get('callback', None)
        error_callback = operation.get('error_callback', None)
        as_dict = operation.get('as_dict', True)
        
        # Crear conexión
        conn = self.connect()
        if not conn:
            if error_callback:
                error_callback("Error de conexión a la base de datos")
            return
        
        try:
            # Realizar operación
            if as_dict:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    # Convertir a lista de diccionarios
                    if results and len(results) > 0:
                        results = [dict(row) for row in results]
            else:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['successful_operations'] += 1
            
            # Llamar callback
            if callback:
                callback(results)
                
        except Exception as e:
            # Rollback
            conn.rollback()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['failed_operations'] += 1
                self.stats['last_error'] = str(e)
            
            # Llamar error_callback
            if error_callback:
                error_callback(str(e))
                
            self.logger.error(f"Error ejecutando fetch query: {str(e)}")
            
        finally:
            # Cerrar conexión
            conn.close()
            with self.lock:
                self.stats['connections_closed'] += 1
    
    def _transaction_operation(self, operation: Dict[str, Any]) -> None:
        """
        Ejecutar operación de tipo transaction.
        
        Args:
            operation: Diccionario con la operación
        """
        queries = operation.get('queries', [])
        callback = operation.get('callback', None)
        error_callback = operation.get('error_callback', None)
        
        # Verificar que hay queries
        if not queries:
            if error_callback:
                error_callback("No se proporcionaron queries para la transacción")
            return
        
        # Crear conexión
        conn = self.connect()
        if not conn:
            if error_callback:
                error_callback("Error de conexión a la base de datos")
            return
        
        try:
            # Ejecutar queries en transacción
            with conn.cursor() as cur:
                for query_data in queries:
                    query = query_data.get('query', '')
                    params = query_data.get('params', None)
                    cur.execute(query, params)
            
            # Confirmar cambios
            conn.commit()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['successful_operations'] += 1
            
            # Llamar callback
            if callback:
                callback(True)
                
        except Exception as e:
            # Rollback
            conn.rollback()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['failed_operations'] += 1
                self.stats['last_error'] = str(e)
            
            # Llamar error_callback
            if error_callback:
                error_callback(str(e))
                
            self.logger.error(f"Error ejecutando transacción: {str(e)}")
            
        finally:
            # Cerrar conexión
            conn.close()
            with self.lock:
                self.stats['connections_closed'] += 1
    
    def execute(self, query: str, params: Optional[tuple] = None, callback: Optional[Callable] = None) -> None:
        """
        Ejecutar query de forma asíncrona (non-blocking).
        
        Args:
            query: Consulta SQL
            params: Parámetros para la consulta
            callback: Función a llamar con el resultado
        """
        # Añadir operación a la cola
        operation = {
            'type': 'execute',
            'query': query,
            'params': params,
            'callback': callback,
            'error_callback': None,
            'timestamp': time.time()
        }
        
        self.operation_queue.put(operation)
    
    def execute_sync(self, query: str, params: Optional[tuple] = None) -> bool:
        """
        Ejecutar query de forma síncrona (blocking).
        
        Args:
            query: Consulta SQL
            params: Parámetros para la consulta
            
        Returns:
            True si se ejecutó correctamente
        """
        # Crear conexión
        conn = self.connect()
        if not conn:
            return False
        
        try:
            # Realizar operación
            with conn.cursor() as cur:
                cur.execute(query, params)
            
            # Confirmar cambios
            conn.commit()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['successful_operations'] += 1
            
            return True
                
        except Exception as e:
            # Rollback
            conn.rollback()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['failed_operations'] += 1
                self.stats['last_error'] = str(e)
                
            self.logger.error(f"Error ejecutando query síncrona: {str(e)}")
            return False
            
        finally:
            # Cerrar conexión
            conn.close()
            with self.lock:
                self.stats['connections_closed'] += 1
    
    def fetch(self, 
             query: str, 
             params: Optional[tuple] = None, 
             callback: Optional[Callable] = None,
             error_callback: Optional[Callable] = None,
             as_dict: bool = True) -> None:
        """
        Obtener datos de forma asíncrona (non-blocking).
        
        Args:
            query: Consulta SQL
            params: Parámetros para la consulta
            callback: Función a llamar con el resultado
            error_callback: Función a llamar en caso de error
            as_dict: Si es True, devuelve resultados como diccionarios
        """
        # Añadir operación a la cola
        operation = {
            'type': 'fetch',
            'query': query,
            'params': params,
            'callback': callback,
            'error_callback': error_callback,
            'as_dict': as_dict,
            'timestamp': time.time()
        }
        
        self.operation_queue.put(operation)
    
    def fetch_sync(self, 
                  query: str, 
                  params: Optional[tuple] = None, 
                  as_dict: bool = True) -> List[Any]:
        """
        Obtener datos de forma síncrona (blocking).
        
        Args:
            query: Consulta SQL
            params: Parámetros para la consulta
            as_dict: Si es True, devuelve resultados como diccionarios
            
        Returns:
            Lista de resultados
        """
        # Crear conexión
        conn = self.connect()
        if not conn:
            return []
        
        try:
            # Realizar operación
            if as_dict:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    # Convertir a lista de diccionarios
                    if results and len(results) > 0:
                        results = [dict(row) for row in results]
            else:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['successful_operations'] += 1
            
            return results
                
        except Exception as e:
            # Rollback
            conn.rollback()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['failed_operations'] += 1
                self.stats['last_error'] = str(e)
                
            self.logger.error(f"Error ejecutando fetch síncrono: {str(e)}")
            return []
            
        finally:
            # Cerrar conexión
            conn.close()
            with self.lock:
                self.stats['connections_closed'] += 1
    
    def transaction(self, 
                   queries: List[Dict[str, Any]], 
                   callback: Optional[Callable] = None,
                   error_callback: Optional[Callable] = None) -> None:
        """
        Ejecutar múltiples queries en una transacción asíncrona.
        
        Args:
            queries: Lista de diccionarios con queries y parámetros
            callback: Función a llamar con el resultado
            error_callback: Función a llamar en caso de error
        """
        # Añadir operación a la cola
        operation = {
            'type': 'transaction',
            'queries': queries,
            'callback': callback,
            'error_callback': error_callback,
            'timestamp': time.time()
        }
        
        self.operation_queue.put(operation)
    
    def transaction_sync(self, queries: List[Dict[str, Any]]) -> bool:
        """
        Ejecutar múltiples queries en una transacción síncrona.
        
        Args:
            queries: Lista de diccionarios con queries y parámetros
            
        Returns:
            True si se ejecutó correctamente
        """
        # Verificar que hay queries
        if not queries:
            return False
        
        # Crear conexión
        conn = self.connect()
        if not conn:
            return False
        
        try:
            # Ejecutar queries en transacción
            with conn.cursor() as cur:
                for query_data in queries:
                    query = query_data.get('query', '')
                    params = query_data.get('params', None)
                    cur.execute(query, params)
            
            # Confirmar cambios
            conn.commit()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['successful_operations'] += 1
            
            return True
                
        except Exception as e:
            # Rollback
            conn.rollback()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['failed_operations'] += 1
                self.stats['last_error'] = str(e)
                
            self.logger.error(f"Error ejecutando transacción síncrona: {str(e)}")
            return False
            
        finally:
            # Cerrar conexión
            conn.close()
            with self.lock:
                self.stats['connections_closed'] += 1
    
    def create_checkpoint(self) -> Optional[str]:
        """
        Crear checkpoint con información de estado.
        
        Returns:
            ID del checkpoint o None si falla
        """
        checkpoint_id = f"checkpoint_{int(time.time())}"
        checkpoint_data = {
            'id': checkpoint_id,
            'timestamp': datetime.now().isoformat(),
            'stats': self.get_stats()
        }
        
        # Guardar checkpoint en la base de datos
        try:
            query = """
                INSERT INTO system_checkpoints (id, type, data, timestamp)
                VALUES (%s, %s, %s, NOW())
            """
            params = (checkpoint_id, 'database', json.dumps(checkpoint_data))
            
            success = self.execute_sync(query, params)
            
            if success:
                with self.lock:
                    self.stats['last_checkpoint'] = checkpoint_id
                return checkpoint_id
        except Exception as e:
            self.logger.error(f"Error al crear checkpoint: {str(e)}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas actuales.
        
        Returns:
            Diccionario con estadísticas
        """
        with self.lock:
            stats_copy = self.stats.copy()
            
            # Añadir información adicional
            stats_copy['queue_size'] = self.operation_queue.qsize()
            stats_copy['is_active'] = self.processing_active
            stats_copy['timestamp'] = datetime.now().isoformat()
            
            return stats_copy
    
    def setup_hypertables(self) -> bool:
        """
        Configurar hipertablas de TimescaleDB.
        
        Returns:
            True si se configuró correctamente
        """
        # Verificar si TimescaleDB está disponible
        conn = self.connect()
        if not conn:
            return False
        
        try:
            # Verificar extensión TimescaleDB
            with conn.cursor() as cur:
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
                extension_exists = cur.fetchone()
                
                if not extension_exists:
                    # Instalar extensión TimescaleDB
                    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
                    conn.commit()
                    self.logger.info("Extensión TimescaleDB instalada")
                else:
                    self.logger.info("Extensión TimescaleDB ya está instalada")
                
                # Crear tablas si no existen
                
                # Tabla para datos de mercado (market_data)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        open DOUBLE PRECISION,
                        high DOUBLE PRECISION,
                        low DOUBLE PRECISION,
                        close DOUBLE PRECISION,
                        volume DOUBLE PRECISION
                    )
                """)
                
                # Crear hipertabla para market_data
                cur.execute("""
                    SELECT create_hypertable('market_data', 'timestamp', 
                                            if_not_exists => TRUE, 
                                            create_default_indexes => TRUE)
                """)
                
                # Tabla para operaciones (trades)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        price DOUBLE PRECISION,
                        amount DOUBLE PRECISION,
                        cost DOUBLE PRECISION,
                        fee DOUBLE PRECISION,
                        realized_pnl DOUBLE PRECISION,
                        strategy TEXT,
                        status TEXT
                    )
                """)
                
                # Crear hipertabla para trades
                cur.execute("""
                    SELECT create_hypertable('trades', 'timestamp', 
                                            if_not_exists => TRUE, 
                                            create_default_indexes => TRUE)
                """)
                
                # Tabla para rendimiento (performance)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS performance (
                        timestamp TIMESTAMPTZ NOT NULL,
                        equity DOUBLE PRECISION,
                        balance DOUBLE PRECISION,
                        drawdown DOUBLE PRECISION,
                        daily_return DOUBLE PRECISION,
                        sharpe_ratio DOUBLE PRECISION,
                        sortino_ratio DOUBLE PRECISION,
                        win_rate DOUBLE PRECISION
                    )
                """)
                
                # Crear hipertabla para performance
                cur.execute("""
                    SELECT create_hypertable('performance', 'timestamp', 
                                            if_not_exists => TRUE, 
                                            create_default_indexes => TRUE)
                """)
                
                # Tabla para checkpoints
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS system_checkpoints (
                        id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        data JSONB,
                        timestamp TIMESTAMPTZ NOT NULL
                    )
                """)
                
                # Confirmar cambios
                conn.commit()
                
                self.logger.info("Hipertablas de TimescaleDB configuradas correctamente")
                return True
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error configurando hipertablas de TimescaleDB: {str(e)}")
            return False
            
        finally:
            conn.close()
    
    def bulk_insert_market_data(self, data: List[Dict[str, Any]], symbol: str) -> bool:
        """
        Insertar datos de mercado en bulk.
        
        Args:
            data: Lista de datos OHLCV
            symbol: Símbolo
            
        Returns:
            True si se insertó correctamente
        """
        if not data:
            return False
        
        # Crear conexión
        conn = self.connect()
        if not conn:
            return False
        
        try:
            # Preparar datos para inserción
            values = []
            for item in data:
                # Convertir timestamp a formato adecuado
                if 'timestamp' in item:
                    timestamp = item['timestamp']
                    if isinstance(timestamp, int):
                        # Convertir de milisegundos a segundos si es necesario
                        if timestamp > 1600000000000:  # Timestamp en milisegundos
                            timestamp = timestamp / 1000
                        # Convertir a ISO
                        timestamp = datetime.fromtimestamp(timestamp).isoformat()
                elif 'time' in item:
                    timestamp = item['time']
                else:
                    timestamp = datetime.now().isoformat()
                
                # Extraer valores OHLCV
                open_price = item.get('open', 0.0)
                high = item.get('high', 0.0)
                low = item.get('low', 0.0)
                close = item.get('close', 0.0)
                volume = item.get('volume', 0.0)
                
                values.append((timestamp, symbol, open_price, high, low, close, volume))
            
            # Insertar datos
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO market_data 
                    (timestamp, symbol, open, high, low, close, volume)
                    VALUES %s
                    ON CONFLICT (timestamp, symbol) DO UPDATE 
                    SET open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                    """,
                    values,
                    template="(%s, %s, %s, %s, %s, %s, %s)"
                )
            
            # Confirmar cambios
            conn.commit()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['successful_operations'] += 1
            
            return True
                
        except Exception as e:
            # Rollback
            conn.rollback()
            
            # Actualizar estadísticas
            with self.lock:
                self.stats['failed_operations'] += 1
                self.stats['last_error'] = str(e)
                
            self.logger.error(f"Error insertando datos de mercado en bulk: {str(e)}")
            return False
            
        finally:
            # Cerrar conexión
            conn.close()
            with self.lock:
                self.stats['connections_closed'] += 1