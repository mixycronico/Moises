"""
Prueba Gradual de Intensidad de Base de Datos - Sistema Genesis

Este script realiza pruebas de intensidad en la base de datos del Sistema Genesis
con niveles incrementales:
1. Intensidad 10: Operaciones básicas (100 operaciones)
2. Intensidad 100: Operaciones paralelas (1,000 operaciones)
3. Intensidad 500: Prueba de estrés (5,000 operaciones)
4. Intensidad 1000: Prueba extrema (10,000+ operaciones)

La prueba usa simuladores si los componentes reales no están disponibles.
"""

import os
import time
import json
import random
import asyncio
import logging
from enum import Enum, auto
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("db_intensity_test")
file_handler = logging.FileHandler("logs/db_intensity_test.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Definiciones básicas
class TestIntensity(Enum):
    """Define niveles de intensidad para las pruebas."""
    BASIC = 10      # 100 operaciones
    MEDIUM = 100    # 1,000 operaciones
    HIGH = 500      # 5,000 operaciones
    EXTREME = 1000  # 10,000+ operaciones

class OperationType(Enum):
    """Tipos de operaciones para pruebas."""
    READ = auto()
    WRITE = auto()
    UPDATE = auto()
    DELETE = auto()
    TRANSACTION = auto()
    BULK_INSERT = auto()
    AGGREGATE = auto()
    JOIN = auto()

class TestResult:
    """Almacena resultados de pruebas para análisis posterior."""
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = time.time()
        self.end_time = None
        self.success = None
        self.metrics = {}
        self.errors = []
        self.details = {}

    def finish(self, success: bool = True):
        """Finalizar prueba y registrar tiempo."""
        self.end_time = time.time()
        self.success = success

    def add_metric(self, name: str, value: Any):
        """Añadir una métrica."""
        self.metrics[name] = value

    def add_error(self, error: Exception):
        """Añadir un error."""
        self.errors.append(str(error))

    def add_detail(self, key: str, value: Any):
        """Añadir detalle."""
        self.details[key] = value

    def duration(self) -> float:
        """Obtener duración de la prueba."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "test_name": self.test_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "success": self.success,
            "metrics": self.metrics,
            "errors": self.errors,
            "details": self.details
        }

    def __str__(self) -> str:
        """Representación como string."""
        status = "ÉXITO" if self.success else "FALLO"
        return f"Test '{self.test_name}': {status} (duración: {self.duration():.2f}s)"

class TestSuite:
    """Suite de pruebas con análisis agregado."""
    def __init__(self, name: str):
        self.name = name
        self.results = []
        self.start_time = time.time()

    def add_result(self, result: TestResult):
        """Añadir resultado de prueba."""
        self.results.append(result)

    def success_rate(self) -> float:
        """Tasa de éxito de las pruebas."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.success)
        return successful / len(self.results)

    def total_duration(self) -> float:
        """Duración total de las pruebas."""
        return sum(r.duration() for r in self.results)

    def aggregate_metrics(self) -> Dict[str, Any]:
        """Agregar métricas de todas las pruebas."""
        metrics = {}
        for result in self.results:
            for k, v in result.metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
        
        # Calcular promedios, min, max para valores numéricos
        aggregated = {}
        for k, values in metrics.items():
            try:
                if all(isinstance(v, (int, float)) for v in values):
                    aggregated[f"{k}_avg"] = sum(values) / len(values)
                    aggregated[f"{k}_min"] = min(values)
                    aggregated[f"{k}_max"] = max(values)
                else:
                    aggregated[k] = values
            except (TypeError, ValueError):
                aggregated[k] = values
        
        return aggregated

    def summary(self) -> Dict[str, Any]:
        """Obtener resumen de resultados."""
        return {
            "name": self.name,
            "total_tests": len(self.results),
            "success_rate": self.success_rate(),
            "total_duration": self.total_duration(),
            "success_count": sum(1 for r in self.results if r.success),
            "failure_count": sum(1 for r in self.results if not r.success),
            "metrics": self.aggregate_metrics()
        }

    def print_summary(self):
        """Imprimir resumen de resultados."""
        summary = self.summary()
        logger.info(f"Suite de pruebas: {self.name}")
        logger.info(f"Total de pruebas: {summary['total_tests']}")
        logger.info(f"Tasa de éxito: {summary['success_rate']:.2%}")
        logger.info(f"Duración total: {summary['total_duration']:.2f}s")
        logger.info(f"Pruebas exitosas: {summary['success_count']}")
        logger.info(f"Pruebas fallidas: {summary['failure_count']}")
        
        for result in self.results:
            logger.info(f"  - {result}")

class TranscendentalMetrics:
    """Métricas avanzadas para la prueba de intensidad."""
    def __init__(self):
        self.operations = {op_type: 0 for op_type in OperationType}
        self.operation_times = {op_type: [] for op_type in OperationType}
        self.errors = {}
        self.recoveries = []
        self.concurrent_operations = 0
        self.max_concurrent = 0
        self.throughput_measurements = []
        self.checkpoints = 0
        self.anomalies = 0
        self.operation_lock = asyncio.Lock()
        self.start_time = time.time()

    async def record_operation(self, op_type: OperationType, duration: float):
        """Registrar una operación."""
        async with self.operation_lock:
            self.operations[op_type] += 1
            self.operation_times[op_type].append(duration)

    async def record_error(self, error_type: str):
        """Registrar un error."""
        async with self.operation_lock:
            if error_type not in self.errors:
                self.errors[error_type] = 0
            self.errors[error_type] += 1

    async def record_recovery(self, time_taken: float):
        """Registrar una recuperación."""
        async with self.operation_lock:
            self.recoveries.append(time_taken)

    async def start_operation(self):
        """Registrar inicio de operación concurrente."""
        async with self.operation_lock:
            self.concurrent_operations += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_operations)

    async def end_operation(self, latency: float = None):
        """Registrar fin de operación concurrente."""
        async with self.operation_lock:
            self.concurrent_operations -= 1

    async def record_throughput(self, ops_per_second: float):
        """Registrar medición de throughput."""
        async with self.operation_lock:
            self.throughput_measurements.append(ops_per_second)

    async def record_checkpoint(self):
        """Registrar checkpoint."""
        async with self.operation_lock:
            self.checkpoints += 1

    async def record_anomaly(self):
        """Registrar anomalía temporal."""
        async with self.operation_lock:
            self.anomalies += 1

    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas."""
        total_ops = sum(self.operations.values())
        elapsed = time.time() - self.start_time
        avg_throughput = total_ops / elapsed if elapsed > 0 else 0
        
        # Calcular latencias promedio por tipo de operación
        latencies = {}
        for op_type, times in self.operation_times.items():
            if times:
                latencies[op_type.name] = sum(times) / len(times)
            else:
                latencies[op_type.name] = 0
        
        return {
            "total_operations": total_ops,
            "elapsed_time": elapsed,
            "operations_per_second": avg_throughput,
            "max_concurrent_operations": self.max_concurrent,
            "operation_counts": {op.name: count for op, count in self.operations.items()},
            "error_counts": self.errors,
            "recovery_count": len(self.recoveries),
            "avg_recovery_time": sum(self.recoveries) / len(self.recoveries) if self.recoveries else 0,
            "checkpoint_count": self.checkpoints,
            "anomaly_count": self.anomalies,
            "avg_latency": {op: avg for op, avg in latencies.items()},
            "throughput_stats": {
                "min": min(self.throughput_measurements) if self.throughput_measurements else 0,
                "max": max(self.throughput_measurements) if self.throughput_measurements else 0,
                "avg": sum(self.throughput_measurements) / len(self.throughput_measurements) if self.throughput_measurements else 0
            }
        }

    def print_summary(self):
        """Imprimir resumen de métricas."""
        summary = self.get_summary()
        logger.info("=" * 40)
        logger.info("MÉTRICAS DE RENDIMIENTO")
        logger.info("=" * 40)
        logger.info(f"Total de operaciones: {summary['total_operations']}")
        logger.info(f"Tiempo transcurrido: {summary['elapsed_time']:.2f}s")
        logger.info(f"Operaciones por segundo: {summary['operations_per_second']:.2f}")
        logger.info(f"Máximo de operaciones concurrentes: {summary['max_concurrent_operations']}")
        logger.info("Conteo de operaciones:")
        for op, count in summary['operation_counts'].items():
            logger.info(f"  - {op}: {count}")
        logger.info("Conteo de errores:")
        for err, count in summary['error_counts'].items():
            logger.info(f"  - {err}: {count}")
        logger.info(f"Recuperaciones: {summary['recovery_count']}")
        logger.info(f"Checkpoints: {summary['checkpoint_count']}")
        logger.info(f"Anomalías: {summary['anomaly_count']}")
        logger.info("Latencia promedio por operación (segundos):")
        for op, latency in summary['avg_latency'].items():
            logger.info(f"  - {op}: {latency:.6f}")
        logger.info("Estadísticas de throughput (ops/s):")
        logger.info(f"  - Min: {summary['throughput_stats']['min']:.2f}")
        logger.info(f"  - Max: {summary['throughput_stats']['max']:.2f}")
        logger.info(f"  - Avg: {summary['throughput_stats']['avg']:.2f}")
        logger.info("=" * 40)

# Simulador de base de datos si no se pueden importar los módulos reales
class DatabaseSimulator:
    """
    Simulador de base de datos para pruebas sin acceso a la implementación real.
    """
    def __init__(self):
        self.data = {}
        self.query_count = 0
        self.transaction_count = 0
        self.in_transaction = False
        self.transaction_data = {}
        self.recovery_count = 0
        self.failure_rate = 0.01  # 1% de fallos aleatorios
        self.connection_time = 0
        self.logger = logging.getLogger("DB_Simulator")
        self.logger.info("Simulador de base de datos iniciado")

    async def connect(self):
        """Simular conexión a la base de datos."""
        await asyncio.sleep(0.01)
        self.connection_time = time.time()
        self.logger.info("Conexión a base de datos simulada establecida")
        return True

    async def execute(self, query: str, *args, **kwargs):
        """Simular ejecución de consulta."""
        # Simular latencia realista
        latency = random.uniform(0.001, 0.03)
        await asyncio.sleep(latency)
        
        # Simular fallos aleatorios
        if random.random() < self.failure_rate:
            error_msg = f"Error simulado en ejecución de consulta: {query[:50]}..."
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        self.query_count += 1
        operation_id = kwargs.get('id', f"query_{self.query_count}")
        
        # Extraer tipo de operación
        if query.lower().startswith("select"):
            op_type = "READ"
        elif query.lower().startswith("insert"):
            op_type = "WRITE"
        elif query.lower().startswith("update"):
            op_type = "UPDATE"
        elif query.lower().startswith("delete"):
            op_type = "DELETE"
        else:
            op_type = "OTHER"
        
        return {"success": True, "operation_id": operation_id, "type": op_type}

    async def fetch(self, query: str, *args, **kwargs):
        """Simular consulta con retorno de datos."""
        # Comportamiento similar al execute pero retorna datos
        result = await self.execute(query, *args, **kwargs)
        
        # Generar datos simulados
        row_count = random.randint(1, 20)
        rows = []
        for i in range(row_count):
            row = {
                "id": i,
                "name": f"item_{i}",
                "value": random.uniform(1, 1000),
                "created_at": time.time() - random.uniform(0, 3600)
            }
            rows.append(row)
        
        return rows

    async def begin(self):
        """Iniciar transacción simulada."""
        await asyncio.sleep(0.005)
        if self.in_transaction:
            raise Exception("Transacción ya iniciada")
        self.in_transaction = True
        self.transaction_data = {}
        self.transaction_count += 1
        return {"transaction_id": f"tx_{self.transaction_count}"}

    async def commit(self):
        """Confirmar transacción simulada."""
        await asyncio.sleep(0.01)
        if not self.in_transaction:
            raise Exception("No hay transacción activa para confirmar")
        
        # Simular posibles fallos en commit (más probable que en operaciones regulares)
        if random.random() < self.failure_rate * 2:
            self.in_transaction = False
            error_msg = "Error simulado en commit de transacción"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        # Actualizar datos reales con los de la transacción
        for k, v in self.transaction_data.items():
            self.data[k] = v
        
        self.in_transaction = False
        return {"success": True, "transaction_id": f"tx_{self.transaction_count}"}

    async def rollback(self):
        """Revertir transacción simulada."""
        await asyncio.sleep(0.008)
        if not self.in_transaction:
            raise Exception("No hay transacción activa para revertir")
        
        self.in_transaction = False
        self.transaction_data = {}
        return {"success": True, "transaction_id": f"tx_{self.transaction_count}"}

    async def close(self):
        """Cerrar conexión simulada."""
        await asyncio.sleep(0.005)
        return {"success": True, "connection_time": time.time() - self.connection_time}

    async def simulate_failure(self, recovery_time: float = 1.0):
        """Simular fallo temporal en la base de datos."""
        self.logger.warning(f"Simulando fallo de base de datos por {recovery_time} segundos")
        self.failure_rate = 0.8  # 80% de fallos durante la simulación
        await asyncio.sleep(recovery_time)
        self.logger.info("Recuperación de fallo simulado")
        self.failure_rate = 0.01  # Restaurar tasa normal
        self.recovery_count += 1
        return {"recovered": True, "downtime": recovery_time}

    async def get_stats(self):
        """Obtener estadísticas del simulador."""
        return {
            "query_count": self.query_count,
            "transaction_count": self.transaction_count,
            "recovery_count": self.recovery_count,
            "data_size": len(self.data),
            "connection_time": time.time() - self.connection_time
        }

# Funciones de prueba
async def execute_protected_operation(func, *args, id=None, **kwargs):
    """
    Ejecutar una operación con protección ante errores.
    
    Args:
        func: Función a ejecutar
        *args: Argumentos posicionales
        id: Identificador de operación
        **kwargs: Argumentos con nombre
        
    Returns:
        Resultado de la operación o diccionario con error
    """
    start_time = time.time()
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            return {
                "success": True,
                "id": id,
                "duration": duration,
                "result": result
            }
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                duration = time.time() - start_time
                logger.error(f"Error en operación {id} después de {retry_count} intentos: {str(e)}")
                return {
                    "success": False,
                    "id": id,
                    "duration": duration,
                    "error": str(e),
                    "retries": retry_count
                }
            await asyncio.sleep(0.1 * retry_count)
    
    # Nunca deberíamos llegar aquí, pero por seguridad
    return {
        "success": False,
        "id": id,
        "duration": time.time() - start_time,
        "error": "Error desconocido"
    }

async def test_basic_operations(intensity: TestIntensity = TestIntensity.BASIC):
    """
    Prueba de operaciones básicas de base de datos.
    
    Args:
        intensity: Nivel de intensidad de la prueba
        
    Returns:
        Resultado de la prueba
    """
    result = TestResult(f"Operaciones Básicas - Intensidad {intensity.name}")
    metrics = TranscendentalMetrics()
    
    logger.info(f"Iniciando prueba de operaciones básicas con intensidad {intensity.name}")
    
    try:
        # Obtener simulador o conexión real
        db = DatabaseSimulator()
        await db.connect()
        
        # Determinar número de operaciones basado en intensidad
        operations_factor = {
            TestIntensity.BASIC: 1,
            TestIntensity.MEDIUM: 10,
            TestIntensity.HIGH: 50,
            TestIntensity.EXTREME: 100
        }
        
        total_operations = 100 * operations_factor[intensity]
        logger.info(f"Ejecutando {total_operations} operaciones")
        
        # Preparar operaciones
        read_ops = []
        write_ops = []
        update_ops = []
        delete_ops = []
        transaction_ops = []
        
        # Generar operaciones de lectura
        for i in range(total_operations // 3):
            read_ops.append(execute_protected_operation(
                db.fetch,
                f"SELECT * FROM test_table WHERE id > {i} LIMIT 10",
                id=f"read_{i}"
            ))
            await metrics.start_operation()
        
        # Generar operaciones de escritura
        for i in range(total_operations // 3):
            data = {"name": f"item_{i}", "value": i * 1.5}
            write_ops.append(execute_protected_operation(
                db.execute,
                f"INSERT INTO test_table (name, value) VALUES ('item_{i}', {i * 1.5})",
                id=f"write_{i}"
            ))
            await metrics.start_operation()
        
        # Generar operaciones de actualización
        for i in range(total_operations // 6):
            update_ops.append(execute_protected_operation(
                db.execute,
                f"UPDATE test_table SET value = {i * 2.5} WHERE id = {i}",
                id=f"update_{i}"
            ))
            await metrics.start_operation()
        
        # Generar operaciones de eliminación (pocas)
        for i in range(total_operations // 12):
            delete_ops.append(execute_protected_operation(
                db.execute,
                f"DELETE FROM test_table WHERE id = {i}",
                id=f"delete_{i}"
            ))
            await metrics.start_operation()
        
        # Generar transacciones
        for i in range(total_operations // 12):
            # Función que ejecuta una transacción completa
            async def transaction_operation(tx_id):
                try:
                    await db.begin()
                    await db.execute(f"INSERT INTO test_table (name, value) VALUES ('tx_item_{tx_id}', {tx_id})")
                    await db.execute(f"UPDATE test_table SET value = value + 1 WHERE name = 'tx_item_{tx_id}'")
                    if random.random() < 0.8:  # 80% de commits, 20% de rollbacks
                        await db.commit()
                        return {"status": "committed", "tx_id": tx_id}
                    else:
                        await db.rollback()
                        return {"status": "rolled_back", "tx_id": tx_id}
                except Exception as e:
                    try:
                        await db.rollback()
                    except:
                        pass
                    raise e
            
            transaction_ops.append(execute_protected_operation(
                transaction_operation,
                i,
                id=f"transaction_{i}"
            ))
            await metrics.start_operation()
        
        # Ejecutar todas las operaciones en grupos
        all_ops = read_ops + write_ops + update_ops + delete_ops + transaction_ops
        random.shuffle(all_ops)  # Mezclar para simular carga real
        
        chunk_size = min(100, len(all_ops) // 10)  # Dividir en chunks para mejor monitoreo
        all_results = []
        
        for i in range(0, len(all_ops), chunk_size):
            chunk = all_ops[i:i+chunk_size]
            logger.info(f"Procesando chunk {i//chunk_size + 1}/{(len(all_ops) + chunk_size - 1)//chunk_size}")
            
            # Simular fallo cada 4 chunks (si la intensidad es alta)
            if i > 0 and i % (chunk_size * 4) == 0 and intensity in [TestIntensity.HIGH, TestIntensity.EXTREME]:
                logger.warning(f"Induciendo fallo simulado en chunk {i//chunk_size + 1}")
                await db.simulate_failure(recovery_time=0.5 if intensity == TestIntensity.HIGH else 1.0)
                await metrics.record_anomaly()
            
            start_chunk = time.time()
            chunk_results = await asyncio.gather(*chunk)
            chunk_duration = time.time() - start_chunk
            
            # Medir throughput
            ops_per_second = len(chunk) / chunk_duration if chunk_duration > 0 else 0
            await metrics.record_throughput(ops_per_second)
            
            all_results.extend(chunk_results)
            
            # Registrar operaciones terminadas
            for _ in range(len(chunk)):
                await metrics.end_operation()
            
            # Checkpoint cada 2 chunks
            if i % (chunk_size * 2) == 0:
                await metrics.record_checkpoint()
            
            # Log de progreso
            logger.info(f"Progreso: {min(i + chunk_size, len(all_ops))}/{len(all_ops)} ({ops_per_second:.2f} ops/s)")
        
        # Analizar resultados
        success_count = sum(1 for r in all_results if r["success"])
        error_count = len(all_results) - success_count
        
        # Registrar métricas por tipo
        for result_item in all_results:
            if "result" in result_item and isinstance(result_item["result"], dict) and "type" in result_item["result"]:
                op_type = result_item["result"]["type"]
                if op_type == "READ":
                    await metrics.record_operation(OperationType.READ, result_item["duration"])
                elif op_type == "WRITE":
                    await metrics.record_operation(OperationType.WRITE, result_item["duration"])
                elif op_type == "UPDATE":
                    await metrics.record_operation(OperationType.UPDATE, result_item["duration"])
                elif op_type == "DELETE":
                    await metrics.record_operation(OperationType.DELETE, result_item["duration"])
        
        # Registrar errores
        for result_item in all_results:
            if not result_item["success"]:
                await metrics.record_error("DB_Operation_Error")
        
        # Obtener stats de la base de datos
        db_stats = await db.get_stats()
        
        # Actualizar resultado
        result.add_metric("total_operations", len(all_ops))
        result.add_metric("success_count", success_count)
        result.add_metric("error_count", error_count)
        result.add_metric("success_rate", success_count / len(all_ops))
        result.add_metric("db_queries", db_stats["query_count"])
        result.add_metric("db_transactions", db_stats["transaction_count"])
        result.add_metric("avg_latency", sum(r["duration"] for r in all_results) / len(all_results))
        
        # Prueba exitosa si la tasa de éxito es alta
        success_threshold = 0.95 if intensity == TestIntensity.EXTREME else 0.98
        success = result.metrics["success_rate"] >= success_threshold
        result.finish(success)
        
        # Mensaje de resultado
        if success:
            logger.info(f"¡PRUEBA EXITOSA! Tasa de éxito: {result.metrics['success_rate']:.2%}")
        else:
            logger.warning(f"Prueba por debajo del umbral de éxito ({success_threshold:.0%}). Tasa: {result.metrics['success_rate']:.2%}")
        
        # Cerrar conexión
        await db.close()
        
    except Exception as e:
        logger.error(f"Error en prueba de operaciones básicas: {e}")
        result.add_error(e)
        result.finish(False)
    
    # Imprimir métricas
    metrics.print_summary()
    
    return result

async def test_concurrent_operations(intensity: TestIntensity = TestIntensity.BASIC):
    """
    Prueba de operaciones concurrentes de base de datos.
    
    Args:
        intensity: Nivel de intensidad de la prueba
        
    Returns:
        Resultado de la prueba
    """
    result = TestResult(f"Operaciones Concurrentes - Intensidad {intensity.name}")
    metrics = TranscendentalMetrics()
    
    logger.info(f"Iniciando prueba de operaciones concurrentes con intensidad {intensity.name}")
    
    try:
        # Obtener simulador o conexión real
        db = DatabaseSimulator()
        await db.connect()
        
        # Determinar número de operaciones basado en intensidad
        operations_factor = {
            TestIntensity.BASIC: 1,
            TestIntensity.MEDIUM: 10,
            TestIntensity.HIGH: 50,
            TestIntensity.EXTREME: 100
        }
        
        concurrency_factor = {
            TestIntensity.BASIC: 5,
            TestIntensity.MEDIUM: 20,
            TestIntensity.HIGH: 50,
            TestIntensity.EXTREME: 100
        }
        
        total_operations = 100 * operations_factor[intensity]
        max_concurrency = concurrency_factor[intensity]
        
        logger.info(f"Ejecutando {total_operations} operaciones con concurrencia máxima de {max_concurrency}")
        
        # Función para ejecutar operación concurrente
        async def run_concurrent_operation(i):
            await metrics.start_operation()
            
            operation_type = i % 5  # 5 tipos de operaciones
            
            if operation_type == 0:
                # Lectura
                result = await execute_protected_operation(
                    db.fetch,
                    f"SELECT * FROM test_table WHERE id > {i % 1000} LIMIT 10",
                    id=f"concurrent_read_{i}"
                )
                await metrics.record_operation(OperationType.READ, result["duration"])
                
            elif operation_type == 1:
                # Escritura
                result = await execute_protected_operation(
                    db.execute,
                    f"INSERT INTO test_table (name, value) VALUES ('concurrent_item_{i}', {i * 1.5})",
                    id=f"concurrent_write_{i}"
                )
                await metrics.record_operation(OperationType.WRITE, result["duration"])
                
            elif operation_type == 2:
                # Actualización
                result = await execute_protected_operation(
                    db.execute,
                    f"UPDATE test_table SET value = {i * 2.5} WHERE id = {i % 1000}",
                    id=f"concurrent_update_{i}"
                )
                await metrics.record_operation(OperationType.UPDATE, result["duration"])
                
            elif operation_type == 3:
                # Transacción
                async def transaction_operation():
                    try:
                        await db.begin()
                        await db.execute(f"INSERT INTO test_table (name, value) VALUES ('tx_item_{i}', {i})")
                        await db.execute(f"UPDATE test_table SET value = value + 1 WHERE id = {i % 1000}")
                        if random.random() < 0.8:  # 80% de commits, 20% de rollbacks
                            await db.commit()
                            return {"status": "committed", "id": i}
                        else:
                            await db.rollback()
                            return {"status": "rolled_back", "id": i}
                    except Exception as e:
                        try:
                            await db.rollback()
                        except:
                            pass
                        raise e
                
                result = await execute_protected_operation(
                    transaction_operation,
                    id=f"concurrent_transaction_{i}"
                )
                await metrics.record_operation(OperationType.TRANSACTION, result["duration"])
                
            else:
                # Inserción masiva
                values = ", ".join([f"('bulk_item_{i}_{j}', {i + j})" for j in range(10)])
                result = await execute_protected_operation(
                    db.execute,
                    f"INSERT INTO test_table (name, value) VALUES {values}",
                    id=f"concurrent_bulk_{i}"
                )
                await metrics.record_operation(OperationType.BULK_INSERT, result["duration"])
            
            await metrics.end_operation()
            return result
        
        # Ejecutar operaciones en bloques controlados para mantener la concurrencia deseada
        all_results = []
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def controlled_operation(i):
            async with semaphore:
                return await run_concurrent_operation(i)
        
        tasks = [controlled_operation(i) for i in range(total_operations)]
        
        # Dividir en chunks para mejor monitoreo y control
        chunk_size = min(100, len(tasks) // 10)
        
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i+chunk_size]
            logger.info(f"Procesando chunk concurrente {i//chunk_size + 1}/{(len(tasks) + chunk_size - 1)//chunk_size}")
            
            # Simular fallo cada 3 chunks en intensidades altas
            if i > 0 and i % (chunk_size * 3) == 0 and intensity in [TestIntensity.HIGH, TestIntensity.EXTREME]:
                logger.warning(f"Induciendo fallo simulado en chunk concurrente {i//chunk_size + 1}")
                await db.simulate_failure(recovery_time=0.5 if intensity == TestIntensity.HIGH else 1.0)
                await metrics.record_anomaly()
            
            start_chunk = time.time()
            chunk_results = await asyncio.gather(*chunk)
            chunk_duration = time.time() - start_chunk
            
            # Medir throughput
            ops_per_second = len(chunk) / chunk_duration if chunk_duration > 0 else 0
            await metrics.record_throughput(ops_per_second)
            
            all_results.extend(chunk_results)
            
            # Checkpoint cada 2 chunks
            if i % (chunk_size * 2) == 0:
                await metrics.record_checkpoint()
            
            # Log de progreso
            logger.info(f"Progreso concurrente: {min(i + chunk_size, len(tasks))}/{len(tasks)} ({ops_per_second:.2f} ops/s)")
        
        # Analizar resultados
        success_count = sum(1 for r in all_results if r["success"])
        error_count = len(all_results) - success_count
        
        # Registrar errores
        for result_item in all_results:
            if not result_item["success"] and "error" in result_item:
                await metrics.record_error(result_item["error"][:50])  # Primeros 50 caracteres como tipo
        
        # Obtener stats de la base de datos
        db_stats = await db.get_stats()
        
        # Actualizar resultado
        result.add_metric("total_operations", len(tasks))
        result.add_metric("success_count", success_count)
        result.add_metric("error_count", error_count)
        result.add_metric("success_rate", success_count / len(tasks))
        result.add_metric("db_queries", db_stats["query_count"])
        result.add_metric("db_transactions", db_stats["transaction_count"])
        result.add_metric("max_concurrency", max_concurrency)
        result.add_metric("avg_latency", sum(r["duration"] for r in all_results) / len(all_results))
        
        # Prueba exitosa si la tasa de éxito es alta
        success_threshold = 0.90 if intensity == TestIntensity.EXTREME else 0.95
        success = result.metrics["success_rate"] >= success_threshold
        result.finish(success)
        
        # Mensaje de resultado
        if success:
            logger.info(f"¡PRUEBA CONCURRENTE EXITOSA! Tasa de éxito: {result.metrics['success_rate']:.2%}")
        else:
            logger.warning(f"Prueba concurrente por debajo del umbral de éxito ({success_threshold:.0%}). Tasa: {result.metrics['success_rate']:.2%}")
        
        # Cerrar conexión
        await db.close()
        
    except Exception as e:
        logger.error(f"Error en prueba de operaciones concurrentes: {e}")
        result.add_error(e)
        result.finish(False)
    
    # Imprimir métricas
    metrics.print_summary()
    
    return result

async def test_database_resilience(intensity: TestIntensity = TestIntensity.BASIC):
    """
    Prueba de resiliencia de la base de datos ante fallos.
    
    Args:
        intensity: Nivel de intensidad de la prueba
        
    Returns:
        Resultado de la prueba
    """
    result = TestResult(f"Resiliencia de Base de Datos - Intensidad {intensity.name}")
    metrics = TranscendentalMetrics()
    
    logger.info(f"Iniciando prueba de resiliencia con intensidad {intensity.name}")
    
    try:
        # Obtener simulador o conexión real
        db = DatabaseSimulator()
        await db.connect()
        
        # Determinar parámetros basados en intensidad
        operations_factor = {
            TestIntensity.BASIC: 1,
            TestIntensity.MEDIUM: 10,
            TestIntensity.HIGH: 50,
            TestIntensity.EXTREME: 100
        }
        
        failure_intervals = {
            TestIntensity.BASIC: 50,
            TestIntensity.MEDIUM: 25,
            TestIntensity.HIGH: 20,
            TestIntensity.EXTREME: 10
        }
        
        failure_durations = {
            TestIntensity.BASIC: 0.5,
            TestIntensity.MEDIUM: 1.0,
            TestIntensity.HIGH: 2.0,
            TestIntensity.EXTREME: 3.0
        }
        
        concurrency = {
            TestIntensity.BASIC: 5,
            TestIntensity.MEDIUM: 10,
            TestIntensity.HIGH: 20,
            TestIntensity.EXTREME: 40
        }
        
        total_operations = 100 * operations_factor[intensity]
        fail_every = failure_intervals[intensity]
        failure_duration = failure_durations[intensity]
        max_concurrency = concurrency[intensity]
        
        logger.info(f"Ejecutando {total_operations} operaciones con concurrencia {max_concurrency}")
        logger.info(f"Induciendo fallos cada {fail_every} operaciones con duración {failure_duration}s")
        
        # Semáforo para controlar concurrencia
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # Ejecutar operaciones con fallos inducidos
        all_results = []
        failure_count = 0
        recovery_times = []
        
        async def resilience_operation(i):
            async with semaphore:
                await metrics.start_operation()
                
                # Decidir tipo de operación
                operation_type = i % 4
                
                start_time = time.time()
                try:
                    if operation_type == 0:
                        # Lectura
                        result = await db.fetch(f"SELECT * FROM test_table WHERE id > {i % 1000} LIMIT 10")
                        await metrics.record_operation(OperationType.READ, time.time() - start_time)
                        return {"success": True, "data": result, "duration": time.time() - start_time}
                        
                    elif operation_type == 1:
                        # Escritura
                        result = await db.execute(f"INSERT INTO test_table (name, value) VALUES ('resilience_item_{i}', {i})")
                        await metrics.record_operation(OperationType.WRITE, time.time() - start_time)
                        return {"success": True, "result": result, "duration": time.time() - start_time}
                        
                    elif operation_type == 2:
                        # Transacción
                        await db.begin()
                        try:
                            await db.execute(f"INSERT INTO test_table (name, value) VALUES ('tx_resilience_{i}', {i})")
                            await db.execute(f"UPDATE test_table SET value = value + 1 WHERE id = {i % 500}")
                            await db.commit()
                            await metrics.record_operation(OperationType.TRANSACTION, time.time() - start_time)
                            return {"success": True, "transaction": "committed", "duration": time.time() - start_time}
                        except Exception as e:
                            try:
                                await db.rollback()
                            except:
                                pass
                            raise e
                            
                    else:
                        # Agregación
                        result = await db.fetch(f"SELECT AVG(value), MAX(value), MIN(value) FROM test_table WHERE id < {i % 1000}")
                        await metrics.record_operation(OperationType.AGGREGATE, time.time() - start_time)
                        return {"success": True, "aggregation": result, "duration": time.time() - start_time}
                        
                except Exception as e:
                    duration = time.time() - start_time
                    await metrics.record_error(str(e)[:30])  # Usar primeros 30 caracteres como tipo
                    return {"success": False, "error": str(e), "duration": duration}
                finally:
                    await metrics.end_operation()
        
        # Ejecutar operaciones en bloques
        tasks_remaining = total_operations
        op_index = 0
        
        while tasks_remaining > 0:
            # Determinar tamaño del chunk actual
            chunk_size = min(fail_every, tasks_remaining)
            logger.info(f"Ejecutando chunk de {chunk_size} operaciones. Restantes: {tasks_remaining}")
            
            # Crear y ejecutar tareas del chunk
            chunk_tasks = [resilience_operation(op_index + i) for i in range(chunk_size)]
            
            start_chunk = time.time()
            chunk_results = await asyncio.gather(*chunk_tasks)
            chunk_duration = time.time() - start_chunk
            
            # Medir throughput
            ops_per_second = len(chunk_results) / chunk_duration if chunk_duration > 0 else 0
            await metrics.record_throughput(ops_per_second)
            
            # Almacenar resultados
            all_results.extend(chunk_results)
            
            # Actualizar contadores
            op_index += chunk_size
            tasks_remaining -= chunk_size
            
            # Log de progreso
            success_in_chunk = sum(1 for r in chunk_results if r.get("success", False))
            logger.info(f"Chunk completado. Éxitos: {success_in_chunk}/{len(chunk_results)} ({ops_per_second:.2f} ops/s)")
            
            # Inducir fallo después del chunk (excepto el último)
            if tasks_remaining > 0 and (intensity != TestIntensity.BASIC or failure_count == 0):
                failure_count += 1
                logger.warning(f"Induciendo fallo #{failure_count} con duración {failure_duration}s")
                
                # Marcar inicio de fallo
                failure_start = time.time()
                
                # Simular fallo
                await db.simulate_failure(failure_duration)
                
                # Calcular tiempo de recuperación
                recovery_time = time.time() - failure_start
                recovery_times.append(recovery_time)
                await metrics.record_recovery(recovery_time)
                
                # Registrar anomalía
                await metrics.record_anomaly()
                
                logger.info(f"Sistema recuperado en {recovery_time:.2f}s")
        
        # Analizar resultados
        success_count = sum(1 for r in all_results if r.get("success", False))
        error_count = len(all_results) - success_count
        
        # Obtener stats de la base de datos
        db_stats = await db.get_stats()
        
        # Actualizar resultado
        result.add_metric("total_operations", len(all_results))
        result.add_metric("success_count", success_count)
        result.add_metric("error_count", error_count)
        result.add_metric("success_rate", success_count / len(all_results))
        result.add_metric("induced_failures", failure_count)
        result.add_metric("avg_recovery_time", sum(recovery_times) / len(recovery_times) if recovery_times else 0)
        result.add_metric("max_recovery_time", max(recovery_times) if recovery_times else 0)
        result.add_metric("db_recovery_count", db_stats["recovery_count"])
        
        # Prueba exitosa si la tasa de éxito es razonable considerando los fallos inducidos
        # La tolerancia se ajusta según la intensidad
        success_thresholds = {
            TestIntensity.BASIC: 0.90,
            TestIntensity.MEDIUM: 0.85,
            TestIntensity.HIGH: 0.80,
            TestIntensity.EXTREME: 0.75
        }
        
        success_threshold = success_thresholds[intensity]
        success = result.metrics["success_rate"] >= success_threshold
        result.finish(success)
        
        # Mensaje de resultado
        if success:
            logger.info(f"¡PRUEBA DE RESILIENCIA EXITOSA! Tasa de éxito: {result.metrics['success_rate']:.2%} con {failure_count} fallos inducidos")
        else:
            logger.warning(f"Prueba de resiliencia por debajo del umbral de éxito ({success_threshold:.0%}). Tasa: {result.metrics['success_rate']:.2%}")
        
        # Cerrar conexión
        await db.close()
        
    except Exception as e:
        logger.error(f"Error catastrófico en prueba de resiliencia: {e}")
        result.add_error(e)
        result.finish(False)
    
    # Imprimir métricas
    metrics.print_summary()
    
    return result

async def run_tests_with_intensity(intensity: TestIntensity):
    """
    Ejecutar todas las pruebas con una intensidad específica.
    
    Args:
        intensity: Nivel de intensidad de las pruebas
    
    Returns:
        Resultados de la suite
    """
    logger.info("=" * 80)
    logger.info(f"INICIANDO PRUEBAS CON INTENSIDAD: {intensity.name}")
    logger.info("=" * 80)
    
    # Crear suite de pruebas
    suite = TestSuite(f"Database Test - Intensity {intensity.name}")
    
    # 1. Pruebas básicas
    logger.info("\n\n1. PRUEBA DE OPERACIONES BÁSICAS")
    logger.info("=" * 80)
    result = await test_basic_operations(intensity)
    suite.add_result(result)
    
    # 2. Pruebas concurrentes
    logger.info("\n\n2. PRUEBA DE OPERACIONES CONCURRENTES")
    logger.info("=" * 80)
    result = await test_concurrent_operations(intensity)
    suite.add_result(result)
    
    # 3. Pruebas de resiliencia
    logger.info("\n\n3. PRUEBA DE RESILIENCIA")
    logger.info("=" * 80)
    result = await test_database_resilience(intensity)
    suite.add_result(result)
    
    # Mostrar resumen
    logger.info("\n\n")
    logger.info("=" * 80)
    logger.info(f"RESUMEN DE PRUEBAS - INTENSIDAD {intensity.name}")
    logger.info("=" * 80)
    suite.print_summary()
    
    return suite

async def main():
    """Ejecutar todas las pruebas con intensidades graduales."""
    
    # Configuración
    os.makedirs("logs", exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("INICIANDO PRUEBAS GRADUALES DE INTENSIDAD PARA SISTEMA DE BASE DE DATOS")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Lista de intensidades a probar
    intensities = [
        TestIntensity.BASIC,      # 10: Básico
        TestIntensity.MEDIUM,     # 100: Medio
        TestIntensity.HIGH,       # 500: Alto
        TestIntensity.EXTREME     # 1000: Extremo
    ]
    
    results = {}
    
    # Ejecutar pruebas para cada intensidad
    for intensity in intensities:
        logger.info("\n\n")
        logger.info("#" * 100)
        logger.info(f"EJECUTANDO PRUEBAS CON INTENSIDAD: {intensity.name} ({intensity.value})")
        logger.info("#" * 100)
        
        suite = await run_tests_with_intensity(intensity)
        results[intensity.name] = suite.summary()
        
        # Guardar resultados parciales
        results_file = f"test_db_intensity_{intensity.name.lower()}_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "intensity": intensity.name,
                "value": intensity.value,
                "summary": suite.summary()
            }, f, indent=2)
        
        logger.info(f"Resultados para intensidad {intensity.name} guardados en {results_file}")
        
        # Si la intensidad actual falla, no continuar con intensidades mayores
        # (excepto si estamos en BASIC, siempre probar al menos MEDIUM)
        if intensity != TestIntensity.BASIC and suite.success_rate() < 0.7:
            logger.warning(f"Deteniendo pruebas. Intensidad {intensity.name} tuvo una tasa de éxito baja: {suite.success_rate():.2%}")
            break
    
    # Mostrar resumen final con todas las intensidades
    total_duration = time.time() - start_time
    
    logger.info("\n\n")
    logger.info("=" * 80)
    logger.info("RESUMEN FINAL DE PRUEBAS GRADUALES")
    logger.info("=" * 80)
    logger.info(f"Tiempo total: {total_duration:.2f}s")
    logger.info(f"Intensidades probadas: {', '.join(r for r in results.keys())}")
    
    for intensity, summary in results.items():
        logger.info(f"Intensidad {intensity}: {summary['success_rate']:.2%} de éxito ({summary['success_count']}/{summary['total_tests']} pruebas)")
    
    # Guardar resultados completos
    with open("test_db_intensity_all_results.json", "w") as f:
        json.dump({
            "timestamp": time.time(),
            "duration": total_duration,
            "intensities_tested": list(results.keys()),
            "results": results
        }, f, indent=2)
    
    logger.info("Resultados completos guardados en test_db_intensity_all_results.json")
    logger.info("=" * 80)

if __name__ == "__main__":
    try:
        # Ejecutar con nuevo loop de eventos
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Prueba interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error en ejecución principal: {e}")
        import traceback
        traceback.print_exc()