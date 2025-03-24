"""
Prueba Express de Intensidad de Base de Datos - Sistema Genesis

Este script realiza pruebas de intensidad en la base de datos del Sistema Genesis
con una versión más rápida y directa para resultados inmediatos.
"""

import os
import time
import json
import random
import asyncio
import logging
from enum import Enum, auto
from typing import Dict, List, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("db_intensity_test_express")
file_handler = logging.FileHandler("logs/db_intensity_express.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Definiciones básicas
class TestIntensity(Enum):
    """Define niveles de intensidad para las pruebas."""
    BASIC = 10      # Básico - 100 operaciones
    MEDIUM = 100    # Medio - 1,000 operaciones 
    HIGH = 500      # Alto - 5,000 operaciones
    EXTREME = 1000  # Extremo - 10,000+ operaciones

class OperationType(Enum):
    """Tipos de operaciones para pruebas."""
    READ = auto()
    WRITE = auto()
    UPDATE = auto()
    DELETE = auto()
    TRANSACTION = auto()

class TestResult:
    """Almacena resultados de pruebas para análisis."""
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
        return self

    def add_metric(self, name: str, value: Any):
        """Añadir una métrica."""
        self.metrics[name] = value
        return self

    def add_error(self, error: Exception):
        """Añadir un error."""
        self.errors.append(str(error))
        return self

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

# Simulador de base de datos simplificado para prueba ultra-rápida
class ExpressDBSimulator:
    """Simulador de base de datos ultra-rápido para pruebas express."""
    def __init__(self, failure_rate=0.01):
        self.data = {}
        self.query_count = 0
        self.write_count = 0
        self.error_count = 0
        self.failure_rate = failure_rate
        self.logger = logging.getLogger("ExpressDB")
        self.logger.info("Simulador de base de datos express iniciado")
        
    async def connect(self):
        """Simular conexión a la base de datos."""
        await asyncio.sleep(0.001)
        return True
        
    async def execute(self, operation_type: OperationType, data: Any = None):
        """
        Ejecutar operación simulada.
        
        Args:
            operation_type: Tipo de operación
            data: Datos asociados a la operación
            
        Returns:
            Resultado de la operación
        """
        # Simular latencia mínima
        await asyncio.sleep(0.0001)
        
        # Simular fallos aleatorios
        if random.random() < self.failure_rate:
            self.error_count += 1
            self.logger.debug(f"Error simulado en operación {operation_type.name}")
            raise Exception(f"Error simulado en operación {operation_type.name}")
        
        self.query_count += 1
        
        if operation_type in [OperationType.WRITE, OperationType.UPDATE]:
            self.write_count += 1
            if data and isinstance(data, dict) and "key" in data:
                self.data[data["key"]] = data.get("value", True)
        
        return {
            "success": True,
            "operation_type": operation_type.name,
            "timestamp": time.time()
        }
    
    async def simulate_failure(self, duration=0.5):
        """Simular fallo temporal en la base de datos."""
        old_rate = self.failure_rate
        self.failure_rate = 0.8
        self.logger.warning(f"Simulando fallo por {duration}s")
        await asyncio.sleep(duration)
        self.failure_rate = old_rate
        return True
    
    def get_stats(self):
        """Obtener estadísticas del simulador."""
        return {
            "queries": self.query_count,
            "writes": self.write_count,
            "errors": self.error_count,
            "data_size": len(self.data)
        }

async def run_intensity_test(intensity: TestIntensity) -> TestResult:
    """
    Ejecutar prueba de intensidad express.
    
    Args:
        intensity: Nivel de intensidad
        
    Returns:
        Resultado de la prueba
    """
    result = TestResult(f"Express Test - {intensity.name}")
    logger.info(f"Iniciando prueba express con intensidad {intensity.name}")
    
    try:
        # Crear simulador
        db = ExpressDBSimulator(failure_rate=0.005)
        await db.connect()
        
        # Determinar parámetros según intensidad
        operations_factor = {
            TestIntensity.BASIC: 1,
            TestIntensity.MEDIUM: 10,
            TestIntensity.HIGH: 50,
            TestIntensity.EXTREME: 100
        }
        
        # Configurar parámetros de la prueba
        total_operations = 100 * operations_factor[intensity]
        max_concurrency = min(500, total_operations // 2)
        
        logger.info(f"Ejecutando {total_operations} operaciones con concurrencia máxima de {max_concurrency}")
        
        # Semáforo para controlar concurrencia
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # Contador de operaciones exitosas
        success_count = 0
        error_count = 0
        operation_latency = []
        
        # Función para ejecutar una operación
        async def perform_operation(i: int):
            async with semaphore:
                start_time = time.time()
                try:
                    # Elegir tipo de operación
                    op_type = random.choice(list(OperationType))
                    
                    # Preparar datos según tipo
                    data = None
                    if op_type in [OperationType.WRITE, OperationType.UPDATE]:
                        data = {
                            "key": f"key_{i % 1000}",
                            "value": {
                                "id": i,
                                "timestamp": time.time(),
                                "data": f"data_{i}"
                            }
                        }
                    
                    # Ejecutar operación
                    await db.execute(op_type, data)
                    
                    # Registrar éxito
                    nonlocal success_count
                    success_count += 1
                    
                    # Registrar latencia
                    latency = time.time() - start_time
                    operation_latency.append(latency)
                    
                    return {"success": True, "operation": i, "type": op_type.name, "latency": latency}
                    
                except Exception as e:
                    # Registrar error
                    nonlocal error_count
                    error_count += 1
                    
                    # Registrar latencia hasta el error
                    latency = time.time() - start_time
                    operation_latency.append(latency)
                    
                    return {"success": False, "operation": i, "error": str(e), "latency": latency}
        
        # Crear todas las tareas
        tasks = [perform_operation(i) for i in range(total_operations)]
        
        # Ejecutar en grupos para monitoreo
        chunk_size = min(1000, max(100, total_operations // 5))
        all_results = []
        
        for i in range(0, len(tasks), chunk_size):
            # Obtener chunk actual
            chunk = tasks[i:i+chunk_size]
            chunk_size_actual = len(chunk)
            
            logger.info(f"Procesando chunk {i//chunk_size + 1}/{(len(tasks) + chunk_size - 1)//chunk_size} ({chunk_size_actual} operaciones)")
            
            # Simular fallo cada 3 chunks para intensidades altas
            if i > 0 and i % (chunk_size * 3) == 0 and intensity in [TestIntensity.HIGH, TestIntensity.EXTREME]:
                logger.warning(f"Induciendo fallo simulado en chunk {i//chunk_size + 1}")
                await db.simulate_failure(0.2)
            
            # Ejecutar chunk
            start_chunk = time.time()
            chunk_results = await asyncio.gather(*chunk)
            chunk_duration = time.time() - start_chunk
            
            # Medir rendimiento
            ops_per_second = chunk_size_actual / chunk_duration if chunk_duration > 0 else 0
            
            # Almacenar resultados
            all_results.extend(chunk_results)
            
            # Mostrar progreso
            logger.info(f"Progreso: {min(i + chunk_size, total_operations)}/{total_operations} ({ops_per_second:.2f} ops/s)")
        
        # Obtener estadísticas
        db_stats = db.get_stats()
        
        # Calcular métricas
        avg_latency = sum(operation_latency) / len(operation_latency) if operation_latency else 0
        success_rate = success_count / total_operations if total_operations > 0 else 0
        throughput = total_operations / result.duration() if result.duration() > 0 else 0
        
        # Actualizar resultado
        result.add_metric("total_operations", total_operations)
        result.add_metric("success_count", success_count)
        result.add_metric("error_count", error_count)
        result.add_metric("success_rate", success_rate)
        result.add_metric("avg_latency", avg_latency)
        result.add_metric("throughput", throughput)
        result.add_metric("max_concurrency", max_concurrency)
        result.add_metric("db_queries", db_stats["queries"])
        result.add_metric("db_writes", db_stats["writes"])
        result.add_metric("db_errors", db_stats["errors"])
        
        # Determinar éxito
        success_threshold = 0.95 if intensity == TestIntensity.BASIC else 0.9
        test_success = success_rate >= success_threshold
        
        # Finalizar prueba
        result.finish(test_success)
        
        # Mostrar resultado
        logger.info(f"Prueba completada: {'ÉXITO' if test_success else 'FALLO'}")
        logger.info(f"Total operaciones: {total_operations}")
        logger.info(f"Operaciones exitosas: {success_count} ({success_rate:.2%})")
        logger.info(f"Errores: {error_count}")
        logger.info(f"Latencia promedio: {avg_latency:.6f}s")
        logger.info(f"Throughput: {throughput:.2f} ops/s")
        logger.info(f"Duración: {result.duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error en prueba express: {e}")
        result.add_error(e)
        result.finish(False)
    
    return result

async def run_all_intensities():
    """Ejecutar pruebas con todas las intensidades."""
    os.makedirs("logs", exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("INICIANDO PRUEBAS EXPRESS DE INTENSIDAD")
    logger.info("=" * 80)
    
    results = {}
    intensities = list(TestIntensity)
    
    for intensity in intensities:
        logger.info(f"\n\nPRUEBA CON INTENSIDAD {intensity.name}")
        logger.info("=" * 50)
        
        result = await run_intensity_test(intensity)
        results[intensity.name] = result.to_dict()
        
        # Guardar resultado individual
        with open(f"test_db_express_{intensity.name.lower()}_results.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Resultado guardado para intensidad {intensity.name}")
    
    # Guardar resultados completos
    with open("test_db_express_all_results.json", "w") as f:
        json.dump({
            "timestamp": time.time(),
            "results": results
        }, f, indent=2)
    
    logger.info("\n\nRESUMEN FINAL")
    logger.info("=" * 50)
    
    for intensity, result in results.items():
        status = "ÉXITO" if result["success"] else "FALLO"
        logger.info(f"{intensity}: {status} - {result['metrics']['success_rate']:.2%} - {result['metrics']['throughput']:.2f} ops/s")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    try:
        asyncio.run(run_all_intensities())
    except KeyboardInterrupt:
        logger.warning("Prueba interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error en ejecución principal: {e}")
        import traceback
        traceback.print_exc()