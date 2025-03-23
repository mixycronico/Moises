"""
Script de prueba para el Sistema Divino de procesamiento de tareas de base de datos.

Este script demuestra las capacidades del Sistema Divino, incluyendo:
- Configuración e inicialización del sistema
- Creación de tareas con diferentes prioridades
- Uso de transacciones divinas
- Monitoreo de rendimiento

El sistema funcionará incluso si Redis y RabbitMQ no están disponibles,
utilizando una cola en memoria como respaldo.
"""

import asyncio
import time
import logging
import random
from datetime import datetime
from typing import Dict, Any, List

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_divine")

# Importar componentes divinos
from genesis.db.divine_integrator import (
    initialize_divine_system,
    get_divine_stats,
    divine_task,
    critical_task,
    high_priority_task,
    low_priority_task,
    background_task,
    divine_transaction,
    DivineDatabaseOperations
)

# Simulación de funciones de base de datos
async def simular_db_session():
    """Simular una sesión de base de datos."""
    return SimulatedDBSession()
    
class SimulatedDBSession:
    """Simulación de sesión de base de datos para pruebas."""
    
    async def execute(self, query, *args):
        """Simular ejecución de consulta."""
        logger.info(f"Ejecutando: {query}")
        logger.info(f"Args: {args}")
        # Simular tiempo de procesamiento
        await asyncio.sleep(random.uniform(0.01, 0.05))
        return {"rows_affected": random.randint(1, 10)}
        
    async def fetch(self, query, *args):
        """Simular lectura de datos."""
        logger.info(f"Consultando: {query}")
        logger.info(f"Args: {args}")
        # Simular tiempo de procesamiento
        await asyncio.sleep(random.uniform(0.01, 0.05))
        # Crear resultados simulados
        rows = []
        for i in range(random.randint(2, 5)):
            row = {
                "id": i,
                "name": f"Item {i}",
                "value": random.random() * 100
            }
            rows.append(SimulatedRow(row))
        return rows
        
    async def begin(self):
        """Iniciar transacción simulada."""
        logger.info("Iniciando transacción")
        return SimulatedTransaction()
        
    async def commit(self):
        """Confirmar cambios simulados."""
        logger.info("Commit")
        await asyncio.sleep(0.01)
        
    async def rollback(self):
        """Revertir cambios simulados."""
        logger.info("Rollback")
        await asyncio.sleep(0.01)
        
    async def close(self):
        """Cerrar sesión simulada."""
        logger.info("Cerrando sesión")
        

class SimulatedTransaction:
    """Simulación de transacción de base de datos."""
    
    async def commit(self):
        """Confirmar transacción simulada."""
        logger.info("Commit de transacción")
        await asyncio.sleep(0.01)
        
    async def rollback(self):
        """Revertir transacción simulada."""
        logger.info("Rollback de transacción")
        await asyncio.sleep(0.01)
        

class SimulatedRow(dict):
    """Simulación de fila de resultados de base de datos."""
    
    def __init__(self, data):
        super().__init__(data)
        
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"No attribute {name}")

# Sobrescribir la función de get_db_session para la prueba
import genesis.db.base
genesis.db.base.get_db_session = simular_db_session

# Funciones de ejemplo para probar el sistema
@critical_task()
async def tarea_critica(data: Dict[str, Any]) -> Dict[str, Any]:
    """Tarea crítica de máxima prioridad."""
    logger.info(f"Ejecutando tarea crítica con datos: {data}")
    # Simular operación importante
    await asyncio.sleep(0.02)
    return {"status": "success", "processed": data}

@high_priority_task()
async def tarea_alta_prioridad(param1: str, param2: int) -> Dict[str, Any]:
    """Tarea de alta prioridad."""
    logger.info(f"Ejecutando tarea de alta prioridad: {param1}, {param2}")
    # Simular operación importante
    await asyncio.sleep(0.03)
    return {"result": param1 * param2}

@divine_task(priority=5)
async def tarea_normal(operation: str) -> str:
    """Tarea de prioridad normal."""
    logger.info(f"Ejecutando tarea normal: {operation}")
    # Simular operación importante
    await asyncio.sleep(0.01)
    return f"Completado: {operation}"

@low_priority_task()
async def tarea_baja_prioridad() -> None:
    """Tarea de baja prioridad."""
    logger.info("Ejecutando tarea de baja prioridad")
    # Simular operación de larga duración
    await asyncio.sleep(0.1)

@background_task()
async def tarea_segundo_plano(data: List[Any]) -> None:
    """Tarea en segundo plano."""
    logger.info(f"Procesando datos en segundo plano: {len(data)} elementos")
    # Simular procesamiento largo
    await asyncio.sleep(0.2)
    logger.info("Procesamiento en segundo plano completado")

async def prueba_divina_transaction():
    """Probar transacciones divinas."""
    logger.info("Probando transacción divina")
    
    try:
        async with divine_transaction():
            # Operaciones que deben ser atómicas
            await tarea_alta_prioridad("valor", 5)
            await tarea_normal("operación dentro de transacción")
            
            # Todo se completa correctamente
            logger.info("Transacción completada correctamente")
            
    except Exception as e:
        logger.error(f"Error en transacción: {e}")
    
    # Probar transacción con error
    try:
        async with divine_transaction():
            await tarea_alta_prioridad("valor", 5)
            
            # Simular error
            raise ValueError("Error simulado en transacción")
            
            # Esta parte no debería ejecutarse
            await tarea_normal("no debería ejecutarse")
            
    except ValueError:
        logger.info("Transacción con error controlada correctamente (rollback automático)")

async def prueba_operaciones_db():
    """Probar operaciones de base de datos divinas."""
    # Crear datos
    await DivineDatabaseOperations.save_critical_data(
        "usuarios",
        {"nombre": "Usuario Prueba", "email": "test@example.com", "created_at": datetime.now().isoformat()}
    )
    
    # Leer datos
    usuarios = await DivineDatabaseOperations.read_data(
        "usuarios",
        {"email": "test@example.com"},
        ["id", "nombre", "email"]
    )
    
    # Actualizar datos
    await DivineDatabaseOperations.update_data(
        "usuarios",
        {"nombre": "Usuario Actualizado"},
        {"email": "test@example.com"}
    )
    
    # Ejecutar transacción compleja
    await DivineDatabaseOperations.execute_with_transaction([
        {
            "query": "INSERT INTO logs (message, level) VALUES (%s, %s)",
            "values": ["Prueba de transacción", "INFO"]
        },
        {
            "query": "UPDATE usuarios SET last_login = %s WHERE email = %s",
            "values": [datetime.now().isoformat(), "test@example.com"]
        }
    ])
    
    # Eliminar datos (en segundo plano)
    await DivineDatabaseOperations.delete_data(
        "logs",
        {"level": "DEBUG", "created_at": {"$lt": datetime.now().isoformat()}}
    )

async def mostrar_estadisticas():
    """Mostrar estadísticas del sistema divino."""
    stats = await get_divine_stats()
    
    logger.info("=== Estadísticas del Sistema Divino ===")
    logger.info(f"Tareas totales: {stats['total_tasks']}")
    logger.info(f"Tareas completadas: {stats['completed_tasks']}")
    logger.info(f"Tareas fallidas: {stats['failed_tasks']}")
    
    if 'avg_latencies' in stats:
        logger.info("Latencias promedio (ms):")
        for carrier, latency in stats['avg_latencies'].items():
            logger.info(f"  - {carrier}: {latency:.2f}")
    
    if 'queue_sizes' in stats:
        logger.info("Tamaños de colas:")
        for queue, size in stats['queue_sizes'].items():
            logger.info(f"  - {queue}: {size}")
    
    if 'config' in stats:
        logger.info(f"Modo: {stats['config']['mode']}")
        logger.info(f"Workers Redis: {stats['config']['redis_workers']}")
        logger.info(f"Workers RabbitMQ: {stats['config']['rabbitmq_workers']}")

async def main():
    """Función principal."""
    logger.info("=== Prueba del Sistema Divino de Cola de Tareas ===")
    
    # Inicializar sistema
    await initialize_divine_system()
    
    # Ejecutar tareas con diferentes prioridades
    tasks = []
    
    # Tarea crítica
    tasks.append(tarea_critica({"id": 1, "operation": "pago"}))
    
    # Tareas de alta prioridad
    for i in range(3):
        tasks.append(tarea_alta_prioridad(f"item_{i}", i * 10))
    
    # Tareas normales
    for i in range(5):
        tasks.append(tarea_normal(f"operación_{i}"))
    
    # Tareas de baja prioridad
    for i in range(2):
        tasks.append(tarea_baja_prioridad())
    
    # Tareas en segundo plano
    tasks.append(tarea_segundo_plano([i for i in range(100)]))
    
    # Esperar a que todas las tareas se completen
    await asyncio.gather(*tasks)
    
    # Probar transacciones
    await prueba_divina_transaction()
    
    # Probar operaciones de base de datos
    await prueba_operaciones_db()
    
    # Mostrar estadísticas
    await asyncio.sleep(1)  # Dar tiempo para actualizar estadísticas
    await mostrar_estadisticas()
    
    logger.info("=== Prueba completada ===")
    
    # Detener sistema para salir limpiamente (opcional, normalmente se dejaría corriendo)
    from genesis.db.divine_task_queue import divine_task_queue
    if divine_task_queue:
        await divine_task_queue.stop()

if __name__ == "__main__":
    asyncio.run(main())