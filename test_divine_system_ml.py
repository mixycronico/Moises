"""
Script de prueba para el Sistema Divino con Machine Learning.

Este script demuestra las capacidades del Sistema Divino con ML, incluyendo:
- Procesamiento ultrarrápido de operaciones de base de datos
- Priorización inteligente basada en ML
- Optimización dinámica de recursos
- Resiliencia extrema ante fallos

El sistema funcionará incluso cuando Redis o RabbitMQ no estén disponibles,
adaptándose automáticamente a las condiciones de ejecución.
"""

import asyncio
import logging
import time
import random
from datetime import datetime
from typing import Dict, Any, List

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_divine_ml")

# Importar sistema divino
from genesis.db.divine_system_integrator import (
    divine_system,
    execute_divine_sql,
    execute_divine_transaction,
    get_divine_system_stats
)

# Simulación de base de datos
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
async def get_simulated_db_session():
    return SimulatedDBSession()
genesis.db.base.get_db_session = get_simulated_db_session

# Pruebas
async def test_basic_queries():
    """Probar consultas básicas."""
    logger.info("=== Probando consultas básicas ===")
    
    # Consultas SELECT
    for i in range(5):
        result = await execute_divine_sql(
            f"SELECT * FROM items WHERE category = 'test' LIMIT {i+1}",
            ["param1", i],
            priority=5
        )
        logger.info(f"Resultado consulta {i+1}: {len(result)} filas")
    
    # Consulta crítica
    result = await execute_divine_sql(
        "SELECT * FROM high_value_trades WHERE amount > 10000",
        priority=9,
        critical=True
    )
    logger.info(f"Resultado consulta crítica: {len(result)} filas")
    
    # Consulta de baja prioridad
    result = await execute_divine_sql(
        "SELECT COUNT(*) FROM audit_logs",
        priority=2
    )
    logger.info(f"Resultado consulta baja prioridad: {result}")

async def test_transactions():
    """Probar transacciones."""
    logger.info("=== Probando transacciones ===")
    
    # Transacción simple
    result = await execute_divine_transaction([
        {
            "query": "INSERT INTO users (name, email) VALUES (%s, %s)",
            "params": ["Test User", "test@example.com"]
        },
        {
            "query": "INSERT INTO profiles (user_id, bio) VALUES (%s, %s)",
            "params": [1, "Test profile"]
        }
    ])
    logger.info(f"Resultado transacción: {result}")
    
    # Transacción con SELECT
    result = await execute_divine_transaction([
        {
            "query": "SELECT * FROM users WHERE email = %s",
            "params": ["test@example.com"]
        },
        {
            "query": "UPDATE users SET last_login = %s WHERE email = %s",
            "params": [datetime.now().isoformat(), "test@example.com"]
        }
    ])
    logger.info(f"Resultado transacción con SELECT: {result}")
    
    # Transacción fallida (simulación)
    try:
        result = await execute_divine_transaction([
            {
                "query": "INSERT INTO restricted_table (value) VALUES (%s)",
                "params": ["test"]
            }
        ])
    except Exception as e:
        logger.info(f"Transacción fallida controlada correctamente: {e}")

async def test_ml_optimization():
    """Probar optimización ML."""
    logger.info("=== Probando optimización ML ===")
    
    # Ejecutar múltiples consultas para entrenar el modelo
    logger.info("Ejecutando 50 consultas para entrenar el modelo ML...")
    
    for i in range(50):
        # Variar prioridad, criticidad y volumen
        priority = random.randint(1, 10)
        critical = random.random() > 0.7
        
        # Generar consulta de diferente complejidad
        if random.random() > 0.7:
            # Consulta compleja (transaccional)
            await execute_divine_transaction([
                {
                    "query": f"INSERT INTO large_table (id, data) VALUES (%s, %s)",
                    "params": [i, "X" * random.randint(10, 1000)]
                },
                {
                    "query": f"UPDATE related_table SET last_update = %s WHERE id = %s",
                    "params": [datetime.now().isoformat(), i]
                }
            ], priority=priority, critical=critical)
        else:
            # Consulta simple
            table = random.choice(["users", "items", "logs", "transactions"])
            size = random.randint(10, 1000)
            await execute_divine_sql(
                f"SELECT * FROM {table} LIMIT {size}",
                priority=priority,
                critical=critical
            )
        
        # Pequeña pausa entre consultas
        await asyncio.sleep(0.05)
    
    # Obtener estadísticas ML
    stats = await get_divine_system_stats()
    logger.info("Estadísticas ML:")
    if "divine_ml" in stats:
        for key, value in stats["divine_ml"].items():
            logger.info(f"  {key}: {value}")

async def test_adaptive_resources():
    """Probar adaptación de recursos."""
    logger.info("=== Probando adaptación de recursos ===")
    
    # Simular carga alta
    logger.info("Simulando carga alta...")
    tasks = []
    for i in range(100):
        tasks.append(execute_divine_sql(
            f"SELECT * FROM large_table WHERE id = %s",
            [i],
            priority=random.randint(1, 10)
        ))
    
    # Ejecutar en paralelo
    await asyncio.gather(*tasks)
    
    # Obtener estadísticas
    stats = await get_divine_system_stats()
    logger.info("Estadísticas después de carga alta:")
    logger.info(f"  Operaciones procesadas: {stats['divine_system']['operations_processed']}")
    logger.info(f"  Tasa de éxito: {stats['divine_system']['success_rate']:.2f}%")
    if "divine_queue" in stats:
        if "workers" in stats["divine_queue"]:
            logger.info(f"  Workers Redis: {stats['divine_queue']['workers'].get('redis', 'N/A')}")
            logger.info(f"  Workers RabbitMQ: {stats['divine_queue']['workers'].get('rabbitmq', 'N/A')}")

async def test_resilience():
    """Probar resiliencia del sistema."""
    logger.info("=== Probando resiliencia ===")
    
    # Simular operaciones con errores
    for i in range(10):
        try:
            if i % 3 == 0:
                # Forzar error
                await execute_divine_sql(
                    "SELECT * FROM non_existent_table",
                    priority=random.randint(1, 10)
                )
            else:
                await execute_divine_sql(
                    f"SELECT * FROM valid_table WHERE id = %s",
                    [i],
                    priority=random.randint(1, 10)
                )
        except Exception as e:
            logger.info(f"Error controlado: {e}")
    
    # Verificar recuperación
    await asyncio.sleep(1)
    
    # Ejecutar consulta válida después de errores
    result = await execute_divine_sql(
        "SELECT * FROM recovery_test",
        priority=7
    )
    logger.info(f"Consulta de recuperación exitosa: {result}")
    
    # Obtener estadísticas
    stats = await get_divine_system_stats()
    logger.info("Estadísticas de resiliencia:")
    logger.info(f"  Operaciones fallidas: {stats['divine_system']['operations_failed']}")
    logger.info(f"  Tasa de éxito: {stats['divine_system']['success_rate']:.2f}%")

async def main():
    """Función principal."""
    logger.info("=== Prueba del Sistema Divino con ML ===")
    
    # Inicializar y arrancar sistema divino
    await divine_system.initialize()
    await divine_system.start()
    
    # Ejecutar pruebas
    await test_basic_queries()
    
    await test_transactions()
    
    await test_ml_optimization()
    
    await test_adaptive_resources()
    
    await test_resilience()
    
    # Mostrar estadísticas finales
    stats = await get_divine_system_stats()
    logger.info("=== Estadísticas finales del Sistema Divino ===")
    logger.info(f"Operaciones procesadas: {stats['divine_system']['operations_processed']}")
    logger.info(f"Operaciones fallidas: {stats['divine_system']['operations_failed']}")
    logger.info(f"Tasa de éxito: {stats['divine_system']['success_rate']:.2f}%")
    logger.info(f"Tiempo de actividad: {stats['divine_system'].get('uptime_seconds', 0):.2f} segundos")
    
    # Detener el sistema divino para salir limpiamente
    await divine_system.stop()
    
    logger.info("=== Pruebas completadas ===")

if __name__ == "__main__":
    asyncio.run(main())