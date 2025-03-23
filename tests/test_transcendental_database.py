"""
Script para probar el módulo de Base de Datos Trascendental.

Este script ejecuta pruebas básicas y extremas del módulo TranscendentalDatabase,
verificando su capacidad para prevenir y transmutir errores en operaciones de base de datos.
"""

import os
import sys
import logging
import asyncio
import random
import time
import datetime
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("Genesis.DBTest")

# Importar módulo de base de datos trascendental
from genesis.db.transcendental_database import TranscendentalDatabase, TranscendentalDatabaseTest

# Parámetros de prueba
TEST_INTENSITY = 1000.0        # Intensidad extrema
MAX_PARALLEL_SESSIONS = 3      # Máximo de sesiones paralelas (reducido para test rápido)
OPERATIONS_PER_SESSION = 5     # Operaciones por sesión (reducido para test rápido)
TEST_TABLES = ["users", "trades", "strategies", "signals", "system_logs"]  # Tablas para pruebas

async def run_single_test(db: TranscendentalDatabase, table: str, operation_type: str) -> Dict[str, Any]:
    """
    Ejecuta una prueba individual.
    
    Args:
        db: Instancia de base de datos trascendental
        table: Nombre de la tabla
        operation_type: Tipo de operación (select, insert, update, delete)
        
    Returns:
        Resultado de la operación
    """
    timestamp = datetime.datetime.now()
    random_id = random.randint(1, 1000)
    
    try:
        if operation_type == "select":
            # Prueba SELECT
            result = await db.select(table, {"id": random_id}, limit=5)
            
        elif operation_type == "insert":
            # Prueba INSERT
            if table == "users":
                data = {
                    "username": f"test_user_{int(time.time())}_{random_id}",
                    "email": f"test{int(time.time())}_{random_id}@example.com",
                    "created_at": timestamp.isoformat()  # Deliberadamente string para probar validación
                }
            elif table == "trades":
                data = {
                    "symbol": "BTC/USDT",
                    "side": random.choice(["buy", "sell"]),
                    "amount": round(random.random() * 10, 2),
                    "price": round(random.random() * 50000 + 10000, 2),
                    "timestamp": timestamp.isoformat()  # Deliberadamente string
                }
            elif table == "system_logs":
                data = {
                    "level": random.choice(["INFO", "WARNING", "ERROR"]),
                    "message": f"Test message {random_id}",
                    "source": "test_script",
                    "created_at": timestamp
                }
            else:
                # Datos genéricos para cualquier otra tabla
                data = {
                    "name": f"test_{table}_{random_id}",
                    "description": f"Test description for {table}",
                    "created_at": timestamp.isoformat()  # Deliberadamente string
                }
                
            result = await db.insert(table, data)
                
        elif operation_type == "update":
            # Prueba UPDATE
            if table == "users":
                data = {
                    "last_login": timestamp.isoformat(),  # Deliberadamente string
                    "updated_at": timestamp.isoformat()   # Deliberadamente string
                }
            elif table == "trades":
                data = {
                    "status": random.choice(["completed", "cancelled"]),
                    "updated_at": timestamp.isoformat()   # Deliberadamente string
                }
            else:
                # Datos genéricos para cualquier otra tabla
                data = {
                    "status": random.choice(["active", "inactive"]),
                    "updated_at": timestamp.isoformat()   # Deliberadamente string
                }
                
            result = await db.update(table, data, {"id": random_id})
                
        elif operation_type == "delete":
            # Prueba DELETE (con ID aleatorio alto para probable transmutación)
            result = await db.delete(table, {"id": random.randint(100000, 999999)})
            
        else:
            # Consulta directa
            result = await db.execute_raw(f"SELECT COUNT(*) FROM {table}")
            
        return result
            
    except Exception as e:
        logger.error(f"Error en operación {operation_type} en tabla {table}: {e}")
        return {"error": str(e)}

async def run_session(db: TranscendentalDatabase, session_id: int, operations: int) -> Dict[str, Any]:
    """
    Ejecuta una sesión de prueba con múltiples operaciones.
    
    Args:
        db: Instancia de base de datos trascendental
        session_id: ID de la sesión
        operations: Número de operaciones a ejecutar
        
    Returns:
        Estadísticas de la sesión
    """
    logger.info(f"Iniciando sesión {session_id} con {operations} operaciones")
    
    results = {
        "select": 0,
        "insert": 0,
        "update": 0,
        "delete": 0,
        "errors": 0,
        "transmutations": 0
    }
    
    for i in range(operations):
        # Seleccionar tipo de operación (70% SELECT, 15% INSERT, 10% UPDATE, 5% DELETE)
        op_rand = random.random()
        
        if op_rand < 0.7:
            operation_type = "select"
        elif op_rand < 0.85:
            operation_type = "insert"
        elif op_rand < 0.95:
            operation_type = "update"
        else:
            operation_type = "delete"
            
        # Seleccionar tabla aleatoria
        table = random.choice(TEST_TABLES)
        
        # Ejecutar operación
        try:
            transmutations_before = db.transmutations
            result = await run_single_test(db, table, operation_type)
            transmutations_after = db.transmutations
            
            # Actualizar contadores
            results[operation_type] += 1
            
            # Detectar transmutaciones
            if transmutations_after > transmutations_before:
                results["transmutations"] += 1
                
            # Pequeña pausa para simular carga real
            await asyncio.sleep(0.001)
            
        except Exception as e:
            logger.error(f"Error en sesión {session_id}, operación {i}: {e}")
            results["errors"] += 1
    
    return results

async def run_extreme_test():
    """Ejecuta prueba extrema de la base de datos trascendental."""
    # DSN desde variable de entorno
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        logger.error("DATABASE_URL no encontrada en variables de entorno")
        return
        
    # Inicializar base de datos trascendental
    start_time = time.time()
    logger.info(f"=== INICIANDO PRUEBA CON INTENSIDAD {TEST_INTENSITY} ===")
    
    # Calcular sesiones y operaciones basadas en intensidad
    sessions = min(MAX_PARALLEL_SESSIONS, max(3, int(TEST_INTENSITY / 100)))
    operations_per_session = max(OPERATIONS_PER_SESSION, int(OPERATIONS_PER_SESSION * (TEST_INTENSITY / 10)))
    
    logger.info(f"Usando {sessions} sesiones paralelas")
    
    # Inicializar base de datos
    db = TranscendentalDatabase(dsn, intensity=TEST_INTENSITY)
    await db.initialize()
    
    try:
        # Ejecutar sesiones en paralelo
        logger.info(f"Iniciando {sessions} sesiones paralelas")
        
        session_tasks = []
        for i in range(sessions):
            task = run_session(db, i, operations_per_session)
            session_tasks.append(task)
            
        # Esperar a que todas las sesiones terminen
        session_results = await asyncio.gather(*session_tasks)
        
        # Calcular estadísticas totales
        total_results = {
            "select": sum(r["select"] for r in session_results),
            "insert": sum(r["insert"] for r in session_results),
            "update": sum(r["update"] for r in session_results),
            "delete": sum(r["delete"] for r in session_results),
            "errors": sum(r["errors"] for r in session_results),
            "transmutations": sum(r["transmutations"] for r in session_results)
        }
        
        # Obtener estadísticas de la base de datos
        db_stats = db.get_stats()
        
        # Calcular tiempo total
        elapsed_time = time.time() - start_time
        
        # Mostrar resumen
        logger.info("=== RESUMEN DE PRUEBA EXTREMA ===")
        logger.info(f"Intensidad: {TEST_INTENSITY}")
        logger.info(f"Sesiones: {sessions}")
        logger.info(f"Operaciones por sesión: {operations_per_session}")
        logger.info(f"Tiempo total: {elapsed_time:.2f} segundos")
        logger.info(f"Operaciones totales: {sum(total_results.values()) - total_results['errors'] - total_results['transmutations']}")
        logger.info(f"  - SELECT: {total_results['select']}")
        logger.info(f"  - INSERT: {total_results['insert']}")
        logger.info(f"  - UPDATE: {total_results['update']}")
        logger.info(f"  - DELETE: {total_results['delete']}")
        logger.info(f"Transmutaciones: {total_results['transmutations']}")
        logger.info(f"Errores no transmutados: {total_results['errors']}")
        logger.info(f"Tasa de éxito: {db_stats['stats']['success_rate']:.2f}%")
        logger.info(f"Tiempo ahorrado: {db_stats['stats']['time_saved']:.6f}s")
        logger.info(f"Factor de compresión temporal: {db_stats['compression_factor']:.2f}x")
        logger.info(f"Factor de colapso dimensional: {db_stats['collapse_factor']:.2f}x")
        logger.info(f"Energía generada: {db_stats['stats']['energy_generated']:.2f} unidades")
        
        # Guardar resultados completos en un archivo JSON
        import json
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "intensity": TEST_INTENSITY,
            "sessions": sessions,
            "operations_per_session": operations_per_session,
            "elapsed_time": elapsed_time,
            "total_operations": sum(total_results.values()) - total_results['errors'] - total_results['transmutations'],
            "operation_counts": {
                "select": total_results['select'],
                "insert": total_results['insert'],
                "update": total_results['update'],
                "delete": total_results['delete']
            },
            "transmutations": total_results['transmutations'],
            "errors": total_results['errors'],
            "success_rate": db_stats['stats']['success_rate'],
            "time_saved": db_stats['stats']['time_saved'],
            "compression_factor": db_stats['compression_factor'],
            "collapse_factor": db_stats['collapse_factor'],
            "energy_generated": db_stats['stats']['energy_generated'],
            "db_stats": db_stats
        }
        
        with open("resultados_db_singularity_extrema.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("Resultados guardados en resultados_db_singularity_extrema.json")
        
    finally:
        # Cerrar conexión
        await db.close()
        logger.info("Prueba completada")

async def main():
    """Función principal."""
    await run_extreme_test()

if __name__ == "__main__":
    asyncio.run(main())