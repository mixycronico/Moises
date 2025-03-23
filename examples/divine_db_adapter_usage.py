"""
Ejemplo de uso del Adaptador Divino de Base de Datos para el Sistema Genesis.

Este script demuestra cómo utilizar las capacidades del DivineDatabaseAdapter
para operaciones síncronas y asíncronas, con manejo de caché y transacciones.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List

from genesis.db.divine_database import get_divine_db_adapter, divine_db

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("divine_db_example")

async def ejemplo_operaciones_basicas():
    """Demostrar operaciones básicas del adaptador divino."""
    logger.info("=== EJEMPLO: OPERACIONES BÁSICAS ===")
    
    # Obtener instancia global del adaptador
    db = divine_db
    
    # === Operaciones Síncronas ===
    logger.info("--- Operaciones Síncronas ---")
    
    # Ejecutar una consulta simple
    count = db.fetch_val_sync("SELECT count(*) FROM information_schema.tables", default=0)
    logger.info(f"Total de tablas en la base de datos: {count}")
    
    # Obtener información de tablas
    tables = db.fetch_all_sync(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' LIMIT 5"
    )
    logger.info(f"Primeras 5 tablas: {', '.join([t['table_name'] for t in tables])}")
    
    # Ejemplo con transacción
    try:
        with db.transaction_sync() as tx:
            # Esta es una operación de solo lectura para el ejemplo
            result = tx.fetch_one("SELECT current_timestamp AS tiempo")
            logger.info(f"Tiempo actual (en transacción): {result['tiempo']}")
    except Exception as e:
        logger.error(f"Error en transacción: {e}")
    
    # === Operaciones Asíncronas ===
    logger.info("--- Operaciones Asíncronas ---")
    
    # Ejecutar una consulta simple
    count = await db.fetch_val_async(
        "SELECT count(*) FROM information_schema.tables", 
        default=0
    )
    logger.info(f"Total de tablas en la base de datos (async): {count}")
    
    # Obtener información de tablas
    tables = await db.fetch_all_async(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' LIMIT 5"
    )
    logger.info(f"Primeras 5 tablas (async): {', '.join([t['table_name'] for t in tables])}")
    
    # Ejemplo de consultas paralelas
    start_time = time.time()
    tasks = [
        db.fetch_val_async("SELECT pg_sleep(0.1), 1 AS valor"),
        db.fetch_val_async("SELECT pg_sleep(0.1), 2 AS valor"),
        db.fetch_val_async("SELECT pg_sleep(0.1), 3 AS valor"),
    ]
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    logger.info(f"Resultados paralelos: {results}")
    logger.info(f"Tiempo total para 3 consultas de 0.1s: {end_time - start_time:.2f}s")
    
    # Ejemplo con transacción
    try:
        async with db.transaction_async() as tx:
            # Esta es una operación de solo lectura para el ejemplo
            result = await tx.fetch_one("SELECT current_timestamp AS tiempo")
            logger.info(f"Tiempo actual (en transacción async): {result['tiempo']}")
    except Exception as e:
        logger.error(f"Error en transacción async: {e}")
    
    # Mostrar estadísticas
    stats = db.get_stats()
    logger.info(f"Estadísticas del adaptador: {stats}")

async def ejemplo_cache():
    """Demostrar el sistema de cache del adaptador divino."""
    logger.info("\n=== EJEMPLO: SISTEMA DE CACHE ===")
    
    # Crear una instancia personalizada con configuración de cache específica
    db_custom = get_divine_db_adapter()
    
    # Configurar cache para una consulta específica
    query = "SELECT pg_sleep(0.2), current_timestamp AS tiempo"
    
    # Primera ejecución (sin cache)
    start = time.time()
    result1 = await db_custom.fetch_val(query)
    time1 = time.time() - start
    logger.info(f"Primera ejecución: {result1} (tiempo: {time1:.4f}s)")
    
    # Segunda ejecución (debería usar cache)
    start = time.time()
    result2 = await db_custom.fetch_val_sync(query)  # Usando versión síncrona
    time2 = time.time() - start
    logger.info(f"Segunda ejecución: {result2} (tiempo: {time2:.4f}s)")
    
    # Tercera ejecución (también debería usar cache)
    start = time.time()
    result3 = await db_custom.fetch_val(query)
    time3 = time.time() - start
    logger.info(f"Tercera ejecución: {result3} (tiempo: {time3:.4f}s)")
    
    # Comprobar si se usó el cache
    logger.info(f"Aceleración con cache: {time1 / time3:.1f}x más rápido")
    
    # Verificar estadísticas del cache
    cache_stats = db_custom.get_stats()["cache_stats"]
    logger.info(f"Estadísticas del cache:")
    logger.info(f"  - Hits: {cache_stats['hits']}")
    logger.info(f"  - Misses: {cache_stats['misses']}")
    logger.info(f"  - Hit ratio: {cache_stats['hit_ratio']:.2%}")
    logger.info(f"  - Entradas: {cache_stats['entries']}")
    logger.info(f"  - Memoria: {cache_stats['memory_usage_bytes'] / 1024:.2f} KB")

async def main():
    """Función principal del ejemplo."""
    logger.info("INICIO DEL EJEMPLO DEL ADAPTADOR DIVINO DE BASE DE DATOS")
    
    try:
        await ejemplo_operaciones_basicas()
        await ejemplo_cache()
        
    except Exception as e:
        logger.error(f"Error en el ejemplo: {e}")
    
    logger.info("FIN DEL EJEMPLO DEL ADAPTADOR DIVINO DE BASE DE DATOS")

if __name__ == "__main__":
    asyncio.run(main())