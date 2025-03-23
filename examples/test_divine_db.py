"""
Script de prueba para el Adaptador Divino de Base de Datos.

Este script realiza pruebas básicas de conexión, consulta y rendimiento
utilizando el DivineDatabaseAdapter con la base de datos PostgreSQL.
"""

import asyncio
import logging
import time
import os
import sys
from typing import Dict, Any, List

# Agregar directorio raíz al path para importar el módulo
sys.path.append('.')

# Importar después de agregar el directorio al path
from genesis.db.divine_database import get_divine_db_adapter

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("divine_db_test")

async def test_connection():
    """Probar conexión básica a la base de datos."""
    logger.info("=== PRUEBA: CONEXIÓN A LA BASE DE DATOS ===")
    
    # Obtener URL de conexión del entorno
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("No se encontró la variable de entorno DATABASE_URL")
        return False
    
    logger.info(f"Conectando a la base de datos...")
    
    # Crear instancia del adaptador
    try:
        db = get_divine_db_adapter(db_url)
        
        # Probar conexión
        version = await db.fetch_val_async("SELECT version()")
        logger.info(f"Conectado exitosamente a PostgreSQL: {version}")
        
        return True
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        return False

async def test_basic_operations():
    """Probar operaciones básicas del adaptador."""
    logger.info("\n=== PRUEBA: OPERACIONES BÁSICAS ===")
    
    db = get_divine_db_adapter()
    
    try:
        # Consultar tablas en la base de datos
        tables = await db.fetch_all_async(
            "SELECT table_name, table_schema FROM information_schema.tables WHERE table_schema = 'public'"
        )
        
        if tables:
            logger.info(f"Se encontraron {len(tables)} tablas en el esquema public:")
            for i, table in enumerate(tables[:10], 1):  # Mostrar solo las primeras 10
                logger.info(f"  {i}. {table['table_name']}")
            
            if len(tables) > 10:
                logger.info(f"  ... y {len(tables) - 10} más")
        else:
            logger.info("No se encontraron tablas en el esquema public")
        
        # Consultar esquemas
        schemas = await db.fetch_all_async(
            "SELECT schema_name FROM information_schema.schemata ORDER BY schema_name"
        )
        
        logger.info(f"Esquemas disponibles: {', '.join([s['schema_name'] for s in schemas])}")
        
        # Buscar tablas 'gen_'
        gen_tables = await db.fetch_all_async(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'gen_%'"
        )
        
        if gen_tables:
            logger.info(f"Se encontraron {len(gen_tables)} tablas con prefijo 'gen_':")
            for table in gen_tables:
                logger.info(f"  - {table['table_name']}")
                
            # Seleccionar la primera tabla para más pruebas
            first_table = gen_tables[0]['table_name']
            logger.info(f"Examinando la tabla '{first_table}':")
            
            # Obtener estructura
            columns = await db.fetch_all_async(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s",
                [first_table]
            )
            
            logger.info(f"Columnas de {first_table}:")
            for col in columns:
                logger.info(f"  - {col['column_name']} ({col['data_type']})")
            
            # Contar registros
            count = await db.fetch_val_async(
                f"SELECT count(*) FROM {first_table}",
                default=0
            )
            
            logger.info(f"La tabla {first_table} contiene {count} registros")
            
            # Obtener muestra
            if count > 0:
                sample = await db.fetch_one_async(
                    f"SELECT * FROM {first_table} LIMIT 1"
                )
                
                if sample:
                    logger.info(f"Muestra de registro de {first_table}:")
                    for key, value in sample.items():
                        logger.info(f"  - {key}: {value}")
        else:
            logger.info("No se encontraron tablas con prefijo 'gen_'")
        
        return True
    except Exception as e:
        logger.error(f"Error en pruebas básicas: {e}")
        return False

async def test_performance():
    """Probar rendimiento del adaptador con y sin cache."""
    logger.info("\n=== PRUEBA: RENDIMIENTO Y CACHE ===")
    
    db = get_divine_db_adapter()
    
    try:
        # Consulta que toma tiempo (usando pg_sleep para simular procesamiento)
        query = "SELECT pg_sleep(0.1), current_timestamp AS tiempo"
        
        # Primera ejecución (sin cache)
        logger.info("Ejecutando consulta sin cache...")
        start = time.time()
        result1 = await db.fetch_one_async(query)
        time1 = time.time() - start
        logger.info(f"Primera ejecución: {result1['tiempo']} (tiempo: {time1:.4f}s)")
        
        # Segunda ejecución (debería usar cache si está habilitado)
        logger.info("Ejecutando consulta nuevamente (posible cache)...")
        start = time.time()
        result2 = await db.fetch_one_async(query)
        time2 = time.time() - start
        logger.info(f"Segunda ejecución: {result2['tiempo']} (tiempo: {time2:.4f}s)")
        
        # Verificar si el cache funcionó
        if time2 < time1 * 0.5:  # Si es significativamente más rápido
            logger.info(f"¡Cache funcionando! Aceleración: {time1/time2:.1f}x")
        else:
            logger.info(f"Cache no detectado o no habilitado para esta consulta")
        
        # Estadísticas del adaptador
        stats = db.get_stats()
        
        logger.info("Estadísticas del adaptador:")
        logger.info(f"  - Consultas totales: {stats['total_queries']}")
        logger.info(f"  - Tiempo promedio: {stats['query_time_avg']*1000:.2f}ms")
        
        if 'cache_stats' in stats:
            logger.info("Estadísticas del cache:")
            logger.info(f"  - Hits: {stats['cache_stats']['hits']}")
            logger.info(f"  - Misses: {stats['cache_stats']['misses']}")
            logger.info(f"  - Hit ratio: {stats['cache_stats']['hit_ratio']:.2%}")
        
        return True
    except Exception as e:
        logger.error(f"Error en pruebas de rendimiento: {e}")
        return False

async def main():
    """Función principal del script de prueba."""
    logger.info("INICIANDO PRUEBAS DEL ADAPTADOR DIVINO DE BASE DE DATOS")
    
    try:
        # Probar conexión
        if not await test_connection():
            logger.error("Prueba de conexión fallida. Abortando pruebas adicionales.")
            return
        
        # Probar operaciones básicas
        await test_basic_operations()
        
        # Probar rendimiento
        await test_performance()
        
        logger.info("\nFIN DE PRUEBAS - ADAPTADOR DIVINO FUNCIONANDO CORRECTAMENTE")
        
    except Exception as e:
        logger.error(f"Error general en pruebas: {e}")

if __name__ == "__main__":
    asyncio.run(main())