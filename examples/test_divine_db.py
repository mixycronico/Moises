"""
Script de prueba para el DivineDatabaseAdapter.

Este script ejecuta pruebas simples de funcionalidad del adaptador divino,
verificando operaciones básicas y características avanzadas.
"""
import os
import time
import asyncio
import logging
import random
from typing import List, Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_divine_db")

# Importar adaptador divino
from genesis.db.divine_database import (
    DivineDatabaseAdapter,
    get_divine_db_adapter
)

# Pruebas síncronas básicas
def test_sincrono_basico():
    """Probar funcionalidades síncronas básicas."""
    logger.info("=== Prueba: Operaciones síncronas básicas ===")
    
    # Obtener instancia
    db = get_divine_db_adapter()
    
    try:
        # Verificar existencia de tablas
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name LIKE 'gen_%'
            ORDER BY table_name
        """
        results = db.fetch_all_sync(query)
        
        if results:
            logger.info(f"Tablas encontradas: {len(results)}")
            for i, row in enumerate(results):
                if i < 10:  # Mostrar solo las primeras 10 tablas
                    logger.info(f"  - {row['table_name']}")
                elif i == 10:
                    logger.info(f"  - ... y {len(results) - 10} más")
        else:
            logger.error("No se encontraron tablas 'gen_*'")
            return False
        
        logger.info("Prueba síncrona básica completada con éxito")
        return True
    
    except Exception as e:
        logger.error(f"Error en prueba síncrona básica: {e}")
        return False

# Pruebas asíncronas básicas
async def test_asincrono_basico():
    """Probar funcionalidades asíncronas básicas."""
    logger.info("=== Prueba: Operaciones asíncronas básicas ===")
    
    # Obtener instancia
    db = get_divine_db_adapter()
    
    try:
        # Verificar existencia de datos
        query = """
            SELECT COUNT(*) as count
            FROM gen_intensity_results
        """
        result = await db.fetch_one_async(query)
        
        if result:
            logger.info(f"Registros en gen_intensity_results: {result['count']}")
        else:
            logger.warning("No se pudo obtener conteo de registros")
        
        # Probar múltiples consultas en paralelo
        queries = [
            "SELECT COUNT(*) as count FROM gen_intensity_results",
            "SELECT COUNT(*) as count FROM gen_components",
            "SELECT COUNT(*) as count FROM gen_metrics",
            "SELECT COUNT(*) as count FROM gen_events"
        ]
        
        tasks = []
        for query in queries:
            tasks.append(db.fetch_one_async(query))
        
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            table_name = queries[i].split("FROM ")[1].strip()
            if result:
                logger.info(f"Conteo en {table_name}: {result['count']}")
            else:
                logger.warning(f"No se pudo obtener conteo de {table_name}")
        
        logger.info("Prueba asíncrona básica completada con éxito")
        return True
    
    except Exception as e:
        logger.error(f"Error en prueba asíncrona básica: {e}")
        return False

# Prueba de caché
def test_cache():
    """Probar funcionalidad de caché."""
    logger.info("=== Prueba: Funcionalidad de caché ===")
    
    # Obtener instancia
    db = get_divine_db_adapter()
    
    try:
        # Limpiar caché actual
        db.clear_cache()
        
        # Primera ejecución (sin caché)
        query = """
            SELECT * 
            FROM gen_intensity_results 
            ORDER BY id 
            LIMIT 100
        """
        
        start = time.time()
        results1 = db.fetch_all_sync(query, use_cache=True)
        time1 = time.time() - start
        
        # Verificar si hay datos para continuar
        if not results1:
            logger.warning("No hay datos para probar caché, omitiendo prueba")
            return True
        
        # Segunda ejecución (con caché)
        start = time.time()
        results2 = db.fetch_all_sync(query, use_cache=True)
        time2 = time.time() - start
        
        # Tercera ejecución (con caché)
        start = time.time()
        results3 = db.fetch_all_sync(query, use_cache=True)
        time3 = time.time() - start
        
        # Verificar resultados
        logger.info(f"Primera ejecución: {time1:.6f}s")
        logger.info(f"Segunda ejecución: {time2:.6f}s")
        logger.info(f"Tercera ejecución: {time3:.6f}s")
        
        if time2 < time1 and time3 < time1:
            improvement = ((time1 - time3) / time1) * 100
            logger.info(f"Mejora por caché: {improvement:.2f}%")
            
            # Verificar estadísticas de caché
            stats = db.get_cache_stats()
            logger.info(f"Estadísticas de caché: {stats}")
            
            if stats['hits'] >= 2:  # Debería haber al menos 2 hits (2da y 3ra ejecución)
                logger.info("Prueba de caché completada con éxito")
                return True
            else:
                logger.warning("La caché no está funcionando correctamente")
                return False
        else:
            logger.warning("No se detectó mejora con caché")
            return False
    
    except Exception as e:
        logger.error(f"Error en prueba de caché: {e}")
        return False

# Prueba de transacciones
async def test_transacciones():
    """Probar funcionalidad de transacciones."""
    logger.info("=== Prueba: Funcionalidad de transacciones ===")
    
    # Obtener instancia
    db = get_divine_db_adapter()
    
    # Prueba 1: Transacción exitosa
    try:
        logger.info("Prueba 1: Transacción exitosa")
        
        # Crear tabla temporal para pruebas
        create_table_query = """
            CREATE TABLE IF NOT EXISTS temp_test_table (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                value FLOAT
            )
        """
        await db.execute_async(create_table_query)
        
        # Insertar datos en transacción
        async with db.transaction_async() as tx:
            for i in range(5):
                query = """
                    INSERT INTO temp_test_table (name, value)
                    VALUES (%(name)s, %(value)s)
                """
                params = {
                    "name": f"test_{i}",
                    "value": random.random() * 100
                }
                await db.execute_async(query, params, transaction=tx)
            
            logger.info("Insertados 5 registros en transacción")
        
        # Verificar datos insertados
        verify_query = "SELECT COUNT(*) as count FROM temp_test_table"
        result = await db.fetch_one_async(verify_query)
        
        if result and result['count'] >= 5:
            logger.info(f"Verificación exitosa: {result['count']} registros encontrados")
        else:
            logger.error("Fallo en transacción exitosa: no se encontraron los registros esperados")
            return False
    
    except Exception as e:
        logger.error(f"Error en prueba de transacción exitosa: {e}")
        return False
    
    # Prueba 2: Transacción con rollback
    try:
        logger.info("Prueba 2: Transacción con rollback")
        
        # Contar registros actuales
        count_query = "SELECT COUNT(*) as count FROM temp_test_table"
        result_before = await db.fetch_one_async(count_query)
        before_count = result_before['count'] if result_before else 0
        
        # Intentar transacción que fallará
        try:
            async with db.transaction_async() as tx:
                # Insertar algunos registros válidos
                for i in range(3):
                    query = """
                        INSERT INTO temp_test_table (name, value)
                        VALUES (%(name)s, %(value)s)
                    """
                    params = {
                        "name": f"will_rollback_{i}",
                        "value": random.random() * 100
                    }
                    await db.execute_async(query, params, transaction=tx)
                
                # Insertar registro inválido para forzar error
                invalid_query = """
                    INSERT INTO temp_test_table (name, value)
                    VALUES (%(name)s, %(invalid_column)s)
                """
                invalid_params = {
                    "name": "invalid_record"
                    # Falta el parámetro invalid_column para forzar error
                }
                await db.execute_async(invalid_query, invalid_params, transaction=tx)
                
                logger.error("¡Error! La transacción inválida no falló como se esperaba")
                return False
                
        except Exception as e:
            logger.info(f"Error esperado en transacción: {e}")
            
            # Verificar que no se insertaron registros (rollback exitoso)
            result_after = await db.fetch_one_async(count_query)
            after_count = result_after['count'] if result_after else 0
            
            if after_count == before_count:
                logger.info("Rollback exitoso: no se insertaron registros")
            else:
                logger.error(f"Fallo en rollback: se insertaron {after_count - before_count} registros")
                return False
    
    except Exception as e:
        logger.error(f"Error en prueba de transacción con rollback: {e}")
        return False
    
    # Limpiar tabla temporal
    try:
        await db.execute_async("DROP TABLE temp_test_table")
        logger.info("Tabla temporal eliminada")
    except Exception as e:
        logger.warning(f"Error al eliminar tabla temporal: {e}")
    
    logger.info("Prueba de transacciones completada con éxito")
    return True

# Función principal para ejecutar todas las pruebas
async def ejecutar_pruebas():
    """Ejecutar todas las pruebas."""
    resultados = {}
    
    # Pruebas síncronas
    resultados["sincrono_basico"] = test_sincrono_basico()
    resultados["cache"] = test_cache()
    
    # Pruebas asíncronas
    resultados["asincrono_basico"] = await test_asincrono_basico()
    resultados["transacciones"] = await test_transacciones()
    
    # Mostrar resumen
    logger.info("\n=== Resumen de Pruebas ===")
    
    all_passed = True
    for name, passed in resultados.items():
        status = "PASÓ" if passed else "FALLÓ"
        logger.info(f"{name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        logger.info("\n✅ Todas las pruebas pasaron exitosamente!")
    else:
        logger.error("\n❌ Algunas pruebas fallaron!")
    
    return all_passed

# Función principal
async def main():
    """Función principal para ejecutar pruebas."""
    logger.info("Iniciando pruebas de DivineDatabaseAdapter...")
    success = await ejecutar_pruebas()
    
    if success:
        logger.info("DivineDatabaseAdapter funciona correctamente!")
    else:
        logger.error("Se encontraron problemas con DivineDatabaseAdapter!")
    
    logger.info("Pruebas completadas.")

if __name__ == "__main__":
    asyncio.run(main())