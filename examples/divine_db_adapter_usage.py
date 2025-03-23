"""
Ejemplos de uso del DivineDatabaseAdapter.

Este script muestra cómo utilizar el adaptador divino de base de datos
en diferentes contextos y escenarios, aprovechando sus características
de resiliencia extrema y soporte híbrido.
"""
import os
import time
import asyncio
import logging
from typing import List, Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("examples.divine_db")

# Importar adaptador divino
from genesis.db.divine_database import (
    DivineDatabaseAdapter,
    get_divine_db_adapter
)

# Ejemplo 1: Uso síncrono simple
def ejemplo_uso_sincrono():
    """Demostrar el uso síncrono básico del adaptador."""
    logger.info("Ejemplo 1: Uso síncrono básico")
    
    # Obtener instancia (singleton)
    db = get_divine_db_adapter()
    
    # Ejecutar consulta simple
    try:
        # Obtener modos de sistema existentes
        query = """
            SELECT DISTINCT mode 
            FROM gen_intensity_results 
            ORDER BY mode
        """
        results = db.fetch_all_sync(query)
        
        if results:
            logger.info(f"Modos de sistema encontrados: {len(results)}")
            for row in results:
                logger.info(f"  - {row['mode']}")
        else:
            logger.info("No se encontraron modos de sistema")
        
        # Otra consulta con parámetros
        query_params = """
            SELECT COUNT(*) as total_tests
            FROM gen_intensity_results
            WHERE intensity >= %(min_intensity)s
        """
        params = {"min_intensity": 5.0}
        result = db.fetch_one_sync(query_params, params)
        
        if result:
            logger.info(f"Total de pruebas con intensidad >= 5.0: {result['total_tests']}")
    
    except Exception as e:
        logger.error(f"Error en ejemplo síncrono: {e}")

# Ejemplo 2: Uso asíncrono simple
async def ejemplo_uso_asincrono():
    """Demostrar el uso asíncrono básico del adaptador."""
    logger.info("Ejemplo 2: Uso asíncrono básico")
    
    # Obtener instancia (singleton)
    db = get_divine_db_adapter()
    
    # Ejecutar consulta simple de forma asíncrona
    try:
        # Obtener estadísticas por componente
        query = """
            SELECT component_id, 
                   COUNT(*) as count,
                   AVG(success_rate) as avg_success_rate
            FROM gen_components
            GROUP BY component_id
            ORDER BY avg_success_rate DESC
        """
        results = await db.fetch_all_async(query)
        
        if results:
            logger.info(f"Componentes encontrados: {len(results)}")
            for row in results:
                logger.info(f"  - {row['component_id']}: {row['avg_success_rate']:.2f} "
                          f"({row['count']} registros)")
        else:
            logger.info("No se encontraron componentes")
        
        # Obtener resultado de prueba específico
        query_params = """
            SELECT *
            FROM gen_intensity_results
            WHERE id = %(id)s
        """
        params = {"id": 1}
        result = await db.fetch_one_async(query_params, params)
        
        if result:
            logger.info(f"Prueba ID 1: {result['mode']} (Intensidad: {result['intensity']}, "
                      f"Éxito: {result['average_success_rate']:.2f})")
        else:
            logger.info("No se encontró la prueba con ID 1")
    
    except Exception as e:
        logger.error(f"Error en ejemplo asíncrono: {e}")

# Ejemplo 3: Transacciones
async def ejemplo_transacciones():
    """Demostrar el uso de transacciones en el adaptador."""
    logger.info("Ejemplo 3: Uso de transacciones")
    
    # Obtener instancia (singleton)
    db = get_divine_db_adapter()
    
    # Ejemplo de transacción asíncrona
    try:
        async with db.transaction_async() as tx:
            # Insertar resultado de prueba
            insert_query = """
                INSERT INTO gen_intensity_results 
                (intensity, mode, average_success_rate, components_count, 
                 total_events, execution_time, timestamp, system_version)
                VALUES 
                (%(intensity)s, %(mode)s, %(success_rate)s, %(comp_count)s,
                 %(events)s, %(exec_time)s, NOW(), %(version)s)
                RETURNING id
            """
            params = {
                "intensity": 10.0,
                "mode": "EXAMPLE_MODE",
                "success_rate": 0.95,
                "comp_count": 5,
                "events": 100,
                "exec_time": 30.5,
                "version": "example_version"
            }
            
            result = await db.fetch_one_async(insert_query, params, transaction=tx)
            result_id = result['id']
            logger.info(f"Insertado registro de prueba con ID: {result_id}")
            
            # Insertar componentes asociados
            for i in range(5):
                comp_query = """
                    INSERT INTO gen_components
                    (results_id, component_id, component_type, success_rate)
                    VALUES
                    (%(result_id)s, %(comp_id)s, %(comp_type)s, %(success)s)
                """
                comp_params = {
                    "result_id": result_id,
                    "comp_id": f"comp_{i}",
                    "comp_type": "EXAMPLE",
                    "success": 0.90 + (i * 0.01)
                }
                
                await db.execute_async(comp_query, comp_params, transaction=tx)
            
            logger.info("Insertados 5 componentes asociados")
            
            # Para demostración: comentar línea siguiente para probar rollback
            # raise Exception("Forzar rollback para demostración")
    
    except Exception as e:
        logger.error(f"Error en transacción (rollback automático): {e}")

# Ejemplo 4: Funcionalidades resilientes
def ejemplo_funcionalidades_resilientes():
    """Demostrar las funcionalidades resilientes del adaptador."""
    logger.info("Ejemplo 4: Funcionalidades resilientes")
    
    # Obtener instancia personalizada con configuración específica
    db = DivineDatabaseAdapter(
        pool_size=10,
        max_retries=3,
        retry_delay=0.1,
        connection_timeout=5.0,
        statement_timeout=10.0,
        idle_in_transaction_timeout=30.0,
        isolation_level="READ COMMITTED"
    )
    
    logger.info("Configuración resiliente aplicada")
    
    # Demostrar cache
    try:
        # Primera ejecución (sin cache)
        start = time.time()
        query = "SELECT * FROM gen_intensity_results LIMIT 10"
        results1 = db.fetch_all_sync(query, use_cache=True)
        time1 = time.time() - start
        
        # Segunda ejecución (con cache)
        start = time.time()
        results2 = db.fetch_all_sync(query, use_cache=True)
        time2 = time.time() - start
        
        if results1 and results2:
            logger.info(f"Primera ejecución: {time1:.6f}s, {len(results1)} registros")
            logger.info(f"Segunda ejecución: {time2:.6f}s, {len(results2)} registros")
            
            if time2 < time1:
                improvement = ((time1 - time2) / time1) * 100
                logger.info(f"Mejora por cache: {improvement:.2f}%")
        else:
            logger.info("No hay datos para mostrar mejora de cache")
        
        # Estadísticas de cache
        cache_stats = db.get_cache_stats()
        logger.info(f"Estadísticas de cache: {cache_stats}")
    
    except Exception as e:
        logger.error(f"Error en ejemplo de funcionalidades resilientes: {e}")

# Función para ejecutar todos los ejemplos
async def ejecutar_ejemplos():
    """Ejecutar todos los ejemplos."""
    # Ejemplos síncronos
    ejemplo_uso_sincrono()
    ejemplo_funcionalidades_resilientes()
    
    # Ejemplos asíncronos
    await ejemplo_uso_asincrono()
    await ejemplo_transacciones()

# Función principal
async def main():
    """Función principal para ejecutar todos los ejemplos."""
    logger.info("Iniciando ejemplos de DivineDatabaseAdapter...")
    await ejecutar_ejemplos()
    logger.info("Ejemplos completados.")

if __name__ == "__main__":
    asyncio.run(main())