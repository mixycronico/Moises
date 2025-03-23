"""
Prueba simple del Adaptador Divino de Base de Datos.
"""
import asyncio
import logging
import os
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_db")

# Importar módulo de base de datos
from genesis.db.divine_database import DivineDatabaseAdapter

async def run_test():
    """Ejecutar prueba simple de la base de datos."""
    # Obtener URL de la base de datos
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("No se encontró la variable de entorno DATABASE_URL")
        return
    
    logger.info(f"Conectando a la base de datos...")
    
    # Crear instancia del adaptador
    db = DivineDatabaseAdapter(db_url)
    
    try:
        # Probar conexión básica
        version = await db.fetch_val_async("SELECT version()")
        logger.info(f"Conectado exitosamente a PostgreSQL: {version}")
        
        # Consultar tablas
        tables = await db.fetch_all_async(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' LIMIT 5"
        )
        
        if tables:
            logger.info(f"Primeras 5 tablas encontradas:")
            for i, table in enumerate(tables, 1):
                logger.info(f"  {i}. {table['table_name']}")
        else:
            logger.info("No se encontraron tablas en el esquema public")
        
        # Probar cache
        logger.info("Probando cache con consulta que toma tiempo...")
        
        query = "SELECT pg_sleep(0.1), current_timestamp AS tiempo"
        
        # Primera ejecución
        start = time.time()
        result1 = await db.fetch_one_async(query)
        time1 = time.time() - start
        logger.info(f"Primera ejecución: {time1:.4f}s")
        
        # Segunda ejecución
        start = time.time()
        result2 = await db.fetch_one_async(query)
        time2 = time.time() - start
        logger.info(f"Segunda ejecución: {time2:.4f}s")
        
        if time2 < time1 * 0.5:
            logger.info(f"¡Cache funcionando correctamente!")
        
        # Probar transacción
        logger.info("Probando transacción asíncrona...")
        async with db.transaction_async() as tx:
            result = await tx.fetch_one("SELECT current_timestamp AS tiempo")
            logger.info(f"Timestamp en transacción: {result['tiempo']}")
        
        # Mostrar estadísticas
        stats = db.get_stats()
        logger.info(f"Estadísticas del adaptador: {stats}")
        
        logger.info("Prueba completada exitosamente")
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
    finally:
        # Cerrar conexiones
        db.close()
        logger.info("Conexiones cerradas")

if __name__ == "__main__":
    asyncio.run(run_test())