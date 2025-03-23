"""
Script para probar el adaptador de base de datos.

Este script verifica que el adaptador funcione correctamente tanto 
en modo síncrono como asíncrono.
"""
import asyncio
import logging
from genesis.db import get_db_adapter, DatabaseAdapter

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Consulta para verificar operación de base de datos
TEST_QUERY = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'gen_%' LIMIT 5"

async def test_async_mode():
    """Probar el adaptador en modo asíncrono."""
    logger.info("Probando adaptador en modo asíncrono...")
    
    # Obtener adaptador
    adapter = get_db_adapter()
    
    try:
        # Ejecutar consulta usando el adaptador
        results = await adapter.fetch(TEST_QUERY)
        
        # Verificar resultados
        if results:
            logger.info(f"Consulta asíncrona exitosa. Primeras tablas encontradas: {[row['table_name'] for row in results]}")
            return True
        else:
            logger.warning("Consulta asíncrona no retornó resultados.")
            return False
    except Exception as e:
        logger.error(f"Error en prueba asíncrona: {e}")
        return False

def test_sync_mode():
    """Probar el adaptador en modo síncrono."""
    logger.info("Probando adaptador en modo síncrono...")
    
    # Obtener adaptador
    adapter = get_db_adapter()
    
    try:
        # Adaptarnos a la ejecución síncrona
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(adapter.fetch(TEST_QUERY))
        loop.close()
        
        # Verificar resultados
        if results:
            logger.info(f"Consulta síncrona exitosa. Primeras tablas encontradas: {[row['table_name'] for row in results]}")
            return True
        else:
            logger.warning("Consulta síncrona no retornó resultados.")
            return False
    except Exception as e:
        logger.error(f"Error en prueba síncrona: {e}")
        return False

async def main():
    """Función principal."""
    logger.info("Iniciando prueba del adaptador de base de datos...")
    
    # Probar modo asíncrono
    async_success = await test_async_mode()
    
    # Probar modo síncrono
    sync_success = test_sync_mode()
    
    # Mostrar resultado final
    if async_success and sync_success:
        logger.info("✅ Adaptador de base de datos funciona correctamente en ambos modos!")
    else:
        logger.error("❌ El adaptador de base de datos falló en al menos un modo.")
        if not async_success:
            logger.error("  - Modo asíncrono: FALLÓ")
        if not sync_success:
            logger.error("  - Modo síncrono: FALLÓ")

if __name__ == "__main__":
    asyncio.run(main())