"""
Ejemplo de uso del adaptador divino de base de datos.

Este script demuestra cómo usar el adaptador divino de base de datos
en contextos tanto síncronos como asíncronos.
"""
import os
import time
import asyncio
import logging
from typing import Dict, Any, List

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("divine_db_example")

# Importamos nuestro adaptador divino
from genesis.db.divine_database import get_divine_db_adapter, divine_db

# Ejemplos de uso sincrónicos
def run_sync_examples():
    """Ejecutar ejemplos síncronos de uso del adaptador divino."""
    logger.info("=== Ejemplos síncronos ===")
    
    # Consulta simple para verificar que la base de datos funciona
    count = divine_db.fetch_val_sync("SELECT count(*) FROM gen_components", default=0)
    logger.info(f"Número de componentes en la base de datos: {count}")
    
    # Consulta con parámetros
    components = divine_db.fetch_all_sync(
        "SELECT * FROM gen_components WHERE tipo = %s LIMIT 5",
        params=["CORE"]
    )
    logger.info(f"Componentes de tipo CORE: {len(components)}")
    
    # Uso de transacciones
    try:
        with divine_db.transaction_sync() as tx:
            # Este es solo un ejemplo de cómo funciona la transacción
            # No ejecutamos realmente ninguna inserción para no modificar la base de datos
            logger.info("Transacción síncrona funcionando correctamente")
    except Exception as e:
        logger.error(f"Error en transacción síncrona: {e}")
    
    # Estadísticas de uso
    stats = divine_db.get_stats()
    logger.info(f"Estadísticas: Consultas síncronas: {stats['queries_sync']}, Tasa de error: {stats['error_rate']:.2%}")
    logger.info(f"Tiempo promedio de consulta: {stats['query_time_avg']*1000:.2f}ms")

# Ejemplos de uso asíncronos
async def run_async_examples():
    """Ejecutar ejemplos asíncronos de uso del adaptador divino."""
    logger.info("=== Ejemplos asíncronos ===")
    
    # Consulta simple
    count = await divine_db.fetch_val_async("SELECT count(*) FROM gen_intensity_results", default=0)
    logger.info(f"Número de resultados de intensidad: {count}")
    
    # Consulta con parámetros
    results = await divine_db.fetch_all_async(
        "SELECT * FROM gen_intensity_results WHERE intensity > %s LIMIT 3",
        params=[1.0]
    )
    logger.info(f"Resultados de intensidad > 1.0: {len(results)}")
    
    # Ejecución concurrente de múltiples consultas
    tasks = [
        divine_db.fetch_val_async("SELECT MAX(intensity) FROM gen_intensity_results", default=0),
        divine_db.fetch_val_async("SELECT MIN(intensity) FROM gen_intensity_results", default=0),
        divine_db.fetch_val_async("SELECT AVG(intensity) FROM gen_intensity_results", default=0)
    ]
    
    max_intensity, min_intensity, avg_intensity = await asyncio.gather(*tasks)
    logger.info(f"Intensidad - Max: {max_intensity}, Min: {min_intensity}, Avg: {avg_intensity}")
    
    # Uso de transacciones
    try:
        async with divine_db.transaction_async() as tx:
            # Este es solo un ejemplo de cómo funciona la transacción
            # No ejecutamos realmente ninguna inserción para no modificar la base de datos
            logger.info("Transacción asíncrona funcionando correctamente")
    except Exception as e:
        logger.error(f"Error en transacción asíncrona: {e}")
    
    # Estadísticas de uso
    stats = divine_db.get_stats()
    logger.info(f"Estadísticas: Consultas asíncronas: {stats['queries_async']}, Tasa de aciertos cache: {stats['cache_hit_ratio']:.2%}")

# Ejemplos que muestran el uso automático basado en contexto
async def run_context_smart_examples():
    """Ejecutar ejemplos de detección automática de contexto."""
    logger.info("=== Ejemplos de detección de contexto ===")
    
    # En un contexto asíncrono, utilizará automáticamente las versiones asíncronas
    count = await divine_db.fetch_val("SELECT count(*) FROM gen_components", default=0)
    logger.info(f"Auto-contexto asíncrono - Componentes: {count}")
    
    # Para demostrar el contexto síncrono, creamos una función síncrona
    def sync_function():
        # En un contexto síncrono, deberíamos usar los métodos específicos síncronos
        # ya que .fetch_val() requiere await en contexto asíncrono
        count = divine_db.fetch_val_sync("SELECT count(*) FROM gen_intensity_results", default=0)
        logger.info(f"Contexto síncrono explícito - Resultados: {count}")
    
    # Ejecutamos la función síncrona
    sync_function()

async def main():
    """Función principal para ejecutar todos los ejemplos."""
    logger.info("Iniciando ejemplos de uso del adaptador divino de base de datos")
    
    # Ejecutar ejemplos síncronos
    run_sync_examples()
    
    # Ejecutar ejemplos asíncronos
    await run_async_examples()
    
    # Ejecutar ejemplos de detección de contexto
    await run_context_smart_examples()
    
    # Mostrar estadísticas finales
    stats = divine_db.get_stats()
    logger.info("=== Estadísticas finales ===")
    logger.info(f"Total consultas: {stats['total_queries']}")
    logger.info(f"- Síncronas: {stats['queries_sync']}")
    logger.info(f"- Asíncronas: {stats['queries_async']}")
    logger.info(f"Tasa de error: {stats['error_rate']:.2%}")
    logger.info(f"Tasa de aciertos caché: {stats['cache_hit_ratio']:.2%}")
    logger.info(f"Tiempo promedio consulta: {stats['query_time_avg']*1000:.2f}ms")
    logger.info(f"Cache - Tamaño: {stats['cache_stats']['size']}/{stats['cache_stats']['max_size']}")

if __name__ == "__main__":
    asyncio.run(main())