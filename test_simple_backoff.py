"""
Prueba simple del sistema de reintentos adaptativos con backoff exponencial.

Esta es una versión simplificada para demostrar el funcionamiento básico
del sistema de reintentos con backoff exponencial.
"""

import asyncio
import logging
import time
import random

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simple_backoff_test")

# Decorador simulado de retry
async def with_retry(func, max_retries=3, base_delay=0.1):
    """
    Ejecutar una función con reintentos usando backoff exponencial.
    
    Args:
        func: Función asíncrona a ejecutar
        max_retries: Número máximo de reintentos
        base_delay: Retraso base en segundos
    
    Returns:
        El resultado de la función o None si fallan todos los intentos
    """
    attempt = 0
    delays = []
    
    while attempt <= max_retries:
        try:
            start_time = time.time()
            result = await func()
            logger.info(f"Intento {attempt + 1}: Éxito en {time.time() - start_time:.3f}s")
            return result
        except Exception as e:
            attempt += 1
            if attempt <= max_retries:
                # Calcular backoff exponencial con jitter
                delay = base_delay * (2 ** (attempt - 1))
                jitter = delay * 0.2
                final_delay = delay + random.uniform(-jitter, jitter)
                delays.append(final_delay)
                
                logger.info(f"Intento {attempt}: Fallo - {e}. Reintentando en {final_delay:.3f}s")
                await asyncio.sleep(final_delay)
            else:
                logger.error(f"Intento {attempt}: Fallo final - {e}")
                raise e
    
    return None

# Función que falla aleatoriamente
async def unreliable_operation(fail_rate=0.7, always_fail_first=False):
    """
    Simular una operación que falla aleatoriamente.
    
    Args:
        fail_rate: Probabilidad de fallo (0.0-1.0)
        always_fail_first: Si True, siempre falla en el primer intento
    
    Returns:
        Un resultado simulado
    
    Raises:
        Exception: Si la operación falla aleatoriamente
    """
    # Simular trabajo
    await asyncio.sleep(0.1)
    
    # Contador global para saber en qué intento estamos
    if not hasattr(unreliable_operation, "call_count"):
        unreliable_operation.call_count = 0
    unreliable_operation.call_count += 1
    
    # Forzar fallo en los primeros dos intentos
    if unreliable_operation.call_count <= 2:
        raise Exception(f"Fallo simulado en intento {unreliable_operation.call_count}")
    
    # Decidir si falla aleatoriamente en los siguientes intentos
    if random.random() < fail_rate:
        raise Exception(f"Fallo aleatorio en intento {unreliable_operation.call_count}")
    
    return {"status": "success", "timestamp": time.time(), "attempt": unreliable_operation.call_count}

async def main():
    """Función principal."""
    logger.info("=== Prueba de Sistema de Reintentos con Backoff Exponencial ===")
    
    # Ejecutar con reintentos
    try:
        logger.info("Ejecutando operación con reintentos...")
        result = await with_retry(lambda: unreliable_operation(0.7))
        logger.info(f"Resultado final: {result}")
        logger.info("✓ La operación se completó exitosamente con reintentos")
    except Exception as e:
        logger.error(f"✗ La operación falló a pesar de los reintentos: {e}")
    
    logger.info("\nPrueba completada.")

if __name__ == "__main__":
    asyncio.run(main())