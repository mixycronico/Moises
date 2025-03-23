"""
Script de prueba para el Procesador Asincrónico Ultra-Cuántico.

Este script demuestra cómo usar el procesador asincrónico para ejecutar 
operaciones paralelas sin deadlocks ni race conditions, con aislamiento
cuántico y capacidades de transmutación de errores.
"""

import asyncio
import logging
import time
import random
from typing import List, Dict, Any

from genesis.core.async_quantum_processor import (
    async_quantum_operation,
    run_isolated,
    quantum_thread_context,
    quantum_process_context
)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestAsyncQuantum")

# Operaciones de prueba
@async_quantum_operation(namespace="calculo", priority=8)
async def calculo_complejo(n: int) -> float:
    """
    Simula un cálculo complejo asincrónico.
    
    Args:
        n: Número de iteraciones
        
    Returns:
        Resultado del cálculo
    """
    logger.info(f"Iniciando cálculo complejo con {n} iteraciones")
    resultado = 0.0
    
    for i in range(n):
        # Simulamos operaciones que toman tiempo
        await asyncio.sleep(0.01)
        resultado += (i * random.random())
        
    logger.info(f"Cálculo complejo completado: {resultado:.2f}")
    return resultado

@async_quantum_operation(namespace="io", priority=5)
async def operacion_io() -> Dict[str, Any]:
    """
    Simula una operación de I/O asincrónica.
    
    Returns:
        Datos obtenidos
    """
    logger.info("Iniciando operación de I/O")
    
    # Simular latencia de red/disco
    await asyncio.sleep(0.2)
    
    # Simular datos obtenidos
    datos = {
        "timestamp": time.time(),
        "valores": [random.random() for _ in range(5)],
        "estado": "completado"
    }
    
    logger.info(f"Operación de I/O completada: {len(datos['valores'])} valores")
    return datos

@async_quantum_operation(namespace="error", priority=9)
async def operacion_con_error(debe_fallar: bool = True) -> Dict[str, Any]:
    """
    Operación que puede fallar para probar transmutación de errores.
    
    Args:
        debe_fallar: Si debe fallar la operación
        
    Returns:
        Datos procesados o error transmutado
    """
    logger.info(f"Iniciando operación que puede fallar: debe_fallar={debe_fallar}")
    
    await asyncio.sleep(0.1)
    
    if debe_fallar:
        logger.warning("Generando error intencionalmente")
        raise ValueError("Error simulado para prueba de transmutación")
        
    return {"estado": "éxito sin error"}

@async_quantum_operation(run_in_thread=True, namespace="thread")
def operacion_bloqueante() -> str:
    """
    Operación bloqueante que se ejecutará en un thread separado.
    
    Returns:
        Resultado de la operación
    """
    logger.info("Iniciando operación bloqueante en thread")
    
    # Simulamos una operación bloqueante que tomaría mucho tiempo
    time.sleep(0.5)
    
    resultado = "Operación bloqueante completada"
    logger.info(resultado)
    return resultado

async def ejecutar_muchas_tareas_paralelas(n: int = 20) -> List[Any]:
    """
    Ejecutar muchas tareas en paralelo para probar concurrencia.
    
    Args:
        n: Número de tareas
        
    Returns:
        Lista de resultados
    """
    logger.info(f"Ejecutando {n} tareas en paralelo")
    
    # Crear tareas variadas
    tareas = []
    for i in range(n):
        if i % 4 == 0:
            tareas.append(calculo_complejo(i + 10))
        elif i % 4 == 1:
            tareas.append(operacion_io())
        elif i % 4 == 2:
            tareas.append(operacion_con_error(i % 2 == 0))
        else:
            tareas.append(operacion_bloqueante())
            
    # Esperar a que todas terminen
    resultados = await asyncio.gather(*tareas, return_exceptions=False)
    logger.info(f"Completadas {len(resultados)} tareas paralelas")
    return resultados

async def main():
    """Función principal."""
    logger.info("Iniciando pruebas del procesador asincrónico ultra-cuántico")
    
    # 1. Ejecutar una operación simple
    resultado = await calculo_complejo(50)
    logger.info(f"Resultado del cálculo: {resultado:.2f}")
    
    # 2. Ejecutar operación que falla, pero se transmuta
    try:
        resultado_error = await operacion_con_error()
        logger.info(f"Resultado operación con error (transmutado): {resultado_error}")
    except Exception as e:
        logger.error(f"Error no transmutado: {e}")
    
    # 3. Ejecutar muchas tareas en paralelo
    resultados_paralelos = await ejecutar_muchas_tareas_paralelas(40)
    logger.info(f"Total de resultados paralelos: {len(resultados_paralelos)}")
    
    # 4. Usar contexto de thread
    with quantum_thread_context() as run_in_thread:
        # Esto se ejecuta en un thread separado
        def tarea_pesada():
            logger.info("Ejecutando tarea pesada en thread")
            time.sleep(1)
            return "Completado en thread"
            
        # Esto no bloquea el bucle de eventos
        resultado_thread = await run_in_thread(tarea_pesada)
        logger.info(f"Resultado thread: {resultado_thread}")
    
    # 5. Usar contexto de proceso
    async with quantum_process_context() as run_in_process:
        # Esto se ejecuta en un proceso separado
        def tarea_intensiva(n):
            logger.info(f"Calculando factorial de {n} en proceso separado")
            resultado = 1
            for i in range(2, n + 1):
                resultado *= i
            return resultado
            
        resultado_proceso = await run_in_process(tarea_intensiva, 10)
        logger.info(f"Factorial calculado en proceso: {resultado_proceso}")
    
    logger.info("Pruebas del procesador asincrónico completadas con éxito")

if __name__ == "__main__":
    asyncio.run(main())