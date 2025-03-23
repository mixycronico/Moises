"""
Ejemplos de uso del importador unificado de datos JSON.

Este script muestra cómo utilizar el importador unificado para
importar datos JSON en diferentes modos y contextos.
"""
import os
import asyncio
import logging
from typing import List, Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("examples.import")

# Importar módulos necesarios
from genesis.db.unified_import import (
    get_unified_importer,
    import_file_sync,
    import_file_async,
    batch_import_files_sync,
    batch_import_files_async
)

# Ejemplo 1: Importar un único archivo de forma síncrona
def ejemplo_importar_archivo_sincrono(file_path: str) -> bool:
    """
    Importar un archivo JSON de forma síncrona.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        True si se importó correctamente, False en caso contrario
    """
    logger.info(f"Ejemplo 1: Importar archivo de forma síncrona: {file_path}")
    success = import_file_sync(file_path)
    logger.info(f"Resultado: {'Éxito' if success else 'Error'}")
    return success

# Ejemplo 2: Importar un único archivo de forma asíncrona
async def ejemplo_importar_archivo_asincrono(file_path: str) -> bool:
    """
    Importar un archivo JSON de forma asíncrona.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        True si se importó correctamente, False en caso contrario
    """
    logger.info(f"Ejemplo 2: Importar archivo de forma asíncrona: {file_path}")
    success = await import_file_async(file_path)
    logger.info(f"Resultado: {'Éxito' if success else 'Error'}")
    return success

# Ejemplo 3: Importar múltiples archivos de forma síncrona
def ejemplo_importar_multiples_archivos_sincronos(file_paths: List[str]) -> Dict[str, bool]:
    """
    Importar múltiples archivos JSON de forma síncrona.
    
    Args:
        file_paths: Lista de rutas a archivos JSON
        
    Returns:
        Diccionario con rutas y resultados
    """
    logger.info(f"Ejemplo 3: Importar {len(file_paths)} archivos de forma síncrona")
    results = batch_import_files_sync(file_paths)
    
    # Mostrar resultados
    for file_path, success in results.items():
        logger.info(f"  - {file_path}: {'Éxito' if success else 'Error'}")
    
    return results

# Ejemplo 4: Importar múltiples archivos de forma asíncrona
async def ejemplo_importar_multiples_archivos_asincronos(file_paths: List[str]) -> Dict[str, bool]:
    """
    Importar múltiples archivos JSON de forma asíncrona.
    
    Args:
        file_paths: Lista de rutas a archivos JSON
        
    Returns:
        Diccionario con rutas y resultados
    """
    logger.info(f"Ejemplo 4: Importar {len(file_paths)} archivos de forma asíncrona")
    results = await batch_import_files_async(file_paths)
    
    # Mostrar resultados
    for file_path, success in results.items():
        logger.info(f"  - {file_path}: {'Éxito' if success else 'Error'}")
    
    return results

# Ejemplo 5: Usar directamente la clase UnifiedImporter
def ejemplo_usar_unified_importer(file_path: str) -> bool:
    """
    Usar directamente la clase UnifiedImporter.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        True si se importó correctamente, False en caso contrario
    """
    logger.info(f"Ejemplo 5: Usar directamente UnifiedImporter: {file_path}")
    importer = get_unified_importer()
    success = importer.import_file_sync(file_path)
    logger.info(f"Resultado: {'Éxito' if success else 'Error'}")
    return success

# Función para ejecutar todos los ejemplos con un archivo de prueba
async def ejecutar_ejemplos(file_path: str, file_paths: List[str]):
    """
    Ejecutar todos los ejemplos.
    
    Args:
        file_path: Ruta a un archivo JSON para ejemplos individuales
        file_paths: Lista de rutas a archivos JSON para ejemplos múltiples
    """
    # Ejemplos síncronos
    ejemplo_importar_archivo_sincrono(file_path)
    ejemplo_importar_multiples_archivos_sincronos(file_paths)
    ejemplo_usar_unified_importer(file_path)
    
    # Ejemplos asíncronos
    await ejemplo_importar_archivo_asincrono(file_path)
    await ejemplo_importar_multiples_archivos_asincronos(file_paths)

async def main():
    """Función principal para ejecutar los ejemplos."""
    # Definir archivo de ejemplo
    resultados_file = os.path.join("datos", "resultados_prueba.json")
    resultados_files = [
        os.path.join("datos", "resultados_prueba.json"),
        os.path.join("datos", "resultados_prueba_extrema.json"),
        os.path.join("datos", "resultados_prueba_simple.json")
    ]
    
    # Verificar que los archivos existen
    if not os.path.isfile(resultados_file):
        logger.error(f"Archivo de ejemplo no encontrado: {resultados_file}")
        logger.info("Por favor, especifique un archivo JSON válido para ejecutar los ejemplos.")
        return
    
    # Filtrar archivos existentes
    existing_files = [f for f in resultados_files if os.path.isfile(f)]
    if not existing_files:
        logger.error("Ninguno de los archivos de ejemplo múltiples fue encontrado")
        logger.info("Por favor, especifique archivos JSON válidos para ejecutar los ejemplos.")
        return
    
    # Ejecutar ejemplos
    await ejecutar_ejemplos(resultados_file, existing_files)

if __name__ == "__main__":
    asyncio.run(main())