#!/usr/bin/env python3
"""
Script unificado para importar datos JSON en la base de datos del sistema Genesis.

Este script puede operar tanto en modo síncrono como asíncrono y maneja todos los tipos
de datos compatibles con las tablas gen_* del sistema Genesis.

Uso:
    python import_data_unified.py [--async] [--dir DIRECTORIO] [archivos...]
"""
import os
import sys
import glob
import logging
import argparse
import asyncio
from typing import List, Dict, Any

from genesis.db.unified_import import import_file_sync, import_file_async, batch_import_files_sync, batch_import_files_async

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("import_data")

def find_json_files(directory: str) -> List[str]:
    """
    Buscar archivos JSON en un directorio.
    
    Args:
        directory: Directorio donde buscar
        
    Returns:
        Lista de rutas a archivos JSON
    """
    if not os.path.isdir(directory):
        logger.error(f"El directorio no existe: {directory}")
        return []
    
    json_files = glob.glob(os.path.join(directory, "*.json"))
    logger.info(f"Se encontraron {len(json_files)} archivos JSON en {directory}")
    return json_files

async def import_async(files: List[str], concurrent: bool = True) -> Dict[str, bool]:
    """
    Importar archivos JSON de forma asíncrona.
    
    Args:
        files: Lista de archivos a importar
        concurrent: Si se deben procesar en paralelo
        
    Returns:
        Diccionario con rutas y resultados
    """
    if not files:
        logger.warning("No hay archivos para importar")
        return {}
    
    if concurrent:
        # Importar archivos en paralelo
        logger.info(f"Importando {len(files)} archivos en paralelo...")
        results = await batch_import_files_async(files)
    else:
        # Importar archivos en secuencia
        logger.info(f"Importando {len(files)} archivos secuencialmente...")
        results = {}
        for file_path in files:
            logger.info(f"Importando {file_path}...")
            success = await import_file_async(file_path)
            results[file_path] = success
    
    return results

def import_sync(files: List[str], concurrent: bool = False) -> Dict[str, bool]:
    """
    Importar archivos JSON de forma síncrona.
    
    Args:
        files: Lista de archivos a importar
        concurrent: No se utiliza en modo síncrono
        
    Returns:
        Diccionario con rutas y resultados
    """
    if not files:
        logger.warning("No hay archivos para importar")
        return {}
    
    logger.info(f"Importando {len(files)} archivos...")
    
    if concurrent:
        # Importar archivos mediante batch síncrono
        results = batch_import_files_sync(files)
    else:
        # Importar archivos en secuencia
        results = {}
        for file_path in files:
            logger.info(f"Importando {file_path}...")
            success = import_file_sync(file_path)
            results[file_path] = success
    
    return results

async def main_async():
    """Función principal para modo asíncrono."""
    parser = argparse.ArgumentParser(description="Importar datos JSON en la base de datos (modo asíncrono)")
    parser.add_argument("files", nargs="*", help="Archivos JSON a importar")
    parser.add_argument("--dir", "-d", help="Directorio con archivos JSON a importar")
    parser.add_argument("--concurrent", "-c", action="store_true", help="Procesar archivos en paralelo")
    
    args = parser.parse_args()
    
    # Recopilar archivos
    files_to_import = list(args.files)
    if args.dir:
        dir_files = find_json_files(args.dir)
        files_to_import.extend(dir_files)
    
    # Importar archivos
    if not files_to_import:
        logger.error("No se especificaron archivos para importar")
        sys.exit(1)
    
    results = await import_async(files_to_import, args.concurrent)
    
    # Imprimir resultados
    success_count = sum(1 for result in results.values() if result)
    logger.info(f"Importación completada: {success_count}/{len(results)} archivos exitosos")
    
    if success_count < len(results):
        for file_path, success in results.items():
            if not success:
                logger.error(f"Error al importar: {file_path}")
        sys.exit(1)
    
    sys.exit(0)

def main_sync():
    """Función principal para modo síncrono."""
    parser = argparse.ArgumentParser(description="Importar datos JSON en la base de datos (modo síncrono)")
    parser.add_argument("files", nargs="*", help="Archivos JSON a importar")
    parser.add_argument("--dir", "-d", help="Directorio con archivos JSON a importar")
    parser.add_argument("--concurrent", "-c", action="store_true", help="Procesar archivos en bloque")
    
    args = parser.parse_args()
    
    # Recopilar archivos
    files_to_import = list(args.files)
    if args.dir:
        dir_files = find_json_files(args.dir)
        files_to_import.extend(dir_files)
    
    # Importar archivos
    if not files_to_import:
        logger.error("No se especificaron archivos para importar")
        sys.exit(1)
    
    results = import_sync(files_to_import, args.concurrent)
    
    # Imprimir resultados
    success_count = sum(1 for result in results.values() if result)
    logger.info(f"Importación completada: {success_count}/{len(results)} archivos exitosos")
    
    if success_count < len(results):
        for file_path, success in results.items():
            if not success:
                logger.error(f"Error al importar: {file_path}")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    # Determinar modo de operación por argumentos
    if "--async" in sys.argv:
        sys.argv.remove("--async")
        asyncio.run(main_async())
    else:
        main_sync()