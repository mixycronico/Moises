"""
Script para crear todas las tablas en la base de datos del sistema Genesis.

Este script lee el archivo SQL con las definiciones de las tablas y
las crea en la base de datos PostgreSQL configurada en el sistema.
"""

import logging
import asyncio
import os
from typing import Optional, List, Dict, Any

from genesis.db.transcendental_database import TranscendentalDatabase

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_tables(sql_path: str) -> bool:
    """
    Crear todas las tablas en la base de datos.
    
    Args:
        sql_path: Ruta al archivo SQL con las definiciones de tablas
        
    Returns:
        True si se crearon correctamente, False en caso contrario
    """
    try:
        # Leer archivo SQL
        logger.info(f"Leyendo archivo SQL: {sql_path}")
        with open(sql_path, 'r') as f:
            sql_commands = f.read()
        
        # Inicializar la base de datos
        logger.info("Inicializando conexión a base de datos")
        db = TranscendentalDatabase()
        
        # Ejecutar comandos SQL
        logger.info("Ejecutando comandos SQL para crear tablas")
        await db.execute(sql_commands)
        
        # Verificar que las tablas se crearon
        logger.info("Verificando tablas creadas")
        tables = await db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        
        if tables:
            logger.info(f"Tablas creadas correctamente: {len(tables)} tablas")
            for table in tables:
                logger.info(f"  - {table[0]}")
            return True
        else:
            logger.error("No se encontraron tablas después de la creación")
            return False
            
    except Exception as e:
        logger.error(f"Error al crear las tablas: {e}")
        return False
        
async def verificar_datos_json(directorio: str) -> List[str]:
    """
    Buscar archivos JSON con datos para importar.
    
    Args:
        directorio: Directorio donde buscar los archivos
        
    Returns:
        Lista de rutas a archivos JSON
    """
    try:
        archivos_json = []
        for archivo in os.listdir(directorio):
            if archivo.endswith('.json') and not archivo == 'genesis_config.json':
                ruta_completa = os.path.join(directorio, archivo)
                if os.path.isfile(ruta_completa):
                    archivos_json.append(ruta_completa)
                    
        logger.info(f"Se encontraron {len(archivos_json)} archivos JSON para importar")
        return archivos_json
    except Exception as e:
        logger.error(f"Error al buscar archivos JSON: {e}")
        return []

async def main():
    """Función principal del script."""
    try:
        logger.info("Iniciando creación de tablas de la base de datos")
        
        # Crear tablas
        success = await create_tables('create_tables.sql')
        
        if success:
            logger.info("Tablas creadas correctamente. Sistema listo para importar datos.")
            
            # Verificar archivos JSON
            json_files = await verificar_datos_json('.')
            if json_files:
                logger.info(f"Archivos JSON listos para importar: {len(json_files)}")
                logger.info("Para importar los datos, ejecute el script import_json_data.py")
            else:
                logger.warning("No se encontraron archivos JSON para importar")
        else:
            logger.error("Error al crear tablas de la base de datos")
            
    except Exception as e:
        logger.error(f"Error en el proceso: {e}")

if __name__ == "__main__":
    # Ejecutar la función principal
    asyncio.run(main())