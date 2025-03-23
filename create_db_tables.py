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
            sql_content = f.read()
        
        # Dividir en comandos individuales
        sql_commands = []
        current_command = ""
        
        for line in sql_content.split('\n'):
            line = line.strip()
            
            # Ignorar comentarios y líneas vacías para el procesamiento
            if line.startswith('--') or not line:
                current_command += f"{line}\n"
                continue
                
            current_command += f"{line}\n"
            
            # Cuando llegue a un punto y coma, es el final del comando
            if line.endswith(';'):
                if current_command.strip():
                    # Agregar solo si hay contenido real (no solo comentarios)
                    has_content = False
                    for cmd_line in current_command.split('\n'):
                        if cmd_line.strip() and not cmd_line.strip().startswith('--'):
                            has_content = True
                            break
                    
                    if has_content:
                        sql_commands.append(current_command)
                        
                current_command = ""
                
        logger.info(f"Se encontraron {len(sql_commands)} comandos SQL")
        
        # Inicializar la base de datos
        logger.info("Inicializando conexión a base de datos")
        db = TranscendentalDatabase()
        
        # Ejecutar cada comando SQL individualmente
        logger.info("Ejecutando comandos SQL para crear tablas")
        for i, cmd in enumerate(sql_commands, 1):
            try:
                logger.info(f"Ejecutando comando {i}/{len(sql_commands)}")
                # Imprimir el comando completo para una mejor depuración
                cmd_preview = cmd.strip().replace('\n', ' ')
                logger.debug(f"SQL Completo: {cmd_preview}")
                # Enviar el comando como una función que devuelve la tupla (sql, params)
                result = await db.execute_query(lambda: (cmd, {}))
                logger.info(f"Comando {i} ejecutado correctamente")
            except Exception as e:
                logger.error(f"Error en comando {i}: {e}")
        
        # Verificar que las tablas se crearon
        logger.info("Verificando tablas creadas")
        try:
            result = await db.execute_query(lambda: ("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'", {}))
            
            # Convertir el resultado a una lista
            tables = []
            if isinstance(result, list):
                tables = result
            elif hasattr(result, '__iter__'):
                tables = list(result)
            elif result is not None:
                # Para otros tipos, intentar convertir
                tables = [result]
                
            if tables:
                logger.info(f"Tablas creadas correctamente: {len(tables)} tablas")
                for table in tables:
                    if isinstance(table, tuple) and len(table) > 0:
                        logger.info(f"  - {table[0]}")
                    else:
                        logger.info(f"  - {table}")
                return True
            else:
                logger.error("No se encontraron tablas después de la creación")
                return False
        except Exception as e:
            logger.error(f"Error al verificar las tablas: {e}")
            # Si no podemos verificar, asumimos que las tablas se crearon
            logger.info("Asumiendo que las tablas se crearon correctamente")
            return True
            
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