"""
Script para actualizar la base de datos del sistema Genesis.

Este script elimina todas las tablas existentes y las crea nuevamente
según la estructura definida en los modelos.
"""

import os
import sys
import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Agregar la ruta raíz al path
sys.path.append(".")

# Importar modelos de base de datos
from genesis.db.models import Base

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """
    Recrear todas las tablas en la base de datos.
    
    Esta función elimina todas las tablas existentes y las crea nuevamente
    según la estructura definida en los modelos.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("Variable de entorno DATABASE_URL no definida")
        return False
    
    logger.info(f"Conectando a la base de datos: {database_url}")
    
    try:
        # Crear motor de base de datos
        engine = create_engine(database_url)
        
        # Eliminar todas las tablas
        logger.info("Eliminando tablas existentes...")
        Base.metadata.drop_all(engine)
        
        # Crear todas las tablas
        logger.info("Creando nuevas tablas...")
        Base.metadata.create_all(engine)
        
        logger.info("Base de datos actualizada exitosamente")
        return True
    
    except SQLAlchemyError as e:
        logger.error(f"Error al actualizar la base de datos: {e}")
        return False

if __name__ == "__main__":
    logger.info("Iniciando actualización de la base de datos...")
    if setup_database():
        logger.info("Actualización completada con éxito")
    else:
        logger.error("Error al actualizar la base de datos")
        sys.exit(1)