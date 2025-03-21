"""
Script para actualizar la base de datos del sistema Genesis.

Este script elimina todas las tablas existentes y las crea nuevamente
según la estructura definida en los modelos.
"""

import os
import sys
import asyncio
import logging

# Agregar la ruta raíz al path
sys.path.append(".")

# Importar modelos y repositorio
from genesis.db.models import Base
from genesis.db.repository import Repository
from genesis.db.paper_trading_models import PaperTradingAccount

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_database():
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
        # Crear repositorio
        repo = Repository(connection_string=database_url)
        
        # Eliminar todas las tablas
        logger.info("Eliminando tablas existentes...")
        await repo.drop_tables(Base)
        
        # Crear todas las tablas
        logger.info("Creando nuevas tablas...")
        await repo.create_tables(Base)
        
        # Verificar que las tablas de paper trading también se hayan creado
        logger.info("Verificando tablas de paper trading...")
        try:
            count = await repo.count(PaperTradingAccount)
            logger.info(f"Tablas de paper trading creadas correctamente. Cuentas existentes: {count}")
        except Exception as e:
            logger.warning(f"No se pudo verificar las tablas de paper trading: {e}")
        
        logger.info("Base de datos actualizada exitosamente")
        return True
    
    except Exception as e:
        logger.error(f"Error al actualizar la base de datos: {e}")
        return False

if __name__ == "__main__":
    logger.info("Iniciando actualización de la base de datos...")
    try:
        success = asyncio.run(setup_database())
        if success:
            logger.info("Actualización completada con éxito")
        else:
            logger.error("Error al actualizar la base de datos")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)