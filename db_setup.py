"""
Script para actualizar la base de datos del sistema Genesis.

Este script elimina todas las tablas existentes y las crea nuevamente
según la estructura definida en los modelos.
"""

import os
import sys
import asyncio
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db_setup")

# Importar modelos
from genesis.db.base import Base
from genesis.db.paper_trading_models import *
from genesis.db.models.crypto_classifier_models import *
from genesis.db.models.scaling_config_models import *

async def setup_database():
    """
    Recrear todas las tablas en la base de datos.
    
    Esta función elimina todas las tablas existentes y las crea nuevamente
    según la estructura definida en los modelos.
    """
    # Obtener URL de la base de datos
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        logger.error("No se encontró la variable de entorno DATABASE_URL")
        return False
    
    logger.info(f"Conectando a la base de datos: {db_url}")
    
    try:
        # Crear motor de base de datos
        engine = create_engine(db_url)
        
        # Eliminar todas las tablas existentes relacionadas con paper trading
        logger.info("Eliminando tablas existentes...")
        Base.metadata.drop_all(engine)
        logger.info("Tablas eliminadas correctamente")
        
        # Crear tablas nuevamente
        logger.info("Creando tablas nuevas...")
        Base.metadata.create_all(engine)
        logger.info("Tablas creadas correctamente")
        
        return True
        
    except Exception as e:
        logger.error(f"Error al actualizar la base de datos: {e}")
        return False

async def main():
    """Función principal."""
    success = await setup_database()
    
    if success:
        logger.info("Base de datos actualizada correctamente")
    else:
        logger.error("Error al actualizar la base de datos")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())