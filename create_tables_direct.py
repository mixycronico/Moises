"""
Script para crear una tabla directamente usando SQLAlchemy.
"""

import asyncio
import logging
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import text

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_sample_table():
    """Crear una tabla de prueba directamente."""
    try:
        # Conectar a la base de datos
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            logger.error("No se encontró la variable de entorno DATABASE_URL")
            return False
            
        # Crear motor SQLAlchemy
        engine = create_async_engine(db_url)
        
        # Crear tabla de prueba
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS gen_test_table (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            value FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        async with engine.begin() as conn:
            await conn.execute(text(create_table_sql))
            
        logger.info("Tabla de prueba creada correctamente")
        
        # Verificar que la tabla existe
        check_sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'gen_test_table'"
        
        async with engine.connect() as conn:
            result = await conn.execute(text(check_sql))
            tables = [row[0] for row in result]
            
        if 'gen_test_table' in tables:
            logger.info("Verificación exitosa: la tabla gen_test_table existe")
            return True
        else:
            logger.error("La tabla gen_test_table no se encontró en la base de datos")
            return False
            
    except Exception as e:
        logger.error(f"Error al crear la tabla de prueba: {e}")
        return False

async def main():
    """Función principal."""
    success = await create_sample_table()
    if success:
        logger.info("Creación de tabla exitosa")
    else:
        logger.error("No se pudo crear la tabla")

if __name__ == "__main__":
    asyncio.run(main())