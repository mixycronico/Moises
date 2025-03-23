"""
Inicializador central para la base de datos del Sistema Genesis.

Este módulo proporciona funciones para inicializar y configurar
correctamente todas las conexiones de base de datos utilizadas en
el sistema Genesis, garantizando la consistencia y resiliencia.
"""
import logging
import asyncio
import time
from typing import Dict, Any, Optional, Tuple

from genesis.db.config import get_database_url, get_db_config
# Importar extensiones para TranscendentalDatabase
from genesis.db.transcendental_extension import initialize_extensions
from genesis.db.base import db_manager
from genesis.db.transcendental_database import transcendental_db

# Configuración de logging central
logger = logging.getLogger("genesis.db.initializer")

async def initialize_database(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Inicializar todas las conexiones de base de datos del sistema.
    
    Esta función centraliza la inicialización de la base de datos,
    configurando tanto el db_manager como la transcendental_db.
    
    Args:
        config: Configuración personalizada opcional
        
    Returns:
        True si la inicialización fue exitosa, False en caso contrario
    """
    try:
        logger.info("Iniciando inicialización de base de datos")
        start_time = time.time()
        
        # Obtener configuración, usando la personalizada si se proporcionó
        if config is None:
            config = {}
        
        # Configuración del pool con valores por defecto
        db_config = get_db_config(config)
        pool_size = db_config.get("pool_size", 20)
        max_overflow = db_config.get("max_overflow", 40)
        pool_recycle = db_config.get("pool_recycle", 300)
        
        # Configurar DatabaseManager
        db_manager.pool_size = pool_size
        db_manager.max_overflow = max_overflow
        db_manager.pool_recycle = pool_recycle
        
        # Configurar URL de base de datos
        db_url = config.get("database_url", get_database_url())
        # Configurar URL en db_manager si tiene el atributo o método adecuado
        if hasattr(db_manager, 'setup_url'):
            db_manager.setup_url(db_url)
        elif hasattr(db_manager, 'database_url'):
            db_manager.database_url = db_url
        elif hasattr(db_manager, 'set_url'):
            db_manager.set_url(db_url)
        # En cualquier caso, la URL se usará en setup()
        
        # Configurar y verificar conexiones
        # No intentamos configurar database_url directamente, usamos métodos públicos
        
        # Inicializar motor principal
        db_manager.setup()
        
        # Para transcendental_db usamos métodos seguros
        if hasattr(transcendental_db, 'configure'):
            transcendental_db.configure(db_url)
        elif hasattr(transcendental_db, 'set_connection_url'):
            transcendental_db.set_connection_url(db_url)
        
        # Verificar conexión
        connection_status = await test_connection()
        if not connection_status:
            logger.error("No se pudo establecer conexión con la base de datos")
            return False
        
        # Crear tablas si es necesario
        await db_manager.create_all_tables()
        
        end_time = time.time()
        logger.info(f"Base de datos inicializada con éxito en {end_time - start_time:.2f} segundos")
        return True
        
    except Exception as e:
        logger.error(f"Error durante la inicialización de la base de datos: {str(e)}")
        return False

async def test_connection() -> bool:
    """
    Probar la conexión a la base de datos.
    
    Returns:
        True si la conexión es exitosa, False en caso contrario
    """
    try:
        if not hasattr(db_manager, "engine") or db_manager.engine is None:
            logger.error("Motor de base de datos no inicializado")
            return False
            
        # Intentar ejecutar una consulta simple
        async with db_manager.engine.connect() as conn:
            # Consultamos la versión de PostgreSQL
            result = await conn.execute("SELECT version()")
            version = await result.scalar()
            
        if version:
            logger.info(f"Conexión a base de datos exitosa: {version}")
            return True
        else:
            logger.warning("Conexión a base de datos establecida pero sin datos")
            return True
            
    except Exception as e:
        logger.error(f"Error al probar conexión a base de datos: {str(e)}")
        return False

async def get_db_status() -> Dict[str, Any]:
    """
    Obtener el estado actual de la base de datos.
    
    Returns:
        Diccionario con información del estado
    """
    status = {
        "connected": False,
        "pool_status": {},
        "transcendental_db_status": {},
        "table_count": 0
    }
    
    try:
        # Verificar conexión
        status["connected"] = await test_connection()
        
        # Obtener estadísticas del pool
        if hasattr(db_manager, "engine") and db_manager.engine is not None:
            if hasattr(db_manager.engine, "pool"):
                pool = db_manager.engine.pool
                status["pool_status"] = {
                    "size": getattr(pool, "size", 0),
                    "overflow": getattr(pool, "overflow", 0),
                    "checkedin": getattr(pool, "checkedin", 0),
                    "checkedout": getattr(pool, "checkedout", 0)
                }
        
        # Obtener estado de transcendental_db
        status["transcendental_db_status"] = transcendental_db.get_stats()
        
        # Contar tablas en la base de datos
        if status["connected"]:
            try:
                async with db_manager.engine.connect() as conn:
                    result = await conn.execute(
                        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
                    )
                    status["table_count"] = await result.scalar()
            except Exception as e:
                logger.warning(f"Error al contar tablas: {str(e)}")
                status["table_count"] = -1
    
    except Exception as e:
        logger.error(f"Error al obtener estado de base de datos: {str(e)}")
    
    return status

async def cleanup_database() -> None:
    """
    Limpiar y cerrar conexiones de base de datos.
    
    Esta función debe llamarse al finalizar la aplicación para cerrar
    correctamente todas las conexiones.
    """
    try:
        if hasattr(db_manager, "engine") and db_manager.engine is not None:
            await db_manager.engine.dispose()
            logger.info("Conexiones de base de datos cerradas correctamente")
    except Exception as e:
        logger.error(f"Error al cerrar conexiones de base de datos: {str(e)}")