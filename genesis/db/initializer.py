"""
Inicializador de base de datos para el Sistema Genesis.

Este módulo proporciona funciones para inicializar la conexión
a la base de datos y configurarla para su uso con el sistema.
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional

from .timescaledb_adapter import TimescaleDBManager
from .db_integration import DatabaseAdapter

# Singleton para mantener la instancia global
_db_adapter = None
_db_manager = None
_initialized = False

async def initialize_database(dsn: Optional[Dict[str, Any]] = None) -> DatabaseAdapter:
    """
    Inicializar adaptador de base de datos.
    
    Args:
        dsn: Configuración de base de datos (opcional)
        
    Returns:
        Adaptador de base de datos configurado
    """
    global _db_adapter, _db_manager, _initialized
    
    logger = logging.getLogger(__name__)
    
    # Si ya está inicializado, devolver instancia existente
    if _initialized and _db_adapter:
        logger.info("Base de datos ya inicializada")
        return _db_adapter
    
    # DSN para conexión a PostgreSQL
    if not dsn:
        dsn = os.environ.get('DATABASE_URL')
        if not dsn:
            logger.warning("Variable DATABASE_URL no definida, usando conexión por defecto")
            dsn = "postgresql://postgres:postgres@localhost:5432/genesis"
    
    # Si dsn es un diccionario, extraer la URL
    if isinstance(dsn, dict):
        dsn_url = dsn.get('url', os.environ.get('DATABASE_URL', "postgresql://postgres:postgres@localhost:5432/genesis"))
        logger.info(f"Iniciando conexión a base de datos (desde config): {dsn_url.split('@')[-1] if '@' in dsn_url else dsn_url}")
        dsn = dsn_url
    else:
        # Es una cadena normal
        logger.info(f"Iniciando conexión a base de datos: {dsn.split('@')[-1] if '@' in dsn else dsn}")
    
    try:
        # Crear gestor de TimescaleDB
        db_manager = TimescaleDBManager(
            dsn=dsn,
            max_workers=2,
            retry_attempts=3,
            enable_checkpoints=True
        )
        
        # Crear adaptador
        db_adapter = DatabaseAdapter(db_manager=db_manager)
        
        # Guardar instancias globales
        _db_manager = db_manager
        _db_adapter = db_adapter
        _initialized = True
        
        logger.info("Base de datos inicializada correctamente")
        
        return db_adapter
        
    except Exception as e:
        logger.error(f"Error inicializando base de datos: {str(e)}")
        raise

def get_db_adapter() -> Optional[DatabaseAdapter]:
    """
    Obtener adaptador de base de datos global.
    
    Returns:
        Adaptador de base de datos o None si no está inicializado
    """
    global _db_adapter
    return _db_adapter

def get_db_manager() -> Optional[TimescaleDBManager]:
    """
    Obtener gestor de TimescaleDB global.
    
    Returns:
        Gestor de TimescaleDB o None si no está inicializado
    """
    global _db_manager
    return _db_manager

def shutdown_database() -> bool:
    """
    Cerrar conexiones a la base de datos.
    
    Returns:
        True si se cerró correctamente
    """
    global _db_adapter, _db_manager, _initialized
    
    logger = logging.getLogger(__name__)
    
    if not _initialized:
        logger.warning("Base de datos no inicializada")
        return False
    
    try:
        # Cerrar adaptador
        if _db_adapter:
            _db_adapter.shutdown()
        
        # Restablecer variables globales
        _db_adapter = None
        _db_manager = None
        _initialized = False
        
        logger.info("Base de datos cerrada correctamente")
        return True
        
    except Exception as e:
        logger.error(f"Error cerrando base de datos: {str(e)}")
        return False

async def test_database_connection(dsn: Optional[str] = None) -> Dict[str, Any]:
    """
    Probar conexión a la base de datos.
    
    Args:
        dsn: Cadena de conexión (opcional)
        
    Returns:
        Diccionario con resultado de la prueba
    """
    logger = logging.getLogger(__name__)
    
    if not dsn:
        dsn = os.environ.get('DATABASE_URL')
        if not dsn:
            logger.warning("Variable DATABASE_URL no definida, usando conexión por defecto")
            dsn = "postgresql://postgres:postgres@localhost:5432/genesis"
    
    # Si dsn es un diccionario, extraer la URL
    if isinstance(dsn, dict):
        dsn_url = dsn.get('url', os.environ.get('DATABASE_URL', "postgresql://postgres:postgres@localhost:5432/genesis"))
        logger.info(f"Probando conexión a base de datos (desde config): {dsn_url.split('@')[-1] if '@' in dsn_url else dsn_url}")
        dsn = dsn_url
    else:
        # Es una cadena normal
        logger.info(f"Probando conexión a base de datos: {dsn.split('@')[-1] if '@' in dsn else dsn}")
    
    try:
        # Crear gestor temporal
        db_manager = TimescaleDBManager(dsn=dsn)
        
        # Probar conexión
        conn = db_manager.connect()
        
        if not conn:
            return {
                'success': False,
                'message': "No se pudo establecer conexión a la base de datos"
            }
        
        # Cerrar conexión
        conn.close()
        
        # Intentar crear hipertablas
        tables_ok = db_manager.setup_hypertables()
        
        # Crear adaptador temporal
        db_adapter = DatabaseAdapter(db_manager=db_manager, initialize=False)
        
        # Probar consulta
        query_result = await db_adapter.fetch("SELECT NOW() as current_time", as_dict=True)
        
        current_time = query_result[0]['current_time'] if query_result else None
        
        # Cerrar adaptador
        db_adapter.shutdown()
        
        return {
            'success': True,
            'message': "Conexión exitosa a la base de datos",
            'details': {
                'current_time': current_time,
                'tables_configured': tables_ok
            }
        }
        
    except Exception as e:
        logger.error(f"Error probando conexión a base de datos: {str(e)}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }