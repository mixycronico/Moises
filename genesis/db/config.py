"""
Configuración de bases de datos para el Sistema Genesis.

Este módulo proporciona configuraciones centralizadas para todas las conexiones
de bases de datos utilizadas en el sistema, incluyendo parámetros para pools
de conexiones, tiempos de expiración y credenciales.
"""
import os
import logging
from typing import Dict, Any, Optional

# Configurar logger
logger = logging.getLogger("genesis.db.config")

# URL de base de datos desde variable de entorno o valor por defecto
DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/genesis"
)

# Configuración de pool de conexiones
DEFAULT_POOL_CONFIG = {
    "pool_size": 20,
    "max_overflow": 40,
    "pool_recycle": 300,  # segundos
    "pool_timeout": 30,
    "pool_pre_ping": True
}

# Configuración de timeout para operaciones
OPERATION_TIMEOUT = 5.0  # segundos

# Configuración para reintentos de conexión
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 0.1,  # segundos
    "max_delay": 1.0,  # segundos
    "jitter": 0.05  # factor de aleatoriedad para evitar tormentas de reintentos
}

# Configuración para cache de conexiones
CACHE_CONFIG = {
    "max_size": 100,
    "ttl": 60  # segundos
}

# Configuración para checkpoints de estado
CHECKPOINT_CONFIG = {
    "enabled": True,
    "interval": 300,  # segundos
    "max_checkpoints": 10
}

def get_database_url() -> str:
    """
    Obtener la URL de conexión a la base de datos.
    
    Returns:
        URL de conexión a la base de datos
    """
    return DATABASE_URL

def get_pool_config() -> Dict[str, Any]:
    """
    Obtener configuración para pool de conexiones.
    
    Returns:
        Diccionario con configuración de pool
    """
    return DEFAULT_POOL_CONFIG.copy()

def get_full_db_config() -> Dict[str, Any]:
    """
    Obtener configuración completa de base de datos.
    
    Returns:
        Diccionario con todas las configuraciones
    """
    return {
        "database_url": get_database_url(),
        "pool_config": get_pool_config(),
        "operation_timeout": OPERATION_TIMEOUT,
        "retry_config": RETRY_CONFIG.copy(),
        "cache_config": CACHE_CONFIG.copy(),
        "checkpoint_config": CHECKPOINT_CONFIG.copy()
    }