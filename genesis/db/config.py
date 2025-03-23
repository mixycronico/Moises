"""
Configuración para el módulo de base de datos transcendental.

Este módulo proporciona funciones y constantes para la configuración
del sistema de base de datos, incluyendo variables de entorno, valores
por defecto y utilidades para recuperar configuraciones.
"""
import os
import logging
from typing import Optional, Dict, Any

# Configuración de logging
logger = logging.getLogger("genesis.db.config")

# Modos transcendentales disponibles
TRASCENDENTAL_MODES = [
    "SINGULARITY_V4",  # Modo por defecto y más avanzado
    "LIGHT",           # Modo Luz
    "DARK_MATTER",     # Modo Materia Oscura
    "DIVINE",          # Modo Divino
    "BIG_BANG",        # Modo Big Bang
    "INTERDIMENSIONAL" # Modo Interdimensional
]

# Valores por defecto para la configuración de base de datos
DEFAULT_DB_CONFIG = {
    "pool_size": 20,
    "max_overflow": 40,
    "pool_recycle": 300,
    "pool_pre_ping": True,
    "connect_args": {"command_timeout": 10},
    "trascendental_mode": "SINGULARITY_V4"
}

def get_database_url() -> str:
    """
    Obtener URL de conexión a la base de datos desde variables de entorno.
    
    Returns:
        URL de conexión
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.warning("Variable de entorno DATABASE_URL no definida, usando valor por defecto")
        # URL por defecto para desarrollo local
        database_url = "postgresql://postgres:postgres@localhost:5432/genesis"
    
    return database_url

def normalize_database_url(url: str) -> str:
    """
    Normalizar URL de base de datos para asegurar compatibilidad.
    
    Args:
        url: URL de conexión original
        
    Returns:
        URL normalizada
    """
    # Convertir URL de Heroku a formato SQLAlchemy
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    return url

def get_db_config(config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Obtener configuración completa de base de datos.
    
    Args:
        config_dict: Diccionario opcional con configuración personalizada
        
    Returns:
        Configuración completa con valores por defecto para campos no especificados
    """
    result = DEFAULT_DB_CONFIG.copy()
    
    # Actualizar con valores proporcionados
    if config_dict:
        result.update(config_dict)
    
    # Asegurar que el modo trascendental es válido
    if result.get("trascendental_mode") not in TRASCENDENTAL_MODES:
        logger.warning(f"Modo trascendental '{result.get('trascendental_mode')}' no válido, usando SINGULARITY_V4")
        result["trascendental_mode"] = "SINGULARITY_V4"
    
    return result