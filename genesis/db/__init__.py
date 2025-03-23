"""
Módulo de Base de Datos del Sistema Genesis.

Este módulo proporciona todas las funcionalidades relacionadas con 
la gestión de bases de datos, incluyendo capacidades transcendentales
para máxima resiliencia y rendimiento.
"""
import logging

# Configuración de logging
logger = logging.getLogger("genesis.db")

# Importar extensiones para TranscendentalDatabase
try:
    from genesis.db.transcendental_extension import initialize_extensions
    # Inicializar extensiones automáticamente
    initialize_extensions()
    logger.info("Extensiones de TranscendentalDatabase inicializadas automáticamente")
except Exception as e:
    logger.warning(f"No se pudieron inicializar extensiones: {str(e)}")

# Módulos públicos
__all__ = [
    "base",
    "transcendental_database",
    "divine_database",
    "initializer",
    "config"
]