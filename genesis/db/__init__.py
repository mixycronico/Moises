"""
Módulo de base de datos para el Sistema Genesis.

Este módulo proporciona clases y funciones para interactuar con la base de datos
del Sistema Genesis, con características avanzadas de resiliencia y rendimiento.
"""
from .divine_database import get_divine_db_adapter, divine_db, DivineDatabaseAdapter, DivineCache

__all__ = ['get_divine_db_adapter', 'divine_db', 'DivineDatabaseAdapter', 'DivineCache']