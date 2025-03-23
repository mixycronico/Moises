"""
Módulo para interacción con la base de datos en el Sistema Genesis.

Este módulo proporciona interfaces para trabajar con la base de datos PostgreSQL,
tanto de forma asíncrona como síncrona, facilitando la integración con diferentes
componentes del sistema.
"""

from genesis.db.base import db_manager, get_db_session
from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.db.sync_database import SyncDatabase
from genesis.db.database_adapter import DatabaseAdapter, get_db_adapter

__all__ = [
    'db_manager',
    'get_db_session',
    'TranscendentalDatabase',
    'SyncDatabase',
    'DatabaseAdapter',
    'get_db_adapter',
]