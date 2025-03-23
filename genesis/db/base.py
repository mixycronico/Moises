"""
Base para todos los modelos SQLAlchemy en Genesis.

Este módulo proporciona la clase Base para todos los modelos SQLAlchemy
con capacidades trascendentales para operaciones resilientes.
"""

from sqlalchemy.orm import DeclarativeBase

# Clase base para todos los modelos trascendentales
class Base(DeclarativeBase):
    """Clase base para todos los modelos con capacidades trascendentales."""
    pass

# Metadatos compartidos para todas las tablas
metadata = Base.metadata