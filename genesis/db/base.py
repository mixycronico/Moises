"""
Base para todos los modelos SQLAlchemy en Genesis.

Este módulo proporciona la clase Base para todos los modelos SQLAlchemy
con capacidades trascendentales para operaciones resilientes.
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import registry

# Crear mapeador con capacidades trascendentales
mapper_registry = registry()

# Clase base para todos los modelos trascendentales
Base = mapper_registry.generate_base()

# Metadatos compartidos para todas las tablas
metadata = Base.metadata