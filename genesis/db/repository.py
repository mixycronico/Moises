"""
Database repository module for data access.

Este módulo proporciona una clase repositorio para acceder a la base de datos
siguiendo el patrón Repository.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Type, TypeVar, Union, Tuple

from sqlalchemy import create_engine, text, select, update, delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.future import select

from genesis.db.models import Base
from genesis.config.settings import settings
from genesis.utils.logger import setup_logging

# Tipo genérico para modelos
T = TypeVar('T')

class Repository:
    """
    Repositorio para acceso a datos de la base de datos.
    
    Esta clase proporciona métodos CRUD y consultas para operaciones de base de datos.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Inicializar el repositorio.
        
        Args:
            connection_string: String de conexión a la base de datos (opcional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Obtener string de conexión
        if not connection_string:
            connection_string = os.environ.get("DATABASE_URL") or settings.get('database.connection_string')
            if not connection_string:
                connection_string = "sqlite:///data/genesis.db"
                
        self.connection_string = connection_string
        
        # Crear motor de base de datos
        self.engine = create_engine(
            self.connection_string,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"check_same_thread": False} if "sqlite" in connection_string else {}
        )
        
        # Crear fábrica de sesiones
        self.session_factory = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session = scoped_session(self.session_factory)
        
        # Configurar motor asíncrono si se usa PostgreSQL
        if connection_string.startswith('postgresql'):
            async_connection_string = connection_string.replace('postgresql', 'postgresql+asyncpg')
            self.async_engine = create_async_engine(
                async_connection_string,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            self.async_session_factory = sessionmaker(
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                bind=self.async_engine
            )
        else:
            self.async_engine = None
            self.async_session_factory = None
            
    async def create_tables(self, base_class: Any) -> None:
        """
        Crear todas las tablas en la base de datos.
        
        Args:
            base_class: Clase base declarativa de SQLAlchemy
        """
        try:
            base_class.metadata.create_all(self.engine)
            self.logger.info("Tablas creadas correctamente en la base de datos")
        except Exception as e:
            self.logger.error(f"Error al crear tablas: {e}")
            raise
            
    async def drop_tables(self, base_class: Any) -> None:
        """
        Eliminar todas las tablas de la base de datos.
        
        Args:
            base_class: Clase base declarativa de SQLAlchemy
        """
        try:
            base_class.metadata.drop_all(self.engine)
            self.logger.info("Tablas eliminadas correctamente de la base de datos")
        except Exception as e:
            self.logger.error(f"Error al eliminar tablas: {e}")
            raise
            
    async def get_by_id(self, model_class: Type[T], id: int) -> Optional[T]:
        """
        Obtener una entidad por su ID.
        
        Args:
            model_class: Clase del modelo
            id: ID de la entidad
            
        Returns:
            Instancia de la entidad o None si no se encuentra
        """
        session = self.session()
        try:
            return session.query(model_class).filter(model_class.id == id).first()
        except Exception as e:
            self.logger.error(f"Error al obtener {model_class.__name__} con ID {id}: {e}")
            return None
        finally:
            session.close()
            
    async def get_by_field(self, model_class: Type[T], field: str, value: Any) -> Optional[T]:
        """
        Obtener una entidad por el valor de un campo.
        
        Args:
            model_class: Clase del modelo
            field: Nombre del campo
            value: Valor a buscar
            
        Returns:
            Instancia de la entidad o None si no se encuentra
        """
        session = self.session()
        try:
            return session.query(model_class).filter(getattr(model_class, field) == value).first()
        except Exception as e:
            self.logger.error(f"Error al obtener {model_class.__name__} con {field}={value}: {e}")
            return None
        finally:
            session.close()
            
    async def get_filtered_one(self, model_class: Type[T], filter_expr: str) -> Optional[T]:
        """
        Obtener una entidad utilizando una expresión de filtro.
        
        Args:
            model_class: Clase del modelo
            filter_expr: Expresión de filtro (SQL)
            
        Returns:
            Instancia de la entidad o None si no se encuentra
        """
        session = self.session()
        try:
            # Convertir string SQL a text()
            filter_sql = text(filter_expr)
            return session.query(model_class).filter(filter_sql).first()
        except Exception as e:
            self.logger.error(f"Error al obtener {model_class.__name__} con filtro {filter_expr}: {e}")
            return None
        finally:
            session.close()
            
    async def query(self, model_class: Type[T], filter_expr: Optional[str] = None) -> List[T]:
        """
        Realizar una consulta con un filtro opcional.
        
        Args:
            model_class: Clase del modelo
            filter_expr: Expresión de filtro (SQL) opcional
            
        Returns:
            Lista de instancias que coinciden con el filtro
        """
        session = self.session()
        try:
            if filter_expr:
                # Convertir string SQL a text()
                filter_sql = text(filter_expr)
                return session.query(model_class).filter(filter_sql).all()
            else:
                return session.query(model_class).all()
        except Exception as e:
            self.logger.error(f"Error al consultar {model_class.__name__} con filtro {filter_expr}: {e}")
            return []
        finally:
            session.close()
            
    async def execute_query(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Ejecutar una consulta SQL personalizada.
        
        Args:
            query: Consulta SQL
            
        Returns:
            Resultados de la consulta o None en caso de error
        """
        session = self.session()
        try:
            result = session.execute(text(query))
            if result.returns_rows:
                # Convertir a lista de diccionarios
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result]
            return None
        except Exception as e:
            self.logger.error(f"Error al ejecutar consulta personalizada: {e}")
            return None
        finally:
            session.close()
            
    async def create(self, entity: T) -> int:
        """
        Crear una nueva entidad.
        
        Args:
            entity: Instancia de la entidad a crear
            
        Returns:
            ID de la entidad creada
        """
        session = self.session()
        try:
            session.add(entity)
            session.commit()
            session.refresh(entity)
            return entity.id
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error al crear entidad {type(entity).__name__}: {e}")
            raise
        finally:
            session.close()
            
    async def update(self, entity: T) -> bool:
        """
        Actualizar una entidad existente.
        
        Args:
            entity: Instancia de la entidad a actualizar
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        session = self.session()
        try:
            session.merge(entity)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error al actualizar entidad {type(entity).__name__}: {e}")
            return False
        finally:
            session.close()
            
    async def delete(self, entity: T) -> bool:
        """
        Eliminar una entidad.
        
        Args:
            entity: Instancia de la entidad a eliminar
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        session = self.session()
        try:
            session.delete(entity)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error al eliminar entidad {type(entity).__name__}: {e}")
            return False
        finally:
            session.close()
            
    async def delete_by_id(self, model_class: Type[T], id: int) -> bool:
        """
        Eliminar una entidad por su ID.
        
        Args:
            model_class: Clase del modelo
            id: ID de la entidad a eliminar
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        session = self.session()
        try:
            entity = session.query(model_class).filter(model_class.id == id).first()
            if entity:
                session.delete(entity)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error al eliminar {model_class.__name__} con ID {id}: {e}")
            return False
        finally:
            session.close()
            
    async def bulk_create(self, entities: List[T]) -> bool:
        """
        Crear múltiples entidades en una sola operación.
        
        Args:
            entities: Lista de entidades a crear
            
        Returns:
            True si se crearon correctamente, False en caso contrario
        """
        if not entities:
            return True
            
        session = self.session()
        try:
            session.add_all(entities)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error al crear múltiples entidades {type(entities[0]).__name__}: {e}")
            return False
        finally:
            session.close()
            
    async def count(self, model_class: Type[T], filter_expr: Optional[str] = None) -> int:
        """
        Contar entidades con un filtro opcional.
        
        Args:
            model_class: Clase del modelo
            filter_expr: Expresión de filtro (SQL) opcional
            
        Returns:
            Número de entidades que coinciden con el filtro
        """
        session = self.session()
        try:
            if filter_expr:
                # Convertir string SQL a text()
                filter_sql = text(filter_expr)
                return session.query(model_class).filter(filter_sql).count()
            else:
                return session.query(model_class).count()
        except Exception as e:
            self.logger.error(f"Error al contar {model_class.__name__} con filtro {filter_expr}: {e}")
            return 0
        finally:
            session.close()