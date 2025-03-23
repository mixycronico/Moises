"""
Módulo base para conexiones de base de datos del Sistema Genesis.

Este módulo proporciona la configuración y conexiones base para SQLAlchemy
con soporte asíncrono para PostgreSQL utilizando AsyncPG.
"""
import logging
import os
from typing import Dict, Any, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

# Configuración de logging
logger = logging.getLogger("genesis.db.base")

# Base declarativa para modelos SQLAlchemy
Base = declarative_base()

# Configuración del motor de base de datos
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost/genesis")

# Convertir URL síncrona a asíncrona si es necesario
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)


class DatabaseManager:
    """Gestor central de conexiones a la base de datos."""
    
    def __init__(self, connection_url: Optional[str] = None, pool_size: int = 20, max_overflow: int = 40, pool_recycle: int = 300):
        """
        Inicializar gestor de base de datos.
        
        Args:
            connection_url: URL de conexión (opcional, usa DATABASE_URL por defecto)
            pool_size: Tamaño del pool de conexiones
            max_overflow: Conexiones máximas adicionales permitidas
            pool_recycle: Tiempo en segundos para reciclar conexiones
        """
        self.connection_url = connection_url or DATABASE_URL
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.engine = None
        self.async_session_factory = None
        
        logger.info(f"DatabaseManager inicializado con pool_size={pool_size}, max_overflow={max_overflow}")
    
    def setup(self):
        """Configurar el motor de base de datos y fábrica de sesiones."""
        if self.engine is None:
            self.engine = create_async_engine(
                self.connection_url,
                echo=False,  # No mostrar consultas SQL (producción)
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=True,  # Verificar conexiones antes de usarlas
                poolclass=QueuePool,
            )
            
            self.async_session_factory = sessionmaker(
                self.engine, expire_on_commit=False, class_=AsyncSession
            )
            
            logger.info("Motor de base de datos asíncrono configurado")
    
    async def create_session(self) -> AsyncSession:
        """
        Crear una nueva sesión asíncrona.
        
        Returns:
            Sesión asíncrona de SQLAlchemy
        """
        if self.async_session_factory is None:
            self.setup()
        return self.async_session_factory()
    
    async def create_all_tables(self):
        """Crear todas las tablas definidas en los modelos."""
        if self.engine is None:
            self.setup()
            
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Tablas creadas en la base de datos")
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del pool de conexiones.
        
        Returns:
            Diccionario con estadísticas
        """
        if self.engine is None:
            return {"status": "not_initialized"}
        
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_recycle": self.pool_recycle,
            "url": self.connection_url.split("@")[1] if "@" in self.connection_url else "**hidden**",
        }


# Instancia global del gestor de base de datos
db_manager = DatabaseManager()

# Función auxiliar para obtener una sesión de base de datos
async def get_db_session() -> AsyncSession:
    """
    Obtener una sesión de base de datos.
    
    Returns:
        Sesión asíncrona de SQLAlchemy
    """
    return await db_manager.create_session()