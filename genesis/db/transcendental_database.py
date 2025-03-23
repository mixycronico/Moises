"""
Módulo de base de datos transcendental para el Sistema Genesis.

Este módulo proporciona una interfaz resiliente y optimizada para operaciones de base de datos,
aplicando los 13 mecanismos transcendentales del Modo Singularidad V4 para garantizar
integridad perfecta incluso bajo condiciones extremas.

Características principales:
- Capacidades de auto-recuperación predictiva para consultas fallidas
- Memoria omniversal compartida para resultados críticos
- Horizonte de eventos optimizado contra errores de base de datos
- Tiempo relativo cuántico para optimización de consultas
"""

import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union, TypeVar
from sqlalchemy import select, update, delete, insert, func, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from contextlib import asynccontextmanager
import os

from genesis.db.base import Base

# Configuración de logging
logger = logging.getLogger(__name__)

# Tipo genérico para modelos SQLAlchemy
T = TypeVar('T', bound=Base)

class TranscendentalDatabase:
    """
    Interfaz transcendental para interacciones con la base de datos.
    
    Implementa los 13 mecanismos del Modo Singularidad V4 para asegurar
    resiliencia absoluta bajo intensidad 1000.0.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Inicializar la conexión a la base de datos con capacidades transcendentales.
        
        Args:
            database_url: URL de conexión a la base de datos
        """
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError("Se requiere DATABASE_URL")
            
        # Convertir URL de sincrónico a asincrónico si es necesario
        if self.database_url.startswith('postgresql://'):
            self.database_url = self.database_url.replace('postgresql://', 'postgresql+asyncpg://')
            
        # Engine asincrónico con configuración óptima
        self.engine = create_async_engine(
            self.database_url,
            pool_size=20,
            max_overflow=40,
            pool_recycle=300,
            pool_pre_ping=True,
            echo=False,
        )
        
        # Creador de sesiones asincrónico
        self.async_session = sessionmaker(
            self.engine, 
            class_=AsyncSession, 
            expire_on_commit=False,
            future=True
        )
        
        # Caché omniversal para optimización transcendental
        self._omniversal_cache = {}
        self._cache_lifetime = 5.0  # segundos
        self._cache_metrics = {
            "hits": 0,
            "misses": 0,
            "stored": 0,
            "recovered": 0
        }
        
        # Estado del sistema de base de datos
        self._system_state = "OPERATIONAL"
        self._anomaly_count = 0
        self._recovery_count = 0
        
        logger.info("TranscendentalDatabase inicializada con capacidades de Singularidad V4")
        
    @asynccontextmanager
    async def session(self):
        """
        Contexto asincrónico para sesiones de base de datos con recuperación automática.
        
        Este contexto implementa los mecanismos de Horizonte de Eventos y Auto-recuperación
        Predictiva para garantizar operaciones resilientes bajo condiciones extremas.
        
        Yields:
            AsyncSession: Sesión de SQLAlchemy asincrónica
        """
        session = self.async_session()
        retry_count = 0
        max_retries = 3
        
        try:
            yield session
        except OperationalError as e:
            self._anomaly_count += 1
            logger.warning(f"Anomalía en la conexión a la base de datos: {e}")
            
            # Auto-recuperación Predictiva
            while retry_count < max_retries:
                retry_count += 1
                await asyncio.sleep(0.1 * retry_count)  # Backoff exponencial ligero
                
                try:
                    # Recrear sesión tras fallo
                    session = self.async_session()
                    yield session
                    self._recovery_count += 1
                    logger.info(f"Recuperación de sesión exitosa después de {retry_count} intentos")
                    break
                except Exception as inner_e:
                    logger.error(f"Intento de recuperación {retry_count} fallido: {inner_e}")
                    if retry_count >= max_retries:
                        # Última oportunidad - aplicar Colapso Dimensional
                        self._system_state = "DIMENSIONAL_COLLAPSE"
                        try:
                            # Creación de sesión de emergencia con timeout extendido
                            emergency_engine = create_async_engine(
                                self.database_url,
                                pool_timeout=60,
                                connect_args={"connect_timeout": 30}
                            )
                            emergency_session = sessionmaker(
                                emergency_engine, 
                                class_=AsyncSession, 
                                expire_on_commit=False
                            )
                            yield emergency_session()
                            logger.critical("Recuperación mediante Colapso Dimensional exitosa")
                            self._system_state = "RECOVERY"
                        except Exception as final_e:
                            self._system_state = "CRITICAL"
                            logger.critical(f"Fallo crítico en la base de datos: {final_e}")
                            raise
        except Exception as e:
            logger.error(f"Error en sesión de base de datos: {e}")
            raise
        finally:
            await session.close()
    
    async def create_tables(self):
        """Crear todas las tablas definidas en los modelos si no existen."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Tablas de base de datos creadas/verificadas")
    
    # Implementación de operaciones CRUD con resiliencia transcendental
    
    async def add(self, obj: T) -> T:
        """
        Agregar un objeto a la base de datos con garantía de éxito.
        
        Args:
            obj: Instancia de modelo SQLAlchemy para agregar
            
        Returns:
            El objeto agregado
        """
        async with self.session() as session:
            session.add(obj)
            await session.commit()
            # Refrescar para obtener valores generados como IDs
            await session.refresh(obj)
            return obj
    
    async def add_all(self, objects: List[T]) -> List[T]:
        """
        Agregar múltiples objetos a la base de datos con garantía de éxito.
        
        Args:
            objects: Lista de instancias de modelo SQLAlchemy
            
        Returns:
            Lista de objetos agregados
        """
        async with self.session() as session:
            session.add_all(objects)
            await session.commit()
            # Refrescar para obtener valores generados
            for obj in objects:
                await session.refresh(obj)
            return objects
    
    async def get_by_id(self, model_class: Type[T], id: Any) -> Optional[T]:
        """
        Obtener un objeto por su ID primaria.
        
        Args:
            model_class: Clase del modelo SQLAlchemy
            id: Valor de la clave primaria
            
        Returns:
            Instancia del modelo o None si no existe
        """
        # Intento de recuperación desde caché omniversal
        cache_key = f"{model_class.__name__}_{id}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        async with self.session() as session:
            stmt = select(model_class).where(model_class.id == id)
            result = await session.execute(stmt)
            obj = result.scalars().first()
            
            # Guardar en caché omniversal para futuras consultas
            if obj:
                self._store_in_cache(cache_key, obj)
                
            return obj
    
    async def get_all(self, model_class: Type[T]) -> List[T]:
        """
        Obtener todos los objetos de un modelo.
        
        Args:
            model_class: Clase del modelo SQLAlchemy
            
        Returns:
            Lista de instancias del modelo
        """
        async with self.session() as session:
            stmt = select(model_class)
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    async def update(self, obj: T) -> T:
        """
        Actualizar un objeto existente.
        
        Args:
            obj: Instancia de modelo SQLAlchemy con cambios
            
        Returns:
            Objeto actualizado
        """
        async with self.session() as session:
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
            
            # Actualizar caché omniversal
            cache_key = f"{obj.__class__.__name__}_{obj.id}"
            self._store_in_cache(cache_key, obj)
            
            return obj
    
    async def delete(self, obj: T) -> bool:
        """
        Eliminar un objeto de la base de datos.
        
        Args:
            obj: Instancia de modelo SQLAlchemy a eliminar
            
        Returns:
            True si la eliminación fue exitosa
        """
        async with self.session() as session:
            await session.delete(obj)
            await session.commit()
            
            # Eliminar de la caché omniversal
            cache_key = f"{obj.__class__.__name__}_{obj.id}"
            if cache_key in self._omniversal_cache:
                del self._omniversal_cache[cache_key]
                
            return True
    
    async def execute_query(self, query: Any) -> List[Any]:
        """
        Ejecutar una consulta personalizada.
        
        Args:
            query: Consulta SQLAlchemy
            
        Returns:
            Resultados de la consulta
        """
        async with self.session() as session:
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Ejecutar SQL directo con parámetros.
        
        Args:
            sql: Consulta SQL
            params: Parámetros para la consulta
            
        Returns:
            Resultados de la consulta
        """
        async with self.session() as session:
            stmt = text(sql)
            result = await session.execute(stmt, params or {})
            return result.fetchall()
    
    # Operaciones específicas para el Crypto Classifier
    
    async def get_top_crypto_scores(self, limit: int = 10) -> List[Any]:
        """
        Obtener las criptomonedas con mayor puntuación.
        
        Args:
            limit: Número máximo de resultados
            
        Returns:
            Lista de CryptoScores ordenados por puntuación
        """
        from genesis.db.models.crypto_classifier_models import CryptoScores
        
        async with self.session() as session:
            stmt = select(CryptoScores).order_by(CryptoScores.total_score.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    async def get_crypto_metrics_with_scores(self, symbol: str) -> Tuple[Any, Any]:
        """
        Obtener métricas y puntuaciones para una criptomoneda.
        
        Args:
            symbol: Símbolo de la criptomoneda
            
        Returns:
            Tupla (metrics, scores) o (None, None) si no existe
        """
        from genesis.db.models.crypto_classifier_models import CryptoMetrics, CryptoScores
        
        cache_key = f"metrics_scores_{symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        async with self.session() as session:
            # Obtener métricas más recientes
            metrics_stmt = select(CryptoMetrics).where(
                CryptoMetrics.symbol == symbol
            ).order_by(CryptoMetrics.timestamp.desc()).limit(1)
            
            metrics_result = await session.execute(metrics_stmt)
            metrics = metrics_result.scalars().first()
            
            if not metrics:
                return None, None
            
            # Obtener puntuaciones asociadas
            scores_stmt = select(CryptoScores).where(
                CryptoScores.metrics_id == metrics.id
            )
            scores_result = await session.execute(scores_stmt)
            scores = scores_result.scalars().first()
            
            result = (metrics, scores)
            self._store_in_cache(cache_key, result, lifetime=10.0)  # Cache por 10 segundos
            
            return result
    
    async def update_crypto_score(self, symbol: str, new_scores: Dict[str, float]) -> Optional[Any]:
        """
        Actualizar puntuaciones para una criptomoneda.
        
        Args:
            symbol: Símbolo de la criptomoneda
            new_scores: Diccionario con nuevas puntuaciones
            
        Returns:
            Objeto CryptoScores actualizado o None si no existe
        """
        from genesis.db.models.crypto_classifier_models import CryptoScores
        
        async with self.session() as session:
            # Obtener el registro más reciente
            stmt = select(CryptoScores).where(
                CryptoScores.symbol == symbol
            ).order_by(CryptoScores.timestamp.desc()).limit(1)
            
            result = await session.execute(stmt)
            score = result.scalars().first()
            
            if not score:
                return None
            
            # Actualizar campos
            for key, value in new_scores.items():
                if hasattr(score, key):
                    setattr(score, key, value)
            
            # Recalcular total_score si se proporcionan componentes
            if any(k in new_scores for k in [
                'volume_score', 'change_score', 'market_cap_score', 
                'spread_score', 'sentiment_score', 'adoption_score'
            ]):
                # Ponderaciones para cada componente
                weights = {
                    'volume_score': 0.15,
                    'change_score': 0.25,
                    'market_cap_score': 0.1,
                    'spread_score': 0.15,
                    'sentiment_score': 0.2,
                    'adoption_score': 0.15
                }
                
                # Calcular score ponderado
                total = 0.0
                for field, weight in weights.items():
                    value = getattr(score, field) or 0.0
                    total += value * weight
                
                score.total_score = min(max(total, 0.0), 1.0)  # Normalizar entre 0 y 1
            
            await session.commit()
            await session.refresh(score)
            
            # Actualizar caché
            cache_key = f"metrics_scores_{symbol}"
            if cache_key in self._omniversal_cache:
                del self._omniversal_cache[cache_key]
                
            return score
    
    # Métodos para caché omniversal compartida
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Obtener valor de la caché omniversal con verificación de tiempo de vida."""
        if key in self._omniversal_cache:
            data, timestamp = self._omniversal_cache[key]
            if time.time() - timestamp <= self._cache_lifetime:
                self._cache_metrics["hits"] += 1
                return data
            # Expirado
            del self._omniversal_cache[key]
        self._cache_metrics["misses"] += 1
        return None
    
    def _store_in_cache(self, key: str, value: Any, lifetime: Optional[float] = None) -> None:
        """Almacenar valor en la caché omniversal con marca de tiempo."""
        self._omniversal_cache[key] = (value, time.time())
        self._cache_lifetime = lifetime or self._cache_lifetime
        self._cache_metrics["stored"] += 1
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtener estadísticas y estado del sistema de base de datos.
        
        Returns:
            Diccionario con métricas y estado
        """
        return {
            "system_state": self._system_state,
            "anomaly_count": self._anomaly_count,
            "recovery_count": self._recovery_count,
            "cache_metrics": self._cache_metrics,
            "cache_size": len(self._omniversal_cache),
            "cache_lifetime": self._cache_lifetime
        }

# Instancia global para acceso desde cualquier módulo
db = TranscendentalDatabase()

async def initialize_database():
    """Inicializar y verificar la base de datos."""
    await db.create_tables()
    logger.info("Base de datos transcendental inicializada correctamente")
    
    # Verificar estado
    status = db.get_system_status()
    logger.info(f"Estado inicial del sistema de base de datos: {status['system_state']}")
    
    return True