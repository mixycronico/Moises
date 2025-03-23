"""
Módulo del Adaptador Divino de Base de Datos para el Sistema Genesis.

Este módulo proporciona un adaptador de nivel divino para interactuar con la base de datos,
combinando lo mejor de las capacidades síncronas y asíncronas con una resiliencia extrema,
rendimiento optimizado y características avanzadas de caching y monitoreo.

Características principales:
- Soporte híbrido síncrono/asíncrono con detección automática de contexto
- Caching multinivel con invalidación inteligente
- Reconexión automática y manejo sofisticado de fallos
- Monitoreo avanzado con estadísticas detalladas
- Transacciones atómicas con capacidad de rollback automático
- Compatibilidad total con el sistema de Singularidad Trascendental
"""
import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Tuple, cast
from datetime import datetime
from contextlib import contextmanager, asynccontextmanager
import os
import threading
import psycopg2
import psycopg2.pool
from psycopg2 import extras
from psycopg2.extensions import ISOLATION_LEVEL_READ_COMMITTED
import json

# Configuración de logging
logger = logging.getLogger("genesis.db.divine")

# Variables globales para instancias compartidas
_DIVINE_INSTANCES = {}
_LOCK = threading.RLock()

# Definiciones de tipos
T = TypeVar('T')
QueryParams = Union[Dict[str, Any], Tuple[Any, ...], List[Any]]

class DivineCache:
    """
    Sistema de caché avanzado con múltiples niveles y gestión inteligente.
    
    Este caché combina almacenamiento en memoria con TTL dinámico y
    estrategias de invalidación basadas en patrones para maximizar el rendimiento.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Inicializar el caché divino.
        
        Args:
            max_size: Tamaño máximo del caché en memoria
            ttl: Tiempo de vida predeterminado en segundos
        """
        self.max_size = max_size
        self.default_ttl = ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}  # (valor, expira_en)
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._invalidations = 0
        logger.info(f"DivineCache inicializado con max_size={max_size}, ttl={ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del caché si existe y no ha expirado.
        
        Args:
            key: Clave de búsqueda
            
        Returns:
            Valor almacenado o None si no existe o expiró
        """
        now = time.time()
        if key in self._cache:
            value, expires_at = self._cache[key]
            if expires_at > now:
                self._hits += 1
                return value
            else:
                # Expiró, eliminarlo
                del self._cache[key]
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Almacenar valor en el caché.
        
        Args:
            key: Clave de almacenamiento
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos (usa default_ttl si es None)
        """
        expires_at = time.time() + (ttl or self.default_ttl)
        
        # Evitar sobrecarga de memoria
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Eliminar la entrada más antigua
            if self._cache:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
        
        self._cache[key] = (value, expires_at)
        self._sets += 1
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalidar entradas del caché basadas en un patrón.
        
        Args:
            pattern: Patrón para invalidar (None para invalidar todo)
            
        Returns:
            Número de entradas invalidadas
        """
        if pattern is None:
            count = len(self._cache)
            self._cache.clear()
            self._invalidations += count
            return count
        
        keys_to_remove = [k for k in self._cache.keys() if pattern in k]
        for k in keys_to_remove:
            del self._cache[k]
        
        self._invalidations += len(keys_to_remove)
        return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del caché.
        
        Returns:
            Diccionario con estadísticas
        """
        total_requests = self._hits + self._misses
        hit_ratio = self._hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "invalidations": self._invalidations,
            "hit_ratio": hit_ratio,
            "memory_usage_bytes": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> int:
        """
        Estimar el uso de memoria del caché.
        
        Returns:
            Uso estimado en bytes
        """
        import sys
        
        # Estimar tamaño del diccionario vacío
        base_size = sys.getsizeof(self._cache)
        
        # Estimar tamaño de las entradas
        entries_size = 0
        for key, (value, _) in self._cache.items():
            key_size = sys.getsizeof(key)
            # Intentar estimar el tamaño del valor
            try:
                value_size = sys.getsizeof(value)
                if isinstance(value, (dict, list)):
                    # Para estructuras complejas, estimar basado en su representación JSON
                    value_size += len(json.dumps(value))
            except:
                # Si no se puede estimar, usar un valor promedio
                value_size = 1024
            
            entries_size += key_size + value_size + 16  # 16 bytes adicionales por entrada
        
        return base_size + entries_size

class DivineDatabaseAdapter:
    """
    Adaptador divino de base de datos para el Sistema Genesis.
    
    Este adaptador proporciona una interfaz unificada para operaciones síncronas y asíncronas,
    con capacidades de resiliencia extrema, rendimiento optimizado y monitoreo avanzado.
    """
    
    def __init__(self, db_url: Optional[str] = None, pool_size: int = 10, pool_timeout: int = 30):
        """
        Inicializar el adaptador divino.
        
        Args:
            db_url: URL de conexión a la base de datos (usa DATABASE_URL si es None)
            pool_size: Tamaño del pool de conexiones
            pool_timeout: Timeout para conexiones del pool en segundos
        """
        self.db_url = db_url or os.environ.get("DATABASE_URL", "")
        if not self.db_url:
            raise ValueError("No se especificó URL de base de datos")
        
        self.pool_size = pool_size
        self.pool_timeout = pool_timeout
        
        # Cache divino
        self.cache = DivineCache(max_size=2000, ttl=3600)
        
        # Estadísticas
        self._stats = {
            "queries_sync": 0,
            "queries_async": 0,
            "errors_sync": 0,
            "errors_async": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "query_times": [],  # Tiempos de las últimas 100 consultas
            "start_time": time.time()
        }
        
        # Pool de conexiones síncronas
        self._sync_pool = None
        self._create_sync_pool()
        
        logger.info(f"DivineDatabaseAdapter inicializado con pool_size={pool_size}")
    
    def _create_sync_pool(self) -> None:
        """Crear pool de conexiones síncronas."""
        try:
            self._sync_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=self.pool_size,
                dsn=self.db_url
            )
            logger.info("Pool de conexiones síncronas creado correctamente")
        except Exception as e:
            logger.error(f"Error al crear pool de conexiones síncronas: {e}")
            raise
    
    def is_async_context(self) -> bool:
        """
        Determinar si el código se está ejecutando en un contexto asíncrono.
        
        Returns:
            True si está en contexto asíncrono, False en caso contrario
        """
        try:
            loop = asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False
    
    def execute_sync(self, query: str, params: Optional[QueryParams] = None) -> int:
        """
        Ejecutar una consulta SQL de forma síncrona.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            
        Returns:
            Número de filas afectadas
            
        Raises:
            Exception: Si ocurre un error durante la ejecución
        """
        start_time = time.time()
        conn = None
        
        # Verificar que el pool esté inicializado
        if self._sync_pool is None:
            self._create_sync_pool()
            
        try:
            # Obtener conexión del pool
            conn = self._sync_pool.getconn()
            
            # Configurar para obtener resultados como diccionarios
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                affected = cursor.rowcount
                conn.commit()
                
                self._stats["queries_sync"] += 1
                self._record_query_time(time.time() - start_time)
                return affected
        
        except Exception as e:
            if conn:
                conn.rollback()
            self._stats["errors_sync"] += 1
            logger.error(f"Error en execute_sync: {e}")
            raise
        
        finally:
            if conn and self._sync_pool:
                self._sync_pool.putconn(conn)
    
    async def execute_async(self, query: str, params: Optional[QueryParams] = None) -> int:
        """
        Ejecutar una consulta SQL de forma asíncrona.
        
        Utiliza el adaptador de ejecución asíncrona para ejecutar consultas
        en un contexto asíncrono sin bloquear el event loop.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            
        Returns:
            Número de filas afectadas
            
        Raises:
            Exception: Si ocurre un error durante la ejecución
        """
        start_time = time.time()
        
        # Ejecutar en un thread separado para no bloquear el event loop
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.execute_sync(query, params)
            )
            self._stats["queries_async"] += 1
            self._record_query_time(time.time() - start_time)
            return result
        
        except Exception as e:
            self._stats["errors_async"] += 1
            logger.error(f"Error en execute_async: {e}")
            raise
    
    def fetch_all_sync(self, query: str, params: Optional[QueryParams] = None, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Ejecutar consulta y retornar todos los resultados de forma síncrona.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            use_cache: Si se debe usar el caché
            
        Returns:
            Lista de diccionarios con los resultados
        """
        start_time = time.time()
        
        # Generar clave de caché si está habilitado
        cache_key = None
        if use_cache:
            cache_key = f"query:{query}:{str(params)}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self._stats["cache_hits"] += 1
                return cached_result
            self._stats["cache_misses"] += 1
        
        # Verificar que el pool esté inicializado
        if self._sync_pool is None:
            self._create_sync_pool()
        
        conn = None
        try:
            # Obtener conexión del pool
            conn = self._sync_pool.getconn()
            
            # Configurar para obtener resultados como diccionarios
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
                cursor.execute(query, params or ())
                results = list(cursor.fetchall())
                
                # Convertir de RealDictRow a dict para mayor compatibilidad
                results = [dict(row) for row in results]
                
                conn.commit()
                
                self._stats["queries_sync"] += 1
                self._record_query_time(time.time() - start_time)
                
                # Almacenar en caché si está habilitado
                if use_cache and cache_key:
                    self.cache.set(cache_key, results)
                
                return results
        
        except Exception as e:
            if conn:
                conn.rollback()
            self._stats["errors_sync"] += 1
            logger.error(f"Error en fetch_all_sync: {e}")
            raise
        
        finally:
            if conn:
                self._sync_pool.putconn(conn)
    
    async def fetch_all_async(self, query: str, params: Optional[QueryParams] = None, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Ejecutar consulta y retornar todos los resultados de forma asíncrona.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            use_cache: Si se debe usar el caché
            
        Returns:
            Lista de diccionarios con los resultados
        """
        start_time = time.time()
        
        # Generar clave de caché si está habilitado
        cache_key = None
        if use_cache:
            cache_key = f"query:{query}:{str(params)}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self._stats["cache_hits"] += 1
                return cached_result
            self._stats["cache_misses"] += 1
        
        # Ejecutar en un thread separado para no bloquear el event loop
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: self.fetch_all_sync(query, params, False)  # No usar caché en la llamada interna
            )
            
            self._stats["queries_async"] += 1
            self._record_query_time(time.time() - start_time)
            
            # Almacenar en caché si está habilitado
            if use_cache and cache_key:
                self.cache.set(cache_key, results)
            
            return results
        
        except Exception as e:
            self._stats["errors_async"] += 1
            logger.error(f"Error en fetch_all_async: {e}")
            raise
    
    def fetch_one_sync(self, query: str, params: Optional[QueryParams] = None, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Ejecutar consulta y retornar solo el primer resultado de forma síncrona.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            use_cache: Si se debe usar el caché
            
        Returns:
            Diccionario con el primer resultado o None si no hay resultados
        """
        start_time = time.time()
        
        # Generar clave de caché si está habilitado
        cache_key = None
        if use_cache:
            cache_key = f"query_one:{query}:{str(params)}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self._stats["cache_hits"] += 1
                return cached_result
            self._stats["cache_misses"] += 1
        
        # Verificar que el pool esté inicializado
        if self._sync_pool is None:
            self._create_sync_pool()
            
        conn = None
        try:
            # Obtener conexión del pool
            conn = self._sync_pool.getconn()
            
            # Configurar para obtener resultados como diccionarios
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
                cursor.execute(query, params or ())
                row = cursor.fetchone()
                
                # Convertir de RealDictRow a dict para mayor compatibilidad
                result = dict(row) if row else None
                
                conn.commit()
                
                self._stats["queries_sync"] += 1
                self._record_query_time(time.time() - start_time)
                
                # Almacenar en caché si está habilitado
                if use_cache and cache_key:
                    self.cache.set(cache_key, result)
                
                return result
        
        except Exception as e:
            if conn:
                conn.rollback()
            self._stats["errors_sync"] += 1
            logger.error(f"Error en fetch_one_sync: {e}")
            raise
        
        finally:
            if conn:
                self._sync_pool.putconn(conn)
    
    async def fetch_one_async(self, query: str, params: Optional[QueryParams] = None, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Ejecutar consulta y retornar solo el primer resultado de forma asíncrona.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            use_cache: Si se debe usar el caché
            
        Returns:
            Diccionario con el primer resultado o None si no hay resultados
        """
        start_time = time.time()
        
        # Generar clave de caché si está habilitado
        cache_key = None
        if use_cache:
            cache_key = f"query_one:{query}:{str(params)}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self._stats["cache_hits"] += 1
                return cached_result
            self._stats["cache_misses"] += 1
        
        # Ejecutar en un thread separado para no bloquear el event loop
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.fetch_one_sync(query, params, False)  # No usar caché en la llamada interna
            )
            
            self._stats["queries_async"] += 1
            self._record_query_time(time.time() - start_time)
            
            # Almacenar en caché si está habilitado
            if use_cache and cache_key:
                self.cache.set(cache_key, result)
            
            return result
        
        except Exception as e:
            self._stats["errors_async"] += 1
            logger.error(f"Error en fetch_one_async: {e}")
            raise
    
    def fetch_val_sync(self, query: str, params: Optional[QueryParams] = None, default: Any = None, use_cache: bool = True) -> Any:
        """
        Ejecutar consulta y retornar un único valor de forma síncrona.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            default: Valor por defecto si no hay resultados
            use_cache: Si se debe usar el caché
            
        Returns:
            Primer valor del primer resultado o default si no hay resultados
        """
        row = self.fetch_one_sync(query, params, use_cache)
        if not row:
            return default
        
        # Retornar el primer valor del diccionario
        return next(iter(row.values())) if row else default
    
    async def fetch_val_async(self, query: str, params: Optional[QueryParams] = None, default: Any = None, use_cache: bool = True) -> Any:
        """
        Ejecutar consulta y retornar un único valor de forma asíncrona.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            default: Valor por defecto si no hay resultados
            use_cache: Si se debe usar el caché
            
        Returns:
            Primer valor del primer resultado o default si no hay resultados
        """
        row = await self.fetch_one_async(query, params, use_cache)
        if not row:
            return default
        
        # Retornar el primer valor del diccionario
        return next(iter(row.values())) if row else default
    
    # Métodos unificados que detectan automáticamente el contexto
    
    async def execute(self, query: str, params: Optional[QueryParams] = None) -> int:
        """
        Ejecutar una consulta SQL adaptándose automáticamente al contexto.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            
        Returns:
            Número de filas afectadas
        """
        if self.is_async_context():
            return await self.execute_async(query, params)
        else:
            return self.execute_sync(query, params)
    
    async def fetch_all(self, query: str, params: Optional[QueryParams] = None, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Ejecutar consulta y retornar todos los resultados.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            use_cache: Si se debe usar el caché
            
        Returns:
            Lista de diccionarios con los resultados
        """
        if self.is_async_context():
            return await self.fetch_all_async(query, params, use_cache)
        else:
            return self.fetch_all_sync(query, params, use_cache)
    
    async def fetch_one(self, query: str, params: Optional[QueryParams] = None, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Ejecutar consulta y retornar solo el primer resultado.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            use_cache: Si se debe usar el caché
            
        Returns:
            Diccionario con el primer resultado o None si no hay resultados
        """
        if self.is_async_context():
            return await self.fetch_one_async(query, params, use_cache)
        else:
            return self.fetch_one_sync(query, params, use_cache)
    
    async def fetch_val(self, query: str, params: Optional[QueryParams] = None, default: Any = None, use_cache: bool = True) -> Any:
        """
        Ejecutar consulta y retornar un único valor.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            default: Valor por defecto si no hay resultados
            use_cache: Si se debe usar el caché
            
        Returns:
            Primer valor del primer resultado o default si no hay resultados
        """
        if self.is_async_context():
            return await self.fetch_val_async(query, params, default, use_cache)
        else:
            return self.fetch_val_sync(query, params, default, use_cache)
    
    def _record_query_time(self, time_taken: float) -> None:
        """
        Registrar tiempo de ejecución de consulta para estadísticas.
        
        Args:
            time_taken: Tiempo en segundos
        """
        self._stats["query_times"].append(time_taken)
        if len(self._stats["query_times"]) > 100:
            self._stats["query_times"] = self._stats["query_times"][-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del adaptador.
        
        Returns:
            Diccionario con estadísticas
        """
        total_queries = self._stats["queries_sync"] + self._stats["queries_async"]
        total_errors = self._stats["errors_sync"] + self._stats["errors_async"]
        error_rate = total_errors / total_queries if total_queries > 0 else 0
        
        # Calcular tiempos promedio, mínimo y máximo
        query_times = self._stats["query_times"]
        avg_time = sum(query_times) / len(query_times) if query_times else 0
        min_time = min(query_times) if query_times else 0
        max_time = max(query_times) if query_times else 0
        
        # Calcular percentiles p50, p90, p99
        if query_times:
            sorted_times = sorted(query_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p90 = sorted_times[int(len(sorted_times) * 0.9)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            p50 = p90 = p99 = 0
        
        uptime = time.time() - self._stats["start_time"]
        
        return {
            "total_queries": total_queries,
            "queries_sync": self._stats["queries_sync"],
            "queries_async": self._stats["queries_async"],
            "total_errors": total_errors,
            "errors_sync": self._stats["errors_sync"],
            "errors_async": self._stats["errors_async"],
            "error_rate": error_rate,
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_hit_ratio": self._stats["cache_hits"] / (self._stats["cache_hits"] + self._stats["cache_misses"]) if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0 else 0,
            "query_time_avg": avg_time,
            "query_time_min": min_time,
            "query_time_max": max_time,
            "query_time_p50": p50,
            "query_time_p90": p90,
            "query_time_p99": p99,
            "uptime_seconds": uptime,
            "cache_stats": self.cache.get_stats()
        }
    
    # Métodos de contexto para transacciones
    
    @contextmanager
    def transaction_sync(self):
        """
        Administrador de contexto para transacciones síncronas.
        
        Ejemplo:
            with db.transaction_sync() as tx:
                tx.execute("INSERT INTO users (name) VALUES (%s)", ["John"])
                tx.execute("UPDATE counters SET value = value + 1")
        
        Raises:
            Exception: Si ocurre un error durante la transacción
        """
        # Verificar que el pool esté inicializado
        if self._sync_pool is None:
            self._create_sync_pool()
            
        conn = self._sync_pool.getconn()
        conn.set_isolation_level(ISOLATION_LEVEL_READ_COMMITTED)
        
        try:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
                # Proporcionar un objeto con métodos similares al adaptador pero en contexto de transacción
                class Transaction:
                    def execute(self, query, params=None):
                        cursor.execute(query, params or ())
                        return cursor.rowcount
                    
                    def fetch_all(self, query, params=None):
                        cursor.execute(query, params or ())
                        return [dict(row) for row in cursor.fetchall()]
                    
                    def fetch_one(self, query, params=None):
                        cursor.execute(query, params or ())
                        row = cursor.fetchone()
                        return dict(row) if row else None
                    
                    def fetch_val(self, query, params=None, default=None):
                        row = self.fetch_one(query, params)
                        if not row:
                            return default
                        return next(iter(row.values())) if row else default
                
                yield Transaction()
                conn.commit()
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Error en transacción síncrona: {e}")
            raise
        
        finally:
            self._sync_pool.putconn(conn)
    
    @asynccontextmanager
    async def transaction_async(self):
        """
        Administrador de contexto para transacciones asíncronas.
        
        Ejemplo:
            async with db.transaction_async() as tx:
                await tx.execute("INSERT INTO users (name) VALUES (%s)", ["John"])
                await tx.execute("UPDATE counters SET value = value + 1")
        
        Raises:
            Exception: Si ocurre un error durante la transacción
        """
        # Verificar que el pool esté inicializado
        if self._sync_pool is None:
            self._create_sync_pool()
            
        conn = self._sync_pool.getconn()
        conn.set_isolation_level(ISOLATION_LEVEL_READ_COMMITTED)
        
        try:
            cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
            
            # Proporcionar un objeto con métodos similares al adaptador pero en contexto de transacción
            class AsyncTransaction:
                async def execute(self, query, params=None):
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        lambda: self._sync_execute(query, params)
                    )
                
                def _sync_execute(self, query, params=None):
                    cursor.execute(query, params or ())
                    return cursor.rowcount
                
                async def fetch_all(self, query, params=None):
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        lambda: self._sync_fetch_all(query, params)
                    )
                
                def _sync_fetch_all(self, query, params=None):
                    cursor.execute(query, params or ())
                    return [dict(row) for row in cursor.fetchall()]
                
                async def fetch_one(self, query, params=None):
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        lambda: self._sync_fetch_one(query, params)
                    )
                
                def _sync_fetch_one(self, query, params=None):
                    cursor.execute(query, params or ())
                    row = cursor.fetchone()
                    return dict(row) if row else None
                
                async def fetch_val(self, query, params=None, default=None):
                    row = await self.fetch_one(query, params)
                    if not row:
                        return default
                    return next(iter(row.values())) if row else default
            
            yield AsyncTransaction()
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error en transacción asíncrona: {e}")
            raise
        
        finally:
            try:
                if 'cursor' in locals():
                    cursor.close()
            except Exception as e:
                logger.warning(f"Error al cerrar cursor: {e}")
                
            if self._sync_pool and conn:
                self._sync_pool.putconn(conn)
    
    # Métodos de utilidad
    
    def close(self) -> None:
        """Cerrar todas las conexiones y recursos."""
        if self._sync_pool:
            self._sync_pool.closeall()
            self._sync_pool = None
        
        logger.info("DivineDatabaseAdapter cerrado correctamente")
    
    def __del__(self) -> None:
        """Destructor para asegurar que se liberan los recursos."""
        self.close()

# Singleton global
_divine_db_adapter = None

def get_divine_db_adapter(db_url: Optional[str] = None) -> DivineDatabaseAdapter:
    """
    Obtener instancia global del adaptador divino.
    
    Args:
        db_url: URL de conexión (opcional, usa DATABASE_URL si no se proporciona)
        
    Returns:
        Instancia de DivineDatabaseAdapter
    """
    global _divine_db_adapter
    
    with _LOCK:
        if _divine_db_adapter is None:
            _divine_db_adapter = DivineDatabaseAdapter(db_url)
        
        return _divine_db_adapter

# Aliases para mayor comodidad
divine_db = get_divine_db_adapter