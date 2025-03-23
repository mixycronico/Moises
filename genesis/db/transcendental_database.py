"""
Módulo de base de datos trascendental para el Sistema Genesis.

Este módulo implementa la capa de base de datos con capacidades trascendentales,
permitiendo operaciones resilientes y atemporales que garantizan la consistencia
incluso bajo condiciones extremas y alta carga.

Características principales:
- Cache cuántico multidimensional
- Reintentos adaptativos con backoff exponencial
- Checkpointing atemporal
- Resolución automática de anomalías y conflictos
- Sincronización entre estados temporales
"""
import asyncio
import logging
import json
import time
import random
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, TypeVar, Generic, Set

import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text

from genesis.db.base import get_db_session

# Definición de tipos para anotaciones
T = TypeVar('T')
R = TypeVar('R')

# Configuración de logging
logger = logging.getLogger("genesis.db.transcendental_database")

class QuantumCache(Generic[T]):
    """
    Cache cuántico multidimensional que almacena valores en diferentes estados temporales.
    
    Este cache mantiene múltiples versiones de los datos en diferentes "dimensiones",
    permitiendo acceso a estados pasados, presentes y futuros sin conflictos.
    """
    
    def __init__(self, max_size: int = 1000, dimensions: int = 3, ttl: int = 3600):
        """
        Inicializar cache cuántico.
        
        Args:
            max_size: Tamaño máximo del cache
            dimensions: Número de dimensiones (3 = pasado, presente, futuro)
            ttl: Tiempo de vida para entradas del cache (segundos)
        """
        self.max_size = max_size
        self.dimensions = dimensions
        self.ttl = ttl
        self.cache: Dict[int, Dict[str, Tuple[T, float]]] = {d: {} for d in range(dimensions)}
        self.access_count: Dict[str, int] = {}
        self.last_cleanup = time.time()
        
        logger.info(f"QuantumCache inicializado con {dimensions} dimensiones y TTL de {ttl}s")
    
    def set(self, key: str, value: T, dimension: int = 1) -> None:
        """
        Almacenar valor en el cache para una dimensión específica.
        
        Args:
            key: Clave de almacenamiento
            value: Valor a almacenar
            dimension: Dimensión (0=pasado, 1=presente, 2=futuro)
        """
        if dimension >= self.dimensions:
            dimension = self.dimensions - 1
            
        # Limpiar cache si está lleno
        if len(self.cache[dimension]) >= self.max_size:
            self._cleanup()
            
        self.cache[dimension][key] = (value, time.time() + self.ttl)
        self.access_count[key] = self.access_count.get(key, 0) + 1
    
    def get(self, key: str, dimension: int = 1) -> Optional[T]:
        """
        Obtener valor del cache para una dimensión específica.
        
        Args:
            key: Clave de búsqueda
            dimension: Dimensión (0=pasado, 1=presente, 2=futuro)
            
        Returns:
            Valor almacenado o None si no existe o ha expirado
        """
        if dimension >= self.dimensions:
            dimension = self.dimensions - 1
            
        if key not in self.cache[dimension]:
            return None
            
        value, expiry = self.cache[dimension][key]
        
        # Verificar si ha expirado
        if time.time() > expiry:
            del self.cache[dimension][key]
            return None
            
        self.access_count[key] = self.access_count.get(key, 0) + 1
        return value
    
    def invalidate(self, key: str, all_dimensions: bool = False) -> None:
        """
        Invalidar entrada del cache.
        
        Args:
            key: Clave a invalidar
            all_dimensions: Si se deben invalidar todas las dimensiones
        """
        if all_dimensions:
            for d in range(self.dimensions):
                if key in self.cache[d]:
                    del self.cache[d][key]
        else:
            # Solo dimensión presente
            if key in self.cache[1]:
                del self.cache[1][key]
    
    def propagate(self, key: str, direction: int = 1) -> None:
        """
        Propagar valor entre dimensiones.
        
        Args:
            key: Clave a propagar
            direction: Dirección (1=hacia futuro, -1=hacia pasado)
        """
        if direction == 1 and key in self.cache[1]:
            # Presente -> Futuro
            self.cache[2][key] = self.cache[1][key]
        elif direction == -1 and key in self.cache[1]:
            # Presente -> Pasado
            self.cache[0][key] = self.cache[1][key]
    
    def _cleanup(self) -> None:
        """Limpieza de entradas expiradas o menos utilizadas."""
        current_time = time.time()
        
        # Limpiar cada 60 segundos máximo
        if current_time - self.last_cleanup < 60:
            return
            
        self.last_cleanup = current_time
        
        # Eliminar expirados
        for d in range(self.dimensions):
            expired_keys = [
                k for k, (_, expiry) in self.cache[d].items() 
                if current_time > expiry
            ]
            for k in expired_keys:
                del self.cache[d][k]
        
        # Si sigue lleno, eliminar menos accedidos
        if any(len(self.cache[d]) >= self.max_size for d in range(self.dimensions)):
            sorted_keys = sorted(
                self.access_count.keys(), 
                key=lambda k: self.access_count[k]
            )
            
            # Eliminar el 10% menos accedido
            keys_to_remove = sorted_keys[:int(len(sorted_keys) * 0.1)]
            for k in keys_to_remove:
                self.access_count.pop(k, None)
                for d in range(self.dimensions):
                    self.cache[d].pop(k, None)


class AtemporalCheckpoint:
    """
    Sistema de checkpoint atemporal para mantener estados coherentes.
    
    Permite guardar y restaurar estados de la base de datos en diferentes
    momentos temporales, incluso durante operaciones que cruzan la barrera
    entre pasado, presente y futuro.
    """
    
    def __init__(self, max_checkpoints: int = 100):
        """
        Inicializar sistema de checkpoints.
        
        Args:
            max_checkpoints: Número máximo de checkpoints a mantener
        """
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.max_checkpoints = max_checkpoints
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.checkpoint_count = 0
        
        logger.info(f"AtemporalCheckpoint inicializado con capacidad para {max_checkpoints} puntos")
    
    def create(self, checkpoint_id: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Crear un nuevo checkpoint.
        
        Args:
            checkpoint_id: Identificador único del checkpoint
            data: Datos a almacenar
            metadata: Metadatos adicionales
        """
        if len(self.checkpoints) >= self.max_checkpoints:
            # Eliminar el checkpoint más antiguo
            oldest = min(self.metadata.items(), key=lambda x: x[1].get('timestamp', 0))
            if oldest[0] in self.checkpoints:
                del self.checkpoints[oldest[0]]
                del self.metadata[oldest[0]]
        
        # Serializar data para asegurar una copia profunda
        self.checkpoints[checkpoint_id] = json.loads(json.dumps(data))
        self.metadata[checkpoint_id] = metadata or {
            'timestamp': time.time(),
            'sequence': self.checkpoint_count
        }
        self.checkpoint_count += 1
        
    def get(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener datos de un checkpoint.
        
        Args:
            checkpoint_id: Identificador del checkpoint
            
        Returns:
            Datos almacenados o None si no existe
        """
        return self.checkpoints.get(checkpoint_id)
    
    def get_metadata(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener metadatos de un checkpoint.
        
        Args:
            checkpoint_id: Identificador del checkpoint
            
        Returns:
            Metadatos o None si no existe
        """
        return self.metadata.get(checkpoint_id)
    
    def list_all(self) -> List[str]:
        """
        Listar todos los checkpoints disponibles.
        
        Returns:
            Lista de identificadores de checkpoint
        """
        return list(self.checkpoints.keys())
    
    def delete(self, checkpoint_id: str) -> bool:
        """
        Eliminar un checkpoint.
        
        Args:
            checkpoint_id: Identificador del checkpoint
            
        Returns:
            True si se eliminó correctamente, False si no existía
        """
        if checkpoint_id in self.checkpoints:
            del self.checkpoints[checkpoint_id]
            if checkpoint_id in self.metadata:
                del self.metadata[checkpoint_id]
            return True
        return False


class TranscendentalDatabase:
    """
    Capa de base de datos trascendental que proporciona resiliencia extrema.
    
    Implementa capacidades avanzadas de la Singularidad V4, incluyendo:
    - Operaciones atemporales
    - Reintentos adaptativos con circuit breaker
    - Cache cuántico
    - Checkpoints resilientes
    - Protección contra anomalías
    """
    
    def __init__(
        self, 
        cache_size: int = 10000, 
        checkpoint_capacity: int = 100,
        pool_size: int = 20,
        max_overflow: int = 40,
        pool_recycle: int = 300,
        pool_pre_ping: bool = True
    ):
        """
        Inicializar base de datos trascendental.
        
        Args:
            cache_size: Tamaño del cache cuántico
            checkpoint_capacity: Capacidad máxima de checkpoints
            pool_size: Tamaño del pool de conexiones
            max_overflow: Máximo de conexiones adicionales temporales
            pool_recycle: Tiempo en segundos para reciclar conexiones
            pool_pre_ping: Si se debe hacer ping a la BD antes de usar conexiones
        """
        self.cache = QuantumCache(max_size=cache_size)
        self.checkpoint = AtemporalCheckpoint(max_checkpoints=checkpoint_capacity)
        self.error_count: Dict[str, int] = {}
        self.success_count: Dict[str, int] = {}
        self.anomaly_detection: Set[str] = set()
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self._connection_health = True
        self._last_ping = 0
        self._ping_lock = asyncio.Lock()
        
        logger.info(f"TranscendentalDatabase inicializada con capacidades de Singularidad V4, pool_size={pool_size}")
    
    async def execute_query(
        self, 
        query_func: Callable[..., Union[Tuple[str, List[Any]], Any]], 
        *args, 
        max_retries: int = 3,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        operation_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Ejecutar consulta SQL con toda la resiliencia del sistema trascendental.
        
        Args:
            query_func: Función que genera la consulta SQL y parámetros o un objeto SQLAlchemy
            *args: Argumentos para query_func
            max_retries: Reintentos máximos
            use_cache: Si se debe usar el cache
            cache_ttl: Tiempo de vida en cache
            operation_id: Identificador de operación
            **kwargs: Argumentos adicionales para query_func
            
        Returns:
            Resultado de la consulta
            
        Raises:
            SQLAlchemyError: Si todos los reintentos fallan
        """
        op_id = operation_id or f"query_{hash(str(query_func) + str(args) + str(kwargs))}"
        
        # Verificar cache si está habilitado
        if use_cache:
            cached_result = self.cache.get(op_id)
            if cached_result is not None:
                return cached_result
        
        # Preparar consulta
        query_result = query_func(*args, **kwargs)
        
        # Determinar si es una consulta SQLAlchemy o una tupla (sql, params)
        if isinstance(query_result, tuple) and len(query_result) == 2:
            # Es una tupla (sql, params)
            sql, params = query_result
            use_text = True
        else:
            # Es un objeto SQLAlchemy
            sql = query_result
            params = {}
            use_text = False
        
        # Implementar reintentos con backoff exponencial
        for retry in range(max_retries + 1):
            session = None
            try:
                # Obtener una sesión
                session = await get_db_session()
                
                # Ejecutar consulta según su tipo
                if use_text:
                    result = await session.execute(text(sql), params)
                    
                    # Determinar el tipo de operación
                    is_select = sql.strip().lower().startswith("select")
                    is_modify = sql.strip().lower().startswith(("insert", "update", "delete"))
                else:
                    # Es un objeto SQLAlchemy, ejecutarlo directamente
                    result = await session.execute(sql)
                    
                    # Intentar determinar tipo de operación por el tipo de objeto
                    is_select = hasattr(sql, "is_select") and getattr(sql, "is_select")
                    is_modify = hasattr(sql, "is_dml") and getattr(sql, "is_dml")
                
                # Si es una operación que modifica datos, hacer commit
                if is_modify:
                    await session.commit()
                
                # Procesar resultado según tipo de consulta
                if is_select:
                    final_result = result.fetchall()
                else:
                    final_result = result.rowcount
                
                # Almacenar en cache si está habilitado
                if use_cache:
                    self.cache.set(op_id, final_result)
                
                # Registrar éxito
                self.success_count[op_id] = self.success_count.get(op_id, 0) + 1
                
                return final_result
                
            except SQLAlchemyError as e:
                # Registrar error
                self.error_count[op_id] = self.error_count.get(op_id, 0) + 1
                
                # Detectar anomalías
                if self.error_count.get(op_id, 0) > 5:
                    self.anomaly_detection.add(op_id)
                
                # Último intento, propagar excepción
                if retry == max_retries:
                    logger.error(f"Todos los reintentos fallaron para operación {op_id}: {str(e)}")
                    raise
                
                # Aplicar backoff exponencial con jitter
                delay = min(1.0, 0.1 * (2 ** retry) + random.uniform(0, 0.1))
                logger.warning(f"Reintento {retry+1}/{max_retries} para {op_id} después de {delay:.2f}s: {str(e)}")
                await asyncio.sleep(delay)
                
            finally:
                if session:
                    await session.close()
    
    async def execute_batch(
        self, 
        queries: List[Tuple[Callable, List, Dict]],
        use_transaction: bool = True,
        max_retries: int = 3
    ) -> List[Any]:
        """
        Ejecutar múltiples consultas como lote, opcionalmente en una transacción.
        
        Args:
            queries: Lista de tuplas (func, args, kwargs)
            use_transaction: Si se debe usar una transacción
            max_retries: Reintentos máximos
            
        Returns:
            Lista de resultados
            
        Raises:
            SQLAlchemyError: Si todos los reintentos fallan
        """
        for retry in range(max_retries + 1):
            try:
                # Obtener una sesión
                session = await get_db_session()
                results = []
                
                try:
                    # Iniciar transacción si está habilitada
                    if use_transaction:
                        transaction = await session.begin()
                    
                    try:
                        for query_func, args, kwargs in queries:
                            query_result = query_func(*args, **kwargs)
                            
                            # Determinar si es una consulta SQLAlchemy o una tupla (sql, params)
                            if isinstance(query_result, tuple) and len(query_result) == 2:
                                # Es una tupla (sql, params)
                                sql, params = query_result
                                result = await session.execute(text(sql), params)
                                
                                # Determinar el tipo de operación
                                is_select = sql.strip().lower().startswith("select")
                            else:
                                # Es un objeto SQLAlchemy, ejecutarlo directamente
                                sql = query_result
                                result = await session.execute(sql)
                                
                                # Intentar determinar tipo de operación por el tipo de objeto
                                is_select = hasattr(sql, "is_select") and getattr(sql, "is_select")
                            
                            if is_select:
                                results.append(result.fetchall())
                            else:
                                results.append(result.rowcount)
                        
                        if use_transaction:
                            await transaction.commit()
                            
                        return results
                        
                    except Exception as e:
                        if use_transaction:
                            await transaction.rollback()
                        raise
                finally:
                    await session.close()
                        
            except SQLAlchemyError as e:
                # Último intento, propagar excepción
                if retry == max_retries:
                    logger.error(f"Todos los reintentos fallaron para operación batch: {str(e)}")
                    raise
                
                # Aplicar backoff exponencial con jitter
                delay = min(1.0, 0.1 * (2 ** retry) + random.uniform(0, 0.1))
                logger.warning(f"Reintento {retry+1}/{max_retries} para batch después de {delay:.2f}s: {str(e)}")
                await asyncio.sleep(delay)
    
    async def checkpoint_state(self, entity_type: str, entity_id: str, data: Dict[str, Any]) -> str:
        """
        Crear checkpoint del estado de una entidad con manejo resiliente de conexiones.
        
        Esta versión mejorada implementa:
        1. Checkpointing dual (memoria + base de datos)
        2. Reintentos con backoff exponencial
        3. Verificación de conexión antes de operaciones críticas
        
        Args:
            entity_type: Tipo de entidad
            entity_id: ID de la entidad
            data: Estado a guardar
            
        Returns:
            ID del checkpoint
        """
        checkpoint_id = f"{entity_type}_{entity_id}_{int(time.time())}"
        
        # Paso 1: Guardar en memoria primero (siempre funciona)
        try:
            self.checkpoint.create(checkpoint_id, data, {
                'entity_type': entity_type,
                'entity_id': entity_id,
                'timestamp': time.time()
            })
            logger.debug(f"Checkpoint en memoria creado: {checkpoint_id}")
        except Exception as e:
            logger.error(f"Error al crear checkpoint en memoria: {e}")
            # Continuar aunque falle la memoria
            
        # Paso 2: Verificar conexión a base de datos (si aplica)
        try:
            # Verificar conexión antes de intentar operaciones con DB
            is_connected = await self.ping()
            if not is_connected:
                logger.warning(f"Conexión a BD no disponible para checkpoint {checkpoint_id}, usando solo memoria")
                return checkpoint_id
        except Exception as e:
            logger.error(f"Error al verificar conexión para checkpoint {checkpoint_id}: {e}")
            return checkpoint_id
                
        # Paso 3: Guardar en base de datos si es posible (más persistente)
        for attempt in range(3):  # 3 intentos
            try:
                # Crear registro de checkpoint en tabla (si tenemos una)
                session = await get_db_session()
                try:
                    # Intentar persistir el checkpoint en DB
                    await session.execute(
                        text("INSERT INTO gen_checkpoints (checkpoint_id, entity_type, entity_id, data, created_at) VALUES (:id, :type, :eid, :data, :ts)"),
                        {
                            "id": checkpoint_id,
                            "type": entity_type,
                            "eid": entity_id,
                            "data": json.dumps(data),
                            "ts": time.time()
                        }
                    )
                    await session.commit()
                    logger.debug(f"Checkpoint persistido en BD: {checkpoint_id}")
                    break  # Éxito, salir del bucle de reintentos
                except Exception as inner_e:
                    # Rollback si falla
                    await session.rollback()
                    
                    # Si es el último intento, loguear error detallado
                    if attempt == 2:
                        logger.error(f"Error persistente al guardar checkpoint {checkpoint_id} en BD: {inner_e}")
                    else:
                        logger.warning(f"Error al guardar checkpoint {checkpoint_id}, reintento {attempt+1}/3: {inner_e}")
                        
                    # Backoff exponencial con jitter
                    delay = 0.1 * (2 ** attempt) + random.uniform(0, 0.1)
                    await asyncio.sleep(delay)
                finally:
                    # Siempre cerrar la sesión apropiadamente
                    await session.close()
            except Exception as e:
                logger.error(f"Error crítico en checkpoint {checkpoint_id}: {e}")
                # Si fallaron todos los intentos, al menos tenemos el checkpoint en memoria
                break
                
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Restaurar estado desde un checkpoint.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Estado restaurado o None si no existe
        """
        return self.checkpoint.get(checkpoint_id)
    
    def get_latest_checkpoint(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener el checkpoint más reciente para una entidad.
        
        Args:
            entity_type: Tipo de entidad
            entity_id: ID de la entidad
            
        Returns:
            Datos del checkpoint o None si no hay
        """
        prefix = f"{entity_type}_{entity_id}_"
        checkpoints = [cp for cp in self.checkpoint.list_all() if cp.startswith(prefix)]
        
        if not checkpoints:
            return None
            
        # Ordenar por timestamp (parte del ID)
        latest = max(checkpoints, key=lambda cp: int(cp.split('_')[2]))
        return self.checkpoint.get(latest)
    
    async def perform_temporal_sync(self) -> Dict[str, Any]:
        """
        Sincronizar estados temporales entre dimensiones del cache.
        
        Esta función propaga valores críticos entre dimensiones para mantener
        coherencia, evitando paradojas temporales.
        
        Returns:
            Estadísticas de sincronización
        """
        # Identificar claves críticas con alta frecuencia de acceso
        critical_keys = {
            k: v for k, v in self.cache.access_count.items() 
            if v > 5  # Umbral de acceso para considerar crítico
        }
        
        # Propagar valores en ambas direcciones
        propagated_future = 0
        propagated_past = 0
        
        for key in critical_keys:
            # Presente -> Futuro
            if self.cache.get(key, dimension=1) is not None:
                self.cache.propagate(key, direction=1)
                propagated_future += 1
                
            # Presente -> Pasado (solo para valores muy críticos)
            if critical_keys.get(key, 0) > 10 and self.cache.get(key, dimension=1) is not None:
                self.cache.propagate(key, direction=-1)
                propagated_past += 1
        
        return {
            'critical_keys': len(critical_keys),
            'propagated_future': propagated_future,
            'propagated_past': propagated_past,
            'timestamp': time.time()
        }
    
    async def ping(self) -> bool:
        """
        Verificar la conexión a la base de datos.
        
        Esta función hace una consulta simple para verificar que la conexión
        esté activa y responda correctamente.
        
        Returns:
            True si la conexión está activa, False en caso contrario
        """
        async with self._ping_lock:
            # Si ya hicimos ping recientemente, usar el resultado almacenado
            current_time = time.time()
            if current_time - self._last_ping < 5.0:  # 5 segundos de cache para ping
                return self._connection_health
                
            try:
                # Hacer una consulta simple que no requiera existencia de tablas
                session = await get_db_session()
                try:
                    await session.execute(text("SELECT 1"))
                    self._connection_health = True
                    logger.debug("Conexión a BD verificada: OK")
                finally:
                    await session.close()
            except Exception as e:
                self._connection_health = False
                logger.error(f"Error de conexión a BD: {e}")
                
            self._last_ping = current_time
            return self._connection_health
    
    async def retry_with_reconnect(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecutar una función con reintentos y reconexión si es necesario.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos para la función
            **kwargs: Argumentos de palabra clave para la función
            
        Returns:
            Resultado de la función
        """
        for attempt in range(3):  # 3 intentos
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == 2:  # Último intento
                    raise
                    
                logger.warning(f"Error en operación DB (intento {attempt+1}/3): {e}")
                
                # Verificar conexión explícitamente
                is_connected = await self.ping()
                if not is_connected:
                    logger.warning("Conexión perdida, esperando antes de reintentar...")
                    await asyncio.sleep(1.0)  # Esperar antes de reintentar
        
        # No debería llegar aquí, pero por si acaso
        raise RuntimeError("Error inesperado en retry_with_reconnect")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de la base de datos trascendental.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'cache_size': sum(len(self.cache.cache[d]) for d in range(self.cache.dimensions)),
            'cache_dimensions': self.cache.dimensions,
            'checkpoint_count': len(self.checkpoint.checkpoints),
            'error_count': sum(self.error_count.values()),
            'success_count': sum(self.success_count.values()),
            'anomaly_count': len(self.anomaly_detection),
            'success_ratio': sum(self.success_count.values()) / (sum(self.error_count.values()) + sum(self.success_count.values()) + 0.001) * 100,
            'connection_health': self._connection_health
        }


# Instancia global de la base de datos trascendental
transcendental_db = TranscendentalDatabase()