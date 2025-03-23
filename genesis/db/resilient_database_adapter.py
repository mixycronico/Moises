"""
Adaptador de base de datos resiliente para el Sistema Genesis.

Este módulo proporciona un adaptador mejorado con manejo robusto de conexiones,
sistema de reintentos con backoff exponencial y pool de conexiones optimizado.
"""
import asyncio
import logging
import time
import functools
import random
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import event

from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.db.sync_database import SyncDatabase

# Configuración de logging
logger = logging.getLogger(__name__)

# Tipo genérico para el resultado de una función
T = TypeVar('T')

class ResilientDatabaseAdapter:
    """
    Adaptador resiliente para interactuar con la base de datos.
    
    Esta clase extiende el DatabaseAdapter base con características adicionales:
    - Pool de conexiones con reconexión automática
    - Reintentos con backoff exponencial
    - Estrategia de jitter para evitar tormentas de conexión
    - Capacidades de checkpoint mejoradas
    """
    
    def __init__(
        self, 
        async_db: Optional[TranscendentalDatabase] = None, 
        sync_db: Optional[SyncDatabase] = None,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_recycle: int = 300,
        pool_pre_ping: bool = True,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 5.0
    ):
        """
        Inicializar el adaptador resiliente con opciones avanzadas.
        
        Args:
            async_db: Instancia de TranscendentalDatabase (opcional)
            sync_db: Instancia de SyncDatabase (opcional)
            pool_size: Tamaño del pool de conexiones
            max_overflow: Máximo de conexiones adicionales temporales
            pool_recycle: Tiempo en segundos para reciclar conexiones
            pool_pre_ping: Si se debe hacer ping a la BD antes de usar conexiones
            max_retries: Número máximo de reintentos para operaciones fallidas
            base_delay: Retraso base inicial para reintentos (segundos)
            max_delay: Retraso máximo para reintentos (segundos)
        """
        self._async_db = async_db
        self._sync_db = sync_db
        self._created_instances = []
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        self._connection_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._active_sessions = 0
        self._checkpoint_state = {}
        
        logger.info(f"ResilientDatabaseAdapter inicializado con pool_size={pool_size}, max_retries={max_retries}")
        
    async def _get_async_db(self) -> TranscendentalDatabase:
        """
        Obtener la instancia de base de datos asíncrona con reconexión automática.
        
        Returns:
            Instancia de TranscendentalDatabase
        """
        async with self._connection_lock:
            if self._async_db is None:
                logger.debug("Creando nueva instancia resiliente de TranscendentalDatabase")
                self._async_db = TranscendentalDatabase(
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_recycle=self.pool_recycle,
                    pool_pre_ping=self.pool_pre_ping
                )
                self._created_instances.append(self._async_db)
            
            # Verificar y reparar la conexión si es necesario
            try:
                # Verificar conexión con operación simple
                await self._async_db.ping()
            except Exception as e:
                logger.warning(f"Conexión perdida con TranscendentalDatabase: {e}")
                logger.info("Intentando reconexión...")
                self._async_db = TranscendentalDatabase(
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_recycle=self.pool_recycle,
                    pool_pre_ping=self.pool_pre_ping
                )
                self._created_instances.append(self._async_db)
                
        return self._async_db
        
    def _get_sync_db(self) -> SyncDatabase:
        """
        Obtener la instancia de base de datos síncrona.
        
        Returns:
            Instancia de SyncDatabase
        """
        if self._sync_db is None:
            logger.debug("Creando nueva instancia de SyncDatabase")
            self._sync_db = SyncDatabase(
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=self.pool_pre_ping
            )
            self._created_instances.append(self._sync_db)
        return self._sync_db
        
    def is_async_context(self) -> bool:
        """
        Determinar si el contexto actual es asíncrono.
        
        Returns:
            True si el contexto es asíncrono, False si es síncrono
        """
        try:
            loop = asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False
    
    async def track_session(self, transaction_id: str, func: Callable):
        """
        Realizar seguimiento de sesiones activas para detectar bloqueos.
        
        Args:
            transaction_id: Identificador único de la transacción
            func: Función a ejecutar dentro del seguimiento
        
        Returns:
            Resultado de la función ejecutada
        """
        async with self._session_lock:
            self._active_sessions += 1
            session_id = random.randint(1000, 9999)
            logger.debug(f"Iniciando sesión {session_id} para transacción {transaction_id}. Sesiones activas: {self._active_sessions}")
        
        start_time = time.time()
        try:
            result = await func()
            return result
        finally:
            execution_time = time.time() - start_time
            async with self._session_lock:
                self._active_sessions -= 1
                logger.debug(f"Finalizando sesión {session_id} para transacción {transaction_id}. "
                           f"Tiempo: {execution_time:.3f}s. Sesiones activas: {self._active_sessions}")
                
                # Detectar posibles problemas si hay muchas sesiones activas o ejecución lenta
                if execution_time > 5.0:
                    logger.warning(f"Transacción {transaction_id} tardó {execution_time:.3f}s en completarse")
                if self._active_sessions > self.pool_size * 0.8:
                    logger.warning(f"Alto número de sesiones activas: {self._active_sessions} de {self.pool_size}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def execute_query(self, query: str, params: Any = None, transaction_id: str = None) -> List[Dict[str, Any]]:
        """
        Ejecutar una consulta SQL con reintentos automáticos.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            transaction_id: Identificador único de la transacción
        
        Returns:
            Resultado de la consulta como lista de diccionarios
        """
        if transaction_id is None:
            transaction_id = f"query_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            
        if self.is_async_context():
            db = await self._get_async_db()
            
            async def execute():
                return await db.execute_query(query, params)
                
            return await self.track_session(transaction_id, execute)
        else:
            db = self._get_sync_db()
            return db.execute_query(query, params)
    
    async def checkpoint_state(self, component_id: str, state: Dict[str, Any], durable: bool = True) -> bool:
        """
        Guardar un checkpoint de estado con manejo de errores mejorado.
        
        Esta implementación:
        1. Guarda primero en memoria (rápido y sin riesgo de error de conexión)
        2. Luego guarda en BD si durable=True (persistente pero podría fallar)
        3. Usa reintentos con backoff exponencial para la persistencia
        
        Args:
            component_id: Identificador del componente
            state: Estado a guardar
            durable: Si se debe persistir en la base de datos
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        # Guardar en memoria primero (siempre funciona)
        checkpoint_id = f"checkpoint_{component_id}_{int(time.time())}"
        self._checkpoint_state[component_id] = {
            "state": state,
            "timestamp": time.time(),
            "checkpoint_id": checkpoint_id
        }
        
        logger.debug(f"Checkpoint en memoria creado para {component_id}")
        
        # Si no se requiere durabilidad, terminar aquí
        if not durable:
            return True
            
        # Intentar guardar en la base de datos con reintentos
        if self.is_async_context():
            try:
                db = await self._get_async_db()
                
                @retry(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=self.base_delay, min=self.base_delay, max=self.max_delay),
                    reraise=True
                )
                async def persist_checkpoint():
                    await db.checkpoint.create(checkpoint_id, state, {
                        "component_id": component_id,
                        "timestamp": time.time()
                    })
                    
                await persist_checkpoint()
                logger.debug(f"Checkpoint persistente creado para {component_id}")
                return True
            except Exception as e:
                logger.error(f"Error al persistir checkpoint para {component_id}: {e}")
                # Fallback: al menos tenemos el checkpoint en memoria
                return False
        else:
            # Versión síncrona simplificada
            try:
                db = self._get_sync_db()
                db.save_checkpoint(checkpoint_id, state, component_id)
                return True
            except Exception as e:
                logger.error(f"Error al persistir checkpoint síncrono para {component_id}: {e}")
                return False
                
    async def restore_from_checkpoint(self, component_id: str, fallback_to_memory: bool = True) -> Optional[Dict[str, Any]]:
        """
        Restaurar estado desde un checkpoint con manejo de errores mejorado.
        
        Args:
            component_id: Identificador del componente
            fallback_to_memory: Si se debe usar el checkpoint en memoria como fallback
            
        Returns:
            Estado restaurado o None si no existe
        """
        if self.is_async_context():
            try:
                db = await self._get_async_db()
                checkpoint_id = f"checkpoint_{component_id}_*"  # Patrón para buscar el más reciente
                
                @retry(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=self.base_delay, min=self.base_delay, max=self.max_delay),
                    reraise=True
                )
                async def retrieve_checkpoint():
                    # Buscar el checkpoint más reciente por patrón
                    checkpoints = await db.list_checkpoints(component_id)
                    if not checkpoints:
                        return None
                    
                    # Tomar el más reciente
                    latest = max(checkpoints, key=lambda c: c.get("timestamp", 0))
                    return await db.checkpoint.get(latest["checkpoint_id"])
                
                state = await retrieve_checkpoint()
                if state:
                    logger.debug(f"Estado restaurado desde checkpoint en BD para {component_id}")
                    # Actualizar también la memoria
                    self._checkpoint_state[component_id] = {
                        "state": state,
                        "timestamp": time.time(),
                        "checkpoint_id": f"memory_{int(time.time())}"
                    }
                    return state
                    
                # Si no hay en BD pero sí en memoria y está permitido el fallback
                if fallback_to_memory and component_id in self._checkpoint_state:
                    logger.debug(f"Estado restaurado desde checkpoint en memoria para {component_id}")
                    return self._checkpoint_state[component_id]["state"]
                    
                return None
            except Exception as e:
                logger.error(f"Error al restaurar checkpoint para {component_id}: {e}")
                
                # Fallback a memoria si está permitido
                if fallback_to_memory and component_id in self._checkpoint_state:
                    logger.debug(f"Fallback: estado restaurado desde memoria para {component_id}")
                    return self._checkpoint_state[component_id]["state"]
                
                return None
        else:
            # Versión síncrona simplificada
            try:
                db = self._get_sync_db()
                state = db.get_latest_checkpoint(component_id)
                
                if not state and fallback_to_memory and component_id in self._checkpoint_state:
                    return self._checkpoint_state[component_id]["state"]
                    
                return state
            except Exception as e:
                logger.error(f"Error al restaurar checkpoint síncrono para {component_id}: {e}")
                
                if fallback_to_memory and component_id in self._checkpoint_state:
                    return self._checkpoint_state[component_id]["state"]
                
                return None
                
    async def close(self):
        """Cerrar todas las conexiones y liberar recursos."""
        for instance in self._created_instances:
            if hasattr(instance, 'close'):
                if asyncio.iscoroutinefunction(instance.close):
                    await instance.close()
                else:
                    instance.close()
        
        logger.info("ResilientDatabaseAdapter cerrado correctamente")

# Instancia global para uso en toda la aplicación
resilient_db = ResilientDatabaseAdapter()