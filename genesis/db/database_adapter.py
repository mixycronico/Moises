"""
Adaptador de base de datos para el Sistema Genesis.

Este módulo proporciona un adaptador que permite usar tanto el módulo
de base de datos asíncrono como el sincrónico, dependiendo del contexto.
"""
import asyncio
import logging
import functools
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic, Tuple

from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.db.sync_database import SyncDatabase

# Configuración de logging
logger = logging.getLogger(__name__)

# Tipo genérico para el resultado de una función
T = TypeVar('T')

class DatabaseAdapter:
    """
    Adaptador para interactuar con la base de datos de forma asíncrona o síncrona.
    
    Esta clase determina automáticamente si debe usar operaciones asíncronas o síncronas
    basándose en el contexto de ejecución y proporciona una interfaz unificada.
    """
    
    def __init__(self, async_db: Optional[TranscendentalDatabase] = None, 
                sync_db: Optional[SyncDatabase] = None):
        """
        Inicializar el adaptador con instancias de bases de datos.
        
        Si no se proporcionan, se crean instancias nuevas cuando sean necesarias.
        
        Args:
            async_db: Instancia de TranscendentalDatabase (opcional)
            sync_db: Instancia de SyncDatabase (opcional)
        """
        self._async_db = async_db
        self._sync_db = sync_db
        self._created_instances = []
        
    async def _get_async_db(self) -> TranscendentalDatabase:
        """
        Obtener la instancia de base de datos asíncrona.
        
        Returns:
            Instancia de TranscendentalDatabase
        """
        if self._async_db is None:
            logger.debug("Creando nueva instancia de TranscendentalDatabase")
            self._async_db = TranscendentalDatabase()
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
            self._sync_db = SyncDatabase()
            self._created_instances.append(self._sync_db)
        return self._sync_db
        
    def is_async_context(self) -> bool:
        """
        Determinar si el código se está ejecutando en un contexto asíncrono.
        
        Returns:
            True si está en contexto asíncrono, False en caso contrario
        """
        try:
            # Intentar obtener el loop actual, si hay uno
            loop = asyncio.get_running_loop()
            return True
        except RuntimeError:
            # Si no hay loop, estamos en un contexto síncrono
            return False
            
    async def execute_query(self, query: str, params: Optional[Union[Dict[str, Any], Tuple, List]] = None,
                          fetch_all: bool = False, fetch_one: bool = False) -> Any:
        """
        Ejecutar una consulta SQL usando el modo asíncrono.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            fetch_all: Si es True, retorna todos los resultados
            fetch_one: Si es True, retorna solo el primer resultado
            
        Returns:
            Resultados de la consulta según las opciones de fetch
        """
        db = await self._get_async_db()
        
        if fetch_all:
            return await db.fetch(query, params)
        elif fetch_one:
            return await db.fetch_one(query, params)
        else:
            return await db.execute(query, params)
            
    def execute_query_sync(self, query: str, params: Optional[Union[Dict[str, Any], Tuple, List]] = None,
                         fetch_all: bool = False, fetch_one: bool = False) -> Any:
        """
        Ejecutar una consulta SQL usando el modo síncrono.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            fetch_all: Si es True, retorna todos los resultados
            fetch_one: Si es True, retorna solo el primer resultado
            
        Returns:
            Resultados de la consulta según las opciones de fetch
        """
        db = self._get_sync_db()
        
        if fetch_all:
            return db.fetch_all(query, params)
        elif fetch_one:
            return db.fetch_one(query, params)
        else:
            return db.execute(query, params)
            
    async def execute(self, query: str, params: Optional[Union[Dict[str, Any], Tuple, List]] = None) -> Any:
        """
        Ejecutar una consulta SQL adaptándose automáticamente al contexto.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            
        Returns:
            Resultado de la ejecución
        """
        if self.is_async_context():
            return await self.execute_query(query, params)
        else:
            return self.execute_query_sync(query, params)
            
    async def fetch(self, query: str, params: Optional[Union[Dict[str, Any], Tuple, List]] = None) -> List[Dict[str, Any]]:
        """
        Ejecutar consulta y retornar todos los resultados.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            
        Returns:
            Lista de diccionarios con los resultados
        """
        if self.is_async_context():
            return await self.execute_query(query, params, fetch_all=True)
        else:
            return self.execute_query_sync(query, params, fetch_all=True)
            
    async def fetch_one(self, query: str, params: Optional[Union[Dict[str, Any], Tuple, List]] = None) -> Optional[Dict[str, Any]]:
        """
        Ejecutar consulta y retornar solo el primer resultado.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            
        Returns:
            Diccionario con el primer resultado o None si no hay resultados
        """
        if self.is_async_context():
            return await self.execute_query(query, params, fetch_one=True)
        else:
            return self.execute_query_sync(query, params, fetch_one=True)
            
    async def fetch_val(self, query: str, params: Optional[Union[Dict[str, Any], Tuple, List]] = None, 
                       default: Any = None) -> Any:
        """
        Ejecutar consulta y retornar un único valor.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            default: Valor por defecto si no hay resultados
            
        Returns:
            Primer valor del primer resultado o default si no hay resultados
        """
        row = await self.fetch_one(query, params)
        if not row:
            return default
            
        # Retornar el primer valor
        return next(iter(row.values()), default)
        
    async def run_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Ejecutar una operación adaptándose al contexto actual.
        
        Esta función determina automáticamente si debe ejecutar la operación
        de forma asíncrona o síncrona basándose en el contexto actual.
        
        Args:
            operation: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la operación
        """
        # Si estamos en un contexto asíncrono y la operación es una corrutina
        if self.is_async_context() and asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        # Si estamos en un contexto síncrono pero la operación es una corrutina
        elif asyncio.iscoroutinefunction(operation):
            # Necesitamos ejecutarla en un nuevo loop
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(operation(*args, **kwargs))
            finally:
                loop.close()
        # Si la operación es una función normal
        else:
            return operation(*args, **kwargs)
            
    def get_native_db(self) -> Union[TranscendentalDatabase, SyncDatabase]:
        """
        Obtener la instancia nativa de base de datos según el contexto.
        
        Returns:
            TranscendentalDatabase o SyncDatabase
        """
        if self.is_async_context():
            # No podemos usar await directamente en funciones sincrónicas
            if self._async_db:
                return self._async_db
            # Necesitamos un manejo especial aquí
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(self._get_async_db(), loop)
            return future.result()
        else:
            return self._get_sync_db()
            
    async def close(self) -> None:
        """Cerrar todas las conexiones creadas por este adaptador."""
        for instance in self._created_instances:
            if isinstance(instance, TranscendentalDatabase):
                await instance.close()
            elif isinstance(instance, SyncDatabase):
                instance.close()
                
        self._created_instances = []
        
    async def __aenter__(self) -> 'DatabaseAdapter':
        """Para usar con async with statement."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Para usar con async with statement."""
        await self.close()

# Singleton global para uso en toda la aplicación
_global_adapter = None

def get_db_adapter() -> DatabaseAdapter:
    """
    Obtener la instancia global del adaptador de base de datos.
    
    Returns:
        Instancia de DatabaseAdapter
    """
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = DatabaseAdapter()
    return _global_adapter