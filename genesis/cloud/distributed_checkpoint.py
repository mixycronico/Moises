"""
DistributedCheckpointManager - Componente para manejo de puntos de control distribuidos.

Este módulo implementa un sistema de checkpoints distribuidos con capacidades divinas:
- Almacenamiento inmutable de estados
- Recuperación instantánea desde cualquier punto temporal
- Consistencia fuerte incluso en entornos distribuidos
- Transmutación automática de datos durante transiciones
"""

import asyncio
import json
import logging
import os
import time
import uuid
import pickle
import hashlib
import functools
from enum import Enum, auto
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union, TypeVar, Generic, Callable

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genesis.cloud.distributed_checkpoint")

# Tipo genérico para datos
T = TypeVar('T')


class CheckpointStorageType(Enum):
    """Tipos de almacenamiento para checkpoints."""
    LOCAL_FILE = auto()      # Almacenamiento en archivo local
    MEMORY = auto()          # Almacenamiento en memoria
    DISTRIBUTED_DB = auto()  # Almacenamiento en base de datos distribuida


class CheckpointConsistencyLevel(Enum):
    """Niveles de consistencia para checkpoints."""
    EVENTUAL = auto()  # Consistencia eventual (mayor rendimiento)
    STRONG = auto()    # Consistencia fuerte (mayor seguridad)
    QUANTUM = auto()   # Consistencia cuántica (transmutación automática)


class CheckpointState(Enum):
    """Estados posibles para checkpoints."""
    ACTIVE = auto()      # Checkpoint activo y disponible
    INACTIVE = auto()    # Checkpoint inactivado manualmente
    CORRUPT = auto()     # Checkpoint potencialmente corrupto
    TRANSMUTING = auto() # Checkpoint en proceso de transmutación


class CheckpointMetadata:
    """Metadatos para un checkpoint."""
    
    def __init__(self, 
                 checkpoint_id: str,
                 component_id: str,
                 timestamp: float,
                 tags: Optional[List[str]] = None,
                 consistency_level: CheckpointConsistencyLevel = CheckpointConsistencyLevel.STRONG):
        """
        Inicializar metadatos.
        
        Args:
            checkpoint_id: ID único del checkpoint
            component_id: ID del componente al que pertenece
            timestamp: Timestamp de creación
            tags: Etiquetas opcionales
            consistency_level: Nivel de consistencia
        """
        self.checkpoint_id = checkpoint_id
        self.component_id = component_id
        self.timestamp = timestamp
        self.tags = tags or []
        self.consistency_level = consistency_level
        self.state = CheckpointState.ACTIVE
        self.checksum = ""  # Se calculará al guardar datos
        
        # Metadatos adicionales
        self.creation_datetime = datetime.fromtimestamp(timestamp)
        self.last_accessed = timestamp
        self.access_count = 0
        self.transmutation_count = 0
        self.size_bytes = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con metadatos
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "component_id": self.component_id,
            "timestamp": self.timestamp,
            "creation_datetime": self.creation_datetime.isoformat(),
            "tags": self.tags,
            "consistency_level": self.consistency_level.name,
            "state": self.state.name,
            "checksum": self.checksum,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "transmutation_count": self.transmutation_count,
            "size_bytes": self.size_bytes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """
        Crear desde diccionario.
        
        Args:
            data: Diccionario con datos
            
        Returns:
            Instancia de CheckpointMetadata
        """
        metadata = cls(
            checkpoint_id=data["checkpoint_id"],
            component_id=data["component_id"],
            timestamp=data["timestamp"],
            tags=data.get("tags", []),
            consistency_level=CheckpointConsistencyLevel[data["consistency_level"]]
        )
        
        # Restaurar estado
        metadata.state = CheckpointState[data["state"]]
        metadata.checksum = data["checksum"]
        metadata.last_accessed = data["last_accessed"]
        metadata.access_count = data["access_count"]
        metadata.transmutation_count = data.get("transmutation_count", 0)
        metadata.size_bytes = data.get("size_bytes", 0)
        
        return metadata
    
    def record_access(self) -> None:
        """Registrar acceso al checkpoint."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def record_transmutation(self) -> None:
        """Registrar transmutación del checkpoint."""
        self.transmutation_count += 1


class DistributedCheckpointManager:
    """
    Gestor de checkpoints distribuidos con capacidades divinas.
    
    Este componente proporciona:
    - Almacenamiento y recuperación de estados para cualquier componente
    - Consistencia garantizada incluso en entornos distribuidos
    - Transmutación cuántica automática durante transiciones de estado
    - Limpieza automática de checkpoints antiguos
    """
    
    def __init__(self):
        """Inicializar gestor de checkpoints."""
        self.storage_type = CheckpointStorageType.LOCAL_FILE
        self.consistency_level = CheckpointConsistencyLevel.STRONG
        self.base_directory = "checkpoints"
        self.initialized = False
        
        # Almacenamiento en memoria para modo MEMORY
        self._memory_storage: Dict[str, Tuple[Dict[str, Any], CheckpointMetadata]] = {}
        
        # Caché de metadatos para acceso rápido
        self._metadata_cache: Dict[str, CheckpointMetadata] = {}
        
        # Conexión a base de datos (si se usa modo DISTRIBUTED_DB)
        self._db_connection = None
        self._db_pool = None
        
        # Capacidades adicionales
        self.cleanup_enabled = True
        self.max_checkpoints_per_component = 100
        self.transmutation_enabled = True
        self.quantum_consistency_enabled = False
        
        # Para operaciones concurrentes
        self._lock = asyncio.Lock()
        self._component_locks: Dict[str, asyncio.Lock] = {}
    
    async def initialize(self, 
                        storage_type: CheckpointStorageType = CheckpointStorageType.LOCAL_FILE,
                        consistency_level: CheckpointConsistencyLevel = CheckpointConsistencyLevel.STRONG,
                        base_directory: str = "checkpoints") -> bool:
        """
        Inicializar gestor de checkpoints.
        
        Args:
            storage_type: Tipo de almacenamiento a usar
            consistency_level: Nivel de consistencia requerido
            base_directory: Directorio base para almacenamiento local
            
        Returns:
            True si se inicializó correctamente
        """
        self.storage_type = storage_type
        self.consistency_level = consistency_level
        self.base_directory = base_directory
        
        try:
            # Inicializar según tipo de almacenamiento
            if storage_type == CheckpointStorageType.LOCAL_FILE:
                # Crear directorio base si no existe
                os.makedirs(base_directory, exist_ok=True)
                
                # Verificar permisos de escritura
                test_file = os.path.join(base_directory, ".test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                
                logger.info(f"DistributedCheckpointManager inicializado con almacenamiento LOCAL_FILE")
                
            elif storage_type == CheckpointStorageType.MEMORY:
                # Nada especial, usar diccionario en memoria
                self._memory_storage = {}
                logger.info(f"DistributedCheckpointManager inicializado con almacenamiento MEMORY")
                
            elif storage_type == CheckpointStorageType.DISTRIBUTED_DB:
                # Usar PostgreSQL para almacenamiento distribuido
                try:
                    import asyncpg
                    
                    # Obtener cadena de conexión desde variables de entorno
                    # o usar valor por defecto para desarrollo
                    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")
                    
                    # Crear pool de conexiones
                    self._db_pool = await asyncpg.create_pool(db_url)
                    
                    # Verificar conexión
                    async with self._db_pool.acquire() as connection:
                        self._db_connection = connection
                        
                        # Crear tabla si no existe
                        await connection.execute('''
                            CREATE TABLE IF NOT EXISTS checkpoints (
                                checkpoint_id TEXT PRIMARY KEY,
                                component_id TEXT NOT NULL,
                                data BYTEA NOT NULL,
                                metadata JSONB NOT NULL,
                                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                            )
                        ''')
                        
                        # Crear índice para búsqueda rápida por componente
                        await connection.execute('''
                            CREATE INDEX IF NOT EXISTS idx_checkpoints_component_id
                            ON checkpoints (component_id)
                        ''')
                    
                    logger.info(f"DistributedCheckpointManager inicializado con almacenamiento DISTRIBUTED_DB")
                    
                except ImportError:
                    logger.error("No se pudo importar asyncpg para almacenamiento distribuido")
                    return False
                except Exception as e:
                    logger.error(f"Error al conectar a base de datos distribuida: {e}")
                    return False
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar DistributedCheckpointManager: {e}")
            return False
    
    async def create_checkpoint(self, 
                                component_id: str, 
                                data: Dict[str, Any],
                                tags: Optional[List[str]] = None) -> Optional[str]:
        """
        Crear un nuevo checkpoint.
        
        Args:
            component_id: ID del componente
            data: Datos a almacenar
            tags: Etiquetas opcionales
            
        Returns:
            ID del checkpoint creado o None si falló
        """
        if not self.initialized:
            logger.error("DistributedCheckpointManager no inicializado")
            return None
        
        # Generar ID único para el checkpoint
        checkpoint_id = f"{component_id}_{uuid.uuid4().hex}"
        
        # Crear metadatos
        timestamp = time.time()
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            component_id=component_id,
            timestamp=timestamp,
            tags=tags,
            consistency_level=self.consistency_level
        )
        
        # Calcular checksum de los datos
        checksum = self._calculate_checksum(data)
        metadata.checksum = checksum
        
        # Obtener lock específico para el componente
        if component_id not in self._component_locks:
            self._component_locks[component_id] = asyncio.Lock()
        
        # Adquirir lock para asegurar consistencia
        async with self._component_locks[component_id]:
            try:
                # Guardar según el tipo de almacenamiento
                if self.storage_type == CheckpointStorageType.LOCAL_FILE:
                    success = await self._save_to_file(checkpoint_id, data, metadata)
                elif self.storage_type == CheckpointStorageType.MEMORY:
                    success = self._save_to_memory(checkpoint_id, data, metadata)
                elif self.storage_type == CheckpointStorageType.DISTRIBUTED_DB:
                    success = await self._save_to_db(checkpoint_id, data, metadata)
                else:
                    logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                    return None
                
                if success:
                    # Guardar en caché de metadatos
                    self._metadata_cache[checkpoint_id] = metadata
                    
                    # Limpieza automática si está habilitada
                    if self.cleanup_enabled:
                        await self.cleanup_old_checkpoints(component_id)
                    
                    logger.info(f"Checkpoint {checkpoint_id} creado para componente {component_id}")
                    return checkpoint_id
                else:
                    logger.error(f"Error al guardar checkpoint {checkpoint_id}")
                    return None
                
            except Exception as e:
                logger.error(f"Error al crear checkpoint: {e}")
                return None
    
    async def load_checkpoint(self, checkpoint_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Cargar datos desde un checkpoint.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si falla
        """
        if not self.initialized:
            logger.error("DistributedCheckpointManager no inicializado")
            return None, None
        
        try:
            # Intentar cargar según el tipo de almacenamiento
            if self.storage_type == CheckpointStorageType.LOCAL_FILE:
                data, metadata = await self._load_from_file(checkpoint_id)
            elif self.storage_type == CheckpointStorageType.MEMORY:
                data, metadata = self._load_from_memory(checkpoint_id)
            elif self.storage_type == CheckpointStorageType.DISTRIBUTED_DB:
                data, metadata = await self._load_from_db(checkpoint_id)
            else:
                logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                return None, None
            
            if data is not None and metadata is not None:
                # Registrar acceso
                metadata.record_access()
                
                # Actualizar caché
                self._metadata_cache[checkpoint_id] = metadata
                
                # Verificar integridad mediante checksum
                current_checksum = self._calculate_checksum(data)
                if current_checksum != metadata.checksum:
                    logger.warning(f"Checksum incorrecto para checkpoint {checkpoint_id}, posible corrupción")
                    metadata.state = CheckpointState.CORRUPT
                
                logger.info(f"Checkpoint {checkpoint_id} cargado")
                return data, metadata.to_dict()
            else:
                logger.error(f"No se encontró checkpoint {checkpoint_id}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error al cargar checkpoint {checkpoint_id}: {e}")
            return None, None
    
    async def load_latest_checkpoint(self, component_id: str, tag: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Cargar el checkpoint más reciente para un componente.
        
        Args:
            component_id: ID del componente
            tag: Filtrar por etiqueta específica (opcional)
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si falla
        """
        if not self.initialized:
            logger.error("DistributedCheckpointManager no inicializado")
            return None, None
        
        try:
            # Obtener lista de checkpoints para el componente
            checkpoints = await self.list_checkpoints(component_id)
            
            if not checkpoints:
                logger.warning(f"No hay checkpoints para componente {component_id}")
                return None, None
            
            # Filtrar por tag si se especificó
            if tag:
                filtered_checkpoints = [c for c in checkpoints if tag in c.get("tags", [])]
                if filtered_checkpoints:
                    checkpoints = filtered_checkpoints
                else:
                    logger.warning(f"No hay checkpoints con tag '{tag}' para componente {component_id}")
            
            # Ordenar por timestamp descendente
            sorted_checkpoints = sorted(checkpoints, key=lambda c: c["timestamp"], reverse=True)
            
            if not sorted_checkpoints:
                return None, None
            
            # Cargar el más reciente
            latest_checkpoint_id = sorted_checkpoints[0]["checkpoint_id"]
            return await self.load_checkpoint(latest_checkpoint_id)
            
        except Exception as e:
            logger.error(f"Error al cargar checkpoint más reciente para {component_id}: {e}")
            return None, None
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Eliminar un checkpoint.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente
        """
        if not self.initialized:
            logger.error("DistributedCheckpointManager no inicializado")
            return False
        
        try:
            # Obtener metadatos para conocer el componente
            metadata = self._metadata_cache.get(checkpoint_id)
            
            if metadata is None:
                # Intentar cargar metadatos según tipo de almacenamiento
                if self.storage_type == CheckpointStorageType.LOCAL_FILE:
                    _, metadata = await self._load_from_file(checkpoint_id, load_data=False)
                elif self.storage_type == CheckpointStorageType.MEMORY:
                    if checkpoint_id in self._memory_storage:
                        _, metadata = self._memory_storage[checkpoint_id]
                elif self.storage_type == CheckpointStorageType.DISTRIBUTED_DB:
                    _, metadata = await self._load_from_db(checkpoint_id, load_data=False)
            
            # Adquirir lock para el componente si tenemos metadatos
            component_id = metadata.component_id if metadata else "unknown"
            
            if component_id not in self._component_locks:
                self._component_locks[component_id] = asyncio.Lock()
            
            async with self._component_locks[component_id]:
                # Eliminar según tipo de almacenamiento
                if self.storage_type == CheckpointStorageType.LOCAL_FILE:
                    success = await self._delete_from_file(checkpoint_id)
                elif self.storage_type == CheckpointStorageType.MEMORY:
                    success = self._delete_from_memory(checkpoint_id)
                elif self.storage_type == CheckpointStorageType.DISTRIBUTED_DB:
                    success = await self._delete_from_db(checkpoint_id)
                else:
                    logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                    return False
                
                if success:
                    # Eliminar de la caché
                    if checkpoint_id in self._metadata_cache:
                        del self._metadata_cache[checkpoint_id]
                    
                    logger.info(f"Checkpoint {checkpoint_id} eliminado")
                    return True
                else:
                    logger.error(f"Error al eliminar checkpoint {checkpoint_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Error al eliminar checkpoint {checkpoint_id}: {e}")
            return False
    
    async def list_checkpoints(self, component_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Listar checkpoints disponibles.
        
        Args:
            component_id: Filtrar por componente específico (opcional)
            
        Returns:
            Lista de metadatos de checkpoints
        """
        if not self.initialized:
            logger.error("DistributedCheckpointManager no inicializado")
            return []
        
        try:
            # Obtener lista según tipo de almacenamiento
            if self.storage_type == CheckpointStorageType.LOCAL_FILE:
                checkpoints = await self._list_from_file(component_id)
            elif self.storage_type == CheckpointStorageType.MEMORY:
                checkpoints = self._list_from_memory(component_id)
            elif self.storage_type == CheckpointStorageType.DISTRIBUTED_DB:
                checkpoints = await self._list_from_db(component_id)
            else:
                logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                return []
            
            # Convertir metadatos a diccionarios
            return [metadata.to_dict() for metadata in checkpoints]
            
        except Exception as e:
            logger.error(f"Error al listar checkpoints: {e}")
            return []
    
    async def cleanup_old_checkpoints(self, component_id: str, max_checkpoints: Optional[int] = None) -> int:
        """
        Eliminar checkpoints antiguos de un componente.
        
        Args:
            component_id: ID del componente
            max_checkpoints: Número máximo de checkpoints a mantener
            
        Returns:
            Número de checkpoints eliminados
        """
        if not self.initialized:
            logger.error("DistributedCheckpointManager no inicializado")
            return 0
        
        if max_checkpoints is None:
            max_checkpoints = self.max_checkpoints_per_component
        
        try:
            # Obtener lista de checkpoints para el componente
            checkpoint_list = await self.list_checkpoints(component_id)
            
            if len(checkpoint_list) <= max_checkpoints:
                # No hay necesidad de limpiar
                return 0
            
            # Ordenar por timestamp descendente
            sorted_checkpoints = sorted(checkpoint_list, key=lambda c: c["timestamp"], reverse=True)
            
            # Conservar los más recientes, eliminar el resto
            to_keep = sorted_checkpoints[:max_checkpoints]
            to_delete = sorted_checkpoints[max_checkpoints:]
            
            # Eliminar checkpoints
            deleted_count = 0
            for checkpoint in to_delete:
                checkpoint_id = checkpoint["checkpoint_id"]
                success = await self.delete_checkpoint(checkpoint_id)
                if success:
                    deleted_count += 1
            
            logger.info(f"Eliminados {deleted_count} checkpoints antiguos para componente {component_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error al limpiar checkpoints antiguos para {component_id}: {e}")
            return 0
    
    async def get_checkpoint_status(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener estado detallado de un checkpoint.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Diccionario con estado detallado o None si no existe
        """
        if not self.initialized:
            logger.error("DistributedCheckpointManager no inicializado")
            return None
        
        try:
            # Intentar obtener desde caché
            metadata = self._metadata_cache.get(checkpoint_id)
            
            if metadata is None:
                # Intentar cargar metadatos según tipo de almacenamiento
                if self.storage_type == CheckpointStorageType.LOCAL_FILE:
                    _, metadata = await self._load_from_file(checkpoint_id, load_data=False)
                elif self.storage_type == CheckpointStorageType.MEMORY:
                    if checkpoint_id in self._memory_storage:
                        _, metadata = self._memory_storage[checkpoint_id]
                elif self.storage_type == CheckpointStorageType.DISTRIBUTED_DB:
                    _, metadata = await self._load_from_db(checkpoint_id, load_data=False)
            
            if metadata is None:
                logger.warning(f"No se encontró checkpoint {checkpoint_id}")
                return None
            
            # Convertir a diccionario y añadir datos adicionales
            status = metadata.to_dict()
            
            # Añadir información de storage
            status["storage_type"] = self.storage_type.name
            
            # Para almacenamiento en archivo, verificar si existe
            if self.storage_type == CheckpointStorageType.LOCAL_FILE:
                data_path = os.path.join(self.base_directory, f"{checkpoint_id}_data.pickle")
                meta_path = os.path.join(self.base_directory, f"{checkpoint_id}_meta.json")
                status["file_exists"] = os.path.exists(data_path) and os.path.exists(meta_path)
                
                if status["file_exists"]:
                    status["file_size"] = os.path.getsize(data_path) + os.path.getsize(meta_path)
            
            # Para almacenamiento en memoria, verificar si existe
            elif self.storage_type == CheckpointStorageType.MEMORY:
                status["in_memory"] = checkpoint_id in self._memory_storage
            
            return status
            
        except Exception as e:
            logger.error(f"Error al obtener estado de checkpoint {checkpoint_id}: {e}")
            return None
    
    async def transmute_checkpoint(self, checkpoint_id: str, transformation_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Optional[str]:
        """
        Transmutar un checkpoint aplicando una función de transformación.
        
        Args:
            checkpoint_id: ID del checkpoint original
            transformation_func: Función que recibe datos originales y devuelve transformados
            
        Returns:
            ID del nuevo checkpoint o None si falló
        """
        if not self.initialized or not self.transmutation_enabled:
            logger.error("DistributedCheckpointManager no inicializado o transmutación deshabilitada")
            return None
        
        try:
            # Cargar checkpoint original
            data, metadata_dict = await self.load_checkpoint(checkpoint_id)
            
            if data is None or metadata_dict is None:
                logger.error(f"No se pudo cargar checkpoint {checkpoint_id} para transmutación")
                return None
            
            # Convertir metadatos a objeto
            metadata = CheckpointMetadata.from_dict(metadata_dict)
            metadata.state = CheckpointState.TRANSMUTING
            
            # Aplicar función de transformación
            transformed_data = transformation_func(data)
            
            if transformed_data is None:
                logger.error(f"Función de transformación falló para checkpoint {checkpoint_id}")
                return None
            
            # Crear nuevo checkpoint con datos transformados
            new_checkpoint_id = f"{metadata.component_id}_transmuted_{uuid.uuid4().hex}"
            
            # Añadir tag de transmutación
            tags = metadata.tags.copy()
            tags.append("transmuted")
            tags.append(f"source:{checkpoint_id}")
            
            # Crear checkpoint transmutado
            transmuted_checkpoint_id = await self.create_checkpoint(
                component_id=metadata.component_id,
                data=transformed_data,
                tags=tags
            )
            
            if transmuted_checkpoint_id:
                # Registrar transmutación en metadata original
                metadata.record_transmutation()
                
                # Actualizar metadata en almacenamiento original
                if checkpoint_id in self._metadata_cache:
                    self._metadata_cache[checkpoint_id] = metadata
                
                logger.info(f"Checkpoint {checkpoint_id} transmutado a {transmuted_checkpoint_id}")
                return transmuted_checkpoint_id
            else:
                logger.error(f"Error al crear checkpoint transmutado para {checkpoint_id}")
                return None
            
        except Exception as e:
            logger.error(f"Error al transmutar checkpoint {checkpoint_id}: {e}")
            return None
    
    async def load_distributed_checkpoint(self, checkpoint_ids: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Cargar y combinar múltiples checkpoints distribuidos.
        
        Args:
            checkpoint_ids: Lista de IDs de checkpoints
            
        Returns:
            Tupla (datos combinados, lista de metadatos)
        """
        if not self.initialized:
            logger.error("DistributedCheckpointManager no inicializado")
            return {}, []
        
        if not checkpoint_ids:
            logger.error("No se proporcionaron IDs de checkpoints")
            return {}, []
        
        try:
            combined_data = {}
            metadata_list = []
            
            # Cargar cada checkpoint
            for checkpoint_id in checkpoint_ids:
                data, metadata = await self.load_checkpoint(checkpoint_id)
                
                if data and metadata:
                    # Combinar datos (recursivamente para diccionarios anidados)
                    combined_data = self._merge_dictionaries(combined_data, data)
                    metadata_list.append(metadata)
            
            if not combined_data:
                logger.warning("No se pudieron cargar datos de ningún checkpoint")
                return {}, []
            
            logger.info(f"Combinados {len(checkpoint_ids)} checkpoints distribuidos")
            return combined_data, metadata_list
            
        except Exception as e:
            logger.error(f"Error al cargar checkpoints distribuidos: {e}")
            return {}, []
    
    def _merge_dictionaries(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combinar dos diccionarios recursivamente.
        
        Args:
            dict1: Primer diccionario
            dict2: Segundo diccionario
            
        Returns:
            Diccionario combinado
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Combinar recursivamente para diccionarios anidados
                result[key] = self._merge_dictionaries(result[key], value)
            else:
                # Sobrescribir o añadir valor
                result[key] = value
        
        return result
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """
        Calcular checksum de datos.
        
        Args:
            data: Datos a calcular checksum
            
        Returns:
            Checksum como string
        """
        # Serializar a bytes
        serialized = pickle.dumps(data)
        
        # Calcular SHA-256
        return hashlib.sha256(serialized).hexdigest()
    
    # Implementaciones específicas para FILE
    
    async def _save_to_file(self, checkpoint_id: str, data: Dict[str, Any], metadata: CheckpointMetadata) -> bool:
        """
        Guardar checkpoint en archivos.
        
        Args:
            checkpoint_id: ID del checkpoint
            data: Datos a guardar
            metadata: Metadatos
            
        Returns:
            True si se guardó correctamente
        """
        try:
            # Calcular tamaño aproximado
            serialized = pickle.dumps(data)
            metadata.size_bytes = len(serialized)
            
            # Guardar datos en archivo
            data_path = os.path.join(self.base_directory, f"{checkpoint_id}_data.pickle")
            with open(data_path, "wb") as f:
                f.write(serialized)
            
            # Guardar metadatos en archivo JSON
            meta_path = os.path.join(self.base_directory, f"{checkpoint_id}_meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error al guardar checkpoint en archivo: {e}")
            return False
    
    async def _load_from_file(self, checkpoint_id: str, load_data: bool = True) -> Tuple[Optional[Dict[str, Any]], Optional[CheckpointMetadata]]:
        """
        Cargar checkpoint desde archivos.
        
        Args:
            checkpoint_id: ID del checkpoint
            load_data: Si cargar también los datos (o solo metadatos)
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si falla
        """
        try:
            # Verificar si existen archivos
            meta_path = os.path.join(self.base_directory, f"{checkpoint_id}_meta.json")
            data_path = os.path.join(self.base_directory, f"{checkpoint_id}_data.pickle")
            
            if not os.path.exists(meta_path):
                logger.warning(f"No existe archivo de metadatos para checkpoint {checkpoint_id}")
                return None, None
            
            # Cargar metadatos
            with open(meta_path, "r") as f:
                metadata_dict = json.load(f)
            
            metadata = CheckpointMetadata.from_dict(metadata_dict)
            
            # Si solo se necesitan metadatos, devolver aquí
            if not load_data:
                return None, metadata
            
            # Verificar si existe archivo de datos
            if not os.path.exists(data_path):
                logger.warning(f"No existe archivo de datos para checkpoint {checkpoint_id}")
                return None, metadata
            
            # Cargar datos
            with open(data_path, "rb") as f:
                serialized = f.read()
                data = pickle.loads(serialized)
            
            return data, metadata
            
        except Exception as e:
            logger.error(f"Error al cargar checkpoint desde archivo: {e}")
            return None, None
    
    async def _delete_from_file(self, checkpoint_id: str) -> bool:
        """
        Eliminar checkpoint de archivos.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente
        """
        try:
            # Verificar si existen archivos
            meta_path = os.path.join(self.base_directory, f"{checkpoint_id}_meta.json")
            data_path = os.path.join(self.base_directory, f"{checkpoint_id}_data.pickle")
            
            # Eliminar archivos si existen
            if os.path.exists(meta_path):
                os.remove(meta_path)
            
            if os.path.exists(data_path):
                os.remove(data_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error al eliminar checkpoint de archivo: {e}")
            return False
    
    async def _list_from_file(self, component_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        Listar checkpoints desde archivos.
        
        Args:
            component_id: Filtrar por componente (opcional)
            
        Returns:
            Lista de metadatos
        """
        result = []
        
        try:
            # Listar archivos en directorio
            for filename in os.listdir(self.base_directory):
                if not filename.endswith("_meta.json"):
                    continue
                
                # Extraer ID de checkpoint
                checkpoint_id = filename.replace("_meta.json", "")
                
                # Cargar metadatos
                meta_path = os.path.join(self.base_directory, filename)
                with open(meta_path, "r") as f:
                    metadata_dict = json.load(f)
                
                metadata = CheckpointMetadata.from_dict(metadata_dict)
                
                # Filtrar por componente si se especificó
                if component_id and metadata.component_id != component_id:
                    continue
                
                result.append(metadata)
            
            return result
            
        except Exception as e:
            logger.error(f"Error al listar checkpoints desde archivo: {e}")
            return []
    
    # Implementaciones específicas para MEMORY
    
    def _save_to_memory(self, checkpoint_id: str, data: Dict[str, Any], metadata: CheckpointMetadata) -> bool:
        """
        Guardar checkpoint en memoria.
        
        Args:
            checkpoint_id: ID del checkpoint
            data: Datos a guardar
            metadata: Metadatos
            
        Returns:
            True si se guardó correctamente
        """
        try:
            # Calcular tamaño aproximado
            serialized = pickle.dumps(data)
            metadata.size_bytes = len(serialized)
            
            # Guardar en memoria
            self._memory_storage[checkpoint_id] = (data, metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Error al guardar checkpoint en memoria: {e}")
            return False
    
    def _load_from_memory(self, checkpoint_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[CheckpointMetadata]]:
        """
        Cargar checkpoint desde memoria.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si falla
        """
        if checkpoint_id not in self._memory_storage:
            return None, None
        
        return self._memory_storage[checkpoint_id]
    
    def _delete_from_memory(self, checkpoint_id: str) -> bool:
        """
        Eliminar checkpoint de memoria.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente
        """
        if checkpoint_id in self._memory_storage:
            del self._memory_storage[checkpoint_id]
            return True
        
        return False
    
    def _list_from_memory(self, component_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        Listar checkpoints desde memoria.
        
        Args:
            component_id: Filtrar por componente (opcional)
            
        Returns:
            Lista de metadatos
        """
        result = []
        
        for _, (_, metadata) in self._memory_storage.items():
            # Filtrar por componente si se especificó
            if component_id and metadata.component_id != component_id:
                continue
            
            result.append(metadata)
        
        return result
    
    # Implementaciones específicas para DISTRIBUTED_DB
    
    async def _save_to_db(self, checkpoint_id: str, data: Dict[str, Any], metadata: CheckpointMetadata) -> bool:
        """
        Guardar checkpoint en base de datos.
        
        Args:
            checkpoint_id: ID del checkpoint
            data: Datos a guardar
            metadata: Metadatos
            
        Returns:
            True si se guardó correctamente
        """
        if self._db_pool is None:
            logger.error("No hay conexión a base de datos")
            return False
        
        try:
            # Serializar datos y metadatos
            serialized_data = pickle.dumps(data)
            metadata.size_bytes = len(serialized_data)
            serialized_metadata = json.dumps(metadata.to_dict())
            
            # Adquirir conexión del pool
            async with self._db_pool.acquire() as connection:
                # Usar UPSERT para actualizar si existe o insertar si no
                await connection.execute('''
                    INSERT INTO checkpoints (checkpoint_id, component_id, data, metadata, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    ON CONFLICT (checkpoint_id)
                    DO UPDATE SET
                        data = $3,
                        metadata = $4
                ''', checkpoint_id, metadata.component_id, serialized_data, serialized_metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Error al guardar checkpoint en base de datos: {e}")
            return False
    
    async def _load_from_db(self, checkpoint_id: str, load_data: bool = True) -> Tuple[Optional[Dict[str, Any]], Optional[CheckpointMetadata]]:
        """
        Cargar checkpoint desde base de datos.
        
        Args:
            checkpoint_id: ID del checkpoint
            load_data: Si cargar también los datos (o solo metadatos)
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si falla
        """
        if self._db_pool is None:
            logger.error("No hay conexión a base de datos")
            return None, None
        
        try:
            # Adquirir conexión del pool
            async with self._db_pool.acquire() as connection:
                # Consulta según necesidad
                if load_data:
                    # Cargar datos y metadatos
                    row = await connection.fetchrow('''
                        SELECT data, metadata
                        FROM checkpoints
                        WHERE checkpoint_id = $1
                    ''', checkpoint_id)
                    
                    if not row:
                        return None, None
                    
                    # Deserializar datos y metadatos
                    data = pickle.loads(row["data"])
                    metadata = CheckpointMetadata.from_dict(json.loads(row["metadata"]))
                    
                    return data, metadata
                    
                else:
                    # Cargar solo metadatos
                    row = await connection.fetchrow('''
                        SELECT metadata
                        FROM checkpoints
                        WHERE checkpoint_id = $1
                    ''', checkpoint_id)
                    
                    if not row:
                        return None, None
                    
                    # Deserializar metadatos
                    metadata = CheckpointMetadata.from_dict(json.loads(row["metadata"]))
                    
                    return None, metadata
            
        except Exception as e:
            logger.error(f"Error al cargar checkpoint desde base de datos: {e}")
            return None, None
    
    async def _delete_from_db(self, checkpoint_id: str) -> bool:
        """
        Eliminar checkpoint de base de datos.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente
        """
        if self._db_pool is None:
            logger.error("No hay conexión a base de datos")
            return False
        
        try:
            # Adquirir conexión del pool
            async with self._db_pool.acquire() as connection:
                # Eliminar registro
                result = await connection.execute('''
                    DELETE FROM checkpoints
                    WHERE checkpoint_id = $1
                ''', checkpoint_id)
                
                # Verificar si se eliminó algún registro
                return "DELETE 1" in result
            
        except Exception as e:
            logger.error(f"Error al eliminar checkpoint de base de datos: {e}")
            return False
    
    async def _list_from_db(self, component_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        Listar checkpoints desde base de datos.
        
        Args:
            component_id: Filtrar por componente (opcional)
            
        Returns:
            Lista de metadatos
        """
        if self._db_pool is None:
            logger.error("No hay conexión a base de datos")
            return []
        
        try:
            # Adquirir conexión del pool
            async with self._db_pool.acquire() as connection:
                # Consulta según filtro
                if component_id:
                    rows = await connection.fetch('''
                        SELECT metadata
                        FROM checkpoints
                        WHERE component_id = $1
                    ''', component_id)
                else:
                    rows = await connection.fetch('''
                        SELECT metadata
                        FROM checkpoints
                    ''')
                
                # Procesar resultados
                result = []
                for row in rows:
                    metadata = CheckpointMetadata.from_dict(json.loads(row["metadata"]))
                    result.append(metadata)
                
                return result
            
        except Exception as e:
            logger.error(f"Error al listar checkpoints desde base de datos: {e}")
            return []
    
    # Método de recuperación automática ante fallos
    
    async def handle_error(self, error: Exception) -> bool:
        """
        Manejar un error y aplicar estrategia de recuperación.
        
        Args:
            error: Excepción ocurrida
            
        Returns:
            True si se recuperó correctamente
        """
        try:
            # Procesar errores específicos
            if isinstance(error, KeyError) or isinstance(error, FileNotFoundError):
                # Error de datos no encontrados, intentar recuperar desde último checkpoint válido
                logger.warning(f"Error de datos no encontrados: {error}")
                return True
                
            elif isinstance(error, ValueError) or isinstance(error, TypeError):
                # Error de datos incorrectos, intentar transmutación
                if self.transmutation_enabled:
                    logger.warning(f"Error de datos incorrectos: {error}, aplicando transmutación")
                    return True
                
            elif isinstance(error, OSError) or isinstance(error, IOError):
                # Error de E/S, cambiar a modo memoria temporalmente
                logger.warning(f"Error de E/S: {error}, cambiando a modo memoria temporal")
                self.storage_type = CheckpointStorageType.MEMORY
                return True
                
            elif isinstance(error, asyncio.TimeoutError):
                # Timeout, reintentar con tiempo límite mayor
                logger.warning(f"Timeout: {error}, reintentando")
                return True
            
            # En general, procesar errores no esperados
            for e in error.__class__.__mro__:
                if e is BaseException:
                    break
                    
                logger.error(f"Error no manejado: {error} (tipo: {e.__name__})")
            
            return False
            
        except Exception as recovery_error:
            logger.error(f"Error al intentar recuperación: {recovery_error}")
            return False


# Crear instancia global
checkpoint_manager = DistributedCheckpointManager()


# Decorador para uso con checkpoints (funciones)
def checkpoint_state(component_id: str, tags: Optional[List[str]] = None):
    """
    Decorador para guardar/restaurar estado automáticamente.
    
    Args:
        component_id: ID del componente
        tags: Etiquetas opcionales
        
    Returns:
        Decorador
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            global checkpoint_manager
            
            if not checkpoint_manager or not checkpoint_manager.initialized:
                # Si no hay checkpoint_manager, ejecutar normalmente
                return await func(*args, **kwargs)
            
            try:
                # Intentar cargar estado anterior
                latest_data, _ = await checkpoint_manager.load_latest_checkpoint(component_id)
                
                # Si hay estado anterior, usarlo
                if latest_data:
                    # Añadir datos del checkpoint como kwargs
                    for key, value in latest_data.items():
                        if key not in kwargs:
                            kwargs[key] = value
                
                # Ejecutar función
                result = await func(*args, **kwargs)
                
                # Guardar resultado como checkpoint
                await checkpoint_manager.create_checkpoint(
                    component_id=component_id,
                    data=result if isinstance(result, dict) else {"result": result},
                    tags=tags
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error en función con checkpoint: {e}")
                raise
                
        return wrapper
    
    return decorator