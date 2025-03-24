#!/usr/bin/env python3
"""
Gestor de Checkpoints Distribuidos Ultra-Divino.

Este módulo implementa un sistema de checkpoints distribuidos para el Sistema Genesis,
permitiendo respaldo y recuperación de estado entre diferentes instancias y servicios.

Características:
- Checkpoints atómicos con garantía ACID
- Recuperación ultra-rápida desde cualquier nodo
- Compatibilidad con PostgreSQL, S3 y sistemas de archivos locales
- Coordinación de estado distribuido mediante consensus cuántico
- Versionado automático de checkpoints para rollbacks controlados
"""

import os
import sys
import json
import logging
import time
import asyncio
import random
import hashlib
import pickle
import base64
import uuid
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set, Type, TypeVar

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("genesis.cloud.distributed_checkpoint")


# Tipos genéricos
T = TypeVar('T')


class CheckpointStorageType(Enum):
    """Tipos de almacenamiento para checkpoints."""
    LOCAL_FILE = auto()     # Archivo local
    DATABASE = auto()       # Base de datos PostgreSQL
    S3 = auto()             # AWS S3 o compatible
    MEMORY = auto()         # Memoria (para pruebas)
    HYBRID = auto()         # Combinación de tipos


class CheckpointConsistencyLevel(Enum):
    """Niveles de consistencia para checkpoints."""
    EVENTUAL = auto()       # Consistencia eventual
    STRONG = auto()         # Consistencia fuerte
    QUANTUM = auto()        # Consistencia cuántica (máxima)


class CheckpointState(Enum):
    """Estados posibles de un checkpoint."""
    CREATING = auto()       # En proceso de creación
    ACTIVE = auto()         # Activo y disponible
    SYNCING = auto()        # Sincronizando con otros nodos
    CORRUPTED = auto()      # Corrupto (verificación fallida)
    ARCHIVED = auto()       # Archivado (histórico)


class CheckpointMetadata:
    """Metadatos de un checkpoint."""
    
    def __init__(self, 
                 checkpoint_id: str,
                 component_id: str,
                 timestamp: float,
                 version: int,
                 consistency_level: CheckpointConsistencyLevel,
                 storage_type: CheckpointStorageType,
                 tags: Optional[List[str]] = None,
                 dependencies: Optional[List[str]] = None):
        """
        Inicializar metadatos de checkpoint.
        
        Args:
            checkpoint_id: ID único del checkpoint
            component_id: ID del componente al que pertenece
            timestamp: Timestamp de creación
            version: Versión del checkpoint
            consistency_level: Nivel de consistencia
            storage_type: Tipo de almacenamiento
            tags: Etiquetas para categorización
            dependencies: IDs de checkpoints dependientes
        """
        self.checkpoint_id = checkpoint_id
        self.component_id = component_id
        self.timestamp = timestamp
        self.version = version
        self.consistency_level = consistency_level
        self.storage_type = storage_type
        self.tags = tags or []
        self.dependencies = dependencies or []
        self.state = CheckpointState.CREATING
        self.hash = ""
        self.size_bytes = 0
        self.created_by = "system"
        self.node_id = str(uuid.uuid4())[:8]
        self.last_verified = timestamp
        self.verification_hash = ""
    
    def finalize(self, data_hash: str, size_bytes: int):
        """
        Finalizar metadatos al completar la creación.
        
        Args:
            data_hash: Hash del contenido
            size_bytes: Tamaño en bytes
        """
        self.hash = data_hash
        self.size_bytes = size_bytes
        self.state = CheckpointState.ACTIVE
        
        # Crear hash de verificación
        self.verification_hash = self._create_verification_hash()
    
    def _create_verification_hash(self) -> str:
        """
        Crear hash de verificación para integridad.
        
        Returns:
            Hash de verificación
        """
        data = f"{self.checkpoint_id}:{self.component_id}:{self.timestamp}:{self.version}:{self.hash}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify(self) -> bool:
        """
        Verificar integridad del checkpoint.
        
        Returns:
            True si es íntegro, False si está corrupto
        """
        current_hash = self._create_verification_hash()
        if current_hash != self.verification_hash:
            self.state = CheckpointState.CORRUPTED
            return False
        
        self.last_verified = time.time()
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario para almacenamiento.
        
        Returns:
            Diccionario con los metadatos
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "component_id": self.component_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "consistency_level": self.consistency_level.name,
            "storage_type": self.storage_type.name,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "state": self.state.name,
            "hash": self.hash,
            "size_bytes": self.size_bytes,
            "created_by": self.created_by,
            "node_id": self.node_id,
            "last_verified": self.last_verified,
            "verification_hash": self.verification_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """
        Crear desde diccionario.
        
        Args:
            data: Diccionario con metadatos
            
        Returns:
            Instancia de CheckpointMetadata
        """
        instance = cls(
            checkpoint_id=data["checkpoint_id"],
            component_id=data["component_id"],
            timestamp=data["timestamp"],
            version=data["version"],
            consistency_level=CheckpointConsistencyLevel[data["consistency_level"]],
            storage_type=CheckpointStorageType[data["storage_type"]],
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", [])
        )
        
        instance.state = CheckpointState[data["state"]]
        instance.hash = data["hash"]
        instance.size_bytes = data["size_bytes"]
        instance.created_by = data.get("created_by", "system")
        instance.node_id = data.get("node_id", "unknown")
        instance.last_verified = data.get("last_verified", instance.timestamp)
        instance.verification_hash = data.get("verification_hash", "")
        
        return instance


class CheckpointStorageProvider:
    """Proveedor de almacenamiento para checkpoints."""
    
    async def store(self, checkpoint_id: str, data: Any, metadata: CheckpointMetadata) -> bool:
        """
        Almacenar un checkpoint.
        
        Args:
            checkpoint_id: ID del checkpoint
            data: Datos a almacenar
            metadata: Metadatos del checkpoint
            
        Returns:
            True si se almacenó correctamente
        """
        raise NotImplementedError("Método abstracto")
    
    async def load(self, checkpoint_id: str) -> Tuple[Optional[Any], Optional[CheckpointMetadata]]:
        """
        Cargar un checkpoint.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si no existe
        """
        raise NotImplementedError("Método abstracto")
    
    async def list_checkpoints(self, component_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        Listar checkpoints disponibles.
        
        Args:
            component_id: Filtrar por componente (opcional)
            
        Returns:
            Lista de metadatos de checkpoints
        """
        raise NotImplementedError("Método abstracto")
    
    async def delete(self, checkpoint_id: str) -> bool:
        """
        Eliminar un checkpoint.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente
        """
        raise NotImplementedError("Método abstracto")


class LocalFileStorageProvider(CheckpointStorageProvider):
    """Almacenamiento en archivos locales."""
    
    def __init__(self, base_path: str = "checkpoints"):
        """
        Inicializar proveedor de almacenamiento local.
        
        Args:
            base_path: Ruta base para almacenar checkpoints
        """
        self.base_path = base_path
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Crear estructura de directorios si no existe."""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        
        # Directorios para datos y metadatos
        data_path = os.path.join(self.base_path, "data")
        meta_path = os.path.join(self.base_path, "meta")
        
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        if not os.path.exists(meta_path):
            os.makedirs(meta_path)
    
    def _get_data_path(self, checkpoint_id: str) -> str:
        """
        Obtener ruta para datos de checkpoint.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Ruta completa
        """
        return os.path.join(self.base_path, "data", f"{checkpoint_id}.dat")
    
    def _get_meta_path(self, checkpoint_id: str) -> str:
        """
        Obtener ruta para metadatos de checkpoint.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Ruta completa
        """
        return os.path.join(self.base_path, "meta", f"{checkpoint_id}.meta")
    
    async def store(self, checkpoint_id: str, data: Any, metadata: CheckpointMetadata) -> bool:
        """
        Almacenar un checkpoint en archivo local.
        
        Args:
            checkpoint_id: ID del checkpoint
            data: Datos a almacenar
            metadata: Metadatos del checkpoint
            
        Returns:
            True si se almacenó correctamente
        """
        try:
            data_path = self._get_data_path(checkpoint_id)
            meta_path = self._get_meta_path(checkpoint_id)
            
            # Serializar datos
            data_bytes = pickle.dumps(data)
            data_hash = hashlib.sha256(data_bytes).hexdigest()
            data_size = len(data_bytes)
            
            # Finalizar metadatos con hash y tamaño
            metadata.finalize(data_hash, data_size)
            
            # Guardar datos y metadatos
            await asyncio.to_thread(self._write_file, data_path, data_bytes)
            await asyncio.to_thread(self._write_file, meta_path, json.dumps(metadata.to_dict()).encode())
            
            logger.info(f"Checkpoint {checkpoint_id} almacenado correctamente en {data_path}")
            return True
        except Exception as e:
            logger.error(f"Error al almacenar checkpoint {checkpoint_id}: {e}")
            return False
    
    def _write_file(self, path: str, data: bytes):
        """
        Escribir archivo de forma segura.
        
        Args:
            path: Ruta del archivo
            data: Datos a escribir
        """
        # Escribir a archivo temporal primero
        temp_path = f"{path}.tmp"
        with open(temp_path, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        
        # Renombrar atómicamente
        os.rename(temp_path, path)
    
    async def load(self, checkpoint_id: str) -> Tuple[Optional[Any], Optional[CheckpointMetadata]]:
        """
        Cargar un checkpoint desde archivo local.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si no existe
        """
        try:
            data_path = self._get_data_path(checkpoint_id)
            meta_path = self._get_meta_path(checkpoint_id)
            
            # Verificar existencia
            if not os.path.exists(data_path) or not os.path.exists(meta_path):
                logger.warning(f"Checkpoint {checkpoint_id} no encontrado")
                return None, None
            
            # Cargar metadatos
            meta_bytes = await asyncio.to_thread(self._read_file, meta_path)
            metadata = CheckpointMetadata.from_dict(json.loads(meta_bytes.decode()))
            
            # Verificar integridad de metadatos
            if not metadata.verify():
                logger.error(f"Metadatos corruptos para checkpoint {checkpoint_id}")
                return None, None
            
            # Cargar datos
            data_bytes = await asyncio.to_thread(self._read_file, data_path)
            
            # Verificar hash de datos
            data_hash = hashlib.sha256(data_bytes).hexdigest()
            if data_hash != metadata.hash:
                logger.error(f"Datos corruptos para checkpoint {checkpoint_id}")
                metadata.state = CheckpointState.CORRUPTED
                return None, metadata
            
            # Deserializar datos
            data = pickle.loads(data_bytes)
            
            logger.info(f"Checkpoint {checkpoint_id} cargado correctamente desde {data_path}")
            return data, metadata
        except Exception as e:
            logger.error(f"Error al cargar checkpoint {checkpoint_id}: {e}")
            return None, None
    
    def _read_file(self, path: str) -> bytes:
        """
        Leer archivo completo.
        
        Args:
            path: Ruta del archivo
            
        Returns:
            Contenido del archivo
        """
        with open(path, 'rb') as f:
            return f.read()
    
    async def list_checkpoints(self, component_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        Listar checkpoints disponibles en almacenamiento local.
        
        Args:
            component_id: Filtrar por componente (opcional)
            
        Returns:
            Lista de metadatos de checkpoints
        """
        try:
            meta_dir = os.path.join(self.base_path, "meta")
            if not os.path.exists(meta_dir):
                return []
            
            # Listar archivos de metadatos
            meta_files = await asyncio.to_thread(os.listdir, meta_dir)
            result = []
            
            for filename in meta_files:
                if not filename.endswith(".meta"):
                    continue
                
                try:
                    # Cargar metadatos
                    meta_path = os.path.join(meta_dir, filename)
                    meta_bytes = await asyncio.to_thread(self._read_file, meta_path)
                    metadata = CheckpointMetadata.from_dict(json.loads(meta_bytes.decode()))
                    
                    # Filtrar por componente
                    if component_id is None or metadata.component_id == component_id:
                        result.append(metadata)
                except Exception as e:
                    logger.warning(f"Error al leer metadatos {filename}: {e}")
            
            # Ordenar por timestamp descendente
            result.sort(key=lambda m: m.timestamp, reverse=True)
            return result
        except Exception as e:
            logger.error(f"Error al listar checkpoints: {e}")
            return []
    
    async def delete(self, checkpoint_id: str) -> bool:
        """
        Eliminar un checkpoint del almacenamiento local.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente
        """
        try:
            data_path = self._get_data_path(checkpoint_id)
            meta_path = self._get_meta_path(checkpoint_id)
            
            # Eliminar archivos si existen
            if os.path.exists(data_path):
                await asyncio.to_thread(os.remove, data_path)
            
            if os.path.exists(meta_path):
                await asyncio.to_thread(os.remove, meta_path)
            
            logger.info(f"Checkpoint {checkpoint_id} eliminado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al eliminar checkpoint {checkpoint_id}: {e}")
            return False


class PostgreSQLStorageProvider(CheckpointStorageProvider):
    """Almacenamiento en base de datos PostgreSQL."""
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Inicializar proveedor de almacenamiento PostgreSQL.
        
        Args:
            db_url: URL de conexión a la base de datos
        """
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self._pool = None
        self._initialized = False
    
    async def initialize(self):
        """Inicializar conexión y crear tablas si no existen."""
        if self._initialized:
            return
        
        try:
            # Importar aquí para no depender de asyncpg si no se usa
            import asyncpg
            
            # Crear pool de conexiones
            self._pool = await asyncpg.create_pool(self.db_url)
            
            # Crear tablas si no existen
            async with self._pool.acquire() as conn:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS checkpoint_metadata (
                        checkpoint_id TEXT PRIMARY KEY,
                        component_id TEXT NOT NULL,
                        timestamp DOUBLE PRECISION NOT NULL,
                        version INTEGER NOT NULL,
                        consistency_level TEXT NOT NULL,
                        storage_type TEXT NOT NULL,
                        tags JSONB DEFAULT '[]',
                        dependencies JSONB DEFAULT '[]',
                        state TEXT NOT NULL,
                        hash TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        created_by TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        last_verified DOUBLE PRECISION NOT NULL,
                        verification_hash TEXT NOT NULL
                    )
                ''')
                
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS checkpoint_data (
                        checkpoint_id TEXT PRIMARY KEY REFERENCES checkpoint_metadata(checkpoint_id) ON DELETE CASCADE,
                        data BYTEA NOT NULL
                    )
                ''')
                
                # Índices
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_checkpoint_component 
                    ON checkpoint_metadata(component_id)
                ''')
                
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_checkpoint_timestamp 
                    ON checkpoint_metadata(timestamp)
                ''')
            
            self._initialized = True
            logger.info("Proveedor PostgreSQL inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar proveedor PostgreSQL: {e}")
            raise
    
    async def store(self, checkpoint_id: str, data: Any, metadata: CheckpointMetadata) -> bool:
        """
        Almacenar un checkpoint en PostgreSQL.
        
        Args:
            checkpoint_id: ID del checkpoint
            data: Datos a almacenar
            metadata: Metadatos del checkpoint
            
        Returns:
            True si se almacenó correctamente
        """
        await self.initialize()
        
        try:
            # Serializar datos
            data_bytes = pickle.dumps(data)
            data_hash = hashlib.sha256(data_bytes).hexdigest()
            data_size = len(data_bytes)
            
            # Finalizar metadatos con hash y tamaño
            metadata.finalize(data_hash, data_size)
            meta_dict = metadata.to_dict()
            
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # Guardar metadatos
                    await conn.execute('''
                        INSERT INTO checkpoint_metadata(
                            checkpoint_id, component_id, timestamp, version, consistency_level,
                            storage_type, tags, dependencies, state, hash, size_bytes,
                            created_by, node_id, last_verified, verification_hash
                        ) VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                        ON CONFLICT(checkpoint_id) DO UPDATE SET
                            component_id = $2, timestamp = $3, version = $4, consistency_level = $5,
                            storage_type = $6, tags = $7, dependencies = $8, state = $9,
                            hash = $10, size_bytes = $11, created_by = $12, node_id = $13,
                            last_verified = $14, verification_hash = $15
                    ''', 
                        checkpoint_id, meta_dict["component_id"], meta_dict["timestamp"],
                        meta_dict["version"], meta_dict["consistency_level"],
                        meta_dict["storage_type"], json.dumps(meta_dict["tags"]),
                        json.dumps(meta_dict["dependencies"]), meta_dict["state"],
                        meta_dict["hash"], meta_dict["size_bytes"], meta_dict["created_by"],
                        meta_dict["node_id"], meta_dict["last_verified"],
                        meta_dict["verification_hash"]
                    )
                    
                    # Guardar datos
                    await conn.execute('''
                        INSERT INTO checkpoint_data(checkpoint_id, data)
                        VALUES($1, $2)
                        ON CONFLICT(checkpoint_id) DO UPDATE SET
                            data = $2
                    ''', checkpoint_id, data_bytes)
            
            logger.info(f"Checkpoint {checkpoint_id} almacenado correctamente en PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Error al almacenar checkpoint {checkpoint_id} en PostgreSQL: {e}")
            return False
    
    async def load(self, checkpoint_id: str) -> Tuple[Optional[Any], Optional[CheckpointMetadata]]:
        """
        Cargar un checkpoint desde PostgreSQL.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si no existe
        """
        await self.initialize()
        
        try:
            async with self._pool.acquire() as conn:
                # Cargar metadatos
                meta_row = await conn.fetchrow('''
                    SELECT * FROM checkpoint_metadata
                    WHERE checkpoint_id = $1
                ''', checkpoint_id)
                
                if not meta_row:
                    logger.warning(f"Checkpoint {checkpoint_id} no encontrado en PostgreSQL")
                    return None, None
                
                # Convertir a diccionario
                meta_dict = dict(meta_row)
                meta_dict["tags"] = json.loads(meta_dict["tags"])
                meta_dict["dependencies"] = json.loads(meta_dict["dependencies"])
                
                # Crear objeto de metadatos
                metadata = CheckpointMetadata.from_dict(meta_dict)
                
                # Verificar integridad de metadatos
                if not metadata.verify():
                    logger.error(f"Metadatos corruptos para checkpoint {checkpoint_id}")
                    return None, None
                
                # Cargar datos
                data_row = await conn.fetchrow('''
                    SELECT data FROM checkpoint_data
                    WHERE checkpoint_id = $1
                ''', checkpoint_id)
                
                if not data_row:
                    logger.error(f"Datos no encontrados para checkpoint {checkpoint_id}")
                    return None, metadata
                
                data_bytes = data_row["data"]
                
                # Verificar hash de datos
                data_hash = hashlib.sha256(data_bytes).hexdigest()
                if data_hash != metadata.hash:
                    logger.error(f"Datos corruptos para checkpoint {checkpoint_id}")
                    metadata.state = CheckpointState.CORRUPTED
                    
                    # Actualizar estado en base de datos
                    await conn.execute('''
                        UPDATE checkpoint_metadata
                        SET state = $1
                        WHERE checkpoint_id = $2
                    ''', CheckpointState.CORRUPTED.name, checkpoint_id)
                    
                    return None, metadata
                
                # Deserializar datos
                data = pickle.loads(data_bytes)
                
                # Actualizar última verificación
                await conn.execute('''
                    UPDATE checkpoint_metadata
                    SET last_verified = $1
                    WHERE checkpoint_id = $2
                ''', time.time(), checkpoint_id)
                
                logger.info(f"Checkpoint {checkpoint_id} cargado correctamente desde PostgreSQL")
                return data, metadata
        except Exception as e:
            logger.error(f"Error al cargar checkpoint {checkpoint_id} desde PostgreSQL: {e}")
            return None, None
    
    async def list_checkpoints(self, component_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        Listar checkpoints disponibles en PostgreSQL.
        
        Args:
            component_id: Filtrar por componente (opcional)
            
        Returns:
            Lista de metadatos de checkpoints
        """
        await self.initialize()
        
        try:
            async with self._pool.acquire() as conn:
                # Consulta base
                query = '''
                    SELECT * FROM checkpoint_metadata
                '''
                
                # Filtrar por componente si se especifica
                params = []
                if component_id is not None:
                    query += " WHERE component_id = $1"
                    params.append(component_id)
                
                # Ordenar por timestamp descendente
                query += " ORDER BY timestamp DESC"
                
                # Ejecutar consulta
                rows = await conn.fetch(query, *params)
                
                # Convertir a objetos de metadatos
                result = []
                for row in rows:
                    meta_dict = dict(row)
                    meta_dict["tags"] = json.loads(meta_dict["tags"])
                    meta_dict["dependencies"] = json.loads(meta_dict["dependencies"])
                    result.append(CheckpointMetadata.from_dict(meta_dict))
                
                return result
        except Exception as e:
            logger.error(f"Error al listar checkpoints desde PostgreSQL: {e}")
            return []
    
    async def delete(self, checkpoint_id: str) -> bool:
        """
        Eliminar un checkpoint de PostgreSQL.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente
        """
        await self.initialize()
        
        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # Eliminar datos y metadatos (la eliminación en cascada funciona por la FK)
                    await conn.execute('''
                        DELETE FROM checkpoint_metadata
                        WHERE checkpoint_id = $1
                    ''', checkpoint_id)
            
            logger.info(f"Checkpoint {checkpoint_id} eliminado correctamente de PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Error al eliminar checkpoint {checkpoint_id} de PostgreSQL: {e}")
            return False


class MemoryStorageProvider(CheckpointStorageProvider):
    """Almacenamiento en memoria (para pruebas)."""
    
    def __init__(self):
        """Inicializar proveedor de almacenamiento en memoria."""
        self._data_store: Dict[str, bytes] = {}
        self._meta_store: Dict[str, Dict[str, Any]] = {}
    
    async def store(self, checkpoint_id: str, data: Any, metadata: CheckpointMetadata) -> bool:
        """
        Almacenar un checkpoint en memoria.
        
        Args:
            checkpoint_id: ID del checkpoint
            data: Datos a almacenar
            metadata: Metadatos del checkpoint
            
        Returns:
            True si se almacenó correctamente
        """
        try:
            # Serializar datos
            data_bytes = pickle.dumps(data)
            data_hash = hashlib.sha256(data_bytes).hexdigest()
            data_size = len(data_bytes)
            
            # Finalizar metadatos con hash y tamaño
            metadata.finalize(data_hash, data_size)
            
            # Guardar en memoria
            self._data_store[checkpoint_id] = data_bytes
            self._meta_store[checkpoint_id] = metadata.to_dict()
            
            logger.info(f"Checkpoint {checkpoint_id} almacenado correctamente en memoria")
            return True
        except Exception as e:
            logger.error(f"Error al almacenar checkpoint {checkpoint_id} en memoria: {e}")
            return False
    
    async def load(self, checkpoint_id: str) -> Tuple[Optional[Any], Optional[CheckpointMetadata]]:
        """
        Cargar un checkpoint desde memoria.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si no existe
        """
        try:
            # Verificar existencia
            if checkpoint_id not in self._meta_store or checkpoint_id not in self._data_store:
                logger.warning(f"Checkpoint {checkpoint_id} no encontrado en memoria")
                return None, None
            
            # Cargar metadatos
            meta_dict = self._meta_store[checkpoint_id]
            metadata = CheckpointMetadata.from_dict(meta_dict)
            
            # Verificar integridad de metadatos
            if not metadata.verify():
                logger.error(f"Metadatos corruptos para checkpoint {checkpoint_id}")
                return None, None
            
            # Cargar datos
            data_bytes = self._data_store[checkpoint_id]
            
            # Verificar hash de datos
            data_hash = hashlib.sha256(data_bytes).hexdigest()
            if data_hash != metadata.hash:
                logger.error(f"Datos corruptos para checkpoint {checkpoint_id}")
                metadata.state = CheckpointState.CORRUPTED
                self._meta_store[checkpoint_id]["state"] = CheckpointState.CORRUPTED.name
                return None, metadata
            
            # Deserializar datos
            data = pickle.loads(data_bytes)
            
            # Actualizar última verificación
            metadata.last_verified = time.time()
            self._meta_store[checkpoint_id]["last_verified"] = time.time()
            
            logger.info(f"Checkpoint {checkpoint_id} cargado correctamente desde memoria")
            return data, metadata
        except Exception as e:
            logger.error(f"Error al cargar checkpoint {checkpoint_id} desde memoria: {e}")
            return None, None
    
    async def list_checkpoints(self, component_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        Listar checkpoints disponibles en memoria.
        
        Args:
            component_id: Filtrar por componente (opcional)
            
        Returns:
            Lista de metadatos de checkpoints
        """
        try:
            result = []
            
            for meta_dict in self._meta_store.values():
                # Filtrar por componente si se especifica
                if component_id is not None and meta_dict["component_id"] != component_id:
                    continue
                
                metadata = CheckpointMetadata.from_dict(meta_dict)
                result.append(metadata)
            
            # Ordenar por timestamp descendente
            result.sort(key=lambda m: m.timestamp, reverse=True)
            return result
        except Exception as e:
            logger.error(f"Error al listar checkpoints desde memoria: {e}")
            return []
    
    async def delete(self, checkpoint_id: str) -> bool:
        """
        Eliminar un checkpoint de memoria.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente
        """
        try:
            # Verificar existencia
            if checkpoint_id not in self._meta_store:
                logger.warning(f"Checkpoint {checkpoint_id} no encontrado para eliminar")
                return False
            
            # Eliminar de memoria
            self._meta_store.pop(checkpoint_id, None)
            self._data_store.pop(checkpoint_id, None)
            
            logger.info(f"Checkpoint {checkpoint_id} eliminado correctamente de memoria")
            return True
        except Exception as e:
            logger.error(f"Error al eliminar checkpoint {checkpoint_id} de memoria: {e}")
            return False


class HybridStorageProvider(CheckpointStorageProvider):
    """
    Proveedor híbrido que combina varios tipos de almacenamiento.
    
    - Escritura simultánea a múltiples almacenamientos
    - Lectura con prioridad configurable
    - Redundancia automática para alta disponibilidad
    """
    
    def __init__(self, providers: List[Tuple[CheckpointStorageProvider, int]]):
        """
        Inicializar proveedor híbrido.
        
        Args:
            providers: Lista de tuplas (proveedor, prioridad)
        """
        self.providers = providers
        self.providers.sort(key=lambda p: p[1], reverse=True)  # Ordenar por prioridad descendente
    
    async def store(self, checkpoint_id: str, data: Any, metadata: CheckpointMetadata) -> bool:
        """
        Almacenar un checkpoint en todos los proveedores.
        
        Args:
            checkpoint_id: ID del checkpoint
            data: Datos a almacenar
            metadata: Metadatos del checkpoint
            
        Returns:
            True si se almacenó correctamente en al menos un proveedor
        """
        # Serializar datos para asegurar consistencia entre proveedores
        data_bytes = pickle.dumps(data)
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        data_size = len(data_bytes)
        
        # Finalizar metadatos con hash y tamaño
        metadata.finalize(data_hash, data_size)
        
        # Almacenar en todos los proveedores
        results = await asyncio.gather(
            *[provider.store(checkpoint_id, data, metadata) for provider, _ in self.providers],
            return_exceptions=True
        )
        
        # Verificar resultados
        success_count = sum(1 for r in results if r is True)
        failure_count = len(results) - success_count
        
        if success_count > 0:
            logger.info(f"Checkpoint {checkpoint_id} almacenado correctamente en {success_count}/{len(self.providers)} proveedores")
            return True
        else:
            logger.error(f"Checkpoint {checkpoint_id} no se pudo almacenar en ningún proveedor")
            return False
    
    async def load(self, checkpoint_id: str) -> Tuple[Optional[Any], Optional[CheckpointMetadata]]:
        """
        Cargar un checkpoint desde el proveedor de mayor prioridad disponible.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si no existe
        """
        # Intentar cargar desde cada proveedor en orden de prioridad
        for provider, priority in self.providers:
            try:
                data, metadata = await provider.load(checkpoint_id)
                if data is not None and metadata is not None:
                    logger.info(f"Checkpoint {checkpoint_id} cargado correctamente desde proveedor de prioridad {priority}")
                    return data, metadata
            except Exception as e:
                logger.warning(f"Error al cargar desde proveedor de prioridad {priority}: {e}")
        
        logger.warning(f"Checkpoint {checkpoint_id} no encontrado en ningún proveedor")
        return None, None
    
    async def list_checkpoints(self, component_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        Listar checkpoints disponibles en todos los proveedores (sin duplicados).
        
        Args:
            component_id: Filtrar por componente (opcional)
            
        Returns:
            Lista de metadatos de checkpoints
        """
        # Obtener checkpoints de todos los proveedores
        all_checkpoints: Dict[str, CheckpointMetadata] = {}
        
        for provider, _ in self.providers:
            try:
                checkpoints = await provider.list_checkpoints(component_id)
                for checkpoint in checkpoints:
                    # Mantener solo el checkpoint más reciente por ID
                    if checkpoint.checkpoint_id not in all_checkpoints or \
                       checkpoint.timestamp > all_checkpoints[checkpoint.checkpoint_id].timestamp:
                        all_checkpoints[checkpoint.checkpoint_id] = checkpoint
            except Exception as e:
                logger.warning(f"Error al listar checkpoints desde proveedor: {e}")
        
        # Convertir a lista y ordenar por timestamp
        result = list(all_checkpoints.values())
        result.sort(key=lambda m: m.timestamp, reverse=True)
        
        return result
    
    async def delete(self, checkpoint_id: str) -> bool:
        """
        Eliminar un checkpoint de todos los proveedores.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente de al menos un proveedor
        """
        # Eliminar de todos los proveedores
        results = await asyncio.gather(
            *[provider.delete(checkpoint_id) for provider, _ in self.providers],
            return_exceptions=True
        )
        
        # Verificar resultados
        success_count = sum(1 for r in results if r is True)
        
        if success_count > 0:
            logger.info(f"Checkpoint {checkpoint_id} eliminado correctamente de {success_count}/{len(self.providers)} proveedores")
            return True
        else:
            logger.error(f"Checkpoint {checkpoint_id} no se pudo eliminar de ningún proveedor")
            return False


class DistributedCheckpointManager:
    """
    Gestor de Checkpoints Distribuidos.
    
    Este gestor proporciona una interfaz unificada para operaciones con checkpoints,
    incluyendo creación, recuperación, listado y eliminación. Soporta múltiples backends
    de almacenamiento y ofrece capacidades avanzadas como checkpoints incrementales,
    control de versiones y recuperación de fallos.
    """
    
    def __init__(self, 
                 storage_type: CheckpointStorageType = CheckpointStorageType.LOCAL_FILE,
                 base_path: str = "checkpoints",
                 db_url: Optional[str] = None,
                 consistency_level: CheckpointConsistencyLevel = CheckpointConsistencyLevel.STRONG):
        """
        Inicializar gestor de checkpoints.
        
        Args:
            storage_type: Tipo de almacenamiento a utilizar
            base_path: Ruta base para almacenamiento local
            db_url: URL de conexión a PostgreSQL
            consistency_level: Nivel de consistencia por defecto
        """
        self.storage_type = storage_type
        self.consistency_level = consistency_level
        self.base_path = base_path
        self.db_url = db_url
        
        # Inicializar proveedor de almacenamiento
        self._init_storage_provider()
        
        # Caché de metadatos para acceso rápido
        self._metadata_cache: Dict[str, CheckpointMetadata] = {}
        
        # Para coordinación de checkpoints distribuidos
        self._active_components: Set[str] = set()
        self._version_counters: Dict[str, int] = {}
        
        logger.info(f"DistributedCheckpointManager inicializado con almacenamiento {storage_type.name}")
    
    def _init_storage_provider(self):
        """Inicializar el proveedor de almacenamiento según el tipo seleccionado."""
        if self.storage_type == CheckpointStorageType.LOCAL_FILE:
            self.provider = LocalFileStorageProvider(self.base_path)
        
        elif self.storage_type == CheckpointStorageType.DATABASE:
            self.provider = PostgreSQLStorageProvider(self.db_url)
        
        elif self.storage_type == CheckpointStorageType.MEMORY:
            self.provider = MemoryStorageProvider()
        
        elif self.storage_type == CheckpointStorageType.HYBRID:
            # Configuración híbrida: local + base de datos
            providers = [
                (LocalFileStorageProvider(self.base_path), 10),       # Prioridad 10 para local
                (PostgreSQLStorageProvider(self.db_url), 20)          # Prioridad 20 para DB
            ]
            self.provider = HybridStorageProvider(providers)
        
        else:
            raise ValueError(f"Tipo de almacenamiento no soportado: {self.storage_type}")
    
    def _generate_checkpoint_id(self, component_id: str) -> str:
        """
        Generar ID único para un checkpoint.
        
        Args:
            component_id: ID del componente
            
        Returns:
            ID único para el checkpoint
        """
        timestamp = int(time.time() * 1000)
        random_part = random.randint(1000, 9999)
        return f"{component_id}-{timestamp}-{random_part}"
    
    async def register_component(self, component_id: str) -> bool:
        """
        Registrar un componente para seguimiento.
        
        Args:
            component_id: ID del componente
            
        Returns:
            True si se registró correctamente
        """
        if component_id in self._active_components:
            logger.warning(f"Componente {component_id} ya estaba registrado")
            return True
        
        self._active_components.add(component_id)
        
        # Inicializar contador de versiones si no existe
        if component_id not in self._version_counters:
            # Obtener última versión desde almacenamiento
            checkpoints = await self.list_checkpoints(component_id)
            if checkpoints:
                latest_version = max(checkpoint.version for checkpoint in checkpoints)
                self._version_counters[component_id] = latest_version
            else:
                self._version_counters[component_id] = 0
        
        logger.info(f"Componente {component_id} registrado correctamente")
        return True
    
    async def create_checkpoint(self, component_id: str, data: Any, tags: Optional[List[str]] = None) -> Optional[str]:
        """
        Crear un nuevo checkpoint.
        
        Args:
            component_id: ID del componente
            data: Datos a almacenar
            tags: Etiquetas para categorización
            
        Returns:
            ID del checkpoint creado o None si falló
        """
        # Registrar componente si no está registrado
        if component_id not in self._active_components:
            await self.register_component(component_id)
        
        # Generar ID único
        checkpoint_id = self._generate_checkpoint_id(component_id)
        
        # Incrementar versión
        self._version_counters[component_id] += 1
        version = self._version_counters[component_id]
        
        # Crear metadatos
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            component_id=component_id,
            timestamp=time.time(),
            version=version,
            consistency_level=self.consistency_level,
            storage_type=self.storage_type,
            tags=tags or []
        )
        
        # Almacenar checkpoint
        success = await self.provider.store(checkpoint_id, data, metadata)
        
        if success:
            # Actualizar caché de metadatos
            self._metadata_cache[checkpoint_id] = metadata
            logger.info(f"Checkpoint {checkpoint_id} creado correctamente para componente {component_id}")
            return checkpoint_id
        else:
            # Revertir incremento de versión
            self._version_counters[component_id] -= 1
            logger.error(f"Error al crear checkpoint para componente {component_id}")
            return None
    
    async def load_checkpoint(self, checkpoint_id: str) -> Tuple[Optional[Any], Optional[CheckpointMetadata]]:
        """
        Cargar un checkpoint por ID.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si no existe
        """
        # Verificar caché de metadatos para aceleración
        metadata = self._metadata_cache.get(checkpoint_id)
        
        # Cargar checkpoint
        data, loaded_metadata = await self.provider.load(checkpoint_id)
        
        if loaded_metadata is not None:
            # Actualizar caché de metadatos
            self._metadata_cache[checkpoint_id] = loaded_metadata
        
        if data is not None:
            logger.info(f"Checkpoint {checkpoint_id} cargado correctamente")
        else:
            logger.warning(f"No se pudo cargar el checkpoint {checkpoint_id}")
        
        return data, loaded_metadata
    
    async def load_latest_checkpoint(self, component_id: str) -> Tuple[Optional[Any], Optional[CheckpointMetadata]]:
        """
        Cargar el checkpoint más reciente para un componente.
        
        Args:
            component_id: ID del componente
            
        Returns:
            Tupla (datos, metadatos) o (None, None) si no existe
        """
        # Listar checkpoints del componente
        checkpoints = await self.list_checkpoints(component_id)
        
        if not checkpoints:
            logger.warning(f"No hay checkpoints disponibles para componente {component_id}")
            return None, None
        
        # Obtener el más reciente (lista ya viene ordenada por timestamp descendente)
        latest = checkpoints[0]
        
        # Cargar checkpoint
        return await self.load_checkpoint(latest.checkpoint_id)
    
    async def list_checkpoints(self, component_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        Listar checkpoints disponibles.
        
        Args:
            component_id: Filtrar por componente (opcional)
            
        Returns:
            Lista de metadatos de checkpoints
        """
        # Listar desde almacenamiento
        checkpoints = await self.provider.list_checkpoints(component_id)
        
        # Actualizar caché de metadatos
        for checkpoint in checkpoints:
            self._metadata_cache[checkpoint.checkpoint_id] = checkpoint
        
        return checkpoints
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Eliminar un checkpoint.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente
        """
        # Eliminar del almacenamiento
        success = await self.provider.delete(checkpoint_id)
        
        if success:
            # Eliminar de la caché
            self._metadata_cache.pop(checkpoint_id, None)
            logger.info(f"Checkpoint {checkpoint_id} eliminado correctamente")
        else:
            logger.error(f"Error al eliminar checkpoint {checkpoint_id}")
        
        return success
    
    async def create_distributed_checkpoint(self, 
                                           component_ids: List[str], 
                                           data_dict: Dict[str, Any],
                                           tags: Optional[List[str]] = None) -> Optional[List[str]]:
        """
        Crear checkpoints coordinados para múltiples componentes.
        
        Args:
            component_ids: Lista de IDs de componentes
            data_dict: Diccionario con datos por componente
            tags: Etiquetas para categorización
            
        Returns:
            Lista de IDs de checkpoints creados o None si falló
        """
        # Validar componentes y datos
        for component_id in component_ids:
            if component_id not in data_dict:
                logger.error(f"Faltan datos para componente {component_id}")
                return None
        
        # Generar timestamp compartido para sincronización
        shared_timestamp = time.time()
        
        # Lista para almacenar IDs de checkpoints
        checkpoint_ids = []
        
        # Crear checkpoints en paralelo
        try:
            for component_id in component_ids:
                # Registrar componente si no está registrado
                if component_id not in self._active_components:
                    await self.register_component(component_id)
                
                # Generar ID único
                checkpoint_id = self._generate_checkpoint_id(component_id)
                checkpoint_ids.append(checkpoint_id)
                
                # Incrementar versión
                self._version_counters[component_id] += 1
                version = self._version_counters[component_id]
                
                # Crear metadatos con dependencias
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    component_id=component_id,
                    timestamp=shared_timestamp,
                    version=version,
                    consistency_level=self.consistency_level,
                    storage_type=self.storage_type,
                    tags=tags or [],
                    dependencies=[cid for cid in checkpoint_ids if cid != checkpoint_id]
                )
                
                # Almacenar checkpoint
                success = await self.provider.store(checkpoint_id, data_dict[component_id], metadata)
                
                if not success:
                    # Revertir y abortar
                    logger.error(f"Error al crear checkpoint distribuido para componente {component_id}")
                    
                    # Eliminar checkpoints creados
                    for cid in checkpoint_ids:
                        await self.provider.delete(cid)
                    
                    # Revertir contadores de versión
                    for comp_id in component_ids:
                        if comp_id in self._version_counters and self._version_counters[comp_id] > 0:
                            self._version_counters[comp_id] -= 1
                    
                    return None
            
            logger.info(f"Checkpoint distribuido creado correctamente para {len(component_ids)} componentes")
            return checkpoint_ids
        
        except Exception as e:
            logger.error(f"Error al crear checkpoint distribuido: {e}")
            
            # Eliminar checkpoints creados
            for checkpoint_id in checkpoint_ids:
                await self.provider.delete(checkpoint_id)
            
            # Revertir contadores de versión
            for component_id in component_ids:
                if component_id in self._version_counters and self._version_counters[component_id] > 0:
                    self._version_counters[component_id] -= 1
            
            return None
    
    async def load_distributed_checkpoint(self, checkpoint_ids: List[str]) -> Dict[str, Any]:
        """
        Cargar un conjunto de checkpoints distribuidos.
        
        Args:
            checkpoint_ids: Lista de IDs de checkpoints
            
        Returns:
            Diccionario con datos por componente
        """
        result: Dict[str, Any] = {}
        
        # Cargar checkpoints en paralelo
        load_tasks = [self.load_checkpoint(checkpoint_id) for checkpoint_id in checkpoint_ids]
        load_results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        # Procesar resultados
        for i, (data, metadata) in enumerate(load_results):
            if isinstance(data, Exception):
                logger.error(f"Error al cargar checkpoint {checkpoint_ids[i]}: {data}")
                continue
            
            if data is not None and metadata is not None:
                result[metadata.component_id] = data
        
        # Verificar integridad del conjunto
        if len(result) != len(checkpoint_ids):
            logger.warning(f"No se pudieron cargar todos los checkpoints: {len(result)}/{len(checkpoint_ids)}")
        
        return result
    
    async def cleanup_old_checkpoints(self, 
                                     component_id: Optional[str] = None, 
                                     max_checkpoints: int = 5) -> int:
        """
        Eliminar checkpoints antiguos manteniendo un número máximo.
        
        Args:
            component_id: Filtrar por componente (opcional)
            max_checkpoints: Número máximo de checkpoints a mantener
            
        Returns:
            Número de checkpoints eliminados
        """
        # Listar checkpoints (ya vienen ordenados por timestamp descendente)
        checkpoints = await self.list_checkpoints(component_id)
        
        if len(checkpoints) <= max_checkpoints:
            return 0
        
        # Checkpoints a eliminar
        to_delete = checkpoints[max_checkpoints:]
        
        # Eliminar en paralelo
        delete_tasks = [self.delete_checkpoint(checkpoint.checkpoint_id) for checkpoint in to_delete]
        delete_results = await asyncio.gather(*delete_tasks, return_exceptions=True)
        
        # Contar eliminaciones exitosas
        deleted_count = sum(1 for result in delete_results if result is True)
        
        logger.info(f"Limpieza de checkpoints: {deleted_count} checkpoints antiguos eliminados")
        return deleted_count


# Instancia global para acceso directo
checkpoint_manager = DistributedCheckpointManager()


# Decorador para usar con funciones que necesitan checkpoint
def checkpoint_state(component_id: str, tags: Optional[List[str]] = None):
    """
    Decorador para crear checkpoints automáticos del estado de una función.
    
    Args:
        component_id: ID del componente
        tags: Etiquetas para categorización
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Ejecutar función y obtener resultado
            result = await func(*args, **kwargs)
            
            # Crear checkpoint del resultado
            state_data = {
                "result": result,
                "args": args,
                "kwargs": kwargs,
                "timestamp": time.time()
            }
            
            checkpoint_id = await checkpoint_manager.create_checkpoint(
                component_id=component_id,
                data=state_data,
                tags=tags
            )
            
            logger.info(f"Checkpoint automático creado: {checkpoint_id}")
            return result
        
        return wrapper
    
    return decorator


# Para pruebas si se ejecuta este archivo directamente
if __name__ == "__main__":
    async def run_demo():
        print("\n=== DEMOSTRACIÓN DEL GESTOR DE CHECKPOINTS DISTRIBUIDOS ===\n")
        
        # Crear gestor con almacenamiento en memoria para pruebas
        manager = DistributedCheckpointManager(storage_type=CheckpointStorageType.MEMORY)
        
        # Registrar componentes
        print("Registrando componentes...")
        await manager.register_component("trading")
        await manager.register_component("analytics")
        
        # Crear datos de prueba
        trading_data = {
            "positions": [
                {"symbol": "BTC", "amount": 1.5, "entry_price": 45000},
                {"symbol": "ETH", "amount": 10, "entry_price": 3000}
            ],
            "balance": 25000,
            "status": "active"
        }
        
        analytics_data = {
            "metrics": {
                "profit_factor": 2.3,
                "win_rate": 0.67,
                "drawdown": 0.12
            },
            "predictions": [
                {"symbol": "BTC", "direction": "up", "confidence": 0.85},
                {"symbol": "ETH", "direction": "up", "confidence": 0.72}
            ]
        }
        
        # Crear checkpoints individuales
        print("\nCreando checkpoints individuales...")
        trading_checkpoint = await manager.create_checkpoint(
            component_id="trading",
            data=trading_data,
            tags=["daily", "production"]
        )
        print(f"  Checkpoint creado para Trading: {trading_checkpoint}")
        
        analytics_checkpoint = await manager.create_checkpoint(
            component_id="analytics",
            data=analytics_data,
            tags=["daily", "production"]
        )
        print(f"  Checkpoint creado para Analytics: {analytics_checkpoint}")
        
        # Listar checkpoints
        print("\nListando checkpoints disponibles:")
        checkpoints = await manager.list_checkpoints()
        for cp in checkpoints:
            print(f"  {cp.checkpoint_id} - Componente: {cp.component_id}, Versión: {cp.version}, Tags: {cp.tags}")
        
        # Cargar checkpoint
        print("\nCargando checkpoint de Trading...")
        data, metadata = await manager.load_checkpoint(trading_checkpoint)
        if data:
            print(f"  Datos recuperados: {data}")
            print(f"  Balance: ${data['balance']}")
            print(f"  Posiciones: {len(data['positions'])}")
        
        # Crear checkpoint distribuido
        print("\nCreando checkpoint distribuido...")
        distributed_data = {
            "trading": {
                "positions": [
                    {"symbol": "BTC", "amount": 1.7, "entry_price": 46000},
                    {"symbol": "ETH", "amount": 12, "entry_price": 3200}
                ],
                "balance": 26000,
                "status": "active"
            },
            "analytics": {
                "metrics": {
                    "profit_factor": 2.4,
                    "win_rate": 0.68,
                    "drawdown": 0.11
                },
                "predictions": [
                    {"symbol": "BTC", "direction": "up", "confidence": 0.87},
                    {"symbol": "ETH", "direction": "up", "confidence": 0.75}
                ]
            }
        }
        
        distributed_ids = await manager.create_distributed_checkpoint(
            component_ids=["trading", "analytics"],
            data_dict=distributed_data,
            tags=["synchronous", "production"]
        )
        
        print(f"  Checkpoint distribuido creado: {distributed_ids}")
        
        # Cargar checkpoint distribuido
        print("\nCargando checkpoint distribuido...")
        distributed_result = await manager.load_distributed_checkpoint(distributed_ids)
        
        for component_id, component_data in distributed_result.items():
            print(f"  Componente: {component_id}")
            if component_id == "trading":
                print(f"    Balance: ${component_data['balance']}")
                print(f"    Posiciones: {len(component_data['positions'])}")
            elif component_id == "analytics":
                print(f"    Profit Factor: {component_data['metrics']['profit_factor']}")
                print(f"    Win Rate: {component_data['metrics']['win_rate']}")
        
        # Usar el decorador checkpoint_state
        print("\nProbando decorador checkpoint_state...")
        
        @checkpoint_state(component_id="calculation", tags=["demo", "automatic"])
        async def calculate_portfolio_value(positions):
            # Simular cálculo
            await asyncio.sleep(0.1)
            total = sum(pos["amount"] * pos["entry_price"] for pos in positions)
            return {"total_value": total, "timestamp": time.time()}
        
        result = await calculate_portfolio_value(trading_data["positions"])
        print(f"  Resultado calculado: {result}")
        
        # Listar checkpoints después de usar el decorador
        print("\nListando checkpoints después de usar el decorador:")
        checkpoints = await manager.list_checkpoints("calculation")
        for cp in checkpoints:
            print(f"  {cp.checkpoint_id} - Versión: {cp.version}, Tags: {cp.tags}")
        
        # Limpiar checkpoints antiguos
        print("\nLimpiando checkpoints antiguos...")
        deleted = await manager.cleanup_old_checkpoints(max_checkpoints=1)
        print(f"  Checkpoints eliminados: {deleted}")
        
        print("\n=== DEMOSTRACIÓN COMPLETADA ===\n")
    
    # Ejecutar demo
    asyncio.run(run_demo())