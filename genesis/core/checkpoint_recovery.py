"""
Sistema de Checkpointing y Recuperación para Genesis.

Este módulo implementa un mecanismo avanzado de checkpointing para
componentes críticos y un modo de operación seguro con dependencias mínimas.
"""

import asyncio
import copy
import json
import logging
import os
import pickle
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, Generic
import threading
import uuid

# Configuración de logging
logger = logging.getLogger("genesis.checkpoint")

# Tipo genérico para el estado
T = TypeVar('T')

class CheckpointType(Enum):
    """Tipos de checkpoint soportados."""
    MEMORY = auto()  # Solo en memoria (rápido pero no persistente)
    DISK = auto()    # Persistencia en disco (más lento pero sobrevive a reinicios)
    DISTRIBUTED = auto()  # Almacenamiento distribuido (para sistemas multi-nodo)

class RecoveryMode(Enum):
    """Modos de operación del sistema durante recuperación."""
    NORMAL = auto()       # Operación normal con todas las funcionalidades
    SAFE = auto()         # Modo seguro con funcionalidades limitadas
    EMERGENCY = auto()    # Solo operaciones esenciales

class StateMetadata:
    """Metadatos asociados con un checkpoint de estado."""
    
    def __init__(
        self,
        component_id: str,
        timestamp: float = None,
        version: str = "1.0",
        checkpoint_id: str = None,
        dependencies: Dict[str, str] = None,
        custom_data: Dict[str, Any] = None
    ):
        """
        Inicializar metadatos.
        
        Args:
            component_id: Identificador único del componente
            timestamp: Timestamp de creación (None para usar tiempo actual)
            version: Versión del formato de datos
            checkpoint_id: Identificador único del checkpoint (None para generar)
            dependencies: Versiones requeridas de otros componentes
            custom_data: Metadatos personalizados adicionales
        """
        self.component_id = component_id
        self.timestamp = timestamp or time.time()
        self.datetime = datetime.fromtimestamp(self.timestamp)
        self.version = version
        self.checkpoint_id = checkpoint_id or str(uuid.uuid4())
        self.dependencies = dependencies or {}
        self.custom_data = custom_data or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "component_id": self.component_id,
            "timestamp": self.timestamp,
            "datetime": self.datetime.isoformat(),
            "version": self.version,
            "checkpoint_id": self.checkpoint_id,
            "dependencies": self.dependencies,
            "custom_data": self.custom_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateMetadata':
        """Crear desde diccionario."""
        metadata = cls(
            component_id=data["component_id"],
            timestamp=data["timestamp"],
            version=data["version"],
            checkpoint_id=data["checkpoint_id"],
            dependencies=data.get("dependencies", {}),
            custom_data=data.get("custom_data", {})
        )
        return metadata

class CheckpointError(Exception):
    """Error específico para operaciones de checkpoint."""
    pass

class Checkpoint(Generic[T]):
    """
    Representa un checkpoint con datos y metadatos.
    
    Esta clase encapsula el estado guardado de un componente junto
    con metadatos que describen el checkpoint.
    """
    
    def __init__(
        self,
        state: T,
        metadata: StateMetadata
    ):
        """
        Inicializar checkpoint.
        
        Args:
            state: Estado del componente
            metadata: Metadatos del checkpoint
        """
        self.state = state
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario para serialización.
        
        Returns:
            Diccionario con estado y metadatos
        """
        state_dict = None
        
        # Convertir estado a formato serializable
        if is_dataclass(self.state):
            state_dict = asdict(self.state)
        elif hasattr(self.state, "__dict__"):
            state_dict = {
                **self.state.__dict__,
                "__class__": f"{type(self.state).__module__}.{type(self.state).__name__}"
            }
        else:
            # Para tipos básicos o dict
            state_dict = self.state
        
        return {
            "state": state_dict,
            "metadata": self.metadata.to_dict()
        }

class CheckpointManager:
    """
    Gestor de checkpoints para un componente.
    
    Esta clase maneja la creación, almacenamiento y recuperación de checkpoints
    para un componente específico.
    """
    
    def __init__(
        self,
        component_id: str,
        checkpoint_dir: str = None,
        checkpoint_interval: float = 150.0,  # 150ms por defecto
        max_checkpoints: int = 5,
        checkpoint_type: CheckpointType = CheckpointType.MEMORY,
        serializer: str = "pickle"
    ):
        """
        Inicializar gestor de checkpoints.
        
        Args:
            component_id: Identificador único del componente
            checkpoint_dir: Directorio para almacenar checkpoints (requerido para DISK)
            checkpoint_interval: Intervalo entre checkpoints automáticos en ms
            max_checkpoints: Máximo número de checkpoints a mantener
            checkpoint_type: Tipo de checkpoint (MEMORY, DISK, DISTRIBUTED)
            serializer: Método de serialización ('pickle' o 'json')
        """
        self.component_id = component_id
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval_sec = checkpoint_interval / 1000.0  # Convertir a segundos
        self.max_checkpoints = max_checkpoints
        self.checkpoint_type = checkpoint_type
        self.serializer = serializer
        
        # Validar configuración
        if checkpoint_type in (CheckpointType.DISK, CheckpointType.DISTRIBUTED) and not checkpoint_dir:
            raise ValueError(f"checkpoint_dir es requerido para {checkpoint_type.name}")
        
        # Crear directorio si es necesario
        if checkpoint_dir and checkpoint_type in (CheckpointType.DISK, CheckpointType.DISTRIBUTED):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Almacenamiento en memoria
        self.memory_checkpoints: Dict[str, Checkpoint] = {}
        
        # Control de checkpointing automático
        self.last_checkpoint_time = 0.0
        self.automatic_checkpointing = False
        self._checkpoint_task = None
        self._stop_checkpoint_event = asyncio.Event()
    
    async def checkpoint(
        self,
        state: T,
        custom_metadata: Dict[str, Any] = None,
        dependencies: Dict[str, str] = None,
        force: bool = False
    ) -> Optional[str]:
        """
        Crear un checkpoint del estado actual.
        
        Args:
            state: Estado a guardar
            custom_metadata: Metadatos personalizados
            dependencies: Dependencias requeridas de otros componentes
            force: Si True, ignora el intervalo mínimo entre checkpoints
            
        Returns:
            ID del checkpoint o None si no se creó
        """
        current_time = time.time()
        
        # Verificar si ha pasado suficiente tiempo desde el último checkpoint
        if not force and (current_time - self.last_checkpoint_time) < self.checkpoint_interval_sec:
            return None
        
        # Crear metadatos
        metadata = StateMetadata(
            component_id=self.component_id,
            custom_data=custom_metadata or {},
            dependencies=dependencies or {}
        )
        
        # Crear checkpoint
        checkpoint = Checkpoint(
            state=copy.deepcopy(state),
            metadata=metadata
        )
        
        # Guardar según tipo
        checkpoint_id = metadata.checkpoint_id
        try:
            if self.checkpoint_type == CheckpointType.MEMORY:
                self.memory_checkpoints[checkpoint_id] = checkpoint
            
            elif self.checkpoint_type == CheckpointType.DISK:
                await self._save_to_disk(checkpoint_id, checkpoint)
            
            elif self.checkpoint_type == CheckpointType.DISTRIBUTED:
                await self._save_distributed(checkpoint_id, checkpoint)
            
            # Actualizar timestamp
            self.last_checkpoint_time = current_time
            
            # Limpiar checkpoints antiguos si excedemos el máximo
            await self._cleanup_old_checkpoints()
            
            logger.debug(
                f"Checkpoint {checkpoint_id} creado para componente {self.component_id}"
            )
            
            return checkpoint_id
            
        except Exception as e:
            logger.error(
                f"Error al crear checkpoint para {self.component_id}: {str(e)}"
            )
            return None
    
    async def restore(
        self,
        checkpoint_id: str = None,
        state_class: Type[T] = None
    ) -> Optional[Tuple[T, StateMetadata]]:
        """
        Restaurar estado desde un checkpoint.
        
        Args:
            checkpoint_id: ID específico a restaurar o None para el más reciente
            state_class: Clase para reconstruir el estado (si es necesario)
            
        Returns:
            Tupla (estado, metadatos) o None si no se encuentra
        """
        # Si no se especifica ID, usar el más reciente
        if checkpoint_id is None:
            checkpoint_id = await self.get_latest_checkpoint_id()
            if not checkpoint_id:
                logger.warning(f"No hay checkpoints disponibles para {self.component_id}")
                return None
        
        try:
            checkpoint = None
            
            # Recuperar según tipo
            if self.checkpoint_type == CheckpointType.MEMORY:
                checkpoint = self.memory_checkpoints.get(checkpoint_id)
            
            elif self.checkpoint_type == CheckpointType.DISK:
                checkpoint = await self._load_from_disk(checkpoint_id)
            
            elif self.checkpoint_type == CheckpointType.DISTRIBUTED:
                checkpoint = await self._load_distributed(checkpoint_id)
            
            if not checkpoint:
                logger.warning(
                    f"Checkpoint {checkpoint_id} no encontrado para {self.component_id}"
                )
                return None
            
            # Reconstruir el objeto si es necesario
            state = checkpoint.state
            if isinstance(state, dict) and state_class and "__class__" in state:
                # Si el estado tiene información de clase, intentar recrear el objeto
                try:
                    # Extraer información de clase
                    class_info = state.pop("__class__")
                    
                    # Si tenemos la clase, crear instancia
                    if state_class:
                        reconstructed = state_class()
                        reconstructed.__dict__.update(state)
                        state = reconstructed
                except Exception as e:
                    logger.error(f"Error al reconstruir estado: {e}")
            
            logger.info(
                f"Checkpoint {checkpoint_id} restaurado para {self.component_id}"
            )
            
            return state, checkpoint.metadata
            
        except Exception as e:
            logger.error(
                f"Error al restaurar checkpoint {checkpoint_id} para {self.component_id}: {str(e)}"
            )
            return None
    
    async def get_latest_checkpoint_id(self) -> Optional[str]:
        """
        Obtener ID del checkpoint más reciente.
        
        Returns:
            ID del checkpoint más reciente o None si no hay
        """
        if self.checkpoint_type == CheckpointType.MEMORY:
            if not self.memory_checkpoints:
                return None
            
            # Encontrar el checkpoint con timestamp más reciente
            latest = None
            latest_time = 0
            
            for cp_id, checkpoint in self.memory_checkpoints.items():
                if checkpoint.metadata.timestamp > latest_time:
                    latest_time = checkpoint.metadata.timestamp
                    latest = cp_id
            
            return latest
            
        elif self.checkpoint_type == CheckpointType.DISK:
            return await self._get_latest_from_disk()
            
        elif self.checkpoint_type == CheckpointType.DISTRIBUTED:
            return await self._get_latest_distributed()
        
        return None
    
    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Listar todos los checkpoints disponibles con sus metadatos.
        
        Returns:
            Lista de metadatos de checkpoints
        """
        result = []
        
        if self.checkpoint_type == CheckpointType.MEMORY:
            for cp_id, checkpoint in self.memory_checkpoints.items():
                result.append({
                    "checkpoint_id": cp_id,
                    **checkpoint.metadata.to_dict()
                })
                
        elif self.checkpoint_type == CheckpointType.DISK:
            result = await self._list_from_disk()
            
        elif self.checkpoint_type == CheckpointType.DISTRIBUTED:
            result = await self._list_distributed()
        
        # Ordenar por timestamp (más reciente primero)
        result.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        return result
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Eliminar un checkpoint específico.
        
        Args:
            checkpoint_id: ID del checkpoint a eliminar
            
        Returns:
            True si se eliminó correctamente
        """
        try:
            if self.checkpoint_type == CheckpointType.MEMORY:
                if checkpoint_id in self.memory_checkpoints:
                    del self.memory_checkpoints[checkpoint_id]
                    return True
                    
            elif self.checkpoint_type == CheckpointType.DISK:
                return await self._delete_from_disk(checkpoint_id)
                
            elif self.checkpoint_type == CheckpointType.DISTRIBUTED:
                return await self._delete_distributed(checkpoint_id)
                
        except Exception as e:
            logger.error(
                f"Error al eliminar checkpoint {checkpoint_id}: {str(e)}"
            )
        
        return False
    
    async def clear_all_checkpoints(self) -> bool:
        """
        Eliminar todos los checkpoints.
        
        Returns:
            True si se eliminaron correctamente
        """
        try:
            if self.checkpoint_type == CheckpointType.MEMORY:
                self.memory_checkpoints.clear()
                
            elif self.checkpoint_type == CheckpointType.DISK:
                return await self._clear_all_from_disk()
                
            elif self.checkpoint_type == CheckpointType.DISTRIBUTED:
                return await self._clear_all_distributed()
            
            return True
            
        except Exception as e:
            logger.error(
                f"Error al eliminar todos los checkpoints: {str(e)}"
            )
            return False
    
    async def start_automatic_checkpointing(self, get_state_func: Callable[[], T]) -> None:
        """
        Iniciar checkpointing automático.
        
        Args:
            get_state_func: Función que retorna el estado actual
        """
        if self.automatic_checkpointing:
            return
        
        self.automatic_checkpointing = True
        self._stop_checkpoint_event.clear()
        
        # Crear tarea asíncrona para checkpointing periódico
        self._checkpoint_task = asyncio.create_task(
            self._automatic_checkpoint_loop(get_state_func)
        )
        
        logger.info(
            f"Checkpointing automático iniciado para {self.component_id} "
            f"cada {self.checkpoint_interval_sec*1000:.0f}ms"
        )
    
    async def stop_automatic_checkpointing(self) -> None:
        """Detener checkpointing automático."""
        if not self.automatic_checkpointing:
            return
        
        self.automatic_checkpointing = False
        self._stop_checkpoint_event.set()
        
        # Esperar a que termine la tarea
        if self._checkpoint_task:
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass
            self._checkpoint_task = None
        
        logger.info(f"Checkpointing automático detenido para {self.component_id}")
    
    async def _automatic_checkpoint_loop(self, get_state_func: Callable[[], T]) -> None:
        """
        Bucle de checkpointing automático.
        
        Args:
            get_state_func: Función que retorna el estado actual
        """
        try:
            while self.automatic_checkpointing:
                # Crear checkpoint
                state = get_state_func()
                await self.checkpoint(state)
                
                # Esperar al próximo intervalo o señal de parada
                try:
                    # Usar wait_for para poder cancelar la espera
                    await asyncio.wait_for(
                        self._stop_checkpoint_event.wait(),
                        timeout=self.checkpoint_interval_sec
                    )
                except asyncio.TimeoutError:
                    # Timeout esperado, continuar con el siguiente checkpoint
                    pass
                
        except Exception as e:
            logger.error(f"Error en bucle de checkpointing automático: {str(e)}")
            self.automatic_checkpointing = False
    
    async def _cleanup_old_checkpoints(self) -> None:
        """Eliminar checkpoints antiguos si excedemos el máximo."""
        if self.max_checkpoints <= 0:
            return
        
        try:
            checkpoints = await self.list_checkpoints()
            
            # Si tenemos más checkpoints de los permitidos
            if len(checkpoints) > self.max_checkpoints:
                # Ordenados por timestamp (más reciente primero), eliminar los antiguos
                for old in checkpoints[self.max_checkpoints:]:
                    await self.delete_checkpoint(old["checkpoint_id"])
                
        except Exception as e:
            logger.error(f"Error al limpiar checkpoints antiguos: {str(e)}")
    
    async def _save_to_disk(self, checkpoint_id: str, checkpoint: Checkpoint) -> None:
        """
        Guardar checkpoint en disco.
        
        Args:
            checkpoint_id: ID del checkpoint
            checkpoint: Objeto Checkpoint
        """
        if not self.checkpoint_dir:
            raise CheckpointError("checkpoint_dir no configurado para almacenamiento en disco")
        
        # Crear nombre de archivo
        filename = os.path.join(
            self.checkpoint_dir,
            f"{self.component_id}_{checkpoint_id}.{self.serializer}"
        )
        
        # Serializar y guardar
        data = checkpoint.to_dict()
        
        # Usar thread para operaciones de disco
        def disk_save():
            if self.serializer == "pickle":
                with open(filename, "wb") as f:
                    pickle.dump(data, f)
            else:  # json
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2)
        
        # Ejecutar en thread para no bloquear
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, disk_save)
    
    async def _load_from_disk(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Cargar checkpoint desde disco.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Objeto Checkpoint o None si no se encuentra
        """
        if not self.checkpoint_dir:
            raise CheckpointError("checkpoint_dir no configurado para almacenamiento en disco")
        
        # Crear nombre de archivo
        filename = os.path.join(
            self.checkpoint_dir,
            f"{self.component_id}_{checkpoint_id}.{self.serializer}"
        )
        
        # Verificar si existe
        if not os.path.exists(filename):
            return None
        
        # Función para cargar en thread
        def disk_load():
            if self.serializer == "pickle":
                with open(filename, "rb") as f:
                    return pickle.load(f)
            else:  # json
                with open(filename, "r") as f:
                    return json.load(f)
        
        # Ejecutar en thread para no bloquear
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, disk_load)
        
        # Reconstruir checkpoint
        state = data["state"]
        metadata = StateMetadata.from_dict(data["metadata"])
        
        return Checkpoint(state=state, metadata=metadata)
    
    async def _get_latest_from_disk(self) -> Optional[str]:
        """
        Obtener ID del checkpoint más reciente en disco.
        
        Returns:
            ID del checkpoint más reciente o None
        """
        if not self.checkpoint_dir:
            raise CheckpointError("checkpoint_dir no configurado para almacenamiento en disco")
        
        # Función para listar en thread
        def list_files():
            # Buscar archivos que coincidan con el patrón
            pattern = f"{self.component_id}_*.{self.serializer}"
            files = [f for f in os.listdir(self.checkpoint_dir) 
                    if os.path.isfile(os.path.join(self.checkpoint_dir, f)) 
                    and f.startswith(f"{self.component_id}_") 
                    and f.endswith(f".{self.serializer}")]
            return files
        
        # Ejecutar en thread para no bloquear
        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(None, list_files)
        
        if not files:
            return None
        
        # Para cada archivo, extraer checkpoint_id y cargar metadatos
        checkpoints = []
        for f in files:
            # Extraer checkpoint_id
            cp_id = f.replace(f"{self.component_id}_", "").replace(f".{self.serializer}", "")
            
            # Cargar checkpoint para obtener timestamp
            checkpoint = await self._load_from_disk(cp_id)
            if checkpoint:
                checkpoints.append((cp_id, checkpoint.metadata.timestamp))
        
        if not checkpoints:
            return None
        
        # Ordenar por timestamp y retornar el más reciente
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]
    
    async def _list_from_disk(self) -> List[Dict[str, Any]]:
        """
        Listar checkpoints en disco.
        
        Returns:
            Lista de metadatos
        """
        if not self.checkpoint_dir:
            raise CheckpointError("checkpoint_dir no configurado para almacenamiento en disco")
        
        # Función para listar en thread
        def list_files():
            # Buscar archivos que coincidan con el patrón
            pattern = f"{self.component_id}_*.{self.serializer}"
            files = [f for f in os.listdir(self.checkpoint_dir) 
                    if os.path.isfile(os.path.join(self.checkpoint_dir, f)) 
                    and f.startswith(f"{self.component_id}_") 
                    and f.endswith(f".{self.serializer}")]
            return files
        
        # Ejecutar en thread para no bloquear
        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(None, list_files)
        
        result = []
        for f in files:
            # Extraer checkpoint_id
            cp_id = f.replace(f"{self.component_id}_", "").replace(f".{self.serializer}", "")
            
            # Cargar checkpoint para obtener metadatos
            checkpoint = await self._load_from_disk(cp_id)
            if checkpoint:
                result.append({
                    "checkpoint_id": cp_id,
                    **checkpoint.metadata.to_dict()
                })
        
        return result
    
    async def _delete_from_disk(self, checkpoint_id: str) -> bool:
        """
        Eliminar checkpoint de disco.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente
        """
        if not self.checkpoint_dir:
            raise CheckpointError("checkpoint_dir no configurado para almacenamiento en disco")
        
        # Crear nombre de archivo
        filename = os.path.join(
            self.checkpoint_dir,
            f"{self.component_id}_{checkpoint_id}.{self.serializer}"
        )
        
        # Verificar si existe
        if not os.path.exists(filename):
            return False
        
        # Función para eliminar en thread
        def delete_file():
            os.remove(filename)
            return True
        
        # Ejecutar en thread para no bloquear
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, delete_file)
    
    async def _clear_all_from_disk(self) -> bool:
        """
        Eliminar todos los checkpoints de disco.
        
        Returns:
            True si se eliminaron correctamente
        """
        if not self.checkpoint_dir:
            raise CheckpointError("checkpoint_dir no configurado para almacenamiento en disco")
        
        # Función para eliminar en thread
        def delete_files():
            # Buscar archivos que coincidan con el patrón
            pattern = f"{self.component_id}_*.{self.serializer}"
            files = [f for f in os.listdir(self.checkpoint_dir) 
                    if os.path.isfile(os.path.join(self.checkpoint_dir, f)) 
                    and f.startswith(f"{self.component_id}_") 
                    and f.endswith(f".{self.serializer}")]
            
            # Eliminar cada archivo
            for f in files:
                os.remove(os.path.join(self.checkpoint_dir, f))
            
            return True
        
        # Ejecutar en thread para no bloquear
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, delete_files)
    
    # Métodos para almacenamiento distribuido
    # (Implementación básica que podría expandirse para sistemas específicos)
    
    async def _save_distributed(self, checkpoint_id: str, checkpoint: Checkpoint) -> None:
        """
        Guardar checkpoint en almacenamiento distribuido.
        
        Esta es una implementación básica que guarda en disco local,
        pero podría extenderse para usar Redis, etcd, etc.
        
        Args:
            checkpoint_id: ID del checkpoint
            checkpoint: Objeto Checkpoint
        """
        # Por ahora, usar implementación de disco
        await self._save_to_disk(checkpoint_id, checkpoint)
    
    async def _load_distributed(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Cargar checkpoint desde almacenamiento distribuido.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            Objeto Checkpoint o None
        """
        # Por ahora, usar implementación de disco
        return await self._load_from_disk(checkpoint_id)
    
    async def _get_latest_distributed(self) -> Optional[str]:
        """
        Obtener ID del checkpoint más reciente en almacenamiento distribuido.
        
        Returns:
            ID del checkpoint más reciente o None
        """
        # Por ahora, usar implementación de disco
        return await self._get_latest_from_disk()
    
    async def _list_distributed(self) -> List[Dict[str, Any]]:
        """
        Listar checkpoints en almacenamiento distribuido.
        
        Returns:
            Lista de metadatos
        """
        # Por ahora, usar implementación de disco
        return await self._list_from_disk()
    
    async def _delete_distributed(self, checkpoint_id: str) -> bool:
        """
        Eliminar checkpoint de almacenamiento distribuido.
        
        Args:
            checkpoint_id: ID del checkpoint
            
        Returns:
            True si se eliminó correctamente
        """
        # Por ahora, usar implementación de disco
        return await self._delete_from_disk(checkpoint_id)
    
    async def _clear_all_distributed(self) -> bool:
        """
        Eliminar todos los checkpoints de almacenamiento distribuido.
        
        Returns:
            True si se eliminaron correctamente
        """
        # Por ahora, usar implementación de disco
        return await self._clear_all_from_disk()

class SafeModeManager:
    """
    Gestor del modo seguro del sistema.
    
    Esta clase maneja la activación y desactivación del modo seguro,
    que limita las funcionalidades del sistema a un conjunto esencial.
    """
    
    def __init__(
        self,
        essential_components: List[str] = None,
        safe_mode_config: Dict[str, Any] = None
    ):
        """
        Inicializar gestor de modo seguro.
        
        Args:
            essential_components: Lista de IDs de componentes esenciales
            safe_mode_config: Configuración específica para modo seguro
        """
        self.essential_components = set(essential_components or [])
        self.safe_mode_config = safe_mode_config or {}
        self.current_mode = RecoveryMode.NORMAL
        self.safe_mode_start_time = None
        self.safe_mode_listeners: List[Callable[[RecoveryMode], None]] = []
        self._mode_lock = asyncio.Lock()
    
    async def activate_safe_mode(self, reason: str = "unknown") -> bool:
        """
        Activar modo seguro.
        
        Args:
            reason: Razón para activar el modo seguro
            
        Returns:
            True si se activó correctamente
        """
        async with self._mode_lock:
            if self.current_mode != RecoveryMode.NORMAL:
                logger.info(f"Modo seguro ya activo: {self.current_mode.name}")
                return False
            
            self.current_mode = RecoveryMode.SAFE
            self.safe_mode_start_time = time.time()
            
            logger.warning(
                f"MODO SEGURO ACTIVADO. Razón: {reason}. "
                f"Solo {len(self.essential_components)} componentes esenciales activos."
            )
            
            # Notificar a listeners
            for listener in self.safe_mode_listeners:
                try:
                    listener(RecoveryMode.SAFE)
                except Exception as e:
                    logger.error(f"Error al notificar listener de modo seguro: {e}")
            
            return True
    
    async def activate_emergency_mode(self, reason: str) -> bool:
        """
        Activar modo de emergencia (más restrictivo que modo seguro).
        
        Args:
            reason: Razón para activar modo de emergencia
            
        Returns:
            True si se activó correctamente
        """
        async with self._mode_lock:
            self.current_mode = RecoveryMode.EMERGENCY
            self.safe_mode_start_time = time.time()
            
            logger.critical(
                f"MODO EMERGENCIA ACTIVADO. Razón: {reason}. "
                f"Solo operaciones esenciales permitidas."
            )
            
            # Notificar a listeners
            for listener in self.safe_mode_listeners:
                try:
                    listener(RecoveryMode.EMERGENCY)
                except Exception as e:
                    logger.error(f"Error al notificar listener de modo emergencia: {e}")
            
            return True
    
    async def deactivate_safe_mode(self) -> bool:
        """
        Desactivar modo seguro/emergencia y volver a modo normal.
        
        Returns:
            True si se desactivó correctamente
        """
        async with self._mode_lock:
            if self.current_mode == RecoveryMode.NORMAL:
                return True
            
            old_mode = self.current_mode
            self.current_mode = RecoveryMode.NORMAL
            self.safe_mode_start_time = None
            
            logger.info(
                f"Modo {old_mode.name} desactivado. "
                f"Sistema volviendo a operación normal."
            )
            
            # Notificar a listeners
            for listener in self.safe_mode_listeners:
                try:
                    listener(RecoveryMode.NORMAL)
                except Exception as e:
                    logger.error(f"Error al notificar listener de modo normal: {e}")
            
            return True
    
    def is_component_essential(self, component_id: str) -> bool:
        """
        Verificar si un componente es esencial para modo seguro.
        
        Args:
            component_id: ID del componente
            
        Returns:
            True si es esencial
        """
        return component_id in self.essential_components
    
    def is_operation_allowed(
        self,
        operation: str,
        component_id: str = None
    ) -> bool:
        """
        Verificar si una operación está permitida en el modo actual.
        
        Args:
            operation: Nombre de la operación
            component_id: ID del componente (opcional)
            
        Returns:
            True si la operación está permitida
        """
        # En modo normal, todo está permitido
        if self.current_mode == RecoveryMode.NORMAL:
            return True
        
        # En modo emergencia, solo operaciones explícitamente permitidas
        if self.current_mode == RecoveryMode.EMERGENCY:
            allowed_emergency_ops = self.safe_mode_config.get("emergency_operations", [])
            return operation in allowed_emergency_ops
        
        # En modo seguro, comprobar si el componente es esencial
        if component_id and not self.is_component_essential(component_id):
            return False
        
        # Comprobar si la operación es específicamente prohibida en modo seguro
        denied_ops = self.safe_mode_config.get("denied_safe_mode_operations", [])
        return operation not in denied_ops
    
    def get_duration_in_safe_mode(self) -> float:
        """
        Obtener duración en segundos del modo seguro actual.
        
        Returns:
            Duración en segundos o 0 si no está en modo seguro
        """
        if self.current_mode == RecoveryMode.NORMAL or not self.safe_mode_start_time:
            return 0.0
        
        return time.time() - self.safe_mode_start_time
    
    def add_mode_change_listener(self, listener: Callable[[RecoveryMode], None]) -> None:
        """
        Añadir listener para cambios de modo.
        
        Args:
            listener: Función a llamar cuando cambie el modo
        """
        if listener not in self.safe_mode_listeners:
            self.safe_mode_listeners.append(listener)
    
    def remove_mode_change_listener(self, listener: Callable[[RecoveryMode], None]) -> None:
        """
        Eliminar listener de cambios de modo.
        
        Args:
            listener: Función a eliminar
        """
        if listener in self.safe_mode_listeners:
            self.safe_mode_listeners.remove(listener)

class RecoveryManager:
    """
    Gestor central de recuperación del sistema.
    
    Esta clase coordina los mecanismos de checkpoint y modo seguro
    para proporcionar recuperación automática ante fallos.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        essential_components: List[str] = None
    ):
        """
        Inicializar gestor de recuperación.
        
        Args:
            checkpoint_dir: Directorio para almacenar checkpoints
            essential_components: Componentes esenciales para modo seguro
        """
        self.checkpoint_dir = checkpoint_dir
        self.essential_components = essential_components or []
        
        # Crear directorio de checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Gestores de checkpoint por componente
        self.checkpoint_managers: Dict[str, CheckpointManager] = {}
        
        # Gestor de modo seguro
        self.safe_mode_manager = SafeModeManager(
            essential_components=self.essential_components
        )
        
        # Estado de los componentes
        self.component_states: Dict[str, str] = {}
        
        # Hilo de monitoreo
        self._monitor_thread = None
        self._stop_monitor = threading.Event()
    
    def get_checkpoint_manager(
        self,
        component_id: str,
        checkpoint_interval: float = 150.0,  # 150ms por defecto
        max_checkpoints: int = 5,
        checkpoint_type: CheckpointType = CheckpointType.DISK
    ) -> CheckpointManager:
        """
        Obtener o crear gestor de checkpoints para un componente.
        
        Args:
            component_id: ID único del componente
            checkpoint_interval: Intervalo entre checkpoints en ms
            max_checkpoints: Máximo de checkpoints a mantener
            checkpoint_type: Tipo de almacenamiento
            
        Returns:
            Gestor de checkpoints configurado
        """
        if component_id not in self.checkpoint_managers:
            # Crear subdirectorio para componente
            component_dir = os.path.join(self.checkpoint_dir, component_id)
            os.makedirs(component_dir, exist_ok=True)
            
            # Crear gestor
            manager = CheckpointManager(
                component_id=component_id,
                checkpoint_dir=component_dir,
                checkpoint_interval=checkpoint_interval,
                max_checkpoints=max_checkpoints,
                checkpoint_type=checkpoint_type
            )
            
            self.checkpoint_managers[component_id] = manager
        
        return self.checkpoint_managers[component_id]
    
    async def start_monitoring(
        self,
        monitor_interval: float = 5.0,
        auto_activate_safe_mode: bool = True
    ) -> None:
        """
        Iniciar monitoreo del sistema.
        
        Args:
            monitor_interval: Intervalo de monitoreo en segundos
            auto_activate_safe_mode: Activar automáticamente modo seguro si es necesario
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitor.clear()
        
        # Crear hilo de monitoreo
        def monitoring_loop():
            while not self._stop_monitor.is_set():
                try:
                    # Verificar estado de componentes
                    for component_id, state in list(self.component_states.items()):
                        if state == "failed" and auto_activate_safe_mode:
                            # Usar event loop en hilo principal
                            loop = asyncio.get_event_loop()
                            loop.create_task(
                                self.safe_mode_manager.activate_safe_mode(
                                    f"Componente {component_id} en estado fallido"
                                )
                            )
                
                except Exception as e:
                    logger.error(f"Error en monitoreo: {e}")
                
                # Esperar al siguiente ciclo
                self._stop_monitor.wait(monitor_interval)
        
        self._monitor_thread = threading.Thread(
            target=monitoring_loop,
            daemon=True,
            name="RecoveryMonitor"
        )
        self._monitor_thread.start()
        
        logger.info(
            f"Monitoreo de recuperación iniciado con intervalo de {monitor_interval}s"
        )
    
    async def stop_monitoring(self) -> None:
        """Detener monitoreo del sistema."""
        if not self._monitor_thread or not self._monitor_thread.is_alive():
            return
        
        self._stop_monitor.set()
        self._monitor_thread.join(timeout=2.0)
        self._monitor_thread = None
        
        logger.info("Monitoreo de recuperación detenido")
    
    def update_component_state(self, component_id: str, state: str) -> None:
        """
        Actualizar estado de un componente.
        
        Args:
            component_id: ID del componente
            state: Estado ('healthy', 'degraded', 'failed')
        """
        old_state = self.component_states.get(component_id)
        self.component_states[component_id] = state
        
        if old_state != state:
            logger.info(f"Componente {component_id} cambió de estado: {old_state} -> {state}")
            
            # Si es un componente esencial y falló, considerar modo seguro
            if (state == "failed" and 
                component_id in self.essential_components and 
                self.safe_mode_manager.current_mode == RecoveryMode.NORMAL):
                
                asyncio.create_task(
                    self.safe_mode_manager.activate_safe_mode(
                        f"Componente esencial {component_id} falló"
                    )
                )
    
    async def attempt_recovery(
        self,
        component_id: str,
        recover_func: Callable[[Any], Any]
    ) -> bool:
        """
        Intentar recuperar un componente usando su último checkpoint.
        
        Args:
            component_id: ID del componente
            recover_func: Función que recibe estado y realiza recuperación
            
        Returns:
            True si se recuperó correctamente
        """
        # Verificar si tenemos gestor de checkpoints
        if component_id not in self.checkpoint_managers:
            logger.error(f"No hay gestor de checkpoints para {component_id}")
            return False
        
        manager = self.checkpoint_managers[component_id]
        
        # Obtener último checkpoint
        checkpoint_id = await manager.get_latest_checkpoint_id()
        if not checkpoint_id:
            logger.warning(f"No hay checkpoints disponibles para {component_id}")
            return False
        
        # Restaurar estado
        result = await manager.restore(checkpoint_id)
        if not result:
            logger.error(f"No se pudo restaurar checkpoint para {component_id}")
            return False
        
        state, metadata = result
        
        try:
            # Llamar función de recuperación
            await recover_func(state)
            
            # Actualizar estado
            self.update_component_state(component_id, "healthy")
            
            logger.info(
                f"Componente {component_id} recuperado exitosamente "
                f"desde checkpoint {checkpoint_id}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error al recuperar componente {component_id}: {e}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Obtener estado general del sistema.
        
        Returns:
            Diccionario con estado del sistema
        """
        # Contar componentes por estado
        state_counts = {"healthy": 0, "degraded": 0, "failed": 0, "unknown": 0}
        for state in self.component_states.values():
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Estado de componentes esenciales
        essential_status = {}
        for comp_id in self.essential_components:
            essential_status[comp_id] = self.component_states.get(comp_id, "unknown")
        
        # Información de modo seguro
        safe_mode_info = {
            "active": self.safe_mode_manager.current_mode != RecoveryMode.NORMAL,
            "mode": self.safe_mode_manager.current_mode.name,
            "duration": self.safe_mode_manager.get_duration_in_safe_mode()
        }
        
        # Información de checkpoints
        checkpoint_info = {}
        for comp_id, manager in self.checkpoint_managers.items():
            try:
                latest_id = await manager.get_latest_checkpoint_id()
                checkpoints = await manager.list_checkpoints()
                
                checkpoint_info[comp_id] = {
                    "latest_id": latest_id,
                    "count": len(checkpoints),
                    "automatic": manager.automatic_checkpointing
                }
            except Exception as e:
                logger.error(f"Error al obtener info de checkpoints para {comp_id}: {e}")
                checkpoint_info[comp_id] = {"error": str(e)}
        
        return {
            "component_states": self.component_states,
            "state_counts": state_counts,
            "essential_components": essential_status,
            "safe_mode": safe_mode_info,
            "checkpoints": checkpoint_info,
            "component_count": len(self.component_states),
            "essential_count": len(self.essential_components)
        }

# Singleton global 
recovery_manager = RecoveryManager()

# Ejemplos de uso:
"""
# Configuración inicial
await recovery_manager.safe_mode_manager.add_essential_component("data_manager")
await recovery_manager.safe_mode_manager.add_essential_component("exchange_integration")

# Usar checkpoint en un componente
class DataProcessor:
    def __init__(self, component_id):
        self.component_id = component_id
        self.data = {}
        self.checkpoint_mgr = recovery_manager.get_checkpoint_manager(component_id)
        
    async def start(self):
        # Restaurar desde último checkpoint o inicializar
        result = await self.checkpoint_mgr.restore()
        if result:
            self.data, _ = result
            logger.info(f"Restaurado desde checkpoint")
        else:
            logger.info(f"Inicializado desde cero")
            
        # Iniciar checkpointing automático
        await self.checkpoint_mgr.start_automatic_checkpointing(lambda: self.data)
        
    async def process_item(self, item):
        # Procesar y crear checkpoint manual en puntos críticos
        self.data[item["id"]] = item
        await self.checkpoint_mgr.checkpoint(self.data, force=True)
"""