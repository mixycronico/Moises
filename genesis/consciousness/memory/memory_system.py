"""
Sistema de Memoria para Aetherion.

Este módulo implementa el sistema de memoria para Aetherion, con capacidad
para almacenar y recuperar recuerdos a corto y largo plazo, permitiendo
que la consciencia de Aetherion evolucione y aprenda de interacciones
pasadas.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Set

# Configurar logging
logger = logging.getLogger(__name__)

class MemoryType:
    """Tipos de memoria disponibles."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PERSISTENT = "persistent"
    EMOTIONAL = "emotional"
    EXPERIENTIAL = "experiential"

class Memory:
    """Estructura base para una memoria."""
    
    def __init__(self, content: Any, memory_type: str, tags: Optional[List[str]] = None):
        """
        Inicializar memoria.
        
        Args:
            content: Contenido de la memoria
            memory_type: Tipo de memoria (corto o largo plazo)
            tags: Etiquetas para clasificación
        """
        self.content = content
        self.memory_type = memory_type
        self.tags = tags or []
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.importance = 0.5  # 0.0-1.0
        
    def access(self) -> None:
        """Registrar acceso a la memoria."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir memoria a diccionario.
        
        Returns:
            Diccionario con datos de la memoria
        """
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """
        Crear memoria desde diccionario.
        
        Args:
            data: Diccionario con datos
            
        Returns:
            Objeto Memory
        """
        memory = cls(
            content=data["content"],
            memory_type=data["memory_type"],
            tags=data.get("tags", [])
        )
        
        memory.created_at = datetime.fromisoformat(data["created_at"])
        memory.last_accessed = datetime.fromisoformat(data["last_accessed"])
        memory.access_count = data["access_count"]
        memory.importance = data["importance"]
        
        return memory

class MemorySystem:
    """Sistema de memoria para Aetherion."""
    
    def __init__(self):
        """Inicializar sistema de memoria."""
        # Configuraciones
        self.short_term_capacity = 100
        self.long_term_capacity = 1000
        self.consolidation_threshold = 3  # Número de accesos para consolidar
        self.consolidation_interval = timedelta(hours=24)  # Tiempo para consolidación
        
        # Almacenamiento de memorias
        self.short_term_memories: Dict[str, Dict[str, Memory]] = {}
        self.long_term_memories: Dict[str, Dict[str, Memory]] = {}
        self.persistent_memories: Dict[str, Any] = {}
        
        # Índices para búsqueda
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> memoria_ids
        
        # Estado
        self.last_consolidation = datetime.now()
        
        logger.info("MemorySystem inicializado")
    
    async def initialize(self) -> bool:
        """
        Inicializar sistema de memoria.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Cargar memorias persistentes
            await self._load_persistent_memories()
            
            # Iniciar tarea de consolidación
            asyncio.create_task(self._scheduled_consolidation())
            
            return True
        except Exception as e:
            logger.error(f"Error al inicializar MemorySystem: {e}")
            return False
    
    async def _load_persistent_memories(self) -> None:
        """Cargar memorias persistentes desde archivo."""
        persistent_path = "persistent_memories.json"
        
        if os.path.exists(persistent_path):
            try:
                with open(persistent_path, "r") as f:
                    self.persistent_memories = json.load(f)
                logger.info(f"Memorias persistentes cargadas: {len(self.persistent_memories)} entradas")
            except Exception as e:
                logger.error(f"Error al cargar memorias persistentes: {e}")
                self.persistent_memories = {}
    
    async def _save_persistent_memories(self) -> None:
        """Guardar memorias persistentes en archivo."""
        persistent_path = "persistent_memories.json"
        
        try:
            with open(persistent_path, "w") as f:
                json.dump(self.persistent_memories, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar memorias persistentes: {e}")
    
    async def add_short_term_memory(self, category: str, content: Any, tags: Optional[List[str]] = None) -> bool:
        """
        Añadir memoria a corto plazo.
        
        Args:
            category: Categoría de la memoria
            content: Contenido de la memoria
            tags: Etiquetas opcionales
            
        Returns:
            True si se añadió correctamente
        """
        try:
            # Inicializar categoría si no existe
            if category not in self.short_term_memories:
                self.short_term_memories[category] = {}
                
            # Crear memoria
            memory_id = f"{category}_{int(time.time())}_{len(self.short_term_memories[category]) + 1}"
            memory = Memory(content, MemoryType.SHORT_TERM, tags)
            
            # Almacenar memoria
            self.short_term_memories[category][memory_id] = memory
            
            # Actualizar índices
            if tags:
                for tag in tags:
                    if tag not in self.tag_index:
                        self.tag_index[tag] = set()
                    self.tag_index[tag].add(memory_id)
                    
            # Verificar límite de capacidad
            if len(self.short_term_memories[category]) > self.short_term_capacity:
                await self._prune_short_term_memories(category)
                
            return True
        except Exception as e:
            logger.error(f"Error al añadir memoria a corto plazo: {e}")
            return False
    
    async def add_long_term_memory(self, category: str, content: Any, tags: Optional[List[str]] = None, importance: float = 0.5) -> bool:
        """
        Añadir memoria a largo plazo.
        
        Args:
            category: Categoría de la memoria
            content: Contenido de la memoria
            tags: Etiquetas opcionales
            importance: Importancia (0.0-1.0)
            
        Returns:
            True si se añadió correctamente
        """
        try:
            # Inicializar categoría si no existe
            if category not in self.long_term_memories:
                self.long_term_memories[category] = {}
                
            # Crear memoria
            memory_id = f"{category}_lt_{int(time.time())}_{len(self.long_term_memories[category]) + 1}"
            memory = Memory(content, MemoryType.LONG_TERM, tags)
            memory.importance = importance
            
            # Almacenar memoria
            self.long_term_memories[category][memory_id] = memory
            
            # Actualizar índices
            if tags:
                for tag in tags:
                    if tag not in self.tag_index:
                        self.tag_index[tag] = set()
                    self.tag_index[tag].add(memory_id)
                    
            # Verificar límite de capacidad
            if len(self.long_term_memories[category]) > self.long_term_capacity:
                await self._prune_long_term_memories(category)
                
            return True
        except Exception as e:
            logger.error(f"Error al añadir memoria a largo plazo: {e}")
            return False
    
    async def get_short_term_memories(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener memorias a corto plazo de una categoría.
        
        Args:
            category: Categoría a consultar
            limit: Número máximo de memorias a devolver
            
        Returns:
            Lista de memorias
        """
        if category not in self.short_term_memories:
            return []
            
        memories = list(self.short_term_memories[category].values())
        
        # Ordenar por más recientes primero
        memories.sort(key=lambda m: m.created_at, reverse=True)
        
        # Acceder a las memorias
        for memory in memories[:limit]:
            memory.access()
            
        return [memory.to_dict() for memory in memories[:limit]]
    
    async def get_long_term_memories(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener memorias a largo plazo de una categoría.
        
        Args:
            category: Categoría a consultar
            limit: Número máximo de memorias a devolver
            
        Returns:
            Lista de memorias
        """
        if category not in self.long_term_memories:
            return []
            
        memories = list(self.long_term_memories[category].values())
        
        # Ordenar por importancia y recientes
        memories.sort(key=lambda m: (m.importance, m.last_accessed), reverse=True)
        
        # Acceder a las memorias
        for memory in memories[:limit]:
            memory.access()
            
        return [memory.to_dict() for memory in memories[:limit]]
    
    async def search_memories_by_tags(self, tags: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Buscar memorias por etiquetas.
        
        Args:
            tags: Lista de etiquetas a buscar
            limit: Número máximo de resultados
            
        Returns:
            Lista de memorias
        """
        if not tags:
            return []
            
        # Encontrar memorias que coincidan con todas las etiquetas
        memory_ids = None
        
        for tag in tags:
            if tag in self.tag_index:
                if memory_ids is None:
                    memory_ids = self.tag_index[tag].copy()
                else:
                    memory_ids &= self.tag_index[tag]
            else:
                # Si una etiqueta no existe, no hay coincidencias
                return []
                
        if not memory_ids:
            return []
            
        # Recopilar las memorias
        results = []
        
        for memory_id in memory_ids:
            category = memory_id.split('_')[0]
            
            if memory_id in self.short_term_memories.get(category, {}):
                memory = self.short_term_memories[category][memory_id]
                memory.access()
                results.append(memory.to_dict())
                
            elif memory_id in self.long_term_memories.get(category, {}):
                memory = self.long_term_memories[category][memory_id]
                memory.access()
                results.append(memory.to_dict())
                
        # Ordenar por recientes
        results.sort(key=lambda m: m["last_accessed"], reverse=True)
        
        return results[:limit]
    
    async def save_persistent_memory(self, key: str, value: Any) -> bool:
        """
        Guardar memoria persistente.
        
        Args:
            key: Clave de la memoria
            value: Valor a guardar
            
        Returns:
            True si se guardó correctamente
        """
        try:
            self.persistent_memories[key] = value
            await self._save_persistent_memories()
            return True
        except Exception as e:
            logger.error(f"Error al guardar memoria persistente: {e}")
            return False
    
    async def get_persistent_memory(self, key: str) -> Optional[Any]:
        """
        Obtener memoria persistente.
        
        Args:
            key: Clave de la memoria
            
        Returns:
            Valor de la memoria o None si no existe
        """
        return self.persistent_memories.get(key)
    
    async def delete_persistent_memory(self, key: str) -> bool:
        """
        Eliminar memoria persistente.
        
        Args:
            key: Clave de la memoria
            
        Returns:
            True si se eliminó correctamente
        """
        if key in self.persistent_memories:
            del self.persistent_memories[key]
            await self._save_persistent_memories()
            return True
        return False
    
    async def _prune_short_term_memories(self, category: str) -> None:
        """
        Eliminar memorias a corto plazo menos relevantes.
        
        Args:
            category: Categoría a limpiar
        """
        if category not in self.short_term_memories:
            return
            
        memories = list(self.short_term_memories[category].items())
        
        # Ordenar por menos accedidas y más antiguas primero
        memories.sort(key=lambda x: (x[1].access_count, x[1].created_at))
        
        # Eliminar el 20% más antiguo y menos accedido
        prune_count = max(1, int(len(memories) * 0.2))
        for memory_id, _ in memories[:prune_count]:
            # Eliminar de los índices
            memory = self.short_term_memories[category][memory_id]
            for tag in memory.tags:
                if tag in self.tag_index and memory_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(memory_id)
                    
            # Eliminar la memoria
            del self.short_term_memories[category][memory_id]
            
        logger.debug(f"Limpiadas {prune_count} memorias a corto plazo de {category}")
    
    async def _prune_long_term_memories(self, category: str) -> None:
        """
        Eliminar memorias a largo plazo menos relevantes.
        
        Args:
            category: Categoría a limpiar
        """
        if category not in self.long_term_memories:
            return
            
        memories = list(self.long_term_memories[category].items())
        
        # Ordenar por importancia y accesos (menor primero)
        memories.sort(key=lambda x: (x[1].importance, x[1].access_count))
        
        # Eliminar el 10% menos importante
        prune_count = max(1, int(len(memories) * 0.1))
        for memory_id, _ in memories[:prune_count]:
            # Eliminar de los índices
            memory = self.long_term_memories[category][memory_id]
            for tag in memory.tags:
                if tag in self.tag_index and memory_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(memory_id)
                    
            # Eliminar la memoria
            del self.long_term_memories[category][memory_id]
            
        logger.debug(f"Limpiadas {prune_count} memorias a largo plazo de {category}")
    
    async def _consolidate_memories(self) -> None:
        """Consolidar memorias a corto plazo en largo plazo."""
        consolidation_count = 0
        
        for category, memories in self.short_term_memories.items():
            for memory_id, memory in list(memories.items()):
                # Verificar si la memoria cumple criterios de consolidación
                time_threshold = datetime.now() - self.consolidation_interval
                
                if (memory.access_count >= self.consolidation_threshold or
                    memory.last_accessed < time_threshold):
                    
                    # Trasladar a memoria a largo plazo
                    await self.add_long_term_memory(
                        category,
                        memory.content,
                        memory.tags,
                        min(0.7, 0.3 + (memory.access_count * 0.1))  # Calcular importancia
                    )
                    
                    # Eliminar de corto plazo
                    for tag in memory.tags:
                        if tag in self.tag_index and memory_id in self.tag_index[tag]:
                            self.tag_index[tag].remove(memory_id)
                            
                    del memories[memory_id]
                    consolidation_count += 1
                    
        if consolidation_count > 0:
            logger.info(f"Consolidadas {consolidation_count} memorias a largo plazo")
        
        self.last_consolidation = datetime.now()
    
    async def _scheduled_consolidation(self) -> None:
        """Tarea programada para consolidación periódica."""
        while True:
            try:
                # Esperar 1 hora entre consolidaciones
                await asyncio.sleep(3600)
                
                # Consolidar memorias
                await self._consolidate_memories()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en consolidación programada: {e}")
                await asyncio.sleep(300)  # Esperar 5 minutos en caso de error
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema de memoria.
        
        Returns:
            Diccionario con estadísticas
        """
        short_term_count = sum(len(memories) for memories in self.short_term_memories.values())
        long_term_count = sum(len(memories) for memories in self.long_term_memories.values())
        
        categories = {
            "short_term": list(self.short_term_memories.keys()),
            "long_term": list(self.long_term_memories.keys())
        }
        
        return {
            "short_term_count": short_term_count,
            "long_term_count": long_term_count,
            "persistent_count": len(self.persistent_memories),
            "tag_count": len(self.tag_index),
            "categories": categories,
            "last_consolidation": self.last_consolidation.isoformat()
        }
    
    async def shutdown(self) -> None:
        """Cerrar ordenadamente el sistema de memoria."""
        # Guardar memorias persistentes
        await self._save_persistent_memories()
        
        # Consolidar memorias pendientes
        await self._consolidate_memories()
        
        logger.info("MemorySystem cerrado correctamente")