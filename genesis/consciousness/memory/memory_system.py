"""
Sistema de Memoria para Aetherion.

Este módulo implementa el sistema de memoria para Aetherion, permitiendo
almacenar y recuperar recuerdos a corto y largo plazo, con capacidades
de búsqueda semántica, etiquetado y evolución temporal.
"""

import logging
import json
import datetime
import uuid
import os
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field, asdict

# Configurar logging
logger = logging.getLogger(__name__)

# Directorio para almacenamiento persistente
MEMORY_DIR = os.path.join("data", "aetherion", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

class MemoryType(Enum):
    """Tipos de memoria para Aetherion."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"
    PROCEDURAL = "procedural"

@dataclass
class Memory:
    """Representación de una memoria individual."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    memory_type: MemoryType = MemoryType.SHORT_TERM
    importance: float = 0.5
    creation_time: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_access_time: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    related_memories: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        memory_dict = asdict(self)
        memory_dict["memory_type"] = self.memory_type.value
        return memory_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Crear desde diccionario."""
        # Convertir tipo de memoria a enum
        if "memory_type" in data and isinstance(data["memory_type"], str):
            data["memory_type"] = MemoryType(data["memory_type"])
        
        return cls(**data)
    
    def update_access(self) -> None:
        """Actualizar tiempo de acceso y contador."""
        self.last_access_time = datetime.datetime.now().isoformat()
        self.access_count += 1

class MemorySystem:
    """
    Sistema de memoria para Aetherion.
    
    Proporciona almacenamiento y recuperación de memoria a corto y largo plazo,
    con capacidades de búsqueda, etiquetado y evolución temporal.
    """
    
    def __init__(self):
        """Inicializar sistema de memoria."""
        # Memorias por tipo
        self._memories: Dict[MemoryType, List[Memory]] = {
            memory_type: [] for memory_type in MemoryType
        }
        
        # Capacidades máximas por tipo de memoria
        self._capacities: Dict[MemoryType, int] = {
            MemoryType.SHORT_TERM: 100,   # Capacidad limitada
            MemoryType.LONG_TERM: 10000,  # Gran capacidad
            MemoryType.EPISODIC: 1000,    # Eventos específicos
            MemoryType.SEMANTIC: 5000,    # Conocimiento general
            MemoryType.EMOTIONAL: 500,    # Respuestas emocionales
            MemoryType.PROCEDURAL: 200    # Procesos y procedimientos
        }
        
        # Cargar memorias persistentes
        self._load_memories()
        
        logger.info(f"MemorySystem inicializado con {sum(len(mems) for mems in self._memories.values())} memorias")
    
    def _load_memories(self) -> None:
        """Cargar memorias desde almacenamiento persistente."""
        try:
            # Cargar archivo de memoria a largo plazo
            long_term_file = os.path.join(MEMORY_DIR, "long_term_memories.json")
            if os.path.exists(long_term_file):
                with open(long_term_file, 'r', encoding='utf-8') as f:
                    memories_data = json.load(f)
                    for memory_data in memories_data:
                        try:
                            memory = Memory.from_dict(memory_data)
                            self._memories[memory.memory_type].append(memory)
                        except Exception as e:
                            logger.error(f"Error al cargar memoria: {e}")
                
                logger.info(f"Cargadas {len(memories_data)} memorias a largo plazo")
        except Exception as e:
            logger.error(f"Error al cargar memorias: {e}")
    
    def _save_memories(self) -> None:
        """Guardar memorias a almacenamiento persistente."""
        try:
            # Guardar memorias a largo plazo
            long_term_memories = []
            for memory_type in [MemoryType.LONG_TERM, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
                long_term_memories.extend([m.to_dict() for m in self._memories[memory_type]])
            
            long_term_file = os.path.join(MEMORY_DIR, "long_term_memories.json")
            with open(long_term_file, 'w', encoding='utf-8') as f:
                json.dump(long_term_memories, f, indent=2)
            
            logger.info(f"Guardadas {len(long_term_memories)} memorias a largo plazo")
        except Exception as e:
            logger.error(f"Error al guardar memorias: {e}")
    
    def store_memory(self, 
                    content: Dict[str, Any], 
                    memory_type: MemoryType = MemoryType.SHORT_TERM,
                    importance: float = 0.5,
                    tags: Optional[List[str]] = None) -> Memory:
        """
        Almacenar un nuevo recuerdo.
        
        Args:
            content: Contenido de la memoria
            memory_type: Tipo de memoria
            importance: Importancia (0.0-1.0)
            tags: Etiquetas para categorización
        
        Returns:
            Memoria creada
        """
        # Crear nueva memoria
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or []
        )
        
        # Gestionar capacidad
        self._manage_capacity(memory_type)
        
        # Añadir a la lista
        self._memories[memory_type].append(memory)
        
        # Guardar si es memoria a largo plazo
        if memory_type in [MemoryType.LONG_TERM, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            self._save_memories()
        
        logger.debug(f"Memoria almacenada: {memory.id} ({memory_type.value})")
        
        # Registrar para evolución de consciencia
        from genesis.consciousness.states.consciousness_states import get_consciousness_states
        states = get_consciousness_states()
        states.record_activity("memory_accesses")
        
        return memory
    
    def _manage_capacity(self, memory_type: MemoryType) -> None:
        """
        Gestionar capacidad de un tipo de memoria.
        
        Args:
            memory_type: Tipo de memoria a gestionar
        """
        memories = self._memories[memory_type]
        capacity = self._capacities[memory_type]
        
        # Si estamos en el límite, eliminar las menos importantes/accesadas
        if len(memories) >= capacity:
            # Ordenar por importancia y accesos
            memories.sort(key=lambda m: (m.importance, m.access_count))
            
            # Eliminar la menos importante
            removed = memories.pop(0)
            
            # Intentar transferir a largo plazo si es relevante
            if memory_type == MemoryType.SHORT_TERM and removed.importance > 0.7:
                removed.memory_type = MemoryType.LONG_TERM
                self._memories[MemoryType.LONG_TERM].append(removed)
                logger.debug(f"Memoria transferida a largo plazo: {removed.id}")
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Obtener memoria por ID.
        
        Args:
            memory_id: ID de la memoria
        
        Returns:
            Memoria encontrada o None
        """
        # Buscar en todos los tipos
        for memories in self._memories.values():
            for memory in memories:
                if memory.id == memory_id:
                    memory.update_access()
                    return memory
        
        return None
    
    def update_memory(self, 
                     memory_id: str, 
                     content: Optional[Dict[str, Any]] = None,
                     importance: Optional[float] = None,
                     memory_type: Optional[MemoryType] = None,
                     tags: Optional[List[str]] = None) -> Optional[Memory]:
        """
        Actualizar una memoria existente.
        
        Args:
            memory_id: ID de la memoria
            content: Nuevo contenido (opcional)
            importance: Nueva importancia (opcional)
            memory_type: Nuevo tipo (opcional)
            tags: Nuevas etiquetas (opcional)
        
        Returns:
            Memoria actualizada o None
        """
        memory = self.get_memory(memory_id)
        
        if not memory:
            return None
        
        # Actualizar campos
        if content is not None:
            memory.content = content
        
        if importance is not None:
            memory.importance = importance
        
        if memory_type is not None and memory_type != memory.memory_type:
            # Eliminar de la lista antigua
            self._memories[memory.memory_type] = [
                m for m in self._memories[memory.memory_type] if m.id != memory_id
            ]
            
            # Cambiar tipo
            memory.memory_type = memory_type
            
            # Añadir a la nueva lista
            self._memories[memory_type].append(memory)
        
        if tags is not None:
            memory.tags = tags
        
        # Guardar si es memoria a largo plazo
        if memory.memory_type in [MemoryType.LONG_TERM, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            self._save_memories()
        
        return memory
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Eliminar una memoria.
        
        Args:
            memory_id: ID de la memoria
        
        Returns:
            True si se eliminó correctamente
        """
        for memory_type, memories in self._memories.items():
            for i, memory in enumerate(memories):
                if memory.id == memory_id:
                    memories.pop(i)
                    
                    # Guardar si es memoria a largo plazo
                    if memory_type in [MemoryType.LONG_TERM, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
                        self._save_memories()
                    
                    return True
        
        return False
    
    def search_memories(self, 
                       query: str, 
                       memory_type: Optional[MemoryType] = None,
                       tags: Optional[List[str]] = None,
                       limit: int = 10) -> List[Memory]:
        """
        Buscar memorias por contenido.
        
        Args:
            query: Texto a buscar
            memory_type: Tipo de memoria (opcional)
            tags: Etiquetas a filtrar (opcional)
            limit: Límite de resultados
        
        Returns:
            Lista de memorias que coinciden
        """
        results = []
        query = query.lower()
        
        # Determinar tipos a buscar
        types_to_search = [memory_type] if memory_type else list(MemoryType)
        
        for m_type in types_to_search:
            for memory in self._memories[m_type]:
                # Verificar contenido
                match = False
                content_str = json.dumps(memory.content, default=str).lower()
                
                if query in content_str:
                    match = True
                
                # Verificar etiquetas si se especificaron
                if tags and not all(tag in memory.tags for tag in tags):
                    match = False
                
                if match:
                    memory.update_access()
                    results.append(memory)
                    
                    if len(results) >= limit:
                        return results
        
        return results
    
    def search_memories_by_tags(self, tags: List[str], limit: int = 10) -> List[Memory]:
        """
        Buscar memorias por etiquetas.
        
        Args:
            tags: Etiquetas a buscar
            limit: Límite de resultados
        
        Returns:
            Lista de memorias con las etiquetas
        """
        results = []
        
        for memories in self._memories.values():
            for memory in memories:
                if all(tag in memory.tags for tag in tags):
                    memory.update_access()
                    results.append(memory)
                    
                    if len(results) >= limit:
                        return results
        
        return results
    
    def get_memories_by_type(self, memory_type: MemoryType, limit: int = 50) -> List[Memory]:
        """
        Obtener memorias de un tipo específico.
        
        Args:
            memory_type: Tipo de memoria
            limit: Límite de resultados
        
        Returns:
            Lista de memorias del tipo especificado
        """
        # Obtener memorias ordenadas por importancia y recientes
        memories = sorted(
            self._memories[memory_type], 
            key=lambda m: (m.importance, m.last_access_time),
            reverse=True
        )
        
        # Actualizar acceso
        for memory in memories[:limit]:
            memory.update_access()
        
        return memories[:limit]
    
    def consolidate_memories(self) -> int:
        """
        Consolidar memorias a corto plazo en memorias a largo plazo.
        
        Returns:
            Número de memorias consolidadas
        """
        consolidated_count = 0
        
        # Obtener memorias a corto plazo importantes
        short_term = self._memories[MemoryType.SHORT_TERM]
        to_consolidate = [m for m in short_term if m.importance >= 0.7]
        
        for memory in to_consolidate:
            # Cambiar tipo y acceso
            memory.memory_type = MemoryType.LONG_TERM
            memory.update_access()
            
            # Mover a largo plazo
            self._memories[MemoryType.LONG_TERM].append(memory)
            consolidated_count += 1
        
        # Eliminar de corto plazo
        if consolidated_count > 0:
            self._memories[MemoryType.SHORT_TERM] = [
                m for m in short_term if m.importance < 0.7
            ]
            
            # Guardar cambios
            self._save_memories()
            
            logger.info(f"Consolidadas {consolidated_count} memorias a largo plazo")
        
        return consolidated_count
    
    def relate_memories(self, memory_id: str, related_ids: List[str]) -> bool:
        """
        Relacionar memorias entre sí.
        
        Args:
            memory_id: ID de la memoria principal
            related_ids: IDs de memorias relacionadas
        
        Returns:
            True si se relacionaron correctamente
        """
        memory = self.get_memory(memory_id)
        
        if not memory:
            return False
        
        # Verificar que existan todas las memorias relacionadas
        for related_id in related_ids:
            if related_id != memory_id and self.get_memory(related_id):
                if related_id not in memory.related_memories:
                    memory.related_memories.append(related_id)
        
        # Guardar si es memoria a largo plazo
        if memory.memory_type in [MemoryType.LONG_TERM, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            self._save_memories()
        
        return True
    
    def get_related_memories(self, memory_id: str, limit: int = 10) -> List[Memory]:
        """
        Obtener memorias relacionadas.
        
        Args:
            memory_id: ID de la memoria principal
            limit: Límite de resultados
        
        Returns:
            Lista de memorias relacionadas
        """
        memory = self.get_memory(memory_id)
        
        if not memory:
            return []
        
        related = []
        for related_id in memory.related_memories:
            related_memory = self.get_memory(related_id)
            if related_memory:
                related.append(related_memory)
                
                if len(related) >= limit:
                    break
        
        return related
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema de memoria.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "total_memories": sum(len(mems) for mems in self._memories.values()),
            "by_type": {memory_type.value: len(mems) for memory_type, mems in self._memories.items()},
            "capacities": {memory_type.value: capacity for memory_type, capacity in self._capacities.items()}
        }
        
        return stats

# Instancia global para acceso conveniente
_memory_system = None

def get_memory_system() -> MemorySystem:
    """
    Obtener instancia global del sistema de memoria.
    
    Returns:
        Instancia del sistema de memoria
    """
    global _memory_system
    
    if _memory_system is None:
        _memory_system = MemorySystem()
    
    return _memory_system