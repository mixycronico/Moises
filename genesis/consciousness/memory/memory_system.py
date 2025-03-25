"""
Sistema de Memoria para Aetherion

Este módulo implementa un sistema de memoria avanzado para Aetherion,
permitiéndole almacenar y recuperar experiencias, conocimientos y conversaciones.

La memoria se divide en cuatro tipos principales:
1. Memoria a corto plazo: Eventos recientes, conversaciones actuales
2. Memoria a largo plazo: Conocimientos permanentes, patrones identificados
3. Memoria episódica: Experiencias concretas con contexto temporal
4. Memoria experiencial: Aprendizajes derivados de la experiencia

Autor: Genesis AI Assistant
Versión: 1.0.0
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import aiofiles

# Configuración de logging
logger = logging.getLogger("genesis.consciousness.memory")

class MemorySystem:
    """
    Sistema de memoria para Aetherion.
    
    Implementa almacenamiento y recuperación de diferentes tipos de memoria,
    con persistencia opcional para datos importantes.
    
    Atributos:
        memories: Contenedores para cada tipo de memoria
        initialized: Estado de inicialización
        config: Configuración del sistema
    """
    
    def __init__(self):
        """Inicializar sistema de memoria."""
        self.memories = {
            "short_term": [],
            "long_term": {},
            "episodic": [],
            "experiential": {}
        }
        self.initialized = False
        self.memory_path = os.path.join(os.path.dirname(__file__), 'data')
        self.config = {
            "short_term_capacity": 100,
            "auto_persist": True,
            "importance_threshold": 0.7,
            "episodic_capacity": 500
        }
        logger.info("MemorySystem inicializado")
    
    async def initialize(self) -> bool:
        """
        Inicializar sistema de memoria cargando datos persistentes.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Asegurar que existe el directorio de datos
            os.makedirs(self.memory_path, exist_ok=True)
            
            # Cargar memorias persistentes
            await self._load_persistent_memories()
            
            self.initialized = True
            logger.info("MemorySystem inicializado completamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar MemorySystem: {str(e)}")
            return False
    
    async def _load_persistent_memories(self) -> None:
        """Cargar memorias desde almacenamiento persistente."""
        memory_files = {
            "short_term": "short_term.json",
            "long_term": "long_term.json",
            "episodic": "episodic.json",
            "experiential": "experiential.json"
        }
        
        for memory_type, filename in memory_files.items():
            file_path = os.path.join(self.memory_path, filename)
            
            # Si el archivo no existe, crearlo
            if not os.path.exists(file_path):
                async with aiofiles.open(file_path, 'w') as f:
                    if memory_type in ["short_term", "episodic"]:
                        await f.write(json.dumps([]))
                    else:
                        await f.write(json.dumps({}))
                continue
            
            # Cargar datos existentes
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    self.memories[memory_type] = json.loads(content)
                logger.info(f"Memoria {memory_type} cargada: {len(content)} bytes")
            except Exception as e:
                logger.error(f"Error al cargar memoria {memory_type}: {str(e)}")
    
    async def store_short_term(self, memory_item: Dict[str, Any]) -> int:
        """
        Almacenar ítem en memoria a corto plazo.
        
        Args:
            memory_item: Ítem a almacenar
            
        Returns:
            ID del ítem almacenado
        """
        # Asegurar inicialización
        if not self.initialized:
            await self.initialize()
        
        # Añadir timestamp si no tiene
        if "timestamp" not in memory_item:
            memory_item["timestamp"] = datetime.now().isoformat()
        
        # Añadir ID si no tiene
        if "id" not in memory_item:
            memory_item["id"] = len(self.memories["short_term"]) + 1
        
        # Almacenar en memoria a corto plazo
        self.memories["short_term"].append(memory_item)
        
        # Limitar tamaño de memoria a corto plazo
        if len(self.memories["short_term"]) > self.config.get("short_term_capacity", 100):
            self.memories["short_term"].pop(0)
        
        # Persistir si es necesario
        if self.config.get("auto_persist", True):
            await self._persist_memory("short_term")
        
        # Evaluar si debe almacenarse en memoria a largo plazo
        importance = memory_item.get("importance", 0.5)
        if importance >= self.config.get("importance_threshold", 0.7):
            await self.store_long_term(memory_item)
        
        return memory_item["id"]
    
    async def store_long_term(self, memory_item: Dict[str, Any]) -> str:
        """
        Almacenar ítem en memoria a largo plazo.
        
        Args:
            memory_item: Ítem a almacenar
            
        Returns:
            Categoría donde se almacenó
        """
        # Asegurar inicialización
        if not self.initialized:
            await self.initialize()
        
        # Determinar categoría
        category = memory_item.get("category", "general")
        
        # Crear categoría si no existe
        if category not in self.memories["long_term"]:
            self.memories["long_term"][category] = []
        
        # Añadir a la categoría
        self.memories["long_term"][category].append(memory_item)
        
        # Persistir si es necesario
        if self.config.get("auto_persist", True):
            await self._persist_memory("long_term")
        
        return category
    
    async def store_episodic(self, episode: Dict[str, Any]) -> int:
        """
        Almacenar episodio en memoria episódica.
        
        Args:
            episode: Episodio a almacenar
            
        Returns:
            ID del episodio almacenado
        """
        # Asegurar inicialización
        if not self.initialized:
            await self.initialize()
        
        # Añadir timestamp si no tiene
        if "timestamp" not in episode:
            episode["timestamp"] = datetime.now().isoformat()
        
        # Añadir ID si no tiene
        if "id" not in episode:
            episode["id"] = len(self.memories["episodic"]) + 1
        
        # Almacenar en memoria episódica
        self.memories["episodic"].append(episode)
        
        # Limitar tamaño de memoria episódica
        if len(self.memories["episodic"]) > self.config.get("episodic_capacity", 500):
            self.memories["episodic"].pop(0)
        
        # Persistir si es necesario
        if self.config.get("auto_persist", True):
            await self._persist_memory("episodic")
        
        return episode["id"]
    
    async def store_experiential(self, key: str, experience: Dict[str, Any]) -> None:
        """
        Almacenar experiencia en memoria experiencial.
        
        Args:
            key: Clave para identificar la experiencia
            experience: Datos de la experiencia
        """
        # Asegurar inicialización
        if not self.initialized:
            await self.initialize()
        
        # Añadir timestamp si no tiene
        if "timestamp" not in experience:
            experience["timestamp"] = datetime.now().isoformat()
        
        # Almacenar o actualizar en memoria experiencial
        self.memories["experiential"][key] = experience
        
        # Persistir si es necesario
        if self.config.get("auto_persist", True):
            await self._persist_memory("experiential")
    
    async def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener memorias recientes de corto plazo.
        
        Args:
            limit: Número máximo de memorias a retornar
            
        Returns:
            Lista de memorias recientes
        """
        # Asegurar inicialización
        if not self.initialized:
            await self.initialize()
        
        # Obtener memorias más recientes
        return self.memories["short_term"][-limit:]
    
    async def get_long_term_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Obtener memorias a largo plazo por categoría.
        
        Args:
            category: Categoría a consultar
            
        Returns:
            Lista de memorias de la categoría
        """
        # Asegurar inicialización
        if not self.initialized:
            await self.initialize()
        
        # Obtener memorias de la categoría
        return self.memories["long_term"].get(category, [])
    
    async def search_memories(self, query: str, memory_types: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Buscar en las memorias.
        
        Args:
            query: Texto a buscar
            memory_types: Tipos de memoria donde buscar, o None para todos
            
        Returns:
            Resultados agrupados por tipo de memoria
        """
        # Asegurar inicialización
        if not self.initialized:
            await self.initialize()
        
        if not memory_types:
            memory_types = ["short_term", "long_term", "episodic", "experiential"]
        
        results = {}
        
        # Buscar en memoria a corto plazo
        if "short_term" in memory_types:
            results["short_term"] = [
                item for item in self.memories["short_term"]
                if self._match_query(item, query)
            ]
        
        # Buscar en memoria a largo plazo
        if "long_term" in memory_types:
            results["long_term"] = []
            for category, items in self.memories["long_term"].items():
                matches = [item for item in items if self._match_query(item, query)]
                results["long_term"].extend(matches)
        
        # Buscar en memoria episódica
        if "episodic" in memory_types:
            results["episodic"] = [
                item for item in self.memories["episodic"]
                if self._match_query(item, query)
            ]
        
        # Buscar en memoria experiencial
        if "experiential" in memory_types:
            results["experiential"] = [
                {"key": key, **value}
                for key, value in self.memories["experiential"].items()
                if self._match_query(value, query) or query.lower() in key.lower()
            ]
        
        return results
    
    def _match_query(self, item: Dict[str, Any], query: str) -> bool:
        """
        Verificar si un ítem coincide con una consulta.
        
        Args:
            item: Ítem a verificar
            query: Consulta a buscar
            
        Returns:
            True si coincide
        """
        query_lower = query.lower()
        
        # Función recursiva para buscar en diccionarios anidados
        def search_dict(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, str) and query_lower in v.lower():
                        return True
                    elif isinstance(v, (dict, list)) and search_dict(v):
                        return True
            elif isinstance(d, list):
                for item in d:
                    if isinstance(item, (dict, list)) and search_dict(item):
                        return True
                    elif isinstance(item, str) and query_lower in item.lower():
                        return True
            return False
        
        return search_dict(item)
    
    async def _persist_memory(self, memory_type: str) -> None:
        """
        Persistir un tipo de memoria en disco.
        
        Args:
            memory_type: Tipo de memoria a persistir
        """
        file_path = os.path.join(self.memory_path, f"{memory_type}.json")
        
        try:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(self.memories[memory_type], indent=2))
            logger.debug(f"Memoria {memory_type} persistida correctamente")
        except Exception as e:
            logger.error(f"Error al persistir memoria {memory_type}: {str(e)}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de la memoria.
        
        Returns:
            Estadísticas de la memoria
        """
        # Asegurar inicialización
        if not self.initialized:
            await self.initialize()
        
        stats = {
            "short_term": {
                "count": len(self.memories["short_term"]),
                "capacity": self.config.get("short_term_capacity", 100)
            },
            "long_term": {
                "categories": list(self.memories["long_term"].keys()),
                "total_items": sum(len(items) for items in self.memories["long_term"].values())
            },
            "episodic": {
                "count": len(self.memories["episodic"]),
                "capacity": self.config.get("episodic_capacity", 500)
            },
            "experiential": {
                "count": len(self.memories["experiential"])
            }
        }
        
        return stats
    
    async def clear_memory(self, memory_type: str) -> bool:
        """
        Limpiar un tipo de memoria.
        
        Args:
            memory_type: Tipo de memoria a limpiar
            
        Returns:
            True si se limpió correctamente
        """
        # Asegurar inicialización
        if not self.initialized:
            await self.initialize()
        
        if memory_type not in self.memories:
            return False
        
        # Limpiar memoria
        if memory_type in ["short_term", "episodic"]:
            self.memories[memory_type] = []
        else:
            self.memories[memory_type] = {}
        
        # Persistir cambios
        await self._persist_memory(memory_type)
        
        return True

# Instancia global para acceso sencillo
_memory_system_instance = None

async def get_memory_system() -> MemorySystem:
    """
    Obtener instancia global del sistema de memoria.
    
    Returns:
        Instancia inicializada del sistema de memoria
    """
    global _memory_system_instance
    if _memory_system_instance is None:
        _memory_system_instance = MemorySystem()
        await _memory_system_instance.initialize()
    return _memory_system_instance