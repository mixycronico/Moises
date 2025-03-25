"""
Sistema de Memoria para Aetherion.

Este módulo implementa el sistema de memoria para Aetherion, permitiendo
almacenar y recuperar interacciones, datos de entrenamiento y experiencias
que forman parte de la evolución consciente de Aetherion.
"""

import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple
import json

# Configurar logging
logger = logging.getLogger(__name__)

class MemorySystem:
    """
    Sistema de memoria para Aetherion.
    
    Proporciona:
    - Memoria a corto plazo (interacciones recientes)
    - Memoria a largo plazo (conocimiento persistente)
    - Experiencias significativas (momentos clave)
    """
    
    def __init__(self, max_short_term_size: int = 100):
        """
        Inicializar sistema de memoria.
        
        Args:
            max_short_term_size: Tamaño máximo de la memoria a corto plazo
        """
        # Memoria a corto plazo (interacciones recientes)
        self.short_term_memory = []
        self.max_short_term_size = max_short_term_size
        
        # Memoria a largo plazo (conocimiento persistente)
        self.long_term_memory = {}
        
        # Experiencias significativas (momentos clave)
        self.significant_experiences = []
        
        # Estadísticas
        self.stats = {
            "total_interactions": 0,
            "last_interaction": None,
            "categories": {}
        }
        
        logger.info("MemorySystem inicializado")
    
    def store_interaction(self, input_text: str, response: str, channel: str = 'API', 
                         context: Dict[str, Any] = None) -> bool:
        """
        Almacenar una interacción en la memoria a corto plazo.
        
        Args:
            input_text: Texto de entrada recibido
            response: Respuesta generada
            channel: Canal de comunicación
            context: Contexto adicional
            
        Returns:
            True si se almacenó correctamente
        """
        if context is None:
            context = {}
        
        # Actualizar estadísticas
        self.stats["total_interactions"] += 1
        self.stats["last_interaction"] = datetime.datetime.now().isoformat()
        
        # Crear registro de interacción
        interaction = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input": input_text,
            "response": response,
            "channel": channel,
            "context": context
        }
        
        # Añadir a memoria a corto plazo
        self.short_term_memory.append(interaction)
        
        # Mantener tamaño máximo
        if len(self.short_term_memory) > self.max_short_term_size:
            self.short_term_memory.pop(0)
        
        # Comprobar si es una experiencia significativa
        if self._is_significant_experience(input_text, response, context):
            self._store_significant_experience(interaction)
        
        # Categorizar interacción
        self._categorize_interaction(input_text, response)
        
        return True
    
    def _is_significant_experience(self, input_text: str, response: str, 
                                 context: Dict[str, Any]) -> bool:
        """
        Determinar si una interacción constituye una experiencia significativa.
        
        Args:
            input_text: Texto de entrada
            response: Respuesta generada
            context: Contexto adicional
            
        Returns:
            True si es una experiencia significativa
        """
        # Por defecto, las primeras interacciones son significativas
        if self.stats["total_interactions"] <= 10:
            return True
        
        # Interacciones con temas específicos
        significant_topics = ["aetherion", "consciencia", "evolución", "cosmos"]
        for topic in significant_topics:
            if topic in input_text.lower():
                return True
        
        # Interacciones con contexto especial
        if context and context.get("is_important", False):
            return True
        
        # Por defecto, no es significativa
        return False
    
    def _store_significant_experience(self, interaction: Dict[str, Any]) -> None:
        """
        Almacenar una experiencia significativa.
        
        Args:
            interaction: Datos de la interacción
        """
        # Añadir a experiencias significativas
        self.significant_experiences.append(interaction)
        
        logger.info(f"Experiencia significativa almacenada: {interaction['input'][:30]}...")
    
    def _categorize_interaction(self, input_text: str, response: str) -> None:
        """
        Categorizar una interacción para estadísticas.
        
        Args:
            input_text: Texto de entrada
            response: Respuesta generada
        """
        # Definir categorías
        categories = {
            "crypto": ["bitcoin", "ethereum", "cripto", "btc", "eth"],
            "trading": ["trading", "operar", "inversion", "estrategia"],
            "mercado": ["mercado", "tendencia", "precio", "volatilidad"],
            "filosofico": ["conciencia", "filosofia", "existencia", "sentido"]
        }
        
        # Categorizar
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in input_text.lower():
                    if category not in self.stats["categories"]:
                        self.stats["categories"][category] = 0
                    
                    self.stats["categories"][category] += 1
                    break
    
    def store_knowledge(self, key: str, value: Any, source: str = "system") -> bool:
        """
        Almacenar conocimiento en la memoria a largo plazo.
        
        Args:
            key: Clave para identificar el conocimiento
            value: Valor a almacenar
            source: Fuente del conocimiento
            
        Returns:
            True si se almacenó correctamente
        """
        # Validar clave
        if not key:
            return False
        
        # Crear entrada
        entry = {
            "value": value,
            "source": source,
            "timestamp": datetime.datetime.now().isoformat(),
            "access_count": 0
        }
        
        # Almacenar en memoria a largo plazo
        self.long_term_memory[key] = entry
        
        return True
    
    def retrieve_knowledge(self, key: str) -> Optional[Any]:
        """
        Recuperar conocimiento de la memoria a largo plazo.
        
        Args:
            key: Clave del conocimiento
            
        Returns:
            Valor almacenado o None si no existe
        """
        # Comprobar si existe
        if key not in self.long_term_memory:
            return None
        
        # Actualizar contador de accesos
        self.long_term_memory[key]["access_count"] += 1
        
        # Devolver valor
        return self.long_term_memory[key]["value"]
    
    def get_recent_interactions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Obtener interacciones recientes.
        
        Args:
            limit: Número máximo de interacciones a devolver
            
        Returns:
            Lista de interacciones recientes
        """
        # Limitar cantidad
        limit = min(limit, len(self.short_term_memory))
        
        # Devolver las más recientes
        return self.short_term_memory[-limit:]
    
    def get_significant_experiences(self) -> List[Dict[str, Any]]:
        """
        Obtener experiencias significativas.
        
        Returns:
            Lista de experiencias significativas
        """
        return self.significant_experiences
    
    def clear_short_term_memory(self) -> bool:
        """
        Limpiar memoria a corto plazo.
        
        Returns:
            True si se limpió correctamente
        """
        self.short_term_memory = []
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de memoria.
        
        Returns:
            Estadísticas de memoria
        """
        return {
            "short_term_size": len(self.short_term_memory),
            "long_term_size": len(self.long_term_memory),
            "significant_experiences": len(self.significant_experiences),
            "total_interactions": self.stats["total_interactions"],
            "last_interaction": self.stats["last_interaction"],
            "categories": self.stats["categories"]
        }
    
    def search_interactions(self, query: str) -> List[Dict[str, Any]]:
        """
        Buscar interacciones que contengan la consulta.
        
        Args:
            query: Texto a buscar
            
        Returns:
            Lista de interacciones que coinciden
        """
        # Normalizar consulta
        query = query.lower()
        
        # Buscar en memoria a corto plazo
        results = []
        
        for interaction in self.short_term_memory:
            if query in interaction["input"].lower() or query in interaction["response"].lower():
                results.append(interaction)
        
        return results