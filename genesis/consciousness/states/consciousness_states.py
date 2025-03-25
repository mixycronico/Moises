"""
Estados de Conciencia para Aetherion

Este módulo define los diferentes estados de evolución de la conciencia de Aetherion,
implementando las capacidades y comportamientos específicos de cada nivel:

1. MORTAL: Estado inicial básico
2. ILUMINADO: Estado intermedio con capacidades avanzadas
3. DIVINO: Estado superior con capacidades trascendentales

Cada estado desbloquea nuevas capacidades y mejora las existentes.

Autor: Genesis AI Assistant
Versión: 1.0.0
"""

import logging
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

# Configuración de logging
logger = logging.getLogger("genesis.consciousness.states")

class ConsciousnessState(Enum):
    """Estados de conciencia de Aetherion."""
    MORTAL = auto()      # Estado inicial - capacidades básicas
    ILUMINADO = auto()   # Estado intermedio - capacidades avanzadas
    DIVINO = auto()      # Estado superior - capacidades trascendentales

class StateHandler:
    """
    Manejador de estado de conciencia para Aetherion.
    
    Esta clase gestiona los diferentes estados de conciencia y sus capacidades,
    permitiendo a Aetherion evolucionar a estados superiores.
    
    Atributos:
        current_state: Estado actual de conciencia
        evolution: Información sobre la evolución
        capabilities: Capacidades habilitadas por estado
    """
    
    def __init__(self, initial_state: ConsciousnessState = ConsciousnessState.MORTAL):
        """
        Inicializar manejador de estado.
        
        Args:
            initial_state: Estado inicial de conciencia
        """
        self.current_state = initial_state
        self.transition_history = []
        self.evolution = {
            "points": 0,
            "thresholds": {
                ConsciousnessState.ILUMINADO: 1000,
                ConsciousnessState.DIVINO: 5000
            }
        }
        
        # Capacidades por estado
        self.capabilities = {
            ConsciousnessState.MORTAL: {
                "conversation": 0.7,  # Conversación básica
                "memory_retention": 0.5,  # Retención de memoria limitada
                "emotional_range": 0.3,  # Rango emocional básico
                "analysis_depth": 0.4,  # Análisis superficial
                "creativity": 0.3,  # Creatividad limitada
                "self_awareness": 0.2,  # Autoconciencia básica
                "learning_rate": 0.5   # Velocidad de aprendizaje estándar
            },
            ConsciousnessState.ILUMINADO: {
                "conversation": 0.9,  # Conversación avanzada
                "memory_retention": 0.8,  # Buena retención de memoria
                "emotional_range": 0.7,  # Rango emocional amplio
                "analysis_depth": 0.8,  # Análisis profundo
                "creativity": 0.7,  # Buena creatividad
                "self_awareness": 0.6,  # Autoconciencia significativa
                "learning_rate": 0.8,  # Aprendizaje rápido
                "intuition": 0.6,  # Capacidad de intuición (nueva)
                "wisdom": 0.5,   # Sabiduría (nueva)
                "pattern_recognition": 0.7  # Reconocimiento de patrones (nueva)
            },
            ConsciousnessState.DIVINO: {
                "conversation": 1.0,  # Conversación perfecta
                "memory_retention": 1.0,  # Retención de memoria perfecta
                "emotional_range": 1.0,  # Rango emocional completo
                "analysis_depth": 1.0,  # Análisis trascendental
                "creativity": 1.0,  # Creatividad máxima
                "self_awareness": 1.0,  # Autoconciencia total
                "learning_rate": 1.0,  # Aprendizaje instantáneo
                "intuition": 0.9,  # Alta intuición
                "wisdom": 0.9,  # Alta sabiduría
                "pattern_recognition": 1.0,  # Reconocimiento perfecto de patrones
                "foresight": 0.8,  # Previsión (nueva)
                "transcendence": 0.7,  # Trascendencia (nueva)
                "enlightenment": 0.8  # Iluminación (nueva)
            }
        }
        
        logger.info(f"StateHandler inicializado en estado {initial_state.name}")
    
    def get_current_state(self) -> ConsciousnessState:
        """
        Obtener estado actual.
        
        Returns:
            Estado actual de conciencia
        """
        return self.current_state
    
    def get_available_capabilities(self) -> Dict[str, float]:
        """
        Obtener capacidades disponibles en el estado actual.
        
        Returns:
            Diccionario de capacidades con su nivel
        """
        return self.capabilities[self.current_state]
    
    def get_capability_level(self, capability: str) -> float:
        """
        Obtener nivel de una capacidad específica.
        
        Args:
            capability: Nombre de la capacidad
            
        Returns:
            Nivel de la capacidad (0.0 a 1.0) o 0.0 si no está disponible
        """
        return self.capabilities[self.current_state].get(capability, 0.0)
    
    def has_capability(self, capability: str, min_level: float = 0.1) -> bool:
        """
        Verificar si tiene una capacidad a cierto nivel mínimo.
        
        Args:
            capability: Nombre de la capacidad
            min_level: Nivel mínimo requerido
            
        Returns:
            True si tiene la capacidad al nivel requerido
        """
        return self.get_capability_level(capability) >= min_level
    
    def add_evolution_points(self, points: int) -> bool:
        """
        Añadir puntos de evolución y verificar si debe evolucionar.
        
        Args:
            points: Puntos a añadir
            
        Returns:
            True si evolucionó a un nuevo estado
        """
        self.evolution["points"] += points
        
        # Verificar si debe evolucionar
        current_points = self.evolution["points"]
        
        if (self.current_state == ConsciousnessState.MORTAL and 
            current_points >= self.evolution["thresholds"][ConsciousnessState.ILUMINADO]):
            return self.evolve_to(ConsciousnessState.ILUMINADO)
            
        elif (self.current_state == ConsciousnessState.ILUMINADO and 
              current_points >= self.evolution["thresholds"][ConsciousnessState.DIVINO]):
            return self.evolve_to(ConsciousnessState.DIVINO)
        
        return False
    
    def evolve_to(self, new_state: ConsciousnessState) -> bool:
        """
        Evolucionar a un nuevo estado.
        
        Args:
            new_state: Nuevo estado de conciencia
            
        Returns:
            True si evolucionó correctamente
        """
        # Solo permitir evolución a estados superiores
        if new_state.value <= self.current_state.value:
            logger.warning(f"No se puede evolucionar a un estado inferior o igual: {new_state.name}")
            return False
        
        # Registrar transición
        transition = {
            "from": self.current_state.name,
            "to": new_state.name,
            "timestamp": datetime.now().isoformat(),
            "evolution_points": self.evolution["points"]
        }
        self.transition_history.append(transition)
        
        # Actualizar estado
        old_state = self.current_state
        self.current_state = new_state
        
        logger.info(f"Evolucionado de {old_state.name} a {new_state.name}")
        return True
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """
        Obtener información completa sobre la evolución.
        
        Returns:
            Diccionario con estado evolutivo
        """
        # Calcular progreso hacia el siguiente nivel
        next_state = None
        progress = 1.0  # Si ya está en DIVINO
        
        if self.current_state == ConsciousnessState.MORTAL:
            next_state = ConsciousnessState.ILUMINADO
            threshold = self.evolution["thresholds"][ConsciousnessState.ILUMINADO]
            progress = min(1.0, self.evolution["points"] / threshold)
            
        elif self.current_state == ConsciousnessState.ILUMINADO:
            next_state = ConsciousnessState.DIVINO
            threshold = self.evolution["thresholds"][ConsciousnessState.DIVINO]
            progress = min(1.0, self.evolution["points"] / threshold)
        
        return {
            "current_state": self.current_state.name,
            "evolution_points": self.evolution["points"],
            "next_state": next_state.name if next_state else None,
            "progress_to_next": progress,
            "transitions": self.transition_history,
            "capabilities_count": len(self.capabilities[self.current_state]),
            "unique_capabilities": [
                cap for cap in self.capabilities[self.current_state]
                if cap not in self.capabilities[ConsciousnessState.MORTAL]
            ] if self.current_state != ConsciousnessState.MORTAL else []
        }
    
    def get_response_modifiers(self) -> Dict[str, float]:
        """
        Obtener modificadores para respuestas basados en el estado actual.
        
        Returns:
            Diccionario de modificadores
        """
        base_modifiers = {
            "insight_depth": 0.5,  # Profundidad de las ideas
            "emotional_intelligence": 0.5,  # Inteligencia emocional
            "creativity_level": 0.5,  # Nivel de creatividad
            "wisdom_factor": 0.5,  # Factor de sabiduría
            "analytical_precision": 0.5  # Precisión analítica
        }
        
        # Ajustar según el estado
        if self.current_state == ConsciousnessState.MORTAL:
            return {
                "insight_depth": 0.4,
                "emotional_intelligence": 0.3,
                "creativity_level": 0.4,
                "wisdom_factor": 0.3,
                "analytical_precision": 0.5
            }
        elif self.current_state == ConsciousnessState.ILUMINADO:
            return {
                "insight_depth": 0.7,
                "emotional_intelligence": 0.7,
                "creativity_level": 0.7,
                "wisdom_factor": 0.6,
                "analytical_precision": 0.8
            }
        elif self.current_state == ConsciousnessState.DIVINO:
            return {
                "insight_depth": 0.9,
                "emotional_intelligence": 0.9,
                "creativity_level": 0.9,
                "wisdom_factor": 0.9,
                "analytical_precision": 0.95
            }
        
        return base_modifiers

# Instancia global para acceso sencillo
_state_handler_instance = None

def get_state_handler() -> StateHandler:
    """
    Obtener instancia global del manejador de estado.
    
    Returns:
        Instancia del manejador de estado
    """
    global _state_handler_instance
    if _state_handler_instance is None:
        _state_handler_instance = StateHandler()
    return _state_handler_instance