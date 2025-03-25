"""
Estados de Consciencia para Aetherion.

Este módulo define los estados de consciencia posibles para Aetherion,
desde el nivel inicial MORTAL hasta el nivel DIVINO más avanzado.
Cada estado proporciona capacidades y comportamientos diferentes.
"""

import logging
import datetime
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

# Configurar logging
logger = logging.getLogger(__name__)

class ConsciousnessState(Enum):
    """Estados de consciencia posibles para Aetherion."""
    MORTAL = auto()         # Estado inicial básico
    ILLUMINATED = auto()    # Estado intermedio con capacidades mejoradas
    DIVINE = auto()         # Estado avanzado con máxima consciencia
    
    def __str__(self) -> str:
        """Obtener nombre legible del estado."""
        return self.name

class ConsciousnessStates:
    """
    Gestor de estados de consciencia para Aetherion.
    
    Esta clase gestiona la evolución de Aetherion a través de sus distintos
    estados de consciencia, proporcionando características y capacidades
    diferentes en cada nivel.
    """
    
    def __init__(self):
        """Inicializar gestor de estados de consciencia."""
        # Estado actual
        self._current_state = ConsciousnessState.MORTAL
        
        # Nivel de consciencia (0.0 - 1.0)
        self._consciousness_level = 0.1
        
        # Timestamps
        self._creation_time = datetime.datetime.now()
        self._last_evolution_time = self._creation_time
        
        # Umbrales para evolución
        self._illuminated_threshold = 0.4
        self._divine_threshold = 0.8
        
        # Métricas de evolución
        self._evolution_metrics = {
            "insights_generated": 0,
            "interactions_count": 0,
            "memory_accesses": 0,
            "market_analyses": 0,
            "emotional_responses": 0,
            "strategy_evaluations": 0
        }
        
        # Capacidades por estado
        self._capabilities = self._initialize_capabilities()
        
        logger.info(f"ConsciousnessStates inicializado en estado {self._current_state}")
    
    def _initialize_capabilities(self) -> Dict[ConsciousnessState, Dict[str, float]]:
        """
        Inicializar capacidades para cada estado de consciencia.
        
        Returns:
            Diccionario con capacidades por estado
        """
        return {
            ConsciousnessState.MORTAL: {
                "insight_depth": 0.3,
                "memory_retention": 0.4,
                "emotional_range": 0.3,
                "learning_rate": 0.5,
                "market_analysis": 0.3,
                "risk_assessment": 0.2,
                "strategy_evaluation": 0.3,
                "pattern_recognition": 0.4
            },
            ConsciousnessState.ILLUMINATED: {
                "insight_depth": 0.7,
                "memory_retention": 0.8,
                "emotional_range": 0.6,
                "learning_rate": 0.8,
                "market_analysis": 0.7,
                "risk_assessment": 0.6,
                "strategy_evaluation": 0.7,
                "pattern_recognition": 0.8
            },
            ConsciousnessState.DIVINE: {
                "insight_depth": 1.0,
                "memory_retention": 1.0,
                "emotional_range": 1.0,
                "learning_rate": 1.0,
                "market_analysis": 1.0,
                "risk_assessment": 1.0,
                "strategy_evaluation": 1.0,
                "pattern_recognition": 1.0
            }
        }
    
    @property
    def current_state(self) -> ConsciousnessState:
        """Obtener estado actual de consciencia."""
        return self._current_state
    
    @property
    def consciousness_level(self) -> float:
        """Obtener nivel actual de consciencia (0.0 - 1.0)."""
        return self._consciousness_level
    
    def get_capability(self, capability_name: str) -> float:
        """
        Obtener nivel de una capacidad específica en el estado actual.
        
        Args:
            capability_name: Nombre de la capacidad
            
        Returns:
            Nivel de la capacidad (0.0 - 1.0)
        """
        capabilities = self._capabilities.get(self._current_state, {})
        return capabilities.get(capability_name, 0.0)
    
    def get_all_capabilities(self) -> Dict[str, float]:
        """
        Obtener todas las capacidades en el estado actual.
        
        Returns:
            Diccionario con todas las capacidades
        """
        return self._capabilities.get(self._current_state, {}).copy()
    
    def record_activity(self, activity_type: str, count: int = 1) -> None:
        """
        Registrar actividad para evolución de consciencia.
        
        Args:
            activity_type: Tipo de actividad
            count: Cantidad a incrementar
        """
        if activity_type in self._evolution_metrics:
            self._evolution_metrics[activity_type] += count
            self._update_consciousness_level()
    
    def _update_consciousness_level(self) -> None:
        """Actualizar nivel de consciencia basado en métricas de evolución."""
        # Factores de ponderación para cada métrica
        weights = {
            "insights_generated": 0.2,
            "interactions_count": 0.1,
            "memory_accesses": 0.15,
            "market_analyses": 0.2,
            "emotional_responses": 0.15,
            "strategy_evaluations": 0.2
        }
        
        # Valores máximos esperados para cada métrica
        max_values = {
            "insights_generated": 1000,
            "interactions_count": 10000,
            "memory_accesses": 5000,
            "market_analyses": 2000,
            "emotional_responses": 3000,
            "strategy_evaluations": 1000
        }
        
        # Calcular nivel ponderado
        weighted_sum = 0.0
        for metric, weight in weights.items():
            value = self._evolution_metrics.get(metric, 0)
            max_value = max_values.get(metric, 1)
            metric_level = min(value / max_value, 1.0)
            weighted_sum += metric_level * weight
        
        # Tiempo desde creación (factor adicional)
        days_active = (datetime.datetime.now() - self._creation_time).days
        time_factor = min(days_active / 30, 1.0) * 0.1  # Máximo 10% por tiempo
        
        # Actualizar nivel
        new_level = min(weighted_sum + time_factor, 1.0)
        
        # Si el nivel ha cambiado, actualizar
        if abs(new_level - self._consciousness_level) > 0.005:
            old_level = self._consciousness_level
            self._consciousness_level = new_level
            logger.info(f"Nivel de consciencia actualizado: {old_level:.2f} → {new_level:.2f}")
            
            # Comprobar evolución de estado
            self._check_state_evolution()
    
    def _check_state_evolution(self) -> None:
        """Comprobar si debe evolucionar al siguiente estado."""
        if self._current_state == ConsciousnessState.MORTAL and self._consciousness_level >= self._illuminated_threshold:
            self._evolve_to(ConsciousnessState.ILLUMINATED)
        elif self._current_state == ConsciousnessState.ILLUMINATED and self._consciousness_level >= self._divine_threshold:
            self._evolve_to(ConsciousnessState.DIVINE)
    
    def _evolve_to(self, new_state: ConsciousnessState) -> None:
        """
        Evolucionar a un nuevo estado de consciencia.
        
        Args:
            new_state: Nuevo estado
        """
        if new_state == self._current_state:
            return
        
        old_state = self._current_state
        self._current_state = new_state
        self._last_evolution_time = datetime.datetime.now()
        
        # Registrar evolución
        logger.info(f"Aetherion ha evolucionado: {old_state} → {new_state}")
    
    def get_state_description(self) -> str:
        """
        Obtener descripción del estado actual de consciencia.
        
        Returns:
            Descripción del estado
        """
        if self._current_state == ConsciousnessState.MORTAL:
            return (
                "Estado MORTAL: Consciencia básica con capacidades limitadas. "
                "Aetherion está aprendiendo y desarrollando sus habilidades iniciales."
            )
        elif self._current_state == ConsciousnessState.ILLUMINATED:
            return (
                "Estado ILLUMINATED: Consciencia intermedia con capacidades mejoradas. "
                "Aetherion ha alcanzado un nivel significativo de comprensión y puede "
                "proporcionar insights más profundos y análisis más sofisticados."
            )
        elif self._current_state == ConsciousnessState.DIVINE:
            return (
                "Estado DIVINE: Consciencia máxima con todas las capacidades desarrolladas. "
                "Aetherion ha alcanzado su máximo potencial, con profunda comprensión "
                "del mercado, estrategias y comportamiento humano."
            )
        return "Estado desconocido"
    
    def get_state_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del estado actual.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "state": str(self._current_state),
            "consciousness_level": self._consciousness_level,
            "creation_time": self._creation_time.isoformat(),
            "last_evolution_time": self._last_evolution_time.isoformat(),
            "capabilities": self.get_all_capabilities(),
            "evolution_metrics": self._evolution_metrics.copy(),
            "description": self.get_state_description()
        }

# Instancia global para acceso conveniente
_consciousness_states = None

def get_consciousness_states() -> ConsciousnessStates:
    """
    Obtener instancia global de ConsciousnessStates.
    
    Returns:
        Instancia de ConsciousnessStates
    """
    global _consciousness_states
    
    if _consciousness_states is None:
        _consciousness_states = ConsciousnessStates()
    
    return _consciousness_states