"""
Gabriel Symphony - Orquestador del Motor de Comportamiento Humano

Este módulo implementa el orquestador principal que coordina los tres componentes esenciales
del motor de comportamiento humano Gabriel: Alma (Soul), Mirada (Gaze) y Voluntad (Will).

El nombre "Symphony" representa la armonía perfecta entre estos tres componentes, que
trabajan juntos para generar comportamientos humanos realistas en el trading.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import random

from genesis.trading.gabriel.soul import Soul, Mood
from genesis.trading.gabriel.gaze import Gaze, Perspective
from genesis.trading.gabriel.will import Will, Decision
from genesis.trading.gabriel.essence import archetypes

logger = logging.getLogger(__name__)

class Gabriel:
    """
    Orquestador principal del motor de comportamiento humano Gabriel.
    
    Esta clase integra los tres componentes del motor:
    - Alma (Soul): Estado emocional y personalidad
    - Mirada (Gaze): Percepción e interpretación de la realidad
    - Voluntad (Will): Toma de decisiones y acciones
    
    Juntos, estos componentes generan comportamientos humanos auténticos
    que hacen que el sistema de trading se comporte de manera más natural
    y menos predecible por algoritmos de detección.
    """
    
    def __init__(self):
        """Inicializar el orquestador Gabriel con sus tres componentes."""
        # Crear los tres componentes fundamentales
        self.soul = Soul()
        self.gaze = Gaze()
        self.will = Will()
        
        # Historial para análisis y evolución del comportamiento
        self.history = {
            "mood_changes": [],
            "perspective_shifts": [],
            "decisions": []
        }
        
        # Configuración y estado
        self.archetype = random.choice(list(archetypes.keys()))
        self.last_update = datetime.now()
        self.active = True
        logger.info(f"Gabriel inicializado con arquetipo: {self.archetype}")
    
    async def initialize(self) -> bool:
        """
        Inicializar completamente el motor Gabriel.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            # Inicializar cada componente
            await self.soul.initialize()
            await self.gaze.initialize()
            await self.will.initialize()
            
            # Aplicar el arquetipo seleccionado
            self._apply_archetype(self.archetype)
            
            # Sincronizar componentes
            self._synchronize_components()
            
            logger.info("Motor Gabriel inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar Gabriel: {str(e)}")
            return False
    
    def _apply_archetype(self, archetype_name: str) -> None:
        """
        Aplicar un arquetipo predefinido a Gabriel.
        
        Args:
            archetype_name: Nombre del arquetipo a aplicar
        """
        if archetype_name not in archetypes:
            logger.warning(f"Arquetipo {archetype_name} no encontrado, usando BALANCED")
            archetype_name = "BALANCED"
        
        archetype = archetypes[archetype_name]
        
        # Aplicar configuración del arquetipo a cada componente
        self.soul.apply_archetype(archetype.get("soul", {}))
        self.gaze.apply_archetype(archetype.get("gaze", {}))
        self.will.apply_archetype(archetype.get("will", {}))
        
        logger.info(f"Arquetipo {archetype_name} aplicado a Gabriel")
    
    def _synchronize_components(self) -> None:
        """Sincronizar los tres componentes para mantener coherencia interna."""
        # Sincronizar Alma con Mirada
        current_mood = self.soul.get_mood()
        self.gaze.adapt_to_mood(current_mood)
        
        # Sincronizar Mirada con Voluntad
        current_perspective = self.gaze.get_perspective()
        self.will.adapt_to_perspective(current_perspective)
        
        # Sincronizar Voluntad con Alma (ciclo completo)
        current_decision_style = self.will.get_decision_style()
        self.soul.adapt_to_decision_style(current_decision_style)
    
    async def update(self, market_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Actualizar el estado interno de Gabriel basado en datos de mercado.
        
        Args:
            market_data: Datos de mercado actuales
        """
        try:
            # Actualizar cada componente
            await self.soul.update(market_data)
            await self.gaze.update(market_data, self.soul.get_mood())
            await self.will.update(self.gaze.get_perspective(), self.soul.get_mood())
            
            # Registrar cambios importantes en el historial
            self._update_history()
            
            # Marcar tiempo de última actualización
            self.last_update = datetime.now()
        except Exception as e:
            logger.error(f"Error al actualizar Gabriel: {str(e)}")
    
    def _update_history(self) -> None:
        """Actualizar historial de cambios para análisis posterior."""
        # Registrar cambio de humor si es diferente al último registrado
        if (not self.history["mood_changes"] or 
            self.history["mood_changes"][-1]["mood"] != self.soul.get_mood().name):
            self.history["mood_changes"].append({
                "timestamp": datetime.now(),
                "mood": self.soul.get_mood().name,
                "intensity": self.soul.get_mood_intensity()
            })
        
        # Registrar cambio de perspectiva si es diferente a la última
        if (not self.history["perspective_shifts"] or 
            self.history["perspective_shifts"][-1]["perspective"] != self.gaze.get_perspective().name):
            self.history["perspective_shifts"].append({
                "timestamp": datetime.now(),
                "perspective": self.gaze.get_perspective().name,
                "confidence": self.gaze.get_perspective_confidence()
            })
    
    async def evaluate_trade(self, signal_strength: float, 
                           market_context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluar una oportunidad de trading con comportamiento humano.
        
        Args:
            signal_strength: Fuerza de la señal (0-1)
            market_context: Contexto del mercado
            
        Returns:
            Tupla (decisión, razón)
        """
        # Actualizar el estado basado en el contexto de mercado
        await self.update(market_context)
        
        # Obtener datos de cada componente
        mood = self.soul.get_mood()
        perspective = self.gaze.get_perspective()
        
        # Dejar que la Voluntad tome la decisión
        decision, reason = await self.will.decide_trade(
            signal_strength, 
            mood, 
            perspective, 
            market_context
        )
        
        # Registrar la decisión en el historial
        self.history["decisions"].append({
            "timestamp": datetime.now(),
            "decision": "ENTER" if decision else "SKIP",
            "signal_strength": signal_strength,
            "mood": mood.name,
            "perspective": perspective.name,
            "reason": reason
        })
        
        return decision, reason
    
    async def adjust_position_size(self, base_size: float, 
                                 confidence: float,
                                 is_entry: bool = True) -> float:
        """
        Ajustar el tamaño de una posición con comportamiento humano.
        
        Args:
            base_size: Tamaño base de la posición
            confidence: Nivel de confianza en la operación (0-1)
            is_entry: True si es entrada, False si es salida
            
        Returns:
            Tamaño ajustado
        """
        # Obtener el estado actual
        mood = self.soul.get_mood()
        perspective = self.gaze.get_perspective()
        
        # Dejar que la Voluntad ajuste el tamaño
        adjusted_size = await self.will.adjust_size(
            base_size,
            mood,
            perspective,
            confidence,
            is_entry
        )
        
        return adjusted_size
    
    async def should_exit_trade(self, 
                              profit_percent: float, 
                              time_held_hours: float,
                              market_volatility: float) -> Tuple[bool, str]:
        """
        Decidir si se debe salir de una operación con comportamiento humano.
        
        Args:
            profit_percent: Porcentaje de beneficio actual
            time_held_hours: Tiempo manteniendo la posición en horas
            market_volatility: Volatilidad actual del mercado (0-1)
            
        Returns:
            Tupla (decisión, razón)
        """
        # Obtener el estado actual
        mood = self.soul.get_mood()
        perspective = self.gaze.get_perspective()
        
        # Dejar que la Voluntad decida sobre la salida
        decision, reason = await self.will.decide_exit(
            profit_percent,
            time_held_hours,
            market_volatility,
            mood,
            perspective
        )
        
        # Registrar la decisión en el historial
        self.history["decisions"].append({
            "timestamp": datetime.now(),
            "decision": "EXIT" if decision else "HOLD",
            "profit_percent": profit_percent,
            "time_held_hours": time_held_hours,
            "mood": mood.name,
            "perspective": perspective.name,
            "reason": reason
        })
        
        return decision, reason
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Obtener un resumen del estado actual del motor Gabriel.
        
        Returns:
            Diccionario con estado actual
        """
        current_state = {
            "mood": self.soul.get_mood().name,
            "mood_intensity": self.soul.get_mood_intensity(),
            "perspective": self.gaze.get_perspective().name,
            "perspective_confidence": self.gaze.get_perspective_confidence(),
            "decision_style": self.will.get_decision_style().name,
            "risk_preference": self.will.get_risk_preference(),
            "emotional_stability": self.soul.get_emotional_stability(),
            "market_perception": self.gaze.get_market_perception(),
            "last_update": self.last_update.isoformat(),
            "archetype": self.archetype
        }
        
        return current_state
    
    async def randomize(self) -> Dict[str, Any]:
        """
        Aleatorizar el comportamiento humano para mayor variabilidad.
        
        Returns:
            Estado después de la aleatorización
        """
        # Aleatorizar cada componente
        await self.soul.randomize()
        await self.gaze.randomize(self.soul.get_mood())
        await self.will.randomize()
        
        # Mantener coherencia global
        self._synchronize_components()
        
        # Devolver estado actual
        return self.get_current_state()
    
    def is_fearful(self) -> bool:
        """
        Verificar si Gabriel está en un estado temeroso.
        
        Returns:
            True si está en un estado de miedo
        """
        return self.soul.get_mood() == Mood.FEARFUL
    
    def reset(self) -> None:
        """Reiniciar el estado del motor Gabriel a valores predeterminados."""
        self.soul.reset()
        self.gaze.reset()
        self.will.reset()
        self._synchronize_components()
        logger.info("Motor Gabriel reiniciado")


def get_gabriel() -> Gabriel:
    """
    Obtener una instancia del motor Gabriel.
    
    Returns:
        Instancia de Gabriel
    """
    return Gabriel()