"""
El Alma de Gabriel - Estados emocionales y personalidad

Este módulo implementa el componente "Alma" (Soul) del motor de comportamiento humano Gabriel,
responsable de los estados emocionales, rasgos de personalidad y la evolución de ambos
en respuesta a estímulos del mercado y eventos.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import random
from enum import Enum, auto

logger = logging.getLogger(__name__)

class Mood(Enum):
    """Estados emocionales del Alma de Gabriel."""
    SERENE = auto()      # Calma - máxima capacidad de evaluación objetiva
    HOPEFUL = auto()     # Esperanzado - ligero optimismo
    NEUTRAL = auto()     # Neutral - estado equilibrado base
    CAUTIOUS = auto()    # Cautioso - evaluación más conservadora
    RESTLESS = auto()    # Inquieto - cierta ansiedad e impaciencia
    FEARFUL = auto()     # Temeroso - predomina el miedo, muy defensivo

class Soul:
    """
    El Alma (Soul) de Gabriel, responsable de sus estados emocionales.
    
    Este componente simula la naturaleza emocional humana, incluyendo:
    - Estados emocionales que cambian según el mercado y eventos
    - Estabilidad emocional que varía según el arquetipo de personalidad
    - Memoria emocional que influye en la evolución de estados
    - Reactividad emocional ante diferentes tipos de eventos
    """
    
    def __init__(self):
        """Inicializar el Alma con sus propiedades esenciales."""
        # Estado emocional actual y su intensidad
        self.current_mood = Mood.NEUTRAL
        self.mood_intensity = 0.5  # 0-1
        
        # Propiedades de personalidad
        self.emotional_stability = 0.7  # 0-1, más alto = más estable
        self.optimism_bias = 0.0  # -1 a 1, negativo = pesimista
        self.risk_sensitivity = 0.5  # 0-1, reactividad a pérdidas potenciales
        self.market_sensitivity = 0.5  # 0-1, cuánto influye el mercado
        
        # Memoria emocional (para evolución natural)
        self.mood_history = []  # Últimos estados para tendencias
        self.market_impacts = []  # Memoria de impactos de mercado
        self.max_history = 10  # Cuántas entradas guardar
        
        # Variables de estado
        self.last_update = datetime.now()
        self.consecutive_mood_days = 0  # Días en el mismo estado
        
        logger.info(f"Alma de Gabriel inicializada con estado {self.current_mood.name}, intensidad {self.mood_intensity:.2f}")
    
    async def initialize(self) -> bool:
        """
        Inicializar completamente el componente Alma.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            # Establecer estado inicial
            self._select_initial_mood()
            
            # Inicializar la memoria emocional
            self.mood_history = [
                {"mood": self.current_mood, "intensity": self.mood_intensity, "timestamp": datetime.now()}
            ]
            
            logger.info(f"Alma inicializada exitosamente con humor {self.current_mood.name}")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar Alma: {str(e)}")
            return False
    
    def _select_initial_mood(self) -> None:
        """Seleccionar un estado emocional inicial basado en la personalidad."""
        # Personalidades más optimistas tienden a iniciar en estados positivos
        if self.optimism_bias > 0.3:
            # Personalidad optimista
            moods = [Mood.SERENE, Mood.HOPEFUL, Mood.NEUTRAL]
            weights = [0.3, 0.5, 0.2]
        elif self.optimism_bias < -0.3:
            # Personalidad pesimista
            moods = [Mood.NEUTRAL, Mood.CAUTIOUS, Mood.RESTLESS]
            weights = [0.3, 0.5, 0.2]
        else:
            # Personalidad neutral
            moods = [Mood.SERENE, Mood.HOPEFUL, Mood.NEUTRAL, Mood.CAUTIOUS]
            weights = [0.2, 0.3, 0.3, 0.2]
        
        self.current_mood = random.choices(moods, weights=weights, k=1)[0]
        self.mood_intensity = random.uniform(0.4, 0.7)
    
    async def update(self, market_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Actualizar el estado emocional basado en datos de mercado.
        
        Args:
            market_data: Datos de mercado actuales
        """
        try:
            # Si no hay datos, aplicar evolución natural
            if not market_data:
                self._natural_mood_evolution()
                return
            
            # Calcular impacto del mercado
            market_impact = self._calculate_market_impact(market_data)
            
            # Aplicar impacto según la sensibilidad
            self._apply_market_impact(market_impact)
            
            # Registrar el impacto en la memoria
            self._record_market_impact(market_impact, market_data)
            
            # Actualizar timestamp
            self.last_update = datetime.now()
        except Exception as e:
            logger.error(f"Error al actualizar estado emocional: {str(e)}")
    
    def _calculate_market_impact(self, market_data: Dict[str, Any]) -> float:
        """
        Calcular el impacto emocional de los datos de mercado.
        
        Args:
            market_data: Datos de mercado actuales
            
        Returns:
            Impacto emocional (-1 a 1)
        """
        impact = 0.0
        
        # Procesar volatilidad
        volatility = market_data.get("volatility", 0.5)
        impact -= (volatility - 0.5) * self.risk_sensitivity * 0.5
        
        # Procesar tendencia
        trend = market_data.get("trend", 0.0)  # -1 a 1
        impact += trend * 0.4
        
        # Procesar cambio de precio reciente
        price_change = market_data.get("price_change", 0.0)  # porcentaje
        impact += max(min(price_change * 0.2, 0.4), -0.4)
        
        # Procesar sentimiento general
        sentiment = market_data.get("sentiment", "neutral")
        if sentiment == "bullish":
            impact += 0.2
        elif sentiment == "bearish":
            impact -= 0.2
        
        # Ajustar por la sensibilidad al mercado y estabilidad emocional
        impact *= self.market_sensitivity
        impact *= (2 - self.emotional_stability)
        
        return max(min(impact, 1.0), -1.0)
    
    def _apply_market_impact(self, impact: float) -> None:
        """
        Aplicar el impacto del mercado al estado emocional.
        
        Args:
            impact: Impacto emocional (-1 a 1)
        """
        # Determinar si hay un cambio de estado
        previous_mood = self.current_mood
        
        # Ajustar intensidad del estado actual
        self.mood_intensity = max(min(self.mood_intensity + impact * 0.3, 1.0), 0.0)
        
        # Determinar si cambia de estado
        # Un impacto fuerte puede cambiar el estado directamente
        threshold = 1.0 - self.emotional_stability
        
        if impact > threshold * 2:
            # Impacto positivo fuerte
            if self.current_mood == Mood.FEARFUL:
                self.current_mood = Mood.RESTLESS
            elif self.current_mood == Mood.RESTLESS:
                self.current_mood = Mood.CAUTIOUS
            elif self.current_mood == Mood.CAUTIOUS:
                self.current_mood = Mood.NEUTRAL
            elif self.current_mood == Mood.NEUTRAL:
                self.current_mood = Mood.HOPEFUL
            elif self.current_mood == Mood.HOPEFUL:
                self.current_mood = Mood.SERENE
        elif impact < -threshold * 2:
            # Impacto negativo fuerte
            if self.current_mood == Mood.SERENE:
                self.current_mood = Mood.HOPEFUL
            elif self.current_mood == Mood.HOPEFUL:
                self.current_mood = Mood.NEUTRAL
            elif self.current_mood == Mood.NEUTRAL:
                self.current_mood = Mood.CAUTIOUS
            elif self.current_mood == Mood.CAUTIOUS:
                self.current_mood = Mood.RESTLESS
            elif self.current_mood == Mood.RESTLESS:
                self.current_mood = Mood.FEARFUL
        
        # Si la intensidad es muy alta, puede cambiar de estado
        if self.mood_intensity > 0.9 and impact > 0:
            # Potencial mejora de humor
            if self.current_mood == Mood.FEARFUL:
                self.current_mood = Mood.RESTLESS
                self.mood_intensity = 0.5
            elif self.current_mood == Mood.RESTLESS:
                self.current_mood = Mood.CAUTIOUS
                self.mood_intensity = 0.5
        elif self.mood_intensity > 0.9 and impact < 0:
            # Potencial empeoramiento
            if self.current_mood == Mood.SERENE:
                self.current_mood = Mood.HOPEFUL
                self.mood_intensity = 0.5
            elif self.current_mood == Mood.HOPEFUL:
                self.current_mood = Mood.NEUTRAL
                self.mood_intensity = 0.5
        
        # Registrar cambio en el historial si hubo cambio
        if previous_mood != self.current_mood:
            self._record_mood_change()
            self.consecutive_mood_days = 0
        else:
            self.consecutive_mood_days += 1
    
    def _record_mood_change(self) -> None:
        """Registrar un cambio de estado emocional en el historial."""
        self.mood_history.append({
            "mood": self.current_mood,
            "intensity": self.mood_intensity,
            "timestamp": datetime.now()
        })
        
        # Limitar tamaño del historial
        if len(self.mood_history) > self.max_history:
            self.mood_history = self.mood_history[-self.max_history:]
    
    def _record_market_impact(self, impact: float, data: Dict[str, Any]) -> None:
        """
        Registrar el impacto de mercado en la memoria.
        
        Args:
            impact: Impacto calculado
            data: Datos de mercado
        """
        # Simplificar los datos para almacenamiento
        simplified_data = {
            "volatility": data.get("volatility", 0.0),
            "trend": data.get("trend", 0.0),
            "price_change": data.get("price_change", 0.0),
            "sentiment": data.get("sentiment", "neutral")
        }
        
        # Registrar
        self.market_impacts.append({
            "impact": impact,
            "data": simplified_data,
            "timestamp": datetime.now()
        })
        
        # Limitar tamaño
        if len(self.market_impacts) > self.max_history:
            self.market_impacts = self.market_impacts[-self.max_history:]
    
    def _natural_mood_evolution(self) -> None:
        """Aplicar evolución natural del estado emocional con el tiempo."""
        # Determinar cuánto tiempo ha pasado desde la última actualización
        time_delta = datetime.now() - self.last_update
        hours_passed = time_delta.total_seconds() / 3600
        
        # La evolución natural tiende hacia el punto de equilibrio personal
        # Personalidades optimistas tienden a estados más positivos
        target_mood = Mood.NEUTRAL
        if self.optimism_bias > 0.3:
            if random.random() < self.optimism_bias:
                target_mood = Mood.HOPEFUL
        elif self.optimism_bias < -0.3:
            if random.random() < abs(self.optimism_bias):
                target_mood = Mood.CAUTIOUS
        
        # Los estados extremos tienden a normalizarse más rápido
        normalization_rate = 0.1
        if self.current_mood in [Mood.SERENE, Mood.FEARFUL]:
            normalization_rate = 0.2
        
        # Aplicar normalización gradual
        if hours_passed > 4 and self.current_mood != target_mood:
            if random.random() < normalization_rate * hours_passed / 24:
                # Mover un paso hacia el objetivo
                if self.current_mood == Mood.SERENE:
                    self.current_mood = Mood.HOPEFUL
                elif self.current_mood == Mood.HOPEFUL and target_mood in [Mood.NEUTRAL, Mood.CAUTIOUS]:
                    self.current_mood = Mood.NEUTRAL
                elif self.current_mood == Mood.NEUTRAL and target_mood == Mood.CAUTIOUS:
                    self.current_mood = Mood.CAUTIOUS
                elif self.current_mood == Mood.CAUTIOUS and target_mood == Mood.NEUTRAL:
                    self.current_mood = Mood.NEUTRAL
                elif self.current_mood == Mood.RESTLESS:
                    self.current_mood = Mood.CAUTIOUS
                elif self.current_mood == Mood.FEARFUL:
                    self.current_mood = Mood.RESTLESS
                
                self._record_mood_change()
        
        # La intensidad también se normaliza
        if hours_passed > 2:
            # Hacia valor medio
            target_intensity = 0.5 + (self.optimism_bias * 0.2)
            intensity_adjust = (target_intensity - self.mood_intensity) * min(hours_passed / 48, 0.5)
            self.mood_intensity += intensity_adjust
    
    def apply_archetype(self, archetype_config: Dict[str, Any]) -> None:
        """
        Aplicar configuración de arquetipo al Alma.
        
        Args:
            archetype_config: Configuración del arquetipo
        """
        # Aplicar propiedades específicas
        self.emotional_stability = archetype_config.get("emotional_stability", self.emotional_stability)
        self.optimism_bias = archetype_config.get("optimism_bias", self.optimism_bias)
        self.risk_sensitivity = archetype_config.get("risk_sensitivity", self.risk_sensitivity)
        self.market_sensitivity = archetype_config.get("market_sensitivity", self.market_sensitivity)
        
        # Actualizar mood inicial según arquetipo si está especificado
        if "initial_mood" in archetype_config:
            mood_name = archetype_config["initial_mood"]
            try:
                self.current_mood = Mood[mood_name]
                self.mood_intensity = archetype_config.get("initial_intensity", 0.5)
                self._record_mood_change()
            except KeyError:
                logger.warning(f"Mood {mood_name} no encontrado, manteniendo {self.current_mood.name}")
    
    def get_mood(self) -> Mood:
        """
        Obtener el estado emocional actual.
        
        Returns:
            Estado emocional actual
        """
        return self.current_mood
    
    def get_mood_intensity(self) -> float:
        """
        Obtener la intensidad del estado emocional actual.
        
        Returns:
            Intensidad (0-1)
        """
        return self.mood_intensity
    
    def get_emotional_stability(self) -> float:
        """
        Obtener nivel de estabilidad emocional.
        
        Returns:
            Estabilidad emocional (0-1)
        """
        return self.emotional_stability
    
    def adapt_to_decision_style(self, decision_style) -> None:
        """
        Adaptarse al estilo de decisión de la Voluntad para mantener coherencia.
        
        Args:
            decision_style: Estilo de decisión actual
        """
        # Ajustes sutiles para mantener coherencia interna
        # No cambia el estado, solo ajusta propiedades
        from genesis.trading.gabriel.will import Decision
        
        if decision_style == Decision.CAUTIOUS:
            # Aumentar ligeramente sensibilidad al riesgo
            self.risk_sensitivity = min(self.risk_sensitivity + 0.1, 1.0)
        elif decision_style == Decision.AGGRESSIVE:
            # Disminuir ligeramente sensibilidad al riesgo
            self.risk_sensitivity = max(self.risk_sensitivity - 0.1, 0.0)
    
    async def randomize(self) -> None:
        """Aleatorizar el estado emocional para mayor variabilidad."""
        # Elegir un estado aleatorio pero coherente con la personalidad
        moods = list(Mood)
        
        # Personalidades optimistas menos probabilidad de estados negativos
        if self.optimism_bias > 0.3:
            weights = [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]  # Favorece positivos
        elif self.optimism_bias < -0.3:
            weights = [0.05, 0.1, 0.15, 0.25, 0.25, 0.2]  # Favorece negativos
        else:
            weights = [0.15, 0.2, 0.2, 0.2, 0.15, 0.1]  # Más equilibrado
        
        self.current_mood = random.choices(moods, weights=weights, k=1)[0]
        self.mood_intensity = random.uniform(0.4, 0.9)
        self._record_mood_change()
        
        logger.info(f"Alma aleatorizada a estado {self.current_mood.name}, intensidad {self.mood_intensity:.2f}")
    
    def reset(self) -> None:
        """Reiniciar el estado a valores predeterminados."""
        self.current_mood = Mood.NEUTRAL
        self.mood_intensity = 0.5
        self.consecutive_mood_days = 0
        self._record_mood_change()