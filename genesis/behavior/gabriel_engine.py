"""
Motor de Comportamiento Humano Gabriel para el Sistema Genesis

Este módulo implementa el motor de comportamiento simulado Gabriel, diseñado para:
1. Emular estados emocionales humanos que afectan las decisiones de trading
2. Proporcionar perfiles de riesgo dinámicos basados en el estado emocional
3. Evaluar oportunidades de trading con sesgos emocionales realistas
4. Añadir variabilidad controlada a las decisiones algorítmicas

La integración con Aetherion permite una conciencia superior que puede
modular y optimizar el comportamiento humano simulado.
"""

import asyncio
import logging
import random
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple, Union

# Configurar logging
logger = logging.getLogger(__name__)

class EmotionalState(Enum):
    """Estados emocionales para el motor de comportamiento."""
    SERENE = auto()    # Calma - máxima capacidad de evaluación objetiva
    HOPEFUL = auto()   # Esperanzado - ligero optimismo
    CAUTIOUS = auto()  # Cauteloso - evaluación más conservadora
    RESTLESS = auto()  # Inquieto - cierta ansiedad e impaciencia
    FEARFUL = auto()   # Temeroso - predomina el miedo, muy defensivo

class RiskProfile:
    """Perfil de riesgo dinámico basado en estado emocional."""
    
    def __init__(self, base_risk_tolerance: float = 0.5):
        """
        Inicializar perfil de riesgo.
        
        Args:
            base_risk_tolerance: Nivel base de tolerancia al riesgo (0-1)
        """
        self.base_risk_tolerance = max(0.1, min(0.9, base_risk_tolerance))
        self.current_risk_tolerance = self.base_risk_tolerance
        self.risk_multipliers = {
            EmotionalState.SERENE: 1.0,    # Neutral
            EmotionalState.HOPEFUL: 1.3,   # Más arriesgado
            EmotionalState.CAUTIOUS: 0.7,  # Más conservador
            EmotionalState.RESTLESS: 1.5,  # Más arriesgado (impulsivo)
            EmotionalState.FEARFUL: 0.4,   # Mucho más conservador
        }
        self.update_timestamp = datetime.now()
        
    def update_for_emotional_state(self, state: EmotionalState) -> None:
        """
        Actualizar tolerancia al riesgo según estado emocional.
        
        Args:
            state: Estado emocional actual
        """
        multiplier = self.risk_multipliers.get(state, 1.0)
        self.current_risk_tolerance = max(0.1, min(0.9, self.base_risk_tolerance * multiplier))
        self.update_timestamp = datetime.now()
        
    def get_max_position_size(self, available_capital: float) -> float:
        """
        Obtener tamaño máximo de posición basado en perfil de riesgo.
        
        Args:
            available_capital: Capital disponible
            
        Returns:
            Tamaño máximo de posición recomendado
        """
        return available_capital * self.current_risk_tolerance
        
    def get_stop_loss_percentage(self) -> float:
        """
        Obtener porcentaje de stop loss recomendado.
        
        Returns:
            Porcentaje de stop loss (0-1)
        """
        # Más conservador (fearful) = stop loss más cercano
        inverse_risk = 1 - self.current_risk_tolerance
        return 0.02 + (inverse_risk * 0.08)  # Entre 2% y 10%
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con datos del perfil
        """
        return {
            "base_risk_tolerance": self.base_risk_tolerance,
            "current_risk_tolerance": self.current_risk_tolerance,
            "max_position_pct": self.current_risk_tolerance * 100,
            "stop_loss_pct": self.get_stop_loss_percentage() * 100,
            "last_update": self.update_timestamp.isoformat()
        }

class GabrielBehaviorEngine:
    """Motor de comportamiento humano simulado Gabriel."""
    
    def __init__(self, personality_seed: Optional[int] = None):
        """
        Inicializar motor de comportamiento.
        
        Args:
            personality_seed: Semilla para personalidad (opcional)
        """
        self.personality_seed = personality_seed or random.randint(1, 1000000)
        random.seed(self.personality_seed)
        
        # Características de personalidad
        self.neuroticism = random.uniform(0.3, 0.7)      # Tendencia a emociones negativas
        self.extraversion = random.uniform(0.3, 0.7)     # Energía, sociabilidad
        self.openness = random.uniform(0.3, 0.7)         # Apertura a nuevas experiencias
        self.conscientiousness = random.uniform(0.3, 0.7)  # Organización, disciplina
        self.agreeableness = random.uniform(0.3, 0.7)    # Empatía, cooperación
        
        # Estado emocional
        self.current_state = EmotionalState.SERENE
        self.previous_states: List[Tuple[EmotionalState, str, datetime]] = []
        self.state_duration = random.uniform(1.0, 3.0)  # Horas
        self.last_state_change = datetime.now()
        
        # Perfil de riesgo
        base_risk = 0.5 - (self.neuroticism * 0.2) + (self.openness * 0.2)
        self.risk_profile = RiskProfile(base_risk_tolerance=base_risk)
        
        # Factores de mercado que afectan el estado
        self.market_sensitivity = 0.3 + (self.neuroticism * 0.4)  # 0.3-0.7
        
        # Historial de decisiones
        self.decision_history: List[Dict[str, Any]] = []
        
        logger.info(f"GabrielBehaviorEngine inicializado con personalidad {self.personality_seed}")
    
    async def initialize(self) -> bool:
        """
        Inicializar motor asíncronamente.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            await self._randomize_initial_state()
            logger.info("GabrielBehaviorEngine inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar GabrielBehaviorEngine: {e}")
            return False
    
    async def _randomize_initial_state(self) -> None:
        """Establecer estado emocional inicial aleatorio."""
        states = list(EmotionalState)
        weights = [
            0.4,  # SERENE - más probable inicialmente
            0.25,  # HOPEFUL
            0.2,  # CAUTIOUS
            0.1,  # RESTLESS
            0.05  # FEARFUL - menos probable inicialmente
        ]
        self.current_state = random.choices(states, weights=weights)[0]
        self.risk_profile.update_for_emotional_state(self.current_state)
        logger.info(f"Estado emocional inicial: {self.current_state.name}")
    
    async def change_emotional_state(self, state: Union[EmotionalState, str], reason: str = "") -> bool:
        """
        Cambiar estado emocional.
        
        Args:
            state: Nuevo estado emocional
            reason: Razón del cambio
            
        Returns:
            True si se cambió correctamente
        """
        try:
            # Convertir str a enum si es necesario
            if isinstance(state, str):
                try:
                    state = EmotionalState[state]
                except KeyError:
                    logger.error(f"Estado emocional inválido: {state}")
                    return False
            
            # Guardar estado anterior
            self.previous_states.append((self.current_state, reason, datetime.now()))
            if len(self.previous_states) > 10:
                self.previous_states.pop(0)
                
            # Actualizar estado
            self.current_state = state
            self.last_state_change = datetime.now()
            self.risk_profile.update_for_emotional_state(state)
            
            logger.info(f"Estado emocional cambiado a {state.name}. Razón: {reason}")
            return True
        except Exception as e:
            logger.error(f"Error al cambiar estado emocional: {e}")
            return False
    
    async def get_emotional_state(self) -> Dict[str, Any]:
        """
        Obtener estado emocional actual.
        
        Returns:
            Diccionario con información del estado
        """
        time_in_state = (datetime.now() - self.last_state_change).total_seconds() / 3600  # Horas
        state_intensity = min(1.0, time_in_state / self.state_duration)
        
        return {
            "state": self.current_state.name,
            "previous_state": self.previous_states[-1][0].name if self.previous_states else None,
            "last_change_reason": self.previous_states[-1][1] if self.previous_states else None,
            "time_in_state_hours": round(time_in_state, 2),
            "state_intensity": round(state_intensity, 2),
            "personality": {
                "neuroticism": round(self.neuroticism, 2),
                "extraversion": round(self.extraversion, 2),
                "openness": round(self.openness, 2),
                "conscientiousness": round(self.conscientiousness, 2),
                "agreeableness": round(self.agreeableness, 2)
            }
        }
    
    async def get_risk_profile(self) -> Dict[str, Any]:
        """
        Obtener perfil de riesgo actual.
        
        Returns:
            Diccionario con información del perfil
        """
        return self.risk_profile.to_dict()
    
    async def evaluate_market_conditions(self, market_data: Dict[str, Any]) -> None:
        """
        Evaluar condiciones de mercado y potencialmente cambiar estado.
        
        Args:
            market_data: Datos de mercado
        """
        # Extraer métricas clave
        volatility = market_data.get("volatility", 0.5)
        trend = market_data.get("trend", 0)  # -1 (bajista) a 1 (alcista)
        sentiment = market_data.get("sentiment", 0)  # -1 (negativo) a 1 (positivo)
        
        # Calcular impacto emocional
        emotional_impact = 0
        
        # Alta volatilidad aumenta neuroticism -> FEARFUL o RESTLESS
        if volatility > 0.7:
            emotional_impact -= volatility * self.market_sensitivity
        elif volatility < 0.3:
            emotional_impact += (1 - volatility) * (1 - self.market_sensitivity)
            
        # Tendencia positiva -> HOPEFUL, negativa -> CAUTIOUS/FEARFUL
        emotional_impact += trend * self.extraversion
        
        # Sentimiento positivo -> HOPEFUL/SERENE, negativo -> CAUTIOUS/FEARFUL
        emotional_impact += sentiment * (1 - self.neuroticism)
        
        # Determinar nuevo estado basado en impacto emocional
        if emotional_impact > 0.5:
            if random.random() < 0.7:
                await self.change_emotional_state(EmotionalState.HOPEFUL, "Condiciones de mercado favorables")
            else:
                await self.change_emotional_state(EmotionalState.SERENE, "Mercado estable")
        elif emotional_impact < -0.5:
            if random.random() < 0.6:
                await self.change_emotional_state(EmotionalState.FEARFUL, "Condiciones de mercado adversas")
            else:
                await self.change_emotional_state(EmotionalState.CAUTIOUS, "Incertidumbre en el mercado")
        elif abs(emotional_impact) < 0.2:
            if random.random() < 0.5:
                await self.change_emotional_state(EmotionalState.RESTLESS, "Mercado sin dirección clara")
    
    async def evaluate_trade_opportunity(self, 
                                        symbol: str, 
                                        signal_strength: float,
                                        risk_reward_ratio: float,
                                        available_capital: float,
                                        market_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluar oportunidad de trading con sesgo emocional.
        
        Args:
            symbol: Símbolo a operar
            signal_strength: Fuerza de la señal (0-1)
            risk_reward_ratio: Ratio riesgo/recompensa
            available_capital: Capital disponible
            market_conditions: Condiciones de mercado (opcional)
            
        Returns:
            Decisión de trading con información emocional
        """
        # Aplicar sesgo emocional a la evaluación
        emotional_bias = 0
        
        if self.current_state == EmotionalState.HOPEFUL:
            emotional_bias = 0.2  # Más optimista
        elif self.current_state == EmotionalState.CAUTIOUS:
            emotional_bias = -0.1  # Más precavido
        elif self.current_state == EmotionalState.RESTLESS:
            emotional_bias = 0.1 if random.random() < 0.5 else -0.1  # Variable
        elif self.current_state == EmotionalState.FEARFUL:
            emotional_bias = -0.3  # Muy negativo
            
        # Ajustar señal con sesgo emocional
        adjusted_signal = max(0, min(1, signal_strength + emotional_bias))
        
        # Determinar umbral de decisión basado en estado emocional
        decision_threshold = 0.5
        if self.current_state == EmotionalState.HOPEFUL:
            decision_threshold = 0.4  # Más tolerante
        elif self.current_state == EmotionalState.CAUTIOUS:
            decision_threshold = 0.6  # Más exigente
        elif self.current_state == EmotionalState.FEARFUL:
            decision_threshold = 0.8  # Muy exigente
            
        # Tomar decisión
        take_trade = adjusted_signal >= decision_threshold
        
        # Calcular tamaño de posición
        position_size = 0
        stop_loss_pct = 0
        if take_trade:
            max_position = self.risk_profile.get_max_position_size(available_capital)
            position_confidence = (adjusted_signal - decision_threshold) / (1 - decision_threshold)
            position_size = max_position * position_confidence
            stop_loss_pct = self.risk_profile.get_stop_loss_percentage()
            
        # Registrar decisión
        decision = {
            "symbol": symbol,
            "original_signal": signal_strength,
            "emotional_bias": emotional_bias,
            "adjusted_signal": adjusted_signal,
            "decision_threshold": decision_threshold,
            "take_trade": take_trade,
            "position_size": position_size,
            "stop_loss_pct": stop_loss_pct,
            "emotional_state": self.current_state.name,
            "timestamp": datetime.now().isoformat()
        }
        
        self.decision_history.append(decision)
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
            
        return decision
        
    async def randomize_state(self) -> None:
        """Cambiar aleatoriamente el estado emocional para simulación."""
        states = list(EmotionalState)
        new_state = random.choice(states)
        await self.change_emotional_state(new_state, "Cambio aleatorio para simulación")
        
    def get_personality_profile(self) -> Dict[str, float]:
        """
        Obtener perfil de personalidad.
        
        Returns:
            Diccionario con rasgos de personalidad
        """
        return {
            "neuroticism": round(self.neuroticism, 2),
            "extraversion": round(self.extraversion, 2),
            "openness": round(self.openness, 2),
            "conscientiousness": round(self.conscientiousness, 2),
            "agreeableness": round(self.agreeableness, 2)
        }
        
    async def get_decision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener historial de decisiones.
        
        Args:
            limit: Número máximo de decisiones a devolver
            
        Returns:
            Lista de decisiones
        """
        return self.decision_history[-limit:]