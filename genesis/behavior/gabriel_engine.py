"""
Motor de Comportamiento Gabriel para Aetherion.

Este módulo implementa el motor de comportamiento Gabriel, que permite a Aetherion
simular estados emocionales y comportamiento humano, proporcionando un enfoque
más natural y humano a sus interacciones y decisiones.
"""

import logging
import random
import datetime
import json
import os
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

# Configurar logging
logger = logging.getLogger(__name__)

# Directorio para almacenamiento persistente
ENGINE_DIR = os.path.join("data", "aetherion", "gabriel")
os.makedirs(ENGINE_DIR, exist_ok=True)

class EmotionalState(Enum):
    """Estados emocionales básicos para Gabriel."""
    SERENE = auto()     # Calma - máxima capacidad de evaluación objetiva
    HOPEFUL = auto()    # Esperanzado - ligero optimismo
    CAUTIOUS = auto()   # Cauteloso - evaluación más conservadora
    RESTLESS = auto()   # Inquieto - cierta ansiedad e impaciencia
    FEARFUL = auto()    # Temeroso - predomina el miedo, muy defensivo
    
    def __str__(self) -> str:
        """Obtener nombre legible del estado."""
        return self.name

class MarketEvent(Enum):
    """Tipos de eventos de mercado que pueden afectar el estado emocional."""
    PRICE_SURGE = auto()          # Subida brusca de precio
    PRICE_DROP = auto()           # Caída brusca de precio
    VOLATILITY_INCREASE = auto()  # Aumento de volatilidad
    VOLATILITY_DECREASE = auto()  # Disminución de volatilidad
    POSITIVE_NEWS = auto()        # Noticias positivas
    NEGATIVE_NEWS = auto()        # Noticias negativas
    TREND_REVERSAL = auto()       # Cambio de tendencia
    PATTERN_RECOGNITION = auto()  # Reconocimiento de patrón
    
    def __str__(self) -> str:
        """Obtener nombre legible del evento."""
        return self.name

class RiskProfile(Enum):
    """Perfiles de riesgo para decisiones de trading."""
    CONSERVATIVE = auto()   # Muy bajo riesgo
    MODERATE = auto()       # Riesgo moderado
    AGGRESSIVE = auto()     # Alto riesgo
    ADAPTIVE = auto()       # Adaptativo según condiciones
    
    def __str__(self) -> str:
        """Obtener nombre legible del perfil."""
        return self.name

class GabrielBehaviorEngine:
    """
    Motor de comportamiento Gabriel para Aetherion.
    
    Esta clase implementa el motor de comportamiento Gabriel, que permite
    a Aetherion simular estados emocionales y comportamiento humano, proporcionando
    un enfoque más natural y humano a sus interacciones y decisiones.
    """
    
    def __init__(self):
        """Inicializar motor de comportamiento Gabriel."""
        # Estado emocional actual
        self._emotional_state = EmotionalState.SERENE
        
        # Intensidad del estado (0.0 - 1.0)
        self._intensity = 0.5
        
        # Características emocionales
        self._traits = {
            "optimism": 0.6,       # Optimismo (0.0 - 1.0)
            "courage": 0.6,        # Valentía (0.0 - 1.0)
            "patience": 0.7,       # Paciencia (0.0 - 1.0)
            "adaptability": 0.8,   # Adaptabilidad (0.0 - 1.0)
            "risk_tolerance": 0.5  # Tolerancia al riesgo (0.0 - 1.0)
        }
        
        # Perfil de riesgo
        self._risk_profile = RiskProfile.ADAPTIVE
        
        # Historial de estados
        self._state_history = []
        
        # Parámetros de evolución
        self._stability = 0.7      # Estabilidad emocional (0.0 - 1.0)
        self._reactivity = 0.6     # Reactividad a eventos (0.0 - 1.0)
        self._recovery_rate = 0.3  # Tasa de recuperación (0.0 - 1.0)
        
        # Cargar estado persistente
        self._load_state()
        
        logger.info(f"GabrielBehaviorEngine inicializado con estado {self._emotional_state}")
    
    def _load_state(self) -> None:
        """Cargar estado desde almacenamiento persistente."""
        try:
            state_file = os.path.join(ENGINE_DIR, "gabriel_state.json")
            
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Cargar estado emocional
                    if "emotional_state" in data:
                        self._emotional_state = EmotionalState[data["emotional_state"]]
                    
                    # Cargar intensidad
                    if "intensity" in data:
                        self._intensity = data["intensity"]
                    
                    # Cargar características
                    if "traits" in data:
                        self._traits.update(data["traits"])
                    
                    # Cargar perfil de riesgo
                    if "risk_profile" in data:
                        self._risk_profile = RiskProfile[data["risk_profile"]]
                    
                    # Cargar historial
                    if "state_history" in data:
                        self._state_history = data["state_history"]
                    
                    logger.info(f"Estado Gabriel cargado: {self._emotional_state}")
        except Exception as e:
            logger.error(f"Error al cargar estado Gabriel: {e}")
    
    def _save_state(self) -> None:
        """Guardar estado a almacenamiento persistente."""
        try:
            state_file = os.path.join(ENGINE_DIR, "gabriel_state.json")
            
            # Preparar datos
            data = {
                "emotional_state": self._emotional_state.name,
                "intensity": self._intensity,
                "traits": self._traits,
                "risk_profile": self._risk_profile.name,
                "state_history": self._state_history[-50:],  # Guardar solo los últimos 50 estados
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            # Guardar a archivo
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Estado Gabriel guardado")
        except Exception as e:
            logger.error(f"Error al guardar estado Gabriel: {e}")
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """
        Obtener estado emocional actual.
        
        Returns:
            Diccionario con información del estado emocional
        """
        return {
            "state": str(self._emotional_state),
            "intensity": self._intensity,
            "optimism": self._traits["optimism"],
            "courage": self._traits["courage"],
            "patience": self._traits["patience"]
        }
    
    def get_risk_profile(self) -> Dict[str, Any]:
        """
        Obtener perfil de riesgo actual.
        
        Returns:
            Diccionario con información del perfil de riesgo
        """
        return {
            "profile": str(self._risk_profile),
            "risk_tolerance": self._traits["risk_tolerance"],
            "adaptability": self._traits["adaptability"]
        }
    
    def set_emotional_state(self, state: EmotionalState, intensity: float, reason: str = "") -> None:
        """
        Establecer estado emocional.
        
        Args:
            state: Nuevo estado emocional
            intensity: Intensidad del estado (0.0 - 1.0)
            reason: Razón del cambio
        """
        # Guardar estado anterior
        old_state = {
            "state": str(self._emotional_state),
            "intensity": self._intensity,
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": reason
        }
        self._state_history.append(old_state)
        
        # Actualizar estado
        self._emotional_state = state
        self._intensity = max(0.0, min(intensity, 1.0))  # Limitar a [0, 1]
        
        # Actualizar perfil de riesgo basado en estado
        self._update_risk_profile()
        
        # Guardar cambios
        self._save_state()
        
        logger.info(f"Estado emocional cambiado a {state} (intensidad: {self._intensity:.2f}): {reason}")
    
    def _update_risk_profile(self) -> None:
        """Actualizar perfil de riesgo basado en estado emocional."""
        # Mapeo de estados a perfiles de riesgo predeterminados
        state_profile_map = {
            EmotionalState.SERENE: RiskProfile.MODERATE,
            EmotionalState.HOPEFUL: RiskProfile.AGGRESSIVE,
            EmotionalState.CAUTIOUS: RiskProfile.MODERATE,
            EmotionalState.RESTLESS: RiskProfile.CONSERVATIVE,
            EmotionalState.FEARFUL: RiskProfile.CONSERVATIVE
        }
        
        # Si estamos en modo adaptativo, ajustar según el estado
        if self._risk_profile == RiskProfile.ADAPTIVE:
            new_profile = state_profile_map.get(self._emotional_state, RiskProfile.MODERATE)
            
            # Aplicar componente aleatorio basado en traits
            if random.random() < self._traits["adaptability"] * 0.2:
                # Pequeña probabilidad de cambiar
                profiles = list(RiskProfile)
                profiles.remove(RiskProfile.ADAPTIVE)  # Excluir adaptativo para evitar recursión
                new_profile = random.choice(profiles)
            
            logger.debug(f"Perfil de riesgo adaptado a {new_profile}")
    
    def randomize_state(self) -> None:
        """Randomizar estado emocional con componente aleatorio."""
        # Elegir estado aleatorio
        states = list(EmotionalState)
        state = random.choice(states)
        
        # Generar intensidad con distribución normal
        intensity = random.normalvariate(0.5, 0.2)
        intensity = max(0.1, min(intensity, 0.9))  # Limitar a [0.1, 0.9]
        
        # Establecer nuevo estado
        self.set_emotional_state(state, intensity, "Randomización programada")
    
    def process_market_event(self, event_type: MarketEvent, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar evento de mercado y actualizar estado emocional.
        
        Args:
            event_type: Tipo de evento
            data: Datos adicionales del evento
        
        Returns:
            Resultado del procesamiento
        """
        # Extraer magnitud del evento (0.0 - 1.0)
        magnitude = data.get("magnitude", 0.5)
        
        # Extraer dirección del evento (-1.0 - 1.0)
        direction = data.get("direction", 0.0)
        
        # Calcular impacto basado en reactividad
        impact = magnitude * self._reactivity
        
        # Determinar cambio de estado e intensidad
        new_state = self._emotional_state
        new_intensity = self._intensity
        
        # Mapa de transiciones por tipo de evento
        if event_type == MarketEvent.PRICE_SURGE:
            if direction > 0:  # Bueno para nosotros
                if random.random() < self._traits["optimism"]:
                    new_state = EmotionalState.HOPEFUL
                else:
                    new_state = EmotionalState.SERENE
            else:  # Malo para nosotros
                if random.random() < 0.7:
                    new_state = EmotionalState.CAUTIOUS
                else:
                    new_state = EmotionalState.RESTLESS
        
        elif event_type == MarketEvent.PRICE_DROP:
            if direction < 0:  # Bueno para nosotros (si tenemos posición short)
                if random.random() < self._traits["optimism"]:
                    new_state = EmotionalState.HOPEFUL
                else:
                    new_state = EmotionalState.SERENE
            else:  # Malo para nosotros
                if random.random() < self._traits["courage"]:
                    new_state = EmotionalState.CAUTIOUS
                else:
                    new_state = EmotionalState.FEARFUL
        
        elif event_type == MarketEvent.VOLATILITY_INCREASE:
            if random.random() < self._traits["risk_tolerance"]:
                new_state = EmotionalState.HOPEFUL
            else:
                new_state = EmotionalState.RESTLESS
        
        elif event_type == MarketEvent.VOLATILITY_DECREASE:
            if random.random() < self._traits["patience"]:
                new_state = EmotionalState.SERENE
            else:
                new_state = EmotionalState.CAUTIOUS
        
        elif event_type == MarketEvent.POSITIVE_NEWS:
            if random.random() < self._traits["optimism"]:
                new_state = EmotionalState.HOPEFUL
            else:
                new_state = EmotionalState.SERENE
        
        elif event_type == MarketEvent.NEGATIVE_NEWS:
            if random.random() < self._traits["courage"]:
                new_state = EmotionalState.CAUTIOUS
            else:
                new_state = EmotionalState.FEARFUL
        
        elif event_type == MarketEvent.TREND_REVERSAL:
            if random.random() < self._traits["adaptability"]:
                new_state = EmotionalState.CAUTIOUS
            else:
                new_state = EmotionalState.RESTLESS
        
        elif event_type == MarketEvent.PATTERN_RECOGNITION:
            if random.random() < self._traits["optimism"]:
                new_state = EmotionalState.HOPEFUL
            else:
                new_state = EmotionalState.SERENE
        
        # Ajustar intensidad basada en el impacto
        new_intensity = min(1.0, max(0.1, self._intensity + impact * direction))
        
        # Aplicar cambio
        event_desc = f"Evento {event_type}, magnitud {magnitude:.2f}, dirección {direction:.2f}"
        self.set_emotional_state(new_state, new_intensity, event_desc)
        
        # Registrar para evolución de consciencia
        from genesis.consciousness.states.consciousness_states import get_consciousness_states
        states = get_consciousness_states()
        states.record_activity("emotional_responses")
        
        # Resultado
        return {
            "old_state": self._state_history[-1]["state"] if self._state_history else "UNKNOWN",
            "new_state": str(new_state),
            "old_intensity": self._state_history[-1]["intensity"] if self._state_history else 0.0,
            "new_intensity": new_intensity,
            "impact": impact,
            "event": str(event_type)
        }
    
    def evaluate_trade_opportunity(self, 
                                 asset: str, 
                                 signal_type: str, 
                                 confidence: float,
                                 **kwargs) -> Dict[str, Any]:
        """
        Evaluar oportunidad de trading con componente emocional.
        
        Args:
            asset: Activo a operar
            signal_type: Tipo de señal (BUY, SELL)
            confidence: Confianza en la señal (0.0 - 1.0)
            **kwargs: Argumentos adicionales
        
        Returns:
            Resultado de la evaluación
        """
        # Factores adicionales
        signal_strength = kwargs.get("signal_strength", 0.5)
        risk_reward_ratio = kwargs.get("risk_reward_ratio", 1.0)
        available_capital = kwargs.get("available_capital", 1000.0)
        
        # Base inicial (puramente racional)
        base_score = confidence * signal_strength * min(2.0, risk_reward_ratio)
        
        # Factores emocionales
        emotional_factors = {
            EmotionalState.SERENE: 1.0,       # Evaluación objetiva
            EmotionalState.HOPEFUL: 1.2,      # Optimista, más propenso a tomar el trade
            EmotionalState.CAUTIOUS: 0.8,     # Cauteloso, menos propenso
            EmotionalState.RESTLESS: 0.9,     # Ligeramente defensivo
            EmotionalState.FEARFUL: 0.5       # Muy defensivo, mucho menos propenso
        }
        
        # Ajustar por estado emocional
        emotional_factor = emotional_factors.get(self._emotional_state, 1.0)
        emotional_factor = 1.0 + (emotional_factor - 1.0) * self._intensity
        
        # Ajustar por perfil de riesgo
        risk_factors = {
            RiskProfile.CONSERVATIVE: 0.7,
            RiskProfile.MODERATE: 1.0,
            RiskProfile.AGGRESSIVE: 1.3,
            RiskProfile.ADAPTIVE: 1.0  # Ya ajustado por estado emocional
        }
        risk_factor = risk_factors.get(self._risk_profile, 1.0)
        
        # Calcular score final
        final_score = base_score * emotional_factor * risk_factor
        
        # Determinar decisión
        threshold = 0.5  # Umbral mínimo para aceptar
        decision = "ACCEPT" if final_score >= threshold else "REJECT"
        
        # Si es acceptable, calcular tamaño de posición
        position_size = 0.0
        if decision == "ACCEPT":
            # Base racional: % basado en confianza y riesgo
            base_size = min(0.2, confidence * 0.1 * risk_reward_ratio)
            
            # Ajuste emocional
            position_size = base_size * emotional_factor * available_capital
        
        # Registrar para evolución de consciencia
        from genesis.consciousness.states.consciousness_states import get_consciousness_states
        states = get_consciousness_states()
        states.record_activity("strategy_evaluations")
        
        # Resultado
        return {
            "asset": asset,
            "signal_type": signal_type,
            "base_score": base_score,
            "emotional_factor": emotional_factor,
            "risk_factor": risk_factor,
            "final_score": final_score,
            "decision": decision,
            "position_size": position_size,
            "emotional_state": str(self._emotional_state),
            "explanation": self._generate_explanation(decision, final_score, emotional_factor)
        }
    
    def _generate_explanation(self, decision: str, score: float, emotional_factor: float) -> str:
        """
        Generar explicación textual de la decisión.
        
        Args:
            decision: Decisión tomada
            score: Puntuación final
            emotional_factor: Factor emocional aplicado
        
        Returns:
            Explicación textual
        """
        if decision == "ACCEPT":
            if self._emotional_state == EmotionalState.SERENE:
                return f"Análisis sereno indica oportunidad favorable (score: {score:.2f})"
            elif self._emotional_state == EmotionalState.HOPEFUL:
                return f"Optimismo respaldado por señales positivas (score: {score:.2f})"
            elif self._emotional_state == EmotionalState.CAUTIOUS:
                return f"A pesar de la cautela, los indicadores son suficientemente sólidos (score: {score:.2f})"
            elif self._emotional_state == EmotionalState.RESTLESS:
                return f"Señales lo bastante fuertes para superar la inquietud (score: {score:.2f})"
            elif self._emotional_state == EmotionalState.FEARFUL:
                return f"A pesar del miedo, esta oportunidad es demasiado buena para ignorarla (score: {score:.2f})"
        else:
            if self._emotional_state == EmotionalState.SERENE:
                return f"Análisis objetivo muestra que no cumple los requisitos (score: {score:.2f})"
            elif self._emotional_state == EmotionalState.HOPEFUL:
                return f"Aunque hay optimismo, los indicadores no son suficientemente fuertes (score: {score:.2f})"
            elif self._emotional_state == EmotionalState.CAUTIOUS:
                return f"La cautela aconseja esperar mejor oportunidad (score: {score:.2f})"
            elif self._emotional_state == EmotionalState.RESTLESS:
                return f"La inquietud amplifica los riesgos percibidos (score: {score:.2f})"
            elif self._emotional_state == EmotionalState.FEARFUL:
                return f"El miedo predomina, mejor esperar confirmación más clara (score: {score:.2f})"
        
        # Default
        return f"Decisión basada en análisis objetivo y estado emocional (score: {score:.2f})"
    
    def recover_emotional_stability(self) -> Dict[str, Any]:
        """
        Recuperar estabilidad emocional gradualmente.
        
        Returns:
            Resultado de la recuperación
        """
        # Solo aplica a estados no serenos
        if self._emotional_state == EmotionalState.SERENE:
            return {
                "changed": False,
                "reason": "Ya en estado sereno"
            }
        
        # Probabilidad de recuperación basada en factores
        recovery_chance = self._recovery_rate * self._stability
        
        # Mayor probabilidad si la intensidad es baja
        if self._intensity < 0.3:
            recovery_chance *= 1.5
        
        # Intentar recuperación
        if random.random() < recovery_chance:
            old_state = self._emotional_state
            old_intensity = self._intensity
            
            # Transicionar a estado más sereno
            if self._emotional_state == EmotionalState.FEARFUL:
                new_state = EmotionalState.CAUTIOUS
            elif self._emotional_state == EmotionalState.RESTLESS:
                new_state = EmotionalState.CAUTIOUS
            elif self._emotional_state == EmotionalState.CAUTIOUS:
                new_state = EmotionalState.SERENE
            elif self._emotional_state == EmotionalState.HOPEFUL:
                new_state = EmotionalState.SERENE
            else:
                new_state = EmotionalState.SERENE
            
            # Reducir intensidad
            new_intensity = max(0.1, self._intensity - 0.2)
            
            # Aplicar cambio
            self.set_emotional_state(new_state, new_intensity, "Recuperación natural")
            
            return {
                "changed": True,
                "old_state": str(old_state),
                "new_state": str(new_state),
                "old_intensity": old_intensity,
                "new_intensity": new_intensity,
                "reason": "Recuperación natural"
            }
        
        return {
            "changed": False,
            "reason": "No se alcanzó umbral de recuperación"
        }
    
    def get_state_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener historial de estados emocionales.
        
        Args:
            limit: Número máximo de estados a retornar
        
        Returns:
            Lista de estados históricos
        """
        return self._state_history[-limit:]
    
    def get_state_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor de comportamiento.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "emotional_state": str(self._emotional_state),
            "intensity": self._intensity,
            "traits": self._traits.copy(),
            "risk_profile": str(self._risk_profile),
            "stability": self._stability,
            "reactivity": self._reactivity,
            "recovery_rate": self._recovery_rate,
            "state_history_length": len(self._state_history)
        }

# Instancia global para acceso conveniente
_behavior_engine = None

def get_behavior_engine() -> GabrielBehaviorEngine:
    """
    Obtener instancia global del motor de comportamiento Gabriel.
    
    Returns:
        Instancia del motor de comportamiento
    """
    global _behavior_engine
    
    if _behavior_engine is None:
        _behavior_engine = GabrielBehaviorEngine()
    
    return _behavior_engine