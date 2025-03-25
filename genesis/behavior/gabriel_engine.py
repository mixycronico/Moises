"""
Motor de Comportamiento Gabriel para Aetherion.

Este módulo implementa el motor de comportamiento humano Gabriel, que simula
estados emocionales y personalidad para humanizar las respuestas de Aetherion
y añadir un componente emocional a las decisiones de trading.
"""

import logging
import datetime
import random
from typing import Dict, Any, List, Optional, Tuple

# Configurar logging
logger = logging.getLogger(__name__)

class GabrielBehaviorEngine:
    """
    Motor de comportamiento humano para simular estados emocionales.
    
    Gabriel simula comportamientos humanos como:
    - Estados emocionales (sereno, esperanzado, cauteloso, inquieto, temeroso)
    - Toma de decisiones influenciada por emociones
    - Cambios de humor basados en eventos del mercado
    - Personalidad adaptativa
    """
    
    # Estados emocionales posibles
    SERENE = "SERENE"       # Calma - máxima capacidad de evaluación objetiva
    HOPEFUL = "HOPEFUL"     # Esperanzado - ligero optimismo
    CAUTIOUS = "CAUTIOUS"   # Cauteloso - evaluación más conservadora
    RESTLESS = "RESTLESS"   # Inquieto - cierta ansiedad e impaciencia
    FEARFUL = "FEARFUL"     # Temeroso - predomina el miedo, muy defensivo
    
    # Características de personalidad
    PERSONALITY_TRAITS = [
        "risk_tolerance",       # Tolerancia al riesgo (0.0 a 1.0)
        "patience",             # Paciencia (0.0 a 1.0)
        "adaptability",         # Adaptabilidad (0.0 a 1.0)
        "emotional_stability",  # Estabilidad emocional (0.0 a 1.0)
        "optimism"              # Optimismo (0.0 a 1.0)
    ]
    
    def __init__(self, personality_seed: Optional[int] = None):
        """
        Inicializar motor de comportamiento.
        
        Args:
            personality_seed: Semilla para generar personalidad
        """
        # Estado emocional actual
        self.emotional_state = {
            "state": self.SERENE,  # Estado inicial: sereno
            "intensity": 0.5,      # Intensidad media
            "last_change": datetime.datetime.now(),
            "duration": 0          # Duración en segundos
        }
        
        # Historial de cambios emocionales
        self.emotional_history = []
        self.emotional_history.append({
            "state": self.emotional_state["state"],
            "intensity": self.emotional_state["intensity"],
            "timestamp": datetime.datetime.now(),
            "reason": "Inicialización"
        })
        
        # Generar personalidad
        if personality_seed is None:
            personality_seed = random.randint(1, 100000)
        
        self.personality_seed = personality_seed
        self.personality = self._generate_personality(personality_seed)
        
        # Contadores e indicadores
        self.emotional_changes = 0
        self.decisions_made = 0
        self.market_events_processed = 0
        
        logger.info(f"GabrielBehaviorEngine inicializado con personalidad {personality_seed}")
    
    def _generate_personality(self, seed: int) -> Dict[str, float]:
        """
        Generar rasgos de personalidad basados en semilla.
        
        Args:
            seed: Semilla para generación
            
        Returns:
            Diccionario con rasgos de personalidad
        """
        # Usar semilla para reproducibilidad
        random.seed(seed)
        
        # Generar rasgos
        personality = {}
        for trait in self.PERSONALITY_TRAITS:
            personality[trait] = round(random.uniform(0.2, 0.8), 2)
        
        # Restaurar aleatoriedad
        random.seed()
        
        return personality
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """
        Obtener estado emocional actual.
        
        Returns:
            Estado emocional actual
        """
        # Actualizar duración
        now = datetime.datetime.now()
        duration = (now - self.emotional_state["last_change"]).total_seconds()
        self.emotional_state["duration"] = duration
        
        return self.emotional_state
    
    def change_emotional_state(self, new_state: str, intensity: float = 0.5, 
                             reason: str = "Cambio manual") -> bool:
        """
        Cambiar estado emocional.
        
        Args:
            new_state: Nuevo estado (SERENE, HOPEFUL, CAUTIOUS, RESTLESS, FEARFUL)
            intensity: Intensidad del estado (0.0 a 1.0)
            reason: Razón del cambio
            
        Returns:
            True si el cambio fue exitoso
        """
        # Validar estado
        valid_states = [self.SERENE, self.HOPEFUL, self.CAUTIOUS, self.RESTLESS, self.FEARFUL]
        if new_state not in valid_states:
            return False
        
        # Validar intensidad
        intensity = max(0.0, min(1.0, intensity))
        
        # Actualizar estado
        old_state = self.emotional_state["state"]
        old_intensity = self.emotional_state["intensity"]
        
        self.emotional_state["state"] = new_state
        self.emotional_state["intensity"] = intensity
        self.emotional_state["last_change"] = datetime.datetime.now()
        self.emotional_state["duration"] = 0
        
        # Registrar cambio
        self.emotional_history.append({
            "state": new_state,
            "intensity": intensity,
            "timestamp": datetime.datetime.now(),
            "reason": reason,
            "previous_state": old_state,
            "previous_intensity": old_intensity
        })
        
        # Actualizar contador
        self.emotional_changes += 1
        
        logger.info(f"Estado emocional cambiado de {old_state} a {new_state} (intensidad: {intensity:.2f})")
        return True
    
    def process_market_event(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar evento del mercado y actualizar estado emocional.
        
        Args:
            event_type: Tipo de evento (price_change, news, trend_change, etc.)
            data: Datos del evento
            
        Returns:
            Estado emocional actualizado
        """
        # Actualizar contador
        self.market_events_processed += 1
        
        # Determinar impacto emocional según tipo de evento
        emotional_impact = self._calculate_emotional_impact(event_type, data)
        
        # Aplicar cambio emocional si hay suficiente impacto
        if abs(emotional_impact["magnitude"]) >= 0.2:
            self.change_emotional_state(
                emotional_impact["state"],
                emotional_impact["intensity"],
                f"Evento de mercado: {event_type}"
            )
        
        return self.get_emotional_state()
    
    def _calculate_emotional_impact(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcular impacto emocional de un evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            
        Returns:
            Impacto emocional
        """
        # Valores por defecto
        impact = {
            "magnitude": 0.0,       # Magnitud del impacto (-1.0 a 1.0)
            "state": self.SERENE,   # Estado emocional resultante
            "intensity": 0.5        # Intensidad emocional resultante
        }
        
        # Calcular impacto según tipo de evento
        if event_type == "price_change":
            # Cambio porcentual del precio
            if "percent_change" in data:
                percent_change = data["percent_change"]
                
                # Escalar según tolerancia al riesgo y estabilidad emocional
                risk_factor = self.personality["risk_tolerance"]
                stability_factor = self.personality["emotional_stability"]
                
                # Calcular magnitud (normalizada entre -1 y 1)
                magnitude = percent_change / 10.0  # 10% cambio = magnitud 1.0
                magnitude *= (2.0 - risk_factor - stability_factor) / 2.0  # Ajustar por personalidad
                
                impact["magnitude"] = max(-1.0, min(1.0, magnitude))
                
                # Determinar estado resultante
                if magnitude >= 0.5:
                    impact["state"] = self.HOPEFUL
                    impact["intensity"] = 0.5 + (magnitude - 0.5)
                elif magnitude >= 0.2:
                    impact["state"] = self.SERENE
                    impact["intensity"] = 0.5 + (magnitude - 0.2) * 2
                elif magnitude <= -0.5:
                    impact["state"] = self.FEARFUL
                    impact["intensity"] = 0.5 + (-magnitude - 0.5)
                elif magnitude <= -0.2:
                    impact["state"] = self.CAUTIOUS
                    impact["intensity"] = 0.5 + (-magnitude - 0.2) * 2
                else:
                    # Cambio pequeño, mantener estado actual pero ajustar intensidad
                    impact["state"] = self.emotional_state["state"]
                    impact["intensity"] = self.emotional_state["intensity"] + magnitude / 4
        
        elif event_type == "market_trend":
            if "trend" in data:
                trend = data["trend"]  # "bullish", "bearish", "neutral", "volatile"
                
                # Ajustar según optimismo
                optimism_factor = self.personality["optimism"]
                
                if trend == "bullish":
                    impact["magnitude"] = 0.3 + (optimism_factor * 0.4)
                    impact["state"] = self.HOPEFUL
                elif trend == "bearish":
                    impact["magnitude"] = -0.3 - ((1 - optimism_factor) * 0.4)
                    impact["state"] = self.CAUTIOUS if optimism_factor > 0.5 else self.FEARFUL
                elif trend == "volatile":
                    impact["magnitude"] = -0.2 - ((1 - self.personality["stability"]) * 0.3)
                    impact["state"] = self.RESTLESS
                
                # Ajustar intensidad
                impact["intensity"] = 0.5 + (abs(impact["magnitude"]) * 0.5)
        
        # Limitar intensidad
        impact["intensity"] = max(0.1, min(1.0, impact["intensity"]))
        
        return impact
    
    def get_risk_profile(self) -> Dict[str, Any]:
        """
        Obtener perfil de riesgo actual basado en personalidad y estado emocional.
        
        Returns:
            Perfil de riesgo
        """
        # Factores base de personalidad
        base_risk_tolerance = self.personality["risk_tolerance"]
        base_patience = self.personality["patience"]
        
        # Ajustar según estado emocional
        emotional_factor = self._calculate_emotional_factor()
        
        # Calcular perfil de riesgo
        risk_tolerance = base_risk_tolerance * emotional_factor["risk_multiplier"]
        patience = base_patience * emotional_factor["patience_multiplier"]
        
        # Determinar perfil general
        if risk_tolerance >= 0.7:
            risk_profile = "aggressive"
        elif risk_tolerance >= 0.4:
            risk_profile = "moderate"
        else:
            risk_profile = "conservative"
        
        return {
            "profile": risk_profile,
            "risk_tolerance": risk_tolerance,
            "patience": patience,
            "emotional_state": self.emotional_state["state"],
            "emotional_intensity": self.emotional_state["intensity"]
        }
    
    def _calculate_emotional_factor(self) -> Dict[str, float]:
        """
        Calcular factores de ajuste basados en estado emocional.
        
        Returns:
            Factores de ajuste
        """
        state = self.emotional_state["state"]
        intensity = self.emotional_state["intensity"]
        
        # Valores por defecto
        factors = {
            "risk_multiplier": 1.0,
            "patience_multiplier": 1.0,
            "decision_speed_multiplier": 1.0
        }
        
        # Ajustar según estado
        if state == self.SERENE:
            # Estado sereno - comportamiento más equilibrado
            factors["risk_multiplier"] = 1.0
            factors["patience_multiplier"] = 1.0 + (intensity * 0.2)
            factors["decision_speed_multiplier"] = 1.0
        
        elif state == self.HOPEFUL:
            # Estado esperanzado - más riesgo, menos paciencia
            factors["risk_multiplier"] = 1.0 + (intensity * 0.3)
            factors["patience_multiplier"] = 1.0 - (intensity * 0.1)
            factors["decision_speed_multiplier"] = 1.0 + (intensity * 0.2)
        
        elif state == self.CAUTIOUS:
            # Estado cauteloso - menos riesgo, más paciencia
            factors["risk_multiplier"] = 1.0 - (intensity * 0.3)
            factors["patience_multiplier"] = 1.0 + (intensity * 0.2)
            factors["decision_speed_multiplier"] = 1.0 - (intensity * 0.2)
        
        elif state == self.RESTLESS:
            # Estado inquieto - más riesgo, menos paciencia
            factors["risk_multiplier"] = 1.0 + (intensity * 0.2)
            factors["patience_multiplier"] = 1.0 - (intensity * 0.3)
            factors["decision_speed_multiplier"] = 1.0 + (intensity * 0.3)
        
        elif state == self.FEARFUL:
            # Estado temeroso - mucho menos riesgo, variable en paciencia
            factors["risk_multiplier"] = 1.0 - (intensity * 0.5)
            factors["patience_multiplier"] = 1.0 - (intensity * 0.3)  # Menos paciencia por ansiedad
            factors["decision_speed_multiplier"] = 0.8  # Más lento en decisiones
        
        return factors
    
    def evaluate_trade_opportunity(self, asset: str, signal_type: str, 
                                 confidence: float, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluar oportunidad de trading con factor emocional.
        
        Args:
            asset: Activo a operar
            signal_type: Tipo de señal (buy, sell, hold)
            confidence: Confianza en la señal (0.0 a 1.0)
            context: Contexto adicional
            
        Returns:
            Decisión con factor emocional
        """
        if context is None:
            context = {}
        
        # Actualizar contador
        self.decisions_made += 1
        
        # Obtener perfil de riesgo actual
        risk_profile = self.get_risk_profile()
        
        # Calcular factor emocional
        emotional_factors = self._calculate_emotional_factor()
        
        # Ajustar confianza según estado emocional
        adjusted_confidence = confidence
        
        if self.emotional_state["state"] == self.HOPEFUL:
            # Más optimista - aumenta confianza en señales de compra
            if signal_type == "buy":
                adjusted_confidence *= (1.0 + (self.emotional_state["intensity"] * 0.2))
            elif signal_type == "sell":
                adjusted_confidence *= (1.0 - (self.emotional_state["intensity"] * 0.1))
        
        elif self.emotional_state["state"] == self.FEARFUL:
            # Más temeroso - aumenta confianza en señales de venta
            if signal_type == "sell":
                adjusted_confidence *= (1.0 + (self.emotional_state["intensity"] * 0.2))
            elif signal_type == "buy":
                adjusted_confidence *= (1.0 - (self.emotional_state["intensity"] * 0.3))
        
        elif self.emotional_state["state"] == self.CAUTIOUS:
            # Más cauteloso - reduce confianza en general
            adjusted_confidence *= (1.0 - (self.emotional_state["intensity"] * 0.15))
        
        # Limitar confianza
        adjusted_confidence = max(0.1, min(1.0, adjusted_confidence))
        
        # Determinar decisión final
        decision = {
            "original_signal": signal_type,
            "original_confidence": confidence,
            "adjusted_confidence": adjusted_confidence,
            "emotional_state": self.emotional_state["state"],
            "emotional_intensity": self.emotional_state["intensity"],
            "risk_profile": risk_profile["profile"]
        }
        
        # Modificar decisión según estado emocional
        threshold = 0.5 - ((risk_profile["risk_tolerance"] - 0.5) * 0.3)  # Umbral ajustado por tolerancia
        
        if adjusted_confidence >= threshold:
            decision["decision"] = signal_type
            decision["execution_speed"] = "normal"
            
            if self.emotional_state["state"] == self.RESTLESS:
                decision["execution_speed"] = "fast"
            elif self.emotional_state["state"] == self.CAUTIOUS:
                decision["execution_speed"] = "slow"
                
            decision["position_size"] = min(1.0, adjusted_confidence * risk_profile["risk_tolerance"] * 1.5)
        else:
            decision["decision"] = "hold"
            decision["execution_speed"] = "n/a"
            decision["position_size"] = 0.0
            decision["reason"] = "Confianza insuficiente dados factores emocionales"
        
        return decision
    
    def get_emotional_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener historial de cambios emocionales.
        
        Args:
            limit: Número máximo de cambios a devolver
            
        Returns:
            Historial de cambios emocionales
        """
        # Limitar cantidad
        limit = min(limit, len(self.emotional_history))
        
        # Devolver los más recientes
        return self.emotional_history[-limit:]
    
    def get_personality_traits(self) -> Dict[str, float]:
        """
        Obtener rasgos de personalidad.
        
        Returns:
            Rasgos de personalidad
        """
        return self.personality
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor.
        
        Returns:
            Estadísticas
        """
        return {
            "emotional_changes": self.emotional_changes,
            "decisions_made": self.decisions_made,
            "market_events_processed": self.market_events_processed,
            "current_state": self.emotional_state["state"],
            "personality_seed": self.personality_seed
        }