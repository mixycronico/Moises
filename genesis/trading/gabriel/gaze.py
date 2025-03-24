"""
La Mirada de Gabriel - Percepción del mercado con ojos humanos

Este módulo simula cómo un humano percibe el mercado, aplicando sesgos
y distorsiones emocionales a datos objetivos. La percepción en estado
FEARFUL (miedo) es particularmente distorsionada, magnificando todos
los riesgos y minimizando las oportunidades al 100%.
"""

from typing import Dict, Any
import random
import logging
from .soul import Mood

logger = logging.getLogger(__name__)

class Gaze:
    """La mirada de Gabriel sobre los mercados, teñida por sus emociones."""
    
    def __init__(self):
        """Inicializa la percepción del mercado con valores por defecto."""
        self.visions = {
            "turbulence": 0.5,    # Percepción de caos y volatilidad
            "promise": 0.5,       # Visión de oportunidades
            "shadow": 0.5,        # Sensación de riesgo
            "wind": "still",      # Dirección percibida: "rising", "falling", "still"
            "clarity": 0.6,       # Claridad de visión (vs. confusión)
            "trust": 0.5,         # Confianza en los datos
        }
        self.historical_perception = []  # Memoria de percepciones pasadas
        
    async def behold(self, omens: Dict[str, Any], mood: Mood) -> Dict[str, Any]:
        """
        Contempla los presagios del mercado con ojos humanos, influenciados por el estado emocional.
        
        Args:
            omens: Datos objetivos del mercado (volatilidad, tendencia, etc.)
            mood: Estado de ánimo actual que colorea la percepción
            
        Returns:
            Percepción subjetiva del mercado
        """
        # Extraer datos objetivos
        turbulence = omens.get("volatility", 0.5)        # Volatilidad real
        trend = omens.get("trend", "still")              # Tendencia objetiva
        surge = omens.get("volume_change", 0.0)          # Cambio de volumen
        price_direction = omens.get("price_direction", 0.0)  # Dirección del precio (-1.0 a 1.0)
        
        # Guardar percepción anterior para análisis
        self.historical_perception.append(self.visions.copy())
        if len(self.historical_perception) > 10:
            self.historical_perception.pop(0)
            
        # Sesgo hacia lo reciente - cuánto influyen los datos nuevos vs. la percepción anterior
        recency_bias = 0.7  # Valor normal
        
        # --- ALTERACIONES POR ESTADO EMOCIONAL ---
        if mood == Mood.DREAD:  # Estado de MIEDO (100% implementación)
            # En estado de miedo, la percepción es COMPLETAMENTE distorsionada
            recency_bias = 1.0  # Cambios inmediatos y dramáticos en la percepción
            
            # 1. Amplificación extrema de la volatilidad y el riesgo
            turbulence = 1.0  # Siempre ve volatilidad máxima
            self.visions["shadow"] = 1.0  # Siempre ve riesgo máximo
            
            # 2. Colapso completo de la visión de oportunidades
            self.visions["promise"] = 0.0  # Ninguna oportunidad visible
            
            # 3. Distorsión total de tendencias positivas
            if trend == "up":
                # Desconfianza absoluta de tendencias alcistas, las ve como trampas
                self.visions["wind"] = "trap"
                self.visions["clarity"] = 0.1  # Visión extremadamente nublada
                logger.debug("En estado de miedo, ignora completamente la tendencia alcista")
            elif trend == "down":
                # Amplificación de tendencias bajistas, las ve como desplomes
                self.visions["wind"] = "collapsing"
                self.visions["clarity"] = 0.8  # Claridad sólo para ver desastres
                logger.debug("En estado de miedo, percibe caída como colapso inminente")
            else:
                # Ve inestabilidad en la calma
                self.visions["wind"] = "unstable"
                self.visions["clarity"] = 0.3
                logger.debug("En estado de miedo, percibe inestabilidad en la calma")
                
            # 4. Confianza nula en datos positivos
            self.visions["trust"] = 0.0 if price_direction > 0 else 0.9
            
            # 5. Memoria de trauma - recuerda selectivamente experiencias negativas
            negative_memories = [p for p in self.historical_perception 
                               if p.get("wind") in ["falling", "collapsing", "trap"]]
            if negative_memories and random.random() < 0.8:
                worst_memory = min(negative_memories, key=lambda x: x.get("promise", 1.0))
                logger.debug("Memoria traumática activada, reviviendo percepción negativa previa")
                # Contamina la percepción actual con el peor recuerdo
                self.visions["promise"] = worst_memory.get("promise", 0.0)
                self.visions["shadow"] = max(self.visions["shadow"], worst_memory.get("shadow", 0.7))
            
        elif mood == Mood.HOPEFUL:
            # En esperanza, se moderan los riesgos y se destacan oportunidades
            recency_bias = 0.6  # Cambios moderados - más resistente a datos negativos
            if trend == "up":
                turbulence *= 0.7  # Menos percepción de volatilidad en tendencia alcista
            self.visions["trust"] = 0.8 if price_direction > 0 else 0.5
            
        elif mood == Mood.BOLD:
            # En confianza, minimiza riesgos y magnifica oportunidades
            recency_bias = 0.8  # Rápida adaptación a datos positivos
            turbulence *= 0.5  # Mucha menos percepción de volatilidad
            self.visions["shadow"] *= 0.8  # Reducción de la percepción de riesgo
            
        elif mood in [Mood.WARY, Mood.FRAUGHT]:
            # En estados cautelosos, amplifica moderadamente los riesgos
            recency_bias = 0.7
            turbulence *= 1.2  # Amplificación moderada de volatilidad
            self.visions["shadow"] = min(0.9, self.visions["shadow"] * 1.2)
        
        # --- ACTUALIZACIÓN DE LA PERCEPCIÓN ---
        
        # Actualizar turbulencia percibida con sesgo de recencia
        self.visions["turbulence"] = self.visions["turbulence"] * (1 - recency_bias) + turbulence * recency_bias
        
        # Actualizar dirección percibida del mercado
        if mood != Mood.DREAD:  # Ya se actualizó completamente para el estado de miedo
            self.visions["wind"] = (
                "rising" if trend == "up" and surge > 0.1 else
                "falling" if trend == "down" and surge > 0.1 else
                "still" if random.random() < 0.2 else self.visions["wind"]
            )
        
        # Ajustar percepción de oportunidad/riesgo basado en la dirección
        if mood != Mood.DREAD:  # Ya se actualizó completamente para el estado de miedo
            adjust = 0.1
            if self.visions["wind"] == "rising":
                self.visions["promise"] = min(1.0, self.visions["promise"] + adjust)
                self.visions["shadow"] = max(0.1, self.visions["shadow"] - adjust * 0.5)
            elif self.visions["wind"] == "falling":
                self.visions["promise"] = max(0.1, self.visions["promise"] - adjust)
                self.visions["shadow"] = min(1.0, self.visions["shadow"] + adjust)
        
        logger.debug(f"Visión del mercado actualizada: {self.visions} (estado: {mood.name})")
        return self.visions
        
    def get_insights(self) -> Dict[str, Any]:
        """
        Obtiene insights adicionales derivados de la percepción actual.
        
        Returns:
            Insights derivados de la percepción
        """
        return {
            "confidence": max(0.0, min(1.0, (self.visions["promise"] - self.visions["shadow"] + 0.5))),
            "market_sentiment": "bearish" if self.visions["shadow"] > self.visions["promise"] + 0.2 else
                               "bullish" if self.visions["promise"] > self.visions["shadow"] + 0.2 else
                               "neutral",
            "perceived_volatility": self.visions["turbulence"],
            "decision_clarity": self.visions["clarity"],
            "trust_level": self.visions["trust"]
        }