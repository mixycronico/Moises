"""
La Mirada de Gabriel - Percepción e interpretación de la realidad

Este módulo implementa el componente "Mirada" (Gaze) del motor de comportamiento humano Gabriel,
responsable de la percepción e interpretación de la información del mercado,
filtrando y procesando los datos a través de un lente emocional humano.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import random
from enum import Enum, auto
import math

from genesis.trading.gabriel.soul import Mood

logger = logging.getLogger(__name__)

class Perspective(Enum):
    """Perspectivas de mercado que puede adoptar la Mirada de Gabriel."""
    BULLISH = auto()      # Alcista - ve oportunidades de subida
    CAUTIOUSLY_BULLISH = auto()  # Cautelosamente alcista
    NEUTRAL = auto()      # Neutral - sin tendencia clara
    CAUTIOUSLY_BEARISH = auto()  # Cautelosamente bajista
    BEARISH = auto()      # Bajista - ve riesgos de bajada
    CONFUSED = auto()     # Confundido - no puede determinar tendencia

class Gaze:
    """
    La Mirada (Gaze) de Gabriel, responsable de la percepción del mercado.
    
    Este componente simula la forma en que un humano interpreta la información:
    - Filtra datos según estado emocional (ignora ciertos patrones cuando está temeroso)
    - Detecta patrones que serían visibles para un humano
    - Mantiene atención selectiva (enfoca en ciertos tipos de señales)
    - Tiene sesgos de percepción que varían con el tiempo
    - Evoluciona lentamente en lugar de cambiar instantáneamente
    """
    
    def __init__(self):
        """Inicializar el componente Mirada con sus propiedades esenciales."""
        # Perspectiva actual y su confianza
        self.current_perspective = Perspective.NEUTRAL
        self.perspective_confidence = 0.5  # 0-1
        
        # Propiedades de percepción
        self.pattern_recognition = 0.7  # 0-1, capacidad de reconocer patrones
        self.attention_span = 0.7  # 0-1, duración de atención
        self.recency_bias = 0.6  # 0-1, influencia de eventos recientes
        self.confirmation_bias = 0.5  # 0-1, buscar confirmación de creencias
        
        # Percepción actual del mercado
        self.perceived_trend = 0.0  # -1 a 1 (bajista a alcista)
        self.perceived_volatility = 0.5  # 0-1
        self.perceived_liquidity = 0.5  # 0-1
        self.perceived_sentiment = "neutral"  # bullish, neutral, bearish
        
        # Capacidad adaptativa
        self.adaptability = 0.6  # 0-1, capacidad de adaptarse a cambios
        
        # Memoria de percepción
        self.perception_history = []  # Historial de percepciones
        self.max_history = 10
        
        # Variables de estado
        self.last_update = datetime.now()
        self.focus_markets = []  # Mercados en los que se enfoca
        self.noticed_patterns = []  # Patrones que ha notado recientemente
        
        logger.info(f"Mirada de Gabriel inicializada con perspectiva {self.current_perspective.name}")
    
    async def initialize(self) -> bool:
        """
        Inicializar completamente el componente Mirada.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            # Establecer perspectiva inicial
            self._select_initial_perspective()
            
            # Inicializar percepciones iniciales
            self.perceived_trend = random.uniform(-0.2, 0.2)
            self.perceived_volatility = random.uniform(0.4, 0.6)
            self.perceived_liquidity = random.uniform(0.4, 0.6)
            
            # Inicializar memoria de percepción
            self.perception_history = [{
                "perspective": self.current_perspective,
                "confidence": self.perspective_confidence,
                "trend": self.perceived_trend,
                "volatility": self.perceived_volatility,
                "liquidity": self.perceived_liquidity,
                "timestamp": datetime.now()
            }]
            
            logger.info(f"Mirada inicializada exitosamente con perspectiva {self.current_perspective.name}")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar Mirada: {str(e)}")
            return False
    
    def _select_initial_perspective(self) -> None:
        """Seleccionar una perspectiva inicial basada en las propiedades de la Mirada."""
        # Distribuir perspectivas iniciales
        perspectives = [
            Perspective.BULLISH,
            Perspective.CAUTIOUSLY_BULLISH,
            Perspective.NEUTRAL,
            Perspective.CAUTIOUSLY_BEARISH,
            Perspective.BEARISH
        ]
        weights = [0.15, 0.25, 0.3, 0.2, 0.1]  # Ligero sesgo optimista
        
        self.current_perspective = random.choices(perspectives, weights=weights, k=1)[0]
        self.perspective_confidence = random.uniform(0.4, 0.7)
    
    async def update(self, market_data: Optional[Dict[str, Any]] = None, mood: Mood = None) -> None:
        """
        Actualizar la perspectiva basada en datos de mercado y estado emocional.
        
        Args:
            market_data: Datos de mercado actuales
            mood: Estado emocional actual del Alma
        """
        try:
            # Evolución natural si no hay datos
            if not market_data:
                self._natural_perspective_evolution()
                return
            
            # Aplicar filtro emocional a los datos
            filtered_data = self._apply_emotional_filter(market_data, mood)
            
            # Actualizar percepción del mercado
            self._update_market_perception(filtered_data)
            
            # Actualizar perspectiva basada en percepción actual
            self._update_perspective(filtered_data, mood)
            
            # Registrar en historial
            self._record_perception()
            
            # Actualizar timestamp
            self.last_update = datetime.now()
        except Exception as e:
            logger.error(f"Error al actualizar la perspectiva: {str(e)}")
    
    def _apply_emotional_filter(self, data: Dict[str, Any], mood: Mood) -> Dict[str, Any]:
        """
        Aplicar un filtro emocional a los datos de mercado.
        
        Args:
            data: Datos originales
            mood: Estado emocional actual
            
        Returns:
            Datos filtrados según el estado emocional
        """
        # Clonar datos para no modificar original
        filtered = data.copy()
        
        # Si está temeroso, sobreestima riesgos y volatilidad
        if mood == Mood.FEARFUL:
            # Exagerar volatilidad
            if "volatility" in filtered:
                filtered["volatility"] = min(filtered["volatility"] * 1.5, 1.0)
            
            # Subestimar tendencias positivas
            if "trend" in filtered and filtered["trend"] > 0:
                filtered["trend"] *= 0.7
            
            # Exagerar tendencias negativas
            if "trend" in filtered and filtered["trend"] < 0:
                filtered["trend"] *= 1.3
            
            # Percibir sentimiento más negativo
            if "sentiment" in filtered:
                if filtered["sentiment"] == "neutral":
                    filtered["sentiment"] = "slightly_bearish"
                elif filtered["sentiment"] == "slightly_bullish":
                    filtered["sentiment"] = "neutral"
        
        # Si está esperanzado, subestima riesgos y exagera oportunidades
        elif mood == Mood.HOPEFUL:
            # Subestimar volatilidad
            if "volatility" in filtered:
                filtered["volatility"] = max(filtered["volatility"] * 0.8, 0.0)
            
            # Exagerar tendencias positivas
            if "trend" in filtered and filtered["trend"] > 0:
                filtered["trend"] = min(filtered["trend"] * 1.3, 1.0)
            
            # Subestimar tendencias negativas
            if "trend" in filtered and filtered["trend"] < 0:
                filtered["trend"] *= 0.7
            
            # Percibir sentimiento más positivo
            if "sentiment" in filtered:
                if filtered["sentiment"] == "neutral":
                    filtered["sentiment"] = "slightly_bullish"
                elif filtered["sentiment"] == "slightly_bearish":
                    filtered["sentiment"] = "neutral"
        
        # Si está inquieto, exagera cambios y fluctuaciones
        elif mood == Mood.RESTLESS:
            # Exagerar cambios recientes
            if "price_change" in filtered:
                filtered["price_change"] *= 1.2
            
            # Percibir mayor actividad
            if "activity" in filtered:
                filtered["activity"] = min(filtered["activity"] * 1.3, 1.0)
        
        # Los estados neutros y serenos tienen filtros más objetivos
        elif mood == Mood.SERENE:
            # Percepción más equilibrada
            if "trend" in filtered:
                filtered["trend"] *= 0.9  # Reducir extremos
        
        return filtered
    
    def _update_market_perception(self, data: Dict[str, Any]) -> None:
        """
        Actualizar la percepción del mercado basada en datos filtrados.
        
        Args:
            data: Datos de mercado filtrados
        """
        # Actualizar trend percibido
        if "trend" in data:
            # Influencia del nuevo dato proporcional a la atención
            influence = self.attention_span * self.recency_bias
            self.perceived_trend = (self.perceived_trend * (1 - influence)) + (data["trend"] * influence)
        
        # Actualizar volatilidad percibida
        if "volatility" in data:
            influence = self.attention_span * self.recency_bias
            self.perceived_volatility = (self.perceived_volatility * (1 - influence)) + (data["volatility"] * influence)
        
        # Actualizar liquidez percibida
        if "liquidity" in data:
            influence = self.attention_span * 0.5  # Menor influencia
            self.perceived_liquidity = (self.perceived_liquidity * (1 - influence)) + (data["liquidity"] * influence)
        
        # Actualizar sentimiento percibido
        if "sentiment" in data:
            # Mapear sentimiento a valor numérico para cálculos
            sentiment_value = {
                "very_bearish": -1.0,
                "bearish": -0.7,
                "slightly_bearish": -0.3,
                "neutral": 0.0,
                "slightly_bullish": 0.3,
                "bullish": 0.7,
                "very_bullish": 1.0
            }.get(data["sentiment"], 0.0)
            
            # Convertir sentimiento actual a valor
            current_sentiment_value = {
                "very_bearish": -1.0,
                "bearish": -0.7,
                "slightly_bearish": -0.3,
                "neutral": 0.0,
                "slightly_bullish": 0.3,
                "bullish": 0.7,
                "very_bullish": 1.0
            }.get(self.perceived_sentiment, 0.0)
            
            # Calcular nuevo valor
            influence = self.attention_span * self.recency_bias
            new_sentiment_value = (current_sentiment_value * (1 - influence)) + (sentiment_value * influence)
            
            # Convertir valor a categoría
            if new_sentiment_value <= -0.8:
                self.perceived_sentiment = "very_bearish"
            elif new_sentiment_value <= -0.5:
                self.perceived_sentiment = "bearish"
            elif new_sentiment_value <= -0.2:
                self.perceived_sentiment = "slightly_bearish"
            elif new_sentiment_value <= 0.2:
                self.perceived_sentiment = "neutral"
            elif new_sentiment_value <= 0.5:
                self.perceived_sentiment = "slightly_bullish"
            elif new_sentiment_value <= 0.8:
                self.perceived_sentiment = "bullish"
            else:
                self.perceived_sentiment = "very_bullish"
    
    def _update_perspective(self, data: Dict[str, Any], mood: Mood) -> None:
        """
        Actualizar la perspectiva basada en la percepción y estado emocional.
        
        Args:
            data: Datos de mercado filtrados
            mood: Estado emocional actual
        """
        previous_perspective = self.current_perspective
        
        # Calcular un valor agregado para la tendencia (entre -1 y 1)
        trend_value = self.perceived_trend * 0.5
        
        # Añadir influencia del sentimiento percibido
        sentiment_value = {
            "very_bearish": -0.8,
            "bearish": -0.6,
            "slightly_bearish": -0.3,
            "neutral": 0.0,
            "slightly_bullish": 0.3,
            "bullish": 0.6,
            "very_bullish": 0.8
        }.get(self.perceived_sentiment, 0.0)
        
        trend_value += sentiment_value * 0.3
        
        # Añadir influencia de señales técnicas
        if "technical_signals" in data:
            # Convertir señales a valor entre -1 y 1
            tech_value = (data["technical_signals"] * 2) - 1
            trend_value += tech_value * 0.2
        
        # Aplicar sesgo de confirmación
        if (self.current_perspective in [Perspective.BULLISH, Perspective.CAUTIOUSLY_BULLISH] and trend_value > 0) or \
           (self.current_perspective in [Perspective.BEARISH, Perspective.CAUTIOUSLY_BEARISH] and trend_value < 0):
            # Reforzar la perspectiva actual
            trend_value *= (1 + (self.confirmation_bias * 0.3))
        
        # Ajustar por estado de ánimo
        if mood == Mood.FEARFUL:
            trend_value -= 0.3
        elif mood == Mood.HOPEFUL:
            trend_value += 0.3
        elif mood == Mood.CAUTIOUS:
            trend_value *= 0.8  # Reduce la magnitud
        
        # Determinar nueva perspectiva basada en el valor agregado
        if trend_value > 0.6:
            new_perspective = Perspective.BULLISH
        elif trend_value > 0.2:
            new_perspective = Perspective.CAUTIOUSLY_BULLISH
        elif trend_value > -0.2:
            new_perspective = Perspective.NEUTRAL
        elif trend_value > -0.6:
            new_perspective = Perspective.CAUTIOUSLY_BEARISH
        else:
            new_perspective = Perspective.BEARISH
        
        # Si los datos son confusos o contradictorios, podría adoptar perspectiva confusa
        if "conflicting_signals" in data and data["conflicting_signals"] > 0.7:
            self.perspective_confidence *= 0.7
            if random.random() < 0.3:
                new_perspective = Perspective.CONFUSED
        
        # Calcular confianza basada en consistencia y claridad de datos
        confidence_adjustment = 0.0
        
        # Mayor confianza si hay muchas señales en la misma dirección
        if "signal_alignment" in data:
            confidence_adjustment += (data["signal_alignment"] - 0.5) * 0.4
        
        # Mayor confianza en menor volatilidad
        confidence_adjustment -= (self.perceived_volatility - 0.5) * 0.3
        
        # Mayor confianza si la perspectiva se mantiene
        if new_perspective == self.current_perspective:
            confidence_adjustment += 0.1
        
        # Aplicar ajuste de confianza
        self.perspective_confidence = max(0.1, min(0.9, self.perspective_confidence + confidence_adjustment))
        
        # Finalmente, actualizar perspectiva
        if new_perspective != self.current_perspective:
            self.current_perspective = new_perspective
            logger.debug(f"Perspectiva cambiada a {self.current_perspective.name}, confianza: {self.perspective_confidence:.2f}")
    
    def _record_perception(self) -> None:
        """Registrar la percepción actual en el historial."""
        self.perception_history.append({
            "perspective": self.current_perspective,
            "confidence": self.perspective_confidence,
            "trend": self.perceived_trend,
            "volatility": self.perceived_volatility,
            "liquidity": self.perceived_liquidity,
            "sentiment": self.perceived_sentiment,
            "timestamp": datetime.now()
        })
        
        # Limitar tamaño del historial
        if len(self.perception_history) > self.max_history:
            self.perception_history = self.perception_history[-self.max_history:]
    
    def _natural_perspective_evolution(self) -> None:
        """
        Evolución natural de la perspectiva con el paso del tiempo.
        
        Sin datos nuevos, la perspectiva tiende gradualmente hacia la neutralidad,
        pero manteniendo un componente aleatorio para simular cambios de opinión.
        """
        # Determinar cuánto tiempo ha pasado
        time_delta = datetime.now() - self.last_update
        hours_passed = time_delta.total_seconds() / 3600
        
        # Calcular factor de evolución (cuánto puede cambiar)
        evolution_factor = min(hours_passed / 12, 1.0) * (1 - self.confirmation_bias)
        
        # Tendencia natural hacia neutralidad
        if hours_passed > 4:
            if self.current_perspective == Perspective.BULLISH:
                if random.random() < evolution_factor * 0.5:
                    self.current_perspective = Perspective.CAUTIOUSLY_BULLISH
                    self.perspective_confidence *= 0.9
            elif self.current_perspective == Perspective.CAUTIOUSLY_BULLISH:
                if random.random() < evolution_factor * 0.3:
                    self.current_perspective = Perspective.NEUTRAL
                    self.perspective_confidence *= 0.8
            elif self.current_perspective == Perspective.BEARISH:
                if random.random() < evolution_factor * 0.5:
                    self.current_perspective = Perspective.CAUTIOUSLY_BEARISH
                    self.perspective_confidence *= 0.9
            elif self.current_perspective == Perspective.CAUTIOUSLY_BEARISH:
                if random.random() < evolution_factor * 0.3:
                    self.current_perspective = Perspective.NEUTRAL
                    self.perspective_confidence *= 0.8
            elif self.current_perspective == Perspective.CONFUSED:
                if random.random() < evolution_factor * 0.7:
                    self.current_perspective = Perspective.NEUTRAL
                    self.perspective_confidence = 0.4
        
        # Disminución gradual de confianza sin nuevos datos
        confidence_decay = evolution_factor * 0.2
        self.perspective_confidence = max(0.2, self.perspective_confidence - confidence_decay)
        
        # Actualizar percepción gradualmente
        if hours_passed > 2:
            # Tendencia tiende a neutralizarse
            self.perceived_trend *= (1 - (evolution_factor * 0.3))
            
            # Sentimiento tiende a neutralizarse
            if self.perceived_sentiment != "neutral" and random.random() < evolution_factor * 0.4:
                current_sentiment_value = {
                    "very_bearish": -1.0,
                    "bearish": -0.7,
                    "slightly_bearish": -0.3,
                    "neutral": 0.0,
                    "slightly_bullish": 0.3,
                    "bullish": 0.7,
                    "very_bullish": 1.0
                }.get(self.perceived_sentiment, 0.0)
                
                # Mover hacia neutral
                new_value = current_sentiment_value * (1 - (evolution_factor * 0.5))
                
                # Convertir valor a categoría
                if new_value <= -0.8:
                    self.perceived_sentiment = "very_bearish"
                elif new_value <= -0.5:
                    self.perceived_sentiment = "bearish"
                elif new_value <= -0.2:
                    self.perceived_sentiment = "slightly_bearish"
                elif new_value <= 0.2:
                    self.perceived_sentiment = "neutral"
                elif new_value <= 0.5:
                    self.perceived_sentiment = "slightly_bullish"
                elif new_value <= 0.8:
                    self.perceived_sentiment = "bullish"
                else:
                    self.perceived_sentiment = "very_bullish"
    
    def apply_archetype(self, archetype_config: Dict[str, Any]) -> None:
        """
        Aplicar configuración de arquetipo a la Mirada.
        
        Args:
            archetype_config: Configuración del arquetipo
        """
        # Aplicar propiedades específicas
        self.pattern_recognition = archetype_config.get("pattern_recognition", self.pattern_recognition)
        self.attention_span = archetype_config.get("attention_span", self.attention_span)
        self.recency_bias = archetype_config.get("recency_bias", self.recency_bias)
        self.confirmation_bias = archetype_config.get("confirmation_bias", self.confirmation_bias)
        self.adaptability = archetype_config.get("adaptability", self.adaptability)
        
        # Actualizar perspectiva inicial si está especificada
        if "initial_perspective" in archetype_config:
            perspective_name = archetype_config["initial_perspective"]
            try:
                self.current_perspective = Perspective[perspective_name]
                self.perspective_confidence = archetype_config.get("initial_confidence", 0.5)
                self._record_perception()
            except KeyError:
                logger.warning(f"Perspectiva {perspective_name} no encontrada, manteniendo {self.current_perspective.name}")
    
    def get_perspective(self) -> Perspective:
        """
        Obtener la perspectiva actual.
        
        Returns:
            Perspectiva actual
        """
        return self.current_perspective
    
    def get_perspective_confidence(self) -> float:
        """
        Obtener la confianza en la perspectiva actual.
        
        Returns:
            Nivel de confianza (0-1)
        """
        return self.perspective_confidence
    
    def get_market_perception(self) -> Dict[str, Any]:
        """
        Obtener la percepción actual del mercado.
        
        Returns:
            Diccionario con la percepción actual
        """
        return {
            "trend": self.perceived_trend,
            "volatility": self.perceived_volatility,
            "liquidity": self.perceived_liquidity,
            "sentiment": self.perceived_sentiment
        }
    
    def adapt_to_mood(self, mood: Mood) -> None:
        """
        Adaptar la percepción al estado emocional para mantener coherencia.
        
        Args:
            mood: Estado emocional actual
        """
        # Ajustes sutiles para mantener coherencia psicológica
        if mood == Mood.FEARFUL:
            # En estado temeroso, aumenta sesgos y disminuye atención
            self.confirmation_bias = min(self.confirmation_bias + 0.1, 0.9)
            self.attention_span = max(self.attention_span - 0.1, 0.3)
            self.recency_bias = min(self.recency_bias + 0.2, 0.9)  # Más influenciable por eventos recientes
        
        elif mood == Mood.SERENE:
            # En estado sereno, disminuye sesgos y aumenta atención
            self.confirmation_bias = max(self.confirmation_bias - 0.1, 0.2)
            self.attention_span = min(self.attention_span + 0.1, 0.9)
            self.recency_bias = max(self.recency_bias - 0.1, 0.3)  # Más equilibrado
    
    async def randomize(self, mood: Optional[Mood] = None) -> None:
        """
        Aleatorizar la perspectiva para mayor variabilidad.
        
        Args:
            mood: Estado emocional actual para mantener coherencia
        """
        # Elegir una perspectiva aleatoria
        perspectives = list(Perspective)
        
        # Si hay un estado emocional, ajustar las probabilidades para coherencia
        if mood == Mood.FEARFUL:
            weights = [0.05, 0.15, 0.2, 0.25, 0.25, 0.1]  # Más propensión a negativo
        elif mood == Mood.HOPEFUL:
            weights = [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]  # Más propensión a positivo
        elif mood == Mood.CAUTIOUS:
            weights = [0.05, 0.2, 0.2, 0.3, 0.2, 0.05]  # Más neutro-cautious
        else:
            weights = [0.15, 0.2, 0.25, 0.2, 0.15, 0.05]  # Bastante equilibrado
        
        self.current_perspective = random.choices(perspectives, weights=weights, k=1)[0]
        self.perspective_confidence = random.uniform(0.4, 0.8)
        
        # Aleatorizar también percepción del mercado
        self.perceived_trend = random.uniform(-0.7, 0.7)
        self.perceived_volatility = random.uniform(0.3, 0.8)
        self.perceived_liquidity = random.uniform(0.3, 0.8)
        
        sentiments = ["bearish", "slightly_bearish", "neutral", "slightly_bullish", "bullish"]
        self.perceived_sentiment = random.choice(sentiments)
        
        # Registrar en historial
        self._record_perception()
        
        logger.info(f"Mirada aleatorizada a {self.current_perspective.name}, confianza: {self.perspective_confidence:.2f}")
    
    def reset(self) -> None:
        """Reiniciar el estado a valores predeterminados."""
        self.current_perspective = Perspective.NEUTRAL
        self.perspective_confidence = 0.5
        self.perceived_trend = 0.0
        self.perceived_volatility = 0.5
        self.perceived_liquidity = 0.5
        self.perceived_sentiment = "neutral"
        self._record_perception()