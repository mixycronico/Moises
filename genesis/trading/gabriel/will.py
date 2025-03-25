"""
La Voluntad de Gabriel - Toma de decisiones y acciones

Este módulo implementa el componente "Voluntad" (Will) del motor de comportamiento humano Gabriel,
responsable de la toma de decisiones y acciones basadas en el estado emocional y la percepción.
Es el componente final que determina las acciones concretas en el trading.

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
from genesis.trading.gabriel.gaze import Perspective

logger = logging.getLogger(__name__)

class Decision(Enum):
    """Estilos de decisión que puede adoptar la Voluntad de Gabriel."""
    BALANCED = auto()     # Equilibrado - evalúa pros y contras
    CAUTIOUS = auto()     # Cauteloso - minimiza riesgos
    AGGRESSIVE = auto()   # Agresivo - busca oportunidades
    ADAPTIVE = auto()     # Adaptativo - cambia según el contexto
    IMPULSIVE = auto()    # Impulsivo - decisiones rápidas
    ANALYTICAL = auto()   # Analítico - evalúa datos en profundidad

class Will:
    """
    La Voluntad (Will) de Gabriel, responsable de la toma de decisiones.
    
    Este componente simula el proceso humano de decisión:
    - Combina estado emocional y perspectiva para decidir acciones
    - Aplica sesgos cognitivos como aversión a la pérdida
    - Gestiona niveles variables de tolerancia al riesgo
    - Implementa diferentes estilos de decisión según contexto
    - Aprende de experiencias pasadas para ajustar decisiones futuras
    """
    
    def __init__(self):
        """Inicializar el componente Voluntad con sus propiedades esenciales."""
        # Estilo de decisión actual y preferencias
        self.decision_style = Decision.BALANCED
        self.risk_preference = 0.5  # 0-1, preferencia de riesgo
        self.patience = 0.6  # 0-1, capacidad de esperar
        self.conviction = 0.5  # 0-1, firmeza en decisiones
        
        # Sesgos cognitivos
        self.loss_aversion = 0.7  # 0-1, aversión a la pérdida
        self.recency_bias = 0.6  # 0-1, influencia de eventos recientes
        self.confirmation_bias = 0.5  # 0-1, buscar confirmar creencias
        self.overconfidence = 0.4  # 0-1, exceso de confianza
        
        # Variables de decisión
        self.min_signal_threshold = 0.3  # Umbral mínimo para señales
        self.exit_profit_threshold = 0.15  # % beneficio para considerar salida
        self.stop_loss_threshold = 0.10  # % pérdida para salida automática
        
        # Memoria de decisiones
        self.decision_history = []  # Historial de decisiones tomadas
        self.max_history = 15
        
        # Variables de estado
        self.last_update = datetime.now()
        self.consecutive_same_decisions = 0  # Decisiones consecutivas similares
        self.recent_success_rate = 0.5  # Tasa de éxito reciente (0-1)
        
        logger.info(f"Voluntad de Gabriel inicializada con estilo {self.decision_style.name}")
    
    async def initialize(self) -> bool:
        """
        Inicializar completamente el componente Voluntad.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            # Establecer estilo de decisión inicial
            self._select_initial_decision_style()
            
            # Inicializar umbrales basados en estilo
            self._adjust_thresholds_to_style()
            
            # Inicializar memoria de decisiones
            self.decision_history = [{
                "style": self.decision_style,
                "risk_level": self.risk_preference,
                "timestamp": datetime.now()
            }]
            
            logger.info(f"Voluntad inicializada exitosamente con estilo {self.decision_style.name}")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar Voluntad: {str(e)}")
            return False
    
    def _select_initial_decision_style(self) -> None:
        """Seleccionar un estilo de decisión inicial."""
        # Distribuir estilos iniciales
        styles = [
            Decision.BALANCED,
            Decision.CAUTIOUS,
            Decision.AGGRESSIVE,
            Decision.ADAPTIVE,
            Decision.ANALYTICAL,
            Decision.IMPULSIVE
        ]
        weights = [0.3, 0.2, 0.15, 0.15, 0.15, 0.05]  # Favorece estilos balanceados
        
        self.decision_style = random.choices(styles, weights=weights, k=1)[0]
        
        # Ajustar propiedades según estilo
        if self.decision_style == Decision.CAUTIOUS:
            self.risk_preference = random.uniform(0.2, 0.4)
            self.patience = random.uniform(0.6, 0.8)
            self.loss_aversion = random.uniform(0.7, 0.9)
        elif self.decision_style == Decision.AGGRESSIVE:
            self.risk_preference = random.uniform(0.6, 0.8)
            self.patience = random.uniform(0.3, 0.5)
            self.loss_aversion = random.uniform(0.3, 0.5)
        elif self.decision_style == Decision.IMPULSIVE:
            self.patience = random.uniform(0.1, 0.3)
            self.conviction = random.uniform(0.7, 0.9)
            self.recency_bias = random.uniform(0.7, 0.9)
        elif self.decision_style == Decision.ANALYTICAL:
            self.patience = random.uniform(0.7, 0.9)
            self.confirmation_bias = random.uniform(0.3, 0.5)
            self.overconfidence = random.uniform(0.2, 0.4)
    
    def _adjust_thresholds_to_style(self) -> None:
        """Ajustar umbrales de decisión según el estilo actual."""
        if self.decision_style == Decision.CAUTIOUS:
            self.min_signal_threshold = 0.5  # Requiere señales más fuertes
            self.exit_profit_threshold = 0.1  # Sale con menor beneficio
            self.stop_loss_threshold = 0.05  # Stop loss más ajustado
        elif self.decision_style == Decision.AGGRESSIVE:
            self.min_signal_threshold = 0.2  # Acepta señales más débiles
            self.exit_profit_threshold = 0.25  # Busca mayor beneficio
            self.stop_loss_threshold = 0.15  # Tolera más pérdida
        elif self.decision_style == Decision.BALANCED:
            self.min_signal_threshold = 0.35
            self.exit_profit_threshold = 0.15
            self.stop_loss_threshold = 0.10
        elif self.decision_style == Decision.ADAPTIVE:
            # Se ajusta dinámicamente según evolución, valores iniciales medios
            self.min_signal_threshold = 0.3
            self.exit_profit_threshold = 0.15
            self.stop_loss_threshold = 0.10
        elif self.decision_style == Decision.IMPULSIVE:
            self.min_signal_threshold = 0.25  # Decisiones rápidas
            self.exit_profit_threshold = 0.12  # Sale rápido
            self.stop_loss_threshold = 0.12  # Sale rápido también en pérdidas
        elif self.decision_style == Decision.ANALYTICAL:
            self.min_signal_threshold = 0.4  # Requiere más datos
            self.exit_profit_threshold = 0.18  # Calcula mejor punto de salida
            self.stop_loss_threshold = 0.08  # Calcula mejor punto de stop
    
    async def update(self, perspective: Perspective, mood: Mood) -> None:
        """
        Actualizar el estado interno de la Voluntad.
        
        Args:
            perspective: Perspectiva actual
            mood: Estado emocional actual
        """
        try:
            # Ajustar estilo de decisión basado en perspectiva y humor
            self._adjust_decision_style(perspective, mood)
            
            # Actualizar umbrales según nuevo estilo
            self._adjust_thresholds_to_style()
            
            # Ajustar preferencias según humor
            self._adjust_preferences_to_mood(mood)
            
            # Evolución natural de decisiones
            self._natural_decision_evolution()
            
            # Registrar en historial si el estilo cambió
            previous_entry = self.decision_history[-1] if self.decision_history else None
            if not previous_entry or previous_entry["style"] != self.decision_style:
                self.decision_history.append({
                    "style": self.decision_style,
                    "risk_level": self.risk_preference,
                    "timestamp": datetime.now()
                })
                
                # Limitar tamaño del historial
                if len(self.decision_history) > self.max_history:
                    self.decision_history = self.decision_history[-self.max_history:]
            
            # Actualizar timestamp
            self.last_update = datetime.now()
        except Exception as e:
            logger.error(f"Error al actualizar Voluntad: {str(e)}")
    
    def _adjust_decision_style(self, perspective: Perspective, mood: Mood) -> None:
        """
        Ajustar el estilo de decisión según la perspectiva y el humor.
        
        Args:
            perspective: Perspectiva actual
            mood: Estado emocional actual
        """
        previous_style = self.decision_style
        
        # Matriz de estilos probables según perspectiva y humor
        # El estilo se ajusta para ser coherente psicológicamente
        style_matrix = {
            # Cuando está en SERENE (calma)
            Mood.SERENE: {
                Perspective.BULLISH: [Decision.BALANCED, Decision.AGGRESSIVE],
                Perspective.CAUTIOUSLY_BULLISH: [Decision.BALANCED, Decision.ADAPTIVE],
                Perspective.NEUTRAL: [Decision.BALANCED, Decision.ANALYTICAL],
                Perspective.CAUTIOUSLY_BEARISH: [Decision.CAUTIOUS, Decision.BALANCED],
                Perspective.BEARISH: [Decision.CAUTIOUS, Decision.ANALYTICAL],
                Perspective.CONFUSED: [Decision.BALANCED, Decision.ANALYTICAL]
            },
            # Cuando está en HOPEFUL (esperanzado)
            Mood.HOPEFUL: {
                Perspective.BULLISH: [Decision.AGGRESSIVE, Decision.BALANCED],
                Perspective.CAUTIOUSLY_BULLISH: [Decision.BALANCED, Decision.AGGRESSIVE],
                Perspective.NEUTRAL: [Decision.BALANCED, Decision.ADAPTIVE],
                Perspective.CAUTIOUSLY_BEARISH: [Decision.BALANCED, Decision.CAUTIOUS],
                Perspective.BEARISH: [Decision.CAUTIOUS, Decision.BALANCED],
                Perspective.CONFUSED: [Decision.ADAPTIVE, Decision.BALANCED]
            },
            # Cuando está en NEUTRAL
            Mood.NEUTRAL: {
                Perspective.BULLISH: [Decision.BALANCED, Decision.ADAPTIVE],
                Perspective.CAUTIOUSLY_BULLISH: [Decision.BALANCED, Decision.ADAPTIVE],
                Perspective.NEUTRAL: [Decision.BALANCED, Decision.ANALYTICAL],
                Perspective.CAUTIOUSLY_BEARISH: [Decision.BALANCED, Decision.CAUTIOUS],
                Perspective.BEARISH: [Decision.CAUTIOUS, Decision.BALANCED],
                Perspective.CONFUSED: [Decision.ANALYTICAL, Decision.CAUTIOUS]
            },
            # Cuando está en CAUTIOUS (cauteloso)
            Mood.CAUTIOUS: {
                Perspective.BULLISH: [Decision.BALANCED, Decision.CAUTIOUS],
                Perspective.CAUTIOUSLY_BULLISH: [Decision.CAUTIOUS, Decision.BALANCED],
                Perspective.NEUTRAL: [Decision.CAUTIOUS, Decision.ANALYTICAL],
                Perspective.CAUTIOUSLY_BEARISH: [Decision.CAUTIOUS, Decision.ANALYTICAL],
                Perspective.BEARISH: [Decision.CAUTIOUS, Decision.ANALYTICAL],
                Perspective.CONFUSED: [Decision.CAUTIOUS, Decision.ANALYTICAL]
            },
            # Cuando está en RESTLESS (inquieto)
            Mood.RESTLESS: {
                Perspective.BULLISH: [Decision.IMPULSIVE, Decision.AGGRESSIVE],
                Perspective.CAUTIOUSLY_BULLISH: [Decision.IMPULSIVE, Decision.ADAPTIVE],
                Perspective.NEUTRAL: [Decision.IMPULSIVE, Decision.ADAPTIVE],
                Perspective.CAUTIOUSLY_BEARISH: [Decision.IMPULSIVE, Decision.CAUTIOUS],
                Perspective.BEARISH: [Decision.IMPULSIVE, Decision.CAUTIOUS],
                Perspective.CONFUSED: [Decision.IMPULSIVE, Decision.ADAPTIVE]
            },
            # Cuando está en FEARFUL (temeroso)
            Mood.FEARFUL: {
                Perspective.BULLISH: [Decision.CAUTIOUS, Decision.ADAPTIVE],
                Perspective.CAUTIOUSLY_BULLISH: [Decision.CAUTIOUS, Decision.BALANCED],
                Perspective.NEUTRAL: [Decision.CAUTIOUS, Decision.ANALYTICAL],
                Perspective.CAUTIOUSLY_BEARISH: [Decision.CAUTIOUS, Decision.ANALYTICAL],
                Perspective.BEARISH: [Decision.CAUTIOUS, Decision.ANALYTICAL],
                Perspective.CONFUSED: [Decision.CAUTIOUS, Decision.ANALYTICAL]
            }
        }
        
        # Obtener opciones según la matriz
        style_options = style_matrix.get(mood, {}).get(perspective, [Decision.BALANCED, Decision.CAUTIOUS])
        
        # Determinar si cambia el estilo
        # Mayor probabilidad de cambio si los estados emocionales o perspectiva son intensos
        change_threshold = 0.3  # Probabilidad base de cambio
        
        # Si el estado actual ya es coherente con la matriz, menor probabilidad de cambio
        if self.decision_style in style_options:
            change_threshold = 0.1
        
        if random.random() < change_threshold:
            # Seleccionar nuevo estilo de las opciones
            self.decision_style = random.choice(style_options)
            
            # Loggear cambio
            if self.decision_style != previous_style:
                logger.debug(f"Estilo de decisión cambiado de {previous_style.name} a {self.decision_style.name}")
    
    def _adjust_preferences_to_mood(self, mood: Mood) -> None:
        """
        Ajustar preferencias según el estado emocional.
        
        Args:
            mood: Estado emocional actual
        """
        # Ajustes específicos por estado emocional
        if mood == Mood.FEARFUL:
            # Cuando tiene miedo, aversión al riesgo y a la pérdida
            self.risk_preference = max(self.risk_preference * 0.8, 0.1)
            self.loss_aversion = min(self.loss_aversion * 1.2, 1.0)
            self.patience = max(self.patience * 0.8, 0.2)
        
        elif mood == Mood.SERENE:
            # Cuando está sereno, más equilibrado en todo
            self.risk_preference = self.risk_preference * 0.8 + 0.5 * 0.2  # Tender al centro
            self.loss_aversion = self.loss_aversion * 0.8 + 0.5 * 0.2
            self.confirmation_bias = self.confirmation_bias * 0.8 + 0.3 * 0.2  # Menor sesgo
        
        elif mood == Mood.HOPEFUL:
            # Cuando está esperanzado, más propenso al riesgo
            self.risk_preference = min(self.risk_preference * 1.1, 0.8)
            self.loss_aversion = max(self.loss_aversion * 0.9, 0.4)
            self.patience = min(self.patience * 1.1, 0.9)  # Más paciente
        
        elif mood == Mood.RESTLESS:
            # Cuando está inquieto, impulsividad y menor paciencia
            self.patience = max(self.patience * 0.7, 0.2)
            self.recency_bias = min(self.recency_bias * 1.2, 0.9)  # Más influenciado por lo reciente
    
    def _natural_decision_evolution(self) -> None:
        """Evolución natural del estilo de decisión y preferencias con el tiempo."""
        # Determinar cuánto tiempo ha pasado
        time_delta = datetime.now() - self.last_update
        hours_passed = time_delta.total_seconds() / 3600
        
        if hours_passed < 1:
            return
        
        # Evolución gradual según resultados recientes
        if self.consecutive_same_decisions > 5:
            # Si ha tomado muchas decisiones similares, tendencia a experimentar cambio
            if random.random() < 0.3:
                all_styles = list(Decision)
                all_styles.remove(self.decision_style)
                potential_style = random.choice(all_styles)
                
                # Mayor probabilidad de cambiar a estilos cercanos
                if (self.decision_style == Decision.AGGRESSIVE and 
                    potential_style in [Decision.BALANCED, Decision.IMPULSIVE]):
                    self.decision_style = potential_style
                elif (self.decision_style == Decision.CAUTIOUS and 
                      potential_style in [Decision.BALANCED, Decision.ANALYTICAL]):
                    self.decision_style = potential_style
                elif random.random() < 0.3:
                    self.decision_style = potential_style
                
                self.consecutive_same_decisions = 0
        
        # Ajuste según tasa de éxito reciente
        if hours_passed > 12:
            if self.recent_success_rate > 0.7:
                # Buenos resultados, reforzar estilo actual
                pass
            elif self.recent_success_rate < 0.3:
                # Malos resultados, considerar cambio
                if random.random() < 0.4:
                    if self.decision_style == Decision.AGGRESSIVE:
                        self.decision_style = Decision.BALANCED
                    elif self.decision_style == Decision.IMPULSIVE:
                        self.decision_style = Decision.ADAPTIVE
                    elif self.decision_style == Decision.BALANCED:
                        self.decision_style = Decision.CAUTIOUS
                    
                    # Ajustar umbrales
                    self._adjust_thresholds_to_style()
        
        # Normalización natural de preferencias
        if hours_passed > 24:
            # Normalizar lentamente hacia valores medios
            for attr, target in [("risk_preference", 0.5), 
                                ("patience", 0.6), 
                                ("loss_aversion", 0.7),
                                ("confirmation_bias", 0.5)]:
                current = getattr(self, attr)
                adjustment = (target - current) * min(hours_passed / 72, 0.3)
                setattr(self, attr, current + adjustment)
    
    def apply_archetype(self, archetype_config: Dict[str, Any]) -> None:
        """
        Aplicar configuración de arquetipo a la Voluntad.
        
        Args:
            archetype_config: Configuración del arquetipo
        """
        # Aplicar propiedades específicas
        self.risk_preference = archetype_config.get("risk_preference", self.risk_preference)
        self.patience = archetype_config.get("patience", self.patience)
        self.conviction = archetype_config.get("conviction", self.conviction)
        self.loss_aversion = archetype_config.get("loss_aversion", self.loss_aversion)
        self.confirmation_bias = archetype_config.get("confirmation_bias", self.confirmation_bias)
        self.overconfidence = archetype_config.get("overconfidence", self.overconfidence)
        
        # Actualizar estilo inicial si está especificado
        if "decision_style" in archetype_config:
            style_name = archetype_config["decision_style"]
            try:
                self.decision_style = Decision[style_name]
                self._adjust_thresholds_to_style()
                self.decision_history.append({
                    "style": self.decision_style,
                    "risk_level": self.risk_preference,
                    "timestamp": datetime.now()
                })
            except KeyError:
                logger.warning(f"Estilo {style_name} no encontrado, manteniendo {self.decision_style.name}")
        
        # Actualizar umbrales específicos
        self.min_signal_threshold = archetype_config.get("min_signal_threshold", self.min_signal_threshold)
        self.exit_profit_threshold = archetype_config.get("exit_profit_threshold", self.exit_profit_threshold)
        self.stop_loss_threshold = archetype_config.get("stop_loss_threshold", self.stop_loss_threshold)
    
    async def decide_trade(self, signal_strength: float, 
                         mood: Mood, 
                         perspective: Perspective,
                         market_context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Decidir si entrar en una operación de trading.
        
        Args:
            signal_strength: Fuerza de la señal (0-1)
            mood: Estado emocional actual
            perspective: Perspectiva actual
            market_context: Contexto del mercado
            
        Returns:
            Tupla (decisión, razón)
        """
        # Actualizar estado interno primero
        await self.update(perspective, mood)
        
        # Aplicar sesgos a la señal
        adjusted_signal = self._apply_biases_to_signal(signal_strength, mood, perspective, market_context)
        
        # Aplicar umbral según estilo
        threshold = self._get_effective_threshold(market_context)
        
        # Evaluar volatilidad del mercado
        volatility = market_context.get("volatility", 0.5)
        high_volatility = volatility > 0.7
        
        # Condiciones especiales que pueden anular la decisión
        if mood == Mood.FEARFUL and high_volatility:
            return False, "Mercado demasiado volátil en estado temeroso"
        
        # Verificar si hay señales mixtas fuertes
        mixed_signals = self._check_for_mixed_signals(perspective, market_context)
        if mixed_signals and self.decision_style in [Decision.CAUTIOUS, Decision.ANALYTICAL]:
            return False, "Señales contradictorias detectadas"
        
        # Decisión final
        decision = adjusted_signal >= threshold
        
        # Registrar decisión para futuras referencias
        self._record_trade_decision(decision, signal_strength, adjusted_signal, mood, perspective)
        
        # Generar explicación humana
        reason = self._generate_decision_reason(decision, adjusted_signal, threshold, mood, perspective)
        
        return decision, reason
    
    def _apply_biases_to_signal(self, signal: float, mood: Mood, 
                              perspective: Perspective, 
                              context: Dict[str, Any]) -> float:
        """
        Aplicar sesgos cognitivos a la señal.
        
        Args:
            signal: Señal original
            mood: Estado emocional
            perspective: Perspectiva
            context: Contexto del mercado
            
        Returns:
            Señal ajustada
        """
        adjusted_signal = signal
        
        # Sesgo de confirmación - Refuerza señales que confirman la perspectiva
        if (perspective in [Perspective.BULLISH, Perspective.CAUTIOUSLY_BULLISH] and signal > 0.5) or \
           (perspective in [Perspective.BEARISH, Perspective.CAUTIOUSLY_BEARISH] and signal < 0.5):
            # La señal confirma la perspectiva, amplificarla
            confirmation_factor = 1.0 + (self.confirmation_bias * 0.3)
            adjusted_signal = min(adjusted_signal * confirmation_factor, 1.0)
        
        # Aversión a la pérdida - Reduce señales en contextos de riesgo percibido
        recent_downtrend = context.get("recent_downtrend", False)
        if recent_downtrend and self.loss_aversion > 0.6:
            loss_aversion_factor = 1.0 - (self.loss_aversion * 0.2)
            adjusted_signal *= loss_aversion_factor
        
        # Efecto del humor
        if mood == Mood.HOPEFUL:
            # Más optimista, ve mejores señales
            mood_factor = 1.0 + (0.1 * self.recency_bias)
            adjusted_signal = min(adjusted_signal * mood_factor, 1.0)
        elif mood == Mood.FEARFUL:
            # Más pesimista, ve peores señales
            mood_factor = 1.0 - (0.2 * self.recency_bias)
            adjusted_signal *= mood_factor
        
        # Efecto de sobreconfianza
        if self.consecutive_same_decisions > 3 and self.overconfidence > 0.5:
            # Ha tomado decisiones similares consecutivas, sobreconfianza
            if adjusted_signal > 0.5:
                confidence_boost = self.overconfidence * 0.15
                adjusted_signal = min(adjusted_signal + confidence_boost, 1.0)
        
        # Asegurar que está en el rango 0-1
        return max(0.0, min(adjusted_signal, 1.0))
    
    def _get_effective_threshold(self, context: Dict[str, Any]) -> float:
        """
        Obtener umbral efectivo según el contexto.
        
        Args:
            context: Contexto del mercado
            
        Returns:
            Umbral efectivo
        """
        # Umbral base según estilo
        threshold = self.min_signal_threshold
        
        # Ajustar según volatilidad
        volatility = context.get("volatility", 0.5)
        if volatility > 0.7:
            # Alta volatilidad, aumentar umbral
            threshold += 0.1
        
        # Ajustar según liquidez
        liquidity = context.get("liquidity", 0.5)
        if liquidity < 0.3:
            # Baja liquidez, aumentar umbral
            threshold += 0.1
        
        # Ajustar según hora del día
        hour = datetime.now().hour
        is_market_active = 8 <= hour <= 20  # Asumiendo horario activo
        if not is_market_active and self.decision_style == Decision.CAUTIOUS:
            threshold += 0.05
        
        return threshold
    
    def _check_for_mixed_signals(self, perspective: Perspective, 
                               context: Dict[str, Any]) -> bool:
        """
        Verificar si hay señales contradictorias.
        
        Args:
            perspective: Perspectiva actual
            context: Contexto del mercado
            
        Returns:
            True si hay señales mixtas fuertes
        """
        # Verifica si hay contradicciones entre diferentes indicadores
        technical_bullish = context.get("technical_bullish", 0.5) > 0.6
        technical_bearish = context.get("technical_bearish", 0.5) > 0.6
        
        fundamental_bullish = context.get("fundamental_bullish", 0.5) > 0.6
        fundamental_bearish = context.get("fundamental_bearish", 0.5) > 0.6
        
        # Contradicción entre técnico y fundamental
        if (technical_bullish and fundamental_bearish) or (technical_bearish and fundamental_bullish):
            return True
        
        # Contradicción con la perspectiva
        if (perspective in [Perspective.BULLISH, Perspective.CAUTIOUSLY_BULLISH] and 
            (technical_bearish or fundamental_bearish)):
            return True
        
        if (perspective in [Perspective.BEARISH, Perspective.CAUTIOUSLY_BEARISH] and 
            (technical_bullish or fundamental_bullish)):
            return True
        
        return False
    
    def _record_trade_decision(self, decision: bool, original_signal: float,
                             adjusted_signal: float, mood: Mood, 
                             perspective: Perspective) -> None:
        """
        Registrar decisión de trading para análisis futuro.
        
        Args:
            decision: Decisión final
            original_signal: Señal original
            adjusted_signal: Señal ajustada
            mood: Estado emocional
            perspective: Perspectiva
        """
        # Actualizar contador de decisiones consecutivas
        last_decision = next((d for d in reversed(self.decision_history) 
                            if "trade_decision" in d), None)
        
        if last_decision and last_decision.get("trade_decision") == decision:
            self.consecutive_same_decisions += 1
        else:
            self.consecutive_same_decisions = 0
    
    def _generate_decision_reason(self, decision: bool, signal: float, 
                                threshold: float, mood: Mood,
                                perspective: Perspective) -> str:
        """
        Generar una explicación humana para la decisión.
        
        Args:
            decision: Decisión tomada
            signal: Fuerza de la señal ajustada
            threshold: Umbral utilizado
            mood: Estado emocional
            perspective: Perspectiva
            
        Returns:
            Explicación en lenguaje natural
        """
        if decision:
            if signal > threshold + 0.2:
                reason = "Señal muy fuerte"
            elif signal > threshold + 0.1:
                reason = "Buena oportunidad"
            else:
                reason = "Señal adecuada"
                
            # Añadir contexto de humor y perspectiva
            if mood == Mood.HOPEFUL:
                reason += ", perspectiva optimista"
            elif mood == Mood.FEARFUL:
                reason += ", pero con cautela por riesgos percibidos"
            
            if perspective == Perspective.BULLISH:
                reason += ", tendencia alcista clara"
            elif perspective == Perspective.CAUTIOUSLY_BULLISH:
                reason += ", tendencia alcista potencial"
        else:
            if signal < threshold - 0.2:
                reason = "Señal muy débil"
            elif signal < threshold - 0.1:
                reason = "Oportunidad insuficiente"
            else:
                reason = "Señal por debajo del umbral"
                
            # Añadir contexto de humor y perspectiva
            if mood == Mood.FEARFUL:
                reason += ", percepción de riesgo elevada"
            
            if perspective == Perspective.BEARISH:
                reason += ", tendencia bajista detectada"
            elif perspective == Perspective.CONFUSED:
                reason += ", mercado confuso"
        
        return reason
    
    async def adjust_size(self, base_size: float, mood: Mood, 
                        perspective: Perspective, confidence: float,
                        is_entry: bool = True) -> float:
        """
        Ajustar tamaño de posición según estado emocional y perspectiva.
        
        Args:
            base_size: Tamaño base recomendado
            mood: Estado emocional actual
            perspective: Perspectiva actual
            confidence: Nivel de confianza en la operación (0-1)
            is_entry: True si es entrada, False si es salida
            
        Returns:
            Tamaño ajustado
        """
        # Factor base según humor
        mood_factor = 1.0
        if mood == Mood.FEARFUL:
            mood_factor = 0.7  # Reduce tamaño en estado temeroso
        elif mood == Mood.HOPEFUL:
            mood_factor = 1.2  # Aumenta tamaño en estado esperanzado
        elif mood == Mood.CAUTIOUS:
            mood_factor = 0.85  # Reduce ligeramente en estado cauteloso
        
        # Factor según perspectiva
        perspective_factor = 1.0
        if perspective == Perspective.BULLISH and is_entry:
            perspective_factor = 1.2  # Aumenta en perspectiva alcista (entrada)
        elif perspective == Perspective.BEARISH and is_entry:
            perspective_factor = 0.8  # Reduce en perspectiva bajista (entrada)
        elif perspective == Perspective.CONFUSED:
            perspective_factor = 0.75  # Reduce en confusión
        
        # Factor según estilo de decisión
        style_factor = 1.0
        if self.decision_style == Decision.CAUTIOUS:
            style_factor = 0.8
        elif self.decision_style == Decision.AGGRESSIVE:
            style_factor = 1.3
        elif self.decision_style == Decision.IMPULSIVE:
            # Impulsivo puede variar más aleatoriamente
            style_factor = random.uniform(0.9, 1.5)
        
        # Confianza afecta directamente
        confidence_factor = 0.7 + (confidence * 0.6)  # Rango 0.7-1.3
        
        # Combinar todos los factores
        total_factor = mood_factor * perspective_factor * style_factor * confidence_factor
        
        # Limitar cambios extremos
        total_factor = max(0.5, min(total_factor, 1.5))
        
        # Aplicar factor a tamaño base
        adjusted_size = base_size * total_factor
        
        # Razonable variabilidad humana - pequeño elemento aleatorio
        human_variability = random.uniform(0.95, 1.05)
        adjusted_size *= human_variability
        
        # Ajustar a preferencia de riesgo general
        risk_adjustment = 0.8 + (self.risk_preference * 0.4)  # 0.8-1.2
        adjusted_size *= risk_adjustment
        
        # Redondear a 2 decimales para que parezca humano
        return round(adjusted_size * 100) / 100
    
    async def decide_exit(self, profit_percent: float, time_held_hours: float,
                        market_volatility: float, mood: Mood,
                        perspective: Perspective) -> Tuple[bool, str]:
        """
        Decidir si salir de una operación.
        
        Args:
            profit_percent: Porcentaje actual de beneficio (puede ser negativo)
            time_held_hours: Cuánto tiempo se ha mantenido la posición (horas)
            market_volatility: Volatilidad actual del mercado (0-1)
            mood: Estado emocional actual
            perspective: Perspectiva actual
            
        Returns:
            Tupla (decisión, razón)
        """
        # Actualizar estado interno primero
        await self.update(perspective, mood)
        
        # Stop loss automático
        if profit_percent <= -self.stop_loss_threshold:
            return True, f"Stop loss automático activado ({-self.stop_loss_threshold*100:.1f}%)"
        
        # Take profit automático - considerando humor y estilo
        effective_profit_threshold = self.exit_profit_threshold
        
        if mood == Mood.FEARFUL:
            effective_profit_threshold *= 0.8  # Sale antes si está temeroso
        elif mood == Mood.HOPEFUL and perspective in [Perspective.BULLISH, Perspective.CAUTIOUSLY_BULLISH]:
            effective_profit_threshold *= 1.3  # Espera más si está esperanzado y alcista
        
        if self.decision_style == Decision.CAUTIOUS:
            effective_profit_threshold *= 0.9  # Sale antes si es cauteloso
        elif self.decision_style == Decision.AGGRESSIVE:
            effective_profit_threshold *= 1.2  # Espera más si es agresivo
        
        # Salida por take profit
        if profit_percent >= effective_profit_threshold:
            # Verificar si es HOPEFUL y reciente, puede querer esperar más
            if mood == Mood.HOPEFUL and time_held_hours < 24 and profit_percent < effective_profit_threshold * 1.5:
                # 50% de probabilidad de mantener más tiempo
                if random.random() < 0.5:
                    return False, "Manteniendo posición rentable con perspectiva de más beneficio"
            
            return True, f"Objetivo de beneficio alcanzado ({profit_percent:.1f}%)"
        
        # Considerar volatilidad para decidir salida en pérdidas pequeñas
        if market_volatility > 0.7 and profit_percent < 0 and self.loss_aversion > 0.6:
            # Alta volatilidad y pérdida, posible salida si es averso a pérdidas
            chance_of_exit = market_volatility * self.loss_aversion
            if random.random() < chance_of_exit:
                return True, "Salida por alta volatilidad y pérdida actual"
        
        # Considerar paciencia para holds largos
        patience_factor = self.patience
        if time_held_hours > 72:  # 3 días o más
            # Test de paciencia para operaciones largas
            if random.random() > patience_factor:
                if profit_percent > 0:
                    return True, f"Realizando beneficio tras {time_held_hours:.1f} horas ({profit_percent:.1f}%)"
                else:
                    return True, f"Cerrando posición perdedora tras {time_held_hours:.1f} horas"
        
        # Decisión de mantener por defecto
        if profit_percent > 0:
            reason = "Manteniendo posición rentable"
        else:
            reason = "Esperando recuperación de posición"
        
        return False, reason
    
    def get_decision_style(self) -> Decision:
        """
        Obtener el estilo de decisión actual.
        
        Returns:
            Estilo de decisión
        """
        return self.decision_style
    
    def get_risk_preference(self) -> float:
        """
        Obtener la preferencia de riesgo actual.
        
        Returns:
            Nivel de preferencia de riesgo (0-1)
        """
        return self.risk_preference
    
    def adapt_to_perspective(self, perspective: Perspective) -> None:
        """
        Adaptar el estilo de decisión a la perspectiva para mantener coherencia.
        
        Args:
            perspective: Perspectiva actual
        """
        # Ajustes sutiles para mantener coherencia psicológica
        if perspective in [Perspective.BULLISH, Perspective.CAUTIOUSLY_BULLISH]:
            # Ajustar hacia estilos más optimistas
            if self.decision_style == Decision.CAUTIOUS and random.random() < 0.3:
                self.decision_style = Decision.BALANCED
        
        elif perspective in [Perspective.BEARISH, Perspective.CAUTIOUSLY_BEARISH]:
            # Ajustar hacia estilos más cautelosos
            if self.decision_style == Decision.AGGRESSIVE and random.random() < 0.3:
                self.decision_style = Decision.BALANCED
        
        elif perspective == Perspective.CONFUSED:
            # En confusión, tender hacia análisis o adaptación
            if random.random() < 0.3:
                self.decision_style = random.choice([Decision.ANALYTICAL, Decision.ADAPTIVE])
    
    async def randomize(self) -> None:
        """Aleatorizar el estilo de decisión y preferencias para mayor variabilidad."""
        # Elegir estilo aleatorio
        self.decision_style = random.choice(list(Decision))
        
        # Aleatorizar preferencias
        self.risk_preference = random.uniform(0.2, 0.8)
        self.patience = random.uniform(0.3, 0.9)
        self.conviction = random.uniform(0.3, 0.9)
        
        # Aleatorizar sesgos
        self.loss_aversion = random.uniform(0.4, 0.9)
        self.recency_bias = random.uniform(0.4, 0.8)
        self.confirmation_bias = random.uniform(0.3, 0.7)
        self.overconfidence = random.uniform(0.2, 0.7)
        
        # Aleatorizar umbrales
        self.min_signal_threshold = random.uniform(0.25, 0.5)
        self.exit_profit_threshold = random.uniform(0.1, 0.25)
        self.stop_loss_threshold = random.uniform(0.08, 0.15)
        
        # Registrar cambio
        self.decision_history.append({
            "style": self.decision_style,
            "risk_level": self.risk_preference,
            "timestamp": datetime.now()
        })
        
        logger.info(f"Voluntad aleatorizada a estilo {self.decision_style.name}, riesgo {self.risk_preference:.2f}")
    
    def reset(self) -> None:
        """Reiniciar el estado a valores predeterminados."""
        self.decision_style = Decision.BALANCED
        self.risk_preference = 0.5
        self.patience = 0.6
        self.conviction = 0.5
        self.loss_aversion = 0.7
        self.min_signal_threshold = 0.3
        self.exit_profit_threshold = 0.15
        self.stop_loss_threshold = 0.10
        self._adjust_thresholds_to_style()