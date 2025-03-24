"""
Motor de Comportamiento Humano (Human Behavior Engine) - Gabriel

Este módulo celestial implementa el corazón del sistema de simulación 
de comportamiento humano para la estrategia Seraphim Pool, permitiendo
que el Sistema Genesis opere con patrones humanos realistas y evite
la detección algorítmica.

Proporciona:
- Patrones de comportamiento humano para decisiones de trading
- Simulación de pausas estratégicas y contemplación
- Introducción de variabilidad subjetiva en decisiones
- Protección contra patrones algorítmicos detectables

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum, auto
import uuid
import numpy as np
from collections import deque

# Configuración de logging
logger = logging.getLogger(__name__)

class EmotionalState(Enum):
    """Estados emocionales simulados para decisiones humanas."""
    NEUTRAL = auto()       # Estado equilibrado, decisiones racionales
    OPTIMISTIC = auto()    # Tendencia a ver oportunidades, más riesgo
    CAUTIOUS = auto()      # Tendencia a evitar pérdidas, menos riesgo
    IMPATIENT = auto()     # Tendencia a decisiones rápidas, menos análisis
    CONFIDENT = auto()     # Tendencia a posiciones más grandes
    ANXIOUS = auto()       # Tendencia a cerrar posiciones pronto
    FEARFUL = auto()       # Miedo extremo, tendencia a reducir posiciones significativamente
    CONTEMPLATIVE = auto() # Tendencia a analizar más tiempo

class RiskTolerance(Enum):
    """Niveles de tolerancia al riesgo para simulación humana."""
    RISK_AVERSE = auto()      # Evita riesgos agresivamente
    CONSERVATIVE = auto()     # Prefiere seguridad pero acepta algo de riesgo
    MODERATE = auto()         # Balance entre riesgo y seguridad
    GROWTH_ORIENTED = auto()  # Acepta más riesgo por más rendimiento
    AGGRESSIVE = auto()       # Busca alto rendimiento, acepta alto riesgo

class DecisionStyle(Enum):
    """Estilos de toma de decisiones para simulación humana."""
    ANALYTICAL = auto()      # Basado en análisis detallado
    INTUITIVE = auto()       # Basado en "sensaciones" e intuición
    METHODICAL = auto()      # Basado en procesos y reglas
    IMPULSIVE = auto()       # Decisiones rápidas con poca contemplación
    CONSULTATIVE = auto()    # Busca múltiples opiniones

class GabrielBehaviorEngine:
    """
    Motor de Comportamiento Humano Gabriel.
    
    Esta clase implementa el sistema de simulación de comportamiento humano
    para la estrategia Seraphim Pool, integrando patrones naturales en
    el proceso de trading automatizado.
    """
    
    def __init__(self):
        """Inicializar motor de comportamiento humano Gabriel."""
        # Estado emocional actual y características de comportamiento
        self.emotional_state = EmotionalState.NEUTRAL
        self.risk_tolerance = RiskTolerance.MODERATE
        self.decision_style = DecisionStyle.ANALYTICAL
        
        # Evolución de estado
        self.emotional_stability = 0.7  # 0-1, qué tan estable es el estado emocional
        self.risk_adaptation_rate = 0.3  # 0-1, qué tan rápido se adapta la tolerancia al riesgo
        self.last_state_change = datetime.now()
        
        # Historial para simulación de patrones más realistas
        self.recent_decisions = deque(maxlen=10)
        self.market_perceptions = {
            "perceived_volatility": 0.5,  # 0-1
            "perceived_opportunity": 0.5,  # 0-1
            "perceived_risk": 0.5,  # 0-1
            "market_sentiment": "neutral"  # bullish, bearish, neutral
        }
        
        # Parámetros de simulación
        self.decision_speed_multiplier = 1.0  # Afecta tiempos de pausa
        self.randomness_factor = 0.2  # Nivel de aleatorización en decisiones
        self.contrarian_tendency = 0.1  # Tendencia a ir contra la corriente
        
        # ID y configuración
        self.instance_id = str(uuid.uuid4())
        self.configuration = self._create_default_configuration()
        
        logger.info("Motor de Comportamiento Humano Gabriel inicializado")
    
    def _create_default_configuration(self) -> Dict[str, Any]:
        """
        Crear configuración predeterminada.
        
        Returns:
            Configuración predeterminada
        """
        return {
            "emotional_state_shifts": {
                "on_profit": 0.2,  # Hacia optimista
                "on_loss": -0.3,   # Hacia ansioso
                "on_market_upturn": 0.1,  # Hacia optimista
                "on_market_downturn": -0.2,  # Hacia cauteloso
                "on_trade_success": 0.15,  # Hacia confiado
                "on_trade_failure": -0.25,  # Hacia ansioso
                "natural_decay": 0.05  # Hacia neutral
            },
            "contemplation_times": {
                EmotionalState.NEUTRAL.name: (10, 20),  # (min, max) segundos
                EmotionalState.OPTIMISTIC.name: (5, 15),
                EmotionalState.CAUTIOUS.name: (15, 30),
                EmotionalState.IMPATIENT.name: (3, 8),
                EmotionalState.CONFIDENT.name: (7, 12),
                EmotionalState.ANXIOUS.name: (12, 25),
                EmotionalState.FEARFUL.name: (18, 35),  # Tendencia a demorar decisiones por miedo
                EmotionalState.CONTEMPLATIVE.name: (20, 40)
            },
            "decision_thresholds": {
                RiskTolerance.RISK_AVERSE.name: 0.8,  # Umbral para entrar
                RiskTolerance.CONSERVATIVE.name: 0.7,
                RiskTolerance.MODERATE.name: 0.6,
                RiskTolerance.GROWTH_ORIENTED.name: 0.5,
                RiskTolerance.AGGRESSIVE.name: 0.4
            },
            "subjective_rejection_chance": {
                DecisionStyle.ANALYTICAL.name: 0.05,  # Probabilidad de rechazo subjetivo
                DecisionStyle.INTUITIVE.name: 0.3,
                DecisionStyle.METHODICAL.name: 0.1,
                DecisionStyle.IMPULSIVE.name: 0.2,
                DecisionStyle.CONSULTATIVE.name: 0.15
            }
        }
    
    async def update_emotional_state(self, trigger: str, magnitude: float = 1.0) -> EmotionalState:
        """
        Actualizar estado emocional basado en un evento desencadenante.
        
        Args:
            trigger: Tipo de evento ("profit", "loss", "market_upturn", etc.)
            magnitude: Magnitud del cambio (multiplicador)
            
        Returns:
            Nuevo estado emocional
        """
        # Mapear trigger a shift
        shift_key = f"on_{trigger}" if f"on_{trigger}" in self.configuration["emotional_state_shifts"] else "natural_decay"
        shift = self.configuration["emotional_state_shifts"][shift_key] * magnitude
        
        # Aplicar estabilidad emocional como amortiguador
        shift *= (1 - self.emotional_stability)
        
        # Aplicar cambio
        current_states = list(EmotionalState)
        current_index = current_states.index(self.emotional_state)
        
        # Si shift es positivo, moverse hacia emociones positivas
        # Si es negativo, hacia emociones negativas
        if shift > 0:
            new_index = min(current_index + 1, len(current_states) - 1)
        elif shift < 0:
            new_index = max(current_index - 1, 0)
        else:
            new_index = current_index
        
        # Elementos aleatorios: a veces los humanos cambian emocionalmente sin razón clara
        if random.random() < self.randomness_factor:
            random_shift = random.choice([-1, 1])
            new_index = max(0, min(len(current_states) - 1, new_index + random_shift))
        
        # Actualizar estado
        old_state = self.emotional_state
        self.emotional_state = current_states[new_index]
        self.last_state_change = datetime.now()
        
        logger.debug(f"Estado emocional cambió: {old_state.name} -> {self.emotional_state.name} (trigger: {trigger})")
        
        return self.emotional_state
    
    async def get_contemplation_time(self, operation_type: str = "general") -> float:
        """
        Obtener tiempo de contemplación basado en estado emocional actual.
        
        Args:
            operation_type: Tipo de operación ("entry", "exit", "general", etc.)
            
        Returns:
            Tiempo de contemplación en segundos
        """
        # Obtener rango para estado actual
        time_range = self.configuration["contemplation_times"].get(
            self.emotional_state.name, 
            (10, 20)  # Default
        )
        
        # Aplicar modificadores
        base_time = random.uniform(time_range[0], time_range[1])
        
        # Modificador específico para FEARFUL (estado de miedo)
        if self.emotional_state == EmotionalState.FEARFUL:
            if operation_type == "entry":
                # Mucho más tiempo para decidir entrar (miedo a tomar riesgos)
                base_time *= 2.5
            elif operation_type == "exit":
                # Mucho menos tiempo para decidir salir (reacción de pánico)
                base_time *= 0.3
            else:
                # Para otras operaciones, tiempo moderadamente alto (cautela)
                base_time *= 1.5
        
        # Modificador por estilo de decisión
        if self.decision_style == DecisionStyle.ANALYTICAL:
            base_time *= 1.3
        elif self.decision_style == DecisionStyle.IMPULSIVE:
            base_time *= 0.6
        elif self.decision_style == DecisionStyle.METHODICAL:
            base_time *= 1.2
        
        # Aplicar multiplicador de velocidad general
        base_time *= self.decision_speed_multiplier
        
        logger.debug(f"Tiempo de contemplación calculado: {base_time:.2f}s para estado {self.emotional_state.name} en operación {operation_type}")
        
        return base_time
    
    async def simulate_human_delay(self, operation_type: str) -> None:
        """
        Simular retraso humano para una operación.
        
        Args:
            operation_type: Tipo de operación ("analysis", "trade_entry", "trade_exit", etc.)
        """
        # Mapear tipo de operación a nuestros tipos de contemplación
        contemplation_type = "general"
        if operation_type in ["trade_entry", "entry_analysis"]:
            contemplation_type = "entry"
        elif operation_type in ["trade_exit", "exit_analysis", "emergency_exit"]:
            contemplation_type = "exit"
        
        # Calcular tiempo de retraso utilizando el tipo específico
        contemplation_time = await self.get_contemplation_time(contemplation_type)
        
        # Ajustes adicionales según la operación específica
        if operation_type == "quick_decision":
            contemplation_time *= 0.5
        elif operation_type == "emergency_exit":
            contemplation_time *= 0.3
        elif operation_type == "complex_analysis":
            contemplation_time *= 1.5
        
        logger.debug(f"Simulando retraso humano de {contemplation_time:.2f}s para {operation_type}")
        
        # En entorno de producción, podríamos desactivar retrasos reales
        # await asyncio.sleep(contemplation_time)
        
        # Registrar decisión para patrones
        self.recent_decisions.append({
            "operation": operation_type,
            "delay": contemplation_time,
            "timestamp": datetime.now().isoformat()
        })
    
    async def should_enter_trade(self, opportunity_score: float, asset_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Decidir si entrar en una operación con comportamiento humano.
        
        Args:
            opportunity_score: Puntuación de oportunidad (0-1)
            asset_data: Datos del activo
            
        Returns:
            Tupla (decisión, razón)
        """
        # Obtener umbral basado en tolerancia al riesgo
        threshold = self.configuration["decision_thresholds"].get(
            self.risk_tolerance.name, 
            0.6  # Default
        )
        
        # Aplicar modificadores de estado emocional
        if self.emotional_state == EmotionalState.OPTIMISTIC:
            threshold *= 0.8  # Más propenso a entrar
        elif self.emotional_state == EmotionalState.CAUTIOUS:
            threshold *= 1.2  # Menos propenso a entrar
        elif self.emotional_state == EmotionalState.CONFIDENT:
            threshold *= 0.85
        elif self.emotional_state == EmotionalState.ANXIOUS:
            threshold *= 1.3
        
        # Percepción del mercado
        if self.market_perceptions["market_sentiment"] == "bullish":
            threshold *= 0.9
        elif self.market_perceptions["market_sentiment"] == "bearish":
            threshold *= 1.1
        
        # Factor contrario (a veces los humanos van contra la corriente)
        if random.random() < self.contrarian_tendency:
            logger.debug("Aplicando tendencia contraria")
            threshold = 1.0 - threshold
        
        # Rechazo subjetivo
        subjective_rejection_chance = self.configuration["subjective_rejection_chance"].get(
            self.decision_style.name, 
            0.1  # Default
        )
        
        if random.random() < subjective_rejection_chance:
            logger.debug(f"Rechazo subjetivo aplicado para {asset_data.get('symbol', 'desconocido')}")
            return False, "subjective_rejection"
        
        # Tomar decisión
        should_enter = opportunity_score >= threshold
        
        reason = (
            "opportunity_exceeds_threshold" if should_enter 
            else "opportunity_below_threshold"
        )
        
        logger.debug(f"Decisión de entrada: {should_enter} para {asset_data.get('symbol', 'desconocido')} "
                   f"(score: {opportunity_score:.2f}, threshold: {threshold:.2f})")
        
        return should_enter, reason
    
    async def should_exit_trade(
        self, 
        unrealized_pnl_pct: float, 
        asset_data: Dict[str, Any],
        entry_time: datetime,
        price_change_rate: float
    ) -> Tuple[bool, str]:
        """
        Decidir si salir de una operación con comportamiento humano.
        
        Args:
            unrealized_pnl_pct: Ganancia/pérdida no realizada en porcentaje
            asset_data: Datos del activo
            entry_time: Tiempo de entrada
            price_change_rate: Tasa de cambio de precio reciente
            
        Returns:
            Tupla (decisión, razón)
        """
        # Umbrales base según tolerancia al riesgo
        take_profit_thresholds = {
            RiskTolerance.RISK_AVERSE.name: 5.0,      # %
            RiskTolerance.CONSERVATIVE.name: 7.0,
            RiskTolerance.MODERATE.name: 10.0,
            RiskTolerance.GROWTH_ORIENTED.name: 15.0,
            RiskTolerance.AGGRESSIVE.name: 20.0
        }
        
        stop_loss_thresholds = {
            RiskTolerance.RISK_AVERSE.name: -3.0,     # %
            RiskTolerance.CONSERVATIVE.name: -5.0,
            RiskTolerance.MODERATE.name: -8.0,
            RiskTolerance.GROWTH_ORIENTED.name: -12.0,
            RiskTolerance.AGGRESSIVE.name: -15.0
        }
        
        # Obtener umbrales para tolerancia actual
        take_profit = take_profit_thresholds.get(self.risk_tolerance.name, 10.0)
        stop_loss = stop_loss_thresholds.get(self.risk_tolerance.name, -8.0)
        
        # Aplicar modificadores de estado emocional
        if self.emotional_state == EmotionalState.OPTIMISTIC:
            take_profit *= 1.2  # Espera más ganancia
            stop_loss *= 1.1    # Tolera más pérdida
        elif self.emotional_state == EmotionalState.CAUTIOUS:
            take_profit *= 0.8  # Acepta menos ganancia
            stop_loss *= 0.9    # Tolera menos pérdida
        elif self.emotional_state == EmotionalState.IMPATIENT:
            take_profit *= 0.7  # Acepta menos ganancia
            stop_loss *= 0.8    # Tolera menos pérdida
        elif self.emotional_state == EmotionalState.CONFIDENT:
            take_profit *= 1.3  # Espera más ganancia
            stop_loss *= 1.2    # Tolera más pérdida
        elif self.emotional_state == EmotionalState.ANXIOUS:
            take_profit *= 0.6  # Acepta menos ganancia
            stop_loss *= 0.7    # Tolera menos pérdida
        elif self.emotional_state == EmotionalState.FEARFUL:
            take_profit *= 0.5  # Sale rápidamente con cualquier ganancia
            stop_loss *= 0.5    # No tolera casi ninguna pérdida
        
        # Factor de tiempo transcurrido: los humanos se impacientan con el tiempo
        time_held = datetime.now() - entry_time
        hours_held = time_held.total_seconds() / 3600
        
        # Impaciencia creciente con el tiempo
        if hours_held > 12:
            take_profit *= 0.9
            stop_loss *= 0.9
        if hours_held > 24:
            take_profit *= 0.8
            stop_loss *= 0.8
        
        # Factor de dirección de precio reciente
        if price_change_rate < -0.01:  # Caída reciente
            take_profit *= 0.9
            stop_loss *= 0.95
        elif price_change_rate > 0.01:  # Subida reciente
            take_profit *= 1.05
            stop_loss *= 1.05
        
        # Rechazo subjetivo (a veces mantienen contra toda lógica)
        subjective_retention = random.random() < 0.1
        if subjective_retention and unrealized_pnl_pct > stop_loss and unrealized_pnl_pct < take_profit:
            logger.debug(f"Retención subjetiva aplicada para {asset_data.get('symbol', 'desconocido')}")
            return False, "subjective_retention"
        
        # Tomar decisión
        if unrealized_pnl_pct >= take_profit:
            return True, "take_profit_triggered"
        elif unrealized_pnl_pct <= stop_loss:
            return True, "stop_loss_triggered"
        
        # Decisiones especiales
        if self.emotional_state == EmotionalState.IMPATIENT and hours_held > 6:
            # Impaciencia lleva a salir "porque sí"
            return True, "impatience_exit"
        
        if self.emotional_state == EmotionalState.ANXIOUS and price_change_rate < -0.005:
            # Ansiedad lleva a salir ante pequeñas caídas
            return True, "anxiety_exit"
            
        if self.emotional_state == EmotionalState.FEARFUL:
            # Estado temeroso - reacciones extremas ante cualquier señal negativa
            if price_change_rate < 0 or unrealized_pnl_pct < 0:
                # Cualquier movimiento negativo o operación en pérdida provoca salida
                return True, "fear_driven_exit"
            elif hours_held > 3:
                # También salida por tiempo si la posición se mantiene demasiado tiempo
                return True, "fear_extended_holding_exit"
        
        # Mantener posición
        return False, "holding_position"
    
    async def adjust_asset_allocation(
        self, 
        base_allocations: Dict[str, float],
        total_capital: float
    ) -> Dict[str, float]:
        """
        Ajustar asignación de capital con comportamiento humano.
        
        Args:
            base_allocations: Asignaciones base
            total_capital: Capital total disponible
            
        Returns:
            Asignaciones ajustadas
        """
        adjusted = {}
        remaining_capital = total_capital
        
        # Ordenar activos (los humanos tienen preferencias)
        sorted_assets = sorted(
            base_allocations.items(),
            key=lambda x: random.random()  # Ordernar con algo de aleatoriedad
        )
        
        for symbol, allocation in sorted_assets:
            # Ajustar según estado emocional
            if self.emotional_state == EmotionalState.CONFIDENT:
                adjustment = random.uniform(1.1, 1.3)  # Posiciones más grandes
            elif self.emotional_state == EmotionalState.ANXIOUS:
                adjustment = random.uniform(0.7, 0.9)  # Posiciones más pequeñas
            elif self.emotional_state == EmotionalState.FEARFUL:
                adjustment = random.uniform(0.4, 0.6)  # Posiciones muy pequeñas por miedo
            elif self.emotional_state == EmotionalState.OPTIMISTIC:
                adjustment = random.uniform(1.05, 1.2)  # Ligeramente más grandes
            elif self.emotional_state == EmotionalState.CAUTIOUS:
                adjustment = random.uniform(0.8, 0.95)  # Ligeramente más pequeñas
            else:
                adjustment = random.uniform(0.9, 1.1)  # Variación ligera
            
            # Aplicar ajuste
            adjusted_allocation = allocation * adjustment
            
            # Asegurar que no exceda el capital restante
            adjusted_allocation = min(adjusted_allocation, remaining_capital)
            
            # Aplicar redondeo humano (a menudo redondean a valores "bonitos")
            if adjusted_allocation > 100:
                adjusted_allocation = round(adjusted_allocation / 10) * 10
            elif adjusted_allocation > 10:
                adjusted_allocation = round(adjusted_allocation)
            
            adjusted[symbol] = adjusted_allocation
            remaining_capital -= adjusted_allocation
        
        # Los humanos a veces dejan algo sin invertir intencionalmente
        if self.emotional_state == EmotionalState.FEARFUL:
            # En estado de miedo extremo, mantienen gran parte sin invertir (hasta un 50%)
            fearful_reserve = max(remaining_capital, total_capital * 0.5)
            logger.debug(f"Manteniendo gran reserva por miedo: ${fearful_reserve:.2f}")
            # Si hemos asignado demasiado, reducir proporcionalmente
            if fearful_reserve > remaining_capital:
                reduction_needed = fearful_reserve - remaining_capital
                total_allocated = sum(adjusted.values())
                if total_allocated > 0:
                    for symbol in adjusted:
                        reduction_ratio = adjusted[symbol] / total_allocated
                        adjusted[symbol] -= reduction_needed * reduction_ratio
                    remaining_capital = fearful_reserve
        elif self.emotional_state == EmotionalState.CAUTIOUS:
            logger.debug(f"Manteniendo reserva cautelosa: ${remaining_capital:.2f}")
        elif self.risk_tolerance in [RiskTolerance.RISK_AVERSE, RiskTolerance.CONSERVATIVE]:
            logger.debug(f"Manteniendo reserva conservadora: ${remaining_capital:.2f}")
        elif remaining_capital > 0 and remaining_capital < total_capital * 0.1:
            # Si queda poco, a veces los humanos lo distribuyen
            for symbol in adjusted:
                adjusted[symbol] += remaining_capital / len(adjusted)
                remaining_capital = 0
        
        logger.debug(f"Asignaciones ajustadas con comportamiento humano, capital sin asignar: ${remaining_capital:.2f}")
        
        return adjusted
    
    async def update_market_perception(self, market_data: Dict[str, Any]) -> None:
        """
        Actualizar percepción del mercado según los datos recibidos.
        
        Args:
            market_data: Datos recientes del mercado
        """
        # Extraer datos relevantes
        volatility = market_data.get("volatility", 0.5)
        trend = market_data.get("trend", "neutral")
        volume = market_data.get("volume_change", 0.0)
        
        # Actualizar percepción con sesgo humano
        # Los humanos tienden a sobre-reaccionar a eventos recientes
        current_perception = self.market_perceptions["perceived_volatility"]
        recency_bias = 0.7  # Cuánto peso dar a datos recientes vs previos
        
        self.market_perceptions["perceived_volatility"] = (
            current_perception * (1 - recency_bias) + 
            volatility * recency_bias
        )
        
        # Actualizar sentimiento
        if trend == "up" and volume > 0.1:
            self.market_perceptions["market_sentiment"] = "bullish"
        elif trend == "down" and volume > 0.1:
            self.market_perceptions["market_sentiment"] = "bearish"
        else:
            # Mantener sentimiento con ligero sesgo a neutral
            current_sentiment = self.market_perceptions["market_sentiment"]
            if random.random() < 0.2:  # 20% de probabilidad de normalización
                self.market_perceptions["market_sentiment"] = "neutral"
        
        # Percepción de oportunidad y riesgo
        if trend == "up":
            # En tendencia alcista los humanos ven más oportunidad, menos riesgo
            self.market_perceptions["perceived_opportunity"] = min(
                1.0, self.market_perceptions["perceived_opportunity"] + 0.1
            )
            self.market_perceptions["perceived_risk"] = max(
                0.1, self.market_perceptions["perceived_risk"] - 0.05
            )
        elif trend == "down":
            # En tendencia bajista los humanos ven menos oportunidad, más riesgo
            self.market_perceptions["perceived_opportunity"] = max(
                0.1, self.market_perceptions["perceived_opportunity"] - 0.1
            )
            self.market_perceptions["perceived_risk"] = min(
                1.0, self.market_perceptions["perceived_risk"] + 0.1
            )
        
        logger.debug(f"Percepción de mercado actualizada: {self.market_perceptions}")
    
    def randomize_human_characteristics(self) -> Dict[str, Any]:
        """
        Aleatorizar características humanas para simular diferentes personalidades.
        
        Returns:
            Características actualizadas
        """
        # Aleatorizar estado emocional
        self.emotional_state = random.choice(list(EmotionalState))
        
        # Aleatorizar tolerancia al riesgo
        self.risk_tolerance = random.choice(list(RiskTolerance))
        
        # Aleatorizar estilo de decisión
        self.decision_style = random.choice(list(DecisionStyle))
        
        # Aleatorizar estabilidad emocional (0.4-0.9)
        self.emotional_stability = random.uniform(0.4, 0.9)
        
        # Aleatorizar adaptabilidad al riesgo (0.2-0.7)
        self.risk_adaptation_rate = random.uniform(0.2, 0.7)
        
        # Aleatorizar tendencia contraria (0.05-0.3)
        self.contrarian_tendency = random.uniform(0.05, 0.3)
        
        # Aleatorizar multiplicador de velocidad (0.8-1.5)
        self.decision_speed_multiplier = random.uniform(0.8, 1.5)
        
        logger.info(f"Características humanas aleatorizadas: "
                  f"Estado emocional={self.emotional_state.name}, "
                  f"Tolerancia al riesgo={self.risk_tolerance.name}, "
                  f"Estilo de decisión={self.decision_style.name}")
        
        return self.get_current_characteristics()
    
    # Alias para mantener compatibilidad con la integración del orquestador
    def randomize(self) -> Dict[str, Any]:
        """
        Alias para randomize_human_characteristics para compatibilidad con el orquestador.
        
        Returns:
            Características actualizadas
        """
        return self.randomize_human_characteristics()
    
    def get_current_characteristics(self) -> Dict[str, Any]:
        """
        Obtener características humanas actuales.
        
        Returns:
            Características actuales
        """
        return {
            "emotional_state": self.emotional_state.name,
            "risk_tolerance": self.risk_tolerance.name,
            "decision_style": self.decision_style.name,
            "emotional_stability": self.emotional_stability,
            "risk_adaptation_rate": self.risk_adaptation_rate,
            "contrarian_tendency": self.contrarian_tendency,
            "decision_speed": self.decision_speed_multiplier,
            "market_perceptions": self.market_perceptions
        }
    
    # Propiedades alias para compatibilidad con el orquestador
    @property
    def mood(self) -> str:
        """Alias del estado emocional para compatibilidad."""
        return self.emotional_state.name
    
    @property
    def risk_profile(self) -> str:
        """Alias de tolerancia al riesgo para compatibilidad."""
        return self.risk_tolerance.name
    
    @property
    def experience_level(self) -> str:
        """Nivel de experiencia simulado (derivado del estilo de decisión)."""
        if self.decision_style == DecisionStyle.ANALYTICAL:
            return "advanced"
        elif self.decision_style == DecisionStyle.METHODICAL:
            return "intermediate"
        elif self.decision_style == DecisionStyle.IMPULSIVE:
            return "beginner"
        else:
            return "intermediate"
    
    def get_current_state(self) -> Dict[str, Any]:
        """Devolver estado actual para compatibilidad con orquestador."""
        return self.get_current_characteristics()
    
    async def get_human_trading_preferences(self) -> Dict[str, Any]:
        """
        Obtener preferencias de trading basadas en características actuales.
        
        Returns:
            Preferencias de trading
        """
        # Obtener tiempos de operación preferidos
        if self.decision_style == DecisionStyle.METHODICAL:
            # Preferencia por operaciones más estructuradas, en horarios específicos
            preferred_times = ["morning", "evening"]
        elif self.decision_style == DecisionStyle.IMPULSIVE:
            # Puede operar en cualquier momento
            preferred_times = ["any"]
        else:
            # Preferencia moderada
            preferred_times = ["morning", "afternoon", "evening"]
        
        # Duración preferida de operaciones
        if self.risk_tolerance in [RiskTolerance.AGGRESSIVE, RiskTolerance.GROWTH_ORIENTED]:
            # Preferencia por operaciones más cortas
            preferred_duration = "short"
        elif self.risk_tolerance in [RiskTolerance.RISK_AVERSE, RiskTolerance.CONSERVATIVE]:
            # Preferencia por operaciones más largas
            preferred_duration = "long"
        else:
            # Duración moderada
            preferred_duration = "medium"
        
        # Enfoque de entrada/salida
        if self.emotional_state == EmotionalState.FEARFUL:
            # Entradas muy calculadas, salidas extremadamente rápidas
            entry_exit_style = "panic_prone"
        elif self.emotional_state in [EmotionalState.ANXIOUS, EmotionalState.IMPATIENT]:
            # Entradas/salidas más rápidas
            entry_exit_style = "quick"
        elif self.emotional_state in [EmotionalState.CONTEMPLATIVE, EmotionalState.CAUTIOUS]:
            # Entradas/salidas más calculadas
            entry_exit_style = "calculated"
        else:
            # Estilo moderado
            entry_exit_style = "balanced"
        
        return {
            "preferred_times": preferred_times,
            "preferred_duration": preferred_duration,
            "entry_exit_style": entry_exit_style,
            "position_sizing": self._get_position_sizing_preference(),
            "diversification_preference": self._get_diversification_preference()
        }
    
    def _get_position_sizing_preference(self) -> str:
        """
        Obtener preferencia de tamaño de posición basada en características actuales.
        
        Returns:
            Preferencia de tamaño de posición
        """
        if self.emotional_state == EmotionalState.CONFIDENT:
            return "large"
        elif self.emotional_state == EmotionalState.ANXIOUS:
            return "small"
        elif self.emotional_state == EmotionalState.FEARFUL:
            return "very_small"  # Posiciones extremadamente pequeñas cuando hay miedo
        elif self.risk_tolerance == RiskTolerance.AGGRESSIVE:
            return "large"
        elif self.risk_tolerance == RiskTolerance.RISK_AVERSE:
            return "small"
        else:
            return "medium"
    
    def _get_diversification_preference(self) -> str:
        """
        Obtener preferencia de diversificación basada en características actuales.
        
        Returns:
            Preferencia de diversificación
        """
        if self.emotional_state == EmotionalState.FEARFUL:
            return "very_high"  # Máxima diversificación en estado temeroso
        elif self.risk_tolerance in [RiskTolerance.RISK_AVERSE, RiskTolerance.CONSERVATIVE]:
            return "high"  # Alta diversificación
        elif self.risk_tolerance == RiskTolerance.AGGRESSIVE:
            return "low"   # Baja diversificación
        elif self.decision_style == DecisionStyle.METHODICAL:
            return "high"  # Alta diversificación
        elif self.decision_style == DecisionStyle.IMPULSIVE:
            return "low"   # Baja diversificación
        else:
            return "medium"  # Diversificación moderada