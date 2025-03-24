"""
Adaptador del Motor de Comportamiento Humano Gabriel para el Sistema Genesis

Este módulo proporciona una capa de adaptación entre el nuevo motor de comportamiento
Gabriel (con su estructura poética de Alma, Mirada y Voluntad) y el sistema existente,
manteniendo la compatibilidad con las interfaces actuales.

Autor: Genesis AI Assistant
"""

import logging
import random
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio

from genesis.trading.gabriel.symphony import Gabriel
from genesis.trading.gabriel.soul import Mood
from genesis.trading.human_behavior_engine import EmotionalState, RiskTolerance, DecisionStyle

logger = logging.getLogger(__name__)

class GabrielAdapter:
    """
    Adaptador que conecta el nuevo motor Gabriel con el sistema existente,
    manteniendo la compatibilidad con la interfaz actual.
    """
    
    def __init__(self):
        """Inicializa el adaptador con una instancia del motor Gabriel."""
        self.gabriel = Gabriel(archetype="balanced_operator")
        
        # Mapeos para traducción entre sistemas
        self._mood_to_emotional_state = {
            "SERENE": EmotionalState.NEUTRAL,
            "HOPEFUL": EmotionalState.OPTIMISTIC,
            "WARY": EmotionalState.CAUTIOUS,
            "RESTLESS": EmotionalState.IMPATIENT,
            "BOLD": EmotionalState.CONFIDENT,
            "FRAUGHT": EmotionalState.ANXIOUS,
            "DREAD": EmotionalState.FEARFUL,
            "PENSIVE": EmotionalState.CONTEMPLATIVE
        }
        
        self._emotional_state_to_mood = {v.name: k for k, v in self._mood_to_emotional_state.items()}
        
        # Configuración inicial
        self._last_update = datetime.now()
        logger.info("GabrielAdapter inicializado correctamente")
    
    # === MÉTODOS DE COMPATIBILIDAD ===
    
    async def randomize_human_characteristics(self) -> Dict[str, Any]:
        """
        Emula el método existente randomize_human_characteristics()
        generando un estado aleatorio en Gabriel.
        
        Returns:
            Características humanas actualizadas en formato compatible
        """
        # Elegir un evento aleatorio para cambiar el estado emocional
        events = ["neutral", "opportunity", "stress", "victory", "defeat"]
        intensities = [0.3, 0.5, 0.7, 0.9]
        
        event = random.choice(events)
        intensity = random.choice(intensities)
        
        # Aplicar el evento al alma de Gabriel
        await self.gabriel.hear(event, intensity)
        
        # Traducir estado actual de Gabriel al formato esperado
        return self._translate_to_legacy_format()
    
    def get_current_characteristics(self) -> Dict[str, Any]:
        """
        Emula el método existente get_current_characteristics()
        traduciendo el estado actual de Gabriel al formato esperado.
        
        Returns:
            Características actuales en formato compatible
        """
        return self._translate_to_legacy_format()
    
    def set_emotional_state(self, state: EmotionalState, reason: str = "manual_override") -> None:
        """
        Emula el método existente set_emotional_state()
        traduciendo y aplicando el estado a Gabriel.
        
        Args:
            state: Estado emocional a establecer
            reason: Motivo del cambio
        """
        # Si es estado FEARFUL, usar método directo
        if state == EmotionalState.FEARFUL:
            self.gabriel.set_fearful(f"manual_override:{reason}")
            logger.info(f"Estado FEARFUL establecido en Gabriel: {reason}")
            return
            
        # Para otros estados, traducir y aplicar
        mood_name = self._emotional_state_to_mood.get(state.name, "SERENE")
        
        # Ejecutar de forma asíncrona
        async def _set_mood():
            await self.gabriel.hear(f"set_{mood_name.lower()}", 1.0)
        
        # Ejecutar la corutina
        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_set_mood(), loop)
            future.result()  # Esperar resultado
        else:
            loop.run_until_complete(_set_mood())
            
        logger.info(f"Estado emocional {state.name} establecido en Gabriel (como {mood_name})")
    
    def set_fearful_state(self, reason: str = "manual_override") -> None:
        """
        Emula el método existente set_fearful_state()
        estableciendo el estado DREAD en Gabriel.
        
        Args:
            reason: Motivo del cambio
        """
        self.gabriel.set_fearful(reason)
        logger.info(f"Estado FEARFUL establecido en Gabriel: {reason}")
    
    # === MÉTODOS DE COMPATIBILIDAD PARA TRADING ===
    
    async def should_enter_trade(self, signal_strength: float, 
                               market_context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Emula el método existente should_enter_trade()
        consultando la decisión de Gabriel.
        
        Args:
            signal_strength: Fuerza de la señal (0.0-1.0)
            market_context: Contexto del mercado
            
        Returns:
            (decisión, razón)
        """
        # Traducir contexto del mercado al formato que espera Gabriel
        gabriel_context = self._translate_market_context(market_context)
        
        # Actualizar percepción del mercado
        await self.gabriel.see(gabriel_context)
        
        # Consultar decisión de entrada
        decision, reason, _ = await self.gabriel.decide_entry(
            signal_strength, 
            {"market_vision": gabriel_context}
        )
        
        return decision, reason
    
    async def should_exit_trade(self, profit_percent: float, 
                              time_in_trade_hours: float, 
                              price_momentum: float) -> Tuple[bool, str]:
        """
        Emula el método existente should_exit_trade()
        consultando la decisión de salida de Gabriel.
        
        Args:
            profit_percent: Porcentaje de ganancia actual
            time_in_trade_hours: Tiempo en la operación (horas)
            price_momentum: Impulso del precio reciente
            
        Returns:
            (decisión, razón)
        """
        # Crear datos de posición para Gabriel
        position_data = {
            "profit_percent": profit_percent,
            "entry_time": datetime.now() - timedelta(hours=time_in_trade_hours),
            "price_momentum": price_momentum,
            "symbol": "GENERIC",
            "side": "buy"
        }
        
        # Consultar decisión de salida
        decision, reason, _ = await self.gabriel.decide_exit(position_data)
        
        return decision, reason
    
    async def adjust_order_size(self, base_size: float, 
                              confidence: float, 
                              is_buy: bool = True) -> float:
        """
        Emula el método existente adjust_order_size()
        consultando el ajuste de tamaño de Gabriel.
        
        Args:
            base_size: Tamaño base de la operación
            confidence: Nivel de confianza (0.0-1.0)
            is_buy: Si es una operación de compra
            
        Returns:
            Tamaño ajustado
        """
        # Consultar ajuste de tamaño
        adjusted_size, _ = await self.gabriel.size_position(
            base_size,
            is_buy,
            {"confidence": confidence}
        )
        
        return adjusted_size
    
    async def validate_trade(self, trade_params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Emula el método existente validate_trade()
        consultando la validación de Gabriel.
        
        Args:
            trade_params: Parámetros de la operación
            
        Returns:
            (válido, razón_rechazo)
        """
        # Consultar validación
        valid, reason, _ = await self.gabriel.validate_operation(trade_params)
        
        return valid, reason
    
    # === MÉTODOS DE ESTADO COMPATIBLES ===
    
    @property 
    def emotional_state(self) -> EmotionalState:
        """Emula la propiedad emotional_state."""
        gabriel_state = self.gabriel.get_current_state()
        mood_name = gabriel_state.get("emotional_state", "SERENE")
        
        return self._get_emotional_state_from_mood(mood_name)
    
    @property
    def risk_tolerance(self) -> RiskTolerance:
        """Emula la propiedad risk_tolerance."""
        gabriel_state = self.gabriel.get_current_state()
        courage = gabriel_state.get("courage", "BALANCED")
        
        if courage == "TIMID":
            return RiskTolerance.RISK_AVERSE
        elif courage == "BALANCED":
            return RiskTolerance.MODERATE
        elif courage == "DARING":
            return RiskTolerance.AGGRESSIVE
        else:
            return RiskTolerance.MODERATE
    
    @property
    def decision_style(self) -> DecisionStyle:
        """Emula la propiedad decision_style."""
        gabriel_state = self.gabriel.get_current_state()
        resolve = gabriel_state.get("resolve", "THOUGHTFUL")
        
        if resolve == "THOUGHTFUL":
            return DecisionStyle.ANALYTICAL
        elif resolve == "INSTINCTIVE":
            return DecisionStyle.IMPULSIVE
        elif resolve == "STEADFAST":
            return DecisionStyle.METHODICAL
        else:
            return DecisionStyle.ANALYTICAL
    
    @property
    def emotional_stability(self) -> float:
        """Emula la propiedad emotional_stability."""
        gabriel_state = self.gabriel.get_current_state()
        return gabriel_state.get("emotional_stability", 0.7)
    
    @property
    def risk_adaptation_rate(self) -> float:
        """Emula la propiedad risk_adaptation_rate."""
        if self.gabriel.is_fearful():
            return 0.9  # Alta adaptación en estado de miedo
        else:
            return 0.4  # Valor normal
    
    @property
    def contrarian_tendency(self) -> float:
        """Emula la propiedad contrarian_tendency."""
        if self.gabriel.is_fearful():
            return 0.1  # Baja tendencia contraria en miedo (sigue la manada)
        else:
            return 0.4  # Valor normal
    
    @property
    def decision_speed_multiplier(self) -> float:
        """Emula la propiedad decision_speed_multiplier."""
        if self.gabriel.is_fearful():
            return 2.0  # Decisiones más rápidas en miedo
        else:
            return 1.0  # Velocidad normal
    
    @property
    def market_perceptions(self) -> Dict[str, float]:
        """Emula la propiedad market_perceptions."""
        gabriel_state = self.gabriel.get_current_state()
        perception = gabriel_state.get("market_perception", {})
        
        # Traducir a formato compatible
        return {
            "perceived_volatility": perception.get("turbulence", 0.5),
            "opportunity_bias": perception.get("promise", 0.5),
            "risk_bias": perception.get("shadow", 0.5),
            "perceived_trend": self._translate_wind(perception.get("wind", "still")),
            "market_clarity": perception.get("clarity", 0.5)
        }
    
    # === MÉTODOS AUXILIARES ===
    
    def _translate_to_legacy_format(self) -> Dict[str, Any]:
        """
        Traduce el estado actual de Gabriel al formato esperado por el sistema existente.
        
        Returns:
            Estado en formato legacy
        """
        gabriel_state = self.gabriel.get_current_state()
        
        # Traducir estado emocional
        mood_name = gabriel_state.get("emotional_state", "SERENE")
        emotional_state = self._get_emotional_state_from_mood(mood_name)
        
        # Traducir otros estados
        courage = gabriel_state.get("courage", "BALANCED")
        resolve = gabriel_state.get("resolve", "THOUGHTFUL")
        
        risk_tolerance = RiskTolerance.RISK_AVERSE if courage == "TIMID" else \
                         RiskTolerance.AGGRESSIVE if courage == "DARING" else \
                         RiskTolerance.MODERATE
                         
        decision_style = DecisionStyle.ANALYTICAL if resolve == "THOUGHTFUL" else \
                         DecisionStyle.IMPULSIVE if resolve == "INSTINCTIVE" else \
                         DecisionStyle.METHODICAL
        
        # Formato compatible completo
        return {
            "emotional_state": emotional_state.name,
            "risk_tolerance": risk_tolerance.name,
            "decision_style": decision_style.name,
            "emotional_stability": gabriel_state.get("emotional_stability", 0.7),
            "risk_adaptation_rate": 0.9 if gabriel_state.get("is_fearful", False) else 0.4,
            "contrarian_tendency": 0.1 if gabriel_state.get("is_fearful", False) else 0.4,
            "decision_speed": 2.0 if gabriel_state.get("is_fearful", False) else 1.0,
            "market_perceptions": self._translate_market_perceptions(gabriel_state)
        }
    
    def _get_emotional_state_from_mood(self, mood_name: str) -> EmotionalState:
        """
        Traduce el nombre de un estado de ánimo de Gabriel a EmotionalState.
        
        Args:
            mood_name: Nombre del estado de ánimo en Gabriel
            
        Returns:
            EmotionalState correspondiente
        """
        emotional_state_name = self._mood_to_emotional_state.get(mood_name, EmotionalState.NEUTRAL)
        
        # Si es string, convertir a enum
        if isinstance(emotional_state_name, str):
            return getattr(EmotionalState, emotional_state_name)
            
        return emotional_state_name
    
    def _translate_market_perceptions(self, gabriel_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Traduce las percepciones de mercado de Gabriel al formato legacy.
        
        Args:
            gabriel_state: Estado completo de Gabriel
            
        Returns:
            Percepciones en formato legacy
        """
        perception = gabriel_state.get("market_perception", {})
        
        return {
            "perceived_volatility": perception.get("turbulence", 0.5),
            "opportunity_bias": perception.get("promise", 0.5),
            "risk_bias": perception.get("shadow", 0.5),
            "perceived_trend": self._translate_wind(perception.get("wind", "still")),
            "market_clarity": perception.get("clarity", 0.5)
        }
    
    def _translate_wind(self, wind: str) -> float:
        """
        Traduce la dirección del viento (mercado) de Gabriel a un valor numérico.
        
        Args:
            wind: Dirección del viento en Gabriel
            
        Returns:
            Valor numérico para perceived_trend
        """
        return {
            "rising": 0.8,
            "falling": 0.2,
            "still": 0.5,
            "trap": 0.3,
            "unstable": 0.45,
            "collapsing": 0.1
        }.get(wind, 0.5)
    
    def _translate_market_context(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traduce el contexto de mercado del formato legacy al formato que espera Gabriel.
        
        Args:
            market_context: Contexto en formato legacy
            
        Returns:
            Contexto en formato para Gabriel
        """
        # Extraer y normalizar datos
        volatility = market_context.get("volatility", 0.5)
        trend_value = market_context.get("trend", 0.5)
        volume_change = market_context.get("volume_change", 0.0)
        
        # Convertir trend numérico a categórico
        trend = "up" if trend_value > 0.6 else "down" if trend_value < 0.4 else "still"
        
        # Formato para Gabriel
        return {
            "volatility": volatility,
            "trend_direction": 1 if trend == "up" else -1 if trend == "down" else 0,
            "volume_change_percent": volume_change * 100,
            "price_direction": market_context.get("price_change", 0.0),
            "market_sentiment": market_context.get("sentiment", "neutral"),
            "risk_level": market_context.get("risk_level", 0.5)
        }

# Función auxiliar para crear adaptador singleton
_gabriel_adapter_instance = None

def get_gabriel_adapter() -> GabrielAdapter:
    """
    Obtiene la instancia singleton del adaptador de Gabriel.
    
    Returns:
        Instancia del adaptador
    """
    global _gabriel_adapter_instance
    if _gabriel_adapter_instance is None:
        _gabriel_adapter_instance = GabrielAdapter()
    return _gabriel_adapter_instance