"""
La Sinfonía de Gabriel - Orquestador del comportamiento humano celestial

Este módulo integra todos los componentes del motor de comportamiento humano
en una interfaz coherente y elegante, uniendo Alma, Mirada y Voluntad en
una simulación humana trascendente.
"""

from typing import Dict, Tuple, Any, Optional, List
from datetime import datetime, timedelta
import logging
import random
import asyncio

from .soul import Soul, Mood
from .gaze import Gaze
from .will import Will
from .essence import essence, archetypes

logger = logging.getLogger(__name__)

class Gabriel:
    """
    Orquestador principal del comportamiento humano celestial.
    
    Este compositor une los tres movimientos (Alma, Mirada, Voluntad)
    en una sinfonía armoniosa que simula comportamiento humano avanzado
    para trading, con especial énfasis en el estado FEARFUL (100% implementado).
    """
    
    def __init__(self, archetype: str = "balanced_operator", custom_config: Optional[Dict[str, Any]] = None):
        """
        Inicializar Gabriel con un arquetipo de personalidad.
        
        Args:
            archetype: Tipo de personalidad predefinida ("cautious_investor", "bold_trader", etc.)
            custom_config: Configuración personalizada que anula valores predeterminados
        """
        # Cargar configuración del arquetipo seleccionado
        config = archetypes.get(archetype, archetypes["balanced_operator"]).copy()
        
        # Aplicar configuraciones personalizadas si se proporcionan
        if custom_config:
            config.update(custom_config)
            
        # Inicializar los componentes principales
        self.soul = Soul(
            mood=getattr(Mood, config.get("base_mood", "SERENE")),
            stability=config.get("stability", 0.7),
            whimsy=config.get("whimsy", 0.15)
        )
        
        self.gaze = Gaze()
        
        self.will = Will(
            courage=config.get("courage", "BALANCED"),
            resolve=config.get("resolve", "THOUGHTFUL"),
            tenets=essence
        )
        
        # Atributos adicionales
        self.archetype = archetype
        self.config = config
        self.market_data_cache = {}
        self.last_major_decision = None
        self.last_market_analysis = None
        self.creation_time = datetime.now()
        self.emotional_triggers = {}
        
        logger.info(f"Gabriel inicializado con arquetipo: {archetype}")
        logger.info(f"Estado emocional inicial: {self.soul.mood.name}")
        logger.info(f"Configuración: Valor={self.will.courage}, Resolución={self.will.resolve}")
    
    # === INTERFAZ PRINCIPAL ===
    
    async def hear(self, whisper: str, intensity: float = 1.0) -> Mood:
        """
        Escucha los susurros del mundo y permite que el alma responda.
        
        Args:
            whisper: El estímulo o evento emocional ("victory", "loss", etc.)
            intensity: Intensidad del evento (0.0-1.0)
            
        Returns:
            El estado de ánimo resultante
        """
        # Si el evento es crítico y nos encontramos en un ciclo bajista, siempre ir a DREAD
        if whisper in ["market_crash", "system_failure", "catastrophic_loss"] and intensity >= 0.8:
            self.soul.set_dread(f"Evento crítico: {whisper}")
            return self.soul.mood
            
        # Registro de eventos emocionales
        if whisper not in self.emotional_triggers:
            self.emotional_triggers[whisper] = []
        self.emotional_triggers[whisper].append({
            "timestamp": datetime.now(),
            "intensity": intensity,
            "previous_mood": self.soul.mood.name
        })
        
        # Dejar que el alma procese el evento
        new_mood = await self.soul.sway(whisper, intensity, essence["emotional_echoes"])
        
        # Adaptaciones adicionales (ej. cambiar a estado de miedo manualmente)
        if whisper == "set_fearful" or (whisper == "loss" and intensity >= 0.9):
            self.soul.set_dread("Activación manual de estado de miedo")
            
        return self.soul.mood
    
    async def see(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza el mercado con ojos humanos, influenciados por el estado emocional.
        
        Args:
            market_data: Datos objetivos del mercado
            
        Returns:
            Percepción subjetiva del mercado y insights derivados
        """
        # Convertir datos del mercado a "presagios" para la mirada
        omens = self._translate_market_data(market_data)
        
        # Dejar que la mirada interprete los presagios
        vision = await self.gaze.behold(omens, self.soul.reflect())
        
        # Obtener insights adicionales
        insights = self.gaze.get_insights()
        
        # Combinar todo en una percepción completa
        perception = {**vision, **insights}
        
        # Guardar análisis para referencia futura
        self.last_market_analysis = {
            "timestamp": datetime.now(),
            "raw_data": market_data,
            "perceived": perception,
            "mood": self.soul.mood.name
        }
        
        return perception
    
    async def decide_entry(self, opportunity_score: float, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Decide si entrar en una operación basándose en la oportunidad y estado emocional.
        
        Args:
            opportunity_score: Puntuación de la oportunidad (0.0-1.0)
            context: Contexto adicional para la decisión
            
        Returns:
            (decisión, razón, detalles)
        """
        # Asegurar que tenemos percepción del mercado
        market_vision = context.get("market_vision", {}) if context else {}
        if not market_vision and self.last_market_analysis:
            market_vision = self.last_market_analysis.get("perceived", {})
            
        # Tomar la decisión
        decision, reason, details = await self.will.dare_to_enter(
            opportunity_score, 
            self.soul.reflect(),
            market_vision
        )
        
        # Registrar decisión importante
        if decision:
            self.last_major_decision = {
                "type": "entry",
                "timestamp": datetime.now(),
                "result": "approved",
                "opportunity": opportunity_score,
                "mood": self.soul.mood.name,
                "reason": reason
            }
        
        return decision, reason, details
    
    async def decide_exit(self, position_data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Decide si salir de una posición existente.
        
        Args:
            position_data: Datos de la posición actual (ganancia, tiempo, etc.)
            
        Returns:
            (decisión, razón, detalles)
        """
        # Extraer información clave
        profit = position_data.get("profit_percent", 0.0)
        entry_time = position_data.get("entry_time", datetime.now() - timedelta(hours=1))
        price_momentum = position_data.get("price_momentum", 0.0)
        
        # Tomar la decisión
        decision, reason, details = await self.will.choose_to_flee(
            profit,
            entry_time,
            price_momentum,
            self.soul.reflect(),
            position_data
        )
        
        # Registrar decisión importante
        if decision:
            self.last_major_decision = {
                "type": "exit",
                "timestamp": datetime.now(),
                "result": "approved",
                "profit": profit,
                "mood": self.soul.mood.name,
                "reason": reason
            }
        
        return decision, reason, details
    
    async def size_position(self, base_size: float, is_buy: bool = True, 
                          context: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Ajusta el tamaño de la posición según estado emocional y contexto.
        
        Args:
            base_size: Tamaño base recomendado
            is_buy: Si es una operación de compra
            context: Contexto adicional
            
        Returns:
            (tamaño_ajustado, detalles)
        """
        # Extraer nivel de confianza del contexto
        confidence = 0.5  # Valor por defecto
        if context:
            confidence = context.get("confidence", 0.5)
        elif self.last_market_analysis:
            confidence = self.last_market_analysis.get("perceived", {}).get("confidence", 0.5)
        
        # Ajustar el tamaño
        adjusted_size, details = await self.will.adjust_position_size(
            base_size,
            self.soul.reflect(),
            confidence,
            is_buy
        )
        
        return adjusted_size, details
    
    async def validate_operation(self, operation_params: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Valida una operación propuesta según criterios subjetivos y estado emocional.
        
        Args:
            operation_params: Parámetros de la operación a validar
            
        Returns:
            (válido, razón_rechazo, detalles)
        """
        # Validar la operación
        valid, reject_reason, details = await self.will.validate_trade(
            operation_params,
            self.soul.reflect()
        )
        
        # Registrar resultado de validación
        if not valid:
            self.last_major_decision = {
                "type": "validation",
                "timestamp": datetime.now(),
                "result": "rejected",
                "operation": operation_params.get("side", "unknown"),
                "reason": reject_reason,
                "mood": self.soul.mood.name
            }
        
        return valid, reject_reason, details
    
    # === MÉTODOS DE ESTADO ===
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Obtiene el estado completo actual del comportamiento.
        
        Returns:
            Diccionario con el estado completo
        """
        return {
            "emotional_state": self.soul.mood.name,
            "is_fearful": self.soul.mood.is_fearful,
            "market_perception": self.gaze.visions,
            "archetype": self.archetype,
            "courage": self.will.courage,
            "resolve": self.will.resolve,
            "emotional_stability": self.soul.stability,
            "last_mood_change": self.soul.last_shift.isoformat(),
            "mood_duration_hours": self.soul.mood_duration,
            "last_decision": self.last_major_decision
        }
    
    def set_fearful(self, reason: str = "manual_activation") -> None:
        """
        Activa directamente el estado de miedo (FEARFUL/DREAD).
        
        Args:
            reason: Motivo para activar el estado de miedo
        """
        self.soul.set_dread(reason)
        logger.warning(f"ESTADO DE MIEDO ACTIVADO: {reason}")
    
    def is_fearful(self) -> bool:
        """
        Verifica si Gabriel está actualmente en estado de miedo.
        
        Returns:
            True si está en estado DREAD (miedo)
        """
        return self.soul.mood.is_fearful
    
    # === MÉTODOS AUXILIARES ===
    
    def _translate_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traduce datos crudos del mercado a "presagios" para la mirada.
        
        Args:
            market_data: Datos crudos del mercado
            
        Returns:
            Datos traducidos como presagios
        """
        # Extraer y normalizar valores clave
        volatility = market_data.get("volatility", 0.5)
        trend = "up" if market_data.get("trend_direction", 0) > 0 else \
                "down" if market_data.get("trend_direction", 0) < 0 else "still"
        volume_change = market_data.get("volume_change_percent", 0.0) / 100.0
        price_direction = market_data.get("price_direction", 0.0)
        
        # Cachear datos para referencia
        self.market_data_cache = {
            "timestamp": datetime.now(),
            "data": market_data
        }
        
        # Convertir a formato de presagios
        return {
            "volatility": volatility,
            "trend": trend,
            "volume_change": volume_change,
            "price_direction": price_direction,
            "market_sentiment": market_data.get("market_sentiment", "neutral"),
            "risk_level": market_data.get("risk_level", 0.5)
        }
    
    async def simulate_reaction(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simula una reacción completa a un evento del mercado.
        
        Args:
            event: Evento del mercado (datos, tipo, etc.)
            
        Returns:
            Resultado completo de la reacción
        """
        # Extraer información del evento
        event_type = event.get("type", "market_update")
        event_data = event.get("data", {})
        event_intensity = event.get("intensity", 0.5)
        
        # 1. Respuesta emocional al evento
        emotional_trigger = {
            "market_update": "neutral",
            "price_jump": "opportunity",
            "price_drop": "stress",
            "volume_spike": "alert",
            "trend_change": "recalibration"
        }.get(event_type, "neutral")
        
        await self.hear(emotional_trigger, event_intensity)
        
        # 2. Percepción actualizada del mercado
        perception = await self.see(event_data)
        
        # 3. Decisión simulada (si aplica)
        decision = None
        if event_type in ["trading_opportunity", "signal_alert"]:
            score = event.get("opportunity_score", 0.5)
            would_enter, reason, details = await self.decide_entry(score, {"market_vision": perception})
            decision = {
                "would_enter": would_enter,
                "reason": reason,
                "details": details
            }
        
        # Resultado completo
        return {
            "emotional_state": self.soul.mood.name,
            "is_fearful": self.soul.mood.is_fearful,
            "market_perception": perception,
            "decision": decision,
            "event_type": event_type,
            "response_time": datetime.now().isoformat()
        }