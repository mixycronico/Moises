"""
Adaptador para el Motor de Comportamiento Humano Gabriel

Este módulo implementa un adaptador que integra el Motor de Comportamiento Humano Gabriel
con el resto del Sistema Genesis, proporcionando una interfaz consistente
para la simulación de trading con comportamiento humano.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

from genesis.trading.gabriel.symphony import Gabriel
from genesis.trading.gabriel.soul import Mood
from genesis.trading.gabriel.gaze import Perspective

logger = logging.getLogger(__name__)

class GabrielBehaviorEngine:
    """
    Adaptador para el Motor de Comportamiento Humano Gabriel.
    
    Esta clase proporciona una interfaz simplificada para integrar Gabriel
    con el resto de los componentes del sistema, ocultando los detalles
    de implementación internos y exponiendo solo las funcionalidades necesarias.
    """
    
    def __init__(self, archetype: str = "COLLECTIVE"):
        """
        Inicializar el adaptador de Gabriel.
        
        Args:
            archetype: Arquetipo de comportamiento ("BALANCED", "COLLECTIVE", etc.)
        """
        self.gabriel = Gabriel(archetype_name=archetype)
        self.is_initialized = False
        self.last_update = datetime.now()
        
        # Métricas de comportamiento
        self.metrics = {
            "total_decisions": 0,
            "positive_decisions": 0,
            "negative_decisions": 0,
            "mood_changes": 0,
        }
        
        logger.info(f"GabrielBehaviorEngine inicializado con arquetipo {archetype}")
    
    async def initialize(self) -> bool:
        """
        Inicializar el motor de comportamiento Gabriel.
        
        Returns:
            True si la inicialización fue exitosa
        """
        if self.is_initialized:
            return True
        
        try:
            result = await self.gabriel.initialize()
            self.is_initialized = result
            return result
        except Exception as e:
            logger.error(f"Error al inicializar GabrielBehaviorEngine: {str(e)}")
            return False
    
    async def process_news(self, news: Dict[str, Any]) -> None:
        """
        Procesar noticias y eventos externos.
        
        Args:
            news: Información de noticias y eventos
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            await self.gabriel.hear(news)
        except Exception as e:
            logger.error(f"Error al procesar noticias: {e}")
            # Caída segura en caso de error con el método hear
    
    async def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar datos de mercado a través de la percepción humana.
        
        Args:
            market_data: Datos crudos del mercado
            
        Returns:
            Interpretación de los datos filtrados por Gabriel
        """
        if not self.is_initialized:
            await self.initialize()
        
        result = await self.gabriel.see(market_data)
        return result
    
    async def react_to_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Reaccionar a eventos específicos del sistema.
        
        Args:
            event_type: Tipo de evento ("trade_executed", "error", "warning", etc.)
            event_data: Datos del evento
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Convertir evento a formato de noticias para Gabriel
        news = {
            "title": f"Event: {event_type}",
            "content": str(event_data),
            "sentiment": self._map_event_to_sentiment(event_type, event_data),
            "importance": self._calculate_event_importance(event_type, event_data),
            "related_to_portfolio": True,
            "impact": self._calculate_event_impact(event_type, event_data)
        }
        
        await self.gabriel.hear(news)
    
    def _map_event_to_sentiment(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """
        Mapear tipo de evento a sentimiento.
        
        Args:
            event_type: Tipo de evento
            event_data: Datos del evento
            
        Returns:
            Sentimiento ("bullish", "neutral", "bearish", etc.)
        """
        if "error" in event_type.lower():
            return "bearish"
        
        if "warning" in event_type.lower():
            return "slightly_bearish"
        
        if "profit" in event_type.lower():
            profit = event_data.get("profit", 0.0)
            if profit > 0:
                return "bullish"
            else:
                return "bearish"
        
        if "executed" in event_type.lower():
            return "neutral"
        
        return "neutral"
    
    def _calculate_event_importance(self, event_type: str, event_data: Dict[str, Any]) -> float:
        """
        Calcular importancia de un evento.
        
        Args:
            event_type: Tipo de evento
            event_data: Datos del evento
            
        Returns:
            Importancia (0-1)
        """
        if "error" in event_type.lower():
            return 0.8
        
        if "warning" in event_type.lower():
            return 0.6
        
        if "profit" in event_type.lower():
            profit = event_data.get("profit", 0.0)
            profit_pct = event_data.get("profit_percent", 0.0)
            
            # Mayor importancia a operaciones grandes
            if abs(profit_pct) > 0.1:  # Más del 10%
                return 0.8
            elif abs(profit_pct) > 0.05:  # Más del 5%
                return 0.6
            else:
                return 0.4
        
        if "executed" in event_type.lower():
            # Importancia según tamaño de operación
            size = event_data.get("size", 0.0)
            account_percentage = event_data.get("account_percentage", 0.0)
            
            if account_percentage > 0.1:  # Más del 10% de la cuenta
                return 0.7
            elif account_percentage > 0.05:  # Más del 5% de la cuenta
                return 0.5
            else:
                return 0.3
        
        return 0.5  # Valor por defecto
    
    def _calculate_event_impact(self, event_type: str, event_data: Dict[str, Any]) -> float:
        """
        Calcular impacto emocional de un evento.
        
        Args:
            event_type: Tipo de evento
            event_data: Datos del evento
            
        Returns:
            Impacto (-1 a 1)
        """
        if "error" in event_type.lower():
            return -0.7
        
        if "warning" in event_type.lower():
            return -0.4
        
        if "profit" in event_type.lower():
            profit = event_data.get("profit", 0.0)
            profit_pct = event_data.get("profit_percent", 0.0)
            
            # Mapear porcentaje a impacto
            impact = min(max(profit_pct * 10, -1.0), 1.0)
            return impact
        
        if "executed" in event_type.lower():
            # Impacto neutro ligeramente positivo
            return 0.1
        
        return 0.0  # Impacto neutro por defecto
    
    async def set_emergency_mode(self, emergency_type: str) -> None:
        """
        Establecer modo de emergencia según el tipo indicado.
        
        Args:
            emergency_type: Tipo de emergencia ("market_crash", "system_failure", "extreme_volatility", etc.)
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Modo de emergencia = establecer estado temeroso
        self.gabriel.set_fearful()
        
        logger.warning(f"Gabriel en modo de emergencia por {emergency_type}")
    
    async def normalize_state(self) -> None:
        """Normalizar el estado de Gabriel a un estado equilibrado."""
        if not self.is_initialized:
            await self.initialize()
        
        # Reiniciar a estado por defecto
        self.gabriel.reset()
        
        logger.info("Estado de Gabriel normalizado")
    
    async def evaluate_trade_opportunity(self, 
                                       symbol: str, 
                                       signal_strength: float, 
                                       market_data: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Evaluar una oportunidad de trading desde perspectiva humana.
        
        Args:
            symbol: Símbolo a evaluar
            signal_strength: Fuerza de la señal (0-1)
            market_data: Datos adicionales del mercado
            
        Returns:
            Tupla (decisión, razón, confianza)
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Añadir símbolo a datos de mercado
        market_data["symbol"] = symbol
        
        # Obtener decisión de Gabriel
        decision, reason, confidence = await self.gabriel.decide_entry(
            signal_strength,
            market_data
        )
        
        # Actualizar métricas
        self.metrics["total_decisions"] += 1
        if decision:
            self.metrics["positive_decisions"] += 1
        else:
            self.metrics["negative_decisions"] += 1
        
        return decision, reason, confidence
    
    async def evaluate_exit_opportunity(self,
                                      position_data: Dict[str, Any],
                                      market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluar una oportunidad de salida desde perspectiva humana.
        
        Args:
            position_data: Datos de la posición actual
            market_data: Datos actuales del mercado
            
        Returns:
            Tupla (decisión de salir, razón)
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Obtener decisión de Gabriel
        exit_decision, reason = await self.gabriel.decide_exit(
            position_data,
            market_data
        )
        
        return exit_decision, reason
    
    async def adjust_position_size(self, 
                                 base_size: float, 
                                 capital: float,
                                 risk_context: Dict[str, Any]) -> float:
        """
        Ajustar tamaño de posición según comportamiento humano.
        
        Args:
            base_size: Tamaño base recomendado por la estrategia
            capital: Capital total disponible
            risk_context: Contexto de riesgo (volatilidad, etc.)
            
        Returns:
            Tamaño ajustado de posición
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Obtener tamaño ajustado de Gabriel
        adjusted_size = await self.gabriel.size_position(
            base_size,
            capital,
            risk_context
        )
        
        return adjusted_size
    
    async def validate_operation(self, 
                               operation_type: str, 
                               details: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validar una operación según comportamiento humano.
        
        Args:
            operation_type: Tipo de operación ("entry", "exit", "adjust")
            details: Detalles de la operación
            
        Returns:
            Tupla (operación válida, razón)
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Obtener validación de Gabriel
        is_valid, reason = await self.gabriel.validate_operation(
            operation_type,
            details
        )
        
        return is_valid, reason
    
    def get_state(self) -> Dict[str, Any]:
        """
        Obtener el estado actual del comportamiento humano.
        
        Returns:
            Diccionario con estado actual
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        # Obtener estado actual de Gabriel
        gabriel_state = self.gabriel.get_current_state()
        
        # Añadir métricas
        state = {
            "status": "initialized",
            "mood": gabriel_state.get("mood", "NEUTRAL"),
            "mood_intensity": gabriel_state.get("mood_intensity", 0.5),
            "perspective": gabriel_state.get("perspective", "NEUTRAL"),
            "decision_style": gabriel_state.get("decision_style", "BALANCED"),
            "risk_preference": gabriel_state.get("risk_preference", 0.5),
            "archetype": gabriel_state.get("archetype", "COLLECTIVE"),
            "metrics": self.metrics
        }
        
        return state
    
    async def randomize(self) -> None:
        """Aleatorizar el comportamiento para mayor variabilidad."""
        if not self.is_initialized:
            await self.initialize()
        
        await self.gabriel.randomize()
        logger.info("Comportamiento de Gabriel aleatorizado")
    
    def get_risk_profile(self) -> Dict[str, float]:
        """
        Obtener el perfil de riesgo actual basado en el estado emocional.
        
        Returns:
            Diccionario con perfil de riesgo
        """
        if not self.is_initialized:
            return {"risk_appetite": 0.5, "risk_tolerance": 0.5}
        
        # Obtener estado actual de Gabriel
        gabriel_state = self.gabriel.get_current_state()
        
        # Calcular perfil de riesgo
        mood = gabriel_state.get("mood", "NEUTRAL")
        mood_intensity = gabriel_state.get("mood_intensity", 0.5)
        risk_preference = gabriel_state.get("risk_preference", 0.5)
        
        # Modificar apetito según estado emocional
        risk_appetite = risk_preference
        if mood == "FEARFUL":
            risk_appetite *= (1 - mood_intensity * 0.5)
        elif mood == "HOPEFUL":
            risk_appetite *= (1 + mood_intensity * 0.3)
        
        # Calcular tolerancia al riesgo
        risk_tolerance = risk_preference * 1.2  # Generalmente mayor que apetito
        if mood == "FEARFUL":
            risk_tolerance *= (1 - mood_intensity * 0.4)
        
        return {
            "risk_appetite": max(0.1, min(risk_appetite, 1.0)),
            "risk_tolerance": max(0.1, min(risk_tolerance, 1.0))
        }
    
    async def get_emotional_state(self) -> Dict[str, Any]:
        """
        Obtener estado emocional detallado.
        
        Returns:
            Diccionario con estado emocional detallado
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Obtener estado de Gabriel
        gabriel_state = self.gabriel.get_current_state()
        
        # Extraer componentes emocionales
        mood = gabriel_state.get("mood", "NEUTRAL")
        mood_intensity = gabriel_state.get("mood_intensity", 0.5)
        emotional_stability = gabriel_state.get("emotional_stability", 0.7)
        
        # Añadir interpretación humana
        mood_descriptions = {
            "SERENE": "Calma y equilibrio mental",
            "HOPEFUL": "Optimismo y expectativas positivas",
            "NEUTRAL": "Estado equilibrado sin tendencia marcada",
            "CAUTIOUS": "Precaución y cierta reserva",
            "RESTLESS": "Inquietud e impaciencia",
            "FEARFUL": "Temor ante riesgos percibidos"
        }
        
        intensity_description = "moderada"
        if mood_intensity > 0.8:
            intensity_description = "muy alta"
        elif mood_intensity > 0.6:
            intensity_description = "alta"
        elif mood_intensity < 0.4:
            intensity_description = "baja"
        elif mood_intensity < 0.2:
            intensity_description = "muy baja"
        
        return {
            "mood": mood,
            "mood_intensity": mood_intensity,
            "emotional_stability": emotional_stability,
            "description": mood_descriptions.get(mood, "Estado desconocido"),
            "intensity_description": intensity_description,
            "human_interpretation": f"Gabriel muestra un estado de {mood.lower()} con intensidad {intensity_description}."
        }