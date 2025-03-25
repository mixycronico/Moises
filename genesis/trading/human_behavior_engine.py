"""
Motor de comportamiento humano para trading

Este módulo proporciona una simulación de comportamiento humano para estrategias de trading,
permitiendo que las decisiones tengan un componente emocional que refleje patrones humanos.

Esta clase es un wrapper alrededor de Gabriel, para mantener compatibilidad con código existente.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import logging
import asyncio
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from genesis.trading.gabriel_adapter import GabrielBehaviorEngine

logger = logging.getLogger(__name__)

class HumanBehaviorEngine:
    """
    Motor de comportamiento humano para trading.
    
    Esta clase proporciona simulación de comportamiento humano para decisiones
    de trading, incluyendo aspectos emocionales y sesgos.
    
    Actúa como un wrapper alrededor de Gabriel, manteniendo compatibilidad
    con código existente que podría estar usando esta clase.
    """
    
    def __init__(self, archetype: str = "BALANCED"):
        """
        Inicializar motor de comportamiento.
        
        Args:
            archetype: Arquetipo de comportamiento
        """
        # Crear instancia de Gabriel
        self.gabriel = GabrielBehaviorEngine(archetype=archetype)
        self.is_initialized = False
        self.archetype = archetype
        
        logger.info(f"HumanBehaviorEngine inicializado con arquetipo {archetype}")
    
    async def initialize(self) -> bool:
        """
        Inicializar motor de comportamiento.
        
        Returns:
            True si la inicialización fue exitosa
        """
        if self.is_initialized:
            return True
        
        try:
            # Inicializar Gabriel
            result = await self.gabriel.initialize()
            self.is_initialized = result
            return result
        except Exception as e:
            logger.error(f"Error al inicializar HumanBehaviorEngine: {str(e)}")
            return False
    
    async def process_market_state(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar estado del mercado con perspectiva humana.
        
        Args:
            market_data: Datos del mercado
            
        Returns:
            Percepción humana del mercado
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Utilizar Gabriel para obtener percepción
        perception = await self.gabriel.process_market_data(market_data)
        return perception
    
    async def evaluate_signal(self, 
                           signal_strength: float, 
                           market_data: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Evaluar señal de trading con comportamiento humano.
        
        Args:
            signal_strength: Fuerza de la señal (0-1)
            market_data: Datos del mercado
            
        Returns:
            Tupla (decisión, razón, confianza)
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Extraer símbolo
        symbol = market_data.get("symbol", "UNKNOWN")
        
        # Utilizar Gabriel para evaluación
        decision, reason, confidence = await self.gabriel.evaluate_trade_opportunity(
            symbol, signal_strength, market_data
        )
        
        return decision, reason, confidence
    
    async def adjust_risk(self, 
                        base_risk: float, 
                        context: Dict[str, Any]) -> float:
        """
        Ajustar nivel de riesgo según estado emocional.
        
        Args:
            base_risk: Nivel de riesgo base (0-1)
            context: Contexto para ajuste
            
        Returns:
            Nivel de riesgo ajustado
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Obtener perfil de riesgo de Gabriel
        risk_profile = self.gabriel.get_risk_profile()
        
        # Aplicar ajuste basado en apetito y tolerancia
        appetite = risk_profile["risk_appetite"]
        tolerance = risk_profile["risk_tolerance"]
        
        # Fórmula de ajuste
        adjustment_factor = (appetite * 0.7) + (tolerance * 0.3)
        adjusted_risk = base_risk * adjustment_factor
        
        # Limitar a rango razonable
        adjusted_risk = max(0.1, min(adjusted_risk, 1.0))
        
        return adjusted_risk
    
    async def adjust_size(self, 
                        base_size: float, 
                        max_size: float, 
                        context: Dict[str, Any]) -> float:
        """
        Ajustar tamaño de posición según comportamiento humano.
        
        Args:
            base_size: Tamaño base calculado
            max_size: Tamaño máximo permitido
            context: Contexto para ajuste
            
        Returns:
            Tamaño ajustado
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Extraer información relevante
        volatility = context.get("volatility", 0.5)
        capital = context.get("capital", max_size * 10)
        
        # Llamar a Gabriel para ajustar tamaño
        risk_context = {
            "volatility": volatility,
            "max_risk": 0.05
        }
        
        adjusted_size = await self.gabriel.adjust_position_size(
            base_size, capital, risk_context
        )
        
        # Asegurar que no excede el máximo
        adjusted_size = min(adjusted_size, max_size)
        
        return adjusted_size
    
    async def evaluate_exit(self, 
                          position_data: Dict[str, Any],
                          market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluar salida de posición con comportamiento humano.
        
        Args:
            position_data: Datos de la posición actual
            market_data: Datos del mercado
            
        Returns:
            Tupla (decisión, razón)
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Utilizar Gabriel para evaluar salida
        exit_decision, reason = await self.gabriel.evaluate_exit_opportunity(
            position_data, market_data
        )
        
        return exit_decision, reason
    
    async def receive_news(self, news: Dict[str, Any]) -> None:
        """
        Procesar noticias que pueden afectar estado emocional.
        
        Args:
            news: Información de noticias
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Enviar noticia a Gabriel
        await self.gabriel.process_news(news)
    
    async def randomize_state(self) -> None:
        """Aleatorizar estado emocional para mayor variedad."""
        if not self.is_initialized:
            await self.initialize()
        
        await self.gabriel.randomize()
    
    async def get_state(self) -> Dict[str, Any]:
        """
        Obtener estado actual del motor.
        
        Returns:
            Diccionario con estado actual
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Combinar estado de Gabriel con configuración propia
        gabriel_state = self.gabriel.get_state()
        
        state = {
            "mood": gabriel_state.get("mood", "NEUTRAL"),
            "mood_intensity": gabriel_state.get("mood_intensity", 0.5),
            "perspective": gabriel_state.get("perspective", "NEUTRAL"),
            "risk_level": gabriel_state.get("risk_preference", 0.5),
            "archetype": self.archetype,
            "metrics": gabriel_state.get("metrics", {})
        }
        
        return state