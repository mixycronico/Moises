"""
Sinfonía de Gabriel - Integración armónica de los componentes humanos

Este módulo implementa la clase principal "Gabriel" que coordina los tres componentes:
Alma (Soul), Mirada (Gaze) y Voluntad (Will), creando una simulación coherente
de comportamiento humano para trading.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import random

from genesis.trading.gabriel.soul import Soul, Mood
from genesis.trading.gabriel.gaze import Gaze, Perspective
from genesis.trading.gabriel.will import Will, Decision
from genesis.trading.gabriel.essence import archetypes

logger = logging.getLogger(__name__)

class Gabriel:
    """
    Motor de comportamiento humano Gabriel para trading.
    
    Esta clase orquesta los tres componentes principales:
    - Alma (Soul): Estados emocionales y personalidad
    - Mirada (Gaze): Percepción e interpretación del mercado
    - Voluntad (Will): Toma de decisiones y acciones
    
    Juntos forman un sistema coherente que simula el comportamiento humano
    en el contexto de trading, con todas sus complejidades emocionales,
    percepciones sesgadas y estilos de decisión variables.
    """
    
    def __init__(self, archetype_name: str = "BALANCED"):
        """
        Inicializar el motor de comportamiento Gabriel.
        
        Args:
            archetype_name: Nombre del arquetipo a utilizar ("BALANCED", "CONSERVATIVE", etc.)
        """
        # Crear componentes principales
        self.soul = Soul()
        self.gaze = Gaze()
        self.will = Will()
        
        # Estado actual
        self.current_market_data = {}
        self.current_positions = {}
        self.performance_metrics = {}
        
        # Configuración
        self.archetype_name = archetype_name
        
        # Estado interno
        self.last_update = datetime.now()
        self.is_initialized = False
        
        logger.info(f"Gabriel inicializado con arquetipo {archetype_name}")
    
    async def initialize(self) -> bool:
        """
        Inicializar completamente el motor de comportamiento Gabriel.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            # Inicializar componentes
            await self.soul.initialize()
            await self.gaze.initialize()
            await self.will.initialize()
            
            # Aplicar arquetipo para mantener coherencia
            self.apply_archetype(self.archetype_name)
            
            # Sincronizar componentes para coherencia
            self._synchronize_components()
            
            self.is_initialized = True
            logger.info(f"Gabriel inicializado exitosamente con arquetipo {self.archetype_name}")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar Gabriel: {str(e)}")
            return False
    
    def apply_archetype(self, archetype_name: str) -> None:
        """
        Aplicar un arquetipo predefinido a todos los componentes.
        
        Args:
            archetype_name: Nombre del arquetipo a aplicar
        """
        if archetype_name not in archetypes:
            logger.warning(f"Arquetipo {archetype_name} no encontrado, usando BALANCED")
            archetype_name = "BALANCED"
        
        archetype = archetypes[archetype_name]
        
        # Aplicar configuración a cada componente
        self.soul.apply_archetype(archetype["soul"])
        self.gaze.apply_archetype(archetype["gaze"])
        self.will.apply_archetype(archetype["will"])
        
        self.archetype_name = archetype_name
        logger.info(f"Arquetipo {archetype_name} aplicado: {archetype['description']}")
    
    def _synchronize_components(self) -> None:
        """Sincronizar componentes para mantener coherencia psicológica."""
        # Alma afecta Mirada
        mood = self.soul.get_mood()
        self.gaze.adapt_to_mood(mood)
        
        # Mirada afecta Voluntad
        perspective = self.gaze.get_perspective()
        self.will.adapt_to_perspective(perspective)
        
        # Voluntad afecta Alma (ciclo completo)
        decision_style = self.will.get_decision_style()
        self.soul.adapt_to_decision_style(decision_style)
    
    async def update(self, market_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Actualizar el estado interno basado en nuevos datos de mercado.
        
        Args:
            market_data: Nuevos datos de mercado
        """
        try:
            # Si hay datos nuevos, actualizar estado de mercado
            if market_data:
                self.current_market_data.update(market_data)
            
            # Actualizar componentes
            await self.soul.update(market_data)
            await self.gaze.update(market_data, self.soul.get_mood())
            await self.will.update(self.gaze.get_perspective(), self.soul.get_mood())
            
            # Sincronizar para mantener coherencia
            self._synchronize_components()
            
            # Actualizar timestamp
            self.last_update = datetime.now()
        except Exception as e:
            logger.error(f"Error al actualizar Gabriel: {str(e)}")
    
    async def hear(self, news: Dict[str, Any]) -> None:
        """
        Procesar noticias y eventos externos.
        
        Args:
            news: Información de noticias y eventos
        """
        # Extraer información relevante
        sentiment = news.get("sentiment", "neutral")
        importance = news.get("importance", 0.5)
        related_to_portfolio = news.get("related_to_portfolio", False)
        
        # Preparar datos para el Alma
        soul_data = {
            "sentiment": sentiment,
            "importance": importance,
            "impact": news.get("impact", 0.0)
        }
        
        # Preparar datos para la Mirada
        gaze_data = {
            "sentiment": sentiment,
            "conflicting_signals": news.get("conflicting_signals", 0.0),
            "technical_signals": news.get("technical_signals", 0.5)
        }
        
        # Actualizar componentes
        await self.soul.update(soul_data)
        await self.gaze.update(gaze_data, self.soul.get_mood())
        
        # Mayor impacto si está relacionado con el portfolio
        if related_to_portfolio and importance > 0.7:
            # Actualizar Will directamente si es importante y relacionado
            await self.will.update(self.gaze.get_perspective(), self.soul.get_mood())
        
        # Sincronizar para mantener coherencia
        self._synchronize_components()
        
        logger.debug(f"Gabriel procesó news con sentiment {sentiment}, importance {importance}")
    
    async def see(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar y filtrar datos de mercado a través de la percepción humana.
        
        Args:
            market_data: Datos crudos del mercado
            
        Returns:
            Interpretación de los datos filtrados por Gabriel
        """
        # Actualizar componentes con nuevos datos
        await self.update(market_data)
        
        # Obtener percepciones actuales
        mood = self.soul.get_mood()
        mood_intensity = self.soul.get_mood_intensity()
        perspective = self.gaze.get_perspective()
        perspective_confidence = self.gaze.get_perspective_confidence()
        
        # Crear respuesta con interpretación humana
        interpretation = {
            "mood": mood.name,
            "mood_intensity": mood_intensity,
            "perspective": perspective.name,
            "confidence": perspective_confidence,
            "market_perception": self.gaze.get_market_perception(),
            "perceived_trend": self.gaze.perceived_trend,
            "human_interpretation": self._generate_market_interpretation(mood, perspective)
        }
        
        return interpretation
    
    def _generate_market_interpretation(self, mood: Mood, perspective: Perspective) -> str:
        """
        Generar interpretación del mercado en lenguaje humano.
        
        Args:
            mood: Estado emocional actual
            perspective: Perspectiva actual
            
        Returns:
            Interpretación en lenguaje natural
        """
        # Frases según estados emocionales
        mood_phrases = {
            Mood.SERENE: [
                "El mercado parece estable y ordenado",
                "Veo patrones claros y definidos",
                "La situación actual permite análisis objetivo"
            ],
            Mood.HOPEFUL: [
                "Hay buenas oportunidades en este mercado",
                "La tendencia parece prometedora",
                "Veo potencial en la dirección actual"
            ],
            Mood.NEUTRAL: [
                "El mercado está en un estado equilibrado",
                "No veo señales claras en ninguna dirección",
                "Es un momento para observar atentamente"
            ],
            Mood.CAUTIOUS: [
                "Hay que proceder con precaución en este mercado",
                "No todo está claro en la tendencia actual",
                "Es mejor mantener cierta reserva ahora"
            ],
            Mood.RESTLESS: [
                "El mercado está moviéndose demasiado rápido",
                "Hay mucha actividad y volatilidad",
                "Es difícil seguir los cambios constantes"
            ],
            Mood.FEARFUL: [
                "El mercado muestra signos preocupantes",
                "Veo riesgos significativos en este momento",
                "Es momento de proteger el capital"
            ]
        }
        
        # Frases según perspectiva de mercado
        perspective_phrases = {
            Perspective.BULLISH: [
                "Tendencia alcista clara",
                "Potencial de crecimiento significativo",
                "Señales de compra fuertes"
            ],
            Perspective.CAUTIOUSLY_BULLISH: [
                "Tendencia positiva moderada",
                "Potencial alcista con reservas",
                "Oportunidades selectivas de compra"
            ],
            Perspective.NEUTRAL: [
                "Mercado sin dirección clara",
                "Balance entre factores positivos y negativos",
                "Momento de evaluar cuidadosamente"
            ],
            Perspective.CAUTIOUSLY_BEARISH: [
                "Señales de debilidad emergentes",
                "Posible corrección a la baja",
                "Aumenta la precaución"
            ],
            Perspective.BEARISH: [
                "Tendencia bajista confirmada",
                "Riesgos significativos a la baja",
                "Señales de venta presentes"
            ],
            Perspective.CONFUSED: [
                "Señales contradictorias",
                "Difícil interpretar la dirección",
                "Alta incertidumbre en el mercado"
            ]
        }
        
        # Seleccionar frases aleatorias para crear interpretación
        mood_phrase = random.choice(mood_phrases.get(mood, mood_phrases[Mood.NEUTRAL]))
        perspective_phrase = random.choice(perspective_phrases.get(perspective, perspective_phrases[Perspective.NEUTRAL]))
        
        return f"{mood_phrase}. {perspective_phrase}."
    
    async def decide_entry(self, signal_strength: float, market_data: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Decidir si entrar en una operación de trading.
        
        Args:
            signal_strength: Fuerza de la señal (0-1)
            market_data: Datos actuales del mercado
            
        Returns:
            Tupla (decisión, razón, confianza)
        """
        # Actualizar estado con datos de mercado actuales
        await self.update(market_data)
        
        # Obtener estado actual
        mood = self.soul.get_mood()
        perspective = self.gaze.get_perspective()
        
        # Decisión de la Voluntad
        decision, reason = await self.will.decide_trade(
            signal_strength, 
            mood, 
            perspective,
            market_data
        )
        
        # Nivel de confianza combinado
        confidence = self.will.conviction * self.gaze.get_perspective_confidence()
        
        return decision, reason, confidence
    
    async def size_position(self, base_size: float, capital: float, 
                          risk_context: Dict[str, Any]) -> float:
        """
        Ajustar tamaño de posición según estado emocional y contexto.
        
        Args:
            base_size: Tamaño base recomendado por la estrategia
            capital: Capital total disponible
            risk_context: Contexto de riesgo (volatilidad, etc.)
            
        Returns:
            Tamaño ajustado de posición
        """
        # Obtener estado actual
        mood = self.soul.get_mood()
        perspective = self.gaze.get_perspective()
        confidence = self.gaze.get_perspective_confidence()
        
        # Ajustar según Voluntad
        adjusted_size = await self.will.adjust_size(
            base_size, 
            mood, 
            perspective, 
            confidence
        )
        
        # Verificar que no excede límites razonables
        max_allowed = capital * 0.25  # Máximo 25% del capital
        adjusted_size = min(adjusted_size, max_allowed)
        
        return adjusted_size
    
    async def decide_exit(self, position_data: Dict[str, Any], 
                        market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Decidir si salir de una posición existente.
        
        Args:
            position_data: Datos de la posición actual
            market_data: Datos actuales del mercado
            
        Returns:
            Tupla (decisión de salir, razón)
        """
        # Actualizar estado con datos de mercado actuales
        await self.update(market_data)
        
        # Obtener información relevante
        profit_percent = position_data.get("profit_percent", 0.0)
        entry_time = position_data.get("entry_time", None)
        symbol = position_data.get("symbol", "")
        
        # Calcular tiempo transcurrido
        if entry_time:
            now = datetime.now()
            time_delta = now - entry_time
            time_held_hours = time_delta.total_seconds() / 3600
        else:
            time_held_hours = 0.0
        
        # Obtener estado actual
        mood = self.soul.get_mood()
        perspective = self.gaze.get_perspective()
        
        # Extraer volatilidad del mercado o usar valor por defecto
        market_volatility = market_data.get("volatility", 0.5)
        
        # Decisión de la Voluntad
        exit_decision, reason = await self.will.decide_exit(
            profit_percent,
            time_held_hours,
            market_volatility,
            mood,
            perspective
        )
        
        return exit_decision, reason
    
    async def validate_operation(self, operation_type: str, details: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validar una operación según el estado emocional y contexto.
        
        Args:
            operation_type: Tipo de operación ("entry", "exit", "adjust")
            details: Detalles de la operación
            
        Returns:
            Tupla (operación válida, razón)
        """
        # Obtener estado actual
        mood = self.soul.get_mood()
        mood_intensity = self.soul.get_mood_intensity()
        
        # Verificaciones específicas según estado emocional
        if mood == Mood.FEARFUL and mood_intensity > 0.7:
            if operation_type == "entry":
                # Alta precaución en estado temeroso
                return False, "Operación rechazada por alta precaución en estado actual"
            elif operation_type == "exit" and details.get("profit_percent", 0.0) > 0:
                # Tomar ganancias en estado temeroso
                return True, "Asegurando ganancias en un entorno percibido como riesgoso"
        
        return True, "Operación validada"
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Obtener el estado emocional y de percepción actual.
        
        Returns:
            Diccionario con estado actual completo
        """
        try:
            mood = self.soul.get_mood()
            mood_intensity = self.soul.get_mood_intensity()
            perspective = self.gaze.get_perspective()
            perspective_confidence = self.gaze.get_perspective_confidence()
            decision_style = self.will.get_decision_style()
            risk_preference = self.will.get_risk_preference()
            emotional_stability = self.soul.get_emotional_stability()
            market_perception = self.gaze.get_market_perception()
            
            state = {
                "mood": mood.name,
                "mood_intensity": mood_intensity,
                "perspective": perspective.name,
                "perspective_confidence": perspective_confidence,
                "decision_style": decision_style.name,
                "risk_preference": risk_preference,
                "emotional_stability": emotional_stability,
                "market_perception": market_perception,
                "archetype": self.archetype_name
            }
            
            return state
        except Exception as e:
            logger.error(f"Error al obtener estado actual: {str(e)}")
            return {"error": str(e)}
    
    async def randomize(self) -> None:
        """Aleatorizar todos los componentes para mayor variabilidad."""
        # Aleatorizar Alma
        await self.soul.randomize()
        
        # Aleatorizar Mirada (con nuevo estado de Alma)
        await self.gaze.randomize(self.soul.get_mood())
        
        # Aleatorizar Voluntad
        await self.will.randomize()
        
        # Sincronizar componentes
        self._synchronize_components()
        
        logger.info("Gabriel aleatorizado completamente")
    
    def set_fearful(self) -> None:
        """Establecer un estado temeroso para situaciones de emergencia."""
        # Aplicar arquetipo conservador temporalmente
        self.apply_archetype("GUARDIAN")
        
        # Forzar un estado temeroso en el Alma
        self.soul.reset()
        self.soul.current_mood = Mood.FEARFUL
        self.soul.mood_intensity = 0.9
        
        # Sincronizar componentes para coherencia
        self._synchronize_components()
        
        logger.warning("Gabriel establecido en estado temeroso de emergencia")
    
    def reset(self) -> None:
        """Reiniciar el estado a valores predeterminados."""
        self.soul.reset()
        self.gaze.reset()
        self.will.reset()
        self._synchronize_components()
        logger.info("Gabriel reiniciado a valores predeterminados")