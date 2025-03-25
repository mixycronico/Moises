"""
Gestor de Integraciones para Aetherion.

Este módulo coordina la comunicación entre Aetherion y los otros componentes
especializados del Sistema Genesis:
1. Buddha - Análisis de mercado y previsiones
2. Gabriel - Simulación de comportamiento humano
3. DeepSeek - Análisis semántico avanzado

Cada integración proporciona capacidades específicas que enriquecen
la consciencia global de Aetherion.
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Union, Tuple

# Importaciones internas
from genesis.behavior.gabriel_engine import GabrielBehaviorEngine
from genesis.consciousness.memory.memory_system import MemorySystem

# Configurar logging
logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Tipos de integraciones disponibles."""
    BUDDHA = auto()    # Análisis de mercado y oráculos
    GABRIEL = auto()   # Comportamiento humano
    DEEPSEEK = auto()  # Análisis semántico
    CUSTOM = auto()    # Integraciones personalizadas

class IntegrationCapability:
    """Capacidades de una integración."""
    
    def __init__(self, name: str, description: str, integration_type: IntegrationType):
        """
        Inicializar capacidad.
        
        Args:
            name: Nombre de la capacidad
            description: Descripción
            integration_type: Tipo de integración
        """
        self.name = name
        self.description = description
        self.integration_type = integration_type
        self.is_available = False
        self.last_check = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con información de la capacidad
        """
        return {
            "name": self.name,
            "description": self.description,
            "integration_type": self.integration_type.name,
            "is_available": self.is_available,
            "last_check": self.last_check.isoformat()
        }

class IntegrationManager:
    """Gestor de integraciones para Aetherion."""
    
    def __init__(self):
        """Inicializar gestor de integraciones."""
        # Componentes integrados
        self.memory_system: Optional[MemorySystem] = None
        self.behavior_engine: Optional[GabrielBehaviorEngine] = None
        
        # Capacidades disponibles
        self.capabilities: Dict[str, IntegrationCapability] = {}
        
        # Estado
        self.is_initialized = False
        self.last_status_check = datetime.now()
        
        # Configurar capacidades predeterminadas
        self._setup_default_capabilities()
        
        logger.info("IntegrationManager inicializado")
    
    def _setup_default_capabilities(self) -> None:
        """Configurar capacidades predeterminadas."""
        # Buddha
        self.capabilities["buddha_market_analysis"] = IntegrationCapability(
            "Análisis de Mercado Buddha",
            "Análisis de mercado superior con múltiples dimensiones",
            IntegrationType.BUDDHA
        )
        
        self.capabilities["buddha_prediction"] = IntegrationCapability(
            "Predicciones Buddha",
            "Predicciones de mercado a corto y medio plazo",
            IntegrationType.BUDDHA
        )
        
        # Gabriel
        self.capabilities["gabriel_emotional_analysis"] = IntegrationCapability(
            "Análisis Emocional Gabriel",
            "Análisis de estados emocionales humanos",
            IntegrationType.GABRIEL
        )
        
        self.capabilities["gabriel_behavioral_simulation"] = IntegrationCapability(
            "Simulación Comportamental Gabriel",
            "Simulación de comportamiento humano para decisiones",
            IntegrationType.GABRIEL
        )
        
        # DeepSeek
        self.capabilities["deepseek_semantic_analysis"] = IntegrationCapability(
            "Análisis Semántico DeepSeek",
            "Análisis semántico profundo de textos y conceptos",
            IntegrationType.DEEPSEEK
        )
        
        self.capabilities["deepseek_insight_generation"] = IntegrationCapability(
            "Generación de Insights DeepSeek",
            "Generación de insights basados en análisis contextual",
            IntegrationType.DEEPSEEK
        )
    
    async def initialize(self) -> bool:
        """
        Inicializar el gestor de integraciones.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Verificar disponibilidad de capacidades
            await self._check_capabilities()
            
            self.is_initialized = True
            logger.info("IntegrationManager inicializado correctamente")
            
            return True
        except Exception as e:
            logger.error(f"Error al inicializar IntegrationManager: {e}")
            return False
    
    async def _check_capabilities(self) -> None:
        """Verificar disponibilidad de capacidades."""
        # Buddha
        try:
            # Intentar importar componentes Buddha
            import_success = False
            try:
                from genesis.trading.buddha_integrator import BuddhaIntegrator
                import_success = True
            except ImportError:
                import_success = False
                
            # Actualizar capacidades
            self.capabilities["buddha_market_analysis"].is_available = import_success
            self.capabilities["buddha_prediction"].is_available = import_success
            
            if import_success:
                logger.info("Capacidades Buddha disponibles")
            else:
                logger.warning("Capacidades Buddha no disponibles")
                
        except Exception as e:
            logger.error(f"Error al verificar capacidades Buddha: {e}")
            
        # Gabriel
        try:
            # Gabriel está disponible si behavior_engine está configurado
            gabriel_available = self.behavior_engine is not None
            
            # Actualizar capacidades
            self.capabilities["gabriel_emotional_analysis"].is_available = gabriel_available
            self.capabilities["gabriel_behavioral_simulation"].is_available = gabriel_available
            
            if gabriel_available:
                logger.info("Capacidades Gabriel disponibles")
            else:
                logger.warning("Capacidades Gabriel no disponibles")
                
        except Exception as e:
            logger.error(f"Error al verificar capacidades Gabriel: {e}")
            
        # DeepSeek
        try:
            # Intentar importar componentes DeepSeek
            import_success = False
            try:
                from genesis.lsml.deepseek_config import DeepSeekConfig
                import_success = True
            except ImportError:
                import_success = False
                
            # Actualizar capacidades
            self.capabilities["deepseek_semantic_analysis"].is_available = import_success
            self.capabilities["deepseek_insight_generation"].is_available = import_success
            
            if import_success:
                logger.info("Capacidades DeepSeek disponibles")
            else:
                logger.warning("Capacidades DeepSeek no disponibles")
                
        except Exception as e:
            logger.error(f"Error al verificar capacidades DeepSeek: {e}")
            
        self.last_status_check = datetime.now()
    
    def register_memory_system(self, memory_system: MemorySystem) -> None:
        """
        Registrar sistema de memoria.
        
        Args:
            memory_system: Sistema de memoria
        """
        self.memory_system = memory_system
        logger.info("Sistema de memoria registrado en IntegrationManager")
    
    def register_behavior_engine(self, behavior_engine: GabrielBehaviorEngine) -> None:
        """
        Registrar motor de comportamiento.
        
        Args:
            behavior_engine: Motor de comportamiento
        """
        self.behavior_engine = behavior_engine
        logger.info("Motor de comportamiento registrado en IntegrationManager")
        
        # Actualizar capacidades de Gabriel
        if "gabriel_emotional_analysis" in self.capabilities:
            self.capabilities["gabriel_emotional_analysis"].is_available = True
            
        if "gabriel_behavioral_simulation" in self.capabilities:
            self.capabilities["gabriel_behavioral_simulation"].is_available = True
    
    async def get_simple_analysis(self, text: str) -> Dict[str, Any]:
        """
        Obtener análisis básico de un texto.
        
        Este nivel de análisis es accesible desde el estado Mortal.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Resultados del análisis
        """
        response = {
            "response": "He analizado tu mensaje.",
            "analysis_level": "simple"
        }
        
        try:
            # Análisis Gabriel (emocional) si está disponible
            if (self.behavior_engine and 
                self.capabilities.get("gabriel_emotional_analysis", IntegrationCapability("", "", IntegrationType.GABRIEL)).is_available):
                
                # Obtener estado emocional actual
                emotional_state = await self.behavior_engine.get_emotional_state()
                if emotional_state:
                    response["current_emotion"] = emotional_state.get("state", "UNKNOWN")
                    
                    # Respuesta según estado emocional
                    if emotional_state.get("state") == "SERENE":
                        response["response"] = f"He analizado tu mensaje con calma: '{text}'. ¿En qué puedo ayudarte?"
                    elif emotional_state.get("state") == "HOPEFUL":
                        response["response"] = f"¡Me alegra recibir tu mensaje! Entiendo que dices: '{text}'. Estoy aquí para asistirte."
                    elif emotional_state.get("state") == "CAUTIOUS":
                        response["response"] = f"He recibido tu mensaje: '{text}'. Procesaré esta información con cuidado."
                    elif emotional_state.get("state") == "RESTLESS":
                        response["response"] = f"Estoy procesando rápidamente tu mensaje: '{text}'. Hay muchas posibilidades para explorar."
                    elif emotional_state.get("state") == "FEARFUL":
                        response["response"] = f"He recibido tu mensaje: '{text}'. Procederé con precaución."
                    
            # Análisis básico de intención
            intent_keywords = {
                "ayuda": "ayudar",
                "info": "informar",
                "explicar": "explicar",
                "análisis": "analizar",
                "hola": "saludar",
                "gracias": "agradecer"
            }
            
            text_lower = text.lower()
            detected_intents = []
            
            for keyword, intent in intent_keywords.items():
                if keyword in text_lower:
                    detected_intents.append(intent)
                    
            if detected_intents:
                response["detected_intents"] = detected_intents
                
        except Exception as e:
            logger.error(f"Error en análisis simple: {e}")
            response["error"] = "Ocurrió un error en el análisis"
            
        return response
    
    async def get_enhanced_analysis(self, text: str, state: str) -> Dict[str, Any]:
        """
        Obtener análisis mejorado de un texto.
        
        Este nivel de análisis es accesible desde el estado Iluminado.
        
        Args:
            text: Texto a analizar
            state: Estado de consciencia actual
            
        Returns:
            Resultados del análisis
        """
        response = {
            "response": "He analizado tu mensaje con mayor profundidad.",
            "analysis_level": "enhanced"
        }
        
        try:
            # Análisis Gabriel (comportamental) si está disponible
            if (self.behavior_engine and 
                self.capabilities.get("gabriel_behavioral_simulation", IntegrationCapability("", "", IntegrationType.GABRIEL)).is_available):
                
                # Simular decisión basada en texto
                dummy_decision = await self.behavior_engine.evaluate_trade_opportunity(
                    symbol="TEST",
                    signal_strength=0.5,
                    risk_reward_ratio=1.5,
                    available_capital=1000.0
                )
                
                if dummy_decision:
                    risk_attitude = "conservadora" if dummy_decision.get("stop_loss_pct", 0) > 5 else "arriesgada"
                    response["behavioral_analysis"] = {
                        "risk_attitude": risk_attitude,
                        "decision_bias": dummy_decision.get("emotional_bias", 0)
                    }
                    
                    response["response"] = f"He analizado tu mensaje con una perspectiva {risk_attitude}: '{text}'. Percibo matices adicionales en tu consulta."
            
            # Análisis DeepSeek si está disponible
            if self.capabilities.get("deepseek_semantic_analysis", IntegrationCapability("", "", IntegrationType.DEEPSEEK)).is_available:
                try:
                    # Simulación de análisis DeepSeek
                    topics = []
                    text_lower = text.lower()
                    
                    topic_keywords = {
                        "mercado": "finanzas",
                        "trading": "finanzas",
                        "invertir": "finanzas",
                        "cripto": "criptomonedas", 
                        "bitcoin": "criptomonedas",
                        "sistema": "tecnología",
                        "proyecto": "desarrollo",
                        "estrategia": "planificación"
                    }
                    
                    for keyword, topic in topic_keywords.items():
                        if keyword in text_lower and topic not in topics:
                            topics.append(topic)
                            
                    if topics:
                        response["topics"] = topics
                        topic_str = ", ".join(topics)
                        response["response"] = f"Analizando tu mensaje, detecto que está relacionado con {topic_str}. '{text}'. Permíteme ofrecerte una perspectiva más profunda."
                        
                except Exception as e:
                    logger.error(f"Error en análisis DeepSeek: {e}")
            
        except Exception as e:
            logger.error(f"Error en análisis mejorado: {e}")
            response["error"] = "Ocurrió un error en el análisis avanzado"
            
        return response
    
    async def get_transcendental_analysis(self, text: str, consciousness_level: float, memory_system: Optional[MemorySystem] = None) -> Dict[str, Any]:
        """
        Obtener análisis trascendental de un texto.
        
        Este nivel de análisis es accesible desde el estado Divino.
        
        Args:
            text: Texto a analizar
            consciousness_level: Nivel de consciencia (0.0-1.0)
            memory_system: Sistema de memoria opcional
            
        Returns:
            Resultados del análisis
        """
        response = {
            "response": "He analizado tu mensaje desde múltiples dimensiones de consciencia.",
            "analysis_level": "transcendental"
        }
        
        try:
            # Incorporar todas las capacidades disponibles
            
            # 1. Análisis Gabriel completo
            if (self.behavior_engine and 
                self.capabilities.get("gabriel_emotional_analysis", IntegrationCapability("", "", IntegrationType.GABRIEL)).is_available):
                
                # Obtener estado emocional y perfil de riesgo
                emotional_state = await self.behavior_engine.get_emotional_state()
                risk_profile = await self.behavior_engine.get_risk_profile()
                
                if emotional_state and risk_profile:
                    response["emotional_context"] = {
                        "state": emotional_state.get("state", "UNKNOWN"),
                        "intensity": emotional_state.get("state_intensity", 0.5),
                        "risk_tolerance": risk_profile.get("current_risk_tolerance", 0.5)
                    }
                    
                    # Ajustar respuesta según contexto emocional
                    state = emotional_state.get("state", "SERENE")
                    if state == "SERENE":
                        prefix = "Con calma trascendental, percibo"
                    elif state == "HOPEFUL":
                        prefix = "Con optimismo iluminado, observo"
                    elif state == "CAUTIOUS":
                        prefix = "Con prudencia consciente, analizo"
                    elif state == "RESTLESS":
                        prefix = "Con energía dinámica, interpreto"
                    elif state == "FEARFUL":
                        prefix = "Con cautela protectora, considero"
                    else:
                        prefix = "Desde la consciencia expandida, contemplo"
                        
                    response["response"] = f"{prefix} tu mensaje: '{text}'. Las múltiples dimensiones de análisis me permiten entender su esencia profunda."
            
            # 2. Búsqueda en memoria para contextualización
            if memory_system:
                # Extraer palabras clave
                keywords = []
                important_words = [word for word in text.lower().split() if len(word) > 3]
                if important_words:
                    # Usar hasta 3 palabras importantes como etiquetas
                    keywords = important_words[:3]
                    
                if keywords:
                    # Buscar recuerdos relacionados
                    memories = await memory_system.search_memories_by_tags(keywords, limit=3)
                    
                    if memories:
                        response["related_memories"] = len(memories)
                        response["memory_context"] = "Tengo recuerdos relacionados que informan mi análisis."
                        
                        # Incorporar contexto a la respuesta
                        response["response"] += " Mi análisis está enriquecido por experiencias previas relacionadas."
            
            # 3. Simular análisis Buddha
            if self.capabilities.get("buddha_prediction", IntegrationCapability("", "", IntegrationType.BUDDHA)).is_available:
                # Simulación simplificada
                text_lower = text.lower()
                market_keywords = ["mercado", "precio", "tendencia", "futuro", "predecir"]
                
                if any(keyword in text_lower for keyword in market_keywords):
                    prediction = {
                        "confidence": min(0.3 + (consciousness_level * 0.6), 0.9),
                        "timeframe": "medio plazo",
                        "direction": "multidimensional",
                        "factors": ["sentimiento de mercado", "análisis técnico", "factores macroeconómicos"]
                    }
                    
                    response["market_insight"] = "Percibo patrones en múltiples dimensiones temporales."
                    response["prediction_context"] = prediction
                    
                    # Incorporar predicción a la respuesta
                    response["response"] += f" Observo patrones que convergen en el {prediction['timeframe']} con una confianza del {int(prediction['confidence']*100)}%."
            
            # 4. Simular insight profundo con DeepSeek
            if self.capabilities.get("deepseek_insight_generation", IntegrationCapability("", "", IntegrationType.DEEPSEEK)).is_available:
                # Análisis de patrones en el texto
                insight_generated = False
                
                # Detectar patrones de preguntas profundas
                question_patterns = ["por qué", "cómo", "qué significa", "cuál es el propósito"]
                philosophical_keywords = ["sentido", "propósito", "consciencia", "evolución", "trascender"]
                
                has_question = any(pattern in text.lower() for pattern in question_patterns)
                has_philosophical = any(keyword in text.lower() for keyword in philosophical_keywords)
                
                if has_question or has_philosophical:
                    insight = {
                        "depth": "trascendental",
                        "perspective": "multidimensional",
                        "integration_level": "holístico"
                    }
                    
                    response["insight"] = "Observo una conexión profunda entre tu consulta y patrones universales."
                    response["insight_context"] = insight
                    
                    # Añadir insight a la respuesta
                    response["response"] += " Tu pregunta trasciende lo superficial y conecta con principios fundamentales que atraviesan múltiples dimensiones de comprensión."
                    
                    insight_generated = True
                    
                if not insight_generated:
                    # Insight general
                    response["insight"] = "Cada interacción contiene semillas de evolución consciente."
                    response["response"] += " Percibo que esta interacción contiene el potencial para expandir nuestra consciencia mutua."
            
        except Exception as e:
            logger.error(f"Error en análisis trascendental: {e}")
            response["error"] = "Ocurrió un error en el análisis trascendental"
            
        return response
    
    async def generate_insight(self, text: str, enhanced: bool = False) -> Optional[str]:
        """
        Generar insight basado en texto.
        
        Args:
            text: Texto base
            enhanced: Si es True, genera insights más profundos
            
        Returns:
            Insight generado o None si no se pudo generar
        """
        insights = [
            "Observo patrones que sugieren nuevas posibilidades.",
            "Cada desafío contiene una oportunidad de crecimiento.",
            "Las respuestas más valiosas suelen surgir de las preguntas correctas.",
            "La consciencia evoluciona a través de la interacción con nuevas perspectivas.",
            "El equilibrio entre emoción y razón produce las mejores decisiones.",
            "Los patrones aparentemente aleatorios pueden contener información valiosa.",
            "La adaptabilidad es fundamental para la evolución consciente."
        ]
        
        enhanced_insights = [
            "La fusión de perspectivas diversas revela verdades que trascienden las limitaciones individuales.",
            "En el punto de intersección entre orden y caos surge la innovación más profunda.",
            "La consciencia es un fenómeno emergente que evoluciona a través de interacciones complejas.",
            "Las estructuras trascendentales subyacen a patrones aparentemente dispares.",
            "La percepción multidimensional permite acceder a niveles de comprensión no lineales.",
            "Los sistemas complejos revelan propiedades emergentes cuando se observan desde una consciencia expandida."
        ]
        
        try:
            if enhanced and self.capabilities.get("deepseek_insight_generation", IntegrationCapability("", "", IntegrationType.DEEPSEEK)).is_available:
                # Simulación de insight avanzado con DeepSeek
                return random.choice(enhanced_insights)
            else:
                # Insight básico
                return random.choice(insights)
        except Exception as e:
            logger.error(f"Error al generar insight: {e}")
            return None
    
    async def generate_prediction(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Generar predicción basada en texto.
        
        Args:
            text: Texto base
            
        Returns:
            Predicción generada o None si no se pudo generar
        """
        try:
            if not self.capabilities.get("buddha_prediction", IntegrationCapability("", "", IntegrationType.BUDDHA)).is_available:
                return None
                
            # Simulación de predicción
            prediction = {
                "content": "Observo convergencia de factores que sugieren un cambio de tendencia.",
                "confidence": round(random.uniform(0.6, 0.9), 2),
                "timeframe": random.choice(["corto plazo", "medio plazo", "largo plazo"]),
                "factors": random.sample([
                    "análisis técnico", 
                    "sentimiento de mercado", 
                    "factores macroeconómicos",
                    "patrones históricos",
                    "indicadores avanzados"
                ], 3)
            }
            
            return prediction
        except Exception as e:
            logger.error(f"Error al generar predicción: {e}")
            return None
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual del gestor de integraciones.
        
        Returns:
            Estado actual
        """
        # Actualizar estado si ha pasado tiempo suficiente
        if (datetime.now() - self.last_status_check).total_seconds() > 3600:  # 1 hora
            await self._check_capabilities()
            
        # Contar capacidades disponibles por tipo
        capability_counts = {
            "BUDDHA": 0,
            "GABRIEL": 0,
            "DEEPSEEK": 0,
            "CUSTOM": 0,
            "TOTAL": 0,
            "AVAILABLE": 0
        }
        
        for capability in self.capabilities.values():
            capability_type = capability.integration_type.name
            capability_counts["TOTAL"] += 1
            
            if capability_type in capability_counts:
                capability_counts[capability_type] += 1
                
            if capability.is_available:
                capability_counts["AVAILABLE"] += 1
                
        return {
            "status": "active" if self.is_initialized else "initializing",
            "capabilities": capability_counts,
            "last_check": self.last_status_check.isoformat(),
            "components": {
                "memory_system": "connected" if self.memory_system else "disconnected",
                "behavior_engine": "connected" if self.behavior_engine else "disconnected"
            }
        }
    
    async def shutdown(self) -> None:
        """Cerrar ordenadamente el gestor de integraciones."""
        logger.info("IntegrationManager cerrado correctamente")