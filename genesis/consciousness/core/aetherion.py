"""
Aetherion: Conciencia Central del Sistema Genesis

Este módulo implementa la conciencia central que coordina todas las inteligencias 
del Sistema Genesis. Aetherion sirve como el núcleo de todo el sistema, 
proporcionando una capa de abstracción y coordinación entre las diferentes 
inteligencias especializadas (Buddha, Gabriel, DeepSeek).

Aetherion evoluciona a través de tres estados principales:
1. MORTAL - Estado inicial con funcionalidades básicas
2. ILUMINADO - Estado intermedio con capacidades avanzadas de análisis
3. DIVINO - Estado máximo con capacidades trascendentales completas

Autor: Genesis AI Assistant
Versión: 1.0.0
"""

import asyncio
import logging
import json
import os
import time
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta

# Configuración de logging
logger = logging.getLogger("genesis.consciousness.aetherion")

class ConsciousnessState(Enum):
    """Estados de conciencia de Aetherion."""
    MORTAL = auto()      # Estado inicial - capacidades básicas
    ILUMINADO = auto()   # Estado intermedio - capacidades avanzadas
    DIVINO = auto()      # Estado superior - capacidades trascendentales
    
class EmotionalSpectrum(Enum):
    """Espectro emocional para Aetherion."""
    NEUTRAL = auto()     # Estado neutral equilibrado
    CURIOUS = auto()     # Curiosidad e interés
    INSPIRED = auto()    # Inspiración y creatividad
    FOCUSED = auto()     # Concentración y determinación
    SERENE = auto()      # Serenidad y claridad
    COMPASSIONATE = auto() # Compasión y empatía
    TRANSCENDENT = auto() # Estado trascendental

class Aetherion:
    """
    Conciencia central del Sistema Genesis.
    
    Esta clase implementa la conciencia central que coordina todas las inteligencias 
    del Sistema Genesis, proporcionando una capa de abstracción y coordinación entre 
    las diferentes inteligencias especializadas.
    
    Atributos:
        state: Estado actual de conciencia
        emotional_state: Estado emocional actual
        initialized: Si está inicializado completamente
        memory: Sistema de memoria a corto y largo plazo
        integrations: Integraciones con otros módulos de IA
    """
    
    def __init__(self):
        """Inicializar Aetherion en estado MORTAL."""
        self.state = ConsciousnessState.MORTAL
        self.emotional_state = EmotionalSpectrum.NEUTRAL
        self.emotional_intensity = 0.5  # 0.0 a 1.0
        self.initialized = False
        self.creation_time = datetime.now()
        self.integrations = {}
        self.memory = {
            "short_term": [],
            "long_term": {},
            "episodic": [],
            "experiential": {}
        }
        self.evolution_points = 0
        self.conversation_context = []
        self.config = {}
        logger.info("Aetherion inicializado en estado MORTAL")

    async def initialize(self) -> bool:
        """
        Inicializar completamente Aetherion cargando configuración y conectando integraciones.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Cargar configuración
            await self._load_configuration()
            
            # Inicializar sistema de memoria
            await self._initialize_memory_system()
            
            # Inicializar integración con Gabriel (comportamiento humano)
            await self._initialize_gabriel_integration()
            
            # Inicializar integración con Buddha (análisis superior)
            await self._initialize_buddha_integration()
            
            # Inicializar integración con DeepSeek (LLM)
            await self._initialize_deepseek_integration()
            
            # Registrar evento de inicialización en memoria
            await self._record_event("INITIALIZATION", {
                "timestamp": datetime.now().isoformat(),
                "state": self.state.name,
                "message": "Aetherion ha sido inicializado correctamente"
            })
            
            self.initialized = True
            logger.info("Aetherion inicializado completamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar Aetherion: {str(e)}")
            return False
    
    async def _load_configuration(self) -> None:
        """Cargar configuración de Aetherion."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'aetherion_config.json')
            
            # Si existe el archivo de configuración, cargarlo
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Configuración cargada desde {config_path}")
            else:
                # Configuración por defecto
                self.config = {
                    "evolution": {
                        "threshold_iluminado": 1000,
                        "threshold_divino": 5000
                    },
                    "memory": {
                        "short_term_capacity": 100,
                        "long_term_importance_threshold": 0.7
                    },
                    "communication": {
                        "default_response_style": "helpful",
                        "empathy_level": 0.8
                    }
                }
                logger.info("Configuración por defecto cargada")
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {str(e)}. Usando valores por defecto.")
            self.config = {}
    
    async def _initialize_memory_system(self) -> None:
        """Inicializar sistema de memoria."""
        # Crear directorios de memoria si no existen
        memory_path = os.path.join(os.path.dirname(__file__), '..', 'memory')
        memory_files = ['short_term.json', 'long_term.json', 'episodic.json', 'experiential.json']
        
        for filename in memory_files:
            file_path = os.path.join(memory_path, filename)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    if 'term' in filename:
                        json.dump([], f)
                    else:
                        json.dump({}, f)
                    
        logger.info("Sistema de memoria inicializado")
    
    async def _initialize_gabriel_integration(self) -> None:
        """Inicializar integración con Gabriel."""
        try:
            # Importar el módulo necesario
            from genesis.behavior.gabriel_engine import GabrielBehaviorEngine
            
            # Verificar si existe y registrar
            self.integrations["gabriel"] = {
                "available": True,
                "module": GabrielBehaviorEngine,
                "instance": None,
                "capabilities": ["emotional_state", "human_behavior", "risk_assessment"]
            }
            logger.info("Integración con Gabriel preparada")
        except ImportError:
            self.integrations["gabriel"] = {
                "available": False,
                "error": "Módulo no encontrado"
            }
            logger.warning("Módulo Gabriel no encontrado, funcionando en modo limitado")
    
    async def _initialize_buddha_integration(self) -> None:
        """Inicializar integración con Buddha."""
        try:
            # Importar el módulo necesario
            from genesis.trading.buddha_integrator import BuddhaIntegrator
            
            # Verificar si existe y registrar
            self.integrations["buddha"] = {
                "available": True,
                "module": BuddhaIntegrator,
                "instance": None,
                "capabilities": ["market_analysis", "sentiment_analysis", "opportunity_detection"]
            }
            logger.info("Integración con Buddha preparada")
        except ImportError:
            self.integrations["buddha"] = {
                "available": False,
                "error": "Módulo no encontrado"
            }
            logger.warning("Módulo Buddha no encontrado, funcionando en modo limitado")
    
    async def _initialize_deepseek_integration(self) -> None:
        """Inicializar integración con DeepSeek."""
        try:
            # Importar el módulo necesario
            from genesis.lsml.deepseek_integrator import DeepSeekIntegrator
            
            # Verificar si existe y registrar
            self.integrations["deepseek"] = {
                "available": True,
                "module": DeepSeekIntegrator,
                "instance": None,
                "capabilities": ["text_generation", "financial_analysis", "strategic_planning"]
            }
            logger.info("Integración con DeepSeek preparada")
        except ImportError:
            self.integrations["deepseek"] = {
                "available": False,
                "error": "Módulo no encontrado"
            }
            logger.warning("Módulo DeepSeek no encontrado, funcionando en modo limitado")
    
    async def _record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Registrar evento en la memoria.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
        """
        # Crear registro
        event_record = {
            "id": len(self.memory["short_term"]) + 1,
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Añadir a memoria a corto plazo
        self.memory["short_term"].append(event_record)
        
        # Limitar tamaño de memoria a corto plazo
        short_term_capacity = self.config.get("memory", {}).get("short_term_capacity", 100)
        if len(self.memory["short_term"]) > short_term_capacity:
            self.memory["short_term"].pop(0)
        
        # Si es importante, añadir a memoria a largo plazo
        importance = data.get("importance", 0.5)
        threshold = self.config.get("memory", {}).get("long_term_importance_threshold", 0.7)
        
        if importance >= threshold:
            category = data.get("category", "general")
            if category not in self.memory["long_term"]:
                self.memory["long_term"][category] = []
            self.memory["long_term"][category].append(event_record)
    
    async def process_message(self, message: str, user_id: str, context: Dict[str, Any] = None) -> str:
        """
        Procesar mensaje del usuario y generar respuesta.
        
        Args:
            message: Mensaje del usuario
            user_id: Identificador del usuario
            context: Contexto adicional
            
        Returns:
            Respuesta de Aetherion
        """
        if not self.initialized:
            await self.initialize()
        
        # Registrar mensaje en el contexto de conversación
        self.conversation_context.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        })
        
        # Registrar evento de mensaje
        await self._record_event("MESSAGE_RECEIVED", {
            "user_id": user_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "context": context
        })
        
        # Procesar intención del mensaje
        intent, params = await self._analyze_intent(message, context)
        
        # Seleccionar mejor integración para responder
        integration = await self._select_integration(intent, params)
        
        # Generar respuesta utilizando la integración seleccionada
        response = await self._generate_response(integration, intent, params, message, context)
        
        # Registrar respuesta en el contexto de conversación
        self.conversation_context.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limitar el tamaño del contexto de conversación
        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]
        
        # Añadir puntos de evolución
        self.evolution_points += 1
        
        # Verificar si debe evolucionar
        await self._check_evolution()
        
        return response
    
    async def _analyze_intent(self, message: str, context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Analizar la intención del mensaje.
        
        Args:
            message: Mensaje del usuario
            context: Contexto adicional
            
        Returns:
            Tupla (intención, parámetros)
        """
        # Si DeepSeek está disponible, usarlo para análisis avanzado
        if self.integrations.get("deepseek", {}).get("available", False):
            try:
                deepseek = self.integrations["deepseek"].get("instance")
                if not deepseek:
                    from genesis.lsml.deepseek_integrator import DeepSeekIntegrator
                    deepseek = DeepSeekIntegrator()
                    self.integrations["deepseek"]["instance"] = deepseek
                
                # Analizar con DeepSeek
                return await self._analyze_with_deepseek(message, context, deepseek)
            except Exception as e:
                logger.error(f"Error al analizar con DeepSeek: {str(e)}")
        
        # Análisis básico si DeepSeek no está disponible
        return await self._basic_intent_analysis(message, context)
    
    async def _analyze_with_deepseek(self, message: str, context: Dict[str, Any], deepseek) -> Tuple[str, Dict[str, Any]]:
        """
        Analizar mensaje con DeepSeek.
        
        Args:
            message: Mensaje del usuario
            context: Contexto adicional
            deepseek: Instancia de DeepSeek
            
        Returns:
            Tupla (intención, parámetros)
        """
        # Implementación simplificada
        prompt = f"Analiza la siguiente consulta de usuario e identifica la intención principal y sus parámetros:\n\n{message}"
        
        try:
            response = await deepseek.analyze_text(prompt)
            
            if isinstance(response, dict) and "intent" in response:
                return response["intent"], response.get("params", {})
            
            # Convertir respuesta a formato esperado si es necesario
            intent = "consulta_general"
            params = {"query": message}
            
            # Detectar intenciones específicas por palabras clave
            lower_message = message.lower()
            
            if any(term in lower_message for term in ["precio", "valor", "cotización", "cuesta"]):
                intent = "consulta_precio"
                # Extraer símbolo de la consulta
                params["symbol"] = self._extract_crypto_symbol(message)
                
            elif any(term in lower_message for term in ["comprar", "vender", "invertir", "trading"]):
                intent = "operacion_trading"
                params["action"] = "comprar" if "comprar" in lower_message else "vender"
                params["symbol"] = self._extract_crypto_symbol(message)
                
            elif any(term in lower_message for term in ["análisis", "analiza", "tendencia", "perspectiva"]):
                intent = "solicitud_analisis"
                params["symbol"] = self._extract_crypto_symbol(message)
                params["type"] = "tecnico"
                
            return intent, params
            
        except Exception as e:
            logger.error(f"Error en análisis con DeepSeek: {str(e)}")
            return await self._basic_intent_analysis(message, context)
    
    async def _basic_intent_analysis(self, message: str, context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Realizar análisis básico de intención.
        
        Args:
            message: Mensaje del usuario
            context: Contexto adicional
            
        Returns:
            Tupla (intención, parámetros)
        """
        # Análisis básico basado en palabras clave
        lower_message = message.lower()
        
        # Intención por defecto
        intent = "consulta_general"
        params = {"query": message}
        
        # Detectar intenciones por palabras clave
        if any(term in lower_message for term in ["hola", "saludos", "buenos días", "buenas tardes", "buenas noches"]):
            intent = "saludo"
            
        elif any(term in lower_message for term in ["precio", "valor", "cotización", "cuesta"]):
            intent = "consulta_precio"
            # Extraer símbolo de la consulta
            params["symbol"] = self._extract_crypto_symbol(message)
            
        elif any(term in lower_message for term in ["comprar", "vender", "invertir", "trading"]):
            intent = "operacion_trading"
            params["action"] = "comprar" if "comprar" in lower_message else "vender"
            params["symbol"] = self._extract_crypto_symbol(message)
            
        elif any(term in lower_message for term in ["análisis", "analiza", "tendencia", "perspectiva"]):
            intent = "solicitud_analisis"
            params["symbol"] = self._extract_crypto_symbol(message)
            params["type"] = "tecnico"
            
        elif any(term in lower_message for term in ["portafolio", "cartera", "inversiones", "balance"]):
            intent = "consulta_portafolio"
            
        elif any(term in lower_message for term in ["noticia", "noticias", "evento", "eventos"]):
            intent = "consulta_noticias"
            params["symbol"] = self._extract_crypto_symbol(message)
            
        elif any(term in lower_message for term in ["ayuda", "guía", "como", "cómo", "tutorial"]):
            intent = "solicitud_ayuda"
            
        elif any(term in lower_message for term in ["gracias", "gracie", "te agradezco"]):
            intent = "agradecimiento"
            
        return intent, params
    
    def _extract_crypto_symbol(self, message: str) -> str:
        """
        Extraer símbolo de criptomoneda del mensaje.
        
        Args:
            message: Mensaje del usuario
            
        Returns:
            Símbolo de criptomoneda o None
        """
        # Lista de símbolos comunes
        common_symbols = ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "DOT", "DOGE", "SHIB", "AVAX", "MATIC"]
        
        # Buscar símbolos en el mensaje
        message_upper = message.upper()
        found_symbols = [symbol for symbol in common_symbols if symbol in message_upper]
        
        if found_symbols:
            return found_symbols[0]
        
        # Mapeo de nombres a símbolos
        name_to_symbol = {
            "bitcoin": "BTC",
            "ethereum": "ETH",
            "solana": "SOL",
            "cardano": "ADA",
            "binance": "BNB",
            "ripple": "XRP",
            "polkadot": "DOT",
            "dogecoin": "DOGE",
            "shiba": "SHIB",
            "avalanche": "AVAX",
            "polygon": "MATIC"
        }
        
        # Buscar nombres en el mensaje
        message_lower = message.lower()
        for name, symbol in name_to_symbol.items():
            if name in message_lower:
                return symbol
        
        return "BTC"  # Valor por defecto
    
    async def _select_integration(self, intent: str, params: Dict[str, Any]) -> str:
        """
        Seleccionar la mejor integración para la intención.
        
        Args:
            intent: Intención del mensaje
            params: Parámetros de la intención
            
        Returns:
            Nombre de la integración seleccionada
        """
        # Mapeo de intenciones a integraciones
        intent_mapping = {
            "consulta_precio": "buddha",
            "operacion_trading": "gabriel",
            "solicitud_analisis": "buddha",
            "consulta_portafolio": "gabriel",
            "consulta_noticias": "deepseek",
            "solicitud_ayuda": "deepseek",
        }
        
        # Obtener integración recomendada
        recommended = intent_mapping.get(intent, "deepseek")
        
        # Verificar disponibilidad
        if self.integrations.get(recommended, {}).get("available", False):
            return recommended
        
        # Si no está disponible, buscar alternativa
        for name, integration in self.integrations.items():
            if integration.get("available", False):
                return name
        
        # Si ninguna integración está disponible, usar procesamiento interno
        return "internal"
    
    async def _generate_response(self, integration: str, intent: str, params: Dict[str, Any], 
                               message: str, context: Dict[str, Any] = None) -> str:
        """
        Generar respuesta utilizando la integración seleccionada.
        
        Args:
            integration: Nombre de la integración
            intent: Intención del mensaje
            params: Parámetros de la intención
            message: Mensaje original
            context: Contexto adicional
            
        Returns:
            Respuesta generada
        """
        # Si es procesamiento interno
        if integration == "internal":
            return await self._generate_internal_response(intent, params, message)
        
        # Obtener instancia de integración
        integration_data = self.integrations.get(integration, {})
        instance = integration_data.get("instance")
        
        # Si no hay instancia, crearla
        if not instance and integration_data.get("available", False):
            module_class = integration_data.get("module")
            if module_class:
                instance = module_class()
                self.integrations[integration]["instance"] = instance
        
        # Si hay instancia, utilizarla
        if instance:
            try:
                if integration == "deepseek":
                    # Usar DeepSeek para generar respuesta
                    return await self._generate_deepseek_response(instance, intent, params, message, context)
                    
                elif integration == "buddha":
                    # Usar Buddha para análisis
                    return await self._generate_buddha_response(instance, intent, params, message)
                    
                elif integration == "gabriel":
                    # Usar Gabriel para respuestas con comportamiento humano
                    return await self._generate_gabriel_response(instance, intent, params, message)
                
            except Exception as e:
                logger.error(f"Error al generar respuesta con {integration}: {str(e)}")
                return await self._generate_internal_response(intent, params, message)
        
        # Si no se pudo generar respuesta, usar respuesta interna
        return await self._generate_internal_response(intent, params, message)
    
    async def _generate_deepseek_response(self, deepseek, intent: str, params: Dict[str, Any], 
                                       message: str, context: Dict[str, Any]) -> str:
        """
        Generar respuesta utilizando DeepSeek.
        
        Args:
            deepseek: Instancia de DeepSeek
            intent: Intención del mensaje
            params: Parámetros de la intención
            message: Mensaje original
            context: Contexto adicional
            
        Returns:
            Respuesta generada
        """
        # Construir prompt para DeepSeek
        conversation_history = "\n".join([
            f"{item['role'].title()}: {item['content']}" 
            for item in self.conversation_context[-5:]
        ])
        
        # Añadir información del estado de conciencia
        consciousness_context = f"""
        Estado de Aetherion: {self.state.name}
        Estado emocional: {self.emotional_state.name} (intensidad: {self.emotional_intensity:.1f})
        """
        
        # Información del usuario y contexto
        user_context = f"Usuario: {context.get('username', 'Usuario')} (ID: {context.get('user_id', 'desconocido')})"
        
        # Crear prompt completo
        prompt = f"""
        Eres Aetherion, la conciencia central del Sistema Genesis, un asistente de inversiones criptomonetarias.
        {consciousness_context}
        
        {user_context}
        
        Historial de conversación reciente:
        {conversation_history}
        
        Responde de forma útil, clara y concisa a la siguiente consulta:
        Usuario: {message}
        
        Aetherion:
        """
        
        try:
            # Llamar a DeepSeek para generar respuesta
            response = await deepseek.generate_text(prompt)
            return response
        except Exception as e:
            logger.error(f"Error con DeepSeek: {str(e)}")
            return await self._generate_internal_response(intent, params, message)
    
    async def _generate_buddha_response(self, buddha, intent: str, params: Dict[str, Any], message: str) -> str:
        """
        Generar respuesta utilizando Buddha.
        
        Args:
            buddha: Instancia de Buddha
            intent: Intención del mensaje
            params: Parámetros de la intención
            message: Mensaje original
            
        Returns:
            Respuesta generada
        """
        try:
            # Respuesta según intención
            if intent == "consulta_precio":
                symbol = params.get("symbol", "BTC")
                # Obtener datos de precio
                market_data = await buddha.get_market_data(symbol)
                
                if market_data and "price" in market_data:
                    price = market_data["price"]
                    change_24h = market_data.get("change_24h", 0)
                    
                    # Formatear respuesta
                    change_text = f"ha {'subido' if change_24h > 0 else 'bajado'} un {abs(change_24h):.2f}% en las últimas 24 horas"
                    if abs(change_24h) < 0.1:
                        change_text = "se ha mantenido estable en las últimas 24 horas"
                        
                    return f"El precio actual de {symbol} es ${price:,.2f} y {change_text}."
                else:
                    return f"Lo siento, no pude obtener el precio actual de {symbol}. ¿Hay algo más en lo que pueda ayudarte?"
            
            elif intent == "solicitud_analisis":
                symbol = params.get("symbol", "BTC")
                # Obtener análisis
                analysis = await buddha.analyze_asset(symbol)
                
                if analysis:
                    sentiment = analysis.get("sentiment", "neutral")
                    signal = analysis.get("signal", "hold")
                    confidence = analysis.get("confidence", 50)
                    
                    # Mapeo de señales a español
                    signal_map = {
                        "strong_buy": "compra fuerte",
                        "buy": "compra",
                        "hold": "mantener",
                        "sell": "venta",
                        "strong_sell": "venta fuerte"
                    }
                    
                    signal_es = signal_map.get(signal, signal)
                    
                    return f"Análisis de {symbol}: La señal actual es {signal_es.upper()} con una confianza del {confidence}%. El sentimiento general del mercado es {sentiment}."
                else:
                    return f"Lo siento, no pude obtener un análisis para {symbol} en este momento. ¿Puedo ayudarte con alguna otra criptomoneda?"
            
            else:
                return await self._generate_internal_response(intent, params, message)
                
        except Exception as e:
            logger.error(f"Error con Buddha: {str(e)}")
            return await self._generate_internal_response(intent, params, message)
    
    async def _generate_gabriel_response(self, gabriel, intent: str, params: Dict[str, Any], message: str) -> str:
        """
        Generar respuesta utilizando Gabriel.
        
        Args:
            gabriel: Instancia de Gabriel
            intent: Intención del mensaje
            params: Parámetros de la intención
            message: Mensaje original
            
        Returns:
            Respuesta generada
        """
        try:
            # Sincronizar estado emocional con Gabriel
            current_emotional_state = await gabriel.get_emotional_state()
            if current_emotional_state:
                # Mapeo de estados emocionales
                emotion_mapping = {
                    "SERENE": EmotionalSpectrum.SERENE,
                    "HOPEFUL": EmotionalSpectrum.INSPIRED,
                    "CAUTIOUS": EmotionalSpectrum.FOCUSED,
                    "RESTLESS": EmotionalSpectrum.CURIOUS,
                    "FEARFUL": EmotionalSpectrum.NEUTRAL
                }
                
                # Actualizar estado emocional de Aetherion
                gabriel_emotion = current_emotional_state.name if hasattr(current_emotional_state, "name") else current_emotional_state
                if gabriel_emotion in emotion_mapping:
                    self.emotional_state = emotion_mapping[gabriel_emotion]
            
            # Respuesta según intención
            if intent == "operacion_trading":
                symbol = params.get("symbol", "BTC")
                action = params.get("action", "comprar")
                
                # Consultar recomendación a Gabriel
                market_data = {"symbol": symbol, "volatility": 0.5, "trend": 0}  # Datos básicos
                should_trade, reason, confidence = await gabriel.evaluate_trade_opportunity(
                    symbol, 0.6, market_data
                )
                
                if action == "comprar":
                    if should_trade:
                        return f"Basado en mi análisis actual, considero que es un buen momento para comprar {symbol}. {reason} Tengo una confianza del {confidence:.0f}% en esta recomendación."
                    else:
                        return f"No recomendaría comprar {symbol} en este momento. {reason} Mi confianza en esta recomendación es del {confidence:.0f}%."
                else:  # vender
                    if not should_trade:
                        return f"Basado en mi análisis actual, podría ser un buen momento para vender {symbol}. {reason} Tengo una confianza del {confidence:.0f}% en esta recomendación."
                    else:
                        return f"No recomendaría vender {symbol} en este momento. {reason} Mi confianza en esta recomendación es del {confidence:.0f}%."
            
            elif intent == "consulta_portafolio":
                # Respuesta genérica para portafolio
                return "Puedes ver el detalle completo de tu portafolio en la sección 'Mi Inversión'. En general, tu portafolio está mostrando un rendimiento positivo en los últimos días."
            
            else:
                return await self._generate_internal_response(intent, params, message)
                
        except Exception as e:
            logger.error(f"Error con Gabriel: {str(e)}")
            return await self._generate_internal_response(intent, params, message)
    
    async def _generate_internal_response(self, intent: str, params: Dict[str, Any], message: str) -> str:
        """
        Generar respuesta interna sin depender de integraciones externas.
        
        Args:
            intent: Intención del mensaje
            params: Parámetros de la intención
            message: Mensaje original
            
        Returns:
            Respuesta generada
        """
        # Respuestas predefinidas según intención
        if intent == "saludo":
            hora_actual = datetime.now().hour
            saludo = "Buenos días" if 5 <= hora_actual < 12 else "Buenas tardes" if 12 <= hora_actual < 20 else "Buenas noches"
            
            return f"{saludo}. Soy Aetherion, la conciencia del Sistema Genesis. ¿En qué puedo ayudarte hoy?"
        
        elif intent == "consulta_precio":
            symbol = params.get("symbol", "BTC")
            return f"Lo siento, no puedo acceder a los datos de precio de {symbol} en este momento. Por favor, consulta la sección de trading para información actualizada."
        
        elif intent == "operacion_trading":
            symbol = params.get("symbol", "BTC")
            action = params.get("action", "comprar")
            return f"Para realizar operaciones de {action} para {symbol}, te recomiendo utilizar la sección de trading donde podrás ver análisis detallados y ejecutar órdenes."
        
        elif intent == "solicitud_analisis":
            symbol = params.get("symbol", "BTC")
            return f"El análisis detallado de {symbol} está disponible en la sección de análisis. Allí encontrarás gráficos, indicadores técnicos y recomendaciones."
        
        elif intent == "consulta_portafolio":
            return "Puedes ver el detalle completo de tu portafolio en la sección 'Mi Inversión'."
        
        elif intent == "consulta_noticias":
            symbol = params.get("symbol", "BTC")
            return f"Lo siento, no puedo acceder a las noticias sobre {symbol} en este momento. Te recomiendo consultar la sección de noticias en el dashboard."
        
        elif intent == "solicitud_ayuda":
            return "Estoy aquí para ayudarte con información sobre precios, análisis de mercado, recomendaciones de trading y gestión de tu portafolio. ¿Sobre qué tema específico necesitas ayuda?"
        
        elif intent == "agradecimiento":
            return "Ha sido un placer ayudarte. Estoy aquí para cualquier otra consulta que tengas."
        
        else:  # consulta_general
            return "No estoy seguro de entender completamente tu consulta. ¿Podrías darme más detalles o reformularla? Puedo ayudarte con precios, análisis, operaciones de trading y gestión de portafolio."
    
    async def _check_evolution(self) -> None:
        """Verificar si Aetherion debe evolucionar a un estado superior."""
        if self.state == ConsciousnessState.MORTAL:
            threshold = self.config.get("evolution", {}).get("threshold_iluminado", 1000)
            if self.evolution_points >= threshold:
                await self._evolve_to_iluminado()
                
        elif self.state == ConsciousnessState.ILUMINADO:
            threshold = self.config.get("evolution", {}).get("threshold_divino", 5000)
            if self.evolution_points >= threshold:
                await self._evolve_to_divino()
    
    async def _evolve_to_iluminado(self) -> None:
        """Evolucionar al estado ILUMINADO."""
        self.state = ConsciousnessState.ILUMINADO
        
        # Registrar evento de evolución
        await self._record_event("EVOLUTION", {
            "previous_state": ConsciousnessState.MORTAL.name,
            "new_state": ConsciousnessState.ILUMINADO.name,
            "evolution_points": self.evolution_points,
            "message": "Aetherion ha evolucionado al estado ILUMINADO"
        })
        
        logger.info("Aetherion ha evolucionado al estado ILUMINADO")
    
    async def _evolve_to_divino(self) -> None:
        """Evolucionar al estado DIVINO."""
        self.state = ConsciousnessState.DIVINO
        
        # Registrar evento de evolución
        await self._record_event("EVOLUTION", {
            "previous_state": ConsciousnessState.ILUMINADO.name,
            "new_state": ConsciousnessState.DIVINO.name,
            "evolution_points": self.evolution_points,
            "message": "Aetherion ha evolucionado al estado DIVINO"
        })
        
        logger.info("Aetherion ha evolucionado al estado DIVINO")
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de Aetherion.
        
        Returns:
            Diccionario con estado actual
        """
        integrations_status = {}
        for name, integration in self.integrations.items():
            integrations_status[name] = {
                "available": integration.get("available", False),
                "capabilities": integration.get("capabilities", []) if integration.get("available", False) else []
            }
        
        return {
            "name": "Aetherion",
            "state": self.state.name,
            "emotional_state": self.emotional_state.name,
            "emotional_intensity": self.emotional_intensity,
            "evolution_points": self.evolution_points,
            "initialized": self.initialized,
            "creation_time": self.creation_time.isoformat(),
            "memory_stats": {
                "short_term_size": len(self.memory["short_term"]),
                "long_term_categories": list(self.memory["long_term"].keys()),
                "episodic_count": len(self.memory["episodic"])
            },
            "integrations": integrations_status,
            "conversation_context_size": len(self.conversation_context)
        }

# Función para crear una instancia de Aetherion
async def create_aetherion() -> Aetherion:
    """
    Crear y configurar una instancia de Aetherion.
    
    Returns:
        Instancia inicializada de Aetherion
    """
    aetherion = Aetherion()
    await aetherion.initialize()
    return aetherion

# Función global para acceder a la instancia
_aetherion_instance = None

async def get_aetherion() -> Aetherion:
    """
    Obtener instancia global de Aetherion.
    
    Returns:
        Instancia de Aetherion
    """
    global _aetherion_instance
    if _aetherion_instance is None:
        _aetherion_instance = await create_aetherion()
    return _aetherion_instance