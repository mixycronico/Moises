"""
Módulo para interacción con modelos DeepSeek.

Este módulo implementa la interfaz para comunicarse con la API de DeepSeek,
permitiendo utilizar sus modelos para análisis avanzado, generación de texto
y otros casos de uso relacionados con el sistema Genesis.
"""

import os
import json
import time
import logging
import asyncio
import random
from typing import Dict, Any, List, Optional, Tuple, Union
import base64
import hashlib
import hmac
try:
    import aiohttp
except ImportError:
    aiohttp = None
try:
    import numpy as np
except ImportError:
    np = None

# Importar configuración de DeepSeek
from genesis.lsml import deepseek_config

logger = logging.getLogger(__name__)

class DeepSeekModel:
    """
    Clase para interacción con modelos DeepSeek.
    
    Esta clase implementa un cliente para la API de DeepSeek,
    permitiendo consultas a sus modelos LLM para análisis 
    avanzado de trading, procesamiento de lenguaje natural
    y generación de estrategias.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                model_version: str = "deepseek-coder-33b-instruct",
                base_url: str = "https://api.deepseek-ai.com/v1",
                timeout: int = 120):
        """
        Inicializar cliente DeepSeek.
        
        Args:
            api_key: Clave API de DeepSeek (opcional, también puede usar DEEPSEEK_API_KEY)
            model_version: Versión del modelo a utilizar
            base_url: URL base para la API
            timeout: Timeout para solicitudes en segundos
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model_version = model_version
        self.base_url = base_url
        self.timeout = timeout
        self.session = None
        self.initialized = False
        self.request_counter = 0
        self.total_tokens = 0
        self.cache = {}
        self.cache_hits = 0
        self.last_request_time = 0
        self.rate_limit_ms = 1000  # 1 solicitud por segundo (ajustable)
        self.metrics = {
            "requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "total_tokens": 0,
            "average_response_time": 0,
            "total_response_time": 0
        }
        
        logger.info(f"DeepSeekModel inicializado con modelo {model_version}")
    
    async def initialize(self) -> bool:
        """
        Inicializar el cliente aiohttp y validar API key.
        
        Returns:
            True si la inicialización fue exitosa, False en caso contrario
        """
        if self.initialized:
            return True
        
        if aiohttp is None:
            logger.error("No se pudo importar aiohttp, asegúrate de tenerlo instalado")
            return False
        
        try:
            if not self.api_key:
                logger.warning("No se proporcionó API key para DeepSeek, algunas funciones estarán limitadas")
                # Aún así inicializamos para permitir modo simulado
            
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self._get_headers()
            )
            
            self.initialized = True
            logger.info(f"Cliente DeepSeek inicializado correctamente para modelo {self.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar cliente DeepSeek: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Cerrar sesión y liberar recursos."""
        if self.session:
            await self.session.close()
            self.session = None
        self.initialized = False
        logger.info("Cliente DeepSeek cerrado correctamente")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Construir headers para solicitudes a la API.
        
        Returns:
            Headers para solicitudes HTTP
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def _rate_limit_check(self) -> None:
        """
        Verificar y aplicar rate limiting.
        """
        current_time = time.time() * 1000  # ms
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_ms:
            sleep_time = (self.rate_limit_ms - time_since_last) / 1000
            logger.debug(f"Rate limiting: esperando {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time() * 1000
    
    def _get_cache_key(self, prompt: str, system_prompt: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generar clave de caché para una consulta.
        
        Args:
            prompt: Prompt principal
            system_prompt: Prompt de sistema (opcional)
            params: Parámetros adicionales (opcional)
            
        Returns:
            Clave de caché
        """
        key_parts = [prompt]
        if system_prompt:
            key_parts.append(system_prompt)
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        
        # Usar hash para clave de caché
        key = hashlib.md5("".join(key_parts).encode()).hexdigest()
        return key
    
    async def query(self, 
                  prompt: str, 
                  system_prompt: Optional[str] = None,
                  market_data: Optional[Dict[str, Any]] = None,
                  temperature: float = 0.7,
                  max_tokens: int = 4096,
                  use_cache: bool = True,
                  include_context: bool = True) -> str:
        """
        Realizar una consulta al modelo DeepSeek.
        
        Args:
            prompt: Prompt principal para la consulta
            system_prompt: Prompt de sistema (opcional)
            market_data: Datos de mercado para contexto (opcional)
            temperature: Temperatura para generación (0.0-1.0)
            max_tokens: Máximo de tokens a generar
            use_cache: Si se debe usar caché para consultas idénticas
            include_context: Si se debe incluir contexto de mercado
            
        Returns:
            Respuesta del modelo como texto
        """
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return "Error: No se pudo inicializar el cliente DeepSeek"
        
        # Agregar contexto de mercado si está disponible
        full_prompt = prompt
        if market_data and include_context:
            try:
                market_context = f"\nContexto de mercado:\n{json.dumps(market_data, indent=2)}"
                full_prompt = f"{prompt}\n\n{market_context}"
            except Exception as e:
                logger.warning(f"Error al serializar datos de mercado: {str(e)}")
        
        # Verificar caché
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(full_prompt, system_prompt, {
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            
            if cache_key in self.cache:
                self.cache_hits += 1
                self.metrics["cached_responses"] += 1
                logger.debug(f"Respuesta obtenida de caché con clave {cache_key}")
                return self.cache[cache_key]
        
        self.request_counter += 1
        self.metrics["requests"] += 1
        self._rate_limit_check()
        
        start_time = time.time()
        
        # Si no hay API key, generar una respuesta simulada
        if not self.api_key:
            result = self._generate_simulated_response(prompt, system_prompt, market_data)
            logger.warning("Usando respuesta simulada debido a falta de API key")
            return result
        
        try:
            payload = {
                "model": self.model_version,
                "messages": []
            }
            
            # Añadir mensaje del sistema si existe
            if system_prompt:
                payload["messages"].append({"role": "system", "content": system_prompt})
            
            # Añadir mensaje del usuario
            payload["messages"].append({"role": "user", "content": full_prompt})
            
            # Añadir parámetros de generación
            payload["temperature"] = temperature
            payload["max_tokens"] = max_tokens
            
            # Realizar solicitud
            if not self.session:
                await self.initialize()
            
            url = f"{self.base_url}/chat/completions"
            response = None
            
            async with self.session.post(url, json=payload) as resp:
                response = await resp.json()
                
                # Procesar respuesta
                if resp.status == 200 and "choices" in response:
                    result = response["choices"][0]["message"]["content"]
                    
                    # Actualizar métricas
                    if "usage" in response:
                        usage = response["usage"]
                        self.total_tokens += usage.get("total_tokens", 0)
                        self.metrics["total_tokens"] += usage.get("total_tokens", 0)
                    
                    # Guardar en caché si está habilitado
                    if use_cache and cache_key:
                        self.cache[cache_key] = result
                    
                    # Actualizar métricas de éxito
                    self.metrics["successful_requests"] += 1
                    
                    # Calcular tiempo de respuesta y actualizar métricas
                    response_time = time.time() - start_time
                    self.metrics["total_response_time"] += response_time
                    self.metrics["average_response_time"] = (
                        self.metrics["total_response_time"] / self.metrics["successful_requests"]
                    )
                    
                    logger.info(f"Consulta a DeepSeek exitosa en {response_time:.2f}s")
                    return result
                else:
                    error_msg = f"Error en API DeepSeek: {response}"
                    logger.error(error_msg)
                    self.metrics["failed_requests"] += 1
                    return f"Error: {error_msg}"
        
        except Exception as e:
            logger.error(f"Error al realizar consulta a DeepSeek: {str(e)}")
            self.metrics["failed_requests"] += 1
            return f"Error: {str(e)}"
    
    def _generate_simulated_response(self, 
                                    prompt: str,
                                    system_prompt: Optional[str] = None,
                                    market_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generar respuesta simulada cuando no hay API key.
        
        Args:
            prompt: Prompt para la consulta
            system_prompt: Prompt de sistema (opcional)
            market_data: Datos de mercado (opcional)
            
        Returns:
            Respuesta simulada
        """
        import random
        
        responses = [
            "Basado en el análisis técnico, se observa una tendencia alcista a corto plazo.",
            "Los indicadores muestran posible sobreventa. Considere posiciones largas con stop loss ajustado.",
            "Divergencia RSI negativa detectada. Precaución con posiciones largas.",
            "Cruce de medias móviles positivo en las últimas velas. Señal potencialmente alcista.",
            "El análisis sugiere volatilidad creciente en las próximas sesiones.",
            "Patrón de consolidación identificado. Observar ruptura del rango.",
            "El sentimiento de mercado es mixto, con tendencia a la cautela."
        ]
        
        # Añadir tiempo de respuesta simulado
        time.sleep(0.5 + (random.random() * 1.5))
        
        # Para prompt de análisis de mercado
        if "mercado" in prompt.lower() or "market" in prompt.lower():
            if market_data and "symbol" in market_data:
                symbol = market_data["symbol"]
                return (
                    f"Análisis de {symbol}: Los indicadores técnicos muestran una tendencia "
                    f"{'alcista' if random.random() > 0.5 else 'bajista'} a corto plazo. "
                    f"RSI: {random.randint(30, 70)}, MACD: {'positivo' if random.random() > 0.5 else 'negativo'}. "
                    f"Recomendación: {'Comprar' if random.random() > 0.6 else 'Vender' if random.random() > 0.7 else 'Mantener'}."
                )
        
        # Para solicitudes de estrategia
        if "estrategia" in prompt.lower() or "strategy" in prompt.lower():
            return (
                "Estrategia recomendada: Utilizar cruce de medias móviles (EMA 9 y EMA 21) para entradas. "
                "Establecer stop loss en 2% y take profit en 4%. Limitar exposición al 5% del capital por operación. "
                "Complementar con análisis RSI para confirmar momentos de entrada. "
                "Prestar atención a niveles clave de soporte/resistencia para ajustar objetivos."
            )
        
        # Respuesta genérica
        return random.choice(responses)
    
    async def analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizar condiciones actuales del mercado.
        
        Args:
            market_data: Datos del mercado a analizar
            
        Returns:
            Análisis de condiciones de mercado
        """
        prompt = """
        Analiza las condiciones actuales del mercado con los datos proporcionados.
        Tu análisis debe incluir:
        1. Tendencia predominante (alcista, bajista, neutral)
        2. Fortaleza de la tendencia
        3. Niveles clave de soporte y resistencia
        4. Indicadores técnicos relevantes
        5. Evaluación de riesgo (alto, medio, bajo)
        6. Volatilidad actual y proyectada
        """
        
        system_prompt = """
        Eres un analista experto en mercados financieros especializado en criptomonedas.
        Tu tarea es analizar objetivamente las condiciones actuales del mercado y proporcionar
        insights accionables. Basa tu análisis estrictamente en los datos proporcionados
        y devuelve tu respuesta en formato JSON estructurado que incluya las categorías
        solicitadas en el prompt.
        """
        
        response = await self.query(
            prompt=prompt,
            system_prompt=system_prompt,
            market_data=market_data
        )
        
        # Intentar extraer JSON de la respuesta
        try:
            # Buscar patrón de JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                analysis = json.loads(json_str)
                return analysis
            else:
                # Si no encontramos JSON, crear estructura manualmente
                return self._extract_market_analysis(response)
        except Exception as e:
            logger.warning(f"Error al parsear análisis de mercado: {str(e)}")
            # Estructura básica con la respuesta en texto
            return {
                "text_analysis": response,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _extract_market_analysis(self, text: str) -> Dict[str, Any]:
        """
        Extraer información estructurada de análisis de mercado en texto.
        
        Args:
            text: Texto de análisis
            
        Returns:
            Análisis estructurado
        """
        analysis = {
            "trend": "neutral",
            "trend_strength": "weak",
            "support_levels": [],
            "resistance_levels": [],
            "technical_indicators": {},
            "risk_assessment": "medium",
            "volatility": "medium",
            "raw_text": text,
            "timestamp": time.time()
        }
        
        # Extraer tendencia
        if "alcista" in text.lower() or "bullish" in text.lower():
            analysis["trend"] = "bullish"
        elif "bajista" in text.lower() or "bearish" in text.lower():
            analysis["trend"] = "bearish"
        
        # Extraer evaluación de riesgo
        if "riesgo alto" in text.lower() or "high risk" in text.lower():
            analysis["risk_assessment"] = "high"
        elif "riesgo bajo" in text.lower() or "low risk" in text.lower():
            analysis["risk_assessment"] = "low"
        
        return analysis
    
    async def analyze_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analizar sentimiento de noticias o fuentes sociales.
        
        Args:
            news_data: Lista de noticias o publicaciones sociales
            
        Returns:
            Análisis de sentimiento
        """
        if not news_data:
            return {"error": "No hay datos de noticias para analizar"}
        
        # Preparar prompt
        news_text = "\n\n".join([
            f"Título: {news.get('title', 'N/A')}\n"
            f"Fecha: {news.get('date', 'N/A')}\n"
            f"Fuente: {news.get('source', 'N/A')}\n"
            f"Contenido: {news.get('content', 'N/A')}"
            for news in news_data[:5]  # Limitar a 5 noticias
        ])
        
        prompt = f"""
        Analiza el sentimiento de las siguientes noticias relacionadas con criptomonedas:
        
        {news_text}
        
        Proporciona una evaluación detallada del sentimiento incluyendo:
        1. Puntuación general de sentimiento (de -100 a +100)
        2. Temas predominantes
        3. Impacto potencial en el mercado
        4. Palabras clave con sentimiento identificado
        """
        
        system_prompt = """
        Eres un analista de sentimiento especializado en mercados financieros y criptomonedas.
        Tu tarea es evaluar objetivamente el sentimiento de noticias y publicaciones,
        determinando si son positivas, negativas o neutras, y cuantificar ese sentimiento.
        Devuelve tu análisis en formato JSON estructurado que incluya las categorías
        solicitadas en el prompt.
        """
        
        response = await self.query(
            prompt=prompt,
            system_prompt=system_prompt,
            market_data=None,  # No pasar para evitar confusión
            include_context=False
        )
        
        # Intentar extraer JSON de la respuesta
        try:
            # Buscar patrón de JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                sentiment = json.loads(json_str)
                return sentiment
            else:
                # Si no encontramos JSON, crear estructura manualmente
                return self._extract_sentiment_analysis(response)
        except Exception as e:
            logger.warning(f"Error al parsear análisis de sentimiento: {str(e)}")
            # Estructura básica con la respuesta en texto
            return {
                "text_analysis": response,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _extract_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Extraer información estructurada de análisis de sentimiento en texto.
        
        Args:
            text: Texto de análisis
            
        Returns:
            Análisis estructurado
        """
        sentiment = {
            "sentiment_score": 0,
            "sentiment_label": "neutral",
            "topics": [],
            "market_impact": "neutral",
            "keywords": {},
            "raw_text": text,
            "timestamp": time.time()
        }
        
        # Extraer sentimiento
        if "positivo" in text.lower() or "positive" in text.lower() or "bullish" in text.lower():
            sentiment["sentiment_label"] = "positive"
            # Estimación aproximada de score
            if "muy positivo" in text.lower() or "very positive" in text.lower():
                sentiment["sentiment_score"] = 75
            else:
                sentiment["sentiment_score"] = 50
        elif "negativo" in text.lower() or "negative" in text.lower() or "bearish" in text.lower():
            sentiment["sentiment_label"] = "negative"
            # Estimación aproximada de score
            if "muy negativo" in text.lower() or "very negative" in text.lower():
                sentiment["sentiment_score"] = -75
            else:
                sentiment["sentiment_score"] = -50
        
        return sentiment
    
    async def generate_trading_strategy(self, 
                                      market_data: Dict[str, Any],
                                      risk_profile: str = "moderate",
                                      time_horizon: str = "medium") -> Dict[str, Any]:
        """
        Generar estrategia de trading basada en datos de mercado.
        
        Args:
            market_data: Datos del mercado
            risk_profile: Perfil de riesgo (conservative, moderate, aggressive)
            time_horizon: Horizonte temporal (short, medium, long)
            
        Returns:
            Estrategia de trading
        """
        # Preparar prompt
        prompt = f"""
        Genera una estrategia de trading detallada basada en los datos de mercado proporcionados.
        
        Perfil de riesgo: {risk_profile}
        Horizonte temporal: {time_horizon}
        
        La estrategia debe incluir:
        1. Configuración específica de indicadores técnicos a utilizar
        2. Condiciones precisas de entrada y salida
        3. Reglas de gestión de riesgo (tamaño de posición, stop loss, take profit)
        4. Recomendaciones de trading concretas para el mercado actual
        5. Escenarios alternativos y plan de contingencia
        """
        
        system_prompt = """
        Eres un estratega de trading experto especializado en mercados de criptomonedas.
        Tu tarea es generar estrategias detalladas y accionables basadas en análisis técnico
        y condiciones actuales del mercado. Prioriza la gestión de riesgo y proporciona
        parámetros específicos para cada componente de la estrategia.
        Devuelve tu estrategia en formato JSON estructurado que incluya las categorías
        solicitadas en el prompt.
        """
        
        response = await self.query(
            prompt=prompt,
            system_prompt=system_prompt,
            market_data=market_data
        )
        
        # Intentar extraer JSON de la respuesta
        try:
            # Buscar patrón de JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                strategy = json.loads(json_str)
                return strategy
            else:
                # Si no encontramos JSON, crear estructura manualmente
                return self._extract_trading_strategy(response, risk_profile, time_horizon)
        except Exception as e:
            logger.warning(f"Error al parsear estrategia de trading: {str(e)}")
            # Estructura básica con la respuesta en texto
            return {
                "text_strategy": response,
                "risk_profile": risk_profile,
                "time_horizon": time_horizon,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _extract_trading_strategy(self, text: str, risk_profile: str, time_horizon: str) -> Dict[str, Any]:
        """
        Extraer información estructurada de estrategia de trading en texto.
        
        Args:
            text: Texto de análisis
            risk_profile: Perfil de riesgo
            time_horizon: Horizonte temporal
            
        Returns:
            Estrategia estructurada
        """
        strategy = {
            "indicators": {},
            "entry_conditions": [],
            "exit_conditions": [],
            "risk_management": {
                "position_size": "2-5% del capital",
                "stop_loss": "2% del precio de entrada",
                "take_profit": "4% del precio de entrada"
            },
            "trade_recommendations": [],
            "alternative_scenarios": [],
            "risk_profile": risk_profile,
            "time_horizon": time_horizon,
            "raw_text": text,
            "timestamp": time.time()
        }
        
        # Extraer recomendaciones de trading
        if "comprar" in text.lower() or "buy" in text.lower():
            strategy["trade_recommendations"].append({
                "action": "buy",
                "reason": "Señal técnica positiva identificada en el análisis"
            })
        elif "vender" in text.lower() or "sell" in text.lower():
            strategy["trade_recommendations"].append({
                "action": "sell",
                "reason": "Señal técnica negativa identificada en el análisis"
            })
        else:
            strategy["trade_recommendations"].append({
                "action": "hold",
                "reason": "No hay señales claras en este momento"
            })
        
        # Identificar indicadores mencionados
        indicator_patterns = [
            "RSI", "MACD", "EMA", "SMA", "Bollinger", "ADX", "ATR", 
            "Estocástico", "Stochastic", "Ichimoku", "DMI", "OBV"
        ]
        
        for indicator in indicator_patterns:
            if indicator.lower() in text.lower():
                strategy["indicators"][indicator] = "Mencionado en estrategia"
        
        return strategy
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas de uso del modelo.
        
        Returns:
            Diccionario de métricas
        """
        metrics = self.metrics.copy()
        metrics["cache_size"] = len(self.cache)
        metrics["cache_hits"] = self.cache_hits
        metrics["model_version"] = self.model_version
        metrics["has_api_key"] = self.api_key is not None
        
        return metrics