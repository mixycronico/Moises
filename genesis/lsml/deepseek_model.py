"""
Módulo para integración con el modelo DeepSeek para el Sistema Genesis.

Este módulo proporciona la infraestructura para utilizar el modelo DeepSeek
para análisis de mercado avanzado, predicción y generación de señales en
el contexto del trading de criptomonedas.

DeepSeek es un modelo de gran escala que puede procesar lenguaje natural
y datos estructurados para extraer insights valiosos y generar predicciones.
"""

import os
import json
import logging
import aiohttp
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class DeepSeekModel:
    """
    Clase para integración con el modelo DeepSeek para análisis avanzado.
    
    Esta clase proporciona la infraestructura para:
    1. Consultar al modelo DeepSeek mediante API
    2. Procesar los datos de mercado en un formato adecuado para el modelo
    3. Interpretar las respuestas y convertirlas en señales accionables
    4. Mantener un contexto de análisis entre peticiones
    """
    
    def __init__(self, api_key: Optional[str] = None, model_version: str = "deepseek-coder-33b-instruct"):
        """
        Inicializar el modelo DeepSeek.
        
        Args:
            api_key: Clave API para DeepSeek (opcional, también puede usar variables de entorno)
            model_version: Versión del modelo a utilizar
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model_version = model_version
        self.base_url = "https://api.deepseek.com/v1"
        self.session = None
        self.context = []
        self.max_context_length = 10  # Número máximo de intercambios en el contexto
        self.initialized = False
        self.temperature = 0.2  # Temperatura baja para respuestas más deterministas
        self.metrics = {
            "requests": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "last_request_timestamp": None
        }
        
        logger.info(f"DeepSeekModel inicializado con model_version={model_version}")
    
    async def initialize(self) -> bool:
        """
        Inicializar el cliente asíncrono y verificar la conexión a la API.
        
        Returns:
            True si la inicialización fue exitosa, False en caso contrario
        """
        if self.initialized:
            return True
            
        if not self.api_key:
            logger.error("No se encontró la clave API de DeepSeek. Configure DEEPSEEK_API_KEY como variable de entorno o pase api_key al constructor.")
            return False
            
        try:
            self.session = aiohttp.ClientSession()
            # Verificar que podemos conectarnos a la API
            await self._validate_api_connection()
            self.initialized = True
            logger.info("DeepSeekModel inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar DeepSeekModel: {str(e)}")
            if self.session:
                await self.session.close()
                self.session = None
            return False
    
    async def _validate_api_connection(self) -> bool:
        """
        Validar que podemos conectarnos a la API de DeepSeek.
        
        Returns:
            True si la conexión es válida, False en caso contrario
        """
        try:
            # Intenta una consulta simple para verificar la conexión
            system_prompt = "Responde con la palabra 'connected' si recibes este mensaje."
            response = await self.query("Test de conexión", system_prompt=system_prompt)
            if "connected" in response.lower():
                logger.info("Conexión a la API de DeepSeek validada correctamente")
                return True
            else:
                logger.warning(f"La API de DeepSeek respondió, pero la respuesta no es la esperada: {response}")
                return False
        except Exception as e:
            logger.error(f"Error al validar la conexión a la API de DeepSeek: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Cerrar la sesión HTTP y liberar recursos."""
        if self.session:
            await self.session.close()
            self.session = None
        self.initialized = False
        logger.info("DeepSeekModel cerrado correctamente")
    
    async def query(self, 
                   prompt: str, 
                   system_prompt: Optional[str] = None,
                   market_data: Optional[Dict[str, Any]] = None,
                   include_context: bool = True,
                   temperature: Optional[float] = None) -> str:
        """
        Realizar una consulta al modelo DeepSeek.
        
        Args:
            prompt: Texto de la consulta
            system_prompt: Instrucciones del sistema (opcional)
            market_data: Datos de mercado para incluir en la consulta (opcional)
            include_context: Si se debe incluir el contexto de conversaciones previas
            temperature: Temperatura para la generación (opcional, usa el valor predeterminado si es None)
            
        Returns:
            Respuesta del modelo como texto
            
        Raises:
            Exception: Si hay un error en la consulta
        """
        if not self.initialized:
            success = await self.initialize()
            if not success:
                raise Exception("No se pudo inicializar DeepSeekModel. Verifique las credenciales y la conexión.")
        
        start_time = datetime.now()
        self.metrics["requests"] += 1
        self.metrics["last_request_timestamp"] = start_time
        
        # Formatear los datos de mercado si están presentes
        formatted_market_data = ""
        if market_data:
            formatted_market_data = "\n\nDatos de mercado actuales:\n"
            formatted_market_data += json.dumps(market_data, indent=2)
        
        # Construir los mensajes para la API
        messages = []
        
        # Incluir el contexto si se solicita
        if include_context and self.context:
            messages.extend(self.context)
        
        # Añadir system prompt si está presente
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Añadir el prompt del usuario con los datos de mercado
        user_content = prompt
        if formatted_market_data:
            user_content += formatted_market_data
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_version,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature
            }
            
            async with self.session.post(f"{self.base_url}/chat/completions", 
                                        headers=headers, 
                                        json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.metrics["failed_responses"] += 1
                    error_msg = f"Error en la API de DeepSeek (status {response.status}): {error_text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                response_data = await response.json()
                
                # Actualizar métricas
                self.metrics["successful_responses"] += 1
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                self.metrics["total_response_time"] += response_time
                self.metrics["average_response_time"] = (
                    self.metrics["total_response_time"] / self.metrics["successful_responses"]
                )
                
                # Extraer la respuesta
                assistant_response = response_data["choices"][0]["message"]["content"]
                
                # Actualizar el contexto
                self.context.append({
                    "role": "user",
                    "content": user_content
                })
                self.context.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                # Limitar el tamaño del contexto
                if len(self.context) > self.max_context_length * 2:
                    self.context = self.context[-self.max_context_length * 2:]
                
                return assistant_response
                
        except Exception as e:
            self.metrics["failed_responses"] += 1
            logger.error(f"Error al consultar la API de DeepSeek: {str(e)}")
            raise
    
    async def analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizar condiciones de mercado utilizando DeepSeek.
        
        Args:
            market_data: Datos del mercado para analizar
            
        Returns:
            Análisis estructurado de las condiciones del mercado
        """
        system_prompt = """
        Eres un analista experto en mercados de criptomonedas. Tu tarea es analizar los datos de mercado 
        proporcionados y ofrecer un análisis detallado de las condiciones actuales del mercado.
        
        Estructura tu respuesta en formato JSON con el siguiente esquema:
        
        {
            "market_sentiment": "bullish | bearish | neutral",
            "trend_analysis": {
                "short_term": "up | down | sideways",
                "medium_term": "up | down | sideways",
                "long_term": "up | down | sideways"
            },
            "volatility": "high | medium | low",
            "key_levels": {
                "support": [price_levels],
                "resistance": [price_levels]
            },
            "risk_assessment": "high | medium | low",
            "opportunities": [
                {
                    "symbol": "ticker",
                    "type": "entry | exit",
                    "direction": "long | short",
                    "confidence": 0.0-1.0,
                    "rationale": "brief explanation"
                }
            ],
            "analysis_summary": "brief textual summary"
        }
        
        Basa tu análisis exclusivamente en los datos proporcionados y utiliza técnicas de análisis técnico y fundamental.
        """
        
        prompt = "Realiza un análisis detallado de las condiciones actuales del mercado de criptomonedas basado en los datos proporcionados."
        
        try:
            response = await self.query(prompt, system_prompt=system_prompt, market_data=market_data, include_context=False)
            
            # Extraer el JSON de la respuesta
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    analysis = json.loads(json_str)
                    logger.info(f"Análisis de mercado completado para {len(market_data)} activos")
                    return analysis
                except json.JSONDecodeError as e:
                    logger.error(f"Error al decodificar respuesta JSON: {str(e)}")
                    logger.debug(f"Respuesta recibida: {response}")
                    return {"error": "Error en formato de respuesta", "raw_response": response}
            else:
                logger.error("No se encontró JSON en la respuesta")
                return {"error": "No JSON found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error en analyze_market_conditions: {str(e)}")
            return {"error": str(e)}
    
    async def generate_trading_strategy(self, 
                                       market_data: Dict[str, Any],
                                       risk_profile: str = "moderate",
                                       time_horizon: str = "medium") -> Dict[str, Any]:
        """
        Generar una estrategia de trading basada en análisis avanzado.
        
        Args:
            market_data: Datos del mercado para analizar
            risk_profile: Perfil de riesgo (conservative, moderate, aggressive)
            time_horizon: Horizonte temporal (short, medium, long)
            
        Returns:
            Estrategia de trading estructurada
        """
        system_prompt = f"""
        Eres un estratega experto en trading de criptomonedas. Tu tarea es generar una estrategia de trading
        detallada y accionable basada en los datos de mercado proporcionados.
        
        Perfil de riesgo: {risk_profile}
        Horizonte temporal: {time_horizon}
        
        Estructura tu respuesta en formato JSON con el siguiente esquema:
        
        {{
            "strategy_name": "nombre descriptivo",
            "strategy_type": "trend-following | mean-reversion | breakout | etc",
            "risk_profile": "{risk_profile}",
            "time_horizon": "{time_horizon}",
            "trade_recommendations": [
                {{
                    "symbol": "ticker",
                    "action": "buy | sell | hold",
                    "position_size": 0.0-1.0,
                    "entry_price": "price or price range",
                    "stop_loss": "price",
                    "take_profit": "price",
                    "rationale": "brief explanation"
                }}
            ],
            "risk_management": {{
                "max_position_size": 0.0-1.0,
                "max_drawdown": "percentage",
                "position_sizing_strategy": "explanation"
            }},
            "success_indicators": [
                "indicator 1",
                "indicator 2"
            ],
            "fallback_plan": "explanation of fallback strategy",
            "strategy_summary": "brief textual summary"
        }}
        
        Basa tu estrategia exclusivamente en los datos proporcionados y asegúrate de que sea coherente con el perfil de riesgo y horizonte temporal especificados.
        """
        
        prompt = f"Genera una estrategia de trading detallada para un perfil de riesgo {risk_profile} y horizonte temporal {time_horizon}."
        
        try:
            response = await self.query(prompt, system_prompt=system_prompt, market_data=market_data, include_context=False)
            
            # Extraer el JSON de la respuesta
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    strategy = json.loads(json_str)
                    logger.info(f"Estrategia de trading generada para perfil {risk_profile} y horizonte {time_horizon}")
                    return strategy
                except json.JSONDecodeError as e:
                    logger.error(f"Error al decodificar respuesta JSON: {str(e)}")
                    logger.debug(f"Respuesta recibida: {response}")
                    return {"error": "Error en formato de respuesta", "raw_response": response}
            else:
                logger.error("No se encontró JSON en la respuesta")
                return {"error": "No JSON found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error en generate_trading_strategy: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_sentiment(self, 
                              news_data: List[Dict[str, Any]],
                              social_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analizar sentimiento a partir de noticias y datos sociales.
        
        Args:
            news_data: Lista de noticias con título, contenido y fecha
            social_data: Lista de posts de redes sociales (opcional)
            
        Returns:
            Análisis de sentimiento estructurado
        """
        system_prompt = """
        Eres un analista experto en sentimiento de mercado para criptomonedas. Tu tarea es analizar
        el sentimiento general y por activo a partir de noticias y datos de redes sociales.
        
        Estructura tu respuesta en formato JSON con el siguiente esquema:
        
        {
            "overall_sentiment": "bullish | bearish | neutral",
            "sentiment_score": -100 to 100,
            "asset_sentiment": [
                {
                    "symbol": "ticker",
                    "sentiment": "bullish | bearish | neutral",
                    "score": -100 to 100,
                    "key_topics": ["topic1", "topic2"],
                    "key_events": ["event1", "event2"]
                }
            ],
            "key_concerns": ["concern1", "concern2"],
            "key_opportunities": ["opportunity1", "opportunity2"],
            "summary": "brief textual summary"
        }
        
        Basa tu análisis exclusivamente en los datos proporcionados.
        """
        
        # Preparar datos para la consulta
        combined_data = {
            "news": news_data
        }
        
        if social_data:
            combined_data["social"] = social_data
        
        prompt = "Analiza el sentimiento del mercado de criptomonedas basado en las noticias y datos sociales proporcionados."
        
        try:
            response = await self.query(prompt, system_prompt=system_prompt, market_data=combined_data, include_context=False)
            
            # Extraer el JSON de la respuesta
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    sentiment = json.loads(json_str)
                    logger.info(f"Análisis de sentimiento completado para {len(news_data)} noticias")
                    return sentiment
                except json.JSONDecodeError as e:
                    logger.error(f"Error al decodificar respuesta JSON: {str(e)}")
                    logger.debug(f"Respuesta recibida: {response}")
                    return {"error": "Error en formato de respuesta", "raw_response": response}
            else:
                logger.error("No se encontró JSON en la respuesta")
                return {"error": "No JSON found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error en analyze_sentiment: {str(e)}")
            return {"error": str(e)}
    
    async def optimize_portfolio(self, 
                               current_portfolio: Dict[str, Any],
                               market_data: Dict[str, Any],
                               constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimizar una cartera de criptomonedas.
        
        Args:
            current_portfolio: Cartera actual con activos y pesos
            market_data: Datos del mercado para analizar
            constraints: Restricciones para la optimización (opcional)
            
        Returns:
            Recomendaciones de optimización de cartera
        """
        system_prompt = """
        Eres un gestor de carteras experto en criptomonedas. Tu tarea es optimizar una cartera
        existente basándote en datos de mercado y restricciones especificadas.
        
        Estructura tu respuesta en formato JSON con el siguiente esquema:
        
        {
            "portfolio_recommendation": {
                "assets": [
                    {
                        "symbol": "ticker",
                        "current_weight": 0.0-1.0,
                        "recommended_weight": 0.0-1.0,
                        "action": "increase | decrease | maintain | add | remove",
                        "rationale": "brief explanation"
                    }
                ],
                "rebalancing_strategy": "explanation",
                "expected_performance": {
                    "expected_return": "percentage",
                    "expected_risk": "percentage",
                    "sharpe_ratio": "value"
                }
            },
            "portfolio_analysis": {
                "diversification_score": 0-10,
                "risk_concentration": "explanation",
                "correlation_assessment": "explanation"
            },
            "optimization_summary": "brief textual summary"
        }
        
        Basa tu optimización exclusivamente en los datos proporcionados y asegúrate de que cumpla con las restricciones especificadas.
        """
        
        # Preparar datos para la consulta
        combined_data = {
            "current_portfolio": current_portfolio,
            "market_data": market_data
        }
        
        if constraints:
            combined_data["constraints"] = constraints
        
        prompt = "Optimiza la cartera de criptomonedas proporcionada basándote en los datos de mercado y restricciones especificadas."
        
        try:
            response = await self.query(prompt, system_prompt=system_prompt, market_data=combined_data, include_context=False)
            
            # Extraer el JSON de la respuesta
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    optimization = json.loads(json_str)
                    logger.info(f"Optimización de cartera completada para {len(current_portfolio)} activos")
                    return optimization
                except json.JSONDecodeError as e:
                    logger.error(f"Error al decodificar respuesta JSON: {str(e)}")
                    logger.debug(f"Respuesta recibida: {response}")
                    return {"error": "Error en formato de respuesta", "raw_response": response}
            else:
                logger.error("No se encontró JSON en la respuesta")
                return {"error": "No JSON found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error en optimize_portfolio: {str(e)}")
            return {"error": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de uso del modelo."""
        return self.metrics
    
    def reset_context(self) -> None:
        """Resetear el contexto de conversación."""
        self.context = []
        logger.info("Contexto de conversación reseteado")