"""
Módulo integrador del modelo DeepSeek con el Sistema Genesis.

Este módulo conecta el modelo DeepSeek con el resto de componentes del Sistema Genesis,
permitiendo utilizar análisis avanzado de lenguaje natural en la toma de decisiones
de trading, análisis de sentimiento, optimización de cartera y más.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

from genesis.lsml.deepseek_model import DeepSeekModel
from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.lsml import deepseek_config

logger = logging.getLogger(__name__)

class DeepSeekIntegrator:
    """
    Integrador entre DeepSeek y el Sistema Genesis.
    
    Esta clase coordina la comunicación entre el modelo DeepSeek y los distintos
    componentes del Sistema Genesis, permitiendo utilizar la potencia de procesamiento
    de lenguaje natural para mejorar las decisiones de trading.
    """
    
    def __init__(self, 
                database: Optional[TranscendentalDatabase] = None,
                api_key: Optional[str] = None, 
                model_version: str = "deepseek-coder-33b-instruct",
                intelligence_factor: float = 1.0):
        """
        Inicializar el integrador DeepSeek.
        
        Args:
            database: Instancia de la base de datos transcendental (opcional)
            api_key: Clave API para DeepSeek (opcional)
            model_version: Versión del modelo a utilizar
            intelligence_factor: Factor de inteligencia (1.0 es normal, >1.0 es avanzado)
        """
        self.deepseek_model = DeepSeekModel(api_key=api_key, model_version=model_version)
        self.database = database
        self.intelligence_factor = max(0.1, min(intelligence_factor, 10.0))  # Limitar entre 0.1 y 10.0
        self.initialized = False
        self.last_analysis = {}
        self.last_strategy = {}
        self.last_sentiment = {}
        self.cached_data = {}
        self.cache_ttl = 300  # 5 minutos
        self.cache_timestamps = {}
        
        logger.info(f"DeepSeekIntegrator inicializado con intelligence_factor={intelligence_factor}")
    
    async def initialize(self) -> bool:
        """
        Inicializar el integrador y sus componentes.
        
        Returns:
            True si la inicialización fue exitosa, False en caso contrario
        """
        if self.initialized:
            return True
        
        # Verificar si DeepSeek está habilitado
        if not deepseek_config.is_enabled():
            logger.info("DeepSeek está desactivado en la configuración, no se inicializará")
            return False
        
        try:
            # Cargar configuración actual
            config = deepseek_config.get_config()
            
            # Actualizar parámetros desde la configuración
            self.intelligence_factor = config.get("intelligence_factor", self.intelligence_factor)
            
            # Si hay cambio de versión del modelo, actualizar
            current_model = self.deepseek_model.model_version
            config_model = config.get("model_version")
            if config_model and config_model != current_model:
                logger.info(f"Actualizando versión del modelo de {current_model} a {config_model}")
                self.deepseek_model.model_version = config_model
            
            # Inicializar el modelo DeepSeek
            model_initialized = await self.deepseek_model.initialize()
            if not model_initialized:
                logger.error("No se pudo inicializar el modelo DeepSeek")
                return False
            
            self.initialized = True
            deepseek_config.set_initialized(True)
            logger.info(f"DeepSeekIntegrator inicializado correctamente con factor de inteligencia {self.intelligence_factor}")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar DeepSeekIntegrator: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Cerrar el integrador y liberar recursos."""
        if self.deepseek_model:
            await self.deepseek_model.close()
        self.initialized = False
        logger.info("DeepSeekIntegrator cerrado correctamente")
    
    async def analyze_trading_opportunities(self, 
                                          market_data: Dict[str, Any],
                                          news_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analizar oportunidades de trading utilizando DeepSeek.
        
        Esta función combina análisis de mercado, sentimiento y generación de estrategias
        para proporcionar una visión completa de las oportunidades de trading actuales.
        
        Args:
            market_data: Datos del mercado para analizar
            news_data: Datos de noticias para analizar sentimiento (opcional)
            
        Returns:
            Análisis completo de oportunidades de trading
        """
        # Verificar si DeepSeek está habilitado
        if not deepseek_config.is_enabled():
            return {
                "error": "DeepSeek está desactivado en la configuración",
                "status": "disabled",
                "timestamp": datetime.now().isoformat()
            }
            
        # Verificar si el integrador está inicializado
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return {"error": "No se pudo inicializar DeepSeekIntegrator"}
        
        try:
            # 1. Realizar análisis de condiciones de mercado
            market_analysis = await self.deepseek_model.analyze_market_conditions(market_data)
            
            # 2. Analizar sentimiento si hay datos de noticias
            sentiment_analysis = None
            if news_data:
                sentiment_analysis = await self.deepseek_model.analyze_sentiment(news_data)
            
            # 3. Generar estrategia de trading
            # Ajustar perfil de riesgo basado en el análisis de mercado y sentimiento
            risk_profile = "moderate"
            if market_analysis.get("risk_assessment") == "high":
                risk_profile = "conservative"
            elif market_analysis.get("risk_assessment") == "low":
                risk_profile = "aggressive"
            
            if sentiment_analysis and sentiment_analysis.get("sentiment_score"):
                sentiment_score = sentiment_analysis.get("sentiment_score")
                if sentiment_score > 50:
                    risk_profile = "aggressive" if risk_profile != "conservative" else "moderate"
                elif sentiment_score < -50:
                    risk_profile = "conservative" if risk_profile != "aggressive" else "moderate"
            
            trading_strategy = await self.deepseek_model.generate_trading_strategy(
                market_data,
                risk_profile=risk_profile,
                time_horizon="medium"
            )
            
            # 4. Combinar todos los análisis en un resultado unificado
            opportunities = {
                "timestamp": datetime.now().isoformat(),
                "market_analysis": market_analysis,
                "sentiment_analysis": sentiment_analysis,
                "trading_strategy": trading_strategy,
                "intelligence_factor": self.intelligence_factor,
                "combined_recommendations": []
            }
            
            # 5. Extraer y combinar recomendaciones
            if trading_strategy and "trade_recommendations" in trading_strategy:
                opportunities["combined_recommendations"] = trading_strategy["trade_recommendations"]
            
            # 6. Guardar resultados para uso futuro
            self.last_analysis = market_analysis
            self.last_strategy = trading_strategy
            if sentiment_analysis:
                self.last_sentiment = sentiment_analysis
            
            # 7. Opcionalmente, guardar en la base de datos
            if self.database:
                await self._store_analysis_in_db(opportunities)
            
            logger.info(f"Análisis de oportunidades de trading completado con {len(opportunities.get('combined_recommendations', []))} recomendaciones")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error en analyze_trading_opportunities: {str(e)}")
            return {"error": str(e)}
    
    async def enhance_trading_signals(self, 
                                    signals: List[Dict[str, Any]],
                                    market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Mejorar señales de trading existentes con análisis DeepSeek.
        
        Esta función toma señales generadas por otros componentes del sistema y las enriquece
        con análisis adicional del modelo DeepSeek.
        
        Args:
            signals: Lista de señales de trading a mejorar
            market_data: Datos del mercado para contexto
            
        Returns:
            Señales de trading mejoradas
        """
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return signals  # Devuelve las señales originales si hay error
        
        if not signals:
            return []
        
        try:
            # Preparar prompt para DeepSeek
            system_prompt = """
            Eres un experto en análisis de señales de trading para criptomonedas. Tu tarea es evaluar 
            y mejorar las señales existentes utilizando análisis avanzado.
            
            Para cada señal, evalúa:
            1. La confianza en la señal basada en datos de mercado
            2. El timing óptimo para la entrada/salida
            3. Riesgos específicos a considerar
            4. Potencial adicional no capturado en la señal original
            
            Devuelve las señales mejoradas en formato JSON con el siguiente esquema:
            
            {
                "enhanced_signals": [
                    {
                        "original_signal": original signal object,
                        "confidence": 0.0-1.0,
                        "timing_adjustment": "explanation",
                        "additional_risks": ["risk1", "risk2"],
                        "additional_opportunities": ["opportunity1", "opportunity2"],
                        "enhanced_take_profit": value or null,
                        "enhanced_stop_loss": value or null,
                        "commentary": "brief explanation"
                    }
                ]
            }
            """
            
            combined_data = {
                "signals": signals,
                "market_data": market_data
            }
            
            prompt = f"Evalúa y mejora estas {len(signals)} señales de trading utilizando análisis avanzado y datos de mercado."
            
            response = await self.deepseek_model.query(
                prompt, 
                system_prompt=system_prompt, 
                market_data=combined_data, 
                include_context=False
            )
            
            # Extraer el JSON de la respuesta
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    enhanced_data = json.loads(json_str)
                    enhanced_signals = enhanced_data.get("enhanced_signals", [])
                    
                    # Fusionar las señales originales con las mejoras
                    result_signals = []
                    for enhanced in enhanced_signals:
                        original = enhanced.get("original_signal", {})
                        # Añadir las mejoras al original
                        original["deepseek_confidence"] = enhanced.get("confidence")
                        original["deepseek_commentary"] = enhanced.get("commentary")
                        original["deepseek_risks"] = enhanced.get("additional_risks")
                        original["deepseek_opportunities"] = enhanced.get("additional_opportunities")
                        
                        # Actualizar niveles si se proporcionan mejoras
                        if enhanced.get("enhanced_take_profit"):
                            original["enhanced_take_profit"] = enhanced.get("enhanced_take_profit")
                        if enhanced.get("enhanced_stop_loss"):
                            original["enhanced_stop_loss"] = enhanced.get("enhanced_stop_loss")
                        
                        result_signals.append(original)
                    
                    logger.info(f"Señales de trading mejoradas: {len(result_signals)} señales procesadas")
                    return result_signals
                except json.JSONDecodeError as e:
                    logger.error(f"Error al decodificar respuesta JSON: {str(e)}")
                    logger.debug(f"Respuesta recibida: {response}")
                    return signals  # Devuelve las señales originales si hay error
            else:
                logger.error("No se encontró JSON en la respuesta")
                return signals
            
        except Exception as e:
            logger.error(f"Error en enhance_trading_signals: {str(e)}")
            return signals  # Devuelve las señales originales si hay error
    
    async def analyze_trading_performance(self, 
                                        trade_history: List[Dict[str, Any]],
                                        performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizar el rendimiento de trading histórico y proporcionar recomendaciones.
        
        Args:
            trade_history: Historial de operaciones
            performance_metrics: Métricas de rendimiento
            
        Returns:
            Análisis y recomendaciones para mejorar el rendimiento
        """
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return {"error": "No se pudo inicializar DeepSeekIntegrator"}
        
        try:
            # Preparar prompt para DeepSeek
            system_prompt = """
            Eres un analista experto en evaluación de rendimiento de trading de criptomonedas. Tu tarea es
            analizar el historial de operaciones y las métricas de rendimiento para identificar patrones,
            fortalezas, debilidades y oportunidades de mejora.
            
            Estructura tu respuesta en formato JSON con el siguiente esquema:
            
            {
                "performance_analysis": {
                    "overall_assessment": "brief overall assessment",
                    "key_strengths": ["strength1", "strength2"],
                    "key_weaknesses": ["weakness1", "weakness2"],
                    "pattern_identification": [
                        {
                            "pattern": "description",
                            "impact": "positive | negative",
                            "frequency": "high | medium | low"
                        }
                    ],
                    "metrics_evaluation": {
                        "profit_factor": "assessment",
                        "win_rate": "assessment",
                        "average_trade": "assessment",
                        "drawdown": "assessment",
                        "consistency": "assessment"
                    }
                },
                "improvement_recommendations": [
                    {
                        "area": "area for improvement",
                        "recommendation": "detailed recommendation",
                        "expected_impact": "high | medium | low",
                        "implementation_difficulty": "high | medium | low"
                    }
                ],
                "strategy_adjustments": [
                    {
                        "current_approach": "description",
                        "recommended_adjustment": "description",
                        "rationale": "explanation"
                    }
                ],
                "summary": "brief textual summary"
            }
            """
            
            combined_data = {
                "trade_history": trade_history,
                "performance_metrics": performance_metrics
            }
            
            prompt = "Analiza el rendimiento histórico de trading y proporciona recomendaciones detalladas para mejorar."
            
            response = await self.deepseek_model.query(
                prompt, 
                system_prompt=system_prompt, 
                market_data=combined_data, 
                include_context=False
            )
            
            # Extraer el JSON de la respuesta
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    analysis = json.loads(json_str)
                    logger.info(f"Análisis de rendimiento completado para {len(trade_history)} operaciones")
                    return analysis
                except json.JSONDecodeError as e:
                    logger.error(f"Error al decodificar respuesta JSON: {str(e)}")
                    logger.debug(f"Respuesta recibida: {response}")
                    return {"error": "Error en formato de respuesta", "raw_response": response}
            else:
                logger.error("No se encontró JSON en la respuesta")
                return {"error": "No JSON found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error en analyze_trading_performance: {str(e)}")
            return {"error": str(e)}
    
    async def explain_trading_decision(self, 
                                     decision: Dict[str, Any],
                                     market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Proporcionar una explicación detallada de una decisión de trading.
        
        Esta función es útil para entender el razonamiento detrás de decisiones
        automáticas tomadas por el sistema.
        
        Args:
            decision: Decisión de trading a explicar
            market_context: Contexto de mercado en el momento de la decisión
            
        Returns:
            Explicación detallada de la decisión
        """
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return {"error": "No se pudo inicializar DeepSeekIntegrator"}
        
        try:
            # Preparar prompt para DeepSeek
            system_prompt = """
            Eres un experto en explicar decisiones de trading de criptomonedas de forma clara y detallada.
            Tu tarea es proporcionar una explicación completa y comprensible de una decisión de trading
            específica, conectando las condiciones del mercado con la lógica detrás de la decisión.
            
            Estructura tu respuesta en formato JSON con el siguiente esquema:
            
            {
                "decision_explanation": {
                    "summary": "brief summary of the decision",
                    "market_conditions": {
                        "key_factors": ["factor1", "factor2"],
                        "relevance_to_decision": "explanation"
                    },
                    "decision_logic": "step-by-step explanation of the decision logic",
                    "alternative_scenarios": [
                        {
                            "scenario": "different market condition",
                            "likely_decision": "what would happen instead"
                        }
                    ],
                    "risk_assessment": "explanation of risks considered"
                },
                "educational_insights": [
                    {
                        "concept": "trading concept relevant to this decision",
                        "explanation": "brief educational explanation",
                        "relevance": "how it applies to this specific case"
                    }
                ],
                "natural_language_explanation": "human-friendly explanation of the decision"
            }
            """
            
            combined_data = {
                "decision": decision,
                "market_context": market_context
            }
            
            prompt = "Explica detalladamente esta decisión de trading y la lógica detrás de ella."
            
            response = await self.deepseek_model.query(
                prompt, 
                system_prompt=system_prompt, 
                market_data=combined_data, 
                include_context=False
            )
            
            # Extraer el JSON de la respuesta
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    explanation = json.loads(json_str)
                    logger.info(f"Explicación generada para decisión de trading del tipo {decision.get('type', 'unknown')}")
                    return explanation
                except json.JSONDecodeError as e:
                    logger.error(f"Error al decodificar respuesta JSON: {str(e)}")
                    logger.debug(f"Respuesta recibida: {response}")
                    return {"error": "Error en formato de respuesta", "raw_response": response}
            else:
                logger.error("No se encontró JSON en la respuesta")
                return {"error": "No JSON found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error en explain_trading_decision: {str(e)}")
            return {"error": str(e)}
    
    async def _store_analysis_in_db(self, analysis: Dict[str, Any]) -> bool:
        """
        Almacenar análisis en la base de datos transcendental.
        
        Args:
            analysis: Análisis a almacenar
            
        Returns:
            True si se almacenó correctamente
        """
        if not self.database:
            return False
        
        try:
            # Crear clave única para el análisis
            key = f"deepseek_analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Intentar almacenar en la base de datos
            success = await self.database.store(key, analysis)
            if success:
                logger.info(f"Análisis almacenado correctamente con clave {key}")
            else:
                logger.warning(f"No se pudo almacenar el análisis con clave {key}")
            
            return success
        except Exception as e:
            logger.error(f"Error al almacenar análisis en DB: {str(e)}")
            return False
    
    async def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener análisis recientes de la base de datos.
        
        Args:
            limit: Número máximo de análisis a recuperar
            
        Returns:
            Lista de análisis recientes
        """
        if not self.database:
            return []
        
        try:
            # Buscar claves de análisis recientes
            prefix = "deepseek_analysis_"
            keys = await self.database.get_keys_with_prefix(prefix)
            
            # Ordenar por fecha (las claves contienen timestamp)
            keys.sort(reverse=True)
            keys = keys[:limit]  # Limitar al número solicitado
            
            # Recuperar análisis
            analyses = []
            for key in keys:
                analysis = await self.database.retrieve(key)
                if analysis:
                    analyses.append(analysis)
            
            logger.info(f"Recuperados {len(analyses)} análisis recientes")
            return analyses
        except Exception as e:
            logger.error(f"Error al recuperar análisis recientes: {str(e)}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de uso del integrador."""
        metrics = {
            "deepseek_model_metrics": self.deepseek_model.get_metrics(),
            "intelligence_factor": self.intelligence_factor,
            "cache_size": len(self.cached_data),
            "last_analysis_timestamp": self.last_analysis.get("timestamp") if self.last_analysis else None,
            "last_strategy_timestamp": self.last_strategy.get("timestamp") if self.last_strategy else None
        }
        return metrics