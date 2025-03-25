"""
Integrador de Análisis para Aetherion.

Este módulo proporciona la integración entre Aetherion y los diferentes módulos
de análisis del Sistema Genesis, permitiendo que la consciencia artificial 
utilice las capacidades analíticas avanzadas para proporcionar insights.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import datetime

# Configurar logging
logger = logging.getLogger(__name__)

class AnalysisIntegrator:
    """Integrador entre Aetherion y los módulos de análisis."""
    
    def __init__(self):
        """Inicializar integrador."""
        self.available_modules = {}
        self.analysis_cache = {}
        self.last_update = None
        
        # Intentar inicializar módulos de análisis
        self._initialize_modules()
        
    def _initialize_modules(self) -> Dict[str, bool]:
        """
        Inicializar módulos de análisis.
        
        Returns:
            Diccionario con estado de inicialización
        """
        modules_status = {}
        
        # Intentar cargar módulos de análisis técnico
        try:
            from genesis.analysis.technical_analysis import TechnicalAnalysisManager
            self.technical_analysis = TechnicalAnalysisManager()
            modules_status["technical_analysis"] = True
            self.available_modules["technical_analysis"] = "TechnicalAnalysisManager"
            logger.info("Módulo de análisis técnico inicializado correctamente")
        except ImportError:
            self.technical_analysis = None
            modules_status["technical_analysis"] = False
            logger.warning("Módulo de análisis técnico no disponible")
        
        # Intentar cargar módulos de análisis fundamental
        try:
            from genesis.analysis.fundamental_analysis import FundamentalAnalysisManager
            self.fundamental_analysis = FundamentalAnalysisManager()
            modules_status["fundamental_analysis"] = True
            self.available_modules["fundamental_analysis"] = "FundamentalAnalysisManager"
            logger.info("Módulo de análisis fundamental inicializado correctamente")
        except ImportError:
            self.fundamental_analysis = None
            modules_status["fundamental_analysis"] = False
            logger.warning("Módulo de análisis fundamental no disponible")
        
        # Intentar cargar módulos de análisis de sentimiento
        try:
            from genesis.analysis.sentiment_analysis import SentimentAnalysisManager
            self.sentiment_analysis = SentimentAnalysisManager()
            modules_status["sentiment_analysis"] = True
            self.available_modules["sentiment_analysis"] = "SentimentAnalysisManager"
            logger.info("Módulo de análisis de sentimiento inicializado correctamente")
        except ImportError:
            self.sentiment_analysis = None
            modules_status["sentiment_analysis"] = False
            logger.warning("Módulo de análisis de sentimiento no disponible")
            
        # Actualizar timestamp
        self.last_update = datetime.datetime.now()
        
        return modules_status
    
    async def analyze_market(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Realizar análisis completo de mercado.
        
        Args:
            symbols: Lista de símbolos para analizar específicamente
            
        Returns:
            Resultados del análisis de mercado
        """
        result = {
            "timestamp": datetime.datetime.now().isoformat(),
            "technical": None,
            "fundamental": None,
            "sentiment": None,
            "summary": {
                "overall_sentiment": "neutral",
                "risk_level": "medium",
                "recommendation": "hold"
            }
        }
        
        # Análisis técnico
        if self.technical_analysis:
            try:
                if hasattr(self.technical_analysis, 'analyze_market'):
                    result["technical"] = await self._run_async_or_sync(
                        self.technical_analysis.analyze_market, symbols=symbols
                    )
            except Exception as e:
                logger.error(f"Error en análisis técnico: {e}")
        
        # Análisis fundamental
        if self.fundamental_analysis:
            try:
                if hasattr(self.fundamental_analysis, 'analyze_market'):
                    result["fundamental"] = await self._run_async_or_sync(
                        self.fundamental_analysis.analyze_market, symbols=symbols
                    )
            except Exception as e:
                logger.error(f"Error en análisis fundamental: {e}")
        
        # Análisis de sentimiento
        if self.sentiment_analysis:
            try:
                if hasattr(self.sentiment_analysis, 'analyze_market_sentiment'):
                    result["sentiment"] = await self._run_async_or_sync(
                        self.sentiment_analysis.analyze_market_sentiment, symbols=symbols
                    )
            except Exception as e:
                logger.error(f"Error en análisis de sentimiento: {e}")
        
        # Actualizar caché
        self.analysis_cache["market"] = result
        self.last_update = datetime.datetime.now()
        
        return result
    
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Realizar análisis completo de un símbolo específico.
        
        Args:
            symbol: Símbolo a analizar
            
        Returns:
            Resultados del análisis
        """
        result = {
            "symbol": symbol,
            "timestamp": datetime.datetime.now().isoformat(),
            "technical": None,
            "fundamental": None,
            "sentiment": None,
            "summary": {
                "overall_sentiment": "neutral",
                "risk_level": "medium",
                "recommendation": "hold"
            }
        }
        
        # Análisis técnico
        if self.technical_analysis:
            try:
                if hasattr(self.technical_analysis, 'analyze_symbol'):
                    result["technical"] = await self._run_async_or_sync(
                        self.technical_analysis.analyze_symbol, symbol=symbol
                    )
            except Exception as e:
                logger.error(f"Error en análisis técnico para {symbol}: {e}")
        
        # Análisis fundamental
        if self.fundamental_analysis:
            try:
                if hasattr(self.fundamental_analysis, 'analyze_symbol'):
                    result["fundamental"] = await self._run_async_or_sync(
                        self.fundamental_analysis.analyze_symbol, symbol=symbol
                    )
            except Exception as e:
                logger.error(f"Error en análisis fundamental para {symbol}: {e}")
        
        # Análisis de sentimiento
        if self.sentiment_analysis:
            try:
                if hasattr(self.sentiment_analysis, 'analyze_symbol_sentiment'):
                    result["sentiment"] = await self._run_async_or_sync(
                        self.sentiment_analysis.analyze_symbol_sentiment, symbol=symbol
                    )
            except Exception as e:
                logger.error(f"Error en análisis de sentimiento para {symbol}: {e}")
        
        # Actualizar caché
        if "symbols" not in self.analysis_cache:
            self.analysis_cache["symbols"] = {}
        self.analysis_cache["symbols"][symbol] = result
        
        return result
    
    async def get_insight(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Obtener insight basado en análisis actual.
        
        Args:
            context: Contexto adicional para el insight
            
        Returns:
            Insight generado
        """
        insight = {
            "timestamp": datetime.datetime.now().isoformat(),
            "insight_text": "No hay suficiente información para generar un insight.",
            "confidence": 0.0,
            "source_modules": []
        }
        
        # Verificar si tenemos suficientes módulos
        if not self.available_modules:
            return insight
        
        # Integrar información de diferentes módulos
        try:
            # Generar insight técnico si está disponible
            if self.technical_analysis and hasattr(self.technical_analysis, 'generate_insight'):
                technical_insight = await self._run_async_or_sync(
                    self.technical_analysis.generate_insight, context=context
                )
                
                if technical_insight:
                    insight["technical_insight"] = technical_insight
                    insight["source_modules"].append("technical_analysis")
                    
                    # Usar el insight técnico si no tenemos otro mejor
                    if insight["confidence"] < technical_insight.get("confidence", 0):
                        insight["insight_text"] = technical_insight.get("text", "")
                        insight["confidence"] = technical_insight.get("confidence", 0)
            
            # Generar insight fundamental si está disponible
            if self.fundamental_analysis and hasattr(self.fundamental_analysis, 'generate_insight'):
                fundamental_insight = await self._run_async_or_sync(
                    self.fundamental_analysis.generate_insight, context=context
                )
                
                if fundamental_insight:
                    insight["fundamental_insight"] = fundamental_insight
                    insight["source_modules"].append("fundamental_analysis")
                    
                    # Usar el insight fundamental si tiene mayor confianza
                    if insight["confidence"] < fundamental_insight.get("confidence", 0):
                        insight["insight_text"] = fundamental_insight.get("text", "")
                        insight["confidence"] = fundamental_insight.get("confidence", 0)
            
            # Generar insight de sentimiento si está disponible
            if self.sentiment_analysis and hasattr(self.sentiment_analysis, 'generate_insight'):
                sentiment_insight = await self._run_async_or_sync(
                    self.sentiment_analysis.generate_insight, context=context
                )
                
                if sentiment_insight:
                    insight["sentiment_insight"] = sentiment_insight
                    insight["source_modules"].append("sentiment_analysis")
                    
                    # Usar el insight de sentimiento si tiene mayor confianza
                    if insight["confidence"] < sentiment_insight.get("confidence", 0):
                        insight["insight_text"] = sentiment_insight.get("text", "")
                        insight["confidence"] = sentiment_insight.get("confidence", 0)
            
        except Exception as e:
            logger.error(f"Error al generar insight: {e}")
            
        return insight
    
    async def _run_async_or_sync(self, func, **kwargs):
        """
        Ejecutar función de forma asíncrona o síncrona según su naturaleza.
        
        Args:
            func: Función a ejecutar
            **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
        """
        if asyncio.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            return func(**kwargs)
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado del integrador.
        
        Returns:
            Estado actual
        """
        return {
            "available_modules": self.available_modules,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "cached_analyses": list(self.analysis_cache.keys()),
            "cached_symbols": list(self.analysis_cache.get("symbols", {}).keys()) if "symbols" in self.analysis_cache else []
        }

# Crear instancia global
integrator = AnalysisIntegrator()

def get_analysis_integrator() -> AnalysisIntegrator:
    """
    Obtener instancia del integrador.
    
    Returns:
        Instancia del integrador
    """
    return integrator