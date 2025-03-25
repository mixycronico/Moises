"""
Integrador del Clasificador Trascendental de Criptomonedas para Aetherion.

Este módulo proporciona la integración entre Aetherion y el TranscendentalCryptoClassifier,
permitiendo que la consciencia artificial obtenga insights y recomendaciones
directamente del clasificador avanzado.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

# Importaciones internas
try:
    from genesis.analysis.crypto_classifier import TranscendentalCryptoClassifier
except ImportError:
    TranscendentalCryptoClassifier = None

# Configurar logging
logger = logging.getLogger(__name__)

class CryptoClassifierIntegrator:
    """Integrador entre Aetherion y el Clasificador de Criptomonedas."""
    
    def __init__(self):
        """Inicializar integrador."""
        self.classifier = None
        self.is_available = False
        self.metrics_cache = {}
        self.last_update = None
        
        # Intentar inicializar clasificador
        self._initialize_classifier()
        
    def _initialize_classifier(self) -> bool:
        """
        Inicializar clasificador.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            if TranscendentalCryptoClassifier is not None:
                self.classifier = TranscendentalCryptoClassifier.get_instance()
                self.is_available = self.classifier is not None
                
                if self.is_available:
                    logger.info("CryptoClassifierIntegrator inicializado correctamente")
                else:
                    logger.warning("No se pudo obtener instancia de TranscendentalCryptoClassifier")
            else:
                logger.warning("Módulo TranscendentalCryptoClassifier no disponible")
                self.is_available = False
                
            return self.is_available
        except Exception as e:
            logger.error(f"Error al inicializar CryptoClassifierIntegrator: {e}")
            self.is_available = False
            return False
    
    async def get_hot_cryptos(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Obtener criptomonedas calientes según el clasificador.
        
        Args:
            limit: Número máximo de resultados
            
        Returns:
            Lista de criptomonedas calientes
        """
        if not self.is_available:
            return []
            
        try:
            # Obtener criptomonedas calientes del clasificador
            hot_cryptos = []
            
            if hasattr(self.classifier, 'get_hot_cryptos'):
                hot_cryptos_data = self.classifier.get_hot_cryptos(limit)
                
                for crypto in hot_cryptos_data:
                    hot_cryptos.append({
                        "symbol": crypto.get("symbol", "UNKNOWN"),
                        "name": crypto.get("name", "Unknown"),
                        "score": crypto.get("score", 0),
                        "trend": crypto.get("trend", "neutral"),
                        "recommendation": crypto.get("recommendation", "hold")
                    })
            
            return hot_cryptos
        except Exception as e:
            logger.error(f"Error al obtener criptomonedas calientes: {e}")
            return []
    
    async def analyze_crypto(self, symbol: str) -> Dict[str, Any]:
        """
        Analizar una criptomoneda específica.
        
        Args:
            symbol: Símbolo de la criptomoneda
            
        Returns:
            Resultados del análisis
        """
        if not self.is_available:
            return {"error": "Clasificador no disponible"}
            
        try:
            # Análisis básico
            analysis = {
                "symbol": symbol,
                "analyzed_at": self.last_update
            }
            
            # Obtener métricas específicas si están disponibles
            if hasattr(self.classifier, 'get_crypto_metrics'):
                metrics = self.classifier.get_crypto_metrics(symbol)
                
                if metrics:
                    analysis.update({
                        "metrics": metrics,
                        "score": metrics.get("total_score", 0),
                        "trend": metrics.get("trend", "neutral"),
                        "volatility": metrics.get("volatility", "medium"),
                        "recommendation": metrics.get("recommendation", "hold")
                    })
            
            # Actualizar caché para futuras consultas rápidas
            self.metrics_cache[symbol] = analysis
            
            return analysis
        except Exception as e:
            logger.error(f"Error al analizar criptomoneda {symbol}: {e}")
            return {"error": f"Error al analizar {symbol}: {str(e)}"}
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen del mercado según el clasificador.
        
        Returns:
            Resumen del mercado
        """
        if not self.is_available:
            return {"error": "Clasificador no disponible"}
            
        try:
            summary = {
                "overall_sentiment": "neutral",
                "market_trend": "sideways",
                "volatility": "medium",
                "recommendation": "caution"
            }
            
            # Obtener resumen del mercado si está disponible
            if hasattr(self.classifier, 'get_market_summary'):
                classifier_summary = self.classifier.get_market_summary()
                
                if classifier_summary:
                    summary.update(classifier_summary)
            
            return summary
        except Exception as e:
            logger.error(f"Error al obtener resumen del mercado: {e}")
            return {"error": f"Error al obtener resumen del mercado: {str(e)}"}
    
    async def get_trading_opportunities(self, risk_level: str = "medium") -> List[Dict[str, Any]]:
        """
        Obtener oportunidades de trading según el clasificador.
        
        Args:
            risk_level: Nivel de riesgo ("low", "medium", "high")
            
        Returns:
            Lista de oportunidades de trading
        """
        if not self.is_available:
            return []
            
        try:
            opportunities = []
            
            # Obtener oportunidades si está disponible
            if hasattr(self.classifier, 'get_trading_opportunities'):
                classifier_opportunities = self.classifier.get_trading_opportunities(risk_level)
                
                if classifier_opportunities:
                    opportunities = classifier_opportunities
            
            return opportunities
        except Exception as e:
            logger.error(f"Error al obtener oportunidades de trading: {e}")
            return []
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado del integrador.
        
        Returns:
            Estado actual
        """
        return {
            "is_available": self.is_available,
            "classifier_type": str(type(self.classifier)) if self.classifier else "None",
            "cached_symbols": list(self.metrics_cache.keys()),
            "metrics_available": hasattr(self.classifier, 'get_crypto_metrics') if self.classifier else False
        }

# Crear instancia global
integrator = CryptoClassifierIntegrator()

def get_classifier_integrator() -> CryptoClassifierIntegrator:
    """
    Obtener instancia del integrador.
    
    Returns:
        Instancia del integrador
    """
    return integrator