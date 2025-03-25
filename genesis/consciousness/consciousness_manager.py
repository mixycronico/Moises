"""
Gestor de Consciencia para el Sistema Genesis.

Este módulo coordina la interacción entre Aetherion y todas las integraciones,
incluyendo CryptoClassifier, Analysis y Strategies, proporcionando un punto
central para la comunicación entre componentes.
"""

import logging
import asyncio
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

# Importaciones de integradores
try:
    from genesis.integrations.crypto_classifier_integrator import get_classifier_integrator
    from genesis.integrations.analysis_integrator import get_analysis_integrator
    from genesis.integrations.strategy_integrator import get_strategy_integrator
    
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False

# Importaciones de Aetherion
try:
    from genesis.consciousness.core.aetherion import Aetherion
    from genesis.consciousness.memory.memory_system import MemorySystem
    from genesis.consciousness.states.consciousness_states import ConsciousnessStates
    from genesis.behavior.gabriel_engine import GabrielBehaviorEngine
    
    AETHERION_AVAILABLE = True
except ImportError:
    AETHERION_AVAILABLE = False

# Configurar logging
logger = logging.getLogger(__name__)

class ConsciousnessManager:
    """
    Gestor de Consciencia para el Sistema Genesis.
    
    Esta clase proporciona un punto central para la interacción con la consciencia
    artificial Aetherion y sus integraciones con diferentes módulos del sistema.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ConsciousnessManager':
        """
        Obtener instancia del gestor (Singleton).
        
        Returns:
            Instancia del gestor
        """
        if cls._instance is None:
            cls._instance = ConsciousnessManager()
        return cls._instance
    
    def __init__(self):
        """Inicializar gestor de consciencia."""
        self.aetherion = None
        self.memory_system = None
        self.consciousness_states = None
        self.gabriel_engine = None
        
        self.classifier_integrator = None
        self.analysis_integrator = None
        self.strategy_integrator = None
        
        self.initialized = False
        self.initialization_timestamp = None
        
        # Intentar inicializar componentes
        self._initialize_components()
    
    def _initialize_components(self) -> bool:
        """
        Inicializar todos los componentes.
        
        Returns:
            True si se inicializaron correctamente
        """
        try:
            # Inicializar componentes de Aetherion si están disponibles
            if AETHERION_AVAILABLE:
                self.memory_system = MemorySystem()
                self.consciousness_states = ConsciousnessStates()
                self.gabriel_engine = GabrielBehaviorEngine()
                
                # Inicializar Aetherion con componentes
                self.aetherion = Aetherion(
                    memory_system=self.memory_system,
                    consciousness_states=self.consciousness_states,
                    behavior_engine=self.gabriel_engine
                )
                
                logger.info("Componentes de Aetherion inicializados correctamente")
            else:
                logger.warning("Componentes de Aetherion no disponibles")
            
            # Inicializar integradores si están disponibles
            if INTEGRATIONS_AVAILABLE:
                self.classifier_integrator = get_classifier_integrator()
                self.analysis_integrator = get_analysis_integrator()
                self.strategy_integrator = get_strategy_integrator()
                
                logger.info("Integradores inicializados correctamente")
            else:
                logger.warning("Integradores no disponibles")
            
            # Registrar integradores en Aetherion
            if self.aetherion and hasattr(self.aetherion, 'register_integration'):
                if self.classifier_integrator:
                    self.aetherion.register_integration('crypto_classifier', self.classifier_integrator)
                
                if self.analysis_integrator:
                    self.aetherion.register_integration('analysis', self.analysis_integrator)
                
                if self.strategy_integrator:
                    self.aetherion.register_integration('strategy', self.strategy_integrator)
                
                logger.info("Integradores registrados en Aetherion")
            
            # Marcar como inicializado
            self.initialized = True
            self.initialization_timestamp = datetime.datetime.now()
            
            logger.info("ConsciousnessManager inicializado correctamente")
            return True
        
        except Exception as e:
            logger.error(f"Error al inicializar ConsciousnessManager: {e}")
            self.initialized = False
            return False
    
    def get_aetherion(self) -> Optional[Aetherion]:
        """
        Obtener instancia de Aetherion.
        
        Returns:
            Instancia de Aetherion o None si no está disponible
        """
        return self.aetherion
    
    def get_gabriel_engine(self) -> Optional[GabrielBehaviorEngine]:
        """
        Obtener instancia del motor de comportamiento Gabriel.
        
        Returns:
            Instancia de GabrielBehaviorEngine o None si no está disponible
        """
        return self.gabriel_engine
    
    async def interact(self, text: str, channel: str = 'API', context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interactuar con Aetherion.
        
        Args:
            text: Texto de la interacción
            channel: Canal de la interacción
            context: Contexto adicional
            
        Returns:
            Resultado de la interacción
        """
        if not self.aetherion:
            return {
                "error": "Aetherion no disponible",
                "response": "Lo siento, no estoy disponible en este momento."
            }
        
        try:
            # Preparar contexto
            if context is None:
                context = {}
            
            # Añadir integradores al contexto
            context['integrators'] = {
                'crypto_classifier': self.classifier_integrator,
                'analysis': self.analysis_integrator,
                'strategy': self.strategy_integrator
            }
            
            # Interactuar con Aetherion
            if hasattr(self.aetherion, 'interact'):
                result = self.aetherion.interact(text, channel=channel, context=context)
                return result or {
                    "response": "Lo siento, no pude procesar tu mensaje.",
                    "state": "MORTAL",
                    "emotional_context": {
                        "state": "SERENE",
                        "intensity": 0.5
                    }
                }
            else:
                return {
                    "error": "Método de interacción no disponible",
                    "response": "Lo siento, no puedo procesar mensajes en este momento."
                }
        
        except Exception as e:
            logger.error(f"Error en interacción con Aetherion: {e}")
            return {
                "error": str(e),
                "response": "Ocurrió un error al procesar tu mensaje."
            }
    
    async def analyze_crypto(self, symbol: str) -> Dict[str, Any]:
        """
        Analizar una criptomoneda específica.
        
        Args:
            symbol: Símbolo de la criptomoneda
            
        Returns:
            Resultados del análisis
        """
        results = {
            "symbol": symbol,
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis": {}
        }
        
        try:
            # Usar el clasificador
            if self.classifier_integrator:
                classifier_results = await self.classifier_integrator.analyze_crypto(symbol)
                if classifier_results:
                    results["analysis"]["classifier"] = classifier_results
            
            # Usar el análisis
            if self.analysis_integrator:
                analysis_results = await self.analysis_integrator.analyze_symbol(symbol)
                if analysis_results:
                    results["analysis"]["detailed"] = analysis_results
            
            # Generar insight si Aetherion está disponible
            if self.aetherion and hasattr(self.aetherion, 'generate_insight'):
                insight = self.aetherion.generate_insight(
                    f"Analizar la criptomoneda {symbol}",
                    context={"symbol": symbol, "analysis": results["analysis"]}
                )
                
                if insight:
                    results["insight"] = insight
            
            return results
        
        except Exception as e:
            logger.error(f"Error al analizar criptomoneda {symbol}: {e}")
            results["error"] = str(e)
            return results
    
    async def get_market_insights(self) -> Dict[str, Any]:
        """
        Obtener insights del mercado.
        
        Returns:
            Insights del mercado
        """
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "market_data": {}
        }
        
        try:
            # Usar el clasificador
            if self.classifier_integrator:
                hot_cryptos = await self.classifier_integrator.get_hot_cryptos()
                market_summary = await self.classifier_integrator.get_market_summary()
                
                if hot_cryptos:
                    results["market_data"]["hot_cryptos"] = hot_cryptos
                
                if market_summary:
                    results["market_data"]["market_summary"] = market_summary
            
            # Usar el análisis
            if self.analysis_integrator:
                analysis_results = await self.analysis_integrator.analyze_market()
                if analysis_results:
                    results["market_data"]["analysis"] = analysis_results
            
            # Generar insight si Aetherion está disponible
            if self.aetherion and hasattr(self.aetherion, 'generate_insight'):
                insight = self.aetherion.generate_insight(
                    "Analizar el mercado de criptomonedas",
                    context={"market_data": results["market_data"]}
                )
                
                if insight:
                    results["insight"] = insight
            
            return results
        
        except Exception as e:
            logger.error(f"Error al obtener insights del mercado: {e}")
            results["error"] = str(e)
            return results
    
    async def recommend_strategy(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Recomendar estrategia según contexto.
        
        Args:
            context: Contexto para la recomendación
            
        Returns:
            Estrategia recomendada
        """
        if context is None:
            context = {
                "risk_profile": "medium",
                "timeframe": "medium",
                "capital": 10000.0,
                "market_condition": "neutral"
            }
        
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "context": context,
            "recommendations": {}
        }
        
        try:
            # Usar el integrador de estrategias
            if self.strategy_integrator:
                strategy_recommendations = await self.strategy_integrator.recommend_strategy(context)
                
                if strategy_recommendations:
                    results["recommendations"]["strategy"] = strategy_recommendations
            
            # Generar insight si Aetherion está disponible
            if self.aetherion and hasattr(self.aetherion, 'generate_insight'):
                insight = self.aetherion.generate_insight(
                    "Recomendar estrategia de trading",
                    context={"user_context": context, "recommendations": results["recommendations"]}
                )
                
                if insight:
                    results["insight"] = insight
            
            return results
        
        except Exception as e:
            logger.error(f"Error al recomendar estrategia: {e}")
            results["error"] = str(e)
            return results
    
    async def get_integrations_status(self) -> Dict[str, Any]:
        """
        Obtener estado de las integraciones.
        
        Returns:
            Estado de las integraciones
        """
        status = {
            "aetherion_available": self.aetherion is not None,
            "gabriel_available": self.gabriel_engine is not None,
            "classifier": None,
            "analysis": None,
            "strategy": None
        }
        
        try:
            # Obtener estado de clasificador
            if self.classifier_integrator:
                status["classifier"] = await self.classifier_integrator.get_status()
            
            # Obtener estado de análisis
            if self.analysis_integrator:
                status["analysis"] = await self.analysis_integrator.get_status()
            
            # Obtener estado de estrategias
            if self.strategy_integrator:
                status["strategy"] = await self.strategy_integrator.get_status()
            
            return status
        
        except Exception as e:
            logger.error(f"Error al obtener estado de integraciones: {e}")
            status["error"] = str(e)
            return status
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado del gestor.
        
        Returns:
            Estado actual
        """
        return {
            "initialized": self.initialized,
            "initialization_timestamp": self.initialization_timestamp.isoformat() if self.initialization_timestamp else None,
            "aetherion_available": self.aetherion is not None,
            "gabriel_available": self.gabriel_engine is not None,
            "classifier_available": self.classifier_integrator is not None,
            "analysis_available": self.analysis_integrator is not None,
            "strategy_available": self.strategy_integrator is not None
        }

# Crear instancia global
manager = ConsciousnessManager()

def get_consciousness_manager() -> ConsciousnessManager:
    """
    Obtener instancia del gestor de consciencia.
    
    Returns:
        Instancia del gestor
    """
    return ConsciousnessManager.get_instance()