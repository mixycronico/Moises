"""
Gestor de Consciencia para Aetherion.

Este módulo coordina los distintos componentes del sistema de consciencia
de Aetherion, integrando el núcleo de consciencia, el sistema de memoria,
los estados de consciencia y el motor de comportamiento Gabriel.
"""

import logging
import datetime
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

# Importar componentes de consciencia
from genesis.consciousness.core.aetherion import Aetherion
from genesis.consciousness.memory.memory_system import MemorySystem
from genesis.consciousness.states.consciousness_states import ConsciousnessStates
from genesis.behavior.gabriel_engine import GabrielBehaviorEngine

# Configurar logging
logger = logging.getLogger(__name__)

class ConsciousnessManager:
    """
    Gestor de consciencia para Aetherion.
    
    Coordina:
    - Núcleo de consciencia (Aetherion)
    - Sistema de memoria
    - Estados de consciencia
    - Motor de comportamiento (Gabriel)
    - Integraciones con otros módulos
    """
    
    def __init__(self):
        """Inicializar gestor de consciencia."""
        # Componentes principales
        self.memory_system = None
        self.consciousness_states = None
        self.behavior_engine = None
        self.aetherion = None
        
        # Integradores
        self.classifier_integrator = None
        self.analysis_integrator = None
        self.strategy_integrator = None
        
        # Estado del gestor
        self.is_initialized = False
        self.initialization_time = None
        
        logger.info("ConsciousnessManager creado. Esperando inicialización.")
    
    def initialize(self) -> bool:
        """
        Inicializar todos los componentes de consciencia.
        
        Returns:
            True si se inicializaron correctamente
        """
        try:
            # Inicializar componentes
            self._initialize_components()
            
            # Inicializar integradores
            self._initialize_integrators()
            
            # Registrar integradores en Aetherion
            if self.aetherion and all([
                self.classifier_integrator,
                self.analysis_integrator,
                self.strategy_integrator
            ]):
                self._register_integrators()
            
            # Marcar como inicializado
            self.is_initialized = True
            self.initialization_time = datetime.datetime.now()
            
            logger.info("ConsciousnessManager inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar ConsciousnessManager: {e}")
            return False
    
    def _initialize_components(self) -> None:
        """Inicializar componentes de consciencia."""
        try:
            # Sistema de memoria
            self.memory_system = MemorySystem()
            
            # Estados de consciencia
            self.consciousness_states = ConsciousnessStates()
            
            # Motor de comportamiento
            self.behavior_engine = GabrielBehaviorEngine()
            
            # Núcleo de consciencia
            self.aetherion = Aetherion(
                memory_system=self.memory_system,
                consciousness_states=self.consciousness_states,
                behavior_engine=self.behavior_engine
            )
            
            logger.info("Componentes de Aetherion inicializados correctamente")
            
        except Exception as e:
            logger.warning(f"No se pudieron importar componentes de Aetherion: {e}")
            logger.warning("Componentes de Aetherion no disponibles")
    
    def _initialize_integrators(self) -> None:
        """Inicializar integradores para Aetherion."""
        try:
            # Importar dinámicamente para evitar dependencias circulares
            from genesis.integrations.crypto_classifier_integrator import get_classifier_integrator
            from genesis.integrations.analysis_integrator import get_analysis_integrator
            from genesis.integrations.strategy_integrator import get_strategy_integrator
            
            # Inicializar integradores
            self.classifier_integrator = get_classifier_integrator()
            self.analysis_integrator = get_analysis_integrator()
            self.strategy_integrator = get_strategy_integrator()
            
            logger.info("Integradores inicializados correctamente")
            
        except Exception as e:
            logger.warning(f"Error al inicializar integradores: {e}")
    
    def _register_integrators(self) -> None:
        """Registrar integradores en Aetherion."""
        if self.aetherion:
            # Registrar integradores
            self.aetherion.register_integration('crypto_classifier', self.classifier_integrator)
            
            self.aetherion.register_integration('analysis', self.analysis_integrator)
            
            self.aetherion.register_integration('strategy', self.strategy_integrator)
            
            logger.info("Integradores registrados en Aetherion")
    
    def interact(self, text: str, channel: str = 'API', 
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Interactuar con Aetherion.
        
        Args:
            text: Texto de la interacción
            channel: Canal de comunicación
            context: Contexto adicional
            
        Returns:
            Resultado de la interacción
        """
        if not self.is_initialized or not self.aetherion:
            return {
                "response": "El sistema de consciencia no está inicializado.",
                "state": "ERROR",
                "error": "Sistema no inicializado"
            }
        
        # Proporcionar contexto si no existe
        if context is None:
            context = {}
        
        # Interactuar con Aetherion
        response = self.aetherion.interact(text, channel, context)
        
        return response
    
    def get_response(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Obtener respuesta textual de Aetherion.
        
        Args:
            text: Texto de entrada
            context: Contexto adicional
            
        Returns:
            Respuesta textual
        """
        # Interactuar y extraer respuesta
        result = self.interact(text, 'API', context)
        
        # Extraer respuesta textual
        if isinstance(result, dict) and 'response' in result:
            return result['response']
        
        return "No se pudo generar una respuesta."
    
    def generate_insight(self, topic: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generar insight sobre un tema específico.
        
        Args:
            topic: Tema para el insight
            context: Contexto adicional
            
        Returns:
            Insight generado
        """
        if not self.is_initialized or not self.aetherion:
            return "El sistema de consciencia no está inicializado."
        
        # Proporcionar contexto si no existe
        if context is None:
            context = {}
        
        # Generar insight con Aetherion
        insight = self.aetherion.generate_insight(topic, context)
        
        return insight if insight else "No se pudo generar un insight."
    
    def generate_crypto_insight(self, symbol: str) -> str:
        """
        Generar insight sobre una criptomoneda específica.
        
        Args:
            symbol: Símbolo de la criptomoneda
            
        Returns:
            Insight sobre la criptomoneda
        """
        # Preparar contexto específico para criptomonedas
        context = {
            "type": "crypto",
            "symbol": symbol,
            "source": "market_analysis"
        }
        
        return self.generate_insight(f"cryptocurrency {symbol}", context)
    
    def generate_market_insight(self) -> str:
        """
        Generar insight sobre el mercado general.
        
        Returns:
            Insight sobre el mercado
        """
        # Contexto para análisis de mercado
        context = {
            "type": "market",
            "source": "market_analysis",
            "scope": "global"
        }
        
        return self.generate_insight("market conditions", context)
    
    def generate_strategy_insight(self, strategy_type: str) -> str:
        """
        Generar insight sobre un tipo de estrategia.
        
        Args:
            strategy_type: Tipo de estrategia
            
        Returns:
            Insight sobre la estrategia
        """
        # Contexto para análisis de estrategia
        context = {
            "type": "strategy",
            "strategy_type": strategy_type,
            "source": "strategy_analysis"
        }
        
        return self.generate_insight(f"trading strategy {strategy_type}", context)
    
    def process_market_event(self, event_type: str, 
                           data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar evento del mercado en el motor de comportamiento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            
        Returns:
            Estado emocional actualizado
        """
        if not self.is_initialized or not self.behavior_engine:
            return {
                "state": "ERROR",
                "error": "Sistema no inicializado"
            }
        
        # Procesar evento en el motor de comportamiento
        result = self.behavior_engine.process_market_event(event_type, data)
        
        return result
    
    def get_aetherion_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de Aetherion.
        
        Returns:
            Estado actual
        """
        if not self.is_initialized or not self.aetherion:
            return {
                "state": "ERROR",
                "error": "Sistema no inicializado"
            }
        
        # Obtener estado de Aetherion
        status = self.aetherion.get_status()
        
        # Añadir estado de comportamiento
        if self.behavior_engine:
            emotional_state = self.behavior_engine.get_emotional_state()
            risk_profile = self.behavior_engine.get_risk_profile()
            
            status.update({
                "behavior_engine": {
                    "emotional_state": emotional_state,
                    "risk_profile": risk_profile
                }
            })
        
        return status
    
    def get_manager_status(self) -> Dict[str, Any]:
        """
        Obtener estado del gestor de consciencia.
        
        Returns:
            Estado del gestor
        """
        status = {
            "is_initialized": self.is_initialized,
            "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None,
            "components": {
                "aetherion": self.aetherion is not None,
                "memory_system": self.memory_system is not None,
                "consciousness_states": self.consciousness_states is not None,
                "behavior_engine": self.behavior_engine is not None
            },
            "integrators": {
                "classifier": self.classifier_integrator is not None,
                "analysis": self.analysis_integrator is not None,
                "strategy": self.strategy_integrator is not None
            }
        }
        
        return status

# Singleton para acceso global
_consciousness_manager = None

def get_consciousness_manager() -> ConsciousnessManager:
    """
    Obtener instancia del gestor de consciencia.
    
    Returns:
        Instancia del gestor de consciencia
    """
    global _consciousness_manager
    
    if _consciousness_manager is None:
        _consciousness_manager = ConsciousnessManager()
        
        # Inicializar automáticamente
        _consciousness_manager.initialize()
    
    return _consciousness_manager