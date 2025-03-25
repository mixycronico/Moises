"""
Integration Manager para Aetherion

Este módulo gestiona las integraciones entre Aetherion y los demás componentes
de inteligencia artificial del Sistema Genesis (Gabriel, Buddha, DeepSeek).

Permite la comunicación bidireccional y la integración de capacidades entre
todos los sistemas para crear una experiencia unificada.

Autor: Genesis AI Assistant
Versión: 1.0.0
"""

import logging
import asyncio
import importlib
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum, auto

# Configuración de logging
logger = logging.getLogger("genesis.consciousness.integrations")

class IntegrationType(Enum):
    """Tipos de integración soportados."""
    BEHAVIOR = auto()  # Comportamiento humano (Gabriel)
    ANALYSIS = auto()  # Análisis superior (Buddha)
    LANGUAGE = auto()  # Procesamiento de lenguaje (DeepSeek)
    TRADING = auto()   # Trading y operaciones
    MEMORY = auto()    # Memoria y persistencia
    OTHER = auto()     # Otros tipos de integración

class IntegrationManager:
    """
    Gestor de integraciones para Aetherion.
    
    Esta clase coordina la comunicación entre Aetherion y los demás 
    componentes de IA del Sistema Genesis.
    
    Atributos:
        integrations: Diccionario de integraciones registradas
        initialized: Estado de inicialización
    """
    
    def __init__(self):
        """Inicializar gestor de integraciones."""
        self.integrations = {}
        self.initialized = False
        logger.info("IntegrationManager inicializado")
    
    async def initialize(self) -> bool:
        """
        Inicializar todas las integraciones.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Descubrir e inicializar integraciones disponibles
            await self._discover_integrations()
            
            self.initialized = True
            logger.info(f"IntegrationManager inicializado completamente. Integraciones: {len(self.integrations)}")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar IntegrationManager: {str(e)}")
            return False
    
    async def _discover_integrations(self) -> None:
        """Descubrir integraciones disponibles en el sistema."""
        # Integración con Gabriel (Comportamiento humano)
        await self._try_register_gabriel()
        
        # Integración con Buddha (Análisis superior)
        await self._try_register_buddha()
        
        # Integración con DeepSeek (LLM)
        await self._try_register_deepseek()
    
    async def _try_register_gabriel(self) -> None:
        """Intentar registrar integración con Gabriel."""
        try:
            # Importar el módulo necesario
            from genesis.behavior.gabriel_engine import GabrielBehaviorEngine
            
            # Crear integración
            integration = {
                "name": "gabriel",
                "type": IntegrationType.BEHAVIOR,
                "module": GabrielBehaviorEngine,
                "instance": None,
                "available": True,
                "capabilities": [
                    "emotional_state",
                    "human_behavior",
                    "risk_assessment",
                    "trade_evaluation"
                ]
            }
            
            # Registrar integración
            self.integrations["gabriel"] = integration
            logger.info("Integración con Gabriel registrada correctamente")
            
        except ImportError:
            logger.warning("Módulo Gabriel no encontrado, no se registrará integración")
            self.integrations["gabriel"] = {
                "name": "gabriel",
                "type": IntegrationType.BEHAVIOR,
                "available": False,
                "error": "Módulo no encontrado"
            }
    
    async def _try_register_buddha(self) -> None:
        """Intentar registrar integración con Buddha."""
        try:
            # Importar el módulo necesario
            from genesis.trading.buddha_integrator import BuddhaIntegrator
            
            # Crear integración
            integration = {
                "name": "buddha",
                "type": IntegrationType.ANALYSIS,
                "module": BuddhaIntegrator,
                "instance": None,
                "available": True,
                "capabilities": [
                    "market_analysis",
                    "sentiment_analysis",
                    "opportunity_detection",
                    "risk_assessment"
                ]
            }
            
            # Registrar integración
            self.integrations["buddha"] = integration
            logger.info("Integración con Buddha registrada correctamente")
            
        except ImportError:
            logger.warning("Módulo Buddha no encontrado, no se registrará integración")
            self.integrations["buddha"] = {
                "name": "buddha",
                "type": IntegrationType.ANALYSIS,
                "available": False,
                "error": "Módulo no encontrado"
            }
    
    async def _try_register_deepseek(self) -> None:
        """Intentar registrar integración con DeepSeek."""
        try:
            # Importar el módulo necesario
            from genesis.lsml.deepseek_integrator import DeepSeekIntegrator
            
            # Crear integración
            integration = {
                "name": "deepseek",
                "type": IntegrationType.LANGUAGE,
                "module": DeepSeekIntegrator,
                "instance": None,
                "available": True,
                "capabilities": [
                    "text_generation",
                    "financial_analysis",
                    "strategic_planning",
                    "intent_detection"
                ]
            }
            
            # Registrar integración
            self.integrations["deepseek"] = integration
            logger.info("Integración con DeepSeek registrada correctamente")
            
        except ImportError:
            logger.warning("Módulo DeepSeek no encontrado, no se registrará integración")
            self.integrations["deepseek"] = {
                "name": "deepseek",
                "type": IntegrationType.LANGUAGE,
                "available": False,
                "error": "Módulo no encontrado"
            }
    
    async def get_integration(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Obtener integración por nombre.
        
        Args:
            name: Nombre de la integración
            
        Returns:
            Información de la integración, o None si no existe
        """
        return self.integrations.get(name)
    
    async def get_integration_instance(self, name: str) -> Optional[Any]:
        """
        Obtener instancia de integración, inicializándola si es necesario.
        
        Args:
            name: Nombre de la integración
            
        Returns:
            Instancia de la integración, o None si no está disponible
        """
        integration = self.integrations.get(name)
        if not integration or not integration.get("available", False):
            return None
        
        # Si no hay instancia, crearla
        if not integration.get("instance"):
            module_class = integration.get("module")
            if module_class:
                try:
                    instance = module_class()
                    if hasattr(instance, "initialize"):
                        await instance.initialize()
                    integration["instance"] = instance
                except Exception as e:
                    logger.error(f"Error al crear instancia de {name}: {str(e)}")
                    return None
        
        return integration.get("instance")
    
    async def get_integrations_by_type(self, integration_type: IntegrationType) -> List[Dict[str, Any]]:
        """
        Obtener integraciones por tipo.
        
        Args:
            integration_type: Tipo de integración
            
        Returns:
            Lista de integraciones del tipo especificado
        """
        return [
            integration for integration in self.integrations.values()
            if integration.get("type") == integration_type and integration.get("available", False)
        ]
    
    async def get_integrations_with_capability(self, capability: str) -> List[Dict[str, Any]]:
        """
        Obtener integraciones que tienen una capacidad específica.
        
        Args:
            capability: Capacidad requerida
            
        Returns:
            Lista de integraciones con la capacidad
        """
        return [
            integration for integration in self.integrations.values()
            if capability in integration.get("capabilities", []) and integration.get("available", False)
        ]
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado de todas las integraciones.
        
        Returns:
            Estado de las integraciones
        """
        return {
            "total_integrations": len(self.integrations),
            "available_integrations": sum(1 for i in self.integrations.values() if i.get("available", False)),
            "integrations": {
                name: {
                    "available": integration.get("available", False),
                    "type": integration.get("type").name if integration.get("type") else None,
                    "capabilities": integration.get("capabilities", []) if integration.get("available", False) else []
                }
                for name, integration in self.integrations.items()
            }
        }

# Instancia global para acceso sencillo
_integration_manager_instance = None

async def get_integration_manager() -> IntegrationManager:
    """
    Obtener instancia global del gestor de integraciones.
    
    Returns:
        Instancia inicializada del gestor de integraciones
    """
    global _integration_manager_instance
    if _integration_manager_instance is None:
        _integration_manager_instance = IntegrationManager()
        await _integration_manager_instance.initialize()
    return _integration_manager_instance