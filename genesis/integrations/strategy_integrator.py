"""
Integrador de Estrategias para Aetherion.

Este módulo proporciona la integración entre Aetherion y las estrategias de trading
del Sistema Genesis, permitiendo que la consciencia artificial pueda recomendar
y evaluar diferentes estrategias según el contexto.
"""

import asyncio
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

# Configurar logging
logger = logging.getLogger(__name__)

class StrategyIntegrator:
    """Integrador entre Aetherion y las estrategias de trading."""
    
    def __init__(self):
        """Inicializar integrador."""
        self.available_strategies = {}
        self.strategy_manager = None
        self.last_evaluation = None
        self.cached_recommendations = {}
        
        # Intentar inicializar gestor de estrategias
        self._initialize_strategy_manager()
        
    def _initialize_strategy_manager(self) -> bool:
        """
        Inicializar gestor de estrategias.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Intentar importar gestor de estrategias
            try:
                from genesis.strategies.strategy_manager import StrategyManager
                self.strategy_manager = StrategyManager()
                logger.info("Gestor de estrategias inicializado correctamente")
                
                # Obtener estrategias disponibles
                if hasattr(self.strategy_manager, 'get_available_strategies'):
                    self.available_strategies = self.strategy_manager.get_available_strategies()
                    
                return True
            except ImportError:
                self.strategy_manager = None
                logger.warning("Gestor de estrategias no disponible")
                return False
                
        except Exception as e:
            logger.error(f"Error al inicializar gestor de estrategias: {e}")
            self.strategy_manager = None
            return False
    
    async def get_available_strategies(self) -> Dict[str, Any]:
        """
        Obtener estrategias disponibles.
        
        Returns:
            Diccionario con estrategias disponibles
        """
        if not self.strategy_manager:
            return {}
            
        try:
            if hasattr(self.strategy_manager, 'get_available_strategies'):
                # Actualizar caché de estrategias disponibles
                self.available_strategies = self.strategy_manager.get_available_strategies()
                
            return self.available_strategies
        except Exception as e:
            logger.error(f"Error al obtener estrategias disponibles: {e}")
            return {}
    
    async def evaluate_strategy(self, strategy_id: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluar una estrategia específica.
        
        Args:
            strategy_id: Identificador de la estrategia
            params: Parámetros de evaluación
            
        Returns:
            Resultados de la evaluación
        """
        if not self.strategy_manager:
            return {"error": "Gestor de estrategias no disponible"}
            
        try:
            result = {
                "strategy_id": strategy_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "params": params or {},
                "performance": None,
                "risk_metrics": None,
                "success": False
            }
            
            # Evaluar estrategia si está disponible
            if hasattr(self.strategy_manager, 'evaluate_strategy'):
                evaluation = await self._run_async_or_sync(
                    self.strategy_manager.evaluate_strategy,
                    strategy_id=strategy_id,
                    params=params
                )
                
                if evaluation:
                    result.update(evaluation)
                    result["success"] = True
            
            # Actualizar última evaluación
            self.last_evaluation = datetime.datetime.now()
            
            return result
        except Exception as e:
            logger.error(f"Error al evaluar estrategia {strategy_id}: {e}")
            return {
                "strategy_id": strategy_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e),
                "success": False
            }
    
    async def recommend_strategy(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Recomendar estrategia según contexto.
        
        Args:
            context: Contexto para la recomendación
            
        Returns:
            Estrategia recomendada
        """
        if not self.strategy_manager:
            return {"error": "Gestor de estrategias no disponible"}
            
        try:
            # Generar contexto predeterminado si no se proporciona
            if context is None:
                context = {
                    "risk_profile": "medium",
                    "timeframe": "medium",
                    "capital": 10000.0,
                    "market_condition": "neutral"
                }
            
            result = {
                "timestamp": datetime.datetime.now().isoformat(),
                "context": context,
                "recommendations": []
            }
            
            # Obtener recomendación si está disponible
            if hasattr(self.strategy_manager, 'recommend_strategy'):
                recommendations = await self._run_async_or_sync(
                    self.strategy_manager.recommend_strategy,
                    context=context
                )
                
                if recommendations:
                    result["recommendations"] = recommendations
                    
                    # Agregar detalles de la estrategia recomendada
                    if recommendations and "strategy_id" in recommendations[0]:
                        strategy_id = recommendations[0]["strategy_id"]
                        if strategy_id in self.available_strategies:
                            result["primary_recommendation"] = {
                                "strategy_id": strategy_id,
                                "details": self.available_strategies[strategy_id]
                            }
            
            # Actualizar caché
            context_key = f"{context.get('risk_profile', 'unknown')}_{context.get('timeframe', 'unknown')}"
            self.cached_recommendations[context_key] = result
            
            return result
        except Exception as e:
            logger.error(f"Error al recomendar estrategia: {e}")
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "context": context,
                "error": str(e)
            }
    
    async def get_strategy_details(self, strategy_id: str) -> Dict[str, Any]:
        """
        Obtener detalles de una estrategia específica.
        
        Args:
            strategy_id: Identificador de la estrategia
            
        Returns:
            Detalles de la estrategia
        """
        if not self.strategy_manager:
            return {"error": "Gestor de estrategias no disponible"}
            
        try:
            if strategy_id in self.available_strategies:
                return self.available_strategies[strategy_id]
                
            # Intentar obtener detalles si no está en caché
            if hasattr(self.strategy_manager, 'get_strategy_details'):
                details = await self._run_async_or_sync(
                    self.strategy_manager.get_strategy_details,
                    strategy_id=strategy_id
                )
                
                if details:
                    # Actualizar caché
                    self.available_strategies[strategy_id] = details
                    return details
            
            return {"error": f"Estrategia {strategy_id} no encontrada"}
        except Exception as e:
            logger.error(f"Error al obtener detalles de estrategia {strategy_id}: {e}")
            return {"error": f"Error al obtener detalles: {str(e)}"}
    
    async def get_trading_insights(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Obtener insights de trading basados en estrategias.
        
        Args:
            context: Contexto para los insights
            
        Returns:
            Insights generados
        """
        if not self.strategy_manager:
            return {"error": "Gestor de estrategias no disponible"}
            
        try:
            result = {
                "timestamp": datetime.datetime.now().isoformat(),
                "insights": []
            }
            
            # Obtener insights si está disponible
            if hasattr(self.strategy_manager, 'get_trading_insights'):
                insights = await self._run_async_or_sync(
                    self.strategy_manager.get_trading_insights,
                    context=context
                )
                
                if insights:
                    result["insights"] = insights
            
            return result
        except Exception as e:
            logger.error(f"Error al obtener insights de trading: {e}")
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
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
            "has_strategy_manager": self.strategy_manager is not None,
            "available_strategies_count": len(self.available_strategies),
            "available_strategy_ids": list(self.available_strategies.keys()),
            "last_evaluation": self.last_evaluation.isoformat() if self.last_evaluation else None,
            "cached_recommendations_count": len(self.cached_recommendations)
        }

# Crear instancia global
integrator = StrategyIntegrator()

def get_strategy_integrator() -> StrategyIntegrator:
    """
    Obtener instancia del integrador.
    
    Returns:
        Instancia del integrador
    """
    return integrator