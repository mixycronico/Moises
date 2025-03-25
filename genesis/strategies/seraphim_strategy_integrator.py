"""
Integrador de la estrategia Seraphim con el Sistema Genesis

Este módulo proporciona la capa de integración entre la estrategia Seraphim y 
el Sistema Genesis completo, permitiendo incorporar el comportamiento humanizado
de Gabriel en la estrategia de trading principal.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

from genesis.trading.gabriel_adapter import GabrielBehaviorEngine
from genesis.trading.seraphim_integration import SeraphimGabrielIntegrator
from genesis.trading.seraphim_orchestrator import SeraphimOrchestrator
from genesis.strategies.strategy_base import Strategy
from genesis.accounting.capital_scaling import CapitalScalingManager

logger = logging.getLogger(__name__)

class SeraphimStrategyIntegrator:
    """
    Integrador de la estrategia Seraphim en el sistema.
    
    Esta clase gestiona la integración de la estrategia Seraphim Pool con
    el Sistema Genesis, permitiendo su uso como estrategia principal.
    """
    
    def __init__(self, 
                 capital_base: float = 10000.0,
                 symbols: List[str] = None,
                 archetype: str = "COLLECTIVE"):
        """
        Inicializar integrador.
        
        Args:
            capital_base: Capital base para la estrategia
            symbols: Lista de símbolos a operar
            archetype: Arquetipo de comportamiento para Gabriel
        """
        self.capital_base = capital_base
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT"]
        self.archetype = archetype
        
        # Componentes
        self.gabriel_integrator = None
        self.orchestrator = None
        self.capital_manager = None
        
        # Estado
        self.is_initialized = False
        self.is_running = False
        self.current_cycle = None
        
        logger.info(f"SeraphimStrategyIntegrator creado con arquetipo {archetype} y {len(self.symbols)} símbolos")
    
    async def initialize(self, capital_manager: Optional[CapitalScalingManager] = None) -> bool:
        """
        Inicializar componentes de la estrategia.
        
        Args:
            capital_manager: Gestor de capital externo (opcional)
            
        Returns:
            True si se inicializó correctamente
        """
        if self.is_initialized:
            return True
            
        try:
            # Inicializar Gabriel
            self.gabriel_integrator = SeraphimGabrielIntegrator(archetype=self.archetype)
            await self.gabriel_integrator.initialize()
            
            # Usar capital manager externo o crear uno propio
            self.capital_manager = capital_manager
            
            # Inicializar orquestador
            self.orchestrator = SeraphimOrchestrator(
                capital_base=self.capital_base,
                symbols=self.symbols,
                behavior_engine=self.gabriel_integrator.gabriel
            )
            
            await self.orchestrator.initialize()
            
            self.is_initialized = True
            logger.info(f"SeraphimStrategyIntegrator inicializado correctamente con {len(self.symbols)} símbolos")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar SeraphimStrategyIntegrator: {str(e)}")
            return False
    
    async def start_trading_cycle(self) -> Dict[str, Any]:
        """
        Iniciar un ciclo de trading.
        
        Returns:
            Información del ciclo iniciado
        """
        if not self.is_initialized:
            await self.initialize()
        
        if self.is_running:
            logger.warning("Ya hay un ciclo de trading en ejecución")
            return {"status": "already_running", "cycle_id": self.current_cycle}
        
        # Iniciar ciclo con orquestador
        self.is_running = True
        cycle = await self.orchestrator.start_cycle()
        self.current_cycle = cycle.get("cycle_id")
        
        logger.info(f"Ciclo de trading iniciado: {self.current_cycle}")
        return cycle
    
    async def stop_trading_cycle(self) -> Dict[str, Any]:
        """
        Detener ciclo de trading actual.
        
        Returns:
            Resultados del ciclo detenido
        """
        if not self.is_running:
            return {"status": "not_running"}
        
        # Detener ciclo
        results = await self.orchestrator.stop_cycle()
        self.is_running = False
        
        # Procesar resultados con Gabriel para actualizar estado emocional
        profit = results.get("profit_percent", 0)
        await self.gabriel_integrator.process_cycle_result(profit, results)
        
        logger.info(f"Ciclo de trading {self.current_cycle} detenido con resultado: {profit*100:.2f}%")
        return results
    
    async def adjust_capital_allocation(self, new_allocation: Dict[str, float]) -> Dict[str, Any]:
        """
        Ajustar asignación de capital entre símbolos.
        
        Args:
            new_allocation: Diccionario {símbolo: porcentaje}
            
        Returns:
            Resultado del ajuste
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Verificar que todos los símbolos estén en la lista soportada
        for symbol in new_allocation:
            if symbol not in self.symbols:
                return {
                    "status": "error", 
                    "message": f"Símbolo no soportado: {symbol}",
                    "supported_symbols": self.symbols
                }
        
        # Verificar que la suma sea <= 100%
        total = sum(new_allocation.values())
        if total > 1.0:
            return {
                "status": "error",
                "message": f"La suma de asignaciones ({total*100:.2f}%) excede el 100%"
            }
        
        # Aplicar nueva asignación
        result = await self.orchestrator.update_capital_allocation(new_allocation)
        
        return {
            "status": "success",
            "message": "Asignación de capital actualizada",
            "allocation": result
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Obtener estado completo del sistema.
        
        Returns:
            Estado del sistema
        """
        if not self.is_initialized:
            return {
                "status": "not_initialized",
                "is_running": False
            }
        
        # Obtener estado del orquestador
        orchestrator_state = await self.orchestrator.get_system_status()
        
        # Obtener estado de Gabriel
        gabriel_state = await self.gabriel_integrator.get_current_state()
        
        # Combinar estado
        return {
            "status": "active" if self.is_running else "idle",
            "is_running": self.is_running,
            "current_cycle": self.current_cycle,
            "capital": orchestrator_state.get("capital", self.capital_base),
            "symbols": self.symbols,
            "emotional_state": gabriel_state.get("emotional_state", {}),
            "risk_profile": gabriel_state.get("risk_profile", {}),
            "cycle_stats": orchestrator_state.get("cycle_stats", {}),
            "positions": orchestrator_state.get("positions", []),
            "last_update": datetime.now().isoformat()
        }
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual del portafolio.
        
        Returns:
            Estado del portafolio
        """
        if not self.is_initialized:
            await self.initialize()
        
        portfolio = await self.orchestrator.get_portfolio_status()
        
        # Añadir recomendaciones basadas en estado emocional
        gabriel_state = await self.gabriel_integrator.get_current_state()
        mood = gabriel_state.get("emotional_state", {}).get("mood", "NEUTRAL")
        intensity = gabriel_state.get("emotional_state", {}).get("mood_intensity", 0.5)
        
        recommendations = []
        
        if mood == "FEARFUL" and intensity > 0.6:
            recommendations.append("Reducir exposición al mercado")
            recommendations.append("Aumentar posiciones defensivas")
        elif mood == "HOPEFUL" and intensity > 0.6:
            recommendations.append("Considerar oportunidades en activos prometedores")
        elif mood == "SERENE":
            recommendations.append("Mantener estrategia basada en análisis objetivo")
        elif mood == "CAUTIOUS":
            recommendations.append("Evaluar cuidadosamente nuevas entradas")
        
        portfolio["gabriel_recommendations"] = recommendations
        portfolio["last_update"] = datetime.now().isoformat()
        
        return portfolio
    
    async def analyze_market(self) -> Dict[str, Any]:
        """
        Realizar análisis de mercado.
        
        Returns:
            Resultados del análisis
        """
        if not self.is_initialized:
            await self.initialize()
        
        market_analysis = await self.orchestrator.analyze_market()
        
        # Añadir percepción humanizada
        for symbol in market_analysis.get("symbols", {}):
            data = market_analysis["symbols"][symbol]
            perception = await self.gabriel_integrator.process_market_data(symbol, data)
            market_analysis["symbols"][symbol]["human_perception"] = perception
        
        market_analysis["last_update"] = datetime.now().isoformat()
        return market_analysis
    
    async def execute_single_trade(self, 
                                  symbol: str, 
                                  side: str, 
                                  amount: float,
                                  price: Optional[float] = None) -> Dict[str, Any]:
        """
        Ejecutar una operación individual.
        
        Args:
            symbol: Símbolo a operar
            side: Lado ("buy"/"sell")
            amount: Cantidad a operar
            price: Precio (opcional, mercado si no se especifica)
            
        Returns:
            Resultado de la operación
        """
        if not self.is_initialized:
            await self.initialize()
        
        if symbol not in self.symbols:
            return {
                "status": "error",
                "message": f"Símbolo no soportado: {symbol}",
                "supported_symbols": self.symbols
            }
        
        # Obtener datos actuales del mercado
        market_data = await self.orchestrator.get_market_data(symbol)
        
        # Consultar a Gabriel
        original_decision = True  # Decisión forzada por el usuario
        signal_strength = 0.8  # Alta confianza al ser manual
        
        # Verificar con Gabriel (principio "todos ganamos o todos perdemos")
        decision, reason, confidence = await self.gabriel_integrator.evaluate_trade_decision(
            original_decision, symbol, signal_strength, market_data
        )
        
        if not decision:
            return {
                "status": "rejected_by_gabriel",
                "message": f"Gabriel rechazó la operación: {reason}",
                "confidence": confidence,
                "symbol": symbol,
                "side": side
            }
        
        # Ejecutar operación
        result = await self.orchestrator.execute_trade(symbol, side, amount, price)
        
        # Añadir contexto de Gabriel
        result["gabriel_approval"] = {
            "confidence": confidence,
            "reason": reason
        }
        
        return result
    
    async def get_investors_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener estado actual de los inversores.
        
        Returns:
            Diccionario con estado por inversor
        """
        if not self.is_initialized:
            await self.initialize()
        
        return await self.orchestrator.get_investors_status()
    
    async def distribute_profits(self, amount: float) -> Dict[str, float]:
        """
        Distribuir ganancias entre inversores según principio "todos ganamos o todos perdemos".
        
        Args:
            amount: Cantidad a distribuir
            
        Returns:
            Distribución por inversor
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Obtener datos de inversores
        investors = await self.orchestrator.get_investors_status()
        
        # Usar Gabriel para distribución justa
        distribution = await self.gabriel_integrator.distribute_profits(amount, investors)
        
        # Registrar distribución
        await self.orchestrator.record_profit_distribution(distribution)
        
        return distribution
    
    async def cleanup(self) -> None:
        """Liberar recursos."""
        if self.is_running:
            await self.stop_trading_cycle()
        
        if self.orchestrator:
            await self.orchestrator.cleanup()