"""
Integración del Motor de Comportamiento Humano Gabriel con el Orquestador Seraphim

Este módulo implementa la capa de integración entre el motor de comportamiento humano
Gabriel y el orquestador Seraphim, permitiendo que todas las decisiones del sistema
estén influenciadas por un comportamiento humanizado.

La integración permite que el principio "todos ganamos o todos perdemos" sea 
ejecutado por un sistema que toma decisiones con influencia de emociones humanas.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

from genesis.trading.gabriel_adapter import GabrielBehaviorEngine
from genesis.trading.human_behavior_engine import HumanBehaviorEngine  # Compatibilidad con código existente

logger = logging.getLogger(__name__)

class SeraphimGabrielIntegrator:
    """
    Integrador entre Gabriel y Seraphim.
    
    Esta clase gestiona la comunicación bidireccional entre el motor de comportamiento
    Gabriel y el orquestador Seraphim, asegurando que las decisiones de trading
    reflejen patrones de comportamiento humano y emocional.
    """
    
    def __init__(self, archetype: str = "COLLECTIVE"):
        """
        Inicializar el integrador.
        
        Args:
            archetype: Arquetipo de comportamiento para Gabriel
        """
        # Crear instancia de Gabriel
        self.gabriel = GabrielBehaviorEngine(archetype=archetype)
        self.is_initialized = False
        self.last_update = datetime.now()
        
        # Estadísticas de integración
        self.stats = {
            "market_analyses": 0,
            "trade_decisions": 0,
            "modified_decisions": 0,  # Decisiones modificadas por Gabriel
            "risk_adjustments": 0,    # Ajustes de riesgo realizados
            "emotional_changes": 0,   # Cambios de estado emocional
        }
        
        # Mapeo de perfiles de riesgo
        self.risk_profiles = {
            "BALANCED": 0.5,        # Perfil equilibrado
            "AGGRESSIVE": 0.7,      # Perfil agresivo
            "CONSERVATIVE": 0.3,    # Perfil conservador
            "GUARDIAN": 0.25,       # Perfil muy conservador
            "EXPLORER": 0.65,       # Perfil explorador
            "COLLECTIVE": 0.45      # Colectivo/pool - "todos ganamos o todos perdemos"
        }
        
        logger.info(f"Integrador Gabriel-Seraphim inicializado con arquetipo {archetype}")
    
    async def initialize(self) -> bool:
        """
        Inicializar el integrador y el motor de comportamiento.
        
        Returns:
            True si la inicialización fue exitosa
        """
        if self.is_initialized:
            return True
        
        try:
            # Inicializar Gabriel
            await self.gabriel.initialize()
            self.is_initialized = True
            logger.info("Integrador Gabriel-Seraphim inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar integrador Gabriel-Seraphim: {str(e)}")
            return False
    
    async def process_market_data(self, 
                                symbol: str, 
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar datos de mercado a través de Gabriel para humanización.
        
        Args:
            symbol: Símbolo del mercado
            market_data: Datos del mercado
            
        Returns:
            Datos procesados con percepción humanizada
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Añadir símbolo a datos
        data = market_data.copy()
        data["symbol"] = symbol
        
        # Obtener percepción de Gabriel
        perception = await self.gabriel.process_market_data(data)
        
        # Incrementar contador
        self.stats["market_analyses"] += 1
        
        return perception
    
    async def evaluate_trade_decision(self, 
                                    original_decision: bool,
                                    symbol: str,
                                    signal_strength: float,
                                    market_data: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Modificar decisión de trading aplicando comportamiento humano.
        
        Args:
            original_decision: Decisión original del sistema
            symbol: Símbolo del mercado
            signal_strength: Fuerza de la señal (0-1)
            market_data: Datos del mercado
            
        Returns:
            Tupla (decisión_final, razón, confianza)
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Consultar a Gabriel
        gabriel_decision, reason, confidence = await self.gabriel.evaluate_trade_opportunity(
            symbol, signal_strength, market_data
        )
        
        # Incrementar contador
        self.stats["trade_decisions"] += 1
        
        # Comprobar si Gabriel modificó la decisión original
        if gabriel_decision != original_decision:
            self.stats["modified_decisions"] += 1
            logger.info(f"Gabriel modificó decisión original para {symbol}: {original_decision} -> {gabriel_decision}")
            
            # Añadir referencia a la decisión original
            if not gabriel_decision:
                reason = f"Gabriel rechazó: {reason} (sistema recomendaba entrar)"
            else:
                reason = f"Gabriel recomendó: {reason} (sistema sugería no entrar)"
        
        return gabriel_decision, reason, confidence
    
    async def adjust_position_size(self, 
                                base_size: float, 
                                capital: float,
                                volatility: float,
                                max_risk_percent: float) -> float:
        """
        Ajustar tamaño de posición según comportamiento humano.
        
        Args:
            base_size: Tamaño base calculado por la estrategia
            capital: Capital total disponible
            volatility: Volatilidad del mercado (0-1)
            max_risk_percent: Porcentaje máximo de riesgo permitido
            
        Returns:
            Tamaño ajustado según comportamiento humano
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Crear contexto de riesgo
        risk_context = {
            "volatility": volatility,
            "max_risk": max_risk_percent,
            "market_conditions": "normal"  # Puede ser 'normal', 'extreme', 'uncertain'
        }
        
        # Solicitar ajuste a Gabriel
        adjusted_size = await self.gabriel.adjust_position_size(
            base_size, capital, risk_context
        )
        
        # Incrementar contador
        self.stats["risk_adjustments"] += 1
        
        return adjusted_size
    
    async def evaluate_exit_decision(self,
                                  original_decision: bool,
                                  position_data: Dict[str, Any],
                                  market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluar decisión de salida con comportamiento humano.
        
        Args:
            original_decision: Decisión original del sistema
            position_data: Datos de la posición actual
            market_data: Datos actuales del mercado
            
        Returns:
            Tupla (decisión_final, razón)
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Consultar a Gabriel
        gabriel_decision, reason = await self.gabriel.evaluate_exit_opportunity(
            position_data, market_data
        )
        
        # Comprobar si Gabriel modificó la decisión original
        if gabriel_decision != original_decision:
            self.stats["modified_decisions"] += 1
            logger.info(f"Gabriel modificó decisión de salida: {original_decision} -> {gabriel_decision}")
            
            # Añadir referencia a la decisión original
            if gabriel_decision:
                reason = f"Gabriel recomendó salir: {reason} (sistema sugería mantener)"
            else:
                reason = f"Gabriel recomendó mantener: {reason} (sistema sugería salir)"
        
        return gabriel_decision, reason
    
    async def process_cycle_result(self, 
                                cycle_profit: float, 
                                cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar resultados de un ciclo de trading para ajustar estado emocional.
        
        Args:
            cycle_profit: Ganancia/pérdida del ciclo (porcentaje)
            cycle_data: Datos adicionales del ciclo
            
        Returns:
            Recomendaciones para el siguiente ciclo
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Crear noticia según resultado
        if cycle_profit > 0:
            sentiment = "bullish" if cycle_profit > 0.03 else "slightly_bullish"
            importance = min(0.5 + cycle_profit * 5, 0.9)  # Mayor importancia a resultados extremos
            
            news = {
                "title": "Ciclo de trading exitoso",
                "content": f"El ciclo de trading ha finalizado con beneficio de {cycle_profit*100:.2f}%",
                "sentiment": sentiment,
                "importance": importance,
                "impact": cycle_profit * 2,  # Impacto emocional proporcional al beneficio
                "related_to_portfolio": True
            }
        else:
            sentiment = "bearish" if cycle_profit < -0.03 else "slightly_bearish"
            importance = min(0.5 + abs(cycle_profit) * 5, 0.9)
            
            news = {
                "title": "Ciclo de trading con pérdidas",
                "content": f"El ciclo de trading ha finalizado con pérdida de {abs(cycle_profit)*100:.2f}%",
                "sentiment": sentiment,
                "importance": importance,
                "impact": cycle_profit * 2,  # Impacto negativo proporcional a la pérdida
                "related_to_portfolio": True
            }
        
        # Procesar noticia
        await self.gabriel.process_news(news)
        self.stats["emotional_changes"] += 1
        
        # Obtener estado actual
        emotional_state = await self.gabriel.get_emotional_state()
        risk_profile = self.gabriel.get_risk_profile()
        
        # Preparar recomendaciones según estado emocional
        recommendations = {
            "state": emotional_state,
            "risk_profile": risk_profile,
            "recommended_actions": [],
            "allocation_adjustment": 1.0,  # Factor multiplicador para asignación de capital
            "risk_adjustment": 1.0         # Factor multiplicador para nivel de riesgo
        }
        
        # Ajustar según estado emocional
        mood = emotional_state["mood"]
        intensity = emotional_state["mood_intensity"]
        
        if mood == "FEARFUL":
            # En estado temeroso, reducir asignaciones y riesgo
            recommendations["allocation_adjustment"] = 1.0 - (intensity * 0.4)
            recommendations["risk_adjustment"] = 1.0 - (intensity * 0.5)
            recommendations["recommended_actions"].append("Reducir exposición al mercado")
            
            if intensity > 0.7:
                recommendations["recommended_actions"].append("Proteger capital con stop loss más ajustados")
        
        elif mood == "HOPEFUL":
            # En estado esperanzado, puede aumentar ligeramente
            recommendations["allocation_adjustment"] = 1.0 + (intensity * 0.2)
            recommendations["risk_adjustment"] = 1.0 + (intensity * 0.1)
            
            if intensity > 0.6:
                recommendations["recommended_actions"].append("Considerar oportunidades en activos de mayor potencial")
        
        elif mood == "SERENE":
            # En estado sereno, decisiones más objetivas
            recommendations["recommended_actions"].append("Mantener estrategia basada en análisis objetivo")
        
        elif mood == "CAUTIOUS":
            # En estado cauteloso, ligeramente conservador
            recommendations["allocation_adjustment"] = 1.0 - (intensity * 0.15)
            recommendations["risk_adjustment"] = 1.0 - (intensity * 0.2)
            recommendations["recommended_actions"].append("Evaluar cuidadosamente nuevas entradas")
        
        return recommendations
    
    async def distribute_profits(self, 
                              total_profit: float, 
                              investor_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Distribuir ganancias entre inversores según el principio "todos ganamos o todos perdemos".
        
        Esta función implementa de forma práctica el principio fundamental del sistema,
        asegurando que las ganancias se distribuyan de manera justa y proporcional.
        
        Args:
            total_profit: Ganancia total del ciclo
            investor_data: Datos de los inversores con sus aportes
            
        Returns:
            Diccionario con distribución de ganancias por inversor
        """
        # Si no hay ganancia, devolver distribución neutral
        if total_profit == 0:
            return {investor_id: 0 for investor_id in investor_data}
        
        # Calcular capital total
        total_capital = sum(data["capital"] for data in investor_data.values())
        
        # Calcular distribución estrictamente proporcional (base)
        base_distribution = {
            investor_id: (data["capital"] / total_capital) * total_profit
            for investor_id, data in investor_data.items()
        }
        
        # Si la ganancia es positiva, aplicar "todos ganamos"
        if total_profit > 0:
            # Todo inversor debe tener al menos un mínimo de ganancia
            min_percent = 0.6  # Porcentaje mínimo proporcional que debe recibir cada inversor
            
            # Calcular ajustes para asegurar que todos ganan
            adjustments = {}
            surplus = 0
            
            for investor_id, share in base_distribution.items():
                min_share = (investor_data[investor_id]["capital"] / total_capital) * total_profit * min_percent
                
                if share < min_share:
                    # Este inversor recibirá más de lo proporcional
                    deficit = min_share - share
                    adjustments[investor_id] = deficit
                else:
                    # Este inversor aporta al surplus
                    surplus_contribution = (share - (min_percent * share / 1.0)) * 0.5
                    adjustments[investor_id] = -surplus_contribution
                    surplus += surplus_contribution
            
            # Redistribuir el surplus según necesidades
            if surplus > 0 and sum(v for v in adjustments.values() if v > 0) > 0:
                # Calcular factor de redistribución
                needed = sum(v for v in adjustments.values() if v > 0)
                redistribution_factor = min(1.0, surplus / needed)
                
                # Aplicar redistribución
                for investor_id in adjustments:
                    if adjustments[investor_id] > 0:  # Necesita más
                        adjustments[investor_id] *= redistribution_factor
            
            # Aplicar ajustes a la distribución base
            final_distribution = {
                investor_id: base_distribution[investor_id] + adjustments.get(investor_id, 0)
                for investor_id in base_distribution
            }
            
            # Verificar que la suma es igual a total_profit (con margen de error)
            sum_distribution = sum(final_distribution.values())
            if abs(sum_distribution - total_profit) > 0.001:
                # Ajustar de forma proporcional para corregir diferencia
                correction_factor = total_profit / sum_distribution
                final_distribution = {
                    investor_id: amount * correction_factor
                    for investor_id, amount in final_distribution.items()
                }
        
        # Si la ganancia es negativa, aplicar "todos perdemos"
        else:
            # Reducir ligeramente el impacto en inversores pequeños
            min_percent = 0.8  # Porcentaje mínimo de pérdida que un inversor pequeño debe asumir
            
            # Calcular ajustes para suavizar pérdidas de inversores pequeños
            adjustments = {}
            surplus_burden = 0
            
            # Identificar inversores pequeños (menos del 10% del capital total)
            small_investor_threshold = total_capital * 0.1
            
            for investor_id, share in base_distribution.items():
                investor_capital = investor_data[investor_id]["capital"]
                is_small = investor_capital < small_investor_threshold
                
                if is_small:
                    # Limitar la pérdida del inversor pequeño
                    max_loss = (investor_capital / total_capital) * total_profit * min_percent
                    reduction = share - max_loss  # Reducción de pérdida (valor negativo)
                    adjustments[investor_id] = reduction
                    surplus_burden += -reduction  # Convertir a valor positivo para suma
            
            # Redistribuir carga adicional entre inversores grandes
            if surplus_burden > 0:
                # Identificar inversores grandes
                large_investors = [
                    investor_id for investor_id, data in investor_data.items()
                    if data["capital"] >= small_investor_threshold
                ]
                
                if large_investors:
                    # Calcular capital total de inversores grandes
                    large_capital = sum(investor_data[inv_id]["capital"] for inv_id in large_investors)
                    
                    # Distribuir carga adicional proporcionalmente
                    for investor_id in large_investors:
                        investor_capital = investor_data[investor_id]["capital"]
                        add_burden = (investor_capital / large_capital) * surplus_burden
                        adjustments[investor_id] = -add_burden  # Pérdida adicional
            
            # Aplicar ajustes a la distribución base
            final_distribution = {
                investor_id: base_distribution[investor_id] + adjustments.get(investor_id, 0)
                for investor_id in base_distribution
            }
            
            # Verificar que la suma es igual a total_profit (con margen de error)
            sum_distribution = sum(final_distribution.values())
            if abs(sum_distribution - total_profit) > 0.001:
                # Ajustar de forma proporcional para corregir diferencia
                correction_factor = total_profit / sum_distribution
                final_distribution = {
                    investor_id: amount * correction_factor
                    for investor_id, amount in final_distribution.items()
                }
        
        return final_distribution
    
    async def set_emergency_mode(self, emergency_type: str) -> None:
        """
        Activar modo de emergencia en Gabriel.
        
        Args:
            emergency_type: Tipo de emergencia
        """
        if not self.is_initialized:
            await self.initialize()
        
        await self.gabriel.set_emergency_mode(emergency_type)
        self.stats["emotional_changes"] += 1
        
        logger.warning(f"Modo de emergencia activado en Gabriel: {emergency_type}")
    
    async def reset_state(self) -> None:
        """Reiniciar estado de Gabriel a valores por defecto."""
        if not self.is_initialized:
            await self.initialize()
        
        await self.gabriel.normalize_state()
        self.stats["emotional_changes"] += 1
        
        logger.info("Estado de Gabriel reiniciado a valores por defecto")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de integración.
        
        Returns:
            Diccionario con estadísticas
        """
        return self.stats.copy()
    
    async def get_current_state(self) -> Dict[str, Any]:
        """
        Obtener estado actual completo.
        
        Returns:
            Diccionario con estado actual
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Obtener estado emocional
        emotional_state = await self.gabriel.get_emotional_state()
        
        # Obtener perfil de riesgo
        risk_profile = self.gabriel.get_risk_profile()
        
        # Combinar información
        state = {
            "emotional_state": emotional_state,
            "risk_profile": risk_profile,
            "stats": self.get_stats(),
            "last_update": self.last_update.isoformat()
        }
        
        return state

# Compatibilidad con código existente
def get_human_behavior_engine(archetype: str = "COLLECTIVE") -> HumanBehaviorEngine:
    """
    Obtener instancia del motor de comportamiento humano.
    
    Esta función proporciona compatibilidad con código existente que espera
    la clase HumanBehaviorEngine.
    
    Args:
        archetype: Arquetipo de comportamiento
        
    Returns:
        Instancia de HumanBehaviorEngine
    """
    return HumanBehaviorEngine(archetype=archetype)

def get_gabriel_adapter(archetype: str = "COLLECTIVE") -> GabrielBehaviorEngine:
    """
    Obtener instancia del adaptador Gabriel.
    
    Args:
        archetype: Arquetipo de comportamiento
        
    Returns:
        Instancia de GabrielBehaviorEngine
    """
    return GabrielBehaviorEngine(archetype=archetype)

def get_seraphim_integrator(archetype: str = "COLLECTIVE") -> SeraphimGabrielIntegrator:
    """
    Obtener instancia del integrador Seraphim-Gabriel.
    
    Args:
        archetype: Arquetipo de comportamiento
        
    Returns:
        Instancia de SeraphimGabrielIntegrator
    """
    return SeraphimGabrielIntegrator(archetype=archetype)