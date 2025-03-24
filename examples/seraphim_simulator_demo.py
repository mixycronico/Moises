"""
Demo del Sistema Genesis Ultra-Divino Trading Nexus 10M usando el Simulador.

Este script ejecuta una demostración del Sistema Genesis con la estrategia
Seraphim Pool utilizando el simulador de exchange para operar sin depender
de conexiones externas a exchanges reales.

La demo muestra:
1. Inicialización del orquestador Seraphim con simulador
2. Inicio de un ciclo de trading
3. Procesamiento del ciclo completo
4. Información en tiempo real sobre el ciclo
5. Comportamiento humano simulado de Gabriel

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina Simulada)
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
import random
from typing import Dict, Any, Optional, List

# Importar componentes del Sistema Genesis
from genesis.trading.seraphim_orchestrator import SeraphimOrchestrator, OrchestratorState
from genesis.simulators import MarketPattern, MarketEventType
from genesis.exchanges.adapter_factory import ExchangeAdapterFactory, AdapterType
from genesis.strategies.seraphim.seraphim_pool import CyclePhase
from genesis.trading.human_behavior_engine import EmotionalState, RiskTolerance, DecisionStyle

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Genesis.SimulatorDemo")

async def setup_simulator(orchestrator: SeraphimOrchestrator) -> bool:
    """
    Configurar el simulador en el orquestador.
    
    Args:
        orchestrator: Orquestador a configurar
        
    Returns:
        True si la configuración fue exitosa
    """
    try:
        logger.info("Configurando simulador de exchange para el orquestador...")
        
        # Crear adaptador simulado
        adapter = await ExchangeAdapterFactory.create_adapter(
            exchange_id="BINANCE",
            adapter_type=AdapterType.SIMULATED,
            config={
                "tick_interval_ms": 500,        # Actualizaciones cada 500ms
                "volatility_factor": 0.005,     # 0.5% de volatilidad
                "error_rate": 0.05,             # 5% de probabilidad de error
                "pattern_duration": 30,         # 30 segundos por patrón
                "enable_failures": True,        # Habilitar fallos simulados
                "default_candle_count": 1000,   # 1000 velas históricas
                "enable_websocket": True        # Habilitar websocket
            }
        )
        
        # Asignar adaptador al orquestador
        orchestrator.exchange_adapter = adapter
        
        # Configurar símbolos iniciales
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
        for symbol in symbols:
            # Configurar símbolos en el simulador
            await adapter.get_ticker(symbol)
            
            # Establecer patrón inicial aleatorio (con bias hacia alcista)
            patterns = [
                MarketPattern.TRENDING_UP, 
                MarketPattern.TRENDING_UP,  # Doble probabilidad de alcista
                MarketPattern.CONSOLIDATION,
                MarketPattern.VOLATILE
            ]
            pattern = random.choice(patterns)
            await adapter.set_market_pattern(symbol, pattern)
            logger.info(f"Configurado {symbol} con patrón {pattern.name}")
        
        # Verificar conexión
        await orchestrator._verify_exchange_connections()
        
        logger.info("Simulador configurado correctamente")
        return True
        
    except Exception as e:
        logger.error(f"Error configurando simulador: {e}")
        return False

async def simulate_market_events(orchestrator: SeraphimOrchestrator) -> None:
    """
    Simular eventos de mercado durante la ejecución.
    
    Args:
        orchestrator: Orquestador con simulador configurado
    """
    try:
        adapter = orchestrator.exchange_adapter
        if not adapter:
            logger.warning("No hay adaptador configurado para simular eventos")
            return
            
        # Programar eventos de mercado aleatorios durante la ejecución
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
        
        # Evento positivo para BTC (noticia positiva)
        await adapter.add_market_event(
            MarketEventType.POSITIVE_NEWS,
            "BTC/USDT",
            impact=0.02,  # 2% de impacto
            delay=15      # Después de 15 segundos
        )
        logger.info("Evento programado: Noticia positiva para BTC en 15s")
        
        # Evento de aumento de volatilidad para ETH
        await adapter.add_market_event(
            MarketEventType.VOLATILITY_INCREASE,
            "ETH/USDT",
            impact=0.01,  # 1% de impacto
            delay=30      # Después de 30 segundos
        )
        logger.info("Evento programado: Aumento de volatilidad para ETH en 30s")
        
        # Evento de compra de ballena para un símbolo aleatorio
        random_symbol = random.choice(symbols)
        await adapter.add_market_event(
            MarketEventType.WHALE_BUY,
            random_symbol,
            impact=0.03,  # 3% de impacto
            delay=45      # Después de 45 segundos
        )
        logger.info(f"Evento programado: Compra de ballena para {random_symbol} en 45s")
        
    except Exception as e:
        logger.error(f"Error programando eventos de mercado: {e}")

async def randomize_human_behavior(orchestrator: SeraphimOrchestrator) -> None:
    """
    Aleatorizar comportamiento humano en la estrategia Seraphim.
    
    Args:
        orchestrator: Orquestador con comportamiento humano a aleatorizar
    """
    try:
        # Verificar que exista el motor de comportamiento
        if (not hasattr(orchestrator, 'behavior_engine') or 
            not orchestrator.behavior_engine or
            not orchestrator.seraphim_strategy or 
            not orchestrator.seraphim_strategy.behavior_engine):
            logger.warning("Motor de comportamiento no disponible para aleatorizar")
            return
            
        # Obtener motor de comportamiento
        behavior_engine = orchestrator.behavior_engine
        
        # Establecer estados emocionales aleatorios
        emotional_states = list(EmotionalState)
        risk_tolerances = list(RiskTolerance)
        decision_styles = list(DecisionStyle)
        
        # Aleatorizar comportamiento
        new_emotional_state = random.choice(emotional_states)
        new_risk_tolerance = random.choice(risk_tolerances)
        new_decision_style = random.choice(decision_styles)
        
        # Aplicar cambios
        behavior_engine.emotional_state = new_emotional_state
        behavior_engine.risk_tolerance = new_risk_tolerance
        behavior_engine.decision_style = new_decision_style
        
        # Actualizar otros parámetros aleatoriamente
        behavior_engine.confidence_level = random.uniform(0.3, 0.9)
        behavior_engine.patience_factor = random.uniform(0.2, 0.8)
        behavior_engine.fomo_susceptibility = random.uniform(0.1, 0.7)
        behavior_engine.loss_aversion = random.uniform(1.2, 2.5)
        
        logger.info(f"Comportamiento humano aleatorizado: {new_emotional_state.name}, "
                   f"{new_risk_tolerance.name}, {new_decision_style.name}")
        
        # Propagar cambios a la estrategia
        orchestrator.seraphim_strategy.behavior_engine = behavior_engine
        
    except Exception as e:
        logger.error(f"Error aleatorizando comportamiento humano: {e}")

async def display_cycle_progress(orchestrator: SeraphimOrchestrator, cycle_id: str) -> None:
    """
    Mostrar progreso del ciclo en tiempo real.
    
    Args:
        orchestrator: Orquestador con ciclo activo
        cycle_id: ID del ciclo a monitorear
    """
    try:
        # Verificar cada segundo el progreso del ciclo
        previous_phase = None
        
        while True:
            # Obtener estado actual
            cycle_status = await orchestrator.get_cycle_status(cycle_id)
            
            # Verificar si el ciclo sigue activo
            if not cycle_status.get("is_active", False):
                logger.info(f"Ciclo {cycle_id} ya no está activo")
                break
                
            # Obtener fase actual
            current_phase = cycle_status.get("cycle_phase")
            
            # Si cambió la fase, mostrar información
            if current_phase != previous_phase:
                previous_phase = current_phase
                logger.info(f"Ciclo {cycle_id} ahora en fase: {current_phase}")
                
                # Información adicional según la fase
                if current_phase == CyclePhase.REVELATION.name:
                    logger.info("Fase de REVELATION: Analizando mercado y seleccionando activos")
                elif current_phase == CyclePhase.EXECUTION.name:
                    logger.info("Fase de EXECUTION: Ejecutando operaciones de trading")
                elif current_phase == CyclePhase.GUARDIANSHIP.name:
                    logger.info("Fase de GUARDIANSHIP: Monitorizando posiciones abiertas")
                elif current_phase == CyclePhase.REFLECTION.name:
                    logger.info("Fase de REFLECTION: Evaluando resultados del ciclo")
                elif current_phase == CyclePhase.DISTRIBUTION.name:
                    logger.info("Fase de DISTRIBUTION: Distribuyendo ganancias")
                    
            # Verificar estado de la estrategia
            strategy_state = cycle_status.get("strategy_state")
            if strategy_state:
                logger.info(f"Estado de estrategia: {strategy_state}")
                
            # Esperar antes de la siguiente verificación
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Error mostrando progreso del ciclo: {e}")

async def run_seraphim_demo():
    """Ejecutar demostración completa de Seraphim con simulador."""
    logger.info("=== Iniciando Demo de Seraphim con Simulador ===")
    
    try:
        # 1. Crear orquestador
        orchestrator = SeraphimOrchestrator()
        logger.info("Orquestador Seraphim creado")
        
        # 2. Configurar simulador
        simulator_ready = await setup_simulator(orchestrator)
        if not simulator_ready:
            logger.error("No se pudo configurar el simulador, abortando demo")
            return
            
        # 3. Inicializar orquestador
        success = await orchestrator.initialize()
        if not success:
            logger.error("No se pudo inicializar el orquestador, abortando demo")
            return
            
        logger.info(f"Orquestador inicializado correctamente, estado: {orchestrator.state.name}")
        
        # 4. Comprobar salud del sistema
        health = await orchestrator.check_system_health()
        logger.info(f"Salud del sistema: {health:.2f}")
        
        # 5. Programar eventos de mercado
        await simulate_market_events(orchestrator)
        
        # 6. Aleatorizar comportamiento humano inicial
        await randomize_human_behavior(orchestrator)
        
        # 7. Iniciar ciclo de trading
        cycle_result = await orchestrator.start_trading_cycle()
        
        if not cycle_result.get("success", False):
            logger.error(f"Error al iniciar ciclo: {cycle_result}")
            return
            
        cycle_id = cycle_result.get("cycle_id")
        logger.info(f"Ciclo iniciado con ID: {cycle_id}")
        
        # 8. Iniciar tarea para mostrar progreso
        progress_task = asyncio.create_task(
            display_cycle_progress(orchestrator, cycle_id)
        )
        
        # 9. Iniciar tarea para aleatorizar comportamiento durante ciclo
        async def periodic_behavior_changes():
            """Cambiar comportamiento periódicamente."""
            while orchestrator.active_cycle_id:
                await asyncio.sleep(20)  # Cada 20 segundos
                await randomize_human_behavior(orchestrator)
                
        behavior_task = asyncio.create_task(periodic_behavior_changes())
        
        # 10. Procesar ciclo hasta completar
        while orchestrator.active_cycle_id:
            # Procesar ciclo
            process_result = await orchestrator.process_cycle()
            
            if not process_result.get("success", False):
                logger.warning(f"Error en procesamiento de ciclo: {process_result}")
                
            # Verificar si se completó
            if not orchestrator.active_cycle_id:
                logger.info("Ciclo completado")
                break
                
            # Esperar antes del siguiente procesamiento
            await asyncio.sleep(3)
            
        # 11. Esperar a que finalicen tareas
        await progress_task
        behavior_task.cancel()
        
        # 12. Obtener resultados finales
        system_overview = await orchestrator.get_system_overview()
        
        logger.info("=== Resultados de la Demo ===")
        logger.info(f"Ciclos completados: {system_overview.get('completed_cycles_count', 0)}")
        logger.info(f"Beneficio total: ${system_overview.get('total_profit', 0):.2f}")
        logger.info(f"Salud final del sistema: {system_overview.get('health_score', 0):.2f}")
        logger.info(f"Tiempo de ejecución: {system_overview.get('uptime', '')}")
        
        logger.info("Demo completada con éxito")
        
    except Exception as e:
        logger.error(f"Error en demo: {e}")
    
if __name__ == "__main__":
    asyncio.run(run_seraphim_demo())