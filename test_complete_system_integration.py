"""
Prueba de integración completa del Sistema Genesis

Este script prueba la integración completa del Sistema Genesis con:
1. Motor de comportamiento humano Gabriel
2. Estrategia Seraphim con principio "todos ganamos o todos perdemos"
3. Sistema de gestión de capital adaptativo
4. Integración con intercambio simulado

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import asyncio
import logging
import json
import random
from datetime import datetime
from typing import Dict, Any, List, Optional

from genesis.init.seraphim_initializer import initialize_seraphim_strategy
from genesis.accounting.capital_scaling import CapitalScalingManager
from genesis.accounting.predictive_scaling import PredictiveScalingEngine
from genesis.exchanges.simulated_exchange_adapter import SimulatedExchangeAdapter
from genesis.exchanges.exchange_simulator import ExchangeSimulator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("test_complete_system")

# Constantes para test
TEST_CAPITAL = 10000.0
TEST_SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]

async def setup_exchange_simulator() -> SimulatedExchangeAdapter:
    """
    Configurar simulador de intercambio para pruebas.
    
    Returns:
        Adaptador de exchange simulado
    """
    logger.info("Configurando simulador de intercambio...")
    
    # Crear simulador
    simulator = ExchangeSimulator()
    
    # Configurar mercados
    for symbol in TEST_SYMBOLS:
        simulator.add_market(
            symbol=symbol,
            base_price=random.uniform(100, 50000),
            volatility=random.uniform(0.1, 0.4)
        )
    
    # Crear adaptador
    adapter = SimulatedExchangeAdapter(simulator=simulator)
    await adapter.initialize()
    
    logger.info(f"Simulador de intercambio configurado con {len(TEST_SYMBOLS)} símbolos")
    return adapter

async def setup_capital_manager() -> CapitalScalingManager:
    """
    Configurar gestor de capital para pruebas.
    
    Returns:
        Gestor de capital
    """
    logger.info("Configurando gestor de capital...")
    
    # Crear motor predictivo
    engine = PredictiveScalingEngine()
    await engine.initialize()
    
    # Crear gestor de capital
    manager = CapitalScalingManager(
        base_capital=TEST_CAPITAL,
        scaling_engine=engine
    )
    await manager.initialize()
    
    logger.info(f"Gestor de capital configurado con ${TEST_CAPITAL:.2f}")
    return manager

async def test_market_analysis(strategy) -> None:
    """
    Probar análisis de mercado con percepción humanizada.
    
    Args:
        strategy: Estrategia integrada
    """
    logger.info("\n=== Prueba de análisis de mercado ===")
    
    # Obtener análisis
    analysis = await strategy.analyze_market()
    
    # Mostrar resultados por símbolo
    for symbol, data in analysis.get("symbols", {}).items():
        perception = data.get("human_perception", {})
        mood = perception.get("mood", "DESCONOCIDO")
        intensity = perception.get("mood_intensity", 0)
        logger.info(f"\nAnálisis para {symbol}:")
        logger.info(f"Precio actual: ${data.get('price', 0):.2f}")
        logger.info(f"Tendencia: {data.get('trend', 0):.4f}")
        logger.info(f"Percepción humana: {mood} (intensidad: {intensity:.2f})")
        
        # Mostrar información adicional si está disponible
        if "signal_strength" in data:
            logger.info(f"Fuerza de señal: {data['signal_strength']:.4f}")
        if "recommendation" in data:
            logger.info(f"Recomendación: {data['recommendation']}")

async def test_trading_cycle(strategy) -> None:
    """
    Probar ciclo completo de trading.
    
    Args:
        strategy: Estrategia integrada
    """
    logger.info("\n=== Prueba de ciclo de trading ===")
    
    # Iniciar ciclo
    cycle = await strategy.start_trading_cycle()
    logger.info(f"Ciclo iniciado: {cycle.get('cycle_id', 'DESCONOCIDO')}")
    
    # Permitir que el ciclo opere por un tiempo
    logger.info("Ciclo en ejecución por 5 segundos...")
    await asyncio.sleep(5)
    
    # Obtener estado
    status = await strategy.get_system_status()
    emotional_state = status.get("emotional_state", {})
    mood = emotional_state.get("mood", "DESCONOCIDO")
    intensity = emotional_state.get("mood_intensity", 0)
    
    logger.info(f"Estado emocional: {mood} (intensidad: {intensity:.2f})")
    logger.info(f"Posiciones abiertas: {len(status.get('positions', []))}")
    
    # Detener ciclo
    results = await strategy.stop_trading_cycle()
    profit = results.get("profit_percent", 0)
    
    logger.info(f"Ciclo finalizado con resultado: {profit*100:.2f}%")
    logger.info(f"Operaciones realizadas: {results.get('total_trades', 0)}")
    
    # Verificar cambio en estado emocional
    status_after = await strategy.get_system_status()
    emotional_state_after = status_after.get("emotional_state", {})
    mood_after = emotional_state_after.get("mood", "DESCONOCIDO")
    intensity_after = emotional_state_after.get("mood_intensity", 0)
    
    logger.info(f"Estado emocional después del ciclo: {mood_after} (intensidad: {intensity_after:.2f})")
    if mood != mood_after or abs(intensity - intensity_after) > 0.05:
        logger.info(f"Cambio emocional detectado: {mood} -> {mood_after}")

async def test_profit_distribution(strategy) -> None:
    """
    Probar distribución de ganancias según principio colectivo.
    
    Args:
        strategy: Estrategia integrada
    """
    logger.info("\n=== Prueba de distribución de ganancias ===")
    
    # Crear datos de inversores simulados
    investors = {
        "investor_1": {"name": "Juan", "capital": 5000.0},
        "investor_2": {"name": "María", "capital": 10000.0},
        "investor_3": {"name": "Pedro", "capital": 3000.0},
        "investor_4": {"name": "Ana", "capital": 1500.0},
        "investor_5": {"name": "Carlos", "capital": 15000.0}
    }
    
    # Calcular capital total
    total_capital = sum(inv["capital"] for inv in investors.values())
    logger.info(f"Capital total: ${total_capital:.2f}")
    
    # Simular ganancia
    profit = 1500.0
    profit_percent = profit / total_capital
    logger.info(f"Ganancia a distribuir: ${profit:.2f} ({profit_percent*100:.2f}%)")
    
    # Calcular distribución estándar (proporcional directa)
    standard_distribution = {
        inv_id: inv["capital"] / total_capital * profit
        for inv_id, inv in investors.items()
    }
    
    # Obtener distribución según estrategia
    gabriel_distribution = await strategy.distribute_profits(profit)
    
    # Mostrar comparación
    logger.info("Comparación de distribuciones:")
    logger.info(f"{'Inversor':<10} {'Capital':<10} {'% del Pool':<12} {'Estándar':<12} {'Gabriel':<12} {'Diferencia':<12}")
    logger.info("-" * 70)
    
    for inv_id, inv in investors.items():
        cap = inv["capital"]
        pool_percent = cap / total_capital * 100
        std = standard_distribution[inv_id]
        gbr = gabriel_distribution.get(inv_id, 0)
        diff = gbr - std
        
        logger.info(f"{inv['name']:<10} ${cap:<9.2f} {pool_percent:<11.2f}% ${std:<11.2f} ${gbr:<11.2f} ${diff:<11.2f}")
    
    # Verificar suma total
    std_sum = sum(standard_distribution.values())
    gbr_sum = sum(gabriel_distribution.values())
    
    logger.info("-" * 70)
    logger.info(f"{'TOTAL':<10} ${total_capital:<9.2f} 100.00%      ${std_sum:<11.2f} ${gbr_sum:<11.2f}")
    
    # Verificar principio "todos ganamos o todos perdemos"
    std_min = min(standard_distribution.values())
    std_max = max(standard_distribution.values())
    gbr_min = min(gabriel_distribution.values())
    gbr_max = max(gabriel_distribution.values())
    
    std_ratio = std_max / std_min if std_min > 0 else float('inf')
    gbr_ratio = gbr_max / gbr_min if gbr_min > 0 else float('inf')
    
    logger.info(f"Ratio máx/mín estándar: {std_ratio:.2f}")
    logger.info(f"Ratio máx/mín Gabriel: {gbr_ratio:.2f}")
    
    if gbr_ratio < std_ratio:
        logger.info("✓ Distribución Gabriel es más equitativa (todos ganamos o todos perdemos)")
    else:
        logger.info("✗ Distribución Gabriel no cumple el principio de equidad")

async def test_single_trades(strategy) -> None:
    """
    Probar ejecución de operaciones individuales.
    
    Args:
        strategy: Estrategia integrada
    """
    logger.info("\n=== Prueba de operaciones individuales ===")
    
    # Ejecutar una compra y una venta
    for symbol in TEST_SYMBOLS[:2]:  # Probar con los primeros dos símbolos
        # Probar compra
        result = await strategy.execute_single_trade(
            symbol=symbol,
            side="buy",
            amount=100.0
        )
        
        logger.info(f"\nOperación de compra en {symbol}:")
        if result.get("status") == "rejected_by_gabriel":
            logger.info(f"Rechazada por Gabriel: {result.get('message')}")
        else:
            logger.info(f"Estado: {result.get('status')}")
            if "gabriel_approval" in result:
                approval = result["gabriel_approval"]
                logger.info(f"Aprobada por Gabriel con confianza: {approval.get('confidence', 0):.2f}")
                logger.info(f"Razón: {approval.get('reason', 'No disponible')}")
        
        # Pequeña pausa
        await asyncio.sleep(1)
        
        # Probar venta
        result = await strategy.execute_single_trade(
            symbol=symbol,
            side="sell",
            amount=50.0
        )
        
        logger.info(f"\nOperación de venta en {symbol}:")
        if result.get("status") == "rejected_by_gabriel":
            logger.info(f"Rechazada por Gabriel: {result.get('message')}")
        else:
            logger.info(f"Estado: {result.get('status')}")
            if "gabriel_approval" in result:
                approval = result["gabriel_approval"]
                logger.info(f"Aprobada por Gabriel con confianza: {approval.get('confidence', 0):.2f}")
                logger.info(f"Razón: {approval.get('reason', 'No disponible')}")
        
        # Pequeña pausa
        await asyncio.sleep(1)

async def main():
    """Función principal."""
    logger.info("Iniciando prueba de integración completa del sistema")
    
    try:
        # Configurar componentes
        exchange = await setup_exchange_simulator()
        capital_manager = await setup_capital_manager()
        
        # Inicializar estrategia
        strategy = await initialize_seraphim_strategy(
            capital_base=TEST_CAPITAL,
            symbols=TEST_SYMBOLS,
            archetype="COLLECTIVE",
            capital_manager=capital_manager
        )
        
        # Asignar exchange a la estrategia
        strategy.orchestrator.exchange_adapter = exchange
        
        # Ejecutar pruebas
        await test_market_analysis(strategy)
        await test_trading_cycle(strategy)
        await test_profit_distribution(strategy)
        await test_single_trades(strategy)
        
        # Limpiar recursos
        await strategy.cleanup()
        
        logger.info("\nPrueba de integración completa finalizada con éxito")
        
    except Exception as e:
        logger.error(f"Error en prueba de integración: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())