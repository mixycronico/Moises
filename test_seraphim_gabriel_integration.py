"""
Prueba de Integración entre Gabriel y Seraphim

Este script demuestra la integración entre el motor de comportamiento humano Gabriel
y el orquestador Seraphim, con énfasis en el principio "todos ganamos o todos perdemos".

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import asyncio
import logging
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from genesis.trading.seraphim_integration import SeraphimGabrielIntegrator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("test_seraphim_gabriel")

# Crear inversores simulados para prueba
INVESTORS = {
    "investor_1": {
        "name": "Juan García",
        "capital": 5000.0,
        "risk_profile": "moderate"
    },
    "investor_2": {
        "name": "María López",
        "capital": 10000.0,
        "risk_profile": "conservative"
    },
    "investor_3": {
        "name": "Pedro Martínez",
        "capital": 3000.0,
        "risk_profile": "aggressive"
    },
    "investor_4": {
        "name": "Ana Fernández",
        "capital": 1500.0,
        "risk_profile": "moderate"
    },
    "investor_5": {
        "name": "Carlos Rodríguez",
        "capital": 15000.0,
        "risk_profile": "moderate"
    }
}

# Símbolos para simulación
SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "DOT/USDT"]

async def create_integrator(archetype: str = "COLLECTIVE") -> SeraphimGabrielIntegrator:
    """
    Crear instancia del integrador Gabriel-Seraphim.
    
    Args:
        archetype: Nombre del arquetipo
        
    Returns:
        Instancia del integrador
    """
    logger.info(f"Creando integrador con arquetipo {archetype}")
    integrator = SeraphimGabrielIntegrator(archetype=archetype)
    await integrator.initialize()
    return integrator

def generate_market_data(symbol: str, trend: float = 0.0) -> Dict[str, Any]:
    """
    Generar datos de mercado simulados.
    
    Args:
        symbol: Símbolo del mercado
        trend: Tendencia (-1 a 1)
        
    Returns:
        Datos de mercado simulados
    """
    volatility = random.uniform(0.3, 0.8)
    sentiment = "bullish" if trend > 0.3 else "bearish" if trend < -0.3 else "neutral"
    price_change = trend * random.uniform(0.5, 1.5)  # Variación aleatoria de la tendencia
    
    return {
        "symbol": symbol,
        "price": 1000 + random.random() * 50000,  # Precio aleatorio
        "trend": trend,
        "volatility": volatility,
        "sentiment": sentiment,
        "price_change": price_change,
        "market_fear": max(0, min(1, 0.5 - trend * 0.5)),  # Más miedo cuando la tendencia es negativa
        "timestamp": datetime.now().isoformat(),
        "volume": random.random() * 1000000,
        "24h_high": 0,
        "24h_low": 0,
    }

async def test_market_reaction(integrator: SeraphimGabrielIntegrator) -> None:
    """
    Probar reacción a condiciones de mercado.
    
    Args:
        integrator: Instancia del integrador
    """
    logger.info("\n=== Prueba de reacción a condiciones de mercado ===")
    symbol = "BTC/USDT"
    
    # Probar diferentes tendencias
    trends = [-0.8, -0.4, 0.0, 0.4, 0.8]
    
    for trend in trends:
        # Generar datos de mercado
        market_data = generate_market_data(symbol, trend)
        
        # Procesar con integrador
        perception = await integrator.process_market_data(symbol, market_data)
        
        # Evaluar decisión original (simulada como basada en tendencia)
        original_decision = trend > 0.0  # Decisión simple basada en tendencia
        signal_strength = 0.5 + abs(trend) * 0.5  # Señal más fuerte cuando la tendencia es más clara
        
        # Obtener decisión de Gabriel
        decision, reason, confidence = await integrator.evaluate_trade_decision(
            original_decision, symbol, signal_strength, market_data
        )
        
        # Mostrar resultados
        trend_description = "muy alcista" if trend > 0.6 else \
                           "alcista" if trend > 0.2 else \
                           "neutral" if trend > -0.2 else \
                           "bajista" if trend > -0.6 else "muy bajista"
        
        logger.info(f"\nMercado {trend_description} (tendencia: {trend:.2f})")
        logger.info(f"Percepción: {perception['mood']} ({perception['mood_intensity']:.2f})")
        logger.info(f"Decisión original: {'ENTRAR' if original_decision else 'NO ENTRAR'}")
        logger.info(f"Decisión final: {'ENTRAR' if decision else 'NO ENTRAR'} (confianza: {confidence:.2f})")
        logger.info(f"Razón: {reason}")
        
        # Si decidió entrar, calcular tamaño
        if decision:
            adjusted_size = await integrator.adjust_position_size(
                1000.0,  # Tamaño base
                10000.0,  # Capital
                market_data["volatility"],
                0.05  # Máximo riesgo 5%
            )
            logger.info(f"Tamaño ajustado: ${adjusted_size:.2f} (base: $1000)")
        
        # Pequeña pausa entre pruebas
        await asyncio.sleep(0.5)

async def test_profit_distribution(integrator: SeraphimGabrielIntegrator) -> None:
    """
    Probar distribución de ganancias/pérdidas siguiendo el principio "todos ganamos o todos perdemos".
    
    Args:
        integrator: Instancia del integrador
    """
    logger.info("\n=== Prueba de distribución de ganancias/pérdidas ===")
    
    # Probar con diferentes escenarios de ganancia/pérdida
    scenarios = [
        {"name": "Ganancia pequeña", "profit": 500.0},
        {"name": "Ganancia sustancial", "profit": 2000.0},
        {"name": "Pérdida pequeña", "profit": -300.0},
        {"name": "Pérdida sustancial", "profit": -1500.0}
    ]
    
    total_capital = sum(investor["capital"] for investor in INVESTORS.values())
    logger.info(f"Capital total del pool: ${total_capital:.2f}")
    
    for scenario in scenarios:
        profit = scenario["profit"]
        profit_percent = profit / total_capital
        
        logger.info(f"\nEscenario: {scenario['name']} (${profit:.2f}, {profit_percent*100:.2f}%)")
        
        # Calcular distribución estándar (proporcional)
        standard_distribution = {
            investor_id: (data["capital"] / total_capital) * profit
            for investor_id, data in INVESTORS.items()
        }
        
        # Obtener distribución según el principio "todos ganamos o todos perdemos"
        seraphim_distribution = await integrator.distribute_profits(profit, INVESTORS)
        
        # Mostrar comparación
        logger.info("Comparación de distribuciones:")
        logger.info(f"{'Inversor':<15} {'Capital':<10} {'% del Pool':<12} {'Estándar':<12} {'Seraphim':<12} {'Diferencia':<12}")
        logger.info("-" * 75)
        
        for investor_id, data in INVESTORS.items():
            capital = data["capital"]
            pool_percent = (capital / total_capital) * 100
            std_profit = standard_distribution[investor_id]
            ser_profit = seraphim_distribution[investor_id]
            difference = ser_profit - std_profit
            
            logger.info(f"{data['name']:<15} ${capital:<9.2f} {pool_percent:<11.2f}% ${std_profit:<11.2f} ${ser_profit:<11.2f} ${difference:<11.2f}")
        
        # Verificar que la suma es correcta
        std_sum = sum(standard_distribution.values())
        ser_sum = sum(seraphim_distribution.values())
        
        logger.info("-" * 75)
        logger.info(f"{'TOTAL':<15} ${total_capital:<9.2f} 100.00%      ${std_sum:<11.2f} ${ser_sum:<11.2f}")
        
        # Crear y guardar gráfica para visualizar diferencias
        try:
            plt.figure(figsize=(12, 6))
            
            # Preparar datos
            investor_names = [INVESTORS[inv_id]["name"] for inv_id in INVESTORS]
            std_values = [standard_distribution[inv_id] for inv_id in INVESTORS]
            ser_values = [seraphim_distribution[inv_id] for inv_id in INVESTORS]
            
            # Configurar barras
            x = np.arange(len(investor_names))
            width = 0.35
            
            # Crear barras
            plt.bar(x - width/2, std_values, width, label='Distribución Estándar')
            plt.bar(x + width/2, ser_values, width, label='Distribución "Todos Ganamos/Perdemos"')
            
            # Añadir detalles
            plt.xlabel('Inversores')
            plt.ylabel('Ganancia/Pérdida ($)')
            plt.title(f'Comparación de Distribución - {scenario["name"]}')
            plt.xticks(x, investor_names, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Guardar gráfica
            plt.savefig(f"seraphim_distribution_{scenario['name'].replace(' ', '_').lower()}.png")
            plt.close()
            
            logger.info(f"Gráfica guardada: seraphim_distribution_{scenario['name'].replace(' ', '_').lower()}.png")
        except Exception as e:
            logger.error(f"Error al crear gráfica: {str(e)}")
        
        # Pequeña pausa entre escenarios
        await asyncio.sleep(0.5)

async def test_cycle_feedback(integrator: SeraphimGabrielIntegrator) -> None:
    """
    Probar retroalimentación de ciclos de trading en el estado emocional.
    
    Args:
        integrator: Instancia del integrador
    """
    logger.info("\n=== Prueba de retroalimentación de ciclos ===")
    
    # Obtener estado inicial
    initial_state = await integrator.get_current_state()
    initial_mood = initial_state["emotional_state"]["mood"]
    initial_intensity = initial_state["emotional_state"]["mood_intensity"]
    
    logger.info(f"Estado emocional inicial: {initial_mood} ({initial_intensity:.2f})")
    
    # Simular ciclos con diferentes resultados
    cycle_results = [
        {"profit": 0.05, "description": "Ganancia moderada"},
        {"profit": -0.02, "description": "Pequeña pérdida"},
        {"profit": 0.12, "description": "Ganancia sustancial"},
        {"profit": -0.08, "description": "Pérdida importante"}
    ]
    
    for i, cycle in enumerate(cycle_results):
        profit = cycle["profit"]
        description = cycle["description"]
        
        logger.info(f"\nCiclo {i+1}: {description} ({profit*100:.2f}%)")
        
        # Datos adicionales del ciclo
        cycle_data = {
            "duration_hours": random.randint(24, 72),
            "total_trades": random.randint(5, 20),
            "winning_trades": random.randint(3, 15),
            "market_volatility": random.uniform(0.3, 0.7)
        }
        
        # Procesar resultado del ciclo
        recommendations = await integrator.process_cycle_result(profit, cycle_data)
        
        # Mostrar cambio emocional
        state = await integrator.get_current_state()
        mood = state["emotional_state"]["mood"]
        intensity = state["emotional_state"]["mood_intensity"]
        
        logger.info(f"Estado emocional: {mood} ({intensity:.2f})")
        logger.info(f"Ajuste de asignación recomendado: {recommendations['allocation_adjustment']:.2f}x")
        logger.info(f"Ajuste de riesgo recomendado: {recommendations['risk_adjustment']:.2f}x")
        
        if recommendations["recommended_actions"]:
            logger.info("Acciones recomendadas:")
            for action in recommendations["recommended_actions"]:
                logger.info(f"- {action}")
        
        # Pequeña pausa entre ciclos
        await asyncio.sleep(0.5)
    
    # Estado final
    final_state = await integrator.get_current_state()
    final_mood = final_state["emotional_state"]["mood"]
    final_intensity = final_state["emotional_state"]["mood_intensity"]
    
    logger.info(f"\nEstado emocional final: {final_mood} ({final_intensity:.2f})")
    logger.info(f"Cambio desde el inicio: {initial_mood} -> {final_mood}")

async def test_emergency_mode(integrator: SeraphimGabrielIntegrator) -> None:
    """
    Probar activación de modo de emergencia y su impacto en decisiones.
    
    Args:
        integrator: Instancia del integrador
    """
    logger.info("\n=== Prueba de modo de emergencia ===")
    symbol = "BTC/USDT"
    
    # Estado inicial
    initial_state = await integrator.get_current_state()
    initial_mood = initial_state["emotional_state"]["mood"]
    initial_intensity = initial_state["emotional_state"]["mood_intensity"]
    initial_risk = initial_state["risk_profile"]["risk_appetite"]
    
    logger.info(f"Estado inicial: {initial_mood} ({initial_intensity:.2f}), Apetito de riesgo: {initial_risk:.2f}")
    
    # Datos de mercado positivos
    market_data = generate_market_data(symbol, 0.5)  # Tendencia alcista
    signal_strength = 0.7  # Señal fuerte
    
    # Decisión normal
    original_decision = True  # Sistema recomienda entrar
    normal_decision, normal_reason, normal_confidence = await integrator.evaluate_trade_decision(
        original_decision, symbol, signal_strength, market_data
    )
    
    # Tamaño normal
    normal_size = 0
    if normal_decision:
        normal_size = await integrator.adjust_position_size(
            1000.0, 10000.0, market_data["volatility"], 0.05
        )
    
    logger.info(f"\nDecisión normal: {'ENTRAR' if normal_decision else 'NO ENTRAR'} (confianza: {normal_confidence:.2f})")
    if normal_decision:
        logger.info(f"Tamaño normal: ${normal_size:.2f}")
    
    # Activar modo de emergencia
    await integrator.set_emergency_mode("market_crash")
    
    # Obtener estado en emergencia
    emergency_state = await integrator.get_current_state()
    emergency_mood = emergency_state["emotional_state"]["mood"]
    emergency_intensity = emergency_state["emotional_state"]["mood_intensity"]
    emergency_risk = emergency_state["risk_profile"]["risk_appetite"]
    
    logger.info(f"\nEstado en emergencia: {emergency_mood} ({emergency_intensity:.2f}), Apetito de riesgo: {emergency_risk:.2f}")
    
    # Decisión en emergencia
    emergency_decision, emergency_reason, emergency_confidence = await integrator.evaluate_trade_decision(
        original_decision, symbol, signal_strength, market_data
    )
    
    # Tamaño en emergencia
    emergency_size = 0
    if emergency_decision:
        emergency_size = await integrator.adjust_position_size(
            1000.0, 10000.0, market_data["volatility"], 0.05
        )
    
    logger.info(f"Decisión emergencia: {'ENTRAR' if emergency_decision else 'NO ENTRAR'} (confianza: {emergency_confidence:.2f})")
    if emergency_decision:
        logger.info(f"Tamaño emergencia: ${emergency_size:.2f}")
    
    # Mostrar impacto
    size_impact = ((emergency_size / normal_size) - 1) * 100 if normal_size > 0 and emergency_size > 0 else 0
    decision_change = normal_decision != emergency_decision
    
    if decision_change:
        logger.info(f"Impacto: Cambio de decisión de {'ENTRAR' if normal_decision else 'NO ENTRAR'} a {'ENTRAR' if emergency_decision else 'NO ENTRAR'}")
    elif emergency_decision and normal_decision:
        logger.info(f"Impacto: Reducción del tamaño en {abs(size_impact):.2f}%")
    
    # Resetear estado
    await integrator.reset_state()
    
    # Verificar normalización
    reset_state = await integrator.get_current_state()
    reset_mood = reset_state["emotional_state"]["mood"]
    reset_intensity = reset_state["emotional_state"]["mood_intensity"]
    
    logger.info(f"\nEstado tras reset: {reset_mood} ({reset_intensity:.2f})")

async def main():
    """Función principal."""
    logger.info("Iniciando pruebas de integración Gabriel-Seraphim")
    
    # Crear integrador con arquetipo colectivo
    integrator = await create_integrator("COLLECTIVE")
    
    # Probar reacción a mercado
    await test_market_reaction(integrator)
    
    # Probar distribución de ganancias/pérdidas
    await test_profit_distribution(integrator)
    
    # Probar retroalimentación de ciclos
    await test_cycle_feedback(integrator)
    
    # Probar modo de emergencia
    await test_emergency_mode(integrator)
    
    logger.info("\nPruebas de integración completadas")

if __name__ == "__main__":
    asyncio.run(main())