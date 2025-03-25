"""
Prueba del Motor de Comportamiento Humano Gabriel

Este script demuestra el ciclo completo del motor de comportamiento Gabriel,
mostrando cómo reacciona a diferentes situaciones de mercado y cómo evoluciona
su estado emocional, percepción y toma de decisiones a lo largo del tiempo.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import asyncio
import logging
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

from genesis.trading.gabriel_adapter import GabrielBehaviorEngine

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("test_gabriel")

# Arquetipos de personalidad disponibles
ARQUETIPOS = [
    "BALANCED",    # Equilibrado
    "AGGRESSIVE",  # Agresivo
    "CONSERVATIVE", # Conservador
    "GUARDIAN",    # Guardián
    "EXPLORER",    # Explorador
    "COLLECTIVE"   # Colectivo
]

# Estados del mercado para simulación
MARKET_STATES = [
    {
        "name": "Mercado alcista fuerte",
        "trend": 0.9,
        "volatility": 0.3,
        "sentiment": "bullish",
        "price_change": 0.08,
        "market_fear": 0.2
    },
    {
        "name": "Mercado alcista moderado",
        "trend": 0.6,
        "volatility": 0.4,
        "sentiment": "bullish",
        "price_change": 0.04,
        "market_fear": 0.3
    },
    {
        "name": "Mercado neutral",
        "trend": 0.1,
        "volatility": 0.5,
        "sentiment": "neutral",
        "price_change": 0.01,
        "market_fear": 0.4
    },
    {
        "name": "Mercado volátil indeciso",
        "trend": 0.0,
        "volatility": 0.8,
        "sentiment": "mixed",
        "price_change": -0.02,
        "market_fear": 0.5
    },
    {
        "name": "Mercado bajista moderado",
        "trend": -0.5,
        "volatility": 0.5,
        "sentiment": "bearish",
        "price_change": -0.04,
        "market_fear": 0.7
    },
    {
        "name": "Mercado bajista fuerte",
        "trend": -0.8,
        "volatility": 0.7,
        "sentiment": "bearish",
        "price_change": -0.09,
        "market_fear": 0.9
    },
    {
        "name": "Crash de mercado",
        "trend": -0.95,
        "volatility": 0.9,
        "sentiment": "panic",
        "price_change": -0.15,
        "market_fear": 0.95
    }
]

# Noticias para simulación
MARKET_NEWS = [
    {
        "title": "Nueva regulación positiva para criptomonedas",
        "content": "Gobierno anuncia marco regulatorio favorable que podría impulsar adopción",
        "sentiment": "bullish",
        "importance": 0.8,
        "impact": 0.7
    },
    {
        "title": "Gran empresa adopta Bitcoin como reserva",
        "content": "Importante multinacional añade Bitcoin a su balance",
        "sentiment": "bullish",
        "importance": 0.7,
        "impact": 0.6
    },
    {
        "title": "Resultados mixtos en mercados tradicionales",
        "content": "Bolsas con comportamiento desigual tras datos económicos",
        "sentiment": "neutral",
        "importance": 0.4,
        "impact": 0.2
    },
    {
        "title": "Datos de inflación mejores de lo esperado",
        "content": "Inflación más baja de lo previsto podría disminuir presión sobre mercados",
        "sentiment": "slightly_bullish",
        "importance": 0.6,
        "impact": 0.4
    },
    {
        "title": "Hackeo a exchange importante",
        "content": "Exchange sufre robo de fondos significativo",
        "sentiment": "bearish",
        "importance": 0.9,
        "impact": -0.8,
        "related_to_portfolio": True
    },
    {
        "title": "Propuesta de prohibición de minería",
        "content": "País importante evalúa prohibir minería de criptomonedas",
        "sentiment": "bearish",
        "importance": 0.7,
        "impact": -0.6
    },
    {
        "title": "Bancos centrales estudian monedas digitales",
        "content": "Varios bancos centrales aceleran desarrollo de CBDCs",
        "sentiment": "mixed",
        "importance": 0.5,
        "impact": 0.0
    }
]

# Símbolos para simulación
SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "DOT/USDT"]

async def create_gabriel(archetype: str = "BALANCED") -> GabrielBehaviorEngine:
    """
    Crear una instancia de Gabriel con el arquetipo indicado.
    
    Args:
        archetype: Nombre del arquetipo
        
    Returns:
        Instancia de GabrielBehaviorEngine
    """
    logger.info(f"Creando Gabriel con arquetipo {archetype}")
    gabriel = GabrielBehaviorEngine(archetype=archetype)
    await gabriel.initialize()
    return gabriel

def generate_market_data(market_state: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Generar datos de mercado para simulación.
    
    Args:
        market_state: Estado base del mercado
        symbol: Símbolo para el que generar datos
        
    Returns:
        Datos de mercado simulados
    """
    # Añadir algo de variación para hacer más realista
    volatility_factor = 1.0 + (random.random() - 0.5) * 0.2  # ±10% 
    
    return {
        "symbol": symbol,
        "price": 1000 + random.random() * 50000,  # Precio aleatorio
        "trend": market_state["trend"] * volatility_factor,
        "volatility": market_state["volatility"] * volatility_factor,
        "sentiment": market_state["sentiment"],
        "price_change": market_state["price_change"] * volatility_factor,
        "market_fear": market_state["market_fear"],
        "timestamp": datetime.now().isoformat(),
        "volume": random.random() * 1000000,
        "24h_high": 0,
        "24h_low": 0,
    }

async def test_gabriel_market_states(gabriel: GabrielBehaviorEngine) -> None:
    """
    Probar cómo Gabriel reacciona a diferentes estados de mercado.
    
    Args:
        gabriel: Instancia de Gabriel
    """
    logger.info("=== Prueba de reacción a estados de mercado ===")
    symbol = "BTC/USDT"
    
    for state in MARKET_STATES:
        # Generar datos de mercado para este estado
        market_data = generate_market_data(state, symbol)
        
        # Obtener percepción de Gabriel
        perception = await gabriel.process_market_data(market_data)
        
        # Obtener decisión con señal moderada
        decision, reason, confidence = await gabriel.evaluate_trade_opportunity(
            symbol, 0.6, market_data
        )
        
        # Mostrar resultados
        logger.info(f"\nMercado: {state['name']}")
        logger.info(f"Percepción: {perception['mood']} ({perception['mood_intensity']:.2f})")
        logger.info(f"Perspectiva: {perception['perspective']} (confianza: {perception['confidence']:.2f})")
        logger.info(f"Interpretación: {perception['human_interpretation']}")
        logger.info(f"Decisión de entrada: {'SÍ' if decision else 'NO'} ({confidence:.2f})")
        logger.info(f"Razón: {reason}")

        # Simular tamaño de posición
        if decision:
            size = await gabriel.adjust_position_size(1000, 10000, {"volatility": market_data["volatility"]})
            logger.info(f"Tamaño ajustado: ${size:.2f} (base: $1000)")
        
        # Pequeña pausa entre estados
        await asyncio.sleep(0.5)

async def test_gabriel_news_impact(gabriel: GabrielBehaviorEngine) -> None:
    """
    Probar el impacto de noticias en el estado emocional de Gabriel.
    
    Args:
        gabriel: Instancia de Gabriel
    """
    logger.info("\n=== Prueba de reacción a noticias ===")
    
    # Estado inicial
    initial_state = await gabriel.get_emotional_state()
    logger.info(f"Estado inicial: {initial_state['mood']} ({initial_state['mood_intensity']:.2f})")
    
    # Procesar varias noticias
    for news in MARKET_NEWS:
        logger.info(f"\nNoticia: {news['title']}")
        logger.info(f"Sentimiento: {news['sentiment']}, Importancia: {news['importance']}")
        
        # Estado antes de la noticia
        before_state = await gabriel.get_emotional_state()
        
        # Procesar noticia
        await gabriel.process_news(news)
        
        # Estado después de la noticia
        after_state = await gabriel.get_emotional_state()
        
        # Mostrar cambio
        logger.info(f"Cambio de estado: {before_state['mood']} -> {after_state['mood']}")
        logger.info(f"Cambio de intensidad: {before_state['mood_intensity']:.2f} -> {after_state['mood_intensity']:.2f}")
        
        # Pequeña pausa entre noticias
        await asyncio.sleep(0.5)

async def test_gabriel_market_cycle(gabriel: GabrielBehaviorEngine) -> None:
    """
    Simular un ciclo completo de mercado y observar cómo Gabriel evoluciona.
    
    Args:
        gabriel: Instancia de Gabriel
    """
    logger.info("\n=== Simulación de ciclo de mercado ===")
    symbol = "BTC/USDT"
    total_days = 30
    entry_price = None
    entry_day = None
    capital = 10000
    
    # Crear ciclo de mercado: subida, meseta, bajada
    cycle = []
    # Fase alcista inicial (10 días)
    for i in range(10):
        factor = i / 10  # 0.0 a 0.9
        cycle.append({
            "day": i + 1,
            "trend": 0.3 + factor * 0.6,  # 0.3 a 0.9
            "volatility": 0.3 + factor * 0.2,  # 0.3 a 0.5
            "sentiment": "bullish",
            "price_change": 0.02 + factor * 0.06,  # 2% a 8%
            "market_fear": 0.3 - factor * 0.1,  # 0.3 a 0.2
            "price": 40000 * (1 + 0.05 * i)  # Precio subiendo 5% cada día
        })
    
    # Fase de meseta (10 días)
    for i in range(10):
        cycle.append({
            "day": i + 11,
            "trend": 0.3 - i * 0.06,  # 0.3 a -0.3
            "volatility": 0.5 + i * 0.02,  # 0.5 a 0.7
            "sentiment": "mixed" if i < 5 else "slightly_bearish",
            "price_change": 0.01 - i * 0.005,  # 1% a -4%
            "market_fear": 0.3 + i * 0.05,  # 0.3 a 0.8
            "price": 40000 * (1 + 0.05 * 10) * (1 + (0.01 - i * 0.005))
        })
    
    # Fase bajista (10 días)
    for i in range(10):
        cycle.append({
            "day": i + 21,
            "trend": -0.3 - i * 0.06,  # -0.3 a -0.9
            "volatility": 0.7 + i * 0.02,  # 0.7 a 0.9
            "sentiment": "bearish",
            "price_change": -0.04 - i * 0.01,  # -4% a -14%
            "market_fear": 0.8 + i * 0.02,  # 0.8 a 1.0
            "price": 40000 * (1 + 0.05 * 10) * (1 - 0.04) * (1 - 0.02 * i)
        })
    
    # Simular paso de días
    position_open = False
    profit = 0
    
    for day_data in cycle:
        logger.info(f"\n--- Día {day_data['day']} ---")
        logger.info(f"Precio: ${day_data['price']:.2f}, Cambio: {day_data['price_change']*100:.1f}%")
        
        # Generar datos de mercado
        market_data = {
            "symbol": symbol,
            "price": day_data["price"],
            "trend": day_data["trend"],
            "volatility": day_data["volatility"],
            "sentiment": day_data["sentiment"],
            "price_change": day_data["price_change"],
            "market_fear": day_data["market_fear"],
            "timestamp": (datetime.now() + timedelta(days=day_data["day"])).isoformat(),
        }
        
        # Obtener estado emocional
        emotional_state = await gabriel.get_emotional_state()
        logger.info(f"Estado emocional: {emotional_state['mood']} ({emotional_state['mood_intensity']:.2f})")
        
        # Si no hay posición abierta, evaluar entrada
        if not position_open:
            signal_strength = 0.6 + day_data["trend"] * 0.2  # Señal de entrada basada en tendencia
            decision, reason, confidence = await gabriel.evaluate_trade_opportunity(
                symbol, signal_strength, market_data
            )
            
            if decision:
                # Abrir posición
                position_open = True
                entry_price = day_data["price"]
                entry_day = day_data["day"]
                
                # Calcular tamaño
                position_size = await gabriel.adjust_position_size(
                    1000, capital, {"volatility": market_data["volatility"]}
                )
                
                # Número de unidades
                units = position_size / entry_price
                
                logger.info(f"ENTRADA en ${entry_price:.2f} - {units:.5f} unidades (${position_size:.2f})")
                logger.info(f"Razón: {reason} (confianza: {confidence:.2f})")
        
        # Si hay posición abierta, evaluar salida
        else:
            current_price = day_data["price"]
            days_held = day_data["day"] - entry_day
            profit_pct = (current_price - entry_price) / entry_price
            
            position_data = {
                "symbol": symbol,
                "entry_price": entry_price,
                "current_price": current_price,
                "profit_percent": profit_pct,
                "days_held": days_held,
                "entry_time": (datetime.now() + timedelta(days=entry_day)).isoformat()
            }
            
            exit_decision, reason = await gabriel.evaluate_exit_opportunity(
                position_data, market_data
            )
            
            logger.info(f"Posición: {days_held} días, P/L: {profit_pct*100:.2f}%")
            
            if exit_decision:
                # Cerrar posición
                position_open = False
                profit += profit_pct
                logger.info(f"SALIDA en ${current_price:.2f} - P/L: {profit_pct*100:.2f}%")
                logger.info(f"Razón: {reason}")
        
        # Pequeña pausa entre días
        await asyncio.sleep(0.2)
    
    # Resultados finales
    logger.info("\n=== Resultados del ciclo de mercado ===")
    logger.info(f"Ganancia/Pérdida acumulada: {profit*100:.2f}%")
    
    # Estado final
    final_state = await gabriel.get_emotional_state()
    logger.info(f"Estado emocional final: {final_state['mood']} ({final_state['mood_intensity']:.2f})")

async def test_gabriel_comparative(arquetipos: List[str]) -> None:
    """
    Comparar diferentes arquetipos de Gabriel en las mismas condiciones.
    
    Args:
        arquetipos: Lista de arquetipos a comparar
    """
    logger.info("\n=== Comparativa de arquetipos ===")
    symbol = "BTC/USDT"
    
    # Seleccionar algunos estados de mercado representativos
    test_states = [
        MARKET_STATES[1],  # Alcista moderado
        MARKET_STATES[3],  # Volátil indeciso
        MARKET_STATES[5]   # Bajista fuerte
    ]
    
    results = {}
    
    for archetype in arquetipos:
        # Crear Gabriel con este arquetipo
        gabriel = await create_gabriel(archetype)
        
        # Resultados para este arquetipo
        archetype_results = []
        
        # Probar cada estado
        for state in test_states:
            market_data = generate_market_data(state, symbol)
            
            # Evaluar decisión
            decision, reason, confidence = await gabriel.evaluate_trade_opportunity(
                symbol, 0.6, market_data
            )
            
            # Si decidió entrar, calcular tamaño
            size = 0
            if decision:
                size = await gabriel.adjust_position_size(1000, 10000, {"volatility": market_data["volatility"]})
            
            # Percepción
            perception = await gabriel.process_market_data(market_data)
            
            # Almacenar resultados
            archetype_results.append({
                "market_state": state["name"],
                "decision": decision,
                "confidence": confidence,
                "size": size,
                "mood": perception["mood"],
                "mood_intensity": perception["mood_intensity"],
                "perspective": perception["perspective"]
            })
        
        # Guardar resultados de este arquetipo
        results[archetype] = archetype_results
    
    # Mostrar resultados comparativos
    for state_idx, state in enumerate(test_states):
        logger.info(f"\n--- {state['name']} ---")
        
        for archetype in arquetipos:
            result = results[archetype][state_idx]
            decision_text = "ENTRAR" if result["decision"] else "NO ENTRAR"
            logger.info(f"{archetype}: {decision_text} (confianza: {result['confidence']:.2f})")
            
            if result["decision"]:
                logger.info(f"  Tamaño: ${result['size']:.2f}")
            
            logger.info(f"  Estado: {result['mood']} ({result['mood_intensity']:.2f}), Perspectiva: {result['perspective']}")

async def test_emergency_mode(gabriel: GabrielBehaviorEngine) -> None:
    """
    Probar modo de emergencia de Gabriel.
    
    Args:
        gabriel: Instancia de Gabriel
    """
    logger.info("\n=== Prueba de modo de emergencia ===")
    symbol = "BTC/USDT"
    
    # Estado inicial
    initial_state = await gabriel.get_emotional_state()
    logger.info(f"Estado inicial: {initial_state['mood']} ({initial_state['mood_intensity']:.2f})")
    
    # Activar modo de emergencia
    await gabriel.set_emergency_mode("market_crash")
    
    # Estado después de activar emergencia
    emergency_state = await gabriel.get_emotional_state()
    logger.info(f"Estado en emergencia: {emergency_state['mood']} ({emergency_state['mood_intensity']:.2f})")
    
    # Probar decisión en modo emergencia
    market_data = generate_market_data(MARKET_STATES[2], symbol)  # Mercado neutral
    decision, reason, confidence = await gabriel.evaluate_trade_opportunity(
        symbol, 0.7, market_data  # Señal fuerte
    )
    
    logger.info(f"Decisión en emergencia: {'SÍ' if decision else 'NO'} ({confidence:.2f})")
    logger.info(f"Razón: {reason}")
    
    # Normalizar estado
    await gabriel.normalize_state()
    
    # Estado después de normalizar
    normal_state = await gabriel.get_emotional_state()
    logger.info(f"Estado tras normalizar: {normal_state['mood']} ({normal_state['mood_intensity']:.2f})")

async def main():
    """Función principal."""
    logger.info("Iniciando pruebas del Motor de Comportamiento Humano Gabriel")
    
    # Crear Gabriel con arquetipo balanceado para pruebas iniciales
    gabriel = await create_gabriel("BALANCED")
    
    # Probar diferentes estados de mercado
    await test_gabriel_market_states(gabriel)
    
    # Probar impacto de noticias
    await test_gabriel_news_impact(gabriel)
    
    # Probar ciclo completo
    await test_gabriel_market_cycle(gabriel)
    
    # Probar modo de emergencia
    await test_emergency_mode(gabriel)
    
    # Comparar diferentes arquetipos
    await test_gabriel_comparative(["BALANCED", "AGGRESSIVE", "CONSERVATIVE", "COLLECTIVE"])
    
    logger.info("\nPruebas de Gabriel completadas")

if __name__ == "__main__":
    asyncio.run(main())