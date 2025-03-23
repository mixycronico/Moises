"""
Script para probar la integración y operación de DeepSeek en el Sistema Genesis.

Este script permite activar, desactivar y probar las capacidades de DeepSeek,
mostrando cómo afecta al análisis y las decisiones de trading.
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from genesis.lsml import deepseek_config
from genesis.lsml.deepseek_model import DeepSeekModel
from genesis.lsml.deepseek_integrator import DeepSeekIntegrator

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('api_integration')

# Datos de prueba para el análisis
SAMPLE_MARKET_DATA = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "timestamp": datetime.now().timestamp(),
    "open": 63500.0,
    "high": 63800.0,
    "low": 63100.0,
    "close": 63650.0,
    "volume": 1200.5,
    "indicators": {
        "rsi": 58.5,
        "macd": {
            "macd": 120.5,
            "signal": 110.2,
            "histogram": 10.3
        },
        "bbands": {
            "upper": 64200.0,
            "middle": 63500.0,
            "lower": 62800.0
        },
        "ema_9": 63400.0,
        "ema_21": 63100.0,
        "atr": 350.0
    },
    "market_state": {
        "trend": "bullish",
        "volatility": "medium",
        "volume_profile": "increasing"
    }
}

SAMPLE_NEWS_DATA = [
    {
        "title": "Bitcoin reaches new all-time high above $73,000",
        "source": "CoinDesk",
        "published_at": "2025-03-22T12:34:56Z",
        "summary": "Bitcoin has reached a new all-time high, breaking above $73,000 for the first time in history.",
        "sentiment": "positive",
        "relevance": 0.95
    },
    {
        "title": "SEC approves spot Ethereum ETFs",
        "source": "Bloomberg",
        "published_at": "2025-03-21T09:15:30Z",
        "summary": "The U.S. Securities and Exchange Commission has approved applications for spot Ethereum ETFs.",
        "sentiment": "positive",
        "relevance": 0.9
    },
    {
        "title": "Major exchange reports security breach",
        "source": "CryptoNews",
        "published_at": "2025-03-20T17:45:20Z",
        "summary": "A major cryptocurrency exchange reported a security breach affecting some user accounts.",
        "sentiment": "negative",
        "relevance": 0.8
    }
]

async def print_config_status():
    """Mostrar estado actual de la configuración DeepSeek."""
    state = deepseek_config.get_state()
    print("\n=== ESTADO ACTUAL DE DEEPSEEK ===")
    print(f"Habilitado: {state['enabled']}")
    print(f"Inicializado: {state['initialized']}")
    print(f"API Key disponible: {state['api_key_available']}")
    print(f"Modelo: {state['config']['model_version']}")
    print(f"Factor de inteligencia: {state['config']['intelligence_factor']}")
    print(f"Última actualización: {state['config']['last_updated']}")
    print(f"Estadísticas: {json.dumps(state['stats'], indent=2)}")
    print("================================\n")

async def toggle_deepseek():
    """Alternar el estado (activar/desactivar) de DeepSeek."""
    new_state = deepseek_config.toggle()
    print(f"\n>>> DeepSeek {'ACTIVADO' if new_state else 'DESACTIVADO'} <<<\n")
    await print_config_status()

async def test_analysis(integrator: DeepSeekIntegrator):
    """Probar análisis de mercado con DeepSeek."""
    print("\n=== EJECUTANDO ANÁLISIS DE OPORTUNIDADES DE TRADING ===")
    result = await integrator.analyze_trading_opportunities(
        market_data=SAMPLE_MARKET_DATA,
        news_data=SAMPLE_NEWS_DATA
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        if result.get('status') == 'disabled':
            print("DeepSeek está desactivado. Utiliza toggle_deepseek() para activarlo.")
        return
    
    print("\nResumen de análisis:")
    if "market_analysis" in result:
        market = result["market_analysis"]
        print(f"- Tendencia: {market.get('trend', 'N/A')}")
        print(f"- Fortaleza: {market.get('strength', 'N/A')}")
        print(f"- Riesgo: {market.get('risk_assessment', 'N/A')}")
    
    if "sentiment_analysis" in result and result["sentiment_analysis"]:
        sentiment = result["sentiment_analysis"]
        print(f"- Sentimiento: {sentiment.get('sentiment', 'N/A')}")
        print(f"- Puntaje: {sentiment.get('sentiment_score', 'N/A')}")
    
    if "combined_recommendations" in result:
        recs = result["combined_recommendations"]
        print(f"\nRecomendaciones ({len(recs)}):")
        for i, rec in enumerate(recs[:3], 1):  # Mostrar solo las primeras 3
            print(f"  {i}. {rec.get('action', 'N/A')} {rec.get('symbol', 'N/A')} - {rec.get('rationale', 'N/A')[:100]}...")
        
        if len(recs) > 3:
            print(f"  ... y {len(recs) - 3} más")
    
    print("================================================\n")

async def process_deepseek_integration():
    """Ejecutar prueba completa de integración con DeepSeek."""
    try:
        # 1. Mostrar configuración actual
        await print_config_status()
        
        # 2. Crear integrador
        integrator = DeepSeekIntegrator()
        
        # 3. Probar con DeepSeek desactivado
        print("\n--- PRUEBA CON DEEPSEEK DESACTIVADO ---")
        deepseek_config.disable()
        await test_analysis(integrator)
        
        # 4. Probar con DeepSeek activado
        print("\n--- PRUEBA CON DEEPSEEK ACTIVADO ---")
        deepseek_config.enable()
        await test_analysis(integrator)
        
        # 5. Cambiar factor de inteligencia
        print("\n--- PRUEBA CON FACTOR DE INTELIGENCIA AUMENTADO ---")
        deepseek_config.set_intelligence_factor(5.0)
        # Reiniciar el integrador para aplicar cambios
        await integrator.close()
        integrator = DeepSeekIntegrator()
        await test_analysis(integrator)
        
        # 6. Limpiar
        await integrator.close()
        print("\nPrueba de integración completada.")
        
    except Exception as e:
        logger.error(f"Error durante la prueba de integración: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(process_deepseek_integration())