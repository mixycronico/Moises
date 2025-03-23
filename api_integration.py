"""
Script para probar la integración y operación de APIs externas en el Sistema Genesis.

Este script permite activar, desactivar y probar las capacidades de múltiples APIs:
- DeepSeek: Análisis avanzado con IA de texto
- Alpha Vantage: Datos históricos y fundamentales
- NewsAPI: Noticias y eventos
- CoinMarketCap: Información de mercado
- Reddit: Análisis de sentimiento social
"""

import asyncio
import logging
import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

# Importar gestor unificado de APIs
from genesis.api_integration import api_manager, initialize, test_apis

# Importaciones específicas para DeepSeek (integración existente)
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


async def print_api_status():
    """Mostrar estado de todas las APIs configuradas."""
    # Inicializar el gestor de APIs
    await initialize()
    
    # Obtener estado actual
    status = api_manager.get_api_status()
    
    print("\n=== ESTADO DE APIS CONFIGURADAS ===")
    for api_name, api_status in status.items():
        print(f"\n{api_name.upper()}:")
        for key, value in api_status.items():
            # No mostrar la clave enmascarada a menos que esté configurada
            if key == "key_masked" and not api_status.get("key_configured", False):
                continue
            print(f"  {key}: {value}")
    
    print("\n=================================\n")


async def test_all_apis():
    """Probar todas las APIs configuradas."""
    print("\n=== PROBANDO TODAS LAS APIS CONFIGURADAS ===\n")
    
    results = await test_apis()
    
    for api_name, result in results.items():
        if api_name == "api_status":
            continue
            
        print(f"\n--- Resultado para {api_name.upper()} ---")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(json.dumps(result, indent=2, default=str)[:500] + "...")
    
    print("\n=== FIN DE PRUEBAS DE APIS ===\n")


async def main(args):
    """Función principal."""
    try:
        if args.all:
            # Probar todas las integraciones
            await print_api_status()
            await test_all_apis()
            await process_deepseek_integration()
        elif args.deepseek:
            # Solo probar DeepSeek
            await process_deepseek_integration()
        elif args.status:
            # Mostrar estado de APIs
            await print_api_status()
        elif args.test:
            # Probar APIs configuradas
            await test_all_apis()
        else:
            # Mostrar ayuda si no se especifica acción
            print("Especifique una acción con --all, --deepseek, --status o --test")
            print("Ejecute con --help para más información")
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        raise
    finally:
        # Cerrar conexiones
        if hasattr(api_manager, 'session') and api_manager.session:
            await api_manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sistema de integración de APIs externas para Genesis')
    
    # Acciones exclusivas
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true', help='Ejecutar todas las pruebas de integración')
    group.add_argument('--deepseek', action='store_true', help='Probar solo la integración con DeepSeek')
    group.add_argument('--status', action='store_true', help='Mostrar estado de todas las APIs configuradas')
    group.add_argument('--test', action='store_true', help='Probar APIs configuradas')
    
    args = parser.parse_args()
    
    # Si no se especifica ninguna acción, mostrar estado por defecto
    if not (args.all or args.deepseek or args.status or args.test):
        args.status = True
    
    asyncio.run(main(args))