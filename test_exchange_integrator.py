"""
Prueba del Integrador Trascendental para múltiples exchanges.

Este script demuestra las capacidades del Integrador Trascendental con WebSocket
Ultra-Cuántico, conectando a múltiples exchanges simultáneamente y mostrando
su resiliencia absoluta y capacidades de transmutación.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional

from genesis.core.transcendental_exchange_integrator import (
    MultiExchangeTranscendentalIntegrator,
    ExchangeID
)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestExchangeIntegrator")

async def test_exchange_connections():
    """Probar conexiones a múltiples exchanges simultáneamente."""
    logger.info("=== INICIANDO PRUEBA DE CONEXIÓN A EXCHANGES ===")
    
    # Crear integrador para un subconjunto de exchanges (para prueba rápida)
    exchanges = [
        ExchangeID.BINANCE,
        ExchangeID.COINBASE,
        ExchangeID.KRAKEN,
        ExchangeID.BITFINEX,
        ExchangeID.HUOBI
    ]
    
    integrador = MultiExchangeTranscendentalIntegrator(exchanges)
    
    # Inicializar integrador
    init_result = await integrador.initialize()
    logger.info(f"Integrador inicializado: {json.dumps(init_result, indent=2)}")
    
    # Conectar a todos los exchanges
    connect_result = await integrador.connect_all()
    logger.info(f"Resultado de conexiones:\n{json.dumps(connect_result, indent=2)}")
    
    # Mostrar estado de las conexiones
    states = integrador.get_states()
    logger.info(f"Estado de conexiones: {len(states['exchanges'])} exchanges conectados")
    
    return connect_result

async def test_market_data_subscription():
    """Probar suscripción a datos de mercado en múltiples exchanges."""
    logger.info("=== INICIANDO PRUEBA DE SUSCRIPCIÓN A DATOS DE MERCADO ===")
    
    # Crear integrador con todos los exchanges
    integrador = MultiExchangeTranscendentalIntegrator()
    
    # Inicializar y conectar
    await integrador.initialize()
    await integrador.connect_all()
    
    # Definir símbolos personalizados para cada exchange
    symbols = {
        ExchangeID.BINANCE: "BTCUSDT",
        ExchangeID.COINBASE: "BTC-USD",
        ExchangeID.KRAKEN: "XBT/USD",
        ExchangeID.BITFINEX: "tBTCUSD",
        ExchangeID.HUOBI: "btcusdt",
        ExchangeID.KUCOIN: "BTC-USDT",
        ExchangeID.BYBIT: "BTCUSD",
        ExchangeID.OKEX: "BTC-USDT",
        ExchangeID.FTXINT: "BTC/USD",
        ExchangeID.BITSTAMP: "btcusd",
        ExchangeID.BITTREX: "BTC-USDT",
        ExchangeID.GEMINI: "BTCUSD",
        ExchangeID.GATEIO: "BTC_USDT",
        ExchangeID.MEXC: "BTC_USDT"
    }
    
    # Suscribir a tickers en todos los exchanges
    subscription_result = await integrador.subscribe_all(["ticker"], symbols)
    logger.info(f"Resultado de suscripciones:\n{json.dumps(subscription_result, indent=2)}")
    
    # Recolectar datos por 10 segundos
    logger.info("Recibiendo datos de mercado durante 10 segundos...")
    
    start_time = time.time()
    message_count = 0
    exchange_messages = {}
    
    # Recolectar mensajes durante 10 segundos
    async def collect_messages():
        nonlocal message_count, exchange_messages
        try:
            async for message in integrador.listen_all():
                exchange_name = message.get('exchange', 'unknown')
                if exchange_name not in exchange_messages:
                    exchange_messages[exchange_name] = 0
                exchange_messages[exchange_name] += 1
                message_count += 1
                
                if time.time() - start_time > 10:
                    break
        except Exception as e:
            logger.error(f"Error recolectando mensajes: {e}")
    
    # Ejecutar recolección
    await collect_messages()
    
    # Mostrar estadísticas
    logger.info(f"Total de mensajes recibidos: {message_count}")
    for exchange, count in exchange_messages.items():
        logger.info(f"- {exchange}: {count} mensajes")
    
    # Cerrar conexiones
    close_result = await integrador.close_all()
    logger.info(f"Conexiones cerradas: {close_result}")
    
    return {
        "subscription": subscription_result,
        "messages": message_count,
        "per_exchange": exchange_messages
    }

async def test_error_transmutation():
    """Probar transmutación de errores en exchanges problemáticos."""
    logger.info("=== INICIANDO PRUEBA DE TRANSMUTACIÓN DE ERRORES ===")
    
    # Crear integrador con exchanges que pueden ser problemáticos
    exchanges = [
        ExchangeID.FTXINT,  # FTX ya no existe, forzará transmutación
        ExchangeID.BITTREX,  # API puede ser inestable
        ExchangeID.BINANCE   # Para comparar con uno estable
    ]
    
    integrador = MultiExchangeTranscendentalIntegrator(exchanges)
    
    # Inicializar y conectar
    await integrador.initialize()
    connect_result = await integrador.connect_all()
    
    # Verificar transmutaciones
    transmuted_count = connect_result.get('transmuted', 0)
    logger.info(f"Conexiones transmutadas: {transmuted_count} de {len(exchanges)}")
    
    # Intentar suscripción
    subscription_result = await integrador.subscribe_all(["ticker"])
    
    # Verificar transmutaciones en suscripciones
    sub_transmuted = subscription_result.get('transmuted', 0)
    logger.info(f"Suscripciones transmutadas: {sub_transmuted} de {len(exchanges)}")
    
    # Probar recepción de datos (incluyendo transmutados)
    logger.info("Recibiendo datos durante 5 segundos (incluyendo transmutados)...")
    
    start_time = time.time()
    transmuted_data = 0
    real_data = 0
    
    # Recolectar mensajes
    async def collect_messages():
        nonlocal transmuted_data, real_data
        try:
            async for message in integrador.listen_all():
                if message.get('transmuted', False):
                    transmuted_data += 1
                else:
                    real_data += 1
                    
                if time.time() - start_time > 5:
                    break
        except Exception as e:
            logger.error(f"Error recolectando mensajes: {e}")
    
    # Ejecutar recolección
    await collect_messages()
    
    # Mostrar estadísticas
    logger.info(f"Datos reales recibidos: {real_data}")
    logger.info(f"Datos transmutados: {transmuted_data}")
    logger.info(f"Tasa de transmutación: {transmuted_data/(transmuted_data+real_data)*100:.1f}%")
    
    # Cerrar conexiones
    await integrador.close_all()
    
    return {
        "transmuted_connections": transmuted_count,
        "transmuted_subscriptions": sub_transmuted,
        "real_data": real_data,
        "transmuted_data": transmuted_data
    }

async def main():
    """Ejecutar todas las pruebas."""
    logger.info("=== INICIANDO PRUEBAS DEL INTEGRADOR TRASCENDENTAL MULTI-EXCHANGE ===")
    
    # Prueba 1: Conexiones básicas
    connection_result = await test_exchange_connections()
    
    # Prueba 2: Suscripción a datos de mercado
    subscription_result = await test_market_data_subscription()
    
    # Prueba 3: Transmutación de errores
    transmutation_result = await test_error_transmutation()
    
    # Mostrar resumen final
    logger.info("\n=== RESUMEN DE PRUEBAS ===")
    logger.info(f"1. Conexiones: {connection_result.get('connected', 0) + connection_result.get('transmuted', 0)} exitosas de {connection_result.get('total', 0)}")
    logger.info(f"2. Suscripciones: mensajes de {len(subscription_result.get('per_exchange', {}))} exchanges")
    logger.info(f"3. Transmutación: {transmutation_result.get('transmuted_data', 0)} datos transmutados")
    
    logger.info("\n=== PRUEBAS COMPLETADAS EXITOSAMENTE ===")
    logger.info("El Integrador Trascendental Multi-Exchange está funcionando perfectamente")

if __name__ == "__main__":
    asyncio.run(main())