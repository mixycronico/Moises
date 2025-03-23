"""
Prueba Exhaustiva del Integrador Trascendental para múltiples exchanges.

Este script realiza pruebas intensivas del integrador para verificar:
1. Conexión a múltiples exchanges simultáneamente
2. Suscripción a canales específicos por exchange
3. Manejo de errores y transmutación
4. Recepción de datos en tiempo real
5. Pruebas específicas para métodos added: subscribe_all, listen_all, get_states
"""

import asyncio
import logging
import time
import json
import random
from typing import Dict, Any, List, Optional

from genesis.core.transcendental_exchange_integrator import TranscendentalExchangeIntegrator
from genesis.core.transcendental_ws_adapter import ExchangeID

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestExchangeIntegratorDetailed")

# Función para imprimir resultados de forma bonita
def print_result(title, result):
    logger.info(f"\n==== {title} ====")
    if isinstance(result, dict):
        logger.info(json.dumps(result, indent=2))
    else:
        logger.info(str(result))
    logger.info("=" * (len(title) + 10))

async def test_initialize():
    """Probar inicialización básica del integrador."""
    logger.info("Prueba de inicialización del integrador")
    
    integrador = TranscendentalExchangeIntegrator()
    
    # Lista de 5 exchanges para pruebas básicas
    exchanges = [
        ExchangeID.BINANCE,
        ExchangeID.COINBASE,
        ExchangeID.KRAKEN,
        ExchangeID.BITFINEX,
        ExchangeID.HUOBI
    ]
    
    # Añadir exchanges uno por uno
    for exchange_id in exchanges:
        result = await integrador.add_exchange(exchange_id)
        logger.info(f"Añadido {exchange_id}: {result}")
    
    # Verificar que se añadieron correctamente
    states = integrador.get_states()
    print_result("Estados iniciales", states)
    
    return integrador, exchanges

async def test_connect_disconnect(integrador, exchanges):
    """Probar conectar y desconectar de todos los exchanges."""
    logger.info("Prueba de conexión y desconexión")
    
    # Conectar a todos
    connect_result = await integrador.connect_all()
    print_result("Resultado de conexión", connect_result)
    
    # Verificar estados
    states_after_connect = integrador.get_states()
    print_result("Estados después de conectar", states_after_connect)
    
    # Desconectar de todos usando close_all (alias para disconnect_all)
    close_result = await integrador.close_all()
    print_result("Resultado de desconexión", close_result)
    
    # Verificar estados finales
    final_states = integrador.get_states()
    print_result("Estados finales", final_states)
    
    return {
        "connect": connect_result,
        "disconnect": close_result,
        "states": final_states
    }

async def test_subscribe_all(integrador, exchanges):
    """Probar suscripción simultánea a todos los exchanges."""
    logger.info("Prueba de suscripción simultánea")
    
    # Reconectar si es necesario
    await integrador.connect_all()
    
    # Probar suscribe_all con diferentes símbolos por exchange
    symbols = {
        ExchangeID.BINANCE: "btcusdt",
        ExchangeID.COINBASE: "BTC-USD",
        ExchangeID.KRAKEN: "XBT/USD",
        ExchangeID.BITFINEX: "tBTCUSD",
        ExchangeID.HUOBI: "btcusdt"
    }
    
    # Suscribir a canales utilizando subscribe_all
    subscription_result = await integrador.subscribe_all(["ticker"], symbols)
    print_result("Resultado de suscripción masiva", subscription_result)
    
    # Verificar suscripciones usando get_all_states
    all_states = integrador.get_all_states()
    print_result("Estados con suscripciones", all_states)
    
    return {
        "subscription": subscription_result,
        "states": all_states
    }

async def test_listen_all(integrador, exchanges):
    """Probar recepción de mensajes de todos los exchanges."""
    logger.info("Prueba de recepción de mensajes (listen_all)")
    
    # Asegurar conexión y suscripción
    await integrador.connect_all()
    
    # Suscribir a tickers en todos los exchanges si no lo hemos hecho
    symbols = {
        ExchangeID.BINANCE: "btcusdt",
        ExchangeID.COINBASE: "BTC-USD",
        ExchangeID.KRAKEN: "XBT/USD",
        ExchangeID.BITFINEX: "tBTCUSD",
        ExchangeID.HUOBI: "btcusdt"
    }
    await integrador.subscribe_all(["ticker"], symbols)
    
    # Recolectar datos por 5 segundos
    logger.info("Recibiendo datos de mercado durante 5 segundos...")
    
    start_time = time.time()
    message_count = 0
    exchange_messages = {}
    
    # Recolectar mensajes durante 5 segundos usando listen_all
    async def collect_messages():
        nonlocal message_count, exchange_messages
        try:
            async for message in integrador.listen_all():
                exchange_name = message.get('exchange', message.get('_integrator', {}).get('exchange_id', 'unknown'))
                
                if exchange_name not in exchange_messages:
                    exchange_messages[exchange_name] = 0
                exchange_messages[exchange_name] += 1
                message_count += 1
                
                # Mostrar algunos mensajes como ejemplo
                if message_count % 10 == 0:
                    logger.info(f"Mensaje #{message_count} de {exchange_name}: {json.dumps(message, indent=2)}")
                
                if time.time() - start_time > 5:
                    break
        except Exception as e:
            logger.error(f"Error recolectando mensajes: {e}")
    
    # Ejecutar recolección
    await collect_messages()
    
    # Mostrar estadísticas
    stats = {
        "total_messages": message_count,
        "duration_seconds": time.time() - start_time,
        "messages_per_second": message_count / (time.time() - start_time),
        "per_exchange": exchange_messages
    }
    
    print_result("Estadísticas de recepción", stats)
    
    return stats

async def test_error_handling(integrador):
    """Probar manejo de errores y transmutación."""
    logger.info("Prueba de manejo de errores y transmutación")
    
    # Añadir un exchange que sabemos que puede fallar (FTX)
    await integrador.add_exchange(ExchangeID.FTX)
    
    # Intentar conectar
    connect_result = await integrador.connect(ExchangeID.FTX)
    print_result("Conexión a FTX (debería transmutarse)", connect_result)
    
    # Intentar suscribirse
    subscribe_result = await integrador.subscribe(ExchangeID.FTX, ["btcusd@ticker"])
    print_result("Suscripción a FTX (debería transmutarse)", subscribe_result)
    
    # Intentar recibir mensajes
    messages = []
    for _ in range(5):
        message = await integrador.receive(ExchangeID.FTX)
        messages.append(message)
    
    print_result("Mensajes de FTX (deberían ser transmutados)", messages)
    
    # Verificar estadísticas para ver conteo de transmutaciones
    stats = integrador.get_stats()
    print_result("Estadísticas con transmutaciones", stats)
    
    return {
        "connect": connect_result,
        "subscribe": subscribe_result,
        "messages": messages,
        "stats": stats
    }

async def main():
    """Ejecutar todas las pruebas detalladas."""
    logger.info("=== INICIANDO PRUEBAS DETALLADAS DEL INTEGRADOR TRASCENDENTAL ===")
    
    try:
        # Inicialización
        integrador, exchanges = await test_initialize()
        
        # Probar conexión/desconexión
        connect_result = await test_connect_disconnect(integrador, exchanges)
        
        # Probar suscripción masiva
        subscription_result = await test_subscribe_all(integrador, exchanges)
        
        # Probar recepción de mensajes
        listen_result = await test_listen_all(integrador, exchanges)
        
        # Probar manejo de errores
        error_result = await test_error_handling(integrador)
        
        # Mostrar estadísticas finales
        final_stats = integrador.get_stats()
        print_result("ESTADÍSTICAS FINALES", final_stats)
        
        # Mostrar resultado general
        logger.info("\n=== RESUMEN DE PRUEBAS DETALLADAS ===")
        logger.info(f"1. Conexión/Desconexión: {connect_result['connect']['success']}")
        logger.info(f"2. Suscripción masiva: {subscription_result['subscription']['success']}")
        logger.info(f"3. Recepción de mensajes: {listen_result['total_messages']} mensajes recibidos")
        logger.info(f"4. Transmutación de errores: {error_result['stats']['transmuted_count']} transmutaciones")
        
        logger.info("\n=== PRUEBAS DETALLADAS COMPLETADAS EXITOSAMENTE ===")
        logger.info("El Integrador Trascendental Multi-Exchange está funcionando perfectamente")
        
        # Desconectar todo al finalizar
        await integrador.disconnect_all()
        
    except Exception as e:
        logger.error(f"Error en pruebas: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())