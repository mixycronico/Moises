"""
Prueba del WebSocket Trascendental para Exchanges con el sistema Genesis.

Este script prueba la conectividad del Sistema Genesis con exchanges de criptomonedas
mediante el WebSocket Trascendental, demostrando su capacidad para:
- Recibir datos de mercado en tiempo real con resiliencia perfecta
- Manejar reconexiones automáticas y transmutación de errores
- Procesar diferentes tipos de datos (trades, orderbook, klines)
- Operar bajo condiciones de red variables
"""

import asyncio
import json
import logging
import time
import sys
import os
from typing import Dict, Any, List, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_exchange_ws_transcendental.log")
    ]
)

logger = logging.getLogger("Test.ExchangeWS")

# Asegurar que el directorio raíz esté en el path de Python
sys.path.insert(0, os.path.abspath('.'))

# Importar componentes necesarios
from genesis.core.exchange_websocket_connector import ExchangeWebSocketHandler

# Mantener estadísticas
stats = {
    "trades_received": 0,
    "klines_received": 0,
    "orderbook_updates": 0,
    "errors": 0,
    "start_time": time.time()
}

async def on_trade(data: Dict[str, Any]) -> None:
    """
    Callback para datos de trades.
    
    Args:
        data: Datos normalizados del trade
    """
    stats["trades_received"] += 1
    
    # Log detallado cada 10 trades
    if stats["trades_received"] % 10 == 0:
        symbol = data.get("symbol", "unknown")
        price = data.get("price", 0)
        quantity = data.get("quantity", 0)
        side = data.get("side", "unknown")
        
        logger.info(f"Trade #{stats['trades_received']}: {symbol} - {price} - {quantity} - {side}")

async def on_kline(data: Dict[str, Any]) -> None:
    """
    Callback para datos de velas (klines).
    
    Args:
        data: Datos normalizados de la vela
    """
    stats["klines_received"] += 1
    
    # Log detallado cada 5 velas
    if stats["klines_received"] % 5 == 0:
        symbol = data.get("symbol", "unknown")
        interval = data.get("interval", "unknown")
        closed = data.get("is_closed", False)
        close = data.get("close", 0)
        
        logger.info(f"Kline #{stats['klines_received']}: {symbol} {interval} - Close: {close} - Closed: {closed}")

async def on_orderbook(data: Dict[str, Any]) -> None:
    """
    Callback para datos de libro de órdenes.
    
    Args:
        data: Datos normalizados del libro de órdenes
    """
    stats["orderbook_updates"] += 1
    
    # Log detallado cada 20 actualizaciones
    if stats["orderbook_updates"] % 20 == 0:
        symbol = data.get("symbol", "unknown")
        bids = len(data.get("bids", []))
        asks = len(data.get("asks", []))
        
        logger.info(f"Orderbook #{stats['orderbook_updates']}: {symbol} - Bids: {bids} - Asks: {asks}")

async def test_single_stream():
    """Prueba de conexión a un único stream de datos."""
    logger.info("=== PRUEBA DE CONEXIÓN A STREAM DE TRADES ===")
    
    # Crear manejador para Binance (testnet)
    handler = ExchangeWebSocketHandler("binance")
    
    # Conectar a stream de trades de BTC/USDT
    success = await handler.connect_to_stream("btcusdt@trade", on_trade)
    assert success, "La conexión al stream de trades debería ser exitosa"
    
    logger.info("Conectado a stream de trades, esperando datos por 30 segundos...")
    
    # Esperar datos durante 30 segundos
    await asyncio.sleep(30)
    
    # Obtener estadísticas
    ws_stats = handler.get_stats()
    logger.info(f"Estadísticas del WebSocket: {json.dumps(ws_stats, indent=2)}")
    logger.info(f"Trades recibidos: {stats['trades_received']}")
    
    # Desconectar
    await handler.disconnect_all()
    
    assert stats["trades_received"] > 0, "Deberían recibirse trades durante la prueba"
    logger.info("Prueba de stream único completada con éxito")

async def test_multiple_streams():
    """Prueba de conexión a múltiples streams simultáneos."""
    logger.info("=== PRUEBA DE CONEXIÓN A MÚLTIPLES STREAMS ===")
    
    # Crear manejador para Binance (testnet)
    handler = ExchangeWebSocketHandler("binance")
    
    # Conectar a varios streams
    await handler.connect_to_stream("btcusdt@trade", on_trade)
    await handler.connect_to_stream("ethusdt@trade", on_trade)
    await handler.connect_to_stream("btcusdt@kline_1m", on_kline)
    await handler.connect_to_stream("btcusdt@depth20", on_orderbook)
    
    logger.info("Conectado a múltiples streams, esperando datos por 60 segundos...")
    
    # Esperar datos durante 60 segundos
    start_time = time.time()
    while time.time() - start_time < 60:
        await asyncio.sleep(10)
        elapsed = time.time() - start_time
        trades_rate = stats["trades_received"] / elapsed
        klines_rate = stats["klines_received"] / elapsed
        orderbook_rate = stats["orderbook_updates"] / elapsed
        
        logger.info(f"Estado tras {elapsed:.1f}s:")
        logger.info(f"  Trades: {stats['trades_received']} ({trades_rate:.2f}/s)")
        logger.info(f"  Klines: {stats['klines_received']} ({klines_rate:.2f}/s)")
        logger.info(f"  Orderbook: {stats['orderbook_updates']} ({orderbook_rate:.2f}/s)")
    
    # Obtener estadísticas finales
    ws_stats = handler.get_stats()
    logger.info(f"Estadísticas del WebSocket: {json.dumps(ws_stats, indent=2)}")
    
    # Métricas
    total_messages = (stats["trades_received"] + stats["klines_received"] + 
                     stats["orderbook_updates"])
    elapsed = time.time() - stats["start_time"]
    rate = total_messages / elapsed
    
    logger.info(f"Rendimiento: {total_messages} mensajes en {elapsed:.2f}s ({rate:.2f} msgs/s)")
    
    # Desconectar
    await handler.disconnect_all()
    
    assert total_messages > 0, "Deberían recibirse mensajes durante la prueba"
    logger.info("Prueba de múltiples streams completada con éxito")

async def test_reconnection():
    """Prueba de reconexión automática."""
    logger.info("=== PRUEBA DE RECONEXIÓN AUTOMÁTICA ===")
    
    # Crear manejador para Binance (testnet)
    handler = ExchangeWebSocketHandler("binance")
    
    # Conectar a stream
    await handler.connect_to_stream("btcusdt@trade", on_trade)
    
    logger.info("Conectado a stream de trades, esperando datos iniciales...")
    await asyncio.sleep(10)
    
    # Número de trades antes de la desconexión
    trades_before = stats["trades_received"]
    logger.info(f"Trades recibidos antes de la desconexión: {trades_before}")
    
    # Simular desconexión
    logger.info("Simulando desconexión forzada...")
    stream_name = "btcusdt@trade"
    if stream_name in handler.connections and handler.connections[stream_name]['ws']:
        # Forzar cierre del WebSocket
        try:
            await handler.connections[stream_name]['ws'].close()
            handler.connections[stream_name]['active'] = False
            logger.info("Conexión cerrada forzadamente")
        except Exception as e:
            logger.error(f"Error al cerrar forzadamente: {e}")
    
    # Esperar reconexión automática y nuevos datos
    logger.info("Esperando reconexión automática y nuevos datos (30s)...")
    await asyncio.sleep(30)
    
    # Verificar estado después de la reconexión
    trades_after = stats["trades_received"]
    new_trades = trades_after - trades_before
    
    logger.info(f"Trades recibidos después de la reconexión: {trades_after}")
    logger.info(f"Nuevos trades tras reconexión: {new_trades}")
    
    # Obtener estadísticas
    ws_stats = handler.get_stats()
    logger.info(f"Estadísticas del WebSocket: {json.dumps(ws_stats, indent=2)}")
    
    # Desconectar
    await handler.disconnect_all()
    
    assert new_trades > 0, "Deberían recibirse nuevos trades tras la reconexión"
    assert ws_stats["reconnections"] > 0, "Debería registrarse al menos una reconexión"
    logger.info("Prueba de reconexión completada con éxito")

async def test_error_transmutation():
    """Prueba de transmutación de errores."""
    logger.info("=== PRUEBA DE TRANSMUTACIÓN DE ERRORES ===")
    
    # Crear manejador para Binance (testnet)
    handler = ExchangeWebSocketHandler("binance")
    
    # Intentar conectar a un stream inválido (debería fallar pero ser transmutado)
    # En lugar de fallar, el mecanismo trascendental debería transmutarlo en energía útil
    success = await handler.connect_to_stream("INVALIDSTREAM@nonexistent", on_trade)
    
    # Verificar que no falla catastróficamente gracias a la transmutación
    logger.info(f"Resultado de conexión a stream inválido: {success}")
    
    # Esperar procesamiento
    await asyncio.sleep(5)
    
    # Obtener estadísticas
    ws_stats = handler.get_stats()
    logger.info(f"Estadísticas tras error: {json.dumps(ws_stats, indent=2)}")
    
    # Verificar transmutación exitosa
    assert ws_stats["errors_transmuted"] > 0, "Debería haber errores transmutados"
    
    # Ahora probar con un stream válido para confirmar funcionamiento normal
    success = await handler.connect_to_stream("btcusdt@trade", on_trade)
    assert success, "La conexión a un stream válido debería funcionar"
    
    # Esperar datos
    await asyncio.sleep(10)
    
    # Desconectar
    await handler.disconnect_all()
    
    logger.info("Prueba de transmutación de errores completada")

async def main():
    """Función principal."""
    logger.info("INICIANDO PRUEBAS DEL WEBSOCKET TRASCENDENTAL PARA EXCHANGES")
    logger.info(f"Hora de inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Ejecutar pruebas una por una
        await test_single_stream()
        await test_multiple_streams()
        await test_reconnection()
        await test_error_transmutation()
        
        logger.info("TODAS LAS PRUEBAS COMPLETADAS CON ÉXITO")
        
    except Exception as e:
        logger.error(f"ERROR EN PRUEBAS: {str(e)}")
        stats["errors"] += 1
        raise
    finally:
        # Resumen final
        elapsed = time.time() - stats["start_time"]
        logger.info("\n=== RESUMEN DE PRUEBAS ===")
        logger.info(f"Duración total: {elapsed:.2f} segundos")
        logger.info(f"Trades recibidos: {stats['trades_received']}")
        logger.info(f"Klines recibidas: {stats['klines_received']}")
        logger.info(f"Actualizaciones de orderbook: {stats['orderbook_updates']}")
        logger.info(f"Errores encontrados: {stats['errors']}")
        
        total_messages = (stats["trades_received"] + stats["klines_received"] + 
                         stats["orderbook_updates"])
        logger.info(f"Total mensajes procesados: {total_messages}")
        logger.info(f"Rendimiento promedio: {total_messages/elapsed:.2f} mensajes/segundo")

if __name__ == "__main__":
    asyncio.run(main())
"""