"""
Demostración de uso del simulador de exchange para el Sistema Genesis Ultra-Divino.

Este script demuestra el uso del simulador de exchange y su integración
con el adaptador, permitiendo probar funcionalidades del sistema sin
depender de conexiones reales a exchanges.

Escenarios demostrados:
1. Inicialización del simulador
2. Obtención de datos de mercado
3. Suscripción a actualizaciones en tiempo real
4. Colocación y gestión de órdenes
5. Simulación de patrones de mercado y eventos
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List

from genesis.simulators import (
    ExchangeSimulator,
    ExchangeSimulatorFactory,
    MarketPattern,
    MarketEventType
)
from genesis.exchanges.simulated_exchange_adapter import SimulatedExchangeAdapter

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Genesis.ExchangeDemo")

# Lista de símbolos para la demostración
DEMO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]

async def demonstrate_market_data(adapter: SimulatedExchangeAdapter):
    """
    Demostrar obtención de datos de mercado.
    
    Args:
        adapter: Adaptador del simulador
    """
    logger.info("=== DEMOSTRACIÓN DE DATOS DE MERCADO ===")
    
    # Obtener tickers para todos los símbolos
    for symbol in DEMO_SYMBOLS:
        ticker = await adapter.get_ticker(symbol)
        logger.info(f"Ticker para {symbol}: precio = {ticker['price']:.2f}, "
                   f"cambio 24h = {ticker['percentage']:.2f}%")
        
    # Obtener libro de órdenes para BTC/USDT
    orderbook = await adapter.get_orderbook("BTC/USDT", limit=5)
    logger.info(f"Libro de órdenes BTC/USDT:")
    logger.info(f"  Mejor compra: {orderbook['bids'][0][0]:.2f} ({orderbook['bids'][0][1]:.4f})")
    logger.info(f"  Mejor venta: {orderbook['asks'][0][0]:.2f} ({orderbook['asks'][0][1]:.4f})")
    
    # Obtener velas para ETH/USDT
    candles = await adapter.get_candles("ETH/USDT", timeframe="1h", limit=5)
    logger.info(f"Últimas 5 velas horarias para ETH/USDT:")
    for candle in candles:
        dt = datetime.fromtimestamp(candle['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
        logger.info(f"  {dt}: Apertura={candle['open']:.2f}, "
                   f"Cierre={candle['close']:.2f}, "
                   f"Alto={candle['high']:.2f}, "
                   f"Bajo={candle['low']:.2f}, "
                   f"Volumen={candle['volume']:.2f}")

async def demonstrate_orders(adapter: SimulatedExchangeAdapter):
    """
    Demostrar colocación y gestión de órdenes.
    
    Args:
        adapter: Adaptador del simulador
    """
    logger.info("=== DEMOSTRACIÓN DE ÓRDENES ===")
    
    # Obtener precio actual de BTC
    ticker = await adapter.get_ticker("BTC/USDT")
    current_price = ticker["price"]
    
    # Colocar una orden de mercado
    market_order = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "market",
        "quantity": 0.1
    }
    
    result = await adapter.place_order(market_order)
    order = result["order"]
    
    logger.info(f"Orden de mercado colocada: ID={order['id']}, "
               f"Estado={order['status']}, "
               f"Precio={order['price']:.2f}, "
               f"Cantidad={order['quantity']:.8f}")
               
    # Colocar una orden límite
    limit_price = current_price * 0.95  # 5% por debajo del precio actual
    limit_order = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "limit",
        "quantity": 0.2,
        "price": limit_price
    }
    
    result = await adapter.place_order(limit_order)
    order = result["order"]
    
    logger.info(f"Orden límite colocada: ID={order['id']}, "
               f"Estado={order['status']}, "
               f"Precio={order['price']:.2f}, "
               f"Cantidad={order['quantity']:.8f}")
               
    # Obtener órdenes activas
    orders = await adapter.get_orders(symbol="BTC/USDT")
    logger.info(f"Total órdenes para BTC/USDT: {len(orders['orders'])}")
    
    # Cancelar la orden límite si no está completada
    if order["status"] != "FILLED":
        cancel_result = await adapter.cancel_order(order["id"])
        logger.info(f"Orden límite cancelada: {cancel_result['order']['status']}")
        
        # Verificar órdenes activas nuevamente
        orders = await adapter.get_orders(symbol="BTC/USDT")
        logger.info(f"Total órdenes activas después de cancelación: {len([o for o in orders['orders'] if o['status'] != 'CANCELED' and o['status'] != 'FILLED'])}")

async def demonstrate_market_patterns(adapter: SimulatedExchangeAdapter):
    """
    Demostrar patrones de mercado y eventos.
    
    Args:
        adapter: Adaptador del simulador
    """
    logger.info("=== DEMOSTRACIÓN DE PATRONES DE MERCADO ===")
    
    # Obtener precio inicial de ETH
    ticker = await adapter.get_ticker("ETH/USDT")
    initial_price = ticker["price"]
    logger.info(f"Precio inicial de ETH/USDT: {initial_price:.2f}")
    
    # Establecer patrón de tendencia alcista
    await adapter.set_market_pattern("ETH/USDT", MarketPattern.TRENDING_UP)
    logger.info("Patrón de tendencia alcista establecido para ETH/USDT")
    
    # Esperar a que el precio cambie
    await asyncio.sleep(3)
    
    # Obtener nuevo precio
    ticker = await adapter.get_ticker("ETH/USDT")
    new_price = ticker["price"]
    change_pct = (new_price - initial_price) / initial_price * 100
    logger.info(f"Nuevo precio de ETH/USDT: {new_price:.2f} (cambio: {change_pct:.2f}%)")
    
    # Añadir evento de mercado (compra de ballena)
    await adapter.add_market_event(MarketEventType.WHALE_BUY, "ETH/USDT", impact=0.03)
    logger.info("Evento de compra de ballena añadido para ETH/USDT (impacto 3%)")
    
    # Esperar a que el evento ocurra
    await asyncio.sleep(2)
    
    # Obtener precio después del evento
    ticker = await adapter.get_ticker("ETH/USDT")
    event_price = ticker["price"]
    event_change_pct = (event_price - new_price) / new_price * 100
    logger.info(f"Precio después del evento: {event_price:.2f} (cambio: {event_change_pct:.2f}%)")
    
    # Establecer patrón de alta volatilidad para SOL
    await adapter.set_market_pattern("SOL/USDT", MarketPattern.VOLATILE)
    logger.info("Patrón de alta volatilidad establecido para SOL/USDT")

async def demonstrate_websocket(adapter: SimulatedExchangeAdapter):
    """
    Demostrar suscripción a datos en tiempo real.
    
    Args:
        adapter: Adaptador del simulador
    """
    logger.info("=== DEMOSTRACIÓN DE WEBSOCKET ===")
    
    # Variable para contar actualizaciones
    update_count = 0
    
    # Definir callback para recibir actualizaciones
    async def ticker_callback(data: Dict[str, Any]):
        nonlocal update_count
        update_count += 1
        logger.info(f"Actualización #{update_count} de ticker BTC/USDT: "
                   f"precio = {data['price']:.2f}, "
                   f"cambio = {data['percentage']:.2f}%")
    
    # Registrar callback para ticker de BTC/USDT
    await adapter.register_callback("ticker:BTC/USDT", ticker_callback)
    logger.info("Callback registrado para ticker:BTC/USDT")
    
    # Esperar algunas actualizaciones
    logger.info("Esperando actualizaciones durante 10 segundos...")
    await asyncio.sleep(10)
    
    # Cancelar suscripción
    await adapter.unregister_callback("ticker:BTC/USDT")
    logger.info("Suscripción cancelada")
    logger.info(f"Total actualizaciones recibidas: {update_count}")

async def run_demo():
    """Función principal para ejecutar la demostración."""
    logger.info("Iniciando demostración de simulador de exchange para Genesis Ultra-Divino")
    
    try:
        # Configuración del simulador
        config = {
            "tick_interval_ms": 500,       # Actualizaciones cada 500ms
            "volatility_factor": 0.005,    # 0.5% de volatilidad
            "error_rate": 0.05,            # 5% de probabilidad de error
            "pattern_duration": 30,        # 30 segundos por patrón
            "enable_failures": True        # Habilitar fallos simulados
        }
        
        # Crear adaptador
        adapter = SimulatedExchangeAdapter("BINANCE", config)
        await adapter.initialize()
        logger.info("Adaptador inicializado correctamente")
        
        # Conectar al simulador
        await adapter.connect()
        logger.info("Conectado al simulador")
        
        # Ejecutar demostraciones
        await demonstrate_market_data(adapter)
        await demonstrate_orders(adapter)
        await demonstrate_market_patterns(adapter)
        await demonstrate_websocket(adapter)
        
        # Cerrar conexión
        await adapter.close()
        logger.info("Demostración completada con éxito")
        
    except Exception as e:
        logger.error(f"Error en la demostración: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_demo())
"""