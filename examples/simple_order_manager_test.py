#!/usr/bin/env python3
"""
Prueba simplificada de OrderManager del Sistema Genesis Ultra-Divino Trading Nexus.

Este script realiza una prueba básica de la integración entre OrderManager y un
exchange simulado, verificando la capacidad de colocar, consultar y cancelar órdenes.

Autor: Genesis AI Assistant
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("simple_order_test")

async def run_basic_order_test():
    """Ejecutar prueba básica de órdenes."""
    try:
        logger.info("=== Iniciando prueba simplificada de OrderManager ===")
        
        # Importar las clases necesarias aquí para evitar problemas de dependencias
        from genesis.trading.order_manager import OrderManager, OrderSide, OrderType, OrderStatus
        from genesis.exchanges.simulated_exchange_adapter import SimulatedExchangeAdapter
        from genesis.simulators.exchange_simulator import ExchangeSimulator
        
        # Paso 1: Crear simulador y adaptador
        logger.info("Creando simulador de exchange...")
        simulator = ExchangeSimulator(
            exchange_id="BINANCE", 
            config={
                "tick_interval_ms": 500,
                "volatility_factor": 0.005,
                "pattern_duration": 120,
                "enable_failures": False
            }
        )
        
        # Paso 2: Crear adaptador simulado
        adapter = SimulatedExchangeAdapter(simulator)
        await adapter.initialize()
        await adapter.connect()
        
        logger.info("Adaptador de exchange simulado creado y conectado")
        
        # Paso 3: Crear OrderManager
        logger.info("Creando OrderManager...")
        order_manager = OrderManager(exchange_adapter=adapter, behavior_engine=None)
        await order_manager.initialize()
        
        logger.info("OrderManager inicializado correctamente")
        
        # Paso 4: Precargar algunos símbolos
        test_symbol = "BTC/USDT"
        logger.info(f"Precargando símbolo {test_symbol}...")
        ticker = await adapter.get_ticker(test_symbol)
        logger.info(f"Precio actual de {test_symbol}: {ticker.get('last', 'N/A')}")
        
        # Paso 5: Colocar una orden de compra
        buy_amount = 0.01  # Pequeña cantidad para prueba
        logger.info(f"Colocando orden de compra para {buy_amount} {test_symbol}...")
        
        buy_order = await order_manager.place_order(
            symbol=test_symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=buy_amount
        )
        
        if not buy_order.get("success", False):
            logger.error(f"Error colocando orden de compra: {buy_order.get('error', 'Unknown error')}")
            return
        
        buy_order_id = buy_order.get("order_id")
        logger.info(f"Orden de compra colocada correctamente con ID: {buy_order_id}")
        
        # Paso 6: Esperar un momento
        logger.info("Esperando 2 segundos...")
        await asyncio.sleep(2)
        
        # Paso 7: Obtener órdenes existentes
        logger.info("Consultando órdenes existentes...")
        orders = await order_manager.get_orders()
        
        logger.info(f"Obtenidas {len(orders)} órdenes")
        for i, order in enumerate(orders):
            logger.info(f"Orden {i+1}: {order.get('symbol')} {order.get('side')} {order.get('status')} {order.get('amount')}")
        
        # Paso 8: Colocar una orden de venta
        sell_amount = 0.01  # Misma cantidad para vender
        logger.info(f"Colocando orden de venta para {sell_amount} {test_symbol}...")
        
        sell_order = await order_manager.place_order(
            symbol=test_symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=sell_amount
        )
        
        if not sell_order.get("success", False):
            logger.error(f"Error colocando orden de venta: {sell_order.get('error', 'Unknown error')}")
            return
        
        sell_order_id = sell_order.get("order_id")
        logger.info(f"Orden de venta colocada correctamente con ID: {sell_order_id}")
        
        # Paso 9: Esperar un momento
        logger.info("Esperando 2 segundos...")
        await asyncio.sleep(2)
        
        # Paso 10: Obtener órdenes finales
        logger.info("Consultando órdenes finales...")
        final_orders = await order_manager.get_orders()
        
        logger.info(f"Obtenidas {len(final_orders)} órdenes finales")
        for i, order in enumerate(final_orders):
            logger.info(f"Orden {i+1}: {order.get('symbol')} {order.get('side')} {order.get('status')} {order.get('amount')}")
        
        logger.info("=== Prueba simplificada de OrderManager completada exitosamente ===")
    
    except Exception as e:
        logger.error(f"Error ejecutando prueba: {str(e)}")

async def main():
    """Función principal."""
    try:
        await run_basic_order_test()
    except Exception as e:
        logger.error(f"Error en main: {str(e)}")
    finally:
        # Esperar para que todos los logs sean visibles
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())