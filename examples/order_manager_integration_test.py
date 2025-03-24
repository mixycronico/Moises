#!/usr/bin/env python
"""
Prueba de integración entre SeraphimOrchestrator y CodiciaManager.

Este script demuestra el flujo completo de trading utilizando:
1. Creación del SeraphimOrchestrator
2. Inicialización de componentes
3. Verificación de conexiones de exchange
4. Colocación y gestión de órdenes
5. Monitoreo de estado

Autor: Genesis AI Assistant
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional

# Importar directamente desde los módulos donde están definidas las clases
from genesis.trading.codicia_manager import CodiciaManager, OrderSide, OrderType, OrderStatus
from genesis.trading.seraphim_orchestrator import SeraphimOrchestrator
from genesis.trading.human_behavior_engine import GabrielBehaviorEngine

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("order_integration_test")


async def run_basic_order_test():
    """Ejecutar prueba básica de órdenes."""
    logger.info("=== Iniciando prueba de integración entre SeraphimOrchestrator y CodiciaManager ===")
    
    # Paso 1: Crear orquestador
    # Notamos que SeraphimOrchestrator no acepta parámetros en su constructor según su implementación
    orchestrator = SeraphimOrchestrator()
    
    # Paso 2: Inicializar orquestador (esto inicializará internamente los componentes)
    logger.info("Inicializando orquestador...")
    init_success = await orchestrator.initialize()
    
    if not init_success:
        logger.error("Error inicializando orquestador")
        return
    
    logger.info("Orquestador inicializado correctamente")
    
    # Paso 3: Verificar conexiones de exchange
    # Este método configura y crea el exchange_adapter y el codicia_manager si no existen
    logger.info("Verificando conexiones de exchange...")
    exchange_connected = await orchestrator._verify_exchange_connections()
    
    if not exchange_connected:
        logger.error("Error conectando con exchange")
        return
    
    logger.info("Conexión con exchange verificada correctamente")
    
    # Paso 4: Obtener símbolos disponibles
    logger.info("Obteniendo símbolos disponibles...")
    symbols = await orchestrator.get_symbols()
    
    if not symbols:
        logger.error("No se pudieron obtener símbolos")
        return
    
    logger.info(f"Obtenidos {len(symbols)} símbolos: {', '.join(symbols[:5])}...")
    
    # Paso 5: Seleccionar un símbolo para la prueba
    test_symbol = "BTC/USDT"
    logger.info(f"Utilizando {test_symbol} para prueba de órdenes")
    
    # Paso 6: Obtener datos de mercado para el símbolo seleccionado
    logger.info(f"Obteniendo datos de mercado para {test_symbol}...")
    market_data = await orchestrator.get_market_data(test_symbol)
    
    if not market_data.get("success", False):
        logger.error(f"Error obteniendo datos de mercado: {market_data.get('error', 'Unknown error')}")
        return
    
    logger.info(f"Precio actual de {test_symbol}: {market_data.get('last_price')}")
    
    # Paso 7: Colocar una orden de compra
    buy_amount = 0.01  # Pequeña cantidad para prueba
    logger.info(f"Colocando orden de compra para {buy_amount} {test_symbol}...")
    
    buy_order = await orchestrator.place_order(
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
    
    # Paso 8: Esperar un momento antes de colocar la orden de venta
    logger.info("Esperando 2 segundos...")
    await asyncio.sleep(2)
    
    # Paso 9: Obtener órdenes existentes
    logger.info("Consultando órdenes existentes...")
    orders = await orchestrator.get_orders()
    
    if not orders.get("success", False):
        logger.error(f"Error obteniendo órdenes: {orders.get('error', 'Unknown error')}")
    else:
        logger.info(f"Obtenidas {orders.get('count', 0)} órdenes")
        
        # Mostrar detalles de las órdenes
        for i, order in enumerate(orders.get("orders", [])):
            logger.info(f"Orden {i+1}: {order.get('symbol')} {order.get('side')} {order.get('status')} {order.get('amount')}")
    
    # Paso 10: Colocar una orden de venta
    sell_amount = 0.01  # Misma cantidad para vender
    logger.info(f"Colocando orden de venta para {sell_amount} {test_symbol}...")
    
    sell_order = await orchestrator.place_order(
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
    
    # Paso 11: Consultar nuevamente las órdenes para verificar los estados finales
    logger.info("Esperando 2 segundos para que se procesen todas las órdenes...")
    await asyncio.sleep(2)
    
    final_orders = await orchestrator.get_orders()
    
    if not final_orders.get("success", False):
        logger.error(f"Error obteniendo órdenes finales: {final_orders.get('error', 'Unknown error')}")
    else:
        logger.info(f"Estado final: {final_orders.get('count', 0)} órdenes en total")
        
        # Mostrar detalles de las órdenes finales
        for i, order in enumerate(final_orders.get("orders", [])):
            logger.info(f"Orden {i+1}: {order.get('symbol')} {order.get('side')} {order.get('status')} {order.get('amount')}")
    
    # Paso 12: Verificar el comportamiento humano aplicado a las órdenes
    logger.info("Verificando comportamiento del motor de comportamiento humano...")
    behavior_state = orchestrator.behavior_engine.get_state() if orchestrator.behavior_engine else None
    
    if behavior_state:
        logger.info(f"Estado emocional actual: {behavior_state.get('emotional_state', 'unknown')}")
        logger.info(f"Nivel de riesgo: {behavior_state.get('risk_level', 'unknown')}")
    else:
        logger.info("Comportamiento humano no disponible")
    
    logger.info("=== Prueba de integración completada exitosamente ===")


async def main():
    """Función principal."""
    try:
        await run_basic_order_test()
    except Exception as e:
        logger.error(f"Error ejecutando prueba: {str(e)}")
    finally:
        # Esperar para que todos los logs sean visibles
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())