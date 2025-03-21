#!/usr/bin/env python
"""
Script para probar la implementación de la clase CCXTExchange con Binance Testnet.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Asegurarnos que podemos importar los módulos de Genesis
sys.path.insert(0, os.getcwd())

async def test_ccxt_wrapper_testnet():
    """Probar la implementación de CCXTExchange con Binance Testnet."""
    print("Iniciando prueba de CCXTExchange con Binance Testnet...")
    
    # Importamos después de configurar sys.path
    from genesis.exchanges.ccxt_wrapper import CCXTExchange
    from genesis.utils.logger import setup_logging
    
    # Verificar si existen las variables de entorno para las keys
    api_key = os.environ.get('BINANCE_TESTNET_API_KEY')
    api_secret = os.environ.get('BINANCE_TESTNET_SECRET')
    
    if not api_key or not api_secret:
        print("❌ No se encontraron API keys en variables de entorno")
        return
    
    print(f"✓ API keys encontradas (longitud: {len(api_key)}, {len(api_secret)})")
    
    # Crear una instancia de CCXTExchange con modo testnet
    exchange = CCXTExchange(
        exchange_id='binance',
        api_key=api_key,
        secret=api_secret,
        config={'testnet': True}  # Activar modo testnet
    )
    
    print("✓ Instancia de CCXTExchange creada con configuración de testnet")
    
    try:
        # Iniciar el componente (esto debería activar el modo testnet internamente)
        print("Iniciando exchange...")
        await exchange.start()
        print("✓ Exchange iniciado correctamente")
        
        # Comprobar que se cargaron los mercados
        print(f"✓ {len(exchange.markets)} mercados disponibles")
        
        # Probar a obtener ticker
        print("Obteniendo ticker de BTC/USDT...")
        ticker = await exchange.fetch_ticker('BTC/USDT')
        print(f"✓ Precio de BTC/USDT: {ticker['last']} USDT")
        
        # Probar a obtener balance
        print("Obteniendo balance...")
        balance = await exchange.fetch_balance()
        print("✓ Balance obtenido")
        
        # Mostrar activos con balance
        assets_with_balance = {}
        for asset, amount in balance.items():
            if amount and amount > 0:
                assets_with_balance[asset] = amount
        
        if assets_with_balance:
            print("Activos con balance:")
            for asset, amount in assets_with_balance.items():
                print(f"  - {asset}: {amount}")
        else:
            print("No se encontraron activos con balance")
        
        # Detener el exchange
        print("Deteniendo exchange...")
        await exchange.stop()
        print("✓ Exchange detenido correctamente")
        
        print("\n✅ Prueba completada con éxito!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Asegurar que se cierre el exchange si hubo un error
        if exchange:
            try:
                await exchange.stop()
                print("Exchange detenido después de error")
            except:
                pass

if __name__ == "__main__":
    asyncio.run(test_ccxt_wrapper_testnet())