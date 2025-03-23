#!/usr/bin/env python
"""
Script para probar la conexión directa con Binance Testnet usando ccxt.
"""

import asyncio
import ccxt.async_support as ccxt
import os
import logging
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_binance_testnet():
    """Probar la conexión directa a Binance Testnet."""
    print("Iniciando prueba directa de conexión a Binance Testnet...")
    
    # Obtener credenciales
    api_key = os.environ.get('BINANCE_TESTNET_API_KEY')
    api_secret = os.environ.get('BINANCE_TESTNET_SECRET')
    
    if not api_key or not api_secret:
        print("❌ No se encontraron las claves API en las variables de entorno")
        return
    
    print(f"✓ Claves API encontradas (longitud: {len(api_key)}, {len(api_secret)})")
    
    try:
        # Crear exchange con configuración específica para testnet
        exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'testnet': True,  # Esto es importante para que ccxt use el modo testnet
                'recvWindow': 10000,
            },
            # Reemplazar todas las URLs con las del testnet
            'urls': {
                'api': 'https://testnet.binance.vision',
                'web': 'https://testnet.binance.vision',
                'www': 'https://testnet.binance.vision',
                'doc': 'https://binance-docs.github.io/apidocs/spot/en',
                'test': 'https://testnet.binance.vision'
            }
        }
        
        print("Creando instancia de exchange...")
        exchange = ccxt.binance(exchange_config)
        print("✓ Instancia de exchange creada")
        
        print("Cargando mercados...")
        await exchange.load_markets()
        print(f"✓ Mercados cargados: {len(exchange.markets)} mercados disponibles")
        
        # Probar a obtener ticker
        print("Obteniendo ticker para BTC/USDT...")
        ticker = await exchange.fetch_ticker('BTC/USDT')
        print(f"✓ Ticker de BTC/USDT: precio actual = {ticker['last']} USDT")
        
        # Probar a obtener balance
        print("Obteniendo balance...")
        balance = await exchange.fetch_balance()
        total_balance = balance['total']
        print("✓ Balance obtenido:")
        
        # Mostrar activos con balance
        for currency, amount in total_balance.items():
            if amount > 0:
                print(f"  - {currency}: {amount}")
        
        # Cerrar el exchange
        print("Cerrando conexión...")
        await exchange.close()
        
        print("\n✅ Prueba completada con éxito! La conexión a Binance Testnet funciona correctamente.")
        
    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        if hasattr(e, 'args') and len(e.args) > 0:
            print(f"  - Detalle: {e.args[0]}")
        
        if 'exchange' in locals():
            await exchange.close()

async def main():
    try:
        await test_binance_testnet()
    except Exception as e:
        print(f"Error no capturado: {e}")

if __name__ == "__main__":
    asyncio.run(main())