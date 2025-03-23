#!/usr/bin/env python
"""
Script para probar la conexión directa con Binance Testnet 
usando ccxt con una configuración básica.
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
        # Crear objeto de exchange con modo de testnet
        binance = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        
        # Configurar para usar testnet
        binance.set_sandbox_mode(True)
        
        print("✓ Exchange configurado en modo testnet")
        
        # Cargar mercados
        print("Cargando mercados...")
        await binance.load_markets()
        print(f"✓ {len(binance.markets)} mercados disponibles")
        
        # Obtener ticker
        print("Obteniendo ticker de BTC/USDT...")
        ticker = await binance.fetch_ticker('BTC/USDT')
        print(f"✓ Precio de BTC/USDT: {ticker['last']} USDT")
        
        # Obtener balance
        print("Obteniendo balance de la cuenta...")
        balance = await binance.fetch_balance()
        print("✓ Balance obtenido")
        
        # Mostrar activos con balance
        assets_with_balance = {asset: amount for asset, amount in balance['total'].items() if amount > 0}
        
        if assets_with_balance:
            print("Activos con balance:")
            for asset, amount in assets_with_balance.items():
                print(f"  - {asset}: {amount}")
        else:
            print("No se encontraron activos con balance")
            
        # Cerrar el exchange
        await binance.close()
        print("✓ Conexión cerrada correctamente")
        
        print("\n✅ Prueba completada con éxito!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if 'binance' in locals():
            await binance.close()

if __name__ == "__main__":
    asyncio.run(test_binance_testnet())