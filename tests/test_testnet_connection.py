#!/usr/bin/env python
"""
Script para probar la conexión con Binance Testnet.
"""

import asyncio
import sys
import os
import logging
import time
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir un timeout para evitar que el script se quede colgado
TIMEOUT = 30  # segundos

# Asegurarnos que podemos importar los módulos de Genesis
sys.path.insert(0, os.getcwd())

async def test_connection():
    """Probar la conexión con Binance Testnet."""
    try:
        print("Iniciando prueba de conexión a Binance Testnet...")
        
        # Importar después de configurar sys.path
        try:
            from genesis.exchanges.ccxt_wrapper import CCXTExchange
            from genesis.utils.logger import setup_logging
            print("✓ Módulos importados correctamente")
        except ImportError as e:
            print(f"❌ Error importando módulos: {e}")
            raise
        
        logger = setup_logging("testnet_test")
        logger.info("Iniciando prueba de conexión a Binance Testnet...")
        
        # Verificar si existen las variables de entorno para las keys
        api_key = os.environ.get('BINANCE_TESTNET_API_KEY')
        api_secret = os.environ.get('BINANCE_TESTNET_SECRET')
        
        if api_key and api_secret:
            logger.info("Usando API keys de variables de entorno")
        else:
            logger.warning("No se encontraron API keys en variables de entorno")
            logger.warning("Se intentará conectar sin autenticación (funcionalidad limitada)")
        
        # Crear exchange con configuración de testnet
        exchange = CCXTExchange(
            exchange_id='binance',
            api_key=api_key,
            secret=api_secret,
            config={'testnet': True}
        )
        
        # Iniciar el exchange
        await exchange.start()
        
        # Probar algunas operaciones básicas
        logger.info("Obteniendo mercados disponibles...")
        markets = exchange.markets
        logger.info(f"Se encontraron {len(markets)} mercados")
        
        # Obtener ticker para BTC/USDT
        logger.info("Obteniendo ticker para BTC/USDT...")
        ticker = await exchange.fetch_ticker("BTC/USDT")
        logger.info(f"Precio actual de BTC/USDT: {ticker['last']}")
        
        # Obtener balance (esto requerirá API keys)
        try:
            logger.info("Intentando obtener balance...")
            balance = await exchange.fetch_balance()
            if balance:
                for asset, amount in balance.items():
                    if amount and amount > 0:
                        logger.info(f"Balance de {asset}: {amount}")
            logger.info("Balance obtenido correctamente")
        except Exception as e:
            logger.warning(f"No se pudo obtener el balance: {e}")
            logger.warning("Esto es normal si no se proporcionaron API keys")
        
        logger.info("Conexión exitosa a Binance Testnet")
        
        # Detener el exchange
        await exchange.stop()
        
    except Exception as e:
        print(f"❌ Error al conectar con Binance Testnet: {e}")
        if 'logger' in locals():
            logger.error(f"Error al conectar con Binance Testnet: {e}")
        raise

if __name__ == "__main__":
    try:
        # Usar un timeout para evitar que el script se quede colgado indefinidamente
        print(f"Ejecutando con timeout de {TIMEOUT} segundos...")
        
        async def run_with_timeout():
            try:
                # Crear una tarea con timeout
                await asyncio.wait_for(test_connection(), timeout=TIMEOUT)
                print("✓ Conexión finalizada correctamente")
            except asyncio.TimeoutError:
                print(f"❌ Timeout después de {TIMEOUT} segundos. La operación tomó demasiado tiempo.")
                print("Es posible que haya problemas con la red o con la conexión a Binance Testnet.")
            except Exception as e:
                print(f"❌ Error durante la conexión: {e}")
                
        asyncio.run(run_with_timeout())
        
    except KeyboardInterrupt:
        print("\n⚠️ Operación cancelada por el usuario")
    except Exception as e:
        print(f"❌ Error grave: {e}")