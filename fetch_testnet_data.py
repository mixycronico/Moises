#!/usr/bin/env python
"""
Script para descargar datos históricos de Binance Testnet.

Este script se conecta a Binance Testnet, descarga datos OHLCV históricos
y los almacena en archivos CSV para usarlos en pruebas.
"""

import asyncio
import os
import sys
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fetch_testnet_data")

# Asegurarnos que podemos importar los módulos de Genesis
sys.path.insert(0, os.getcwd())

SYMBOLS = ['BTC/USDT']  # Reducido a solo BTC/USDT para pruebas rápidas
TIMEFRAMES = ['1h', '4h']  # Reducido a solo 1h y 4h para pruebas rápidas
DATA_DIR = './data/testnet'

async def save_to_csv(symbol: str, timeframe: str, data: List[List[float]]) -> None:
    """
    Guardar datos OHLCV en un archivo CSV.
    
    Args:
        symbol: Símbolo de trading
        timeframe: Marco temporal
        data: Datos OHLCV
    """
    # Crear directorio si no existe
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Limpiar símbolo para nombre de archivo
    clean_symbol = symbol.replace('/', '_')
    
    # Crear DataFrame
    df = pd.DataFrame(
        data,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # Convertir timestamp a datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Guardar a CSV
    filename = f"{DATA_DIR}/{clean_symbol}_{timeframe}.csv"
    df.to_csv(filename, index=False)
    logger.info(f"Guardados {len(data)} registros en {filename}")

async def fetch_testnet_data():
    """
    Descargar datos históricos de Binance Testnet.
    """
    logger.info("Iniciando descarga de datos históricos de Binance Testnet")
    
    try:
        # Importar CCXT
        import ccxt.async_support as ccxt
        logger.info("Biblioteca CCXT importada correctamente")
        
        # Verificar si existen las variables de entorno para las keys
        api_key = os.environ.get('BINANCE_TESTNET_API_KEY')
        api_secret = os.environ.get('BINANCE_TESTNET_SECRET')
        
        if not api_key or not api_secret:
            logger.error("No se encontraron las claves API en variables de entorno")
            return
        
        logger.info("Claves API configuradas correctamente")
        
        # Configurar exchange con modo testnet
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        
        # Activar modo sandbox/testnet
        exchange.set_sandbox_mode(True)
        logger.info("Modo testnet activado")
        
        # Cargar mercados
        logger.info("Cargando mercados...")
        await exchange.load_markets()
        logger.info(f"Mercados cargados. {len(exchange.markets)} mercados disponibles")
        
        # Comprobar que los símbolos están disponibles
        available_symbols = []
        for symbol in SYMBOLS:
            if symbol in exchange.markets:
                available_symbols.append(symbol)
                logger.info(f"Símbolo {symbol} disponible")
            else:
                logger.warning(f"Símbolo {symbol} no disponible en el exchange")
        
        # Descargar datos para cada símbolo y timeframe
        for symbol in available_symbols:
            for timeframe in TIMEFRAMES:
                try:
                    logger.info(f"Descargando {symbol} - {timeframe}...")
                    
                    # Obtener datos históricos
                    since = exchange.parse8601('2023-01-01T00:00:00Z')
                    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since)
                    
                    if ohlcv and len(ohlcv) > 0:
                        logger.info(f"Descargados {len(ohlcv)} registros para {symbol} - {timeframe}")
                        
                        # Guardar en CSV
                        await save_to_csv(symbol, timeframe, ohlcv)
                    else:
                        logger.warning(f"No se encontraron datos para {symbol} - {timeframe}")
                    
                    # Esperar para respetar límites de API
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error descargando {symbol} - {timeframe}: {e}")
                    # Continuar con el siguiente par de símbolo/timeframe
        
        # Cerrar la conexión con el exchange
        await exchange.close()
        logger.info("Conexión con el exchange cerrada correctamente")
        
    except Exception as e:
        logger.error(f"Error en la descarga de datos: {e}")
        if 'exchange' in locals():
            await exchange.close()

async def main():
    """Función principal."""
    try:
        await fetch_testnet_data()
        logger.info("Proceso de descarga finalizado")
    except Exception as e:
        logger.error(f"Error en el proceso principal: {e}")

if __name__ == "__main__":
    logger.info("Iniciando script de descarga de datos de Binance Testnet")
    asyncio.run(main())