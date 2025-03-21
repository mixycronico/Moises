#!/usr/bin/env python
"""
Script para descargar datos históricos de Binance Testnet y guardarlos en archivos.

Este script se conecta a Binance Testnet, descarga datos OHLCV históricos
y los almacena en archivos CSV para usarlos en pruebas.
"""

import os
import sys
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Añadimos el directorio raíz al path para importar los módulos de Genesis
sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis.config.settings import settings
from genesis.exchanges.ccxt_wrapper import CCXTExchange

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Símbolos para los que descargar datos
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

# Timeframes a descargar
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Período de datos a descargar (en días)
DAYS_TO_FETCH = 30

# Directorio donde guardar los datos
DATA_DIR = Path(__file__).parent.parent / "data" / "historical"


async def save_to_csv(symbol: str, timeframe: str, data: List[List[float]]) -> None:
    """
    Guardar datos OHLCV en un archivo CSV.
    
    Args:
        symbol: Símbolo de trading
        timeframe: Marco temporal
        data: Datos OHLCV
    """
    # Crear directorio si no existe
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Formatear símbolo para nombre de archivo
    symbol_file = symbol.replace('/', '_')
    
    # Crear dataframe
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convertir timestamp a datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Ordenar por timestamp
    df = df.sort_values('timestamp')
    
    # Guardar a CSV
    filename = DATA_DIR / f"{symbol_file}_{timeframe}.csv"
    df.to_csv(filename, index=False)
    
    logger.info(f"Guardados {len(df)} registros en {filename}")


async def main():
    """Función principal del script."""
    logger.info("Iniciando descarga de datos históricos de Binance Testnet")
    
    # Configurar conexión a Binance Testnet
    exchange = CCXTExchange(
        exchange_id="binance", 
        config={
            "testnet": True
        }
    )
    
    # Iniciar exchange
    try:
        await exchange.start()
        logger.info("Conectado a Binance Testnet")
    except Exception as e:
        logger.error(f"Error al conectar a Binance Testnet: {e}")
        return
    
    # Calcular fecha de inicio
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    
    since = int(start_date.timestamp() * 1000)  # Convertir a milisegundos
    
    # Descargar datos para cada símbolo y timeframe
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            try:
                logger.info(f"Descargando datos para {symbol} - {timeframe}")
                
                # Descargar datos OHLCV
                ohlcv_data = await exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000  # Máximo permitido por la mayoría de exchanges
                )
                
                if ohlcv_data:
                    # Guardar a CSV
                    await save_to_csv(symbol, timeframe, ohlcv_data)
                else:
                    logger.warning(f"No se obtuvieron datos para {symbol} - {timeframe}")
                
                # Evitar superar límites de API
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error al descargar datos para {symbol} - {timeframe}: {e}")
    
    # Cerrar conexión al exchange
    await exchange.stop()
    logger.info("Proceso completado")


if __name__ == "__main__":
    asyncio.run(main())