#!/usr/bin/env python
"""
Script para descargar datos históricos de Binance Testnet.

Este script se conecta a Binance Testnet, descarga datos OHLCV históricos
y los almacena en la base de datos para usarlos en pruebas.
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
from genesis.db.repository import Repository
from genesis.db.models import Candle, Symbol, Exchange

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


async def ensure_db_records(repo: Repository) -> Dict[str, int]:
    """
    Asegurarse de que existan los registros de exchange y símbolos en la base de datos.
    
    Args:
        repo: Repositorio para acceder a la base de datos
        
    Returns:
        Diccionario con IDs de símbolos por nombre
    """
    # Verificar/crear registro de exchange para Binance
    exchange = await repo.get_by_field(Exchange, "name", "binance")
    if not exchange:
        exchange = Exchange(
            name="binance",
            description="Binance Cryptocurrency Exchange",
            enabled=True
        )
        exchange_id = await repo.create(exchange)
        logger.info(f"Creado registro de exchange para Binance (ID: {exchange_id})")
    else:
        exchange_id = exchange.id
        logger.info(f"Exchange Binance ya existe en BD (ID: {exchange_id})")
    
    # Verificar/crear registros de símbolos
    symbol_ids = {}
    for symbol_name in SYMBOLS:
        base, quote = symbol_name.split('/')
        
        symbol = await repo.get_by_field(Symbol, "name", symbol_name)
        if not symbol:
            symbol = Symbol(
                name=symbol_name,
                exchange_id=exchange_id,
                base_asset=base,
                quote_asset=quote,
                enabled=True
            )
            symbol_id = await repo.create(symbol)
            logger.info(f"Creado registro de símbolo para {symbol_name} (ID: {symbol_id})")
        else:
            symbol_id = symbol.id
            logger.info(f"Símbolo {symbol_name} ya existe en BD (ID: {symbol_id})")
            
        symbol_ids[symbol_name] = symbol_id
    
    return symbol_ids


async def store_candle_data(
    repo: Repository,
    symbol_id: int,
    timeframe: str,
    data: List[Dict[str, Any]]
) -> int:
    """
    Almacenar datos de velas en la base de datos.
    
    Args:
        repo: Repositorio para acceder a la base de datos
        symbol_id: ID del símbolo en la base de datos
        timeframe: Marco temporal de las velas
        data: Datos de velas formateados
        
    Returns:
        Número de velas almacenadas
    """
    count = 0
    for candle in data:
        # Convertir timestamp a datetime
        timestamp = datetime.fromtimestamp(candle['timestamp'] / 1000)
        
        # Verificar si la vela ya existe
        existing = await repo.query(
            Candle, 
            f"symbol_id = {symbol_id} AND timeframe = '{timeframe}' AND timestamp = '{timestamp}'"
        )
        
        if not existing:
            # Crear registro de vela
            candle_record = Candle(
                symbol_id=symbol_id,
                timeframe=timeframe,
                timestamp=timestamp,
                open=candle['open'],
                high=candle['high'],
                low=candle['low'],
                close=candle['close'],
                volume=candle['volume']
            )
            
            await repo.create(candle_record)
            count += 1
    
    return count


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
        
    # Inicializar repositorio
    repo = Repository()
    
    # Asegurar que existan registros en la BD
    symbol_ids = await ensure_db_records(repo)
    
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
                
                # Formatear datos
                formatted_data = []
                for candle in ohlcv_data:
                    timestamp, open_price, high, low, close, volume = candle
                    formatted_data.append({
                        'timestamp': timestamp,
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': volume
                    })
                
                # Almacenar en base de datos
                count = await store_candle_data(
                    repo=repo,
                    symbol_id=symbol_ids[symbol],
                    timeframe=timeframe,
                    data=formatted_data
                )
                
                logger.info(f"Almacenadas {count} nuevas velas para {symbol} - {timeframe}")
                
                # Evitar superar límites de API
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error al descargar datos para {symbol} - {timeframe}: {e}")
    
    # Cerrar conexión al exchange
    await exchange.stop()
    logger.info("Proceso completado")


if __name__ == "__main__":
    asyncio.run(main())