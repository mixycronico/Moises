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
from datetime import datetime, timedelta
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

# Ordenamos los símbolos y timeframes por prioridad
# Primero los timeframes más importantes para el análisis a largo plazo
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']  # Principales pares para trading
TIMEFRAMES = ['1d', '4h', '1h', '15m', '5m', '1m']  # Primero los timeframes más importantes
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

async def fetch_ohlcv_data(exchange, symbol, timeframe, start_date, end_date=None):
    """
    Obtener datos históricos OHLCV completos entre fechas
    haciendo múltiples solicitudes si es necesario.
    
    Args:
        exchange: Instancia del exchange
        symbol: Símbolo de trading
        timeframe: Marco temporal
        start_date: Fecha de inicio en formato ISO
        end_date: Fecha de fin en formato ISO (opcional, por defecto hasta hoy)
        
    Returns:
        Lista con datos OHLCV
    """
    logger.info(f"Obteniendo datos históricos completos para {symbol} - {timeframe}")
    
    # Convertir fechas a timestamps
    since = exchange.parse8601(start_date)
    until = exchange.parse8601(end_date) if end_date else int(datetime.now().timestamp() * 1000)
    
    # Límite máximo de velas por petición
    limit = 1000
    
    # Almacenar todos los datos
    all_ohlcv = []
    
    # Fecha actual para las peticiones
    current_since = since
    
    # Contador de peticiones
    request_count = 0
    
    # Bucle para obtener todos los datos solicitados
    while current_since < until:
        try:
            # Realizar petición
            batch = await exchange.fetch_ohlcv(symbol, timeframe, current_since, limit)
            request_count += 1
            
            if not batch or len(batch) == 0:
                logger.warning(f"No se obtuvieron datos para {symbol} - {timeframe} desde {current_since}")
                break
                
            logger.info(f"Obtenidos {len(batch)} registros para {symbol} - {timeframe} (petición #{request_count})")
            
            # Añadir datos a la lista completa
            all_ohlcv.extend(batch)
            
            # Avanzar a la siguiente fecha
            last_timestamp = batch[-1][0]  # Timestamp de la última vela
            
            # Si el último timestamp es igual al actual, avanzamos un poco para evitar bucles
            if last_timestamp == current_since:
                # Avanzamos el tiempo según el timeframe
                if timeframe == '1m':
                    current_since += 60 * 1000  # 1 minuto en ms
                elif timeframe == '5m':
                    current_since += 5 * 60 * 1000
                elif timeframe == '15m':
                    current_since += 15 * 60 * 1000
                elif timeframe == '1h':
                    current_since += 60 * 60 * 1000
                elif timeframe == '4h':
                    current_since += 4 * 60 * 60 * 1000
                elif timeframe == '1d':
                    current_since += 24 * 60 * 60 * 1000
                else:
                    current_since += 24 * 60 * 60 * 1000  # Por defecto, un día
            else:
                current_since = last_timestamp + 1  # Avanzamos 1ms después del último dato
            
            # Respetar límites de API
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error en la petición para {symbol} - {timeframe}: {e}")
            await asyncio.sleep(2)  # Esperar más tiempo si hay error
            # Avanzar un poco para intentar superar el punto problemático
            current_since += 24 * 60 * 60 * 1000  # Avanzar un día
    
    # Eliminar duplicados (si los hay)
    if all_ohlcv:
        # Convertir a DataFrame para facilitar la eliminación de duplicados
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Convertir de nuevo a lista
        final_ohlcv = df.values.tolist()
        
        logger.info(f"Total de registros únicos obtenidos para {symbol} - {timeframe}: {len(final_ohlcv)}")
        return final_ohlcv
    
    return all_ohlcv

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
        
        # Fechas para obtener datos históricos (5 años)
        start_date = '2020-01-01T00:00:00Z'  # 5 años atrás
        
        # Descargar datos para cada símbolo y timeframe
        for symbol in available_symbols:
            for timeframe in TIMEFRAMES:
                try:
                    logger.info(f"Descargando historial completo de {symbol} - {timeframe}...")
                    
                    # Obtener datos históricos completos
                    # Para timeframes pequeños (1m, 5m) limitamos a un período más corto 
                    # para no saturar la API y asegurar que obtenemos primero los datos más importantes
                    if timeframe == '1m':
                        tf_start_date = '2023-01-01T00:00:00Z'  # Para 1m solo 2 años
                    elif timeframe == '5m':
                        tf_start_date = '2022-01-01T00:00:00Z'  # Para 5m solo 3 años
                    else:
                        tf_start_date = start_date  # Para el resto, los 5 años completos
                        
                    ohlcv = await fetch_ohlcv_data(exchange, symbol, timeframe, tf_start_date)
                    
                    if ohlcv and len(ohlcv) > 0:
                        logger.info(f"Descargados {len(ohlcv)} registros históricos para {symbol} - {timeframe}")
                        
                        # Guardar en CSV
                        await save_to_csv(symbol, timeframe, ohlcv)
                    else:
                        logger.warning(f"No se encontraron datos históricos para {symbol} - {timeframe}")
                    
                    # Esperar para respetar límites de API
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error descargando historial de {symbol} - {timeframe}: {e}")
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