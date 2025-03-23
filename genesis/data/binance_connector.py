"""
Conector para obtener datos históricos y en tiempo real de Binance.

Este módulo proporciona funciones para obtener datos OHLCV y orderbook
desde Binance, tanto históricos como en tiempo real a través de WebSockets.
Integra con el Sistema Genesis para alimentar modelos de ML y RL.
"""

import pandas as pd
import numpy as np
import logging
import time
import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import ccxt
import ccxt.async_support as ccxtasync
from functools import lru_cache

class BinanceConnector:
    """
    Conector para Binance que proporciona acceso a datos históricos y en tiempo real.
    
    Esta clase facilita la obtención de datos OHLCV, orderbook y trades desde
    Binance, tanto para backtesting como para trading en vivo.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = True,
                 use_async: bool = True,
                 cache_dir: str = './cache/market_data'):
        """
        Inicializar conector de Binance.
        
        Args:
            api_key: API key de Binance (opcional)
            api_secret: API secret de Binance (opcional)
            testnet: Si es True, usa Binance Testnet
            use_async: Si es True, usa versión asíncrona de ccxt
            cache_dir: Directorio para caché de datos
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.use_async = use_async
        self.cache_dir = cache_dir
        
        # Crear directorio de caché si no existe
        os.makedirs(cache_dir, exist_ok=True)
        
        # Inicializar exchange
        self.exchange = None
        self.async_exchange = None
        
        # Inicializar exchange
        self._initialize_exchange()
        
        self.logger.info(f"BinanceConnector inicializado (testnet={testnet}, async={use_async})")
    
    def _initialize_exchange(self) -> None:
        """Inicializar instancia del exchange."""
        # Configuración común
        config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True
        }
        
        if self.testnet:
            config['test'] = True
        
        # Inicializar versión síncrona
        self.exchange = ccxt.binance(config)
        
        # Inicializar versión asíncrona si se requiere
        if self.use_async:
            self.async_exchange = ccxtasync.binance(config)
    
    def fetch_ohlcv(self, 
                    symbol: str, 
                    timeframe: str = '1h', 
                    since: Optional[int] = None,
                    limit: int = 1000) -> pd.DataFrame:
        """
        Obtener datos OHLCV (velas) desde Binance.
        
        Args:
            symbol: Símbolo del par (ej. 'BTC/USDT')
            timeframe: Marco temporal ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            since: Timestamp en milisegundos desde cuando obtener datos
            limit: Número máximo de registros a obtener
            
        Returns:
            DataFrame con datos OHLCV
        """
        try:
            # Obtener datos
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convertir timestamp a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error obteniendo OHLCV para {symbol}: {str(e)}")
            raise
    
    async def fetch_ohlcv_async(self, 
                           symbol: str, 
                           timeframe: str = '1h', 
                           since: Optional[int] = None,
                           limit: int = 1000) -> pd.DataFrame:
        """
        Obtener datos OHLCV (velas) desde Binance de forma asíncrona.
        
        Args:
            symbol: Símbolo del par (ej. 'BTC/USDT')
            timeframe: Marco temporal ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            since: Timestamp en milisegundos desde cuando obtener datos
            limit: Número máximo de registros a obtener
            
        Returns:
            DataFrame con datos OHLCV
        """
        if not self.use_async or self.async_exchange is None:
            # Si no se configuró para async, usar versión síncrona
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self.fetch_ohlcv(symbol, timeframe, since, limit)
            )
        
        try:
            # Obtener datos de forma asíncrona
            ohlcv = await self.async_exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convertir timestamp a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error obteniendo OHLCV async para {symbol}: {str(e)}")
            raise
    
    async def fetch_full_ohlcv(self, 
                          symbol: str, 
                          timeframe: str = '1h', 
                          start_date: Union[str, datetime, int] = None,
                          end_date: Union[str, datetime, int] = None) -> pd.DataFrame:
        """
        Obtener datos OHLCV completos entre fechas realizando múltiples llamadas.
        
        Args:
            symbol: Símbolo del par (ej. 'BTC/USDT')
            timeframe: Marco temporal ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            start_date: Fecha de inicio (datetime, string 'YYYY-MM-DD' o timestamp)
            end_date: Fecha de fin (datetime, string 'YYYY-MM-DD' o timestamp)
            
        Returns:
            DataFrame con datos OHLCV completos
        """
        # Convertir fechas a timestamp en ms
        if start_date is None:
            # Por defecto, 1 mes atrás
            start_date = datetime.now() - timedelta(days=30)
        
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        
        if isinstance(start_date, datetime):
            start_ms = int(start_date.timestamp() * 1000)
        else:
            start_ms = int(start_date)  # Asumir que ya es timestamp en ms
        
        if end_date is None:
            # Por defecto, ahora
            end_date = datetime.now()
        
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if isinstance(end_date, datetime):
            end_ms = int(end_date.timestamp() * 1000)
        else:
            end_ms = int(end_date)  # Asumir que ya es timestamp en ms
        
        # Obtener información de timeframes
        timeframe_ms = self._get_timeframe_ms(timeframe)
        limit = 1000  # Máximo número de velas por solicitud
        
        # Calcular número de solicitudes necesarias
        total_candles = (end_ms - start_ms) / timeframe_ms
        num_requests = int(total_candles / limit) + 1
        
        self.logger.info(f"Obteniendo OHLCV completo para {symbol} ({timeframe}): {num_requests} solicitudes")
        
        # Descargar datos en partes
        all_data = []
        current_start = start_ms
        
        for i in range(num_requests):
            if current_start >= end_ms:
                break
            
            # Obtener datos para este segmento
            df = await self.fetch_ohlcv_async(symbol, timeframe, current_start, limit)
            
            if df.empty:
                break
            
            all_data.append(df)
            
            # Actualizar inicio para la siguiente solicitud
            if len(df) < limit:
                break
            
            # Usar la última timestamp + timeframe como inicio para la siguiente solicitud
            last_timestamp = df['timestamp'].iloc[-1]
            current_start = int(last_timestamp.timestamp() * 1000) + timeframe_ms
            
            # Esperar un poco para evitar rate limiting
            await asyncio.sleep(0.2)
        
        if not all_data:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Combinar todos los datos
        full_data = pd.concat(all_data)
        
        # Eliminar duplicados y ordenar por timestamp
        full_data = full_data.drop_duplicates(subset='timestamp')
        full_data = full_data.sort_values('timestamp').reset_index(drop=True)
        
        return full_data
    
    def _get_timeframe_ms(self, timeframe: str) -> int:
        """
        Convertir timeframe a milisegundos.
        
        Args:
            timeframe: Marco temporal ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            
        Returns:
            Duración del timeframe en milisegundos
        """
        # Map timeframe to milliseconds
        timeframe_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000
        }
        
        if timeframe not in timeframe_map:
            raise ValueError(f"Timeframe no soportado: {timeframe}")
        
        return timeframe_map[timeframe]
    
    def fetch_order_book(self, 
                        symbol: str, 
                        limit: int = 100) -> Dict[str, Any]:
        """
        Obtener libro de órdenes para un símbolo.
        
        Args:
            symbol: Símbolo del par (ej. 'BTC/USDT')
            limit: Número de niveles de precio
            
        Returns:
            Diccionario con libro de órdenes
        """
        try:
            # Obtener orderbook
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            
            return orderbook
            
        except Exception as e:
            self.logger.error(f"Error obteniendo orderbook para {symbol}: {str(e)}")
            raise
    
    async def fetch_order_book_async(self, 
                               symbol: str, 
                               limit: int = 100) -> Dict[str, Any]:
        """
        Obtener libro de órdenes para un símbolo de forma asíncrona.
        
        Args:
            symbol: Símbolo del par (ej. 'BTC/USDT')
            limit: Número de niveles de precio
            
        Returns:
            Diccionario con libro de órdenes
        """
        if not self.use_async or self.async_exchange is None:
            # Si no se configuró para async, usar versión síncrona
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self.fetch_order_book(symbol, limit)
            )
        
        try:
            # Obtener orderbook de forma asíncrona
            orderbook = await self.async_exchange.fetch_order_book(symbol, limit)
            
            return orderbook
            
        except Exception as e:
            self.logger.error(f"Error obteniendo orderbook async para {symbol}: {str(e)}")
            raise
    
    def fetch_tickers(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Obtener tickers para uno o varios símbolos.
        
        Args:
            symbols: Lista de símbolos (None = todos)
            
        Returns:
            Diccionario con tickers por símbolo
        """
        try:
            # Obtener tickers
            tickers = self.exchange.fetch_tickers(symbols)
            
            return tickers
            
        except Exception as e:
            symbol_str = ','.join(symbols) if symbols else 'all'
            self.logger.error(f"Error obteniendo tickers para {symbol_str}: {str(e)}")
            raise
    
    async def fetch_tickers_async(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Obtener tickers para uno o varios símbolos de forma asíncrona.
        
        Args:
            symbols: Lista de símbolos (None = todos)
            
        Returns:
            Diccionario con tickers por símbolo
        """
        if not self.use_async or self.async_exchange is None:
            # Si no se configuró para async, usar versión síncrona
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self.fetch_tickers(symbols)
            )
        
        try:
            # Obtener tickers de forma asíncrona
            tickers = await self.async_exchange.fetch_tickers(symbols)
            
            return tickers
            
        except Exception as e:
            symbol_str = ','.join(symbols) if symbols else 'all'
            self.logger.error(f"Error obteniendo tickers async para {symbol_str}: {str(e)}")
            raise
    
    def save_ohlcv_to_cache(self, 
                           symbol: str, 
                           timeframe: str, 
                           data: pd.DataFrame) -> str:
        """
        Guardar datos OHLCV en caché.
        
        Args:
            symbol: Símbolo del par
            timeframe: Marco temporal
            data: DataFrame con datos
            
        Returns:
            Ruta donde se guardaron los datos
        """
        # Crear directorio para el símbolo si no existe
        symbol_dir = os.path.join(self.cache_dir, symbol.replace('/', '_'))
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Generar nombre de archivo
        filename = f"{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(symbol_dir, filename)
        
        # Guardar datos
        data.to_csv(filepath, index=False)
        
        self.logger.info(f"Datos OHLCV guardados en: {filepath}")
        
        return filepath
    
    def load_ohlcv_from_cache(self, 
                             symbol: str, 
                             timeframe: str,
                             max_age_days: int = 1) -> Optional[pd.DataFrame]:
        """
        Cargar datos OHLCV desde caché.
        
        Args:
            symbol: Símbolo del par
            timeframe: Marco temporal
            max_age_days: Edad máxima de los datos en días
            
        Returns:
            DataFrame con datos o None si no hay datos
        """
        # Directorio para el símbolo
        symbol_dir = os.path.join(self.cache_dir, symbol.replace('/', '_'))
        
        if not os.path.exists(symbol_dir):
            return None
        
        # Buscar archivos que coincidan con el timeframe
        prefix = f"{timeframe}_"
        matching_files = [f for f in os.listdir(symbol_dir) if f.startswith(prefix) and f.endswith('.csv')]
        
        if not matching_files:
            return None
        
        # Obtener el archivo más reciente
        latest_file = max(matching_files)
        filepath = os.path.join(symbol_dir, latest_file)
        
        # Verificar edad del archivo
        file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))).days
        
        if file_age > max_age_days:
            self.logger.info(f"Datos en caché para {symbol} ({timeframe}) son demasiado antiguos: {file_age} días")
            return None
        
        # Cargar datos
        try:
            data = pd.read_csv(filepath)
            
            # Convertir timestamp a datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Datos OHLCV cargados desde: {filepath}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error cargando datos desde caché: {str(e)}")
            return None
    
    async def close(self) -> None:
        """Cerrar conexiones."""
        if self.async_exchange is not None:
            await self.async_exchange.close()
            self.async_exchange = None
        
        self.logger.info("Conexiones cerradas")