"""
Simple CCXT exchange wrapper implementation.

This module provides a simplified wrapper around the CCXT library that doesn't depend
on the event bus or other components, making it suitable for demos and scripts.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple

import ccxt.async_support as ccxt
from ccxt.base.errors import NetworkError, ExchangeError, RequestTimeout

class SimpleCCXTExchange:
    """
    Simplified CCXT exchange wrapper implementation.
    
    This class wraps the CCXT library to provide a minimal interface
    for interacting with cryptocurrency exchanges without dependencies.
    """
    
    def __init__(
        self, exchange_id: str, 
        api_key: Optional[str] = None, 
        secret: Optional[str] = None,
        password: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the simplified CCXT exchange wrapper.
        
        Args:
            exchange_id: CCXT exchange ID (e.g., 'binance', 'kraken')
            api_key: API key for authentication
            secret: API secret for authentication
            password: Additional password/passphrase if required
            config: Additional exchange-specific configuration
        """
        self.exchange_id = exchange_id
        
        # Default configuration
        self.config = {
            'enableRateLimit': True,
            'timeout': 30000,  # milliseconds
        }
        
        # Update with user-provided config
        if config:
            self.config.update(config)
        
        # Initialize exchange
        self.api_key = api_key
        self.secret = secret
        self.exchange = None
        
        # Create the CCXT exchange instance
        exchange_class = getattr(ccxt, exchange_id)
        
        # Preparar configuración
        exchange_config = {
            'apiKey': api_key,
            'secret': secret,
            'password': password
        }
        
        # Si estamos en modo testnet, lo configuramos después de crear la instancia
        self.use_testnet = bool(self.config.get('testnet'))
        
        # Agregar el resto de la configuración
        exchange_config.update(self.config)
        
        # Crear instancia del exchange
        self.exchange = exchange_class(exchange_config)
        
        # Rate limiting
        self.rate_limit = {
            'last_request_time': 0.0,
            'min_time_between_requests': 0.1  # 10 requests per second max
        }
        
        # Market cache
        self.markets = {}
        self.tickers_cache = {}
        self.ticker_cache_ttl = 10.0  # seconds
    
    async def initialize(self) -> bool:
        """Initialize the exchange wrapper."""
        try:
            print(f"Initializing {self.exchange_id} exchange")
            
            # Si estamos en modo testnet, lo configuramos antes de cargar los mercados
            if self.use_testnet:
                print(f"Activando modo testnet para {self.exchange_id}")
                # Usar el método oficial de CCXT para activar el modo sandbox/testnet
                self.exchange.set_sandbox_mode(True)
            
            await self.exchange.load_markets()
            self.markets = self.exchange.markets
            
            print(f"Initialized {self.exchange_id} with {len(self.markets)} markets")
            
            # Verificar la conexión con una operación básica
            try:
                if self.exchange_id.lower() == 'binance':
                    ticker = await self.fetch_ticker('BTC/USDT')
                    print(f"Conexión verificada. Precio BTC/USDT: {ticker['last']}")
            except Exception as ticker_e:
                print(f"Prueba de conexión no completada: {ticker_e}")
            
            return True
        except Exception as e:
            print(f"Error initializing exchange: {e}")
            return False
    
    async def close(self) -> None:
        """Close the exchange connection."""
        if self.exchange:
            print(f"Closing {self.exchange_id} exchange connection")
            await self.exchange.close()
    
    async def _execute_with_retry(self, method_name: str, *args, **kwargs) -> Any:
        """
        Execute a CCXT method with automatic retry on temporary errors.
        
        Args:
            method_name: Name of the CCXT method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
            
        Returns:
            Result of the method call
        
        Raises:
            ExchangeError: If the operation fails after all retries
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized")
        
        method = getattr(self.exchange, method_name)
        max_retries = 3
        retry_delay = 1.0  # seconds
        
        # Apply basic rate limiting
        now = time.time()
        time_since_last = now - self.rate_limit['last_request_time']
        if time_since_last < self.rate_limit['min_time_between_requests']:
            await asyncio.sleep(self.rate_limit['min_time_between_requests'] - time_since_last)
        
        self.rate_limit['last_request_time'] = time.time()
        
        for attempt in range(max_retries):
            try:
                result = await method(*args, **kwargs)
                return result
            except (NetworkError, RequestTimeout) as e:
                if attempt < max_retries - 1:
                    print(f"Temporary error on {method_name}, retrying ({attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"Failed {method_name} after {max_retries} attempts: {e}")
                    raise
            except ExchangeError as e:
                print(f"Exchange error on {method_name}: {e}")
                raise
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data for a symbol with caching.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Ticker data dictionary
        """
        now = time.time()
        
        # Check cache first
        if symbol in self.tickers_cache:
            timestamp, ticker = self.tickers_cache[symbol]
            if now - timestamp < self.ticker_cache_ttl:
                return ticker
        
        # Fetch fresh data
        ticker = await self._execute_with_retry('fetch_ticker', symbol)
        
        # Update cache
        self.tickers_cache[symbol] = (now, ticker)
        
        return ticker
    
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = '1h', 
        since: Optional[int] = None, limit: Optional[int] = None
    ) -> List[List[float]]:
        """
        Fetch OHLCV (candlestick) data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string ('1m', '5m', '1h', etc.)
            since: Starting timestamp in milliseconds
            limit: Maximum number of candles to fetch
            
        Returns:
            List of OHLCV candles
        """
        return await self._execute_with_retry('fetch_ohlcv', symbol, timeframe, since, limit)
    
    async def fetch_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch order book for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of orders to fetch
            
        Returns:
            Order book data
        """
        return await self._execute_with_retry('fetch_order_book', symbol, limit)
    
    async def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance.
        
        Returns:
            Account balance data
        """
        return await self._execute_with_retry('fetch_balance')