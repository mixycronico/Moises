"""
CCXT exchange wrapper implementation.

This module provides a wrapper around the CCXT library for unified
cryptocurrency exchange functionality.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union

import ccxt.async_support as ccxt
from ccxt.base.errors import NetworkError, ExchangeError, RequestTimeout

from genesis.config.settings import settings
from genesis.core.base import Component
from genesis.exchanges.base import Exchange
from genesis.utils.logger import setup_logging


class CCXTExchange(Exchange):
    """
    CCXT exchange wrapper implementation.
    
    This class wraps the CCXT library to provide a consistent interface
    for interacting with cryptocurrency exchanges.
    """
    
    def __init__(
        self, exchange_id: str, 
        api_key: Optional[str] = None, 
        secret: Optional[str] = None,
        password: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the CCXT exchange wrapper.
        
        Args:
            exchange_id: CCXT exchange ID (e.g., 'binance', 'kraken')
            api_key: API key for authentication
            secret: API secret for authentication
            password: Additional password/passphrase if required
            config: Additional exchange-specific configuration
        """
        super().__init__(f"exchange.{exchange_id}", exchange_id)
        
        self.logger = setup_logging(f"exchange.{exchange_id}")
        
        # Default configuration
        self.config = {
            'enableRateLimit': True,
            'timeout': settings.get('exchanges.connection_timeout', 30) * 1000,  # milliseconds
            'asyncio_loop': asyncio.get_event_loop()
        }
        
        # Update with user-provided config
        if config:
            self.config.update(config)
        
        # Initialize exchange
        self.api_key = api_key
        self.secret = secret
        self.exchange: Optional[ccxt.Exchange] = None
        
        # Create the CCXT exchange instance
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'password': password,
            **self.config
        })
        
        # Rate limiting
        self.rate_limit = {
            'last_request_time': 0.0,
            'min_time_between_requests': 1.0 / settings.get('exchanges.rate_limit.max_requests', 10)
        }
        
        # Market cache
        self.markets: Dict[str, Any] = {}
        self.tickers_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self.ticker_cache_ttl = 10.0  # seconds
        
    async def start(self) -> None:
        """Start the exchange component."""
        await super().start()
        
        try:
            self.logger.info(f"Initializing {self.exchange_id} exchange")
            await self.exchange.load_markets()
            self.markets = self.exchange.markets
            
            self.logger.info(f"Initialized {self.exchange_id} with {len(self.markets)} markets")
            
            # Emit exchange ready event
            await self.emit_event("exchange.ready", {
                "exchange_id": self.exchange_id,
                "market_count": len(self.markets)
            })
        except Exception as e:
            self.logger.error(f"Error initializing exchange: {e}")
            await self.emit_event("exchange.error", {
                "exchange_id": self.exchange_id,
                "error": str(e)
            })
            raise
    
    async def stop(self) -> None:
        """Stop the exchange component."""
        if self.exchange:
            self.logger.info(f"Closing {self.exchange_id} exchange connection")
            await self.exchange.close()
        
        await super().stop()
    
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
                    self.logger.warning(
                        f"Temporary error on {method_name}, retrying ({attempt+1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"Failed {method_name} after {max_retries} attempts: {e}")
                    raise
            except ExchangeError as e:
                self.logger.error(f"Exchange error on {method_name}: {e}")
                raise
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Handle events from the event bus."""
        if event_type == "market.ticker_request" and data.get("exchange_id") == self.exchange_id:
            symbol = data.get("symbol")
            if symbol:
                try:
                    ticker = await self.fetch_ticker(symbol)
                    await self.emit_event("market.ticker", {
                        "exchange_id": self.exchange_id,
                        "symbol": symbol,
                        "ticker": ticker
                    })
                except Exception as e:
                    self.logger.error(f"Error fetching ticker for {symbol}: {e}")
                    await self.emit_event("market.error", {
                        "exchange_id": self.exchange_id,
                        "symbol": symbol,
                        "error": str(e)
                    })
    
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
        return await self._execute_with_retry(
            'fetch_ohlcv', symbol, timeframe, since, limit
        )
    
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
    
    async def create_order(
        self, symbol: str, order_type: str, side: str, 
        amount: float, price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type ('limit', 'market')
            side: Order side ('buy', 'sell')
            amount: Order amount
            price: Order price (for limit orders)
            
        Returns:
            Order data
        """
        # Check if we're in dry run mode
        if settings.get('trading.dry_run', True):
            self.logger.info(
                f"DRY RUN: Would create {order_type} {side} order for {amount} {symbol} @ {price}"
            )
            # Return a simulated order response
            return {
                'id': f'dry_run_{int(time.time())}',
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': price,
                'status': 'closed',  # Simulate immediate completion
                'timestamp': int(time.time() * 1000),
                'datetime': self.exchange.iso8601(int(time.time() * 1000)),
                'fee': None,
                'cost': amount * (price or 0),
                'filled': amount,
                'remaining': 0,
                'info': {'dry_run': True}
            }
        
        # Execute real order
        return await self._execute_with_retry(
            'create_order', symbol, order_type, side, amount, price
        )
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol (optional for some exchanges)
            
        Returns:
            Cancellation result
        """
        if settings.get('trading.dry_run', True):
            self.logger.info(f"DRY RUN: Would cancel order {order_id} for {symbol}")
            return {'id': order_id, 'status': 'canceled'}
        
        return await self._execute_with_retry('cancel_order', order_id, symbol)
    
    async def fetch_orders(self, symbol: Optional[str] = None, since: Optional[int] = None, 
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch orders history.
        
        Args:
            symbol: Trading pair symbol
            since: Starting timestamp in milliseconds
            limit: Maximum number of orders to fetch
            
        Returns:
            List of orders
        """
        return await self._execute_with_retry('fetch_orders', symbol, since, limit)
    
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch open orders.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of open orders
        """
        return await self._execute_with_retry('fetch_open_orders', symbol)
    
    async def fetch_closed_orders(self, symbol: Optional[str] = None, 
                           since: Optional[int] = None, 
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch closed orders.
        
        Args:
            symbol: Trading pair symbol
            since: Starting timestamp in milliseconds
            limit: Maximum number of orders to fetch
            
        Returns:
            List of closed orders
        """
        return await self._execute_with_retry('fetch_closed_orders', symbol, since, limit)
    
    async def fetch_my_trades(self, symbol: Optional[str] = None, 
                       since: Optional[int] = None, 
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch trades history.
        
        Args:
            symbol: Trading pair symbol
            since: Starting timestamp in milliseconds
            limit: Maximum number of trades to fetch
            
        Returns:
            List of trades
        """
        return await self._execute_with_retry('fetch_my_trades', symbol, since, limit)
