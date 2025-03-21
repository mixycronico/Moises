"""
Base classes for exchange integration.

This module defines the interfaces for exchange operations, providing
a consistent abstraction over different exchange APIs.
"""

import abc
from typing import Dict, List, Any, Optional, Tuple

from genesis.core.base import Component


class Exchange(Component):
    """Base abstract class for exchange integration."""
    
    def __init__(self, name: str, exchange_id: str):
        """
        Initialize the exchange component.
        
        Args:
            name: Component name
            exchange_id: Exchange identifier (e.g., 'binance', 'kraken')
        """
        super().__init__(name)
        self.exchange_id = exchange_id
    
    @abc.abstractmethod
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Ticker data dictionary
        """
        pass
    
    @abc.abstractmethod
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
        pass
    
    @abc.abstractmethod
    async def fetch_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch order book for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of orders to fetch
            
        Returns:
            Order book data
        """
        pass
    
    @abc.abstractmethod
    async def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance.
        
        Returns:
            Account balance data
        """
        pass
    
    @abc.abstractmethod
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
        pass
    
    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol (optional for some exchanges)
            
        Returns:
            Cancellation result
        """
        pass
    
    @abc.abstractmethod
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
        pass
    
    @abc.abstractmethod
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch open orders.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of open orders
        """
        pass
    
    @abc.abstractmethod
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
        pass
    
    @abc.abstractmethod
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
        pass
