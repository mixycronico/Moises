"""
Market data fetching and processing.

This module handles the acquisition and basic processing of market data
from exchanges and other sources.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from genesis.config.settings import settings
from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class MarketDataManager(Component):
    """
    Manager for fetching and distributing market data.
    
    This component fetches market data from exchanges and distributes it
    to other components through the event bus.
    """
    
    def __init__(self, name: str = "market_data"):
        """
        Initialize the market data manager.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Configuration
        self.default_interval = settings.get('trading.interval', '1h')
        self.default_symbols = settings.get('trading.default_symbols', ["BTC/USDT"])
        self.update_interval = 60  # seconds between checks
        
        # State tracking
        self.subscriptions: Dict[str, Set[str]] = {}  # symbol -> intervals
        self.last_update: Dict[Tuple[str, str], float] = {}  # (symbol, interval) -> timestamp
        self.candle_cache: Dict[Tuple[str, str], List[List[float]]] = {}  # (symbol, interval) -> candles
    
    async def start(self) -> None:
        """Start the market data manager."""
        await super().start()
        
        # Initialize with default symbols
        for symbol in self.default_symbols:
            self.add_subscription(symbol, self.default_interval)
        
        # Start update loop
        asyncio.create_task(self._update_loop())
        
        self.logger.info(f"Market data manager started with {len(self.subscriptions)} symbols")
    
    async def stop(self) -> None:
        """Stop the market data manager."""
        self.running = False
        await super().stop()
        self.logger.info("Market data manager stopped")
    
    def add_subscription(self, symbol: str, interval: str) -> None:
        """
        Add a market data subscription.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval for candles
        """
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = set()
        
        self.subscriptions[symbol].add(interval)
        self.logger.info(f"Added subscription for {symbol} at {interval} interval")
    
    def remove_subscription(self, symbol: str, interval: Optional[str] = None) -> None:
        """
        Remove a market data subscription.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval to remove, or None for all intervals
        """
        if symbol not in self.subscriptions:
            return
        
        if interval is None:
            # Remove all intervals for this symbol
            del self.subscriptions[symbol]
            self.logger.info(f"Removed all subscriptions for {symbol}")
        elif interval in self.subscriptions[symbol]:
            # Remove specific interval
            self.subscriptions[symbol].remove(interval)
            self.logger.info(f"Removed subscription for {symbol} at {interval} interval")
            
            # Clean up if no intervals left
            if not self.subscriptions[symbol]:
                del self.subscriptions[symbol]
    
    async def fetch_latest_candles(
        self, 
        symbol: str, 
        interval: str = '1h',
        limit: int = 100
    ) -> Optional[List[List[float]]]:
        """
        Fetch latest candles for a symbol and interval.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            limit: Maximum number of candles to fetch
            
        Returns:
            List of OHLCV candles or None if failed
        """
        try:
            # Emit a request for candles from the exchange
            await self.emit_event("market.ohlcv_request", {
                "exchange_id": settings.get("exchanges.default_exchange", "binance"),
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            })
            
            # Wait for a short period to allow the exchange to respond
            await asyncio.sleep(0.5)
            
            # Check if we've received the data in our cache
            cache_key = (symbol, interval)
            if cache_key in self.candle_cache:
                return self.candle_cache[cache_key]
            
            return None
        except Exception as e:
            self.logger.error(f"Error fetching candles for {symbol}@{interval}: {e}")
            return None
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Handle events from the event bus.
        
        Args:
            event_type: Event type
            data: Event data
            source: Source component
        """
        if event_type == "market.ohlcv":
            # Cache received candle data
            symbol = data.get("symbol")
            interval = data.get("interval")
            candles = data.get("candles")
            
            if symbol and interval and candles:
                cache_key = (symbol, interval)
                self.candle_cache[cache_key] = candles
                self.last_update[cache_key] = time.time()
                
                # Re-emit as market.data for strategies to consume
                await self.emit_event("market.data", {
                    "symbol": symbol,
                    "interval": interval,
                    "candles": candles
                })
    
    async def _update_loop(self) -> None:
        """Background loop to update market data."""
        while self.running:
            try:
                update_tasks = []
                
                # Check each subscription
                for symbol, intervals in self.subscriptions.items():
                    for interval in intervals:
                        cache_key = (symbol, interval)
                        current_time = time.time()
                        
                        # Determine if update is needed
                        should_update = cache_key not in self.last_update or \
                                      (current_time - self.last_update[cache_key] > 
                                       self._get_interval_seconds(interval))
                        
                        if should_update:
                            update_tasks.append(self.fetch_latest_candles(symbol, interval))
                
                # Wait for all updates to complete
                if update_tasks:
                    await asyncio.gather(*update_tasks)
                
                # Sleep between updates
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in market data update loop: {e}")
                await asyncio.sleep(10)  # Sleep longer on error
    
    @staticmethod
    def _get_interval_seconds(interval: str) -> int:
        """
        Convert interval string to seconds.
        
        Args:
            interval: Interval string like '1m', '1h', '1d'
            
        Returns:
            Number of seconds in the interval
        """
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 24 * 60 * 60
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60
        else:
            return 3600  # Default to 1 hour
