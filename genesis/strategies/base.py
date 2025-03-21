"""
Base classes for trading strategies.

This module defines the interface for all trading strategies in the Genesis system.
"""

import abc
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from genesis.core.base import Component
from genesis.config.settings import settings


class Strategy(Component):
    """
    Base abstract class for all trading strategies.
    
    A strategy consumes market data and produces trading signals.
    """
    
    def __init__(self, name: str):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
        """
        super().__init__(name)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.running = False
        
    async def start(self) -> None:
        """Start the strategy."""
        self.logger.info(f"Starting strategy: {self.name}")
        self.running = True
        
    async def stop(self) -> None:
        """Stop the strategy."""
        self.logger.info(f"Stopping strategy: {self.name}")
        self.running = False
    
    @abc.abstractmethod
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a trading signal from market data.
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV or other market data
            
        Returns:
            Signal information including direction and metadata
        """
        pass
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Handle events from the event bus.
        
        Args:
            event_type: Event type
            data: Event data
            source: Source component of the event
        """
        if event_type == "market.data":
            symbol = data.get("symbol")
            candles = data.get("candles")
            
            if not symbol or not candles:
                return
            
            # Convert to dataframe
            df = pd.DataFrame(
                candles, 
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Generate signal
            try:
                signal = await self.generate_signal(symbol, df)
                
                if signal:
                    await self.emit_event("strategy.signal", {
                        "symbol": symbol,
                        "strategy": self.name,
                        "signal": signal
                    })
            except Exception as e:
                await self.emit_event("strategy.error", {
                    "symbol": symbol,
                    "strategy": self.name,
                    "error": str(e)
                })
        
        elif event_type == "trade.opened":
            # Update position tracking
            symbol = data.get("symbol")
            position = data.get("position")
            
            if symbol and position:
                self.positions[symbol] = position
        
        elif event_type == "trade.closed":
            # Remove from position tracking
            symbol = data.get("symbol")
            
            if symbol and symbol in self.positions:
                del self.positions[symbol]


class SignalType:
    """Define common signal types."""
    
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"
    CLOSE = "close"
