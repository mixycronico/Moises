"""
Trend following strategies implementation.

This module provides strategies based on trend following principles,
such as moving averages and other trend indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from genesis.strategies.base import Strategy, SignalType


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover strategy.
    
    This strategy generates signals based on crossovers between fast
    and slow moving averages.
    """
    
    def __init__(
        self, 
        fast_period: int = 20, 
        slow_period: int = 50,
        name: str = "ma_crossover"
    ):
        """
        Initialize the strategy.
        
        Args:
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
            name: Strategy name
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on MA crossover.
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV price data
            
        Returns:
            Signal details including direction and metadata
        """
        # Check if we have enough data
        if len(data) < self.slow_period:
            return {"type": SignalType.HOLD, "reason": "Not enough data"}
        
        # Calculate moving averages
        fast_ma = data['close'].rolling(window=self.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.slow_period).mean()
        
        # Current and previous crossover status
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        
        # Check for crossover
        signal_type = SignalType.HOLD
        reason = "No crossover"
        
        # Golden cross (fast crosses above slow)
        if prev_fast <= prev_slow and current_fast > current_slow:
            signal_type = SignalType.BUY
            reason = "Golden cross"
        
        # Death cross (fast crosses below slow)
        elif prev_fast >= prev_slow and current_fast < current_slow:
            signal_type = SignalType.SELL
            reason = "Death cross"
        
        # Exit existing position?
        if signal_type in (SignalType.BUY, SignalType.SELL) and symbol in self.positions:
            current_position = self.positions[symbol]["side"]
            
            # If new signal is opposite our position, we should exit
            if (signal_type == SignalType.SELL and current_position == "long") or \
               (signal_type == SignalType.BUY and current_position == "short"):
                signal_type = SignalType.EXIT
                reason = f"Exit {current_position} position due to {reason}"
        
        return {
            "type": signal_type,
            "reason": reason,
            "fast_ma": current_fast,
            "slow_ma": current_slow,
            "strength": abs(current_fast - current_slow) / current_slow
        }


class MACDStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) strategy.
    
    Generates signals based on MACD and signal line crossovers.
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        name: str = "macd"
    ):
        """
        Initialize the MACD strategy.
        
        Args:
            fast_period: Period for the fast EMA
            slow_period: Period for the slow EMA
            signal_period: Period for the signal line
            name: Strategy name
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    @staticmethod
    def calculate_macd(
        close_prices: pd.Series,
        fast_period: int,
        slow_period: int,
        signal_period: int
    ) -> tuple:
        """
        Calculate MACD, signal, and histogram.
        
        Args:
            close_prices: Series of closing prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (MACD, signal, histogram)
        """
        # Calculate EMAs
        fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd = fast_ema - slow_ema
        
        # Calculate signal line
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd - signal
        
        return macd, signal, histogram
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on MACD.
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV price data
            
        Returns:
            Signal details including direction and metadata
        """
        # Check if we have enough data
        if len(data) < self.slow_period + self.signal_period:
            return {"type": SignalType.HOLD, "reason": "Not enough data"}
        
        # Calculate MACD
        macd, signal, histogram = self.calculate_macd(
            data['close'], 
            self.fast_period, 
            self.slow_period, 
            self.signal_period
        )
        
        # Current and previous values
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        current_hist = histogram.iloc[-1]
        
        prev_macd = macd.iloc[-2]
        prev_signal = signal.iloc[-2]
        prev_hist = histogram.iloc[-2]
        
        # Define signal type
        signal_type = SignalType.HOLD
        reason = "No signal"
        
        # Bullish crossover (MACD crosses above signal)
        if prev_macd <= prev_signal and current_macd > current_signal:
            signal_type = SignalType.BUY
            reason = "MACD crossed above signal line"
        
        # Bearish crossover (MACD crosses below signal)
        elif prev_macd >= prev_signal and current_macd < current_signal:
            signal_type = SignalType.SELL
            reason = "MACD crossed below signal line"
        
        # Zero line crossover (optional)
        elif prev_macd <= 0 and current_macd > 0:
            signal_type = SignalType.BUY
            reason = "MACD crossed above zero line"
        
        elif prev_macd >= 0 and current_macd < 0:
            signal_type = SignalType.SELL
            reason = "MACD crossed below zero line"
        
        # Histogram reversal (optional)
        elif prev_hist < 0 and current_hist > 0:
            signal_type = SignalType.BUY
            reason = "Histogram turned positive"
        
        elif prev_hist > 0 and current_hist < 0:
            signal_type = SignalType.SELL
            reason = "Histogram turned negative"
        
        # Exit existing position?
        if signal_type in (SignalType.BUY, SignalType.SELL) and symbol in self.positions:
            current_position = self.positions[symbol]["side"]
            
            # If new signal is opposite our position, we should exit
            if (signal_type == SignalType.SELL and current_position == "long") or \
               (signal_type == SignalType.BUY and current_position == "short"):
                signal_type = SignalType.EXIT
                reason = f"Exit {current_position} position due to {reason}"
        
        return {
            "type": signal_type,
            "reason": reason,
            "macd": current_macd,
            "signal": current_signal,
            "histogram": current_hist,
            "strength": abs(current_macd - current_signal) / abs(current_signal) if current_signal != 0 else 0
        }
