"""
Mean reversion strategies implementation.

This module provides strategies based on mean reversion principles,
such as RSI, Bollinger Bands, and other oscillators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from genesis.strategies.base import Strategy, SignalType


class RSIStrategy(Strategy):
    """
    Relative Strength Index (RSI) strategy.
    
    This strategy generates signals based on overbought and oversold conditions
    using the RSI indicator.
    """
    
    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        name: str = "rsi"
    ):
        """
        Initialize the RSI strategy.
        
        Args:
            period: Period for the RSI calculation
            overbought: RSI level considered overbought
            oversold: RSI level considered oversold
            name: Strategy name
        """
        super().__init__(name)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Series of price data
            period: RSI period
            
        Returns:
            Series containing RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on RSI.
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV price data
            
        Returns:
            Signal details including direction and metadata
        """
        # Check if we have enough data
        if len(data) < self.period + 1:
            return {"type": SignalType.HOLD, "reason": "Not enough data"}
        
        # Calculate RSI
        rsi = self.calculate_rsi(data['close'], self.period)
        
        # Current and previous values
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # Define signal type
        signal_type = SignalType.HOLD
        reason = "RSI in neutral zone"
        
        # Check oversold -> neutral (buy)
        if prev_rsi <= self.oversold and current_rsi > self.oversold:
            signal_type = SignalType.BUY
            reason = f"RSI crossed above oversold level ({self.oversold})"
        
        # Check overbought -> neutral (sell)
        elif prev_rsi >= self.overbought and current_rsi < self.overbought:
            signal_type = SignalType.SELL
            reason = f"RSI crossed below overbought level ({self.overbought})"
        
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
            "rsi": current_rsi,
            "strength": (abs(current_rsi - 50) / 50) ** 2  # Squared distance from neutral
        }


class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands strategy.
    
    This strategy generates signals based on price movements relative to Bollinger Bands.
    """
    
    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        name: str = "bollinger_bands"
    ):
        """
        Initialize the Bollinger Bands strategy.
        
        Args:
            period: Period for the moving average
            std_dev: Number of standard deviations for the bands
            name: Strategy name
        """
        super().__init__(name)
        self.period = period
        self.std_dev = std_dev
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int, std_dev: float) -> tuple:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Series of price data
            period: Period for the moving average
            std_dev: Number of standard deviations for the bands
            
        Returns:
            Tuple of (middle band, upper band, lower band)
        """
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = prices.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return middle_band, upper_band, lower_band
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV price data
            
        Returns:
            Signal details including direction and metadata
        """
        # Check if we have enough data
        if len(data) < self.period:
            return {"type": SignalType.HOLD, "reason": "Not enough data"}
        
        # Calculate Bollinger Bands
        middle, upper, lower = self.calculate_bollinger_bands(
            data['close'], self.period, self.std_dev
        )
        
        # Current and previous values
        current_price = data['close'].iloc[-1]
        previous_price = data['close'].iloc[-2]
        
        current_lower = lower.iloc[-1]
        previous_lower = lower.iloc[-2]
        
        current_upper = upper.iloc[-1]
        previous_upper = upper.iloc[-2]
        
        # Define signal type
        signal_type = SignalType.HOLD
        reason = "Price within bands"
        
        # Price crossing below lower band (oversold, potential buy)
        if previous_price >= previous_lower and current_price < current_lower:
            signal_type = SignalType.BUY
            reason = "Price crossed below lower band"
        
        # Price crossing above upper band (overbought, potential sell)
        elif previous_price <= previous_upper and current_price > current_upper:
            signal_type = SignalType.SELL
            reason = "Price crossed above upper band"
        
        # Price crossing back above lower band (potential buy)
        elif previous_price <= previous_lower and current_price > current_lower:
            signal_type = SignalType.BUY
            reason = "Price crossed back above lower band"
        
        # Price crossing back below upper band (potential sell)
        elif previous_price >= previous_upper and current_price < current_upper:
            signal_type = SignalType.SELL
            reason = "Price crossed back below upper band"
        
        # Exit existing position?
        if signal_type in (SignalType.BUY, SignalType.SELL) and symbol in self.positions:
            current_position = self.positions[symbol]["side"]
            
            # If new signal is opposite our position, we should exit
            if (signal_type == SignalType.SELL and current_position == "long") or \
               (signal_type == SignalType.BUY and current_position == "short"):
                signal_type = SignalType.EXIT
                reason = f"Exit {current_position} position due to {reason}"
        
        # Calculate %B indicator (position within bands, 0 to 1)
        percent_b = (current_price - current_lower) / (current_upper - current_lower) if current_upper != current_lower else 0.5
        
        # Calculate bandwidth
        bandwidth = (current_upper - current_lower) / current_middle if current_middle else 0
        
        return {
            "type": signal_type,
            "reason": reason,
            "price": current_price,
            "middle": middle.iloc[-1],
            "upper": current_upper,
            "lower": current_lower,
            "percent_b": percent_b,
            "bandwidth": bandwidth,
            "strength": abs(0.5 - percent_b) * 2  # 0 at middle, 1 at bands
        }
