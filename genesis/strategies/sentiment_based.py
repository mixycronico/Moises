"""
Sentiment-based strategies implementation.

This module provides strategies that incorporate market sentiment data
from social media, news, and other sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from genesis.strategies.base import Strategy, SignalType


class SentimentStrategy(Strategy):
    """
    Sentiment-based trading strategy.
    
    This strategy generates signals based on market sentiment analysis
    and combines it with technical indicators for confirmation.
    """
    
    def __init__(
        self,
        sentiment_threshold: float = 0.3,
        sentiment_period: int = 24,  # hours
        name: str = "sentiment"
    ):
        """
        Initialize the sentiment strategy.
        
        Args:
            sentiment_threshold: Threshold for positive/negative sentiment
            sentiment_period: Period (in hours) to consider sentiment data
            name: Strategy name
        """
        super().__init__(name)
        self.sentiment_threshold = sentiment_threshold
        self.sentiment_period = sentiment_period
        self.sentiment_data = {}  # Cache for sentiment data
    
    async def fetch_sentiment(self, symbol: str) -> float:
        """
        Fetch sentiment data for a symbol.
        
        This method would typically call an external API or service.
        For now, we'll use a simple simulation based on current time.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Sentiment score from -1.0 (extremely negative) to 1.0 (extremely positive)
        """
        # Check if we have cached data that's still fresh
        if symbol in self.sentiment_data:
            timestamp, score = self.sentiment_data[symbol]
            if datetime.now() - timestamp < timedelta(hours=1):  # Cache for 1 hour
                return score
        
        # In a real implementation, this would call a sentiment API
        # For now, simulate based on symbol's first character and current minute
        seed = ord(symbol[0]) + datetime.now().minute
        sentiment = (seed % 100) / 50.0 - 1.0  # Range from -1.0 to 1.0
        
        # Cache the result
        self.sentiment_data[symbol] = (datetime.now(), sentiment)
        
        return sentiment
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on sentiment and price data.
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV price data
            
        Returns:
            Signal details including direction and metadata
        """
        # Fetch sentiment data
        sentiment = await self.fetch_sentiment(symbol)
        
        # Calculate simple moving average for confirmation
        if len(data) >= 20:
            sma = data['close'].rolling(window=20).mean().iloc[-1]
            current_price = data['close'].iloc[-1]
            price_trend = "up" if current_price > sma else "down"
        else:
            price_trend = "unknown"
        
        # Define signal type
        signal_type = SignalType.HOLD
        reason = "Neutral sentiment"
        
        # Strong positive sentiment
        if sentiment > self.sentiment_threshold:
            # Confirm with price trend
            if price_trend == "up":
                signal_type = SignalType.BUY
                reason = "Strong positive sentiment with upward price trend"
            else:
                reason = "Positive sentiment but no trend confirmation"
        
        # Strong negative sentiment
        elif sentiment < -self.sentiment_threshold:
            # Confirm with price trend
            if price_trend == "down":
                signal_type = SignalType.SELL
                reason = "Strong negative sentiment with downward price trend"
            else:
                reason = "Negative sentiment but no trend confirmation"
        
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
            "sentiment": sentiment,
            "price_trend": price_trend,
            "strength": abs(sentiment)  # Higher absolute sentiment = stronger signal
        }


class SocialVolumeStrategy(Strategy):
    """
    Social media volume strategy.
    
    This strategy generates signals based on abnormal increases in social media
    activity around a cryptocurrency or token, which may precede price movements.
    """
    
    def __init__(
        self,
        volume_threshold: float = 2.0,  # Multiple of average volume
        lookback_period: int = 24,  # hours
        name: str = "social_volume"
    ):
        """
        Initialize the social volume strategy.
        
        Args:
            volume_threshold: Threshold for significant volume increase (multiple of avg)
            lookback_period: Period (in hours) to consider for baseline
            name: Strategy name
        """
        super().__init__(name)
        self.volume_threshold = volume_threshold
        self.lookback_period = lookback_period
        self.social_data = {}  # Cache for social volume data
    
    async def fetch_social_volume(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch social media volume data for a symbol.
        
        This method would typically call an external API or service.
        For now, we'll use a simple simulation.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Dictionary with volume and sentiment data
        """
        # Check if we have cached data that's still fresh
        if symbol in self.social_data:
            timestamp, data = self.social_data[symbol]
            if datetime.now() - timestamp < timedelta(minutes=15):  # Cache for 15 min
                return data
        
        # In a real implementation, this would call a social media API
        # For now, simulate based on symbol and time
        seed = ord(symbol[0]) + datetime.now().hour
        current_volume = 100 + (seed % 200)  # Base volume
        
        # Add some randomness for spikes
        if datetime.now().minute % 10 == 0:  # 10% chance of spike
            current_volume *= 3
        
        avg_volume = 150  # Simulate average volume
        sentiment = ((seed % 10) - 5) / 5.0  # Range from -1.0 to 1.0
        
        result = {
            "current_volume": current_volume,
            "avg_volume": avg_volume,
            "volume_ratio": current_volume / avg_volume,
            "sentiment": sentiment,
            "mentions": current_volume
        }
        
        # Cache the result
        self.social_data[symbol] = (datetime.now(), result)
        
        return result
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on social media volume.
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV price data
            
        Returns:
            Signal details including direction and metadata
        """
        # Fetch social volume data
        social_data = await self.fetch_social_volume(symbol)
        
        # Extract key metrics
        volume_ratio = social_data["volume_ratio"]
        sentiment = social_data["sentiment"]
        
        # Define signal type
        signal_type = SignalType.HOLD
        reason = "Normal social volume"
        
        # Volume spike with positive sentiment
        if volume_ratio > self.volume_threshold and sentiment > 0.2:
            signal_type = SignalType.BUY
            reason = f"Social volume spike ({volume_ratio:.2f}x) with positive sentiment"
        
        # Volume spike with negative sentiment
        elif volume_ratio > self.volume_threshold and sentiment < -0.2:
            signal_type = SignalType.SELL
            reason = f"Social volume spike ({volume_ratio:.2f}x) with negative sentiment"
        
        # Volume spike with neutral sentiment
        elif volume_ratio > self.volume_threshold * 1.5:  # Higher threshold for neutral
            # Look at price momentum to decide direction
            if len(data) >= 12:
                price_change = (data['close'].iloc[-1] / data['close'].iloc[-12]) - 1
                if price_change > 0.02:  # 2% increase
                    signal_type = SignalType.BUY
                    reason = f"Social volume spike ({volume_ratio:.2f}x) with price momentum"
                elif price_change < -0.02:  # 2% decrease
                    signal_type = SignalType.SELL
                    reason = f"Social volume spike ({volume_ratio:.2f}x) with downward price momentum"
        
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
            "volume_ratio": volume_ratio,
            "sentiment": sentiment,
            "mentions": social_data["mentions"],
            "strength": (volume_ratio - 1) * abs(sentiment) if sentiment != 0 else volume_ratio - 1
        }
