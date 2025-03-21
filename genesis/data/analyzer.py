"""
Market data analysis module.

This module provides components for analyzing market data
and detecting patterns and anomalies.
"""

import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from genesis.core.base import Component
from genesis.utils.logger import setup_logging
from genesis.data.indicators import calculate_indicators


class MarketAnalyzer(Component):
    """
    Analyzer for market data.
    
    This component processes raw market data, calculates indicators,
    and detects patterns and anomalies.
    """
    
    def __init__(self, name: str = "market_analyzer"):
        """
        Initialize the market analyzer.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # State tracking
        self.analyzed_data: Dict[Tuple[str, str], pd.DataFrame] = {}  # (symbol, interval) -> DataFrame
    
    async def start(self) -> None:
        """Start the market analyzer."""
        await super().start()
        self.logger.info("Market analyzer started")
    
    async def stop(self) -> None:
        """Stop the market analyzer."""
        await super().stop()
        self.logger.info("Market analyzer stopped")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Handle events from the event bus.
        
        Args:
            event_type: Event type
            data: Event data
            source: Source component
        """
        if event_type == "market.data":
            symbol = data.get("symbol")
            interval = data.get("interval", "1h")
            candles = data.get("candles")
            
            if not symbol or not candles:
                return
            
            try:
                # Process and analyze the data
                analyzed_data = await self._analyze_data(symbol, interval, candles)
                
                # Store the analyzed data
                cache_key = (symbol, interval)
                self.analyzed_data[cache_key] = analyzed_data
                
                # Emit the analyzed data
                await self.emit_event("market.analyzed", {
                    "symbol": symbol,
                    "interval": interval,
                    "data": analyzed_data.to_dict(orient="records")
                })
                
                # Check for patterns
                patterns = self._detect_patterns(analyzed_data)
                if patterns:
                    await self.emit_event("market.patterns", {
                        "symbol": symbol,
                        "interval": interval,
                        "patterns": patterns
                    })
                
                # Check for anomalies
                anomalies = self._detect_anomalies(analyzed_data)
                if anomalies:
                    await self.emit_event("market.anomalies", {
                        "symbol": symbol,
                        "interval": interval,
                        "anomalies": anomalies
                    })
            except Exception as e:
                self.logger.error(f"Error analyzing data for {symbol}@{interval}: {e}")
    
    async def _analyze_data(
        self, symbol: str, interval: str, candles: List[List[float]]
    ) -> pd.DataFrame:
        """
        Analyze market data.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            candles: OHLCV candles
            
        Returns:
            DataFrame with analyzed data
        """
        # Convert candles to DataFrame
        df = pd.DataFrame(
            candles, 
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Calculate indicators
        indicators_df = calculate_indicators(df)
        
        return indicators_df
    
    def _detect_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect patterns in the data.
        
        Args:
            data: DataFrame with market data and indicators
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Example: Detect golden cross (50 SMA crosses above 200 SMA)
        if len(data) > 200:
            try:
                current = data.iloc[-1]
                previous = data.iloc[-2]
                
                # Check for golden cross
                if previous["sma_50"] <= previous["sma_200"] and current["sma_50"] > current["sma_200"]:
                    patterns.append({
                        "type": "golden_cross",
                        "description": "50-period SMA crossed above 200-period SMA",
                        "strength": "strong",
                        "timestamp": current.name.isoformat()
                    })
                
                # Check for death cross
                elif previous["sma_50"] >= previous["sma_200"] and current["sma_50"] < current["sma_200"]:
                    patterns.append({
                        "type": "death_cross",
                        "description": "50-period SMA crossed below 200-period SMA",
                        "strength": "strong",
                        "timestamp": current.name.isoformat()
                    })
                
                # Check for bullish MACD crossover
                if "macd" in current and "macd_signal" in current:
                    if previous["macd"] <= previous["macd_signal"] and current["macd"] > current["macd_signal"]:
                        patterns.append({
                            "type": "macd_bullish_cross",
                            "description": "MACD crossed above signal line",
                            "strength": "medium",
                            "timestamp": current.name.isoformat()
                        })
                    elif previous["macd"] >= previous["macd_signal"] and current["macd"] < current["macd_signal"]:
                        patterns.append({
                            "type": "macd_bearish_cross",
                            "description": "MACD crossed below signal line",
                            "strength": "medium",
                            "timestamp": current.name.isoformat()
                        })
            except Exception as e:
                self.logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    def _detect_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the data.
        
        Args:
            data: DataFrame with market data and indicators
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        try:
            # Check for unusual volume
            if len(data) > 20:
                # Calculate average volume and standard deviation
                avg_volume = data["volume"].rolling(window=20).mean()
                std_volume = data["volume"].rolling(window=20).std()
                
                # Flag unusual volume (more than 3 standard deviations)
                if data["volume"].iloc[-1] > avg_volume.iloc[-1] + 3 * std_volume.iloc[-1]:
                    anomalies.append({
                        "type": "unusual_volume",
                        "description": "Volume spike detected",
                        "severity": "high",
                        "value": float(data["volume"].iloc[-1]),
                        "avg_value": float(avg_volume.iloc[-1]),
                        "timestamp": data.index[-1].isoformat()
                    })
            
            # Check for unusual price movement
            if len(data) > 5:
                # Calculate average price change
                price_changes = data["close"].pct_change()
                avg_change = price_changes.abs().rolling(window=20).mean()
                
                # Flag unusual price change (more than 5 times average)
                latest_change = abs(price_changes.iloc[-1])
                if not pd.isna(latest_change) and not pd.isna(avg_change.iloc[-1]):
                    if latest_change > 5 * avg_change.iloc[-1]:
                        anomalies.append({
                            "type": "unusual_price_change",
                            "description": "Large price change detected",
                            "severity": "high",
                            "value": float(latest_change),
                            "avg_value": float(avg_change.iloc[-1]),
                            "timestamp": data.index[-1].isoformat()
                        })
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
