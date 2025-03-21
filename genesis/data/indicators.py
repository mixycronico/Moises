"""
Technical indicators calculation module.

This module provides functions for calculating various technical indicators
from price and volume data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for a DataFrame of OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicators
    """
    result = df.copy()
    
    # Add basic price transformations
    result["returns"] = result["close"].pct_change()
    result["log_returns"] = np.log(result["close"] / result["close"].shift(1))
    
    # Moving Averages
    result = add_moving_averages(result)
    
    # Momentum Indicators
    result = add_momentum_indicators(result)
    
    # Volatility Indicators
    result = add_volatility_indicators(result)
    
    # Volume Indicators
    result = add_volume_indicators(result)
    
    # Trend Indicators
    result = add_trend_indicators(result)
    
    # Fill NaN values (alternative: drop them)
    result.fillna(method="bfill", inplace=True)
    
    return result


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add moving average indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicators
    """
    # Simple Moving Averages
    for period in [10, 20, 50, 100, 200]:
        df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
    
    # Exponential Moving Averages
    for period in [9, 12, 26]:
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    
    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    
    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicators
    """
    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df["stoch_k"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
    
    # ROC (Rate of Change)
    df["roc"] = df["close"].pct_change(periods=10) * 100
    
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicators
    """
    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    
    # ATR (Average True Range)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(window=14).mean()
    
    # Historical Volatility
    df["volatility"] = df["log_returns"].rolling(window=20).std() * np.sqrt(252)
    
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicators
    """
    # Volume Moving Average
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    
    # Volume Oscillator
    df["volume_5_sma"] = df["volume"].rolling(window=5).mean()
    df["volume_10_sma"] = df["volume"].rolling(window=10).mean()
    df["volume_oscillator"] = ((df["volume_5_sma"] - df["volume_10_sma"]) / df["volume_10_sma"]) * 100
    
    # Chaikin Money Flow
    mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    mf_volume = mf_multiplier * df["volume"]
    df["cmf"] = mf_volume.rolling(window=20).sum() / df["volume"].rolling(window=20).sum()
    
    # On-Balance Volume (OBV)
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    
    return df


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicators
    """
    # ADX (Average Directional Index)
    if len(df) > 14:
        # True Range
        df["tr"] = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        
        # Plus and Minus Directional Movement
        df["plus_dm"] = ((df["high"] - df["high"].shift()) > (df["low"].shift() - df["low"])) & \
                        ((df["high"] - df["high"].shift()) > 0)
        df["plus_dm"] = df["plus_dm"] * (df["high"] - df["high"].shift())
        df["plus_dm"] = df["plus_dm"].fillna(0)
        
        df["minus_dm"] = ((df["low"].shift() - df["low"]) > (df["high"] - df["high"].shift())) & \
                         ((df["low"].shift() - df["low"]) > 0)
        df["minus_dm"] = df["minus_dm"] * (df["low"].shift() - df["low"])
        df["minus_dm"] = df["minus_dm"].fillna(0)
        
        # Smooth with 14-period Wilder's smoothing
        df["tr_14"] = df["tr"].rolling(window=14).mean()
        df["plus_di_14"] = 100 * (df["plus_dm"].rolling(window=14).mean() / df["tr_14"])
        df["minus_di_14"] = 100 * (df["minus_dm"].rolling(window=14).mean() / df["tr_14"])
        
        # ADX
        df["dx"] = 100 * (df["plus_di_14"] - df["minus_di_14"]).abs() / (df["plus_di_14"] + df["minus_di_14"])
        df["adx"] = df["dx"].rolling(window=14).mean()
    
    # Ichimoku Cloud (simplified)
    df["tenkan_sen"] = (df["high"].rolling(window=9).max() + df["low"].rolling(window=9).min()) / 2
    df["kijun_sen"] = (df["high"].rolling(window=26).max() + df["low"].rolling(window=26).min()) / 2
    df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(26)
    df["senkou_span_b"] = ((df["high"].rolling(window=52).max() + df["low"].rolling(window=52).min()) / 2).shift(26)
    df["chikou_span"] = df["close"].shift(-26)
    
    return df
