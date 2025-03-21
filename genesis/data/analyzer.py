"""
Analizador de datos del mercado para el sistema Genesis.

Este módulo proporciona herramientas avanzadas para el análisis
de datos del mercado, detección de patrones y generación de
señales basadas en indicadores técnicos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

from genesis.data.indicators import calculate_indicators


class MarketAnalyzer:
    """
    Analizador de datos del mercado.
    
    Esta clase proporciona métodos para analizar datos del mercado,
    detectar patrones, y generar señales basadas en indicadores técnicos.
    """
    
    def __init__(self):
        """Inicializar el analizador de mercado."""
        pass
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara los datos del mercado para el análisis.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame preparado con indicadores
        """
        if df.empty:
            return df
            
        # Asegurar que tenemos las columnas necesarias
        required_columns = ["open", "high", "low", "close", "volume"]
        
        # Convertir nombres de columnas a minúsculas
        df.columns = [col.lower() for col in df.columns]
        
        # Verificar columnas
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en los datos")
                
        # Calcular indicadores técnicos
        result = calculate_indicators(df)
        
        return result
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Detecta patrones de velas en los datos.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con patrones detectados y sus posiciones
        """
        if df.empty:
            return {}
            
        # Asegurarse de que tenemos las columnas necesarias
        required_columns = ["open", "high", "low", "close"]
        df_lower = df.copy()
        df_lower.columns = [col.lower() for col in df_lower.columns]
        
        for col in required_columns:
            if col not in df_lower.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en los datos")
                
        patterns = {}
        
        # Doji
        doji = self._detect_doji(df_lower)
        if doji:
            patterns["doji"] = doji
            
        # Hammer
        hammer = self._detect_hammer(df_lower)
        if hammer:
            patterns["hammer"] = hammer
            
        # Shooting Star
        shooting_star = self._detect_shooting_star(df_lower)
        if shooting_star:
            patterns["shooting_star"] = shooting_star
            
        # Engulfing
        bullish_engulfing, bearish_engulfing = self._detect_engulfing(df_lower)
        if bullish_engulfing:
            patterns["bullish_engulfing"] = bullish_engulfing
        if bearish_engulfing:
            patterns["bearish_engulfing"] = bearish_engulfing
            
        return patterns
    
    def _detect_doji(self, df: pd.DataFrame) -> List[int]:
        """
        Detecta patrones Doji en los datos.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Lista de índices donde se detectaron Dojis
        """
        # Calcular la diferencia entre apertura y cierre
        df["body"] = abs(df["close"] - df["open"])
        
        # Calcular el rango de la vela
        df["range"] = df["high"] - df["low"]
        
        # Definir un umbral para el cuerpo pequeño (típicamente 5% del rango)
        threshold = 0.05
        
        # Detectar Dojis (cuerpo muy pequeño respecto al rango)
        df["is_doji"] = (df["body"] / df["range"]) < threshold
        
        # Devolver los índices donde se detectaron Dojis
        return df.index[df["is_doji"]].tolist()
    
    def _detect_hammer(self, df: pd.DataFrame) -> List[int]:
        """
        Detecta patrones Hammer en los datos.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Lista de índices donde se detectaron Hammers
        """
        # Calcular el cuerpo y las sombras
        df["body"] = abs(df["close"] - df["open"])
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        
        # Calcular el rango total
        df["range"] = df["high"] - df["low"]
        
        # Condiciones para un Hammer:
        # 1. Sombra inferior larga (al menos 2 veces el cuerpo)
        # 2. Cuerpo pequeño en la parte superior
        # 3. Sombra superior muy pequeña o inexistente
        df["is_hammer"] = (
            (df["lower_shadow"] > 2 * df["body"]) &
            (df["upper_shadow"] < 0.1 * df["range"]) &
            (df["body"] < 0.3 * df["range"])
        )
        
        # Devolver los índices donde se detectaron Hammers
        return df.index[df["is_hammer"]].tolist()
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> List[int]:
        """
        Detecta patrones Shooting Star en los datos.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Lista de índices donde se detectaron Shooting Stars
        """
        # Calcular el cuerpo y las sombras
        df["body"] = abs(df["close"] - df["open"])
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        
        # Calcular el rango total
        df["range"] = df["high"] - df["low"]
        
        # Condiciones para un Shooting Star:
        # 1. Sombra superior larga (al menos 2 veces el cuerpo)
        # 2. Cuerpo pequeño en la parte inferior
        # 3. Sombra inferior muy pequeña o inexistente
        df["is_shooting_star"] = (
            (df["upper_shadow"] > 2 * df["body"]) &
            (df["lower_shadow"] < 0.1 * df["range"]) &
            (df["body"] < 0.3 * df["range"])
        )
        
        # Devolver los índices donde se detectaron Shooting Stars
        return df.index[df["is_shooting_star"]].tolist()
    
    def _detect_engulfing(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        Detecta patrones Engulfing en los datos.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Tupla de (índices de patrones bullish, índices de patrones bearish)
        """
        # Identificar velas alcistas y bajistas
        df["is_bullish"] = df["close"] > df["open"]
        df["is_bearish"] = df["close"] < df["open"]
        
        # Detectar patrones engulfing
        bullish_engulfing = []
        bearish_engulfing = []
        
        for i in range(1, len(df)):
            # Patrón alcista: vela actual alcista envuelve una vela anterior bajista
            if (df["is_bullish"].iloc[i] and 
                df["is_bearish"].iloc[i-1] and 
                df["open"].iloc[i] < df["close"].iloc[i-1] and 
                df["close"].iloc[i] > df["open"].iloc[i-1]):
                bullish_engulfing.append(i)
                
            # Patrón bajista: vela actual bajista envuelve una vela anterior alcista
            if (df["is_bearish"].iloc[i] and 
                df["is_bullish"].iloc[i-1] and 
                df["open"].iloc[i] > df["close"].iloc[i-1] and 
                df["close"].iloc[i] < df["open"].iloc[i-1]):
                bearish_engulfing.append(i)
                
        return bullish_engulfing, bearish_engulfing
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales de trading basadas en indicadores técnicos.
        
        Args:
            df: DataFrame con datos OHLCV e indicadores
            
        Returns:
            DataFrame con señales generadas
        """
        if df.empty:
            return df
            
        # Crear una copia para trabajar
        result = df.copy()
        
        # Inicializar columna de señales
        result["signal"] = 0  # 0: sin señal, 1: compra, -1: venta
        
        # Señales basadas en cruce de medias móviles
        result.loc[(result["sma_20"] > result["sma_50"]) & 
                  (result["sma_20"].shift(1) <= result["sma_50"].shift(1)), "signal"] = 1
                  
        result.loc[(result["sma_20"] < result["sma_50"]) & 
                  (result["sma_20"].shift(1) >= result["sma_50"].shift(1)), "signal"] = -1
        
        # Señales basadas en RSI
        result.loc[(result["rsi"] < 30) & (result["rsi"].shift(1) >= 30), "signal"] = 1
        result.loc[(result["rsi"] > 70) & (result["rsi"].shift(1) <= 70), "signal"] = -1
        
        # Señales basadas en MACD
        result.loc[(result["macd"] > result["macd_signal"]) & 
                  (result["macd"].shift(1) <= result["macd_signal"].shift(1)), "signal"] = 1
                  
        result.loc[(result["macd"] < result["macd_signal"]) & 
                  (result["macd"].shift(1) >= result["macd_signal"].shift(1)), "signal"] = -1
        
        # Señales basadas en Bollinger Bands
        result.loc[result["close"] < result["bb_lower"], "signal"] = 1
        result.loc[result["close"] > result["bb_upper"], "signal"] = -1
        
        return result
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula puntos de pivote (tradicionales).
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con puntos de pivote calculados
        """
        if df.empty:
            return df
            
        # Crear una copia para trabajar
        result = df.copy()
        
        # Asegurar que tenemos las columnas necesarias
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in result.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en los datos")
        
        # Calcular puntos de pivote para cada fila usando datos del día anterior
        for i in range(1, len(result)):
            prev_high = result["high"].iloc[i-1]
            prev_low = result["low"].iloc[i-1]
            prev_close = result["close"].iloc[i-1]
            
            # Punto de pivote central
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Niveles de soporte
            s1 = (2 * pivot) - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
            
            # Niveles de resistencia
            r1 = (2 * pivot) - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)
            
            # Almacenar los resultados
            result.loc[result.index[i], "pivot"] = pivot
            result.loc[result.index[i], "s1"] = s1
            result.loc[result.index[i], "s2"] = s2
            result.loc[result.index[i], "s3"] = s3
            result.loc[result.index[i], "r1"] = r1
            result.loc[result.index[i], "r2"] = r2
            result.loc[result.index[i], "r3"] = r3
        
        return result
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calcula niveles de retroceso de Fibonacci.
        
        Args:
            df: DataFrame con datos OHLCV
            period: Período para encontrar máximos y mínimos
            
        Returns:
            DataFrame con niveles Fibonacci calculados
        """
        if df.empty or len(df) < period:
            return df
            
        # Crear una copia para trabajar
        result = df.copy()
        
        # Niveles de Fibonacci comunes
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        # Calcular máximos y mínimos recientes
        for i in range(period, len(result)):
            segment = result.iloc[i-period:i]
            high = segment["high"].max()
            low = segment["low"].min()
            
            # Calcular niveles de retroceso (para tendencia alcista)
            if result["close"].iloc[i-1] > result["close"].iloc[i-period]:
                # Tendencia alcista, retroceso desde máximo a mínimo
                diff = high - low
                for level in fib_levels:
                    level_name = f"fib_{int(level * 1000)}"
                    result.loc[result.index[i], level_name] = high - (diff * level)
            else:
                # Tendencia bajista, retroceso desde mínimo a máximo
                diff = high - low
                for level in fib_levels:
                    level_name = f"fib_{int(level * 1000)}"
                    result.loc[result.index[i], level_name] = low + (diff * level)
        
        return result
    
    def analyze_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza la estructura del mercado identificando niveles clave.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con análisis de la estructura del mercado
        """
        if df.empty:
            return {}
            
        # Preparar datos
        result = df.copy()
        
        # Detectar máximos y mínimos locales
        result["is_local_max"] = (result["high"] > result["high"].shift(1)) & (result["high"] > result["high"].shift(-1))
        result["is_local_min"] = (result["low"] < result["low"].shift(1)) & (result["low"] < result["low"].shift(-1))
        
        # Identificar niveles de soporte y resistencia
        local_maxima = result[result["is_local_max"]]["high"].tolist()
        local_minima = result[result["is_local_min"]]["low"].tolist()
        
        # Tendencia actual
        current_price = result["close"].iloc[-1]
        sma20 = result["sma_20"].iloc[-1] if "sma_20" in result.columns else None
        sma50 = result["sma_50"].iloc[-1] if "sma_50" in result.columns else None
        sma200 = result["sma_200"].iloc[-1] if "sma_200" in result.columns else None
        
        # Determinar la tendencia
        trend = "neutral"
        if sma20 and sma50 and sma200:
            if sma20 > sma50 > sma200:
                trend = "bullish"
            elif sma20 < sma50 < sma200:
                trend = "bearish"
        
        # Calcular volatilidad reciente
        volatility = result["close"].pct_change().std() * np.sqrt(252)  # Anualizada
        
        # Analizar volumen
        avg_volume = result["volume"].mean()
        latest_volume = result["volume"].iloc[-1]
        volume_trend = "increasing" if latest_volume > avg_volume else "decreasing"
        
        # Soporte y resistencia cercanos
        resistance_levels = [level for level in local_maxima if level > current_price]
        support_levels = [level for level in local_minima if level < current_price]
        
        nearest_resistance = min(resistance_levels) if resistance_levels else None
        nearest_support = max(support_levels) if support_levels else None
        
        return {
            "trend": trend,
            "current_price": current_price,
            "volatility": volatility,
            "volume_trend": volume_trend,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "support_levels": sorted(support_levels),
            "resistance_levels": sorted(resistance_levels),
            "local_maxima": local_maxima,
            "local_minima": local_minima
        }
    
    def detect_divergences(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Detecta divergencias entre precio e indicadores.
        
        Args:
            df: DataFrame con datos OHLCV e indicadores
            
        Returns:
            Diccionario con divergencias detectadas y sus posiciones
        """
        if df.empty or "rsi" not in df.columns:
            return {}
            
        # Detectar máximos y mínimos locales en precio
        df["price_high"] = (df["close"] > df["close"].shift(1)) & (df["close"] > df["close"].shift(-1))
        df["price_low"] = (df["close"] < df["close"].shift(1)) & (df["close"] < df["close"].shift(-1))
        
        # Detectar máximos y mínimos locales en RSI
        df["rsi_high"] = (df["rsi"] > df["rsi"].shift(1)) & (df["rsi"] > df["rsi"].shift(-1))
        df["rsi_low"] = (df["rsi"] < df["rsi"].shift(1)) & (df["rsi"] < df["rsi"].shift(-1))
        
        # Divergencia alcista: precio hace mínimos más bajos pero RSI hace mínimos más altos
        bullish_divergence = []
        for i in range(2, len(df) - 2):
            # Buscar mínimos en precio
            if df["price_low"].iloc[i]:
                prev_lows = df[df["price_low"]].iloc[:i]
                if not prev_lows.empty:
                    prev_low_idx = prev_lows.index[-1]
                    
                    # Verificar si hay divergencia
                    if (df["close"].loc[prev_low_idx] > df["close"].iloc[i] and
                        df["rsi"].loc[prev_low_idx] < df["rsi"].iloc[i]):
                        bullish_divergence.append(i)
        
        # Divergencia bajista: precio hace máximos más altos pero RSI hace máximos más bajos
        bearish_divergence = []
        for i in range(2, len(df) - 2):
            # Buscar máximos en precio
            if df["price_high"].iloc[i]:
                prev_highs = df[df["price_high"]].iloc[:i]
                if not prev_highs.empty:
                    prev_high_idx = prev_highs.index[-1]
                    
                    # Verificar si hay divergencia
                    if (df["close"].loc[prev_high_idx] < df["close"].iloc[i] and
                        df["rsi"].loc[prev_high_idx] > df["rsi"].iloc[i]):
                        bearish_divergence.append(i)
                        
        return {
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence
        }