"""
Indicadores técnicos para análisis de mercado.

Este módulo proporciona funcionalidades para el cálculo de indicadores
técnicos utilizados en el análisis de mercados financieros.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional


class TechnicalIndicators:
    """
    Clase para el cálculo de indicadores técnicos.
    
    Esta clase implementa varios indicadores técnicos utilizados
    en el análisis de mercados financieros.
    """
    
    def __init__(self):
        """Inicializar la clase de indicadores técnicos."""
        pass
        
    def calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calcular la Media Móvil Simple (Simple Moving Average - SMA).
        
        Args:
            data: Serie de precios
            period: Período para el cálculo
            
        Returns:
            Array con los valores de SMA
        """
        return self.sma(data, period)
        
    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calcular la Media Móvil Exponencial (Exponential Moving Average - EMA).
        
        Args:
            data: Serie de precios
            period: Período para el cálculo
            
        Returns:
            Array con los valores de EMA
        """
        return self.ema(data, period)
        
    def calculate_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calcular el Índice de Fuerza Relativa (Relative Strength Index - RSI).
        
        Args:
            data: Serie de precios
            period: Período para el cálculo
            
        Returns:
            Array con los valores de RSI
        """
        return self.rsi(data, period)
        
    def calculate_macd(self, data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcular MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Serie de precios
            fast_period: Período corto para EMA
            slow_period: Período largo para EMA
            signal_period: Período para la línea de señal
            
        Returns:
            Tupla con (línea MACD, línea de señal, histograma)
        """
        return self.macd(data, fast_period, slow_period, signal_period)
        
    def calculate_bollinger_bands(self, data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcular Bandas de Bollinger.
        
        Args:
            data: Serie de precios
            period: Período para la media móvil
            std_dev: Número de desviaciones estándar
            
        Returns:
            Tupla con (banda superior, media, banda inferior)
        """
        return self.bollinger_bands(data, period, std_dev)
        
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calcular el Rango Medio Verdadero (Average True Range - ATR).
        
        Args:
            high: Serie de precios altos
            low: Serie de precios bajos
            close: Serie de precios de cierre
            period: Período para el cálculo
            
        Returns:
            Array con los valores de ATR
        """
        return self.atr(high, low, close, period)
    
    def sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calcular la Media Móvil Simple (Simple Moving Average - SMA).
        
        Args:
            data: Serie de precios
            period: Período para el cálculo
            
        Returns:
            Array con los valores de SMA
        """
        if len(data) < period:
            return np.array([np.nan] * len(data))
        
        result = np.zeros_like(data) * np.nan
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
            
        return result
    
    def ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calcular la Media Móvil Exponencial (Exponential Moving Average - EMA).
        
        Args:
            data: Serie de precios
            period: Período para el cálculo
            
        Returns:
            Array con los valores de EMA
        """
        if len(data) < period:
            return np.array([np.nan] * len(data))
        
        result = np.zeros_like(data) * np.nan
        # Factor de suavizado
        alpha = 2 / (period + 1)
        
        # Inicializar con SMA
        result[period - 1] = np.mean(data[:period])
        
        # Calcular EMA recursivamente
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            
        return result
    
    def rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calcular el Índice de Fuerza Relativa (Relative Strength Index - RSI).
        
        Args:
            data: Serie de precios
            period: Período para el cálculo
            
        Returns:
            Array con los valores de RSI
        """
        if len(data) < period + 1:
            return np.array([np.nan] * len(data))
        
        # Calcular diferencias diarias
        delta = np.zeros_like(data)
        delta[1:] = data[1:] - data[:-1]
        
        # Separar ganancias y pérdidas
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Inicializar arrays de resultado
        avg_gain = np.zeros_like(data) * np.nan
        avg_loss = np.zeros_like(data) * np.nan
        rs = np.zeros_like(data) * np.nan
        rsi = np.zeros_like(data) * np.nan
        
        # Primera media de ganancias y pérdidas
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        # Calcular promedios móviles suavizados
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
            
        # Calcular RS y RSI
        # Cuando no hay pérdidas (todos los precios suben), RSI debe ser 100
        rs = np.zeros_like(avg_gain)
        for i in range(period, len(data)):
            if avg_loss[i] == 0:
                rs[i] = 100.0  # Si no hay pérdidas, RS es infinito (representamos como un valor alto)
            else:
                rs[i] = avg_gain[i] / avg_loss[i]
        
        # Calcular RSI usando la fórmula RSI = 100 - (100 / (1 + RS))
        for i in range(period, len(data)):
            if avg_loss[i] == 0:
                rsi[i] = 100.0  # No hay pérdidas, RSI = 100 (sobrecompra extrema)
            else:
                rsi[i] = 100.0 - (100.0 / (1.0 + rs[i]))
        
        return rsi
    
    def macd(self, data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcular MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Serie de precios
            fast_period: Período corto para EMA
            slow_period: Período largo para EMA
            signal_period: Período para la línea de señal
            
        Returns:
            Tupla con (línea MACD, línea de señal, histograma)
        """
        fast_ema = self.ema(data, fast_period)
        slow_ema = self.ema(data, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = self.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def bollinger_bands(self, data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcular Bandas de Bollinger.
        
        Args:
            data: Serie de precios
            period: Período para la media móvil
            std_dev: Número de desviaciones estándar
            
        Returns:
            Tupla con (banda superior, media, banda inferior)
        """
        if len(data) < period:
            nan_arr = np.array([np.nan] * len(data))
            return nan_arr, nan_arr, nan_arr
        
        # Calcular la media móvil
        middle_band = self.sma(data, period)
        
        # Calcular la desviación estándar
        std = np.zeros_like(data) * np.nan
        for i in range(period - 1, len(data)):
            std[i] = np.std(data[i - period + 1:i + 1], ddof=1)
        
        # Calcular bandas
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    def atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calcular el Rango Medio Verdadero (Average True Range - ATR).
        
        Args:
            high: Serie de precios altos
            low: Serie de precios bajos
            close: Serie de precios de cierre
            period: Período para el cálculo
            
        Returns:
            Array con los valores de ATR
        """
        if len(high) < 2:
            return np.array([np.nan] * len(high))
        
        true_range = np.zeros_like(high) * np.nan
        
        # Primer valor
        true_range[0] = high[0] - low[0]
        
        # Resto de valores
        for i in range(1, len(high)):
            hl = high[i] - low[i]  # Alto - Bajo actual
            hpc = abs(high[i] - close[i-1])  # Alto actual - Cierre previo
            lpc = abs(low[i] - close[i-1])  # Bajo actual - Cierre previo
            true_range[i] = max(hl, hpc, lpc)
        
        # Calcular ATR
        atr = np.zeros_like(high) * np.nan
        
        # Inicializar con la media simple de los primeros valores
        if len(true_range) >= period:
            atr[period-1] = np.mean(true_range[:period])
            
            # Calcular ATR con media móvil suavizada
            for i in range(period, len(high)):
                atr[i] = ((period - 1) * atr[i-1] + true_range[i]) / period
        
        return atr
    
    def adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcular el Índice Direccional Medio (Average Directional Index - ADX).
        
        Args:
            high: Serie de precios altos
            low: Serie de precios bajos
            close: Serie de precios de cierre
            period: Período para el cálculo
            
        Returns:
            Tupla con (ADX, +DI, -DI)
        """
        if len(high) < period + 1:
            nan_arr = np.array([np.nan] * len(high))
            return nan_arr, nan_arr, nan_arr
        
        # Calcular True Range
        atr_values = self.atr(high, low, close, period)
        
        # Calcular movimientos direccionales
        plus_dm = np.zeros_like(high) * np.nan
        minus_dm = np.zeros_like(high) * np.nan
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Suavizar DM
        plus_di = np.zeros_like(high) * np.nan
        minus_di = np.zeros_like(high) * np.nan
        
        # Inicializar con la media simple de los primeros valores
        if len(plus_dm) >= period:
            plus_di[period] = 100 * np.sum(plus_dm[1:period+1]) / np.sum(atr_values[:period+1])
            minus_di[period] = 100 * np.sum(minus_dm[1:period+1]) / np.sum(atr_values[:period+1])
            
            # Calcular DI suavizado
            for i in range(period + 1, len(high)):
                plus_di[i] = 100 * ((period - 1) * plus_di[i-1] + plus_dm[i]) / ((period - 1) * atr_values[i-1] + atr_values[i])
                minus_di[i] = 100 * ((period - 1) * minus_di[i-1] + minus_dm[i]) / ((period - 1) * atr_values[i-1] + atr_values[i])
        
        # Calcular ADX
        adx_values = np.zeros_like(high) * np.nan
        dx = np.zeros_like(high) * np.nan
        
        for i in range(period, len(high)):
            if not (np.isnan(plus_di[i]) or np.isnan(minus_di[i])):
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) > 0 else 0
        
        # Suavizar DX para obtener ADX
        if len(dx) >= 2 * period:
            adx_values[2 * period - 1] = np.mean(dx[period:(2 * period)])
            
            for i in range(2 * period, len(high)):
                adx_values[i] = ((period - 1) * adx_values[i-1] + dx[i]) / period
        
        return adx_values, plus_di, minus_di
    
    def stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcular el Oscilador Estocástico.
        
        Args:
            high: Serie de precios altos
            low: Serie de precios bajos
            close: Serie de precios de cierre
            k_period: Período para %K
            d_period: Período para %D
            
        Returns:
            Tupla con (%K, %D)
        """
        if len(high) < k_period:
            nan_arr = np.array([np.nan] * len(high))
            return nan_arr, nan_arr
        
        # Calcular %K
        k = np.zeros_like(close) * np.nan
        
        for i in range(k_period - 1, len(close)):
            high_val = np.max(high[i - k_period + 1:i + 1])
            low_val = np.min(low[i - k_period + 1:i + 1])
            
            if high_val - low_val > 0:
                k[i] = 100 * ((close[i] - low_val) / (high_val - low_val))
            else:
                k[i] = 50  # Valor por defecto si el rango es 0
        
        # Calcular %D (media móvil simple de %K)
        d = self.sma(k, d_period)
        
        return k, d