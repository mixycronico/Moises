"""
Generador de señales de trading.

Este módulo proporciona funcionalidades para la generación de señales
de trading basadas en indicadores técnicos.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional

from genesis.analysis.indicators import TechnicalIndicators


class SignalGenerator:
    """
    Generador de señales de trading.
    
    Esta clase analiza indicadores técnicos para generar señales
    de compra, venta o mantener posición.
    """
    
    # Constantes para tipos de señal
    SIGNAL_BUY = "buy"
    SIGNAL_SELL = "sell"
    SIGNAL_HOLD = "hold"
    SIGNAL_EXIT = "exit"
    
    # Constantes para compatibilidad con pruebas
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"  # Para mantener compatibilidad con algunos tests que usan NEUTRAL en lugar de HOLD
    EXIT = "EXIT"
    
    def __init__(self, indicators: Optional[TechnicalIndicators] = None):
        """
        Inicializar el generador de señales.
        
        Args:
            indicators: Instancia de TechnicalIndicators
        """
        self.indicators = indicators or TechnicalIndicators()
        
    def generate_ema_signal(self, prices: np.ndarray, short_period: int = 9, long_period: int = 21) -> str:
        """
        Genera señales basadas en cruce de EMAs.
        
        Args:
            prices: Serie de precios
            short_period: Período de EMA corta
            long_period: Período de EMA larga
            
        Returns:
            Señal generada ("BUY", "SELL", "HOLD")
        """
        if short_period >= long_period:
            return self.HOLD
            
        if len(prices) < long_period + 2:  # Necesitamos al menos dos puntos para detectar un cruce
            return self.HOLD
            
        # Calcular EMAs
        ema_short = self.indicators.calculate_ema(prices, short_period)
        ema_long = self.indicators.calculate_ema(prices, long_period)
        
        # Obtener las últimas dos posiciones válidas para detectar cruces
        last_idx = len(prices) - 1
        prev_idx = last_idx - 1
        
        # Verificar que tenemos valores válidos
        if np.isnan(ema_short[last_idx]) or np.isnan(ema_long[last_idx]) or \
           np.isnan(ema_short[prev_idx]) or np.isnan(ema_long[prev_idx]):
            return self.HOLD
        
        # Cruce alcista (EMA rápida cruza por encima de la lenta)
        if ema_short[prev_idx] <= ema_long[prev_idx] and ema_short[last_idx] > ema_long[last_idx]:
            return self.BUY
        
        # Cruce bajista (EMA rápida cruza por debajo de la lenta)
        elif ema_short[prev_idx] >= ema_long[prev_idx] and ema_short[last_idx] < ema_long[last_idx]:
            return self.SELL
        
        # Sin cruce
        return self.HOLD
        
    def generate_rsi_signal(self, prices: np.ndarray, period: int = 14, overbought: float = 70, oversold: float = 30) -> str:
        """
        Genera señales basadas en RSI.
        
        Args:
            prices: Serie de precios
            period: Período para RSI
            overbought: Nivel de sobrecompra
            oversold: Nivel de sobreventa
            
        Returns:
            Señal generada ("BUY", "SELL", "HOLD")
        """
        if period <= 0:
            return self.HOLD
            
        if len(prices) < period + 1:
            return self.HOLD
            
        # Calcular RSI
        rsi_values = self.indicators.calculate_rsi(prices, period)
        
        # Obtener el último valor
        last_idx = len(prices) - 1
        last_rsi = rsi_values[last_idx]
        
        # Verificar valor válido
        if np.isnan(last_rsi):
            return self.HOLD
            
        # Sobreventa (compra)
        if last_rsi <= oversold:
            return self.BUY
            
        # Sobrecompra (venta)
        elif last_rsi >= overbought:
            return self.SELL
            
        # Neutral
        return self.HOLD
        
    def generate_macd_signal(self, prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> str:
        """
        Genera señales basadas en MACD.
        
        Args:
            prices: Serie de precios
            fast_period: Período corto
            slow_period: Período largo
            signal_period: Período de señal
            
        Returns:
            Señal generada ("BUY", "SELL", "HOLD")
        """
        if fast_period >= slow_period:
            return self.HOLD
            
        if len(prices) < slow_period + signal_period + 1:
            return self.HOLD
            
        # Calcular MACD
        macd_line, signal_line, histogram = self.indicators.calculate_macd(
            prices, fast_period, slow_period, signal_period
        )
        
        # Obtener las últimas dos posiciones válidas para detectar cruces
        last_idx = len(prices) - 1
        prev_idx = last_idx - 1
        
        # Verificar valores válidos
        if np.isnan(macd_line[last_idx]) or np.isnan(signal_line[last_idx]) or \
           np.isnan(macd_line[prev_idx]) or np.isnan(signal_line[prev_idx]):
            return self.HOLD
            
        # Cruce alcista (MACD cruza por encima de la línea de señal)
        if macd_line[prev_idx] <= signal_line[prev_idx] and macd_line[last_idx] > signal_line[last_idx]:
            return self.BUY
            
        # Cruce bajista (MACD cruza por debajo de la línea de señal)
        elif macd_line[prev_idx] >= signal_line[prev_idx] and macd_line[last_idx] < signal_line[last_idx]:
            return self.SELL
            
        # Sin cruce
        return self.HOLD
        
    def generate_bollinger_bands_signal(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> str:
        """
        Genera señales basadas en Bandas de Bollinger.
        
        Args:
            prices: Serie de precios
            period: Período para la media móvil
            std_dev: Número de desviaciones estándar
            
        Returns:
            Señal generada ("BUY", "SELL", "HOLD")
        """
        if len(prices) < period:
            return self.HOLD
            
        # Calcular Bandas de Bollinger
        upper, middle, lower = self.indicators.calculate_bollinger_bands(prices, period, std_dev)
        
        # Obtener el último precio y bandas
        last_idx = len(prices) - 1
        last_price = prices[last_idx]
        
        # Verificar valores válidos
        if np.isnan(upper[last_idx]) or np.isnan(middle[last_idx]) or np.isnan(lower[last_idx]):
            return self.HOLD
            
        # Precio por debajo de la banda inferior (compra)
        if last_price <= lower[last_idx]:
            return self.BUY
            
        # Precio por encima de la banda superior (venta)
        elif last_price >= upper[last_idx]:
            return self.SELL
            
        # Precio dentro de las bandas (mantener)
        return self.HOLD
        
    def combine_signals(self, signals: List[str], method: str = "majority") -> str:
        """
        Combina múltiples señales en una sola.
        
        Args:
            signals: Lista de señales ("BUY", "SELL", "HOLD")
            method: Método de combinación ("majority", "conservative", "weighted")
            
        Returns:
            Señal combinada ("BUY", "SELL", "HOLD")
        """
        if not signals:
            return self.HOLD
            
        # Contar señales
        buy_count = signals.count(self.BUY)
        sell_count = signals.count(self.SELL)
        hold_count = signals.count(self.HOLD)
        
        # Método por mayoría
        if method == "majority":
            if buy_count > sell_count and buy_count > hold_count:
                return self.BUY
            elif sell_count > buy_count and sell_count > hold_count:
                return self.SELL
            else:
                return self.HOLD
                
        # Método conservador (sólo si todas coinciden)
        elif method == "conservative":
            if buy_count == len(signals):
                return self.BUY
            elif sell_count == len(signals):
                return self.SELL
            else:
                return self.HOLD
                
        # Método ponderado (aquí simplificado, en realidad se usarían pesos)
        elif method == "weighted":
            # En esta implementación simple, damos más peso a las señales no neutrales
            if buy_count > 0 and buy_count >= sell_count:
                return self.BUY
            elif sell_count > 0:
                return self.SELL
            else:
                return self.HOLD
                
        # Método desconocido
        return self.HOLD
    
    def ema_crossover(self, data: np.ndarray, fast_period: int = 12, slow_period: int = 26) -> Dict[str, Any]:
        """
        Generar señales basadas en cruces de EMA.
        
        Args:
            data: Serie de precios
            fast_period: Período de EMA rápida
            slow_period: Período de EMA lenta
            
        Returns:
            Diccionario con tipo de señal y metadatos
        """
        # Validar períodos
        if fast_period >= slow_period:
            return {"signal": self.SIGNAL_HOLD, "error": "fast_period debe ser menor que slow_period"}
        
        # Validar datos
        if len(data) < slow_period + 2:  # Necesitamos al menos dos puntos para detectar un cruce
            return {"signal": self.SIGNAL_HOLD, "error": "datos insuficientes"}
        
        # Calcular EMAs
        fast_ema = self.indicators.ema(data, fast_period)
        slow_ema = self.indicators.ema(data, slow_period)
        
        # Obtener las últimas dos posiciones válidas para detectar cruces
        last_idx = len(data) - 1
        prev_idx = last_idx - 1
        
        # Verificar que tenemos valores válidos
        if np.isnan(fast_ema[last_idx]) or np.isnan(slow_ema[last_idx]) or \
           np.isnan(fast_ema[prev_idx]) or np.isnan(slow_ema[prev_idx]):
            return {"signal": self.SIGNAL_HOLD, "error": "valores inválidos en EMAs"}
        
        # Cruce alcista (EMA rápida cruza por encima de la lenta)
        if fast_ema[prev_idx] <= slow_ema[prev_idx] and fast_ema[last_idx] > slow_ema[last_idx]:
            return {
                "signal": self.SIGNAL_BUY,
                "strength": (fast_ema[last_idx] - slow_ema[last_idx]) / slow_ema[last_idx] * 100,
                "fast_ema": fast_ema[last_idx],
                "slow_ema": slow_ema[last_idx]
            }
        
        # Cruce bajista (EMA rápida cruza por debajo de la lenta)
        elif fast_ema[prev_idx] >= slow_ema[prev_idx] and fast_ema[last_idx] < slow_ema[last_idx]:
            return {
                "signal": self.SIGNAL_SELL,
                "strength": (slow_ema[last_idx] - fast_ema[last_idx]) / slow_ema[last_idx] * 100,
                "fast_ema": fast_ema[last_idx],
                "slow_ema": slow_ema[last_idx]
            }
        
        # Sin cruce
        return {
            "signal": self.SIGNAL_HOLD,
            "fast_ema": fast_ema[last_idx],
            "slow_ema": slow_ema[last_idx]
        }
    
    def rsi_signal(self, data: np.ndarray, period: int = 14, overbought: float = 70, oversold: float = 30) -> Dict[str, Any]:
        """
        Generar señales basadas en RSI.
        
        Args:
            data: Serie de precios
            period: Período para el cálculo de RSI
            overbought: Nivel de sobrecompra
            oversold: Nivel de sobreventa
            
        Returns:
            Diccionario con tipo de señal y metadatos
        """
        # Validar período
        if period <= 0:
            return {"signal": self.SIGNAL_HOLD, "error": "período inválido"}
        
        # Validar datos
        if len(data) < period + 1:
            return {"signal": self.SIGNAL_HOLD, "error": "datos insuficientes"}
        
        # Calcular RSI
        rsi_values = self.indicators.rsi(data, period)
        
        # Obtener el último valor
        last_idx = len(data) - 1
        last_rsi = rsi_values[last_idx]
        
        # Verificar valor válido
        if np.isnan(last_rsi):
            return {"signal": self.SIGNAL_HOLD, "error": "valor RSI inválido"}
        
        # Sobreventa (potencial compra)
        if last_rsi <= oversold:
            return {
                "signal": self.SIGNAL_BUY,
                "strength": (oversold - last_rsi) / oversold * 100,
                "rsi": last_rsi
            }
        
        # Sobrecompra (potencial venta)
        elif last_rsi >= overbought:
            return {
                "signal": self.SIGNAL_SELL,
                "strength": (last_rsi - overbought) / (100 - overbought) * 100,
                "rsi": last_rsi
            }
        
        # Neutral
        return {
            "signal": self.SIGNAL_HOLD,
            "rsi": last_rsi
        }
    
    def macd_signal(self, data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, Any]:
        """
        Generar señales basadas en MACD.
        
        Args:
            data: Serie de precios
            fast_period: Período corto para EMA
            slow_period: Período largo para EMA
            signal_period: Período para la línea de señal
            
        Returns:
            Diccionario con tipo de señal y metadatos
        """
        # Validar períodos
        if fast_period >= slow_period:
            return {"signal": self.SIGNAL_HOLD, "error": "fast_period debe ser menor que slow_period"}
        
        # Validar datos
        if len(data) < slow_period + signal_period + 1:
            return {"signal": self.SIGNAL_HOLD, "error": "datos insuficientes"}
        
        # Calcular MACD
        macd_line, signal_line, histogram = self.indicators.macd(data, fast_period, slow_period, signal_period)
        
        # Obtener las últimas dos posiciones válidas para detectar cruces
        last_idx = len(data) - 1
        prev_idx = last_idx - 1
        
        # Verificar valores válidos
        if np.isnan(macd_line[last_idx]) or np.isnan(signal_line[last_idx]) or \
           np.isnan(macd_line[prev_idx]) or np.isnan(signal_line[prev_idx]):
            return {"signal": self.SIGNAL_HOLD, "error": "valores MACD inválidos"}
        
        # Cruce alcista (MACD cruza por encima de la línea de señal)
        if macd_line[prev_idx] <= signal_line[prev_idx] and macd_line[last_idx] > signal_line[last_idx]:
            return {
                "signal": self.SIGNAL_BUY,
                "strength": abs(histogram[last_idx]) * 100,
                "macd": macd_line[last_idx],
                "signal": signal_line[last_idx],
                "histogram": histogram[last_idx]
            }
        
        # Cruce bajista (MACD cruza por debajo de la línea de señal)
        elif macd_line[prev_idx] >= signal_line[prev_idx] and macd_line[last_idx] < signal_line[last_idx]:
            return {
                "signal": self.SIGNAL_SELL,
                "strength": abs(histogram[last_idx]) * 100,
                "macd": macd_line[last_idx],
                "signal": signal_line[last_idx],
                "histogram": histogram[last_idx]
            }
        
        # Sin cruce
        return {
            "signal": self.SIGNAL_HOLD,
            "macd": macd_line[last_idx],
            "signal": signal_line[last_idx],
            "histogram": histogram[last_idx]
        }
    
    def bollinger_bands_signal(self, data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Dict[str, Any]:
        """
        Generar señales basadas en Bandas de Bollinger.
        
        Args:
            data: Serie de precios
            period: Período para la media móvil
            std_dev: Número de desviaciones estándar
            
        Returns:
            Diccionario con tipo de señal y metadatos
        """
        # Validar datos
        if len(data) < period:
            return {"signal": self.SIGNAL_HOLD, "error": "datos insuficientes"}
        
        # Calcular Bandas de Bollinger
        upper, middle, lower = self.indicators.bollinger_bands(data, period, std_dev)
        
        # Obtener el último valor
        last_idx = len(data) - 1
        last_price = data[last_idx]
        
        # Verificar valores válidos
        if np.isnan(upper[last_idx]) or np.isnan(middle[last_idx]) or np.isnan(lower[last_idx]):
            return {"signal": self.SIGNAL_HOLD, "error": "valores de bandas inválidos"}
        
        # Cercanía a la banda inferior (potencial compra)
        if last_price <= lower[last_idx]:
            # Calcular qué tan cerca está del límite inferior
            distance = (lower[last_idx] - last_price) / (upper[last_idx] - lower[last_idx]) * 100
            return {
                "signal": self.SIGNAL_BUY,
                "strength": min(distance, 100),  # Limitar a 100%
                "price": last_price,
                "upper": upper[last_idx],
                "middle": middle[last_idx],
                "lower": lower[last_idx]
            }
        
        # Cercanía a la banda superior (potencial venta)
        elif last_price >= upper[last_idx]:
            # Calcular qué tan cerca está del límite superior
            distance = (last_price - upper[last_idx]) / (upper[last_idx] - lower[last_idx]) * 100
            return {
                "signal": self.SIGNAL_SELL,
                "strength": min(distance, 100),  # Limitar a 100%
                "price": last_price,
                "upper": upper[last_idx],
                "middle": middle[last_idx],
                "lower": lower[last_idx]
            }
        
        # Dentro de las bandas (neutral)
        return {
            "signal": self.SIGNAL_HOLD,
            "price": last_price,
            "upper": upper[last_idx],
            "middle": middle[last_idx],
            "lower": lower[last_idx]
        }
    
    def combine_signals(self, signals: List[Dict[str, Any]], method: str = "majority") -> Dict[str, Any]:
        """
        Combinar múltiples señales para obtener una señal final.
        
        Args:
            signals: Lista de señales
            method: Método de combinación ('majority', 'conservative', 'weighted')
            
        Returns:
            Diccionario con tipo de señal y metadatos
        """
        if not signals:
            return {"signal": self.SIGNAL_HOLD, "error": "sin señales"}
        
        # Contar señales por tipo
        signal_counts = {
            self.SIGNAL_BUY: 0,
            self.SIGNAL_SELL: 0,
            self.SIGNAL_HOLD: 0,
            self.SIGNAL_EXIT: 0
        }
        
        for s in signals:
            signal_type = s.get("signal", self.SIGNAL_HOLD)
            if signal_type in signal_counts:
                signal_counts[signal_type] += 1
        
        # Método por mayoría (el tipo de señal más frecuente)
        if method == "majority":
            majority_signal = max(signal_counts, key=signal_counts.get)
            count = signal_counts[majority_signal]
            total = sum(signal_counts.values())
            
            return {
                "signal": majority_signal,
                "confidence": (count / total) * 100 if total > 0 else 0,
                "buy_count": signal_counts[self.SIGNAL_BUY],
                "sell_count": signal_counts[self.SIGNAL_SELL],
                "hold_count": signal_counts[self.SIGNAL_HOLD],
                "exit_count": signal_counts[self.SIGNAL_EXIT],
                "method": "majority"
            }
        
        # Método conservador (señal solo si todas coinciden)
        elif method == "conservative":
            total = sum(signal_counts.values())
            
            if signal_counts[self.SIGNAL_BUY] == total:
                return {"signal": self.SIGNAL_BUY, "confidence": 100, "method": "conservative"}
            elif signal_counts[self.SIGNAL_SELL] == total:
                return {"signal": self.SIGNAL_SELL, "confidence": 100, "method": "conservative"}
            elif signal_counts[self.SIGNAL_EXIT] == total:
                return {"signal": self.SIGNAL_EXIT, "confidence": 100, "method": "conservative"}
            else:
                return {"signal": self.SIGNAL_HOLD, "confidence": 0, "method": "conservative"}
        
        # Método ponderado (considera la fuerza de las señales)
        elif method == "weighted":
            weighted_signals = {
                self.SIGNAL_BUY: 0,
                self.SIGNAL_SELL: 0,
                self.SIGNAL_HOLD: 0,
                self.SIGNAL_EXIT: 0
            }
            
            for s in signals:
                signal_type = s.get("signal", self.SIGNAL_HOLD)
                strength = s.get("strength", 1)
                
                if signal_type in weighted_signals:
                    weighted_signals[signal_type] += strength
            
            strongest_signal = max(weighted_signals, key=weighted_signals.get)
            total_strength = sum(weighted_signals.values())
            
            return {
                "signal": strongest_signal,
                "confidence": (weighted_signals[strongest_signal] / total_strength) * 100 if total_strength > 0 else 0,
                "buy_strength": weighted_signals[self.SIGNAL_BUY],
                "sell_strength": weighted_signals[self.SIGNAL_SELL],
                "hold_strength": weighted_signals[self.SIGNAL_HOLD],
                "exit_strength": weighted_signals[self.SIGNAL_EXIT],
                "method": "weighted"
            }
        
        # Método no válido
        return {"signal": self.SIGNAL_HOLD, "error": f"método de combinación no válido: {method}"}