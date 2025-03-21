"""
Tests intermedios para TechnicalIndicators.

Este módulo contiene tests unitarios para casos intermedios de cálculo
de indicadores técnicos, incluyendo casos límite y validaciones de patrones.
"""

import unittest
import pytest
import numpy as np
import pandas as pd

from genesis.analysis.indicators import TechnicalIndicators


class TestTechnicalIndicatorsIntermediate(unittest.TestCase):
    """Tests intermedios para TechnicalIndicators."""
    
    def setUp(self):
        """Configurar el entorno de prueba."""
        self.indicators = TechnicalIndicators()
        
        # Datos de tendencia alcista clara
        self.uptrend_data = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,
                                     13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 
                                     17.0, 17.5, 18.0, 18.5, 19.0, 19.5])
        
        # Datos de tendencia bajista clara
        self.downtrend_data = np.array([20.0, 19.5, 19.0, 18.5, 18.0, 17.5, 17.0,
                                       16.5, 16.0, 15.5, 15.0, 14.5, 14.0, 13.5, 
                                       13.0, 12.5, 12.0, 11.5, 11.0, 10.5])
        
        # Datos con patrón de onda
        self.wave_data = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 8.0, 9.0, 
                                  10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 
                                  10.0, 9.0, 8.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        
        # Datos volatiles
        self.volatile_data = np.array([10.0, 12.0, 9.0, 13.0, 8.0, 14.0, 7.0, 15.0, 
                                      8.0, 16.0, 9.0, 17.0, 10.0, 18.0, 11.0, 19.0, 
                                      12.0, 20.0, 13.0, 21.0])
        
        # Datos planos
        self.flat_data = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 
                                  10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 
                                  10.0, 10.0, 10.0, 10.0])
        
        # Datos OHLC
        self.high_data = np.array([12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 
                                  16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 
                                  20.0, 20.5, 21.0, 21.5])
        
        self.low_data = np.array([9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 
                                 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 
                                 17.0, 17.5, 18.0, 18.5])
        
        self.close_data = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 
                                   14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 
                                   18.0, 18.5, 19.0, 19.5])
    
    def test_sma_detects_trends(self):
        """Verificar que SMA detecta correctamente tendencias."""
        # SMA en tendencia alcista
        sma_up = self.indicators.sma(self.uptrend_data, 5)
        
        # En tendencia alcista, la SMA debe ser menor que el precio actual
        # después de unos cuantos períodos
        for i in range(10, len(self.uptrend_data)):
            self.assertLess(sma_up[i], self.uptrend_data[i])
        
        # SMA en tendencia bajista
        sma_down = self.indicators.sma(self.downtrend_data, 5)
        
        # En tendencia bajista, la SMA debe ser mayor que el precio actual
        # después de unos cuantos períodos
        for i in range(10, len(self.downtrend_data)):
            self.assertGreater(sma_down[i], self.downtrend_data[i])
    
    def test_ema_responds_faster_than_sma(self):
        """Verificar que EMA responde más rápido a los cambios de precio que SMA."""
        period = 10
        
        # Calcular SMA y EMA para datos con cambio repentino
        # Crear datos con cambio brusco (10, 10, 10, ... 20, 20, 20)
        sudden_change = np.array([10.0] * 15 + [20.0] * 15)
        
        sma = self.indicators.sma(sudden_change, period)
        ema = self.indicators.ema(sudden_change, period)
        
        # Después del cambio, EMA debería responder más rápido
        # Comparar 5 puntos después del cambio
        for i in range(16, 21):
            self.assertGreater(ema[i], sma[i])
    
    def test_rsi_pattern_recognition(self):
        """Verificar que RSI identifica correctamente patrones de sobrecompra/sobreventa."""
        period = 14
        
        # RSI en tendencia alcista sostenida
        rsi_up = self.indicators.rsi(self.uptrend_data, period)
        
        # En tendencia alcista sostenida, el RSI debería estar alto
        self.assertTrue(np.nanmean(rsi_up[period:]) > 60)
        
        # RSI en tendencia bajista sostenida
        rsi_down = self.indicators.rsi(self.downtrend_data, period)
        
        # En tendencia bajista sostenida, el RSI debería estar bajo
        self.assertTrue(np.nanmean(rsi_down[period:]) < 40)
        
        # RSI en datos con patrón de onda
        rsi_wave = self.indicators.rsi(self.wave_data, period)
        
        # En patrón de onda, el RSI debería oscilar
        # Verificamos que haya valores tanto altos como bajos
        rsi_valid = rsi_wave[~np.isnan(rsi_wave)]
        self.assertTrue(np.any(rsi_valid > 60))
        self.assertTrue(np.any(rsi_valid < 40))
    
    def test_macd_confirms_trend_changes(self):
        """Verificar que MACD confirma cambios de tendencia."""
        # Crear datos con cambio de tendencia
        trend_change = np.concatenate([self.uptrend_data, self.downtrend_data])
        
        # Calcular MACD
        macd_line, signal_line, histogram = self.indicators.macd(trend_change)
        
        # Verificar que el histograma cambia de positivo a negativo
        # durante el cambio de tendencia
        
        # Encontrar índice de cambio de tendencia
        change_idx = len(self.uptrend_data)
        
        # Verificar algunos puntos antes del cambio (debería ser positivo)
        for i in range(change_idx - 5, change_idx):
            if not np.isnan(histogram[i]):
                self.assertTrue(histogram[i] >= 0)
        
        # Verificar que eventualmente se vuelve negativo después del cambio
        # (podría tomar algunos puntos para reflejar el cambio)
        self.assertTrue(np.any(histogram[change_idx + 10:] < 0))
    
    def test_bollinger_bands_volatility(self):
        """Verificar que Bandas de Bollinger reflejan correctamente la volatilidad."""
        period = 10
        std_dev = 2.0
        
        # Calcular Bandas de Bollinger para datos planos y volatiles
        upper_flat, middle_flat, lower_flat = self.indicators.bollinger_bands(self.flat_data, period, std_dev)
        upper_vol, middle_vol, lower_vol = self.indicators.bollinger_bands(self.volatile_data, period, std_dev)
        
        # Calcular el ancho de las bandas (distancia entre superior e inferior)
        band_width_flat = upper_flat - lower_flat
        band_width_vol = upper_vol - lower_vol
        
        # El ancho promedio debería ser mayor para datos volátiles
        mean_width_flat = np.nanmean(band_width_flat[period:])
        mean_width_vol = np.nanmean(band_width_vol[period:])
        
        self.assertGreater(mean_width_vol, mean_width_flat)
    
    def test_atr_reflects_volatility(self):
        """Verificar que ATR refleja correctamente la volatilidad."""
        period = 14
        
        # Crear datos de alta y baja volatilidad
        # Para baja volatilidad, usamos datos con cambios pequeños
        high_low_vol = np.array([10.1, 10.2, 10.3, 10.2, 10.1, 10.0, 10.1, 10.2, 
                               10.3, 10.2, 10.1, 10.0, 10.1, 10.2, 10.3, 10.2, 
                               10.1, 10.0, 10.1, 10.2])
        
        low_low_vol = np.array([10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 
                              10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 
                              10.0, 10.1, 10.0, 10.1])
        
        close_low_vol = np.array([10.05, 10.15, 10.05, 10.15, 10.05, 10.15, 10.05, 10.15, 
                                10.05, 10.15, 10.05, 10.15, 10.05, 10.15, 10.05, 10.15, 
                                10.05, 10.15, 10.05, 10.15])
        
        # Para alta volatilidad, usamos datos con cambios grandes
        high_high_vol = np.array([10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 
                                13.0, 15.0, 14.0, 16.0, 15.0, 17.0, 16.0, 18.0, 
                                17.0, 19.0, 18.0, 20.0])
        
        low_high_vol = np.array([9.0, 10.0, 9.0, 11.0, 10.0, 12.0, 11.0, 13.0, 
                               12.0, 14.0, 13.0, 15.0, 14.0, 16.0, 15.0, 17.0, 
                               16.0, 18.0, 17.0, 19.0])
        
        close_high_vol = np.array([9.5, 10.5, 9.5, 11.5, 10.5, 12.5, 11.5, 13.5, 
                                 12.5, 14.5, 13.5, 15.5, 14.5, 16.5, 15.5, 17.5, 
                                 16.5, 18.5, 17.5, 19.5])
        
        # Calcular ATR
        atr_low_vol = self.indicators.atr(high_low_vol, low_low_vol, close_low_vol, period)
        atr_high_vol = self.indicators.atr(high_high_vol, low_high_vol, close_high_vol, period)
        
        # ATR promedio debería ser mayor para datos volátiles
        mean_atr_low = np.nanmean(atr_low_vol[period:])
        mean_atr_high = np.nanmean(atr_high_vol[period:])
        
        self.assertGreater(mean_atr_high, mean_atr_low)
    
    def test_adx_trend_strength(self):
        """Verificar que ADX mide correctamente la fuerza de la tendencia."""
        period = 14
        
        # ADX para tendencia alcista fuerte
        adx_up, plus_di_up, minus_di_up = self.indicators.adx(
            self.high_data, self.low_data, self.uptrend_data, period
        )
        
        # ADX para datos planos (sin tendencia)
        # Usamos los mismos high/low pero con precios planos
        adx_flat, plus_di_flat, minus_di_flat = self.indicators.adx(
            self.high_data, self.low_data, self.flat_data, period
        )
        
        # ADX promedio debería ser mayor para tendencia fuerte
        # Ignoramos valores NaN
        valid_idx = ~np.isnan(adx_up) & ~np.isnan(adx_flat)
        
        if np.any(valid_idx):
            mean_adx_up = np.mean(adx_up[valid_idx])
            mean_adx_flat = np.mean(adx_flat[valid_idx])
            
            self.assertGreater(mean_adx_up, mean_adx_flat)
        
        # En tendencia alcista, +DI debería ser mayor que -DI
        valid_idx = ~np.isnan(plus_di_up) & ~np.isnan(minus_di_up)
        
        if np.any(valid_idx):
            self.assertTrue(np.all(plus_di_up[valid_idx] > minus_di_up[valid_idx]))


# Configurar tests para pytest
@pytest.fixture
def indicators():
    """Fixture que proporciona una instancia de TechnicalIndicators."""
    return TechnicalIndicators()


@pytest.fixture
def trend_data():
    """Fixture que proporciona datos con diferentes tendencias para pruebas."""
    return {
        'uptrend': np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 
                             15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5]),
        'downtrend': np.array([20.0, 19.5, 19.0, 18.5, 18.0, 17.5, 17.0, 16.5, 16.0, 15.5, 
                               15.0, 14.5, 14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 11.0, 10.5]),
        'sideways': np.array([15.0, 15.2, 14.8, 15.1, 14.9, 15.3, 14.7, 15.2, 14.8, 15.1, 
                              14.9, 15.3, 14.7, 15.2, 14.8, 15.1, 14.9, 15.3, 14.7, 15.2])
    }


def test_sma_with_gaps(indicators):
    """Verificar el comportamiento de SMA con datos que tienen NaN."""
    # Datos con valores NaN
    data_with_gaps = np.array([10.0, np.nan, 12.0, 13.0, np.nan, 15.0, 16.0, 17.0, 18.0, 19.0])
    
    # Calcular SMA
    period = 5
    sma = indicators.sma(data_with_gaps, period)
    
    # Verificar que sma tiene NaN en las posiciones correctas
    assert np.isnan(sma[0])  # Primeros (period-1) valores son NaN
    assert np.isnan(sma[1])  # NaN en los datos originales también causa NaN en SMA
    assert np.isnan(sma[4])  # NaN en los datos originales también causa NaN en SMA


def test_ema_with_extreme_values(indicators):
    """Verificar el comportamiento de EMA con valores extremos."""
    # Datos con valores extremos
    data_with_extremes = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 100.0, 13.0, 13.5, 14.0, 14.5])
    
    # Calcular EMA
    period = 5
    ema = indicators.ema(data_with_extremes, period)
    
    # Después del valor extremo, EMA debería aumentar significativamente
    # pero luego debería disminuir gradualmente
    assert ema[5] > ema[4] + 10  # Aumento significativo después del pico
    assert ema[6] < ema[5]  # Disminución después del pico
    assert ema[7] < ema[6]  # Continúa disminuyendo


def test_rsi_with_flat_prices(indicators):
    """Verificar el comportamiento de RSI con precios planos."""
    # Datos con precios planos
    flat_data = np.array([10.0] * 20)
    
    # Calcular RSI
    period = 14
    rsi = indicators.rsi(flat_data, period)
    
    # Con precios planos, RSI debería ser 50 (neutral)
    for i in range(period, len(rsi)):
        if not np.isnan(rsi[i]):
            assert abs(rsi[i] - 50.0) < 1e-6  # Comprobación con tolerancia numérica


def test_rsi_with_only_gains(indicators):
    """Verificar el comportamiento de RSI con solo ganancias."""
    # Datos con solo ganancias
    only_gains = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                          20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0])
    
    # Calcular RSI
    period = 14
    rsi = indicators.rsi(only_gains, period)
    
    # Con solo ganancias, RSI debería estar en sobrecompra (cerca de 100)
    for i in range(period, len(rsi)):
        if not np.isnan(rsi[i]):
            assert rsi[i] > 99.0  # Debería estar muy cerca de 100


def test_rsi_with_only_losses(indicators):
    """Verificar el comportamiento de RSI con solo pérdidas."""
    # Datos con solo pérdidas
    only_losses = np.array([30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0,
                           20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0])
    
    # Calcular RSI
    period = 14
    rsi = indicators.rsi(only_losses, period)
    
    # Con solo pérdidas, RSI debería estar en sobreventa (cerca de 0)
    for i in range(period, len(rsi)):
        if not np.isnan(rsi[i]):
            assert rsi[i] < 1.0  # Debería estar muy cerca de 0


if __name__ == "__main__":
    unittest.main()