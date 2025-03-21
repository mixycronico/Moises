"""
Tests avanzados para TechnicalIndicators.

Este módulo contiene tests unitarios para casos avanzados de cálculo
de indicadores técnicos, incluyendo escenarios extremos, comparación con
implementaciones de referencia, y análisis de rendimiento.
"""

import unittest
import pytest
import numpy as np
import pandas as pd
import time
import logging

from genesis.analysis.indicators import TechnicalIndicators

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTechnicalIndicatorsAdvanced(unittest.TestCase):
    """Tests avanzados para TechnicalIndicators."""
    
    def setUp(self):
        """Configurar el entorno de prueba."""
        self.indicators = TechnicalIndicators()
        
        # Datos con incrementos de precio agresivos
        self.exp_growth_data = np.array([10.0, 11.0, 12.1, 13.3, 14.6, 16.1, 17.7, 19.5, 
                                        21.4, 23.6, 25.9, 28.5, 31.4, 34.5, 38.0, 41.8, 
                                        46.0, 50.6, 55.6, 61.2, 67.3, 74.0, 81.4, 89.6, 
                                        98.5, 108.4, 119.2, 131.1, 144.2, 158.7])
        
        # Datos con caídas de precio agresivas
        self.exp_decline_data = np.array([150.0, 136.4, 124.0, 112.7, 102.5, 93.2, 84.7, 
                                         77.0, 70.0, 63.6, 57.9, 52.6, 47.8, 43.5, 39.5, 
                                         35.9, 32.6, 29.7, 27.0, 24.5, 22.3, 20.3, 18.4, 
                                         16.8, 15.2, 13.9, 12.6, 11.4, 10.4, 9.5])
        
        # Datos con valores extremos (crash)
        self.crash_data = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 
                                   108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 40.0, 
                                   41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 
                                   50.0, 51.0, 52.0, 53.0, 54.0])
        
        # Datos OHLC para ATR extremo
        self.high_volatile_high = np.array([12.0, 15.0, 25.0, 17.0, 30.0, 40.0, 35.0, 25.0, 
                                           45.0, 50.0, 35.0, 45.0, 40.0, 35.0, 30.0, 60.0, 
                                           55.0, 45.0, 40.0, 35.0])
        
        self.high_volatile_low = np.array([8.0, 10.0, 15.0, 10.0, 15.0, 25.0, 20.0, 15.0, 
                                          25.0, 35.0, 25.0, 30.0, 25.0, 25.0, 20.0, 35.0, 
                                          40.0, 35.0, 30.0, 25.0])
        
        self.high_volatile_close = np.array([10.0, 12.0, 20.0, 15.0, 25.0, 35.0, 25.0, 20.0, 
                                            35.0, 40.0, 30.0, 40.0, 30.0, 30.0, 25.0, 55.0, 
                                            45.0, 40.0, 35.0, 30.0])
    
    def test_sma_with_extreme_growth(self):
        """Verificar SMA con crecimiento exponencial de precios."""
        # Calcular SMA
        period = 10
        sma = self.indicators.sma(self.exp_growth_data, period)
        
        # En crecimiento exponencial, SMA debería quedar muy por debajo del precio actual
        for i in range(period, len(self.exp_growth_data)):
            self.assertLess(sma[i], self.exp_growth_data[i])
        
        # La diferencia debería aumentar con el tiempo
        for i in range(period + 5, len(self.exp_growth_data) - 1):
            diff_current = self.exp_growth_data[i] - sma[i]
            diff_next = self.exp_growth_data[i+1] - sma[i+1]
            self.assertLess(diff_current, diff_next)
    
    def test_ema_responds_to_extreme_events(self):
        """Verificar que EMA responde adecuadamente a eventos extremos (crash)."""
        # Calcular EMA
        period = 10
        ema = self.indicators.ema(self.crash_data, period)
        
        # Identificar el punto de crash
        crash_idx = 15  # Índice donde ocurre la caída brusca
        
        # Antes del crash, EMA debería estar cerca del precio
        pre_crash_diff = abs(ema[crash_idx-1] - self.crash_data[crash_idx-1])
        self.assertLess(pre_crash_diff, 5.0)  # Diferencia no mayor a 5
        
        # Después del crash, EMA debería responder, pero no tan bruscamente como el precio
        post_crash_drop_price = self.crash_data[crash_idx-1] - self.crash_data[crash_idx]
        post_crash_drop_ema = ema[crash_idx-1] - ema[crash_idx]
        
        self.assertLess(post_crash_drop_ema, post_crash_drop_price)  # EMA cae menos que el precio
        self.assertGreater(post_crash_drop_ema, 0)  # EMA sí responde a la caída
    
    def test_rsi_with_extreme_scenarios(self):
        """Verificar RSI en escenarios extremos."""
        period = 14
        
        # RSI en crecimiento exponencial
        rsi_growth = self.indicators.rsi(self.exp_growth_data, period)
        
        # En crecimiento exponencial sostenido, RSI debería estar en sobrecompra
        valid_rsi_growth = rsi_growth[~np.isnan(rsi_growth)]
        self.assertTrue(np.all(valid_rsi_growth[-5:] > 70))
        
        # RSI en caída exponencial
        rsi_decline = self.indicators.rsi(self.exp_decline_data, period)
        
        # En caída exponencial sostenida, RSI debería estar en sobreventa
        valid_rsi_decline = rsi_decline[~np.isnan(rsi_decline)]
        self.assertTrue(np.all(valid_rsi_decline[-5:] < 30))
        
        # RSI en crash
        rsi_crash = self.indicators.rsi(self.crash_data, period)
        
        # Después del crash, RSI debería caer bruscamente
        crash_idx = 15  # Índice donde ocurre la caída brusca
        if not np.isnan(rsi_crash[crash_idx]) and not np.isnan(rsi_crash[crash_idx+1]):
            self.assertGreater(rsi_crash[crash_idx-1], rsi_crash[crash_idx])
            self.assertLess(rsi_crash[crash_idx+1], 40)  # Debería estar en territorio bajista
    
    def test_macd_in_trending_markets(self):
        """Verificar MACD en mercados con tendencias fuertes."""
        # MACD en crecimiento exponencial
        macd_line, signal_line, histogram = self.indicators.macd(self.exp_growth_data)
        
        # En tendencia alcista fuerte, MACD debería estar por encima de la señal
        # y el histograma debería ser positivo
        valid_idx = ~np.isnan(macd_line) & ~np.isnan(signal_line)
        if np.any(valid_idx):
            self.assertTrue(np.all(macd_line[valid_idx][-5:] > signal_line[valid_idx][-5:]))
            self.assertTrue(np.all(histogram[valid_idx][-5:] > 0))
        
        # MACD en caída exponencial
        macd_line, signal_line, histogram = self.indicators.macd(self.exp_decline_data)
        
        # En tendencia bajista fuerte, MACD debería estar por debajo de la señal
        # y el histograma debería ser negativo
        valid_idx = ~np.isnan(macd_line) & ~np.isnan(signal_line)
        if np.any(valid_idx):
            self.assertTrue(np.all(macd_line[valid_idx][-5:] < signal_line[valid_idx][-5:]))
            self.assertTrue(np.all(histogram[valid_idx][-5:] < 0))
    
    def test_bollinger_bands_volatility_expansion(self):
        """Verificar expansión de Bandas de Bollinger en períodos de alta volatilidad."""
        period = 10
        std_dev = 2.0
        
        # Crear datos con cambio de volatilidad
        # Primero datos planos, luego alta volatilidad
        low_vol = np.array([10.0] * 15)
        high_vol = np.array([10.0, 15.0, 5.0, 20.0, 0.0, 25.0, 5.0, 30.0, 10.0, 25.0, 
                            5.0, 20.0, 0.0, 15.0, 5.0])
        vol_change_data = np.concatenate([low_vol, high_vol])
        
        # Calcular Bandas de Bollinger
        upper, middle, lower = self.indicators.bollinger_bands(vol_change_data, period, std_dev)
        
        # Calcular ancho de las bandas
        band_width = upper - lower
        
        # Eliminar NaN
        valid_band_width = band_width[~np.isnan(band_width)]
        
        # El ancho de banda debería aumentar después del aumento de volatilidad
        if len(valid_band_width) > 20:  # Asegurar que tenemos suficientes puntos
            low_vol_width = np.mean(valid_band_width[5:15])  # Promedio durante baja volatilidad
            high_vol_width = np.mean(valid_band_width[-10:])  # Promedio durante alta volatilidad
            self.assertGreater(high_vol_width, low_vol_width * 2)  # Al menos el doble de ancho
    
    def test_atr_with_extreme_volatility(self):
        """Verificar ATR con volatilidad extrema."""
        period = 10
        
        # Calcular ATR para datos altamente volátiles
        atr = self.indicators.atr(
            self.high_volatile_high, self.high_volatile_low, self.high_volatile_close, period
        )
        
        # ATR debería reflejar la alta volatilidad
        valid_atr = atr[~np.isnan(atr)]
        
        if len(valid_atr) > 0:
            # Para este conjunto de datos, un ATR promedio alto indicaría volatilidad
            # Verificamos que sea al menos 10% del precio promedio
            avg_price = np.mean(self.high_volatile_close)
            avg_atr = np.mean(valid_atr)
            self.assertGreater(avg_atr, avg_price * 0.1)
    
    def test_performance_large_dataset(self):
        """Verificar rendimiento con conjuntos de datos grandes."""
        # Crear un conjunto de datos grande
        large_data = np.random.normal(100, 10, 10000)  # 10000 puntos
        
        # Medir tiempo para calcular SMA
        start_time = time.time()
        _ = self.indicators.sma(large_data, 50)
        sma_time = time.time() - start_time
        
        # Medir tiempo para calcular EMA
        start_time = time.time()
        _ = self.indicators.ema(large_data, 50)
        ema_time = time.time() - start_time
        
        # Medir tiempo para calcular RSI
        start_time = time.time()
        _ = self.indicators.rsi(large_data, 14)
        rsi_time = time.time() - start_time
        
        # Registrar tiempos
        logger.info(f"Tiempo SMA (10000 puntos): {sma_time:.4f}s")
        logger.info(f"Tiempo EMA (10000 puntos): {ema_time:.4f}s")
        logger.info(f"Tiempo RSI (10000 puntos): {rsi_time:.4f}s")
        
        # SMA debería ser bastante rápido
        self.assertLess(sma_time, 1.0)  # Menos de 1 segundo
        
        # EMA debería ser similar a SMA en rendimiento
        self.assertLess(abs(ema_time - sma_time), 0.5)  # No más de 0.5s de diferencia
        
        # RSI es más complejo, pero no debería ser excesivamente lento
        self.assertLess(rsi_time, 5.0)  # Menos de 5 segundos


# Configurar tests para pytest
@pytest.fixture
def indicators():
    """Fixture que proporciona una instancia de TechnicalIndicators."""
    return TechnicalIndicators()


@pytest.fixture
def extreme_data():
    """Fixture que proporciona datos extremos para pruebas avanzadas."""
    return {
        'gap_up': np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 
                           30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5]),
        'gap_down': np.array([30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 
                             15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5]),
        'spike_up': np.array([10.0, 10.5, 11.0, 11.5, 12.0, 50.0, 13.0, 13.5, 14.0, 14.5, 
                             15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5]),
        'spike_down': np.array([20.0, 20.5, 21.0, 21.5, 22.0, 5.0, 23.0, 23.5, 24.0, 24.5, 
                               25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5])
    }


def test_indicators_with_data_gaps(indicators, extreme_data):
    """Verificar comportamiento de indicadores con gaps en los datos."""
    # Datos con gap (salto) alcista
    gap_up_data = extreme_data['gap_up']
    
    # Calcular SMA, EMA, RSI
    period = 5
    sma = indicators.sma(gap_up_data, period)
    ema = indicators.ema(gap_up_data, period)
    rsi = indicators.rsi(gap_up_data, period)
    
    # Verificar que SMA reacciona lentamente al gap
    pre_gap_idx = 9  # Último índice antes del gap
    post_gap_idx = 10  # Primer índice después del gap
    
    # El gap en los datos es de 14.5 a 30.0 (diferencia de 15.5)
    price_gap = gap_up_data[post_gap_idx] - gap_up_data[pre_gap_idx]
    
    # SMA no debería saltar tanto como el precio
    if not np.isnan(sma[post_gap_idx]):
        sma_jump = sma[post_gap_idx] - sma[pre_gap_idx]
        assert sma_jump < price_gap
    
    # EMA debería responder más rápido que SMA pero menos que el precio
    if not np.isnan(ema[post_gap_idx]):
        ema_jump = ema[post_gap_idx] - ema[pre_gap_idx]
        assert ema_jump < price_gap
        if not np.isnan(sma[post_gap_idx]):
            assert ema_jump > sma_jump  # EMA responde más rápido que SMA
    
    # RSI debería mostrar sobrecompra extrema después del gap alcista
    if not np.isnan(rsi[post_gap_idx]):
        assert rsi[post_gap_idx] > 80  # RSI muy alto después del gap alcista


def test_indicators_with_price_spikes(indicators, extreme_data):
    """Verificar comportamiento de indicadores con picos extremos en los datos."""
    # Datos con pico alcista
    spike_up_data = extreme_data['spike_up']
    
    # Calcular Bandas de Bollinger
    period = 5
    std_dev = 2.0
    upper, middle, lower = indicators.bollinger_bands(spike_up_data, period, std_dev)
    
    # Identificar el índice del pico
    spike_idx = 5  # Índice del pico en spike_up_data
    
    # Verificar que el pico está fuera de las bandas (si ya se han calculado)
    if not np.isnan(upper[spike_idx]):
        assert spike_up_data[spike_idx] > upper[spike_idx]  # El pico debería estar por encima de la banda superior
    
    # Verificar que las bandas se expanden después del pico
    if not np.isnan(upper[spike_idx+1]) and not np.isnan(lower[spike_idx+1]):
        pre_spike_width = upper[spike_idx-1] - lower[spike_idx-1]
        post_spike_width = upper[spike_idx+1] - lower[spike_idx+1]
        assert post_spike_width > pre_spike_width  # Las bandas deberían expandirse


def test_rsi_with_flat_then_spike(indicators):
    """Verificar comportamiento de RSI con datos planos seguidos de un pico."""
    # Crear datos planos seguidos de un pico
    flat_then_spike = np.array([10.0] * 20 + [20.0] + [10.0] * 10)
    
    # Calcular RSI
    period = 14
    rsi = indicators.rsi(flat_then_spike, period)
    
    # Identificar el índice del pico
    spike_idx = 20  # Índice del pico
    
    # Con datos planos, RSI debería estar cerca de 50
    if not np.isnan(rsi[spike_idx-1]):
        assert abs(rsi[spike_idx-1] - 50.0) < 5.0  # Cerca de 50 antes del pico
    
    # El pico debería causar un RSI alto
    if not np.isnan(rsi[spike_idx+1]):
        assert rsi[spike_idx+1] > 70  # RSI alto después del pico
    
    # RSI debería regresar gradualmente a 50
    if not np.isnan(rsi[spike_idx+10]):
        assert abs(rsi[spike_idx+10] - 50.0) < 10.0  # Acercándose a 50 después de varios períodos


def test_atr_with_sudden_volatility_change(indicators):
    """Verificar comportamiento de ATR con cambio repentino de volatilidad."""
    # Crear datos con cambio repentino de volatilidad
    length = 30
    low_vol_range = 1.0
    high_vol_range = 10.0
    
    # Primero baja volatilidad, luego alta
    np.random.seed(42)  # Para reproducibilidad
    
    high = np.zeros(length)
    low = np.zeros(length)
    close = np.zeros(length)
    
    # Precios con baja volatilidad
    base_price = 100.0
    for i in range(15):
        close[i] = base_price + np.random.uniform(-low_vol_range, low_vol_range)
        high[i] = close[i] + np.random.uniform(0, low_vol_range)
        low[i] = close[i] - np.random.uniform(0, low_vol_range)
    
    # Precios con alta volatilidad
    for i in range(15, length):
        close[i] = base_price + np.random.uniform(-high_vol_range, high_vol_range)
        high[i] = close[i] + np.random.uniform(0, high_vol_range)
        low[i] = close[i] - np.random.uniform(0, high_vol_range)
    
    # Calcular ATR
    period = 5
    atr = indicators.atr(high, low, close, period)
    
    # ATR debería aumentar después del cambio de volatilidad
    if not np.isnan(atr[20]) and not np.isnan(atr[10]):
        assert atr[20] > atr[10] * 2  # Al menos el doble después del cambio


if __name__ == "__main__":
    unittest.main()