"""
Tests básicos para TechnicalIndicators.

Este módulo contiene tests unitarios para la funcionalidad básica
de cálculo de indicadores técnicos, verificando la implementación 
correcta de los algoritmos fundamentales.
"""

import unittest
import pytest
import numpy as np
import logging

from genesis.analysis.indicators import TechnicalIndicators

# Configurar logging para los tests
logging.basicConfig(level=logging.INFO)


class TestTechnicalIndicatorsBasic(unittest.TestCase):
    """Tests básicos para TechnicalIndicators."""
    
    def setUp(self):
        """Configurar el entorno de prueba."""
        self.indicators = TechnicalIndicators()
        
        # Datos de prueba simple
        self.sample_data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 
                                     20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 
                                     30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0])
        
        # Datos de precio OHLC para ATR
        self.high_data = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0])
        self.low_data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        self.close_data = np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    
    def test_sma_calculation(self):
        """Verificar cálculo de Media Móvil Simple (SMA)."""
        # Calcular SMA de 5 períodos
        period = 5
        sma = self.indicators.sma(self.sample_data, period)
        
        # Verificar que los primeros (period-1) valores son NaN
        for i in range(period - 1):
            self.assertTrue(np.isnan(sma[i]))
        
        # Verificar valores esperados
        expected_first_sma = np.mean(self.sample_data[0:period])
        self.assertAlmostEqual(sma[period-1], expected_first_sma)
        
        # Verificar un valor adicional
        expected_second_sma = np.mean(self.sample_data[1:period+1])
        self.assertAlmostEqual(sma[period], expected_second_sma)
    
    def test_ema_calculation(self):
        """Verificar cálculo de Media Móvil Exponencial (EMA)."""
        # Calcular EMA de 5 períodos
        period = 5
        ema = self.indicators.ema(self.sample_data, period)
        
        # Verificar que los primeros (period-1) valores son NaN
        for i in range(period - 1):
            self.assertTrue(np.isnan(ema[i]))
        
        # Verificar el primer valor (que debe ser igual a la SMA)
        expected_first_ema = np.mean(self.sample_data[0:period])
        self.assertAlmostEqual(ema[period-1], expected_first_ema)
        
        # Verificar el segundo valor
        alpha = 2 / (period + 1)
        expected_second_ema = alpha * self.sample_data[period] + (1 - alpha) * expected_first_ema
        self.assertAlmostEqual(ema[period], expected_second_ema)
    
    def test_rsi_calculation(self):
        """Verificar cálculo de RSI."""
        # Datos con tendencia alcista clara para RSI
        uptrend_data = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5,
                                 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5])
        
        # Calcular RSI
        period = 14
        rsi = self.indicators.rsi(uptrend_data, period)
        
        # Verificar que los primeros valores son NaN
        for i in range(period):
            self.assertTrue(np.isnan(rsi[i]))
        
        # En una tendencia alcista consistente, RSI debería ser alto
        self.assertTrue(rsi[period] > 70)
        
        # Datos con tendencia bajista clara para RSI
        downtrend_data = np.array([20.0, 19.5, 19.0, 18.5, 18.0, 17.5, 17.0, 16.5, 16.0, 15.5,
                                   15.0, 14.5, 14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 11.0, 10.5])
        
        # Calcular RSI
        rsi = self.indicators.rsi(downtrend_data, period)
        
        # En una tendencia bajista consistente, RSI debería ser bajo
        self.assertTrue(rsi[period] < 30)
    
    def test_macd_calculation(self):
        """Verificar cálculo de MACD."""
        # Calcular MACD
        fast_period = 12
        slow_period = 26
        signal_period = 9
        
        # Necesitamos suficientes datos para el cálculo
        sample_data = np.arange(1.0, 101.0, 1.0)  # 100 puntos de datos
        
        macd_line, signal_line, histogram = self.indicators.macd(
            sample_data, fast_period, slow_period, signal_period
        )
        
        # Verificar que los primeros valores son NaN
        for i in range(slow_period - 1):
            self.assertTrue(np.isnan(macd_line[i]))
        
        for i in range(slow_period + signal_period - 2):
            self.assertTrue(np.isnan(signal_line[i]))
        
        # Verificar que el histograma es la diferencia entre MACD y señal
        valid_idx = slow_period + signal_period - 1
        self.assertAlmostEqual(histogram[valid_idx], macd_line[valid_idx] - signal_line[valid_idx])
    
    def test_bollinger_bands_calculation(self):
        """Verificar cálculo de Bandas de Bollinger."""
        # Calcular Bandas de Bollinger
        period = 20
        std_dev = 2.0
        
        upper, middle, lower = self.indicators.bollinger_bands(
            self.sample_data, period, std_dev
        )
        
        # Verificar que los primeros valores son NaN
        for i in range(period - 1):
            self.assertTrue(np.isnan(upper[i]))
            self.assertTrue(np.isnan(middle[i]))
            self.assertTrue(np.isnan(lower[i]))
        
        # Verificar que la banda media es la SMA
        expected_sma = np.mean(self.sample_data[0:period])
        self.assertAlmostEqual(middle[period-1], expected_sma)
        
        # Verificar que las bandas superior e inferior están a la distancia correcta
        std_value = np.std(self.sample_data[0:period], ddof=1)
        expected_upper = expected_sma + (std_dev * std_value)
        expected_lower = expected_sma - (std_dev * std_value)
        
        self.assertAlmostEqual(upper[period-1], expected_upper)
        self.assertAlmostEqual(lower[period-1], expected_lower)
    
    def test_atr_calculation(self):
        """Verificar cálculo de ATR."""
        # Calcular ATR
        period = 5
        atr = self.indicators.atr(
            self.high_data, self.low_data, self.close_data, period
        )
        
        # Verificar que los primeros valores son NaN
        for i in range(period - 1):
            self.assertTrue(np.isnan(atr[i]))
        
        # Calcular true range para el primer punto
        tr_1 = self.high_data[0] - self.low_data[0]
        
        # Calcular true range para el segundo punto
        tr_2 = max(
            self.high_data[1] - self.low_data[1],
            abs(self.high_data[1] - self.close_data[0]),
            abs(self.low_data[1] - self.close_data[0])
        )
        
        # Verificar que los valores de TR son correctos
        self.assertEqual(tr_1, 2.0)  # 12 - 10 = 2
        self.assertEqual(tr_2, 2.0)  # 13 - 11 = 2


# Configurar tests para pytest
@pytest.fixture
def indicators():
    """Fixture que proporciona una instancia de TechnicalIndicators."""
    return TechnicalIndicators()


@pytest.fixture
def sample_data():
    """Fixture que proporciona datos de muestra para pruebas."""
    return np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 
                     20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 
                     30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0])


def test_sma_with_invalid_period(indicators, sample_data):
    """Verificar el comportamiento de SMA con período inválido."""
    # Período más largo que los datos
    period = len(sample_data) + 1
    sma = indicators.sma(sample_data, period)
    
    # Todos los valores deberían ser NaN
    assert np.all(np.isnan(sma))


def test_ema_with_invalid_period(indicators, sample_data):
    """Verificar el comportamiento de EMA con período inválido."""
    # Período más largo que los datos
    period = len(sample_data) + 1
    ema = indicators.ema(sample_data, period)
    
    # Todos los valores deberían ser NaN
    assert np.all(np.isnan(ema))


def test_proxy_methods(indicators, sample_data):
    """Verificar que los métodos proxy llaman correctamente a los métodos de implementación."""
    period = 5
    
    # SMA
    sma_direct = indicators.sma(sample_data, period)
    sma_proxy = indicators.calculate_sma(sample_data, period)
    assert np.array_equal(sma_direct, sma_proxy)
    
    # EMA
    ema_direct = indicators.ema(sample_data, period)
    ema_proxy = indicators.calculate_ema(sample_data, period)
    assert np.array_equal(ema_direct, ema_proxy)
    
    # RSI
    rsi_direct = indicators.rsi(sample_data, period)
    rsi_proxy = indicators.calculate_rsi(sample_data, period)
    assert np.array_equal(rsi_direct, rsi_proxy)


if __name__ == "__main__":
    unittest.main()