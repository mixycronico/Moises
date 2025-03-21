"""
Tests básicos para indicadores técnicos y generación de señales.

Este módulo contiene tests simples y directos para verificar 
las funcionalidades básicas de cálculo de indicadores técnicos
y generación de señales de trading.
"""

import unittest
import pytest
import numpy as np
import pandas as pd

# Importar componentes de análisis técnico
from genesis.analysis.indicators import TechnicalIndicators
from genesis.analysis.signal_generator import SignalGenerator


class TestTechnicalIndicatorsBasic(unittest.TestCase):
    """Tests básicos para indicadores técnicos."""
    
    def setUp(self):
        """Configuración inicial para los tests."""
        self.indicators = TechnicalIndicators()
        
        # Datos simples para pruebas
        # Tendencia alcista clara
        self.uptrend_prices = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0])
        
        # Tendencia bajista clara
        self.downtrend_prices = np.array([120.0, 118.0, 116.0, 114.0, 112.0, 110.0, 108.0, 106.0, 104.0, 102.0, 100.0])
        
        # Datos sideways (sin tendencia)
        self.sideways_prices = np.array([100.0, 102.0, 101.0, 103.0, 100.0, 102.0, 101.0, 103.0, 100.0, 102.0, 101.0])
    
    def test_calculate_sma_basic(self):
        """Verifica el cálculo básico de una media móvil simple (SMA)."""
        # Ejecutar
        sma_uptrend = self.indicators.calculate_sma(self.uptrend_prices, period=5)
        sma_downtrend = self.indicators.calculate_sma(self.downtrend_prices, period=5)
        
        # Verificar
        # En una tendencia alcista, el último valor de SMA debe ser mayor que el primero
        self.assertGreater(sma_uptrend[-1], sma_uptrend[4], 
                          "SMA en tendencia alcista debe ser creciente")
        
        # En una tendencia bajista, el último valor de SMA debe ser menor que el primero
        self.assertLess(sma_downtrend[-1], sma_downtrend[4], 
                       "SMA en tendencia bajista debe ser decreciente")
        
        # Verificar valor específico para tendencia alcista
        expected_last_value = (108.0 + 110.0 + 112.0 + 114.0 + 116.0) / 5  # Media de los últimos 5 valores
        self.assertAlmostEqual(sma_uptrend[-2], expected_last_value, delta=0.001, 
                              f"SMA incorrecto: esperado {expected_last_value}, obtenido {sma_uptrend[-2]}")
    
    def test_calculate_ema_basic(self):
        """Verifica el cálculo básico de una media móvil exponencial (EMA)."""
        # Ejecutar
        ema_uptrend = self.indicators.calculate_ema(self.uptrend_prices, period=5)
        ema_downtrend = self.indicators.calculate_ema(self.downtrend_prices, period=5)
        
        # Verificar
        # En una tendencia alcista, el último valor de EMA debe ser mayor que el primero válido
        self.assertGreater(ema_uptrend[-1], ema_uptrend[4], 
                          "EMA en tendencia alcista debe ser creciente")
        
        # En una tendencia bajista, el último valor de EMA debe ser menor que el primero válido
        self.assertLess(ema_downtrend[-1], ema_downtrend[4], 
                       "EMA en tendencia bajista debe ser decreciente")
        
        # Verificar que EMA reacciona más rápido que SMA
        sma_uptrend = self.indicators.calculate_sma(self.uptrend_prices, period=5)
        self.assertGreater(ema_uptrend[-1], sma_uptrend[-1], 
                          "EMA debe reaccionar más rápido que SMA en tendencia alcista")
    
    def test_calculate_rsi_basic(self):
        """Verifica el cálculo básico del Índice de Fuerza Relativa (RSI)."""
        # Ejecutar
        rsi_uptrend = self.indicators.calculate_rsi(self.uptrend_prices, period=6)
        rsi_downtrend = self.indicators.calculate_rsi(self.downtrend_prices, period=6)
        
        # Verificar
        # En una tendencia alcista fuerte, el RSI debe estar en zona de sobrecompra (>70)
        self.assertGreater(rsi_uptrend[-1], 70.0, 
                          f"RSI en tendencia alcista debe estar en sobrecompra, obtenido: {rsi_uptrend[-1]}")
        
        # En una tendencia bajista fuerte, el RSI debe estar en zona de sobreventa (<30)
        self.assertLess(rsi_downtrend[-1], 30.0, 
                       f"RSI en tendencia bajista debe estar en sobreventa, obtenido: {rsi_downtrend[-1]}")
    
    def test_calculate_bollinger_bands_basic(self):
        """Verifica el cálculo básico de las Bandas de Bollinger."""
        # Ejecutar con un periodo corto para pruebas
        upper, middle, lower = self.indicators.calculate_bollinger_bands(
            self.sideways_prices, window=5, num_std_dev=2)
        
        # Verificar
        # La banda media debe ser igual a la SMA
        sma = self.indicators.calculate_sma(self.sideways_prices, period=5)
        np.testing.assert_array_almost_equal(middle[4:], sma[4:], decimal=6, 
                                           err_msg="La banda media debe ser igual a la SMA")
        
        # La banda superior debe estar por encima de la media
        self.assertTrue(np.all(upper[4:] > middle[4:]), 
                       "La banda superior debe estar siempre por encima de la media")
        
        # La banda inferior debe estar por debajo de la media
        self.assertTrue(np.all(lower[4:] < middle[4:]), 
                       "La banda inferior debe estar siempre por debajo de la media")
        
        # La distancia entre las bandas debe ser constante en datos sideways
        # (2 * desviación estándar arriba y abajo)
        upper_distance = upper[4:] - middle[4:]
        lower_distance = middle[4:] - lower[4:]
        np.testing.assert_array_almost_equal(upper_distance, lower_distance, decimal=6, 
                                           err_msg="Las distancias a las bandas deben ser simétricas")


@pytest.fixture
def indicators():
    """Fixture que proporciona una instancia de TechnicalIndicators."""
    return TechnicalIndicators()


@pytest.fixture
def signal_generator(indicators):
    """Fixture que proporciona una instancia de SignalGenerator."""
    return SignalGenerator(indicators)


def test_generate_rsi_buy_signal(indicators, signal_generator):
    """Verifica que se genera una señal de compra cuando el RSI está en sobreventa."""
    # Crear datos artificiales con RSI en sobreventa
    prices = np.array([100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 86.0, 84.0, 82.0, 80.0])
    
    # Verificar que el RSI está realmente en sobreventa
    rsi = indicators.calculate_rsi(prices, period=6)
    assert rsi[-1] < 30.0, f"RSI debe estar en sobreventa para esta prueba, obtenido: {rsi[-1]}"
    
    # Generar señal
    signal = signal_generator.generate_rsi_signal(prices)
    
    # Verificar
    assert signal == "BUY", f"Señal incorrecta, esperaba BUY, obtenido: {signal}"


def test_generate_rsi_sell_signal(indicators, signal_generator):
    """Verifica que se genera una señal de venta cuando el RSI está en sobrecompra."""
    # Crear datos artificiales con RSI en sobrecompra
    prices = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0])
    
    # Verificar que el RSI está realmente en sobrecompra
    rsi = indicators.calculate_rsi(prices, period=6)
    assert rsi[-1] > 70.0, f"RSI debe estar en sobrecompra para esta prueba, obtenido: {rsi[-1]}"
    
    # Generar señal
    signal = signal_generator.generate_rsi_signal(prices)
    
    # Verificar
    assert signal == "SELL", f"Señal incorrecta, esperaba SELL, obtenido: {signal}"


def test_generate_bollinger_bands_signals(indicators, signal_generator):
    """Verifica la generación de señales basadas en Bandas de Bollinger."""
    # Datos sideways con un valor final por encima de la banda superior
    prices_high = np.array([100.0, 102.0, 101.0, 103.0, 100.0, 102.0, 101.0, 103.0, 100.0, 102.0, 110.0])
    
    # Datos sideways con un valor final por debajo de la banda inferior
    prices_low = np.array([100.0, 102.0, 101.0, 103.0, 100.0, 102.0, 101.0, 103.0, 100.0, 102.0, 90.0])
    
    # Generar señales
    signal_high = signal_generator.generate_bollinger_bands_signal(prices_high)
    signal_low = signal_generator.generate_bollinger_bands_signal(prices_low)
    
    # Verificar
    assert signal_high == "SELL", f"Señal incorrecta para precio alto, esperaba SELL, obtenido: {signal_high}"
    assert signal_low == "BUY", f"Señal incorrecta para precio bajo, esperaba BUY, obtenido: {signal_low}"


if __name__ == "__main__":
    unittest.main()