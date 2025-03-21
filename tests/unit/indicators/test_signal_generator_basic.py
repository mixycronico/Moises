"""
Tests básicos para SignalGenerator.

Este módulo contiene tests unitarios para la funcionalidad básica
de generación de señales de trading basadas en indicadores técnicos.
"""

import unittest
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from genesis.analysis.indicators import TechnicalIndicators
from genesis.analysis.signal_generator import SignalGenerator


class TestSignalGeneratorBasic(unittest.TestCase):
    """Tests básicos para SignalGenerator."""
    
    def setUp(self):
        """Configurar el entorno de prueba."""
        self.indicators = TechnicalIndicators()
        self.signal_generator = SignalGenerator(self.indicators)
        
        # Datos de prueba simple
        self.sample_data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 
                                    20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 
                                    30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0])
    
    def test_ema_signal_no_crossover(self):
        """Verificar que no hay señal si no hay cruce de EMA."""
        # Mock de calculate_ema para devolver valores que no cruzan
        with patch.object(self.indicators, 'calculate_ema') as mock_calculate_ema:
            mock_calculate_ema.side_effect = lambda data, period: (
                np.array([np.nan] * (len(data) - 2) + [25.0, 26.0]) if period == 9 else
                np.array([np.nan] * (len(data) - 2) + [20.0, 21.0])  # siempre por debajo
            )
            
            signal = self.signal_generator.generate_ema_signal(self.sample_data, 9, 21)
            self.assertEqual(signal, self.signal_generator.HOLD)
    
    def test_ema_signal_buy_crossover(self):
        """Verificar que se genera señal de compra en cruce al alza."""
        # Mock de calculate_ema para simular un cruce al alza
        with patch.object(self.indicators, 'calculate_ema') as mock_calculate_ema:
            mock_calculate_ema.side_effect = lambda data, period: (
                np.array([np.nan] * (len(data) - 2) + [19.0, 22.0]) if period == 9 else  # cruza al alza
                np.array([np.nan] * (len(data) - 2) + [20.0, 21.0])
            )
            
            signal = self.signal_generator.generate_ema_signal(self.sample_data, 9, 21)
            self.assertEqual(signal, self.signal_generator.BUY)
    
    def test_ema_signal_sell_crossover(self):
        """Verificar que se genera señal de venta en cruce a la baja."""
        # Mock de calculate_ema para simular un cruce a la baja
        with patch.object(self.indicators, 'calculate_ema') as mock_calculate_ema:
            mock_calculate_ema.side_effect = lambda data, period: (
                np.array([np.nan] * (len(data) - 2) + [21.0, 19.0]) if period == 9 else  # cruza a la baja
                np.array([np.nan] * (len(data) - 2) + [20.0, 20.0])
            )
            
            signal = self.signal_generator.generate_ema_signal(self.sample_data, 9, 21)
            self.assertEqual(signal, self.signal_generator.SELL)
    
    def test_rsi_signal_oversold(self):
        """Verificar que se genera señal de compra en sobreventa (RSI bajo)."""
        # Mock de calculate_rsi para simular condición de sobreventa
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            mock_calculate_rsi.return_value = np.array([np.nan] * (len(self.sample_data) - 1) + [25.0])  # RSI bajo
            
            signal = self.signal_generator.generate_rsi_signal(self.sample_data)
            self.assertEqual(signal, self.signal_generator.BUY)
    
    def test_rsi_signal_overbought(self):
        """Verificar que se genera señal de venta en sobrecompra (RSI alto)."""
        # Mock de calculate_rsi para simular condición de sobrecompra
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            mock_calculate_rsi.return_value = np.array([np.nan] * (len(self.sample_data) - 1) + [75.0])  # RSI alto
            
            signal = self.signal_generator.generate_rsi_signal(self.sample_data)
            self.assertEqual(signal, self.signal_generator.SELL)
    
    def test_rsi_signal_neutral(self):
        """Verificar que se genera señal neutral cuando RSI está en rango medio."""
        # Mock de calculate_rsi para simular condición neutral
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            mock_calculate_rsi.return_value = np.array([np.nan] * (len(self.sample_data) - 1) + [50.0])  # RSI neutro
            
            signal = self.signal_generator.generate_rsi_signal(self.sample_data)
            self.assertEqual(signal, self.signal_generator.HOLD)
    
    def test_macd_signal_bullish(self):
        """Verificar que se genera señal de compra en cruce alcista de MACD."""
        # Usar períodos más pequeños para evitar error de datos insuficientes
        fast_period = 3
        slow_period = 5
        signal_period = 2
        
        # Mock de macd para simular un cruce alcista
        with patch.object(self.indicators, 'macd') as mock_macd:
            # Simulamos un cruce alcista donde:
            # - En el penúltimo punto, MACD (-0.2) < Signal (-0.1)
            # - En el último punto, MACD (0.1) > Signal (0.0)
            macd_line = np.array([np.nan] * (len(self.sample_data) - 2) + [-0.2, 0.1])
            signal_line = np.array([np.nan] * (len(self.sample_data) - 2) + [-0.1, 0.0])
            histogram = macd_line - signal_line
            
            # Solo mockear cuando se llame con nuestros períodos específicos
            mock_macd.return_value = (macd_line, signal_line, histogram)
            
            # Ejecutar con períodos reducidos
            result = self.signal_generator.macd_signal(
                self.sample_data, 
                fast_period=fast_period, 
                slow_period=slow_period, 
                signal_period=signal_period
            )
            
            self.assertEqual(result["signal"], self.signal_generator.SIGNAL_BUY)
    
    def test_macd_signal_bearish(self):
        """Verificar que se genera señal de venta en cruce bajista de MACD."""
        # Usar períodos más pequeños para evitar error de datos insuficientes
        fast_period = 3
        slow_period = 5
        signal_period = 2
        
        # Mock de macd para simular un cruce bajista
        with patch.object(self.indicators, 'macd') as mock_macd:
            # Simulamos un cruce bajista donde:
            # - En el penúltimo punto, MACD (0.2) > Signal (0.1)
            # - En el último punto, MACD (-0.1) < Signal (0.0)
            macd_line = np.array([np.nan] * (len(self.sample_data) - 2) + [0.2, -0.1])
            signal_line = np.array([np.nan] * (len(self.sample_data) - 2) + [0.1, 0.0])
            histogram = macd_line - signal_line
            
            mock_macd.return_value = (macd_line, signal_line, histogram)
            
            # Ejecutar con períodos reducidos
            result = self.signal_generator.macd_signal(
                self.sample_data, 
                fast_period=fast_period, 
                slow_period=slow_period, 
                signal_period=signal_period
            )
            
            self.assertEqual(result["signal"], self.signal_generator.SIGNAL_SELL)
    
    def test_macd_signal_neutral(self):
        """Verificar que se genera señal neutral cuando no hay cruce de MACD."""
        # Usar períodos más pequeños para evitar error de datos insuficientes
        fast_period = 3
        slow_period = 5
        signal_period = 2
        
        # Mock de macd para simular un no-cruce
        with patch.object(self.indicators, 'macd') as mock_macd:
            mock_macd.return_value = (
                np.array([np.nan] * (len(self.sample_data) - 2) + [0.1, 0.2]),  # MACD siempre por encima
                np.array([np.nan] * (len(self.sample_data) - 2) + [0.0, 0.0]),  # Señal
                np.array([np.nan] * (len(self.sample_data) - 2) + [0.1, 0.2])   # Histograma
            )
            
            # Ejecutar con períodos reducidos
            result = self.signal_generator.macd_signal(
                self.sample_data, 
                fast_period=fast_period, 
                slow_period=slow_period, 
                signal_period=signal_period
            )
            
            self.assertEqual(result["signal"], self.signal_generator.SIGNAL_HOLD)
    
    def test_bollinger_bands_signal_buy(self):
        """Verificar que se genera señal de compra cuando el precio toca la banda inferior."""
        # Último precio es igual a banda inferior (compra)
        with patch.object(self.indicators, 'calculate_bollinger_bands') as mock_calculate_bb:
            mock_calculate_bb.return_value = (
                np.array([np.nan] * (len(self.sample_data) - 1) + [45.0]),  # Banda superior
                np.array([np.nan] * (len(self.sample_data) - 1) + [40.0]),  # Banda media
                np.array([np.nan] * (len(self.sample_data) - 1) + [35.0])   # Banda inferior
            )
            
            # El último precio es 35 (igual a la banda inferior)
            signal = self.signal_generator.generate_bollinger_bands_signal(
                np.append(self.sample_data[:-1], 35.0)  # Reemplazar último valor
            )
            self.assertEqual(signal, self.signal_generator.BUY)
    
    def test_bollinger_bands_signal_sell(self):
        """Verificar que se genera señal de venta cuando el precio toca la banda superior."""
        # Último precio es igual a banda superior (venta)
        with patch.object(self.indicators, 'calculate_bollinger_bands') as mock_calculate_bb:
            mock_calculate_bb.return_value = (
                np.array([np.nan] * (len(self.sample_data) - 1) + [45.0]),  # Banda superior
                np.array([np.nan] * (len(self.sample_data) - 1) + [40.0]),  # Banda media
                np.array([np.nan] * (len(self.sample_data) - 1) + [35.0])   # Banda inferior
            )
            
            # El último precio es 45 (igual a la banda superior)
            signal = self.signal_generator.generate_bollinger_bands_signal(
                np.append(self.sample_data[:-1], 45.0)  # Reemplazar último valor
            )
            self.assertEqual(signal, self.signal_generator.SELL)
    
    def test_bollinger_bands_signal_hold(self):
        """Verificar que se genera señal neutral cuando el precio está entre bandas."""
        # Último precio está entre bandas (neutral)
        with patch.object(self.indicators, 'calculate_bollinger_bands') as mock_calculate_bb:
            mock_calculate_bb.return_value = (
                np.array([np.nan] * (len(self.sample_data) - 1) + [45.0]),  # Banda superior
                np.array([np.nan] * (len(self.sample_data) - 1) + [40.0]),  # Banda media
                np.array([np.nan] * (len(self.sample_data) - 1) + [35.0])   # Banda inferior
            )
            
            # El último precio es 40 (igual a la banda media)
            signal = self.signal_generator.generate_bollinger_bands_signal(
                np.append(self.sample_data[:-1], 40.0)  # Reemplazar último valor
            )
            self.assertEqual(signal, self.signal_generator.HOLD)


# Configurar tests para pytest
@pytest.fixture
def indicators():
    """Fixture que proporciona una instancia simulada de TechnicalIndicators."""
    return MagicMock(spec=TechnicalIndicators)


@pytest.fixture
def signal_generator(indicators):
    """Fixture que proporciona una instancia de SignalGenerator con indicadores simulados."""
    return SignalGenerator(indicators)


@pytest.fixture
def sample_data():
    """Fixture que proporciona datos de muestra para pruebas."""
    return np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 
                    20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 
                    30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0])


def test_ema_signal_with_insufficient_data(signal_generator, sample_data):
    """Verificar comportamiento con datos insuficientes en EMA."""
    # Datos insuficientes
    few_data = np.array([10.0, 11.0, 12.0])
    
    signal = signal_generator.generate_ema_signal(few_data, 9, 21)
    assert signal == signal_generator.HOLD


def test_rsi_signal_with_insufficient_data(signal_generator, sample_data):
    """Verificar comportamiento con datos insuficientes en RSI."""
    # Datos insuficientes
    few_data = np.array([10.0, 11.0, 12.0])
    
    signal = signal_generator.generate_rsi_signal(few_data)
    assert signal == signal_generator.HOLD


def test_macd_signal_with_insufficient_data(signal_generator, sample_data):
    """Verificar comportamiento con datos insuficientes en MACD."""
    # Datos insuficientes
    few_data = np.array([10.0, 11.0, 12.0])
    
    signal = signal_generator.macd_signal(few_data)
    assert signal["signal"] == signal_generator.SIGNAL_HOLD


def test_bollinger_bands_signal_with_insufficient_data(signal_generator, sample_data):
    """Verificar comportamiento con datos insuficientes en Bandas de Bollinger."""
    # Datos insuficientes
    few_data = np.array([10.0, 11.0, 12.0])
    
    signal = signal_generator.generate_bollinger_bands_signal(few_data)
    assert signal == signal_generator.HOLD


def test_combine_signals_majority_buy(signal_generator):
    """Verificar combinación de señales por mayoría con resultado compra."""
    signals = [
        {"signal": signal_generator.SIGNAL_BUY},
        {"signal": signal_generator.SIGNAL_BUY},
        {"signal": signal_generator.SIGNAL_SELL}
    ]
    result = signal_generator.combine_signals(signals, "majority")
    assert result["signal"] == signal_generator.SIGNAL_BUY


def test_combine_signals_majority_sell(signal_generator):
    """Verificar combinación de señales por mayoría con resultado venta."""
    signals = [
        {"signal": signal_generator.SIGNAL_SELL},
        {"signal": signal_generator.SIGNAL_SELL},
        {"signal": signal_generator.SIGNAL_BUY}
    ]
    result = signal_generator.combine_signals(signals, "majority")
    assert result["signal"] == signal_generator.SIGNAL_SELL


def test_combine_signals_majority_hold(signal_generator):
    """Verificar combinación de señales por mayoría con resultado mantener."""
    signals = [
        {"signal": signal_generator.SIGNAL_HOLD},
        {"signal": signal_generator.SIGNAL_HOLD},
        {"signal": signal_generator.SIGNAL_BUY}
    ]
    result = signal_generator.combine_signals(signals, "majority")
    assert result["signal"] == signal_generator.SIGNAL_HOLD


def test_combine_signals_conservative_buy(signal_generator):
    """Verificar combinación de señales conservadora con resultado compra."""
    signals = [
        {"signal": signal_generator.SIGNAL_BUY},
        {"signal": signal_generator.SIGNAL_BUY},
        {"signal": signal_generator.SIGNAL_BUY}
    ]
    result = signal_generator.combine_signals(signals, "conservative")
    assert result["signal"] == signal_generator.SIGNAL_BUY


def test_combine_signals_conservative_mixed(signal_generator):
    """Verificar combinación de señales conservadora con señales mixtas."""
    signals = [
        {"signal": signal_generator.SIGNAL_BUY},
        {"signal": signal_generator.SIGNAL_SELL},
        {"signal": signal_generator.SIGNAL_HOLD}
    ]
    result = signal_generator.combine_signals(signals, "conservative")
    assert result["signal"] == signal_generator.SIGNAL_HOLD


def test_combine_signals_weighted(signal_generator):
    """Verificar combinación de señales ponderada."""
    signals = [
        {"signal": signal_generator.SIGNAL_BUY},
        {"signal": signal_generator.SIGNAL_HOLD},
        {"signal": signal_generator.SIGNAL_HOLD}
    ]
    result = signal_generator.combine_signals(signals, "weighted")
    assert result["signal"] == signal_generator.SIGNAL_BUY


if __name__ == "__main__":
    unittest.main()