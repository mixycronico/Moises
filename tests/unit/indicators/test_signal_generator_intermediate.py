"""
Tests intermedios para SignalGenerator.

Este módulo contiene tests unitarios para casos intermedios de generación
de señales, incluyendo escenarios de mercado realistas y combinación de señales.
"""

import unittest
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from genesis.analysis.indicators import TechnicalIndicators
from genesis.analysis.signal_generator import SignalGenerator


class TestSignalGeneratorIntermediate(unittest.TestCase):
    """Tests intermedios para SignalGenerator."""
    
    def setUp(self):
        """Configurar el entorno de prueba."""
        self.indicators = TechnicalIndicators()
        self.signal_generator = SignalGenerator(self.indicators)
        
        # Tendencia alcista
        self.uptrend_data = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,
                                     13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 
                                     17.0, 17.5, 18.0, 18.5, 19.0, 19.5,
                                     20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0,
                                     23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5])
        
        # Tendencia bajista
        self.downtrend_data = np.array([25.0, 24.5, 24.0, 23.5, 23.0, 22.5, 22.0,
                                       21.5, 21.0, 20.5, 20.0, 19.5, 19.0, 18.5, 
                                       18.0, 17.5, 17.0, 16.5, 16.0, 15.5,
                                       15.0, 14.5, 14.0, 13.5, 13.0, 12.5, 12.0,
                                       11.5, 11.0, 10.5, 10.0, 9.5, 9.0, 8.5])
        
        # Datos laterales (sideways)
        self.sideways_data = np.array([15.0, 15.2, 14.8, 15.1, 14.9, 15.3, 14.7, 15.2, 
                                      14.8, 15.1, 14.9, 15.3, 14.7, 15.2, 14.8, 15.1, 
                                      14.9, 15.3, 14.7, 15.2, 14.8, 15.1, 14.9, 15.3, 
                                      14.7, 15.2, 14.8, 15.1, 14.9, 15.3, 14.7, 15.2])
        
        # Cambio de tendencia (de alcista a bajista)
        self.trend_change_up_to_down = np.concatenate([self.uptrend_data, self.downtrend_data])
        
        # Cambio de tendencia (de bajista a alcista)
        self.trend_change_down_to_up = np.concatenate([self.downtrend_data, self.uptrend_data])
    
    def test_ema_signal_during_strong_uptrend(self):
        """Verificar señales EMA durante tendencia alcista fuerte."""
        # En una tendencia alcista fuerte, la EMA corta debería estar por encima de la larga
        with patch.object(self.indicators, 'calculate_ema') as mock_calculate_ema:
            # Simular EMA corta por encima de EMA larga (consistentemente)
            mock_calculate_ema.side_effect = lambda data, period: (
                np.array([np.nan] * (len(data) - 2) + [25.0, 26.0]) if period == 9 else
                np.array([np.nan] * (len(data) - 2) + [20.0, 21.0])
            )
            
            # No debería generar señal de compra porque no hay cruce (ya está en posición)
            signal = self.signal_generator.generate_ema_signal(self.uptrend_data, 9, 21)
            self.assertEqual(signal, self.signal_generator.HOLD)
    
    def test_ema_crossover_at_trend_change(self):
        """Verificar que el cruce de EMA captura el cambio de tendencia."""
        # Si hay un cambio de tendencia, debería haber un cruce EMA
        # Primero hacemos una simulación manual del cruce
        
        # Para tendencia alcista a bajista
        with patch.object(self.indicators, 'calculate_ema') as mock_calculate_ema:
            # Simular EMA corta cruzando hacia abajo la EMA larga (cambio de tendencia)
            mock_calculate_ema.side_effect = lambda data, period: (
                np.array([np.nan] * (len(data) - 2) + [25.0, 24.0]) if period == 9 else
                np.array([np.nan] * (len(data) - 2) + [24.5, 24.5])
            )
            
            signal = self.signal_generator.generate_ema_signal(self.trend_change_up_to_down, 9, 21)
            self.assertEqual(signal, self.signal_generator.SELL)
        
        # Para tendencia bajista a alcista
        with patch.object(self.indicators, 'calculate_ema') as mock_calculate_ema:
            # Simular EMA corta cruzando hacia arriba la EMA larga (cambio de tendencia)
            mock_calculate_ema.side_effect = lambda data, period: (
                np.array([np.nan] * (len(data) - 2) + [10.0, 11.0]) if period == 9 else
                np.array([np.nan] * (len(data) - 2) + [10.5, 10.5])
            )
            
            signal = self.signal_generator.generate_ema_signal(self.trend_change_down_to_up, 9, 21)
            self.assertEqual(signal, self.signal_generator.BUY)
    
    def test_rsi_signals_in_different_market_conditions(self):
        """Verificar señales RSI en diferentes condiciones de mercado."""
        # En tendencia alcista fuerte, RSI debería indicar sobrecompra
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            mock_calculate_rsi.return_value = np.array([np.nan] * (len(self.uptrend_data) - 1) + [75.0])
            
            signal = self.signal_generator.generate_rsi_signal(self.uptrend_data)
            self.assertEqual(signal, self.signal_generator.SELL)
        
        # En tendencia bajista fuerte, RSI debería indicar sobreventa
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            mock_calculate_rsi.return_value = np.array([np.nan] * (len(self.downtrend_data) - 1) + [25.0])
            
            signal = self.signal_generator.generate_rsi_signal(self.downtrend_data)
            self.assertEqual(signal, self.signal_generator.BUY)
        
        # En mercado lateral, RSI debería oscilar sin tendencia definida
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            mock_calculate_rsi.return_value = np.array([np.nan] * (len(self.sideways_data) - 1) + [50.0])
            
            signal = self.signal_generator.generate_rsi_signal(self.sideways_data)
            self.assertEqual(signal, self.signal_generator.HOLD)
    
    def test_rsi_signal_near_thresholds(self):
        """Verificar comportamiento de señales RSI cerca de los umbrales."""
        # Justo en el umbral de sobreventa
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            mock_calculate_rsi.return_value = np.array([np.nan] * (len(self.downtrend_data) - 1) + [30.0])
            
            signal = self.signal_generator.generate_rsi_signal(self.downtrend_data)
            self.assertEqual(signal, self.signal_generator.BUY)
        
        # Justo por encima del umbral de sobreventa
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            mock_calculate_rsi.return_value = np.array([np.nan] * (len(self.downtrend_data) - 1) + [30.1])
            
            signal = self.signal_generator.generate_rsi_signal(self.downtrend_data)
            self.assertEqual(signal, self.signal_generator.HOLD)
        
        # Justo en el umbral de sobrecompra
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            mock_calculate_rsi.return_value = np.array([np.nan] * (len(self.uptrend_data) - 1) + [70.0])
            
            signal = self.signal_generator.generate_rsi_signal(self.uptrend_data)
            self.assertEqual(signal, self.signal_generator.SELL)
        
        # Justo por debajo del umbral de sobrecompra
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            mock_calculate_rsi.return_value = np.array([np.nan] * (len(self.uptrend_data) - 1) + [69.9])
            
            signal = self.signal_generator.generate_rsi_signal(self.uptrend_data)
            self.assertEqual(signal, self.signal_generator.HOLD)
    
    def test_macd_catches_momentum_change(self):
        """Verificar que MACD captura cambios de momentum."""
        # Simular un cambio de tendencia alcista a bajista
        with patch.object(self.indicators, 'calculate_macd') as mock_calculate_macd:
            # MACD cruza por debajo de la línea de señal
            mock_calculate_macd.return_value = (
                np.array([np.nan] * (len(self.trend_change_up_to_down) - 2) + [0.5, -0.1]),  # MACD
                np.array([np.nan] * (len(self.trend_change_up_to_down) - 2) + [0.0, 0.0]),   # Señal
                np.array([np.nan] * (len(self.trend_change_up_to_down) - 2) + [0.5, -0.1])   # Histograma
            )
            
            signal = self.signal_generator.generate_macd_signal(self.trend_change_up_to_down)
            self.assertEqual(signal, self.signal_generator.SELL)
        
        # Simular un cambio de tendencia bajista a alcista
        with patch.object(self.indicators, 'calculate_macd') as mock_calculate_macd:
            # MACD cruza por encima de la línea de señal
            mock_calculate_macd.return_value = (
                np.array([np.nan] * (len(self.trend_change_down_to_up) - 2) + [-0.5, 0.1]),  # MACD
                np.array([np.nan] * (len(self.trend_change_down_to_up) - 2) + [0.0, 0.0]),   # Señal
                np.array([np.nan] * (len(self.trend_change_down_to_up) - 2) + [-0.5, 0.1])   # Histograma
            )
            
            signal = self.signal_generator.generate_macd_signal(self.trend_change_down_to_up)
            self.assertEqual(signal, self.signal_generator.BUY)
    
    def test_bollinger_bands_in_different_volatility(self):
        """Verificar señales de Bandas de Bollinger con diferentes volatilidades."""
        # Mercado con baja volatilidad (bandas estrechas)
        with patch.object(self.indicators, 'calculate_bollinger_bands') as mock_calculate_bb:
            mock_calculate_bb.return_value = (
                np.array([np.nan] * (len(self.sideways_data) - 1) + [15.5]),  # Banda superior
                np.array([np.nan] * (len(self.sideways_data) - 1) + [15.0]),  # Banda media
                np.array([np.nan] * (len(self.sideways_data) - 1) + [14.5])   # Banda inferior
            )
            
            # Precio en la banda superior
            signal = self.signal_generator.generate_bollinger_bands_signal(
                np.append(self.sideways_data[:-1], 15.5)
            )
            self.assertEqual(signal, self.signal_generator.SELL)
            
            # Precio en la banda inferior
            signal = self.signal_generator.generate_bollinger_bands_signal(
                np.append(self.sideways_data[:-1], 14.5)
            )
            self.assertEqual(signal, self.signal_generator.BUY)
        
        # Mercado con alta volatilidad (bandas anchas)
        with patch.object(self.indicators, 'calculate_bollinger_bands') as mock_calculate_bb:
            mock_calculate_bb.return_value = (
                np.array([np.nan] * (len(self.sideways_data) - 1) + [20.0]),  # Banda superior
                np.array([np.nan] * (len(self.sideways_data) - 1) + [15.0]),  # Banda media
                np.array([np.nan] * (len(self.sideways_data) - 1) + [10.0])   # Banda inferior
            )
            
            # Precio cerca de la banda media
            signal = self.signal_generator.generate_bollinger_bands_signal(
                np.append(self.sideways_data[:-1], 16.0)
            )
            self.assertEqual(signal, self.signal_generator.HOLD)
    
    def test_complex_signal_combination(self):
        """Verificar la combinación de señales complejas."""
        # Señales contradictorias
        signals = [
            {"signal": self.signal_generator.SIGNAL_BUY, "strength": 5.0},    # RSI dice comprar con fuerza alta
            {"signal": self.signal_generator.SIGNAL_SELL, "strength": 2.0},   # MACD dice vender con fuerza baja
            {"signal": self.signal_generator.SIGNAL_BUY, "strength": 4.0},    # Bollinger dice comprar con fuerza media
            {"signal": self.signal_generator.SIGNAL_HOLD, "strength": 1.0},   # EMA dice mantener con fuerza mínima
            {"signal": self.signal_generator.SIGNAL_BUY, "strength": 3.0}     # Otro indicador dice comprar con fuerza media
        ]
        
        # Método por mayoría
        result = self.signal_generator.combine_signals(signals, "majority")
        self.assertEqual(result["signal"], self.signal_generator.SIGNAL_BUY)
        
        # Método conservador (dado que hay contradicción, debería ser HOLD)
        result = self.signal_generator.combine_signals(signals, "conservative")
        self.assertEqual(result["signal"], self.signal_generator.SIGNAL_HOLD)
        
        # Método weighted (privilegia señales activas sobre HOLD)
        result = self.signal_generator.combine_signals(signals, "weighted")
        self.assertEqual(result["signal"], self.signal_generator.SIGNAL_BUY)
    
    def test_signal_override(self):
        """Verificar combinación con señal de override (salida forzada)."""
        # Señales con salida forzada
        signals = [
            {"signal": self.signal_generator.SIGNAL_BUY, "strength": 3.0},     # RSI dice comprar
            {"signal": self.signal_generator.SIGNAL_SELL, "strength": 2.0},    # MACD dice vender
            {"signal": self.signal_generator.SIGNAL_EXIT, "strength": 2.0},    # Stop-loss activado (salida forzada)
            {"signal": self.signal_generator.SIGNAL_BUY, "strength": 3.0}      # Bollinger dice comprar
        ]
        
        # La señal EXIT debería tener prioridad (aunque no está implementado en el método actual)
        # Este es un test para una posible mejora futura
        # En la implementación actual se ignora EXIT y se usa la mayoría
        result = self.signal_generator.combine_signals(signals, "majority")
        
        # Con la implementación actual, la mayoría es BUY
        self.assertEqual(result["signal"], self.signal_generator.SIGNAL_BUY)


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
def market_data():
    """Fixture que proporciona datos de mercado para diferentes escenarios."""
    return {
        'uptrend': np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                            20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0]),
        'downtrend': np.array([40.0, 39.0, 38.0, 37.0, 36.0, 35.0, 34.0, 33.0, 32.0, 31.0,
                              30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0,
                              20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0]),
        'volatile': np.array([20.0, 22.0, 19.0, 23.0, 18.0, 24.0, 17.0, 25.0, 16.0, 26.0,
                             15.0, 27.0, 14.0, 28.0, 13.0, 29.0, 12.0, 30.0, 11.0, 31.0,
                             10.0, 32.0, 9.0, 33.0, 8.0, 34.0, 7.0, 35.0, 6.0, 36.0]),
        'sideways': np.array([25.0, 25.5, 24.5, 25.2, 24.8, 25.3, 24.7, 25.4, 24.6, 25.5,
                             24.5, 25.6, 24.4, 25.7, 24.3, 25.8, 24.2, 25.9, 24.1, 26.0,
                             24.0, 26.1, 23.9, 26.2, 23.8, 26.3, 23.7, 26.4, 23.6, 26.5])
    }


def test_ema_signal_with_custom_periods(signal_generator, market_data):
    """Verificar señales EMA con períodos personalizados."""
    # Configurar el mock para retornar EMAs específicas
    signal_generator.indicators.calculate_ema.side_effect = lambda data, period: (
        np.array([np.nan] * (len(data) - 2) + [25.0, 26.0]) if period == 5 else  # EMA rápida
        np.array([np.nan] * (len(data) - 2) + [24.0, 25.0])  # EMA lenta
    )
    
    # Períodos personalizados: 5 (rápida) y 10 (lenta)
    signal = signal_generator.generate_ema_signal(market_data['uptrend'], 5, 10)
    assert signal == signal_generator.BUY


def test_rsi_signal_with_custom_thresholds(signal_generator, market_data):
    """Verificar señales RSI con umbrales personalizados."""
    # Configurar RSI en un valor intermedio
    signal_generator.indicators.calculate_rsi.return_value = np.array(
        [np.nan] * (len(market_data['uptrend']) - 1) + [45.0]
    )
    
    # Con umbrales estándar (30/70) debería ser HOLD
    signal = signal_generator.generate_rsi_signal(market_data['uptrend'])
    assert signal == signal_generator.HOLD
    
    # Con umbrales personalizados (40/60) debería ser SELL
    signal = signal_generator.generate_rsi_signal(market_data['uptrend'], 
                                                 overbought=60, oversold=40)
    assert signal == signal_generator.SELL


def test_bollinger_bands_signal_with_custom_std(signal_generator, market_data):
    """Verificar señales de Bandas de Bollinger con desviación estándar personalizada."""
    # Configurar el mock para probar diferentes desviaciones estándar
    def mock_bb_with_diff_std(data, period=20, std_dev=2.0):
        middle = 25.0
        # Ancho de banda proporcional a std_dev
        width = 5.0 * std_dev  # 10 para std_dev=2, 5 para std_dev=1
        return (
            np.array([np.nan] * (len(data) - 1) + [middle + width/2]),  # Banda superior
            np.array([np.nan] * (len(data) - 1) + [middle]),            # Banda media
            np.array([np.nan] * (len(data) - 1) + [middle - width/2])   # Banda inferior
        )
    
    signal_generator.indicators.calculate_bollinger_bands.side_effect = mock_bb_with_diff_std
    
    # Precio justo en el medio
    test_data = np.append(market_data['sideways'][:-1], 25.0)
    
    # Con std_dev=2.0 (defecto), el precio está dentro de las bandas
    signal = signal_generator.generate_bollinger_bands_signal(test_data)
    assert signal == signal_generator.HOLD
    
    # Con std_dev=1.0, las bandas son más estrechas
    # El mismo precio podría estar cerca del límite superior/inferior
    signal = signal_generator.generate_bollinger_bands_signal(
        np.append(market_data['sideways'][:-1], 27.49),  # Justo debajo de banda superior
        std_dev=1.0
    )
    assert signal == signal_generator.HOLD
    
    signal = signal_generator.generate_bollinger_bands_signal(
        np.append(market_data['sideways'][:-1], 27.51),  # Justo arriba de banda superior
        std_dev=1.0
    )
    assert signal == signal_generator.SELL


def test_invalid_signal_combination_method(signal_generator):
    """Verificar comportamiento con método de combinación inválido."""
    signals = [
        {"signal": signal_generator.SIGNAL_BUY, "strength": 1.0},
        {"signal": signal_generator.SIGNAL_SELL, "strength": 1.0},
        {"signal": signal_generator.SIGNAL_HOLD, "strength": 1.0}
    ]
    
    # Método inválido debería devolver HOLD con error
    result = signal_generator.combine_signals(signals, "invalid_method")
    assert result["signal"] == signal_generator.SIGNAL_HOLD
    assert "error" in result


if __name__ == "__main__":
    unittest.main()