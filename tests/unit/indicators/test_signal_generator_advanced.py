"""
Tests avanzados para SignalGenerator.

Este módulo contiene tests unitarios para casos avanzados de generación
de señales, incluyendo escenarios de mercado complejos, combinación
avanzada de señales y patrones de mercado específicos.
"""

import unittest
import pytest
import numpy as np
import pandas as pd
import logging
from unittest.mock import patch, MagicMock, call

from genesis.analysis.indicators import TechnicalIndicators
from genesis.analysis.signal_generator import SignalGenerator

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSignalGeneratorAdvanced(unittest.TestCase):
    """Tests avanzados para SignalGenerator."""
    
    def setUp(self):
        """Configurar el entorno de prueba."""
        self.indicators = MagicMock(spec=TechnicalIndicators)
        self.signal_generator = SignalGenerator(self.indicators)
        
        # Datos de mercado con patrones específicos
        
        # Doble techo (M)
        self.double_top_data = np.array([
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
            20.0, 19.0, 18.0, 17.0, 18.0, 19.0, 20.0, 19.0, 18.0, 17.0,
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0
        ])
        
        # Doble suelo (W)
        self.double_bottom_data = np.array([
            20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0,
            10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0,
            14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0
        ])
        
        # Tendencia alcista con correcciones
        self.bullish_with_corrections = np.array([
            10.0, 11.0, 12.0, 13.0, 12.5, 12.0, 13.0, 14.0, 15.0, 16.0,
            15.5, 15.0, 14.5, 15.5, 16.5, 17.5, 18.5, 18.0, 17.5, 18.5,
            19.5, 20.5, 21.5, 21.0, 20.5, 21.5, 22.5, 23.5, 24.5, 25.0
        ])
        
        # Tendencia bajista con rebotes
        self.bearish_with_rallies = np.array([
            25.0, 24.0, 23.0, 22.0, 22.5, 23.0, 22.0, 21.0, 20.0, 19.0,
            19.5, 20.0, 20.5, 19.5, 18.5, 17.5, 16.5, 17.0, 17.5, 16.5,
            15.5, 14.5, 13.5, 14.0, 14.5, 13.5, 12.5, 11.5, 10.5, 10.0
        ])
        
        # Datos con alta volatilidad
        self.high_volatility_data = np.array([
            20.0, 22.0, 18.0, 24.0, 16.0, 26.0, 14.0, 28.0, 12.0, 30.0,
            10.0, 28.0, 14.0, 26.0, 18.0, 24.0, 22.0, 18.0, 24.0, 16.0,
            26.0, 14.0, 28.0, 12.0, 30.0, 10.0, 28.0, 14.0, 26.0, 18.0
        ])
        
        # Datos con baja volatilidad
        self.low_volatility_data = np.array([
            20.0, 20.2, 19.8, 20.1, 19.9, 20.3, 19.7, 20.4, 19.6, 20.5,
            19.5, 20.4, 19.6, 20.3, 19.7, 20.2, 19.8, 20.1, 19.9, 20.0,
            20.1, 19.9, 20.2, 19.8, 20.3, 19.7, 20.4, 19.6, 20.5, 19.5
        ])
    
    def test_ema_signal_with_double_top(self):
        """Verificar señales EMA en patrón de doble techo."""
        # Simular respuesta de EMAs para un doble techo
        with patch.object(self.indicators, 'calculate_ema') as mock_calculate_ema:
            # Configurar EMAs para simular cruce bajista después del segundo techo
            # Primer techo: EMA corta por encima de la larga
            # Segundo techo: EMA corta cruza por debajo de la larga
            mock_calculate_ema.side_effect = lambda data, period: (
                np.array([np.nan] * 15 + [19.0, 20.0, 19.0, 18.0, 16.5, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0]) 
                if period == 9 else
                np.array([np.nan] * 15 + [17.0, 18.0, 18.5, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0])
            )
            
            # Verificar señales en varios puntos
            # Al comienzo del patrón, no tenemos suficientes datos
            signal = self.signal_generator.generate_ema_signal(self.double_top_data[:10], 9, 21)
            self.assertEqual(signal, self.signal_generator.HOLD)
            
            # En el primer techo, todavía no hay cruce
            signal = self.signal_generator.generate_ema_signal(self.double_top_data[:15], 9, 21)
            self.assertEqual(signal, self.signal_generator.HOLD)
            
            # Después del segundo techo, debería haber un cruce bajista
            signal = self.signal_generator.generate_ema_signal(self.double_top_data[:20], 9, 21)
            self.assertEqual(signal, self.signal_generator.SELL)
    
    def test_rsi_signal_in_market_cycles(self):
        """Verificar señales RSI en diferentes ciclos de mercado."""
        # Simular respuestas de RSI para diferentes ciclos de mercado
        
        # En tendencia alcista con correcciones, RSI debería dar señales de compra en sobreventa
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            # Simular RSI que cae a zona de sobreventa durante correcciones
            mock_rsi_values = np.array([np.nan] * 4 + [60.0, 50.0, 30.0, 35.0, 45.0, 55.0,
                                       65.0, 55.0, 45.0, 30.0, 35.0, 45.0, 55.0, 65.0, 55.0,
                                       45.0, 30.0, 35.0, 45.0, 55.0, 65.0, 75.0, 65.0, 55.0,
                                       45.0, 35.0])
            mock_calculate_rsi.return_value = mock_rsi_values
            
            # Verificar señales en puntos específicos
            # En sobreventa durante corrección, debería dar señal de compra
            signal = self.signal_generator.generate_rsi_signal(self.bullish_with_corrections[:13])
            self.assertEqual(signal, self.signal_generator.HOLD)
            
            signal = self.signal_generator.generate_rsi_signal(self.bullish_with_corrections[:14])
            self.assertEqual(signal, self.signal_generator.BUY)
            
            # En sobrecompra durante rally, debería dar señal de venta
            signal = self.signal_generator.generate_rsi_signal(self.bullish_with_corrections[:22])
            self.assertEqual(signal, self.signal_generator.SELL)
    
    def test_macd_complex_signal_patterns(self):
        """Verificar patrones complejos de señales MACD."""
        # Simular MACD para un cambio de tendencia complejo
        with patch.object(self.indicators, 'calculate_macd') as mock_calculate_macd:
            # Simular MACD y señal en un patrón de doble suelo
            # En doble suelo, MACD suele formar una "divergencia alcista"
            # (precios hacen doble mínimo, pero MACD hace mínimo más alto)
            macd_line = np.array([np.nan] * 10 + [-2.0, -1.5, -1.0, -0.5, -1.0, -1.5, -1.2, -0.8, -0.4, 0.0, 
                                 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0])
            
            signal_line = np.array([np.nan] * 10 + [-1.5, -1.4, -1.3, -1.2, -1.1, -1.1, -1.1, -1.0, -0.8, -0.6, 
                                   -0.4, -0.2, 0.0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4])
            
            histogram = macd_line - signal_line
            
            mock_calculate_macd.return_value = (macd_line, signal_line, histogram)
            
            # MACD debería dar señal de compra cuando el MACD cruza por encima de la señal
            # después del segundo suelo
            signal = self.signal_generator.generate_macd_signal(self.double_bottom_data[:20])
            self.assertEqual(signal, self.signal_generator.BUY)
    
    def test_bollinger_bands_in_high_low_volatility(self):
        """Verificar señales de Bandas de Bollinger en diferentes volatilidades."""
        # Alta volatilidad: bandas anchas
        with patch.object(self.indicators, 'calculate_bollinger_bands') as mock_calculate_bb:
            # Simular bandas anchas para alta volatilidad
            # El último precio está muy por encima de la media
            mock_calculate_bb.return_value = (
                np.array([np.nan] * (len(self.high_volatility_data) - 1) + [32.0]),  # Banda superior
                np.array([np.nan] * (len(self.high_volatility_data) - 1) + [20.0]),  # Banda media
                np.array([np.nan] * (len(self.high_volatility_data) - 1) + [8.0])    # Banda inferior
            )
            
            # El último precio es 18.0, está dentro de las bandas
            signal = self.signal_generator.generate_bollinger_bands_signal(self.high_volatility_data)
            self.assertEqual(signal, self.signal_generator.HOLD)
            
            # Simular precio por encima de la banda superior
            high_price_data = np.append(self.high_volatility_data[:-1], 33.0)
            signal = self.signal_generator.generate_bollinger_bands_signal(high_price_data)
            self.assertEqual(signal, self.signal_generator.SELL)
            
            # Simular precio por debajo de la banda inferior
            low_price_data = np.append(self.high_volatility_data[:-1], 7.0)
            signal = self.signal_generator.generate_bollinger_bands_signal(low_price_data)
            self.assertEqual(signal, self.signal_generator.BUY)
        
        # Baja volatilidad: bandas estrechas
        with patch.object(self.indicators, 'calculate_bollinger_bands') as mock_calculate_bb:
            # Simular bandas estrechas para baja volatilidad
            mock_calculate_bb.return_value = (
                np.array([np.nan] * (len(self.low_volatility_data) - 1) + [20.5]),  # Banda superior
                np.array([np.nan] * (len(self.low_volatility_data) - 1) + [20.0]),  # Banda media
                np.array([np.nan] * (len(self.low_volatility_data) - 1) + [19.5])   # Banda inferior
            )
            
            # El último precio es 19.5, está en la banda inferior
            signal = self.signal_generator.generate_bollinger_bands_signal(self.low_volatility_data)
            self.assertEqual(signal, self.signal_generator.BUY)
            
            # Simular precio ligeramente por encima de la banda superior
            high_price_data = np.append(self.low_volatility_data[:-1], 20.6)
            signal = self.signal_generator.generate_bollinger_bands_signal(high_price_data)
            self.assertEqual(signal, self.signal_generator.SELL)
    
    def test_integration_multiple_indicators(self):
        """Verificar la integración de múltiples indicadores para toma de decisiones."""
        # Mockear todos los indicadores para simular una situación específica
        
        # Escenario 1: Todas las señales apuntan a compra
        with patch.object(self.signal_generator, 'generate_ema_signal', return_value=self.signal_generator.BUY), \
             patch.object(self.signal_generator, 'generate_rsi_signal', return_value=self.signal_generator.BUY), \
             patch.object(self.signal_generator, 'generate_macd_signal', return_value=self.signal_generator.BUY), \
             patch.object(self.signal_generator, 'generate_bollinger_bands_signal', return_value=self.signal_generator.BUY):
            
            # Crear diccionarios para las señales con fuerza de señal
            signals = [
                {"signal": self.signal_generator.SIGNAL_BUY, "strength": 3.0},  # EMA
                {"signal": self.signal_generator.SIGNAL_BUY, "strength": 3.0},  # RSI
                {"signal": self.signal_generator.SIGNAL_BUY, "strength": 3.0},  # MACD
                {"signal": self.signal_generator.SIGNAL_BUY, "strength": 3.0}   # Bollinger
            ]
            
            # Todas las estrategias de combinación deberían dar compra
            result_majority = self.signal_generator.combine_signals(signals, "majority")
            result_conservative = self.signal_generator.combine_signals(signals, "conservative")
            result_weighted = self.signal_generator.combine_signals(signals, "weighted")
            
            self.assertEqual(result_majority["signal"], self.signal_generator.SIGNAL_BUY)
            self.assertEqual(result_conservative["signal"], self.signal_generator.SIGNAL_BUY)
            self.assertEqual(result_weighted["signal"], self.signal_generator.SIGNAL_BUY)
        
        # Escenario 2: Señales mixtas
        with patch.object(self.signal_generator, 'generate_ema_signal', return_value=self.signal_generator.BUY), \
             patch.object(self.signal_generator, 'generate_rsi_signal', return_value=self.signal_generator.SELL), \
             patch.object(self.signal_generator, 'generate_macd_signal', return_value=self.signal_generator.HOLD), \
             patch.object(self.signal_generator, 'generate_bollinger_bands_signal', return_value=self.signal_generator.BUY):
            
            # Crear diccionarios para las señales con fuerza de señal
            signals = [
                {"signal": self.signal_generator.SIGNAL_BUY, "strength": 3.0},     # EMA con señal fuerte
                {"signal": self.signal_generator.SIGNAL_SELL, "strength": 2.0},    # RSI
                {"signal": self.signal_generator.SIGNAL_HOLD, "strength": 1.0},    # MACD
                {"signal": self.signal_generator.SIGNAL_BUY, "strength": 2.0}      # Bollinger
            ]
            
            # Las estrategias deberían dar resultados diferentes
            result_majority = self.signal_generator.combine_signals(signals, "majority")
            result_conservative = self.signal_generator.combine_signals(signals, "conservative")
            result_weighted = self.signal_generator.combine_signals(signals, "weighted")
            
            # Mayoría: 2 BUY, 1 SELL, 1 HOLD -> BUY
            self.assertEqual(result_majority["signal"], self.signal_generator.SIGNAL_BUY)
            
            # Conservador: Señales mixtas -> HOLD
            self.assertEqual(result_conservative["signal"], self.signal_generator.SIGNAL_HOLD)
            
            # Weighted: Favorece BUY sobre HOLD por la fuerza de señal -> BUY
            self.assertEqual(result_weighted["signal"], self.signal_generator.SIGNAL_BUY)
    
    def test_response_to_market_regime_changes(self):
        """Verificar respuesta a cambios en el régimen de mercado."""
        # Crear datos que simulan un cambio de régimen (ej: de tendencia a rango)
        regime_change_data = np.concatenate([
            # Primera parte: Tendencia alcista
            np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]),
            # Segunda parte: Mercado en rango
            np.array([25.0, 24.0, 25.5, 24.5, 25.0, 24.0, 25.5, 24.5, 25.0, 24.0, 25.5, 24.5, 25.0, 24.0, 25.5])
        ])
        
        # Mockear las respuestas de indicadores en diferentes regímenes
        
        # RSI en tendencia vs. RSI en rango
        with patch.object(self.indicators, 'calculate_rsi') as mock_calculate_rsi:
            # En tendencia, RSI se mantiene alto
            # En rango, RSI oscila entre sobrecompra y sobreventa
            mock_rsi_values_trend = np.array([np.nan] * 14 + [65.0])
            mock_rsi_values_range = np.array([np.nan] * 14 + [30.0])  # Sobreventa
            
            # Verificar señal en tendencia
            mock_calculate_rsi.return_value = mock_rsi_values_trend
            signal_trend = self.signal_generator.generate_rsi_signal(regime_change_data[:15])
            
            # Verificar señal en rango
            mock_calculate_rsi.return_value = mock_rsi_values_range
            signal_range = self.signal_generator.generate_rsi_signal(regime_change_data[15:30])
            
            # En tendencia alcista, RSI alto no necesariamente significa venta (por encima del umbral)
            self.assertNotEqual(signal_trend, self.signal_generator.BUY)
            
            # En rango, RSI bajo es una señal de compra más fiable
            self.assertEqual(signal_range, self.signal_generator.BUY)
        
        # Bandas de Bollinger en tendencia vs. en rango
        with patch.object(self.indicators, 'calculate_bollinger_bands') as mock_calculate_bb:
            # En tendencia, las bandas son más anchas y el precio tiende a moverse a lo largo de una banda
            mock_calculate_bb.return_value = (
                np.array([np.nan] * 14 + [26.0]),  # Banda superior
                np.array([np.nan] * 14 + [22.0]),  # Banda media
                np.array([np.nan] * 14 + [18.0])   # Banda inferior
            )
            
            # El último precio es 24.0, está dentro de las bandas pero cerca de la superior
            signal_trend = self.signal_generator.generate_bollinger_bands_signal(regime_change_data[:15])
            
            # En rango, las bandas son más estrechas y el precio rebota entre ellas
            mock_calculate_bb.return_value = (
                np.array([np.nan] * 14 + [26.0]),  # Banda superior
                np.array([np.nan] * 14 + [25.0]),  # Banda media
                np.array([np.nan] * 14 + [24.0])   # Banda inferior
            )
            
            # El último precio es 25.5, está cerca de la banda superior
            signal_range = self.signal_generator.generate_bollinger_bands_signal(
                np.append(regime_change_data[15:29], 25.5)
            )
            
            # En tendencia, precio cerca de banda no necesariamente genera señal
            self.assertEqual(signal_trend, self.signal_generator.HOLD)
            
            # En rango, precio en banda superior es señal de venta más fiable
            self.assertEqual(signal_range, self.signal_generator.SELL)
    
    def test_advanced_signal_combinations_with_weighting(self):
        """Verificar combinaciones avanzadas de señales con ponderación."""
        # Implementar una versión personalizada de combine_signals que utiliza pesos
        # Este test simula una posible mejora futura del método
        
        def weighted_combine_signals(signals, weights):
            """Combinar señales con pesos específicos para cada indicador."""
            if len(signals) != len(weights):
                return {"signal": self.signal_generator.SIGNAL_HOLD, "strength": 0.0, "error": "Mismatch in signals and weights length"}
                
            buy_score = 0
            sell_score = 0
            
            for i, signal_data in enumerate(signals):
                signal = signal_data["signal"]
                if signal == self.signal_generator.SIGNAL_BUY:
                    buy_score += weights[i]
                elif signal == self.signal_generator.SIGNAL_SELL:
                    sell_score += weights[i]
            
            if buy_score > sell_score and buy_score > 0.5:
                return {"signal": self.signal_generator.SIGNAL_BUY, "strength": buy_score}
            elif sell_score > buy_score and sell_score > 0.5:
                return {"signal": self.signal_generator.SIGNAL_SELL, "strength": sell_score}
            else:
                return {"signal": self.signal_generator.SIGNAL_HOLD, "strength": max(buy_score, sell_score)}
        
        # Caso 1: Señales mixtas con pesos iguales
        signals = [
            {"signal": self.signal_generator.SIGNAL_BUY, "strength": 1.0},    # EMA
            {"signal": self.signal_generator.SIGNAL_SELL, "strength": 1.0},   # RSI
            {"signal": self.signal_generator.SIGNAL_BUY, "strength": 1.0},    # MACD
            {"signal": self.signal_generator.SIGNAL_HOLD, "strength": 1.0}    # Bollinger
        ]
        
        equal_weights = [0.25, 0.25, 0.25, 0.25]
        result = weighted_combine_signals(signals, equal_weights)
        
        # Con pesos iguales, tenemos 0.5 BUY, 0.25 SELL, 0.25 HOLD
        # Buy_score = 0.5, Sell_score = 0.25, Buy_score > Sell_score pero no > 0.5
        # Por lo tanto, el resultado es HOLD
        self.assertEqual(result["signal"], self.signal_generator.SIGNAL_HOLD)
        
        # Caso 2: Señales mixtas con pesos personalizados
        # Damos más peso a MACD y menos a Bollinger
        custom_weights = [0.3, 0.3, 0.3, 0.1]
        result = weighted_combine_signals(signals, custom_weights)
        
        # Con pesos personalizados, tenemos 0.6 BUY, 0.3 SELL, 0.1 HOLD
        # Buy_score = 0.6, Sell_score = 0.3, Buy_score > Sell_score y > 0.5
        # Por lo tanto, el resultado es BUY
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
def complex_market_data():
    """Fixture que proporciona datos de mercado complejos para pruebas avanzadas."""
    return {
        'choppy_market': np.array([10.0, 10.5, 9.8, 10.2, 9.5, 10.0, 10.5, 9.8, 10.2, 9.5,
                                  10.0, 10.5, 9.8, 10.2, 9.5, 10.0, 10.5, 9.8, 10.2, 9.5]),
        'breaking_support': np.array([20.0, 19.8, 20.1, 19.5, 20.0, 19.3, 19.7, 19.1, 19.5, 18.9,
                                     19.2, 18.6, 19.0, 18.4, 18.8, 18.2, 18.6, 18.0, 17.5, 17.0]),
        'breaking_resistance': np.array([10.0, 10.2, 9.9, 10.5, 10.0, 10.7, 10.3, 10.9, 10.5, 11.1,
                                        10.8, 11.4, 11.0, 11.6, 11.2, 11.8, 11.4, 12.0, 12.5, 13.0])
    }


def test_signal_detection_in_choppy_market(signal_generator, indicators, complex_market_data):
    """Verificar generación de señales en mercado rango/lateral."""
    choppy_data = complex_market_data['choppy_market']
    
    # En mercado lateral, RSI suele ser buen indicador
    # Mockear RSI para simular oscilaciones en mercado lateral
    
    # RSI bajo (sobreventa) debería generar señal de compra
    indicators.calculate_rsi.return_value = np.array([np.nan] * (len(choppy_data) - 1) + [25.0])
    signal = signal_generator.generate_rsi_signal(choppy_data)
    assert signal == signal_generator.BUY
    
    # RSI alto (sobrecompra) debería generar señal de venta
    indicators.calculate_rsi.return_value = np.array([np.nan] * (len(choppy_data) - 1) + [75.0])
    signal = signal_generator.generate_rsi_signal(choppy_data)
    assert signal == signal_generator.SELL
    
    # En mercado lateral, las EMAs suelen cruzarse frecuentemente, generando falsas señales
    def mock_ema_in_choppy_market(data, period):
        # Simular EMAs cercanas que se cruzan frecuentemente
        if period == 9:  # EMA corta
            return np.array([np.nan] * (len(data) - 2) + [10.1, 9.9])  # Cruza hacia abajo
        else:  # EMA larga
            return np.array([np.nan] * (len(data) - 2) + [10.0, 10.0])  # Se mantiene estable
    
    indicators.calculate_ema.side_effect = mock_ema_in_choppy_market
    signal = signal_generator.generate_ema_signal(choppy_data, 9, 21)
    assert signal == signal_generator.SELL  # Señal de venta (cruce hacia abajo)
    
    # En mercado lateral, MACD suele oscilar cerca de cero
    def mock_macd_in_choppy_market(data, fast_period=12, slow_period=26, signal_period=9):
        # Simular MACD oscilando cerca de cero
        macd_line = np.array([np.nan] * (len(data) - 2) + [0.05, -0.05])  # Cruza hacia abajo
        signal_line = np.array([np.nan] * (len(data) - 2) + [0.0, 0.0])    # Se mantiene en cero
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    indicators.calculate_macd.side_effect = mock_macd_in_choppy_market
    signal = signal_generator.generate_macd_signal(choppy_data)
    assert signal == signal_generator.SELL  # Señal de venta (cruce hacia abajo)


def test_detect_breakout_signals(signal_generator, indicators, complex_market_data):
    """Verificar detección de señales en ruptura de soporte/resistencia."""
    # Datos de ruptura de resistencia (hacia arriba)
    breakout_up_data = complex_market_data['breaking_resistance']
    
    # En ruptura, Bollinger Bands son útiles
    # Simular que el precio rompe la banda superior
    def mock_bb_for_breakout_up(data, period=20, std_dev=2.0):
        return (
            np.array([np.nan] * (len(data) - 1) + [12.0]),  # Banda superior
            np.array([np.nan] * (len(data) - 1) + [11.0]),  # Banda media
            np.array([np.nan] * (len(data) - 1) + [10.0])   # Banda inferior
        )
    
    indicators.calculate_bollinger_bands.side_effect = mock_bb_for_breakout_up
    
    # El último precio (13.0) está por encima de la banda superior
    signal = signal_generator.generate_bollinger_bands_signal(breakout_up_data)
    assert signal == signal_generator.SELL  # Tradicionalmente señal de venta, pero en breakout podría ser continuación
    
    # Datos de ruptura de soporte (hacia abajo)
    breakout_down_data = complex_market_data['breaking_support']
    
    # Simular que el precio rompe la banda inferior
    def mock_bb_for_breakout_down(data, period=20, std_dev=2.0):
        return (
            np.array([np.nan] * (len(data) - 1) + [19.0]),  # Banda superior
            np.array([np.nan] * (len(data) - 1) + [18.0]),  # Banda media
            np.array([np.nan] * (len(data) - 1) + [17.5])   # Banda inferior
        )
    
    indicators.calculate_bollinger_bands.side_effect = mock_bb_for_breakout_down
    
    # El último precio (17.0) está por debajo de la banda inferior
    signal = signal_generator.generate_bollinger_bands_signal(breakout_down_data)
    assert signal == signal_generator.BUY  # Tradicionalmente señal de compra, pero en breakdown podría ser continuación


def test_combine_signals_with_empty_list(signal_generator):
    """Verificar behavior de combine_signals con lista vacía."""
    result = signal_generator.combine_signals([])
    assert result["signal"] == signal_generator.SIGNAL_HOLD
    assert "error" in result


def test_combine_signals_with_custom_method(signal_generator):
    """Verificar que un método de combinación desconocido devuelve HOLD."""
    signals = [
        {"signal": signal_generator.SIGNAL_BUY, "strength": 1.0},
        {"signal": signal_generator.SIGNAL_SELL, "strength": 1.0}
    ]
    result = signal_generator.combine_signals(signals, method="non_existent_method")
    assert result["signal"] == signal_generator.SIGNAL_HOLD
    assert "error" in result


def test_api_methods_implementation(signal_generator, indicators):
    """Verificar que los métodos de API devuelven el formato correcto."""
    # Verificar método ema_crossover
    indicators.ema.return_value = np.array([np.nan] * 9 + [10.0])
    
    result = signal_generator.ema_crossover(np.array([10.0] * 10))
    assert isinstance(result, dict)
    assert "signal" in result
    
    # Verificar método rsi_signal
    indicators.rsi.return_value = np.array([np.nan] * 9 + [50.0])
    
    result = signal_generator.rsi_signal(np.array([10.0] * 10))
    assert isinstance(result, dict)
    assert "signal" in result
    
    # Verificar método macd_signal
    indicators.macd.return_value = (
        np.array([np.nan] * 9 + [0.1]),  # MACD
        np.array([np.nan] * 9 + [0.0]),  # Señal
        np.array([np.nan] * 9 + [0.1])   # Histograma
    )
    
    result = signal_generator.macd_signal(np.array([10.0] * 10))
    assert isinstance(result, dict)
    assert "signal" in result
    
    # Verificar método bollinger_bands_signal
    indicators.bollinger_bands.return_value = (
        np.array([np.nan] * 9 + [11.0]),  # Banda superior
        np.array([np.nan] * 9 + [10.0]),  # Banda media
        np.array([np.nan] * 9 + [9.0])    # Banda inferior
    )
    
    result = signal_generator.bollinger_bands_signal(np.array([10.0] * 10))
    assert isinstance(result, dict)
    assert "signal" in result


if __name__ == "__main__":
    unittest.main()