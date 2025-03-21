"""
Tests básicos para las estrategias de trading.

Este módulo prueba las funcionalidades básicas de las estrategias de trading,
incluyendo la generación de señales y la configuración de parámetros.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock

from genesis.strategies.base import Strategy
from genesis.strategies.moving_average import MovingAverageStrategy
from genesis.strategies.rsi import RSIStrategy
from genesis.strategies.bollinger_bands import BollingerBandsStrategy
from genesis.strategies.macd import MACDStrategy


class TestStrategy(Strategy):
    """Estrategia simple para propósitos de prueba."""
    
    def __init__(self, name="test_strategy", parameters=None):
        """Inicializar estrategia de prueba."""
        super().__init__(name=name, parameters=parameters or {})
        
    async def generate_signal(self, symbol, data):
        """Generar señal de prueba."""
        # Implementación simple para tests
        if len(data) == 0:
            return {"signal": "hold", "strength": 0}
            
        # Señal basada en el último precio vs. precio anterior
        last_price = data[-1]
        if len(data) > 1:
            prev_price = data[-2]
            if last_price > prev_price:
                return {"signal": "buy", "strength": 1}
            elif last_price < prev_price:
                return {"signal": "sell", "strength": 1}
        
        return {"signal": "hold", "strength": 0}


@pytest.fixture
def sample_price_data():
    """Proporciona datos de precios básicos para pruebas."""
    return np.array([100.0, 101.0, 102.0, 101.5, 100.5, 100.0, 101.0, 102.0, 103.0, 104.0])


@pytest.fixture
def sample_ohlcv_data():
    """Proporciona datos OHLCV básicos para pruebas."""
    # Estructura: timestamp, open, high, low, close, volume
    data = [
        [1609459200000, 100.0, 102.0, 99.0, 101.0, 1000.0],
        [1609545600000, 101.0, 103.0, 100.0, 102.0, 1100.0],
        [1609632000000, 102.0, 104.0, 101.0, 103.0, 1200.0],
        [1609718400000, 103.0, 105.0, 102.0, 104.0, 1300.0],
        [1609804800000, 104.0, 106.0, 103.0, 105.0, 1400.0],
        [1609891200000, 105.0, 107.0, 104.0, 106.0, 1500.0],
        [1609977600000, 106.0, 108.0, 105.0, 107.0, 1600.0],
        [1610064000000, 107.0, 109.0, 106.0, 108.0, 1700.0],
        [1610150400000, 108.0, 110.0, 107.0, 109.0, 1800.0],
        [1610236800000, 109.0, 111.0, 108.0, 110.0, 1900.0]
    ]
    return np.array(data)


@pytest.fixture
def test_strategy():
    """Proporciona una instancia de la estrategia de prueba."""
    return TestStrategy()


@pytest.mark.asyncio
async def test_strategy_basic_interface(test_strategy):
    """Probar la interfaz básica de una estrategia."""
    # Verificar atributos iniciales
    assert test_strategy.name == "test_strategy"
    assert isinstance(test_strategy.parameters, dict)
    
    # Verificar serialización
    serialized = test_strategy.to_dict()
    assert serialized["name"] == "test_strategy"
    assert "parameters" in serialized
    
    # Probar configuración de parámetros
    test_strategy.set_parameter("test_param", 100)
    assert test_strategy.parameters["test_param"] == 100
    assert test_strategy.get_parameter("test_param") == 100
    assert test_strategy.get_parameter("missing_param", default=50) == 50


@pytest.mark.asyncio
async def test_strategy_basic_signal_generation(test_strategy, sample_price_data):
    """Probar generación básica de señales."""
    # Señal con datos vacíos
    signal = await test_strategy.generate_signal("BTC/USDT", np.array([]))
    assert signal["signal"] == "hold"
    
    # Señal con datos que indican compra (subida de precio)
    signal = await test_strategy.generate_signal("BTC/USDT", np.array([100.0, 101.0]))
    assert signal["signal"] == "buy"
    
    # Señal con datos que indican venta (bajada de precio)
    signal = await test_strategy.generate_signal("BTC/USDT", np.array([101.0, 100.0]))
    assert signal["signal"] == "sell"


@pytest.mark.asyncio
async def test_moving_average_strategy_basic(sample_price_data):
    """Probar configuración y funcionalidad básica de estrategia de medias móviles."""
    # Crear estrategia con parámetros
    ma_strategy = MovingAverageStrategy(
        parameters={
            "fast_period": 3,
            "slow_period": 6,
            "signal_period": 9
        }
    )
    
    # Verificar parámetros
    assert ma_strategy.get_parameter("fast_period") == 3
    assert ma_strategy.get_parameter("slow_period") == 6
    
    # Mockear el cálculo de medias móviles para simular un cruce alcista
    with patch.object(ma_strategy, '_calculate_moving_averages') as mock_calculate:
        # Simular cruce donde EMA rápida cruza por encima de EMA lenta
        mock_calculate.return_value = (
            np.array([98.0, 99.0, 100.0, 101.0, 102.0]),  # EMA rápida
            np.array([97.0, 98.0, 99.0, 100.0, 100.0])    # EMA lenta
        )
        
        # Generar señal
        signal = await ma_strategy.generate_signal("BTC/USDT", sample_price_data)
        
        # Verificar que la señal es de compra
        assert signal["signal"] == "buy"
        assert signal["strength"] > 0


@pytest.mark.asyncio
async def test_rsi_strategy_basic(sample_price_data):
    """Probar configuración y funcionalidad básica de estrategia RSI."""
    # Crear estrategia con parámetros
    rsi_strategy = RSIStrategy(
        parameters={
            "period": 14,
            "overbought": 70,
            "oversold": 30
        }
    )
    
    # Verificar parámetros
    assert rsi_strategy.get_parameter("period") == 14
    assert rsi_strategy.get_parameter("overbought") == 70
    assert rsi_strategy.get_parameter("oversold") == 30
    
    # Mockear el cálculo de RSI para simular condición de sobreventa (señal de compra)
    with patch.object(rsi_strategy, '_calculate_rsi') as mock_calculate_rsi:
        # RSI en zona de sobreventa
        mock_calculate_rsi.return_value = 25.0
        
        # Generar señal
        signal = await rsi_strategy.generate_signal("BTC/USDT", sample_price_data)
        
        # Verificar que la señal es de compra
        assert signal["signal"] == "buy"
        assert "strength" in signal
    
    # Mockear el cálculo de RSI para simular condición de sobrecompra (señal de venta)
    with patch.object(rsi_strategy, '_calculate_rsi') as mock_calculate_rsi:
        # RSI en zona de sobrecompra
        mock_calculate_rsi.return_value = 75.0
        
        # Generar señal
        signal = await rsi_strategy.generate_signal("BTC/USDT", sample_price_data)
        
        # Verificar que la señal es de venta
        assert signal["signal"] == "sell"
        assert "strength" in signal


@pytest.mark.asyncio
async def test_bollinger_bands_strategy_basic(sample_price_data):
    """Probar configuración y funcionalidad básica de estrategia de Bandas de Bollinger."""
    # Crear estrategia con parámetros
    bb_strategy = BollingerBandsStrategy(
        parameters={
            "period": 20,
            "std_dev": 2.0
        }
    )
    
    # Verificar parámetros
    assert bb_strategy.get_parameter("period") == 20
    assert bb_strategy.get_parameter("std_dev") == 2.0
    
    # Mockear el cálculo de Bandas de Bollinger para simular precio cerca de banda inferior (señal de compra)
    with patch.object(bb_strategy, '_calculate_bollinger_bands') as mock_calculate_bb:
        # Precio cerca de banda inferior
        mock_calculate_bb.return_value = (
            105.0,  # Upper band
            100.0,  # Middle band
            95.0    # Lower band
        )
        
        # Último precio = 96, cerca de banda inferior
        last_price = 96.0
        
        # Mockear para retornar el último precio
        with patch.object(bb_strategy, '_get_last_price', return_value=last_price):
            # Generar señal
            signal = await bb_strategy.generate_signal("BTC/USDT", sample_price_data)
            
            # Verificar que la señal es de compra
            assert signal["signal"] == "buy"
            assert "strength" in signal
    
    # Mockear para precio cerca de banda superior (señal de venta)
    with patch.object(bb_strategy, '_calculate_bollinger_bands') as mock_calculate_bb:
        # Precio cerca de banda superior
        mock_calculate_bb.return_value = (
            105.0,  # Upper band
            100.0,  # Middle band
            95.0    # Lower band
        )
        
        # Último precio = 104, cerca de banda superior
        last_price = 104.0
        
        # Mockear para retornar el último precio
        with patch.object(bb_strategy, '_get_last_price', return_value=last_price):
            # Generar señal
            signal = await bb_strategy.generate_signal("BTC/USDT", sample_price_data)
            
            # Verificar que la señal es de venta
            assert signal["signal"] == "sell"
            assert "strength" in signal


@pytest.mark.asyncio
async def test_macd_strategy_basic(sample_price_data):
    """Probar configuración y funcionalidad básica de estrategia MACD."""
    # Crear estrategia con parámetros
    macd_strategy = MACDStrategy(
        parameters={
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        }
    )
    
    # Verificar parámetros
    assert macd_strategy.get_parameter("fast_period") == 12
    assert macd_strategy.get_parameter("slow_period") == 26
    assert macd_strategy.get_parameter("signal_period") == 9
    
    # Mockear el cálculo de MACD para simular cruce alcista (señal de compra)
    with patch.object(macd_strategy, '_calculate_macd') as mock_calculate_macd:
        # MACD cruza por encima de línea de señal
        mock_calculate_macd.return_value = (
            0.5,    # MACD line
            0.4,    # Signal line
            0.1     # Histogram
        )
        
        # Generar señal
        signal = await macd_strategy.generate_signal("BTC/USDT", sample_price_data)
        
        # Verificar que la señal es de compra
        assert signal["signal"] == "buy"
        assert "strength" in signal
    
    # Mockear el cálculo de MACD para simular cruce bajista (señal de venta)
    with patch.object(macd_strategy, '_calculate_macd') as mock_calculate_macd:
        # MACD cruza por debajo de línea de señal
        mock_calculate_macd.return_value = (
            0.3,    # MACD line
            0.4,    # Signal line
            -0.1    # Histogram
        )
        
        # Generar señal
        signal = await macd_strategy.generate_signal("BTC/USDT", sample_price_data)
        
        # Verificar que la señal es de venta
        assert signal["signal"] == "sell"
        assert "strength" in signal


@pytest.mark.asyncio
async def test_strategy_parameter_validation():
    """Probar validación de parámetros en estrategias."""
    # Intentar crear estrategia con parámetros inválidos
    with pytest.raises(ValueError):
        # Período negativo no es válido
        MovingAverageStrategy(parameters={"fast_period": -5})
    
    # Probar límites en RSI
    with pytest.raises(ValueError):
        # Zona de sobrecompra menor que sobreventa no tiene sentido
        RSIStrategy(parameters={"overbought": 30, "oversold": 70})
    
    # Probar parámetros de Bollinger Bands
    with pytest.raises(ValueError):
        # Desviación estándar negativa no es válida
        BollingerBandsStrategy(parameters={"std_dev": -1.0})


@pytest.mark.asyncio
async def test_strategy_with_insufficient_data(test_strategy):
    """Probar comportamiento de estrategias con datos insuficientes."""
    # Datos insuficientes para cálculos
    short_data = np.array([100.0, 101.0])
    
    # Crear estrategia que requiere más datos
    ma_strategy = MovingAverageStrategy(
        parameters={"fast_period": 5, "slow_period": 10}
    )
    
    # Debería manejar datos insuficientes sin errores
    signal = await ma_strategy.generate_signal("BTC/USDT", short_data)
    
    # La señal debería ser "hold" cuando los datos son insuficientes
    assert signal["signal"] == "hold"
    assert "insufficient_data" in signal["additional_data"]
"""