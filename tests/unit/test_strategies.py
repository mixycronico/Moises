"""
Pruebas unitarias para las estrategias de trading del sistema Genesis.

Este módulo prueba las implementaciones de estrategias de trading,
incluyendo cálculos de indicadores, generación de señales y optimización.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import asyncio

from genesis.strategies.base import Strategy, SignalType
from genesis.strategies.mean_reversion import RSIStrategy, BollingerBandsStrategy
from genesis.strategies.trend_following import MovingAverageCrossover, MACDStrategy
from genesis.strategies.sentiment_based import SentimentStrategy


class TestStrategy(Strategy):
    """Estrategia para propósitos de prueba."""
    
    def __init__(self, name="test_strategy"):
        super().__init__(name)
        self.signals = []
    
    async def generate_signal(self, symbol, data):
        """Generar una señal de prueba."""
        signal = {
            "symbol": symbol,
            "timestamp": data.index[-1],
            "signal_type": SignalType.BUY if np.random.random() > 0.5 else SignalType.SELL,
            "price": data["close"].iloc[-1],
            "strength": 0.8,
            "metadata": {}
        }
        self.signals.append(signal)
        return signal


@pytest.fixture
def sample_ohlcv_data():
    """Datos OHLCV de ejemplo para pruebas."""
    # Crear un DataFrame con 100 filas de datos OHLCV
    np.random.seed(42)  # Para reproducibilidad
    
    dates = pd.date_range("2025-01-01", periods=100, freq="1h")
    prices = np.random.normal(40000, 1000, 100).cumsum()  # Camino aleatorio
    
    df = pd.DataFrame({
        "open": prices + np.random.normal(0, 100, 100),
        "high": prices + np.random.normal(200, 100, 100),
        "low": prices - np.random.normal(200, 100, 100),
        "close": prices + np.random.normal(0, 100, 100),
        "volume": np.random.lognormal(10, 1, 100),
    }, index=dates)
    
    # Renombrar las columnas a minúsculas para compatibilidad
    df.columns = ["open", "high", "low", "close", "volume"]
    
    return df


@pytest.mark.asyncio
async def test_base_strategy():
    """Probar la funcionalidad básica de Strategy."""
    strategy = TestStrategy()
    event_bus = MagicMock()
    strategy.event_bus = event_bus
    
    # Probar inicio y parada
    await strategy.start()
    assert strategy.running is True
    
    await strategy.stop()
    assert strategy.running is False
    
    # Probar emisión de eventos
    test_data = {"test": "data"}
    await strategy.emit_event("test_event", test_data)
    
    event_bus.emit.assert_called_once_with("test_event", test_data, "test_strategy")


@pytest.mark.asyncio
async def test_rsi_strategy(sample_ohlcv_data):
    """Probar la estrategia RSI."""
    # Configurar la estrategia
    rsi_strategy = RSIStrategy(period=14, overbought=70, oversold=30)
    
    # Generar una señal
    signal = await rsi_strategy.generate_signal("BTCUSDT", sample_ohlcv_data)
    
    # Verificar que la señal tenga el formato correcto
    assert "symbol" in signal
    assert "timestamp" in signal
    assert "signal_type" in signal
    assert "price" in signal
    assert "metadata" in signal
    
    # Verificar que el RSI se calculó correctamente
    assert "rsi" in signal["metadata"]
    assert 0 <= signal["metadata"]["rsi"] <= 100


@pytest.mark.asyncio
async def test_bollinger_bands_strategy(sample_ohlcv_data):
    """Probar la estrategia de Bandas de Bollinger."""
    # Configurar la estrategia
    bb_strategy = BollingerBandsStrategy(period=20, std_dev=2.0)
    
    # Generar una señal
    signal = await bb_strategy.generate_signal("BTCUSDT", sample_ohlcv_data)
    
    # Verificar la señal
    assert "symbol" in signal
    assert "signal_type" in signal
    assert "price" in signal
    
    # Verificar las bandas de Bollinger en los metadatos
    assert "middle_band" in signal["metadata"]
    assert "upper_band" in signal["metadata"]
    assert "lower_band" in signal["metadata"]
    
    # Verificar lógica básica
    if signal["signal_type"] == SignalType.BUY:
        # En una compra, el precio debería estar cerca de la banda inferior
        assert signal["price"] <= signal["metadata"]["middle_band"]
    elif signal["signal_type"] == SignalType.SELL:
        # En una venta, el precio debería estar cerca de la banda superior
        assert signal["price"] >= signal["metadata"]["middle_band"]


@pytest.mark.asyncio
async def test_ma_crossover_strategy(sample_ohlcv_data):
    """Probar la estrategia de cruce de medias móviles."""
    # Configurar la estrategia
    ma_strategy = MovingAverageCrossover(fast_period=10, slow_period=30)
    
    # Generar una señal
    signal = await ma_strategy.generate_signal("BTCUSDT", sample_ohlcv_data)
    
    # Verificar la señal
    assert "symbol" in signal
    assert "signal_type" in signal
    assert "price" in signal
    
    # Verificar los datos de MA en los metadatos
    assert "fast_ma" in signal["metadata"]
    assert "slow_ma" in signal["metadata"]
    
    # Verificar lógica de cruce
    fast_ma = signal["metadata"]["fast_ma"]
    slow_ma = signal["metadata"]["slow_ma"]
    
    if signal["signal_type"] == SignalType.BUY:
        # En un cruce alcista, la MA rápida debería superar a la lenta
        assert fast_ma >= slow_ma
    elif signal["signal_type"] == SignalType.SELL:
        # En un cruce bajista, la MA rápida debería estar por debajo de la lenta
        assert fast_ma <= slow_ma


@pytest.mark.asyncio
async def test_macd_strategy(sample_ohlcv_data):
    """Probar la estrategia MACD."""
    # Configurar la estrategia
    macd_strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    
    # Generar una señal
    signal = await macd_strategy.generate_signal("BTCUSDT", sample_ohlcv_data)
    
    # Verificar la señal
    assert "symbol" in signal
    assert "signal_type" in signal
    assert "price" in signal
    
    # Verificar los componentes del MACD en los metadatos
    assert "macd" in signal["metadata"]
    assert "signal" in signal["metadata"]
    assert "histogram" in signal["metadata"]
    
    # Verificar la lógica del MACD
    macd = signal["metadata"]["macd"]
    macd_signal = signal["metadata"]["signal"]
    histogram = signal["metadata"]["histogram"]
    
    # El histograma debería ser la diferencia entre MACD y línea de señal
    assert abs(histogram - (macd - macd_signal)) < 1e-6
    
    if signal["signal_type"] == SignalType.BUY:
        # En una señal de compra, el MACD debería cruzar sobre la línea de señal
        assert histogram > 0
    elif signal["signal_type"] == SignalType.SELL:
        # En una señal de venta, el MACD debería cruzar bajo la línea de señal
        assert histogram < 0


@pytest.mark.asyncio
async def test_sentiment_strategy(sample_ohlcv_data):
    """Probar la estrategia basada en sentimiento."""
    # Configurar la estrategia
    sentiment_strategy = SentimentStrategy(sentiment_threshold=0.3)
    
    # Mockear el método de obtención de sentimiento
    with patch.object(sentiment_strategy, 'fetch_sentiment', return_value=asyncio.Future()) as mock_fetch:
        # Configurar el valor de retorno
        mock_fetch.return_value.set_result(0.5)  # Sentimiento positivo
        
        # Generar una señal
        signal = await sentiment_strategy.generate_signal("BTCUSDT", sample_ohlcv_data)
        
        # Verificar que se llamó al método para obtener sentimiento
        mock_fetch.assert_called_once_with("BTCUSDT")
        
        # Verificar la señal
        assert "symbol" in signal
        assert "signal_type" in signal
        assert "price" in signal
        
        # Verificar el sentimiento en los metadatos
        assert "sentiment_score" in signal["metadata"]
        assert signal["metadata"]["sentiment_score"] == 0.5
        
        # Verificar la lógica basada en sentimiento
        if signal["signal_type"] == SignalType.BUY:
            # En una señal de compra, el sentimiento debería ser positivo
            assert signal["metadata"]["sentiment_score"] > sentiment_strategy.sentiment_threshold
        elif signal["signal_type"] == SignalType.SELL:
            # En una señal de venta, el sentimiento debería ser negativo
            assert signal["metadata"]["sentiment_score"] < -sentiment_strategy.sentiment_threshold