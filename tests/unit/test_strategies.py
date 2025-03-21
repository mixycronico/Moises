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
    # El mockeo del logger ahora se realiza a nivel global en conftest.py
    strategy = TestStrategy()
    
    # Crear un mock para el event_bus con un método emit async
    event_bus = MagicMock()
    # Configurar el método emit para devolver un Future completado
    mock_future = asyncio.Future()
    mock_future.set_result(None)
    event_bus.emit = MagicMock(return_value=mock_future)
    
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
    """Probar la estrategia RSI básica."""
    # Configurar la estrategia
    rsi_strategy = RSIStrategy(period=14, overbought=70, oversold=30)
    
    # Generar una señal
    signal = await rsi_strategy.generate_signal("BTCUSDT", sample_ohlcv_data)
    
    # Verificar que la señal tenga el formato correcto
    assert "type" in signal
    assert signal["type"] in [SignalType.BUY, SignalType.SELL, SignalType.HOLD, SignalType.EXIT]
    assert "reason" in signal
    
    # Verificar que el RSI está en el diccionario principal
    assert "rsi" in signal
    assert 0 <= signal["rsi"] <= 100
    
    # Obtener el valor de RSI
    rsi_value = signal["rsi"]
    if signal["type"] == SignalType.BUY:
        # En una señal de compra, el RSI debería haber cruzado desde zona de sobreventa
        assert rsi_value > 30, f"RSI = {rsi_value} debería ser > 30 para señal de compra"
    elif signal["type"] == SignalType.SELL:
        # En una señal de venta, el RSI debería haber cruzado desde zona de sobrecompra
        assert rsi_value < 70, f"RSI = {rsi_value} debería ser < 70 para señal de venta"


@pytest.mark.asyncio
async def test_rsi_strategy_custom_thresholds():
    """Probar la estrategia RSI con umbrales personalizados."""
    # En lugar de intentar generar precios que produzcan un RSI específico,
    # vamos a mockear directamente el método calculate_rsi para tener control total
    
    # Crear un DataFrame simple
    dates = pd.date_range("2025-01-01", periods=30, freq="1h")
    prices = np.linspace(100, 200, 30)  # Precios lineales de 100 a 200
    
    df = pd.DataFrame({
        "open": prices - 1,
        "high": prices + 2,
        "low": prices - 2,
        "close": prices,
        "volume": np.random.lognormal(10, 1, 30),
    }, index=dates)
    
    # Configurar la estrategia con umbrales (25/75)
    rsi_strategy = RSIStrategy(period=10, overbought=75, oversold=25)
    
    # Caso 1: Mockear un cruce del RSI desde sobreventa (RSI cruza desde 20 a 30)
    with patch.object(rsi_strategy, 'calculate_rsi') as mock_calculate_rsi:
        # Configurar el mock para devolver una serie donde el RSI cruza desde sobreventa
        mock_rsi_initial = pd.Series([float('nan')] * (len(df) - 2))
        mock_rsi_values = pd.Series([20.0, 30.0])  # RSI anterior y actual
        mock_rsi = pd.concat([mock_rsi_initial, mock_rsi_values])
        mock_calculate_rsi.return_value = mock_rsi
        
        # Generar una señal
        signal = await rsi_strategy.generate_signal("BTCUSDT", df)
        
        # Verificar que el RSI se usó correctamente
        mock_calculate_rsi.assert_called_once()
        
        # Verificar que la señal es de compra por cruce desde sobreventa
        assert signal["type"] == SignalType.BUY, f"Tipo de señal debería ser BUY pero es {signal['type']}, RSI={signal['rsi']}"
        assert "crossed above" in signal["reason"].lower() or "cruce" in signal["reason"].lower()
        
    # Caso 2: Mockear un cruce del RSI desde sobrecompra (RSI cruza desde 80 a 70)
    with patch.object(rsi_strategy, 'calculate_rsi') as mock_calculate_rsi:
        # Configurar el mock para devolver una serie donde el RSI cruza desde sobrecompra
        mock_rsi_initial = pd.Series([float('nan')] * (len(df) - 2))
        mock_rsi_values = pd.Series([80.0, 70.0])  # RSI anterior y actual
        mock_rsi = pd.concat([mock_rsi_initial, mock_rsi_values])
        mock_calculate_rsi.return_value = mock_rsi
        
        # Generar una señal
        signal = await rsi_strategy.generate_signal("BTCUSDT", df)
        
        # Verificar que la señal es de venta por cruce desde sobrecompra
        assert signal["type"] == SignalType.SELL, f"Tipo de señal debería ser SELL pero es {signal['type']}, RSI={signal['rsi']}"
        assert "crossed below" in signal["reason"].lower() or "cruce" in signal["reason"].lower()


@pytest.mark.asyncio
async def test_rsi_strategy_insufficient_data():
    """Probar el comportamiento de la estrategia RSI con datos insuficientes."""
    # Crear datos OHLCV insuficientes (menos del período necesario)
    dates = pd.date_range("2025-01-01", periods=5, freq="1h")
    prices = np.array([100, 102, 101, 103, 105])
    
    df = pd.DataFrame({
        "open": prices - 1,
        "high": prices + 2,
        "low": prices - 2,
        "close": prices,
        "volume": np.random.lognormal(10, 1, 5),
    }, index=dates)
    
    # Configurar la estrategia con un período más largo que los datos disponibles
    rsi_strategy = RSIStrategy(period=14, overbought=70, oversold=30)
    
    # Generar una señal
    signal = await rsi_strategy.generate_signal("BTCUSDT", df)
    
    # Debería devolver una señal HOLD debido a datos insuficientes
    assert signal["type"] == SignalType.HOLD
    assert "datos" in signal["reason"].lower() or "data" in signal["reason"].lower()


@pytest.mark.asyncio
async def test_bollinger_bands_strategy(sample_ohlcv_data):
    """Probar la estrategia de Bandas de Bollinger básica."""
    # Configurar la estrategia
    bb_strategy = BollingerBandsStrategy(period=20, std_dev=2.0)
    
    # Generar una señal
    signal = await bb_strategy.generate_signal("BTCUSDT", sample_ohlcv_data)
    
    # Verificar que la señal tenga el formato correcto
    assert "type" in signal
    assert signal["type"] in [SignalType.BUY, SignalType.SELL, SignalType.HOLD, SignalType.EXIT]
    assert "reason" in signal
    assert "metadata" in signal
    
    # Verificar las bandas de Bollinger en los metadatos
    assert "middle_band" in signal["metadata"]
    assert "upper_band" in signal["metadata"]
    assert "lower_band" in signal["metadata"]
    
    # Verificar coherencia en los cálculos
    middle_band = signal["metadata"]["middle_band"]
    upper_band = signal["metadata"]["upper_band"]
    lower_band = signal["metadata"]["lower_band"]
    
    # Las bandas deben estar ordenadas correctamente
    assert lower_band <= middle_band <= upper_band
    
    # Verificar lógica de señales según la posición del precio
    current_price = sample_ohlcv_data["close"].iloc[-1]
    
    if signal["type"] == SignalType.BUY:
        # Dos casos típicos para señal de compra:
        # 1. Precio por debajo o cerca de la banda inferior
        # 2. Precio cruzando hacia arriba la banda inferior
        reason = signal["reason"].lower()
        if "por debajo" in reason or "crossed below" in reason:
            assert current_price <= lower_band * 1.05  # Permitir un margen del 5%
        elif "cruzando" in reason or "crossed back above" in reason:
            assert current_price >= lower_band * 0.95  # Permitir un margen del 5%
    
    elif signal["type"] == SignalType.SELL:
        # Dos casos típicos para señal de venta:
        # 1. Precio por encima o cerca de la banda superior
        # 2. Precio cruzando hacia abajo la banda superior
        reason = signal["reason"].lower()
        if "por encima" in reason or "crossed above" in reason:
            assert current_price >= upper_band * 0.95  # Permitir un margen del 5%
        elif "cruzando" in reason or "crossed back below" in reason:
            assert current_price <= upper_band * 1.05  # Permitir un margen del 5%


@pytest.mark.asyncio
async def test_bollinger_bands_strategy_narrow_bands():
    """Probar la estrategia de Bandas de Bollinger con bandas estrechas (baja volatilidad)."""
    # Crear datos OHLCV con baja volatilidad
    dates = pd.date_range("2025-01-01", periods=30, freq="1h")
    
    # Precios con baja volatilidad alrededor de 100
    prices = np.array([100 + np.sin(i/5) * 2 for i in range(30)])
    
    df = pd.DataFrame({
        "open": prices - 0.5,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": np.random.lognormal(10, 0.2, 30),  # Volumen también estable
    }, index=dates)
    
    # Configurar la estrategia con menor desviación estándar
    bb_strategy = BollingerBandsStrategy(period=10, std_dev=1.5)
    
    # Generar una señal
    signal = await bb_strategy.generate_signal("BTCUSDT", df)
    
    # Verificar las bandas
    middle_band = signal["metadata"]["middle_band"]
    upper_band = signal["metadata"]["upper_band"]
    lower_band = signal["metadata"]["lower_band"]
    
    # En períodos de baja volatilidad, las bandas deben estar cercanas 
    band_width = (upper_band - lower_band) / middle_band
    assert band_width < 0.1, f"Ancho de banda ({band_width}) debería ser menor a 0.1 para baja volatilidad"
    
    # Verificar el tipo de señal
    current_price = df["close"].iloc[-1]
    if current_price > upper_band:
        assert signal["type"] == SignalType.SELL
    elif current_price < lower_band:
        assert signal["type"] == SignalType.BUY
    else:
        assert signal["type"] == SignalType.HOLD


@pytest.mark.asyncio
async def test_bollinger_bands_strategy_wide_bands():
    """Probar la estrategia de Bandas de Bollinger con bandas anchas (alta volatilidad)."""
    # Crear datos OHLCV con alta volatilidad
    dates = pd.date_range("2025-01-01", periods=30, freq="1h")
    
    # Base para generar precios con alta volatilidad
    base_prices = np.linspace(100, 130, 30)
    # Agregar ruido aleatorio de alta amplitud
    noise = np.random.normal(0, 15, 30)
    prices = base_prices + noise
    
    df = pd.DataFrame({
        "open": prices - 5,
        "high": prices + 10,
        "low": prices - 10,
        "close": prices,
        "volume": np.random.lognormal(10, 1, 30),  # Volumen también volátil
    }, index=dates)
    
    # Configurar la estrategia
    bb_strategy = BollingerBandsStrategy(period=10, std_dev=2.0)
    
    # Generar una señal
    signal = await bb_strategy.generate_signal("BTCUSDT", df)
    
    # Verificar las bandas
    middle_band = signal["metadata"]["middle_band"]
    upper_band = signal["metadata"]["upper_band"]
    lower_band = signal["metadata"]["lower_band"]
    
    # En períodos de alta volatilidad, las bandas deben estar separadas
    band_width = (upper_band - lower_band) / middle_band
    assert band_width > 0.2, f"Ancho de banda ({band_width}) debería ser mayor a 0.2 para alta volatilidad"
    
    # Verificar el tipo de señal basado en la posición del precio actual
    current_price = df["close"].iloc[-1]
    previous_price = df["close"].iloc[-2]
    
    # Las pruebas de señal dependen de dónde quede el precio al final, pero la lógica debe ser consistente
    if signal["type"] == SignalType.BUY:
        assert (current_price < lower_band * 1.05 or 
                (previous_price < lower_band and current_price > lower_band * 0.95))
    elif signal["type"] == SignalType.SELL:
        assert (current_price > upper_band * 0.95 or 
                (previous_price > upper_band and current_price < upper_band * 1.05))


@pytest.mark.asyncio
async def test_ma_crossover_strategy(sample_ohlcv_data):
    """Probar la estrategia de cruce de medias móviles."""
    # Configurar la estrategia
    ma_strategy = MovingAverageCrossover(fast_period=10, slow_period=30)
    
    # Generar una señal
    signal = await ma_strategy.generate_signal("BTCUSDT", sample_ohlcv_data)
    
    # Verificar que la señal tenga el formato correcto
    assert "type" in signal
    assert signal["type"] in [SignalType.BUY, SignalType.SELL, SignalType.HOLD, SignalType.EXIT]
    assert "reason" in signal
    assert "metadata" in signal
    
    # Verificar los datos de MA en los metadatos
    assert "fast_ma" in signal["metadata"]
    assert "slow_ma" in signal["metadata"]
    
    # Verificar lógica de cruce
    fast_ma = signal["metadata"]["fast_ma"]
    slow_ma = signal["metadata"]["slow_ma"]
    
    if signal["type"] == SignalType.BUY:
        # En un cruce alcista, la MA rápida debería superar a la lenta o está cerca de cruzarla
        assert fast_ma >= slow_ma * 0.95
        assert "cruce" in signal["reason"].lower() or "cross" in signal["reason"].lower()
    elif signal["type"] == SignalType.SELL:
        # En un cruce bajista, la MA rápida debería estar por debajo de la lenta o cerca
        assert fast_ma <= slow_ma * 1.05
        assert "cruce" in signal["reason"].lower() or "cross" in signal["reason"].lower()


@pytest.mark.asyncio
async def test_macd_strategy(sample_ohlcv_data):
    """Probar la estrategia MACD."""
    # Configurar la estrategia
    macd_strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    
    # Generar una señal
    signal = await macd_strategy.generate_signal("BTCUSDT", sample_ohlcv_data)
    
    # Verificar que la señal tenga el formato correcto
    assert "type" in signal
    assert signal["type"] in [SignalType.BUY, SignalType.SELL, SignalType.HOLD, SignalType.EXIT]
    assert "reason" in signal
    assert "metadata" in signal
    
    # Verificar los componentes del MACD en los metadatos
    assert "macd" in signal["metadata"]
    assert "signal_line" in signal["metadata"]
    assert "histogram" in signal["metadata"]
    
    # Verificar la lógica del MACD
    macd = signal["metadata"]["macd"]
    macd_signal = signal["metadata"]["signal_line"]
    histogram = signal["metadata"]["histogram"]
    
    # El histograma debería ser la diferencia entre MACD y línea de señal
    assert abs(histogram - (macd - macd_signal)) < 1e-6
    
    if signal["type"] == SignalType.BUY:
        # En una señal de compra, el MACD debería cruzar sobre la línea de señal
        assert histogram > 0
        assert "cruce" in signal["reason"].lower() or "cross" in signal["reason"].lower()
    elif signal["type"] == SignalType.SELL:
        # En una señal de venta, el MACD debería cruzar bajo la línea de señal
        assert histogram < 0
        assert "cruce" in signal["reason"].lower() or "cross" in signal["reason"].lower()


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
        
        # Verificar que la señal tenga el formato correcto
        assert "type" in signal
        assert signal["type"] in [SignalType.BUY, SignalType.SELL, SignalType.HOLD, SignalType.EXIT]
        assert "reason" in signal
        assert "metadata" in signal
        
        # Verificar el sentimiento en los metadatos
        assert "sentiment_score" in signal["metadata"]
        assert signal["metadata"]["sentiment_score"] == 0.5
        
        # Verificar la lógica basada en sentimiento
        if signal["type"] == SignalType.BUY:
            # En una señal de compra, el sentimiento debería ser positivo
            assert signal["metadata"]["sentiment_score"] > sentiment_strategy.sentiment_threshold
            assert "positivo" in signal["reason"].lower() or "positive" in signal["reason"].lower()
        elif signal["type"] == SignalType.SELL:
            # En una señal de venta, el sentimiento debería ser negativo
            assert signal["metadata"]["sentiment_score"] < -sentiment_strategy.sentiment_threshold
            assert "negativo" in signal["reason"].lower() or "negative" in signal["reason"].lower()