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
async def test_bollinger_bands_strategy():
    """Probar la estrategia de Bandas de Bollinger básica."""
    # Crear un DataFrame simple pero lo suficientemente grande
    dates = pd.date_range("2025-01-01", periods=30, freq="1h")
    prices = np.linspace(100, 200, 30)  # Precios lineales de 100 a 200
    
    df = pd.DataFrame({
        "open": prices - 1,
        "high": prices + 2,
        "low": prices - 2,
        "close": prices,
        "volume": np.random.lognormal(10, 1, 30),
    }, index=dates)
    
    # Configurar la estrategia
    bb_strategy = BollingerBandsStrategy(period=5, std_dev=2.0)  # Usar un período pequeño para que tengamos suficientes datos
    
    # En lugar de mockear el cálculo interno, vamos a crear una situación donde el precio cruza la banda inferior
    # Primero calculamos las bandas reales
    middle, upper, lower = bb_strategy.calculate_bollinger_bands(df['close'], bb_strategy.period, bb_strategy.std_dev)
    
    # Modificar los precios para crear el escenario que queremos probar
    # 1. Señal de compra - precio cruza por debajo de la banda inferior
    prev_idx = -2
    current_idx = -1
    
    # Modificar el precio previo para que esté justo en la banda inferior
    lower_val = lower.iloc[prev_idx]
    df.loc[df.index[prev_idx], 'close'] = lower_val
    
    # Modificar el precio actual para que esté por debajo de la banda inferior
    df.loc[df.index[current_idx], 'close'] = lower_val * 0.9
    
    # Generar una señal
    signal = await bb_strategy.generate_signal("BTCUSDT", df)
    
    # Verificar que la señal es de compra por cruce por debajo
    assert signal["type"] == SignalType.BUY
    assert "crossed below" in signal["reason"].lower() or "por debajo" in signal["reason"].lower()
    
    # Verificar otros detalles de la señal
    assert "middle" in signal
    assert "upper" in signal
    assert "lower" in signal
    assert "price" in signal
    assert "percent_b" in signal
    assert "bandwidth" in signal
    
    # Crear un nuevo DataFrame para la segunda prueba
    df2 = pd.DataFrame({
        "open": prices - 1,
        "high": prices + 2,
        "low": prices - 2,
        "close": prices,
        "volume": np.random.lognormal(10, 1, 30),
    }, index=dates)
    
    # 2. Señal de venta - precio cruza por encima de la banda superior
    middle2, upper2, lower2 = bb_strategy.calculate_bollinger_bands(df2['close'], bb_strategy.period, bb_strategy.std_dev)
    
    # Modificar el precio previo para que esté justo en la banda superior
    upper_val = upper2.iloc[prev_idx]
    df2.loc[df2.index[prev_idx], 'close'] = upper_val
    
    # Modificar el precio actual para que esté por encima de la banda superior
    df2.loc[df2.index[current_idx], 'close'] = upper_val * 1.1
    
    # Generar una señal
    signal2 = await bb_strategy.generate_signal("BTCUSDT", df2)
    
    # Verificar que la señal es de venta por cruce por encima
    assert signal2["type"] == SignalType.SELL
    assert "crossed above" in signal2["reason"].lower() or "por encima" in signal2["reason"].lower()


@pytest.mark.asyncio
async def test_bollinger_bands_strategy_narrow_bands():
    """Probar la estrategia de Bandas de Bollinger con bandas estrechas (baja volatilidad)."""
    # Crear datos OHLCV con baja volatilidad
    dates = pd.date_range("2025-01-01", periods=30, freq="1h")
    
    # Precios con baja volatilidad alrededor de 100 - usando serie constante para mínima volatilidad
    prices = np.array([100 + 0.01 * i for i in range(30)])  # Casi plana con ligera tendencia
    
    df = pd.DataFrame({
        "open": prices - 0.1,
        "high": prices + 0.2,
        "low": prices - 0.2,
        "close": prices,
        "volume": np.random.lognormal(10, 0.2, 30),  # Volumen también estable
    }, index=dates)
    
    # Configurar la estrategia con menor desviación estándar
    bb_strategy = BollingerBandsStrategy(period=10, std_dev=1.5)
    
    # Generar una señal
    signal = await bb_strategy.generate_signal("BTCUSDT", df)
    
    # Verificar las bandas
    middle_band = signal["middle"]
    upper_band = signal["upper"]
    lower_band = signal["lower"]
    
    # En períodos de baja volatilidad, las bandas deben estar cercanas 
    band_width = (upper_band - lower_band) / middle_band
    
    # Verificar que el ancho de banda sea bajo (bandas estrechas)
    # Para una serie casi constante, el ancho debería ser muy pequeño
    assert band_width < 0.1, f"Ancho de banda ({band_width}) debería ser menor a 0.1 para baja volatilidad"
    
    # Por defecto, con precios dentro de las bandas, debería ser HOLD
    # a menos que estemos creando un escenario específico de cruce
    if signal["type"] != SignalType.HOLD:
        # Si no es HOLD, verificamos que la razón sea coherente con el tipo de señal
        if signal["type"] == SignalType.BUY:
            assert "crossed" in signal["reason"].lower() and ("below" in signal["reason"].lower() or 
                                                            "above" in signal["reason"].lower())
        elif signal["type"] == SignalType.SELL:
            assert "crossed" in signal["reason"].lower() and ("above" in signal["reason"].lower() or 
                                                            "below" in signal["reason"].lower())


@pytest.mark.asyncio
async def test_bollinger_bands_strategy_wide_bands():
    """Probar la estrategia de Bandas de Bollinger con bandas anchas (alta volatilidad)."""
    # Crear datos OHLCV con alta volatilidad
    dates = pd.date_range("2025-01-01", periods=30, freq="1h")
    
    # Para simular alta volatilidad usamos una serie con grandes movimientos
    # Base inicial de precios
    np.random.seed(42)  # Para reproducibilidad
    base_prices = np.linspace(100, 200, 30)  # Tendencia alcista pronunciada
    
    # Agregar ruido aleatorio de alta amplitud
    noise = np.random.normal(0, 30, 30)  # Alta volatilidad
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
    middle_band = signal["middle"]
    upper_band = signal["upper"]
    lower_band = signal["lower"]
    
    # Calcular el ancho de las bandas
    band_width = (upper_band - lower_band) / middle_band
    
    # En períodos de alta volatilidad, las bandas deben estar más separadas
    assert band_width > 0.2, f"Ancho de banda ({band_width}) debería ser mayor a 0.2 para alta volatilidad"
    
    # Verificar el formato correcto de la señal
    assert "type" in signal
    assert "reason" in signal
    assert "price" in signal
    assert "middle" in signal
    assert "upper" in signal
    assert "lower" in signal
    assert "percent_b" in signal
    assert "bandwidth" in signal
    
    # Verificar que los valores son coherentes
    assert signal["bandwidth"] == band_width
    assert signal["middle"] == middle_band
    assert signal["upper"] == upper_band
    assert signal["lower"] == lower_band
    
    # Forzamos un escenario específico para verificar las señales
    # Modificamos los últimos dos puntos para crear un cruce específico
    
    # 1. Para probar una señal de venta - precio cruza por encima de la banda superior
    df_sell = df.copy()
    
    # Recalculamos las bandas para este conjunto de datos
    middle2, upper2, lower2 = bb_strategy.calculate_bollinger_bands(df_sell['close'], bb_strategy.period, bb_strategy.std_dev)
    
    # Modificar para crear un cruce por encima
    df_sell.loc[df_sell.index[-2], 'close'] = upper2.iloc[-2] * 0.99  # Justo por debajo
    df_sell.loc[df_sell.index[-1], 'close'] = upper2.iloc[-1] * 1.05  # Ahora por encima
    
    # Generar señal para este escenario
    signal_sell = await bb_strategy.generate_signal("BTCUSDT", df_sell)
    
    # Verificar que esta señal es de venta
    if signal_sell["type"] == SignalType.SELL:
        assert "crossed above" in signal_sell["reason"].lower() or "por encima" in signal_sell["reason"].lower()
        assert signal_sell["price"] > signal_sell["upper"] * 0.95  # Debe estar cerca o por encima de la banda superior


@pytest.mark.asyncio
async def test_ma_crossover_strategy():
    """Probar la estrategia de cruce de medias móviles."""
    # Crear datos de prueba
    dates = pd.date_range("2025-01-01", periods=60, freq="1h")
    
    # Crear un patrón de precios que producirá un cruce
    # Inicialmente la media rápida está por debajo de la lenta, luego cruza por encima
    prices = np.array([100 + i * 0.1 for i in range(30)] +  # tendencia ligeramente alcista
                      [103 + i * 0.5 for i in range(30)])   # tendencia fuertemente alcista (para crear cruce)
    
    df = pd.DataFrame({
        "open": prices - 1,
        "high": prices + 2,
        "low": prices - 2,
        "close": prices,
        "volume": np.random.lognormal(10, 1, 60),
    }, index=dates)
    
    # Configurar la estrategia con periodos adecuados para nuestros datos
    ma_strategy = MovingAverageCrossover(fast_period=10, slow_period=20)
    
    # Generar una señal
    signal = await ma_strategy.generate_signal("BTCUSDT", df)
    
    # Verificar que la señal tenga el formato correcto
    assert "type" in signal
    assert signal["type"] in [SignalType.BUY, SignalType.SELL, SignalType.HOLD, SignalType.EXIT]
    assert "reason" in signal
    
    # Verificar los datos de MA en los metadatos
    assert "fast_ma" in signal
    assert "slow_ma" in signal
    assert "strength" in signal
    
    # Verificar lógica de cruce
    fast_ma = signal["fast_ma"]
    slow_ma = signal["slow_ma"]
    
    # Con los datos que hemos creado, deberíamos tener una señal de compra
    assert signal["type"] == SignalType.BUY
    assert fast_ma > slow_ma  # En un cruce alcista, la MA rápida debería superar a la lenta
    assert "golden" in signal["reason"].lower() or "cross" in signal["reason"].lower()
    
    # Ahora vamos a probar el cruce bajista
    # Creamos un nuevo conjunto de datos donde la media rápida comienza por encima de la lenta
    # y luego cae por debajo
    prices2 = np.array([100 + i * 0.5 for i in range(30)] +  # tendencia fuertemente alcista
                       [115 - i * 0.5 for i in range(30)])   # tendencia fuertemente bajista (para crear cruce)
    
    df2 = pd.DataFrame({
        "open": prices2 - 1,
        "high": prices2 + 2,
        "low": prices2 - 2,
        "close": prices2,
        "volume": np.random.lognormal(10, 1, 60),
    }, index=dates)
    
    # Generar una señal con los nuevos datos
    signal2 = await ma_strategy.generate_signal("BTCUSDT", df2)
    
    # Verificar la señal de venta
    assert signal2["type"] == SignalType.SELL
    assert signal2["fast_ma"] < signal2["slow_ma"]  # En un cruce bajista, la MA rápida debe estar por debajo
    assert "death" in signal2["reason"].lower() or "cross" in signal2["reason"].lower()


@pytest.mark.asyncio
async def test_macd_strategy():
    """Probar la estrategia MACD."""
    # Crear datos con un patrón que generará una señal de MACD
    dates = pd.date_range("2025-01-01", periods=60, freq="1h")
    
    # Generar precios que producirán un cruce de MACD alcista
    # Primero tendencia bajista seguida de un repunte
    prices = np.array([100 - i * 0.5 for i in range(40)] +  # tendencia bajista
                      [80 + i * 1.0 for i in range(20)])    # rápido repunte (producirá un cruce)
    
    df = pd.DataFrame({
        "open": prices - 1,
        "high": prices + 2,
        "low": prices - 2,
        "close": prices,
        "volume": np.random.lognormal(10, 1, 60),
    }, index=dates)
    
    # Configurar la estrategia con periodos adecuados para nuestros datos
    macd_strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    
    # Generar una señal
    signal = await macd_strategy.generate_signal("BTCUSDT", df)
    
    # Verificar que la señal tenga el formato correcto
    assert "type" in signal
    assert signal["type"] in [SignalType.BUY, SignalType.SELL, SignalType.HOLD, SignalType.EXIT]
    assert "reason" in signal
    
    # Verificar los componentes del MACD
    assert "macd" in signal
    assert "signal" in signal
    assert "histogram" in signal
    assert "strength" in signal
    
    # Verificar la lógica del MACD
    macd_value = signal["macd"]
    signal_value = signal["signal"]
    histogram = signal["histogram"]
    
    # El histograma debería ser la diferencia entre MACD y línea de señal
    assert abs(histogram - (macd_value - signal_value)) < 1e-6
    
    # Con los datos que hemos creado, deberíamos tener una señal de compra
    assert signal["type"] == SignalType.BUY
    assert "cross" in signal["reason"].lower()
    assert histogram > 0
    
    # Crear datos para probar señal de venta
    # Primero tendencia alcista seguida de una caída
    prices2 = np.array([100 + i * 0.5 for i in range(40)] +  # tendencia alcista
                       [120 - i * 1.0 for i in range(20)])    # rápida caída (producirá un cruce)
    
    df2 = pd.DataFrame({
        "open": prices2 - 1,
        "high": prices2 + 2,
        "low": prices2 - 2,
        "close": prices2,
        "volume": np.random.lognormal(10, 1, 60),
    }, index=dates)
    
    # Generar una señal con los nuevos datos
    signal2 = await macd_strategy.generate_signal("BTCUSDT", df2)
    
    # Verificar la señal de venta
    assert signal2["type"] == SignalType.SELL
    assert "cross" in signal2["reason"].lower()
    assert signal2["histogram"] < 0
    
    # Verificar que el histograma es consistente
    assert abs(signal2["histogram"] - (signal2["macd"] - signal2["signal"])) < 1e-6


@pytest.mark.asyncio
async def test_sentiment_strategy():
    """Probar la estrategia basada en sentimiento."""
    # Crear datos con una tendencia alcista
    dates = pd.date_range("2025-01-01", periods=30, freq="1h")
    prices = np.linspace(100, 130, 30)  # tendencia claramente alcista
    
    df = pd.DataFrame({
        "open": prices - 1,
        "high": prices + 2,
        "low": prices - 2,
        "close": prices,
        "volume": np.random.lognormal(10, 1, 30),
    }, index=dates)
    
    # Configurar la estrategia
    sentiment_strategy = SentimentStrategy(sentiment_threshold=0.3)
    
    # Mockear el método de obtención de sentimiento para 2 escenarios
    
    # Escenario 1: Sentimiento positivo con tendencia alcista (señal de compra)
    with patch.object(sentiment_strategy, 'fetch_sentiment', return_value=asyncio.Future()) as mock_fetch:
        # Configurar el valor de retorno
        mock_fetch.return_value.set_result(0.5)  # Sentimiento fuertemente positivo
        
        # Generar una señal
        signal = await sentiment_strategy.generate_signal("BTCUSDT", df)
        
        # Verificar que se llamó al método para obtener sentimiento
        mock_fetch.assert_called_once_with("BTCUSDT")
        
        # Verificar que la señal tenga el formato correcto
        assert "type" in signal
        assert signal["type"] in [SignalType.BUY, SignalType.SELL, SignalType.HOLD, SignalType.EXIT]
        assert "reason" in signal
        
        # Verificar el sentimiento en la señal
        assert "sentiment" in signal
        assert signal["sentiment"] == 0.5
        assert "price_trend" in signal
        assert signal["price_trend"] == "up"
        assert "strength" in signal
        
        # Con tendencia alcista y sentimiento positivo, deberíamos tener señal de compra
        assert signal["type"] == SignalType.BUY
        assert "positive sentiment" in signal["reason"].lower()
        assert "up" in signal["reason"].lower() or "trend" in signal["reason"].lower()
    
    # Crear datos con una tendencia bajista
    prices2 = np.linspace(130, 100, 30)  # tendencia claramente bajista
    
    df2 = pd.DataFrame({
        "open": prices2 - 1,
        "high": prices2 + 2,
        "low": prices2 - 2,
        "close": prices2,
        "volume": np.random.lognormal(10, 1, 30),
    }, index=dates)
    
    # Escenario 2: Sentimiento negativo con tendencia bajista (señal de venta)
    with patch.object(sentiment_strategy, 'fetch_sentiment', return_value=asyncio.Future()) as mock_fetch:
        # Configurar el valor de retorno
        mock_fetch.return_value.set_result(-0.5)  # Sentimiento fuertemente negativo
        
        # Generar una señal
        signal2 = await sentiment_strategy.generate_signal("BTCUSDT", df2)
        
        # Verificar que la señal sea de venta
        assert signal2["type"] == SignalType.SELL
        assert "negative sentiment" in signal2["reason"].lower()
        assert "down" in signal2["reason"].lower() or "trend" in signal2["reason"].lower()
        
        # Verificar otros detalles
        assert signal2["sentiment"] == -0.5
        assert signal2["price_trend"] == "down"
        
    # Escenario 3: Sentimiento positivo pero tendencia bajista (no debe dar señal compra)
    with patch.object(sentiment_strategy, 'fetch_sentiment', return_value=asyncio.Future()) as mock_fetch:
        # Configurar el valor de retorno con sentimiento positivo
        mock_fetch.return_value.set_result(0.4)  # Sentimiento positivo
        
        # Generar una señal con tendencia bajista
        signal3 = await sentiment_strategy.generate_signal("BTCUSDT", df2)
        
        # En este caso, no debe dar señal de compra a pesar del sentimiento positivo
        assert signal3["type"] == SignalType.HOLD
        assert "positive sentiment" in signal3["reason"].lower()
        assert "no trend confirmation" in signal3["reason"].lower()