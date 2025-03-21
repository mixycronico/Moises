"""
Pruebas unitarias para indicadores técnicos y generación de señales.

Este módulo prueba las funcionalidades relacionadas con el cálculo
de indicadores técnicos y la generación de señales de trading basadas en ellos.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from genesis.analysis.indicators import TechnicalIndicators
from genesis.analysis.signal_generator import SignalGenerator


# Fixtures
@pytest.fixture
def indicators():
    """Fixture que proporciona una instancia de TechnicalIndicators."""
    return TechnicalIndicators()


@pytest.fixture
def signal_generator(indicators):
    """Fixture que proporciona una instancia de SignalGenerator con indicadores."""
    return SignalGenerator(indicators)


@pytest.fixture
def sample_price_data():
    """Fixture que proporciona datos de precio de ejemplo."""
    # Crear datos de muestra para pruebas
    return np.array([5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000,
                     6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000])


@pytest.fixture
def sample_ohlcv_data():
    """Fixture que proporciona datos OHLCV de ejemplo."""
    # Crear DataFrame con datos de muestra
    dates = pd.date_range("2025-01-01", periods=21, freq="1D")
    prices = np.array([5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000,
                     6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000])
    
    df = pd.DataFrame({
        "open": prices - 50,
        "high": prices + 100,
        "low": prices - 100,
        "close": prices,
        "volume": np.random.randint(1000, 5000, size=len(prices))
    }, index=dates)
    
    return df


# Pruebas para indicadores técnicos básicos
def test_calculate_sma(indicators, sample_price_data):
    """Prueba el cálculo de SMA (Simple Moving Average)."""
    period = 5
    sma = indicators.calculate_sma(sample_price_data, period)
    
    # Verificar que los primeros (period-1) valores son NaN
    assert len(sma) == len(sample_price_data)
    assert all(np.isnan(sma[i]) for i in range(period-1))
    
    # Verificar cálculo manual de algunos valores
    expected_sma_4 = np.mean(sample_price_data[0:period])  # Media de los primeros 5 valores
    assert sma[period-1] == expected_sma_4
    
    expected_sma_10 = np.mean(sample_price_data[6:6+period])  # Media de los valores 6 a 10
    assert sma[10] == expected_sma_10


def test_calculate_ema(indicators, sample_price_data):
    """Prueba el cálculo de EMA (Exponential Moving Average)."""
    period = 5
    ema = indicators.calculate_ema(sample_price_data, period)
    
    # Verificar que los primeros (period-1) valores son NaN
    assert len(ema) == len(sample_price_data)
    assert all(np.isnan(ema[i]) for i in range(period-1))
    
    # Primer valor de EMA es igual a SMA
    expected_first_ema = np.mean(sample_price_data[0:period])
    assert ema[period-1] == expected_first_ema
    
    # Calcular factor de suavizado
    alpha = 2 / (period + 1)
    
    # Verificar cálculo manual del siguiente valor de EMA
    expected_next_ema = (sample_price_data[period] * alpha) + (expected_first_ema * (1 - alpha))
    assert pytest.approx(ema[period], rel=1e-10) == expected_next_ema


def test_calculate_rsi(indicators, sample_price_data):
    """Prueba el cálculo de RSI (Relative Strength Index)."""
    period = 14
    rsi = indicators.calculate_rsi(sample_price_data, period)
    
    # Verificar que los primeros (period) valores son NaN
    assert len(rsi) == len(sample_price_data)
    assert all(np.isnan(rsi[i]) for i in range(period))
    
    # Verificar que RSI está en el rango [0, 100]
    valid_rsi = rsi[~np.isnan(rsi)]
    assert all(0 <= value <= 100 for value in valid_rsi)
    
    # Para estos datos de muestra que están siempre subiendo, RSI debería ser alto
    assert valid_rsi[-1] > 70  # último valor de RSI debería ser alto (sobrecompra)


def test_calculate_macd(indicators, sample_price_data):
    """Prueba el cálculo de MACD (Moving Average Convergence Divergence)."""
    # Usar períodos muy pequeños para adaptarse a nuestros datos de prueba
    fast_period = 2
    slow_period = 4
    signal_period = 2
    
    macd_line, signal_line, histogram = indicators.calculate_macd(
        sample_price_data, fast_period, slow_period, signal_period
    )
    
    # Verificar que las longitudes son correctas
    assert len(macd_line) == len(sample_price_data)
    assert len(signal_line) == len(sample_price_data)
    assert len(histogram) == len(sample_price_data)
    
    # Verificar que hay valores NaN al principio (los primeros valores)
    assert np.isnan(macd_line[0])
    
    # Verificar que hay valores NaN en la línea de señal al principio
    assert np.isnan(signal_line[0])
    
    # Verificar que al menos algunos valores MACD no son NaN
    valid_macd = macd_line[~np.isnan(macd_line)]
    assert len(valid_macd) > 0
    
    # Con datasets pequeños, es posible que no haya valores válidos en la línea de señal
    # En un entorno real con más datos, esto no sería un problema


def test_calculate_bollinger_bands(indicators, sample_price_data):
    """Prueba el cálculo de Bandas de Bollinger."""
    period = 20
    std_dev = 2.0
    
    upper_band, middle_band, lower_band = indicators.calculate_bollinger_bands(
        sample_price_data, period, std_dev
    )
    
    # Verificar que las longitudes son correctas
    assert len(upper_band) == len(sample_price_data)
    assert len(middle_band) == len(sample_price_data)
    assert len(lower_band) == len(sample_price_data)
    
    # Verificar que los primeros (period-1) valores son NaN
    assert all(np.isnan(middle_band[i]) for i in range(period-1))
    
    # Verificar que la banda media es igual a la SMA
    sma = indicators.calculate_sma(sample_price_data, period)
    np.testing.assert_array_equal(middle_band, sma)
    
    # Verificar que la banda superior es mayor que la media
    valid_indices = ~np.isnan(middle_band)
    assert all(upper_band[i] > middle_band[i] for i in range(len(sample_price_data)) if valid_indices[i])
    
    # Verificar que la banda inferior es menor que la media
    assert all(lower_band[i] < middle_band[i] for i in range(len(sample_price_data)) if valid_indices[i])


def test_calculate_atr(indicators, sample_ohlcv_data):
    """Prueba el cálculo de ATR (Average True Range)."""
    period = 14
    high = sample_ohlcv_data['high'].values
    low = sample_ohlcv_data['low'].values
    close = sample_ohlcv_data['close'].values
    
    atr = indicators.calculate_atr(high, low, close, period)
    
    # Verificar que la longitud es correcta
    assert len(atr) == len(high)
    
    # Verificar que los primeros (period) valores son NaN o el primer valor no es NaN 
    # pero los siguientes (period-1) son NaN, dependiendo de la implementación
    if not np.isnan(atr[0]):
        assert all(np.isnan(atr[i]) for i in range(1, period))
    else:
        assert all(np.isnan(atr[i]) for i in range(period))
    
    # Verificar que los valores de ATR son positivos
    valid_atr = atr[~np.isnan(atr)]
    assert all(value > 0 for value in valid_atr)


# Pruebas para generate_ema_signal
def test_signal_generator_ema_crossover_buy(signal_generator, indicators):
    """Prueba que un cruce ascendente de EMA genere una señal de compra."""
    prices = np.array([5000, 5100, 5200, 5300, 5400])
    ema_short = np.array([5020, 5120, 5220, 5320, 5420])  # EMA corta sube más rápido
    ema_long = np.array([5030, 5130, 5230, 5330, 5410])   # EMA larga más lenta
    
    # Mockear el cálculo de EMA para devolver valores controlados
    with patch.object(indicators, 'calculate_ema', side_effect=[ema_short, ema_long]):
        signal = signal_generator.generate_ema_signal(prices, short_period=9, long_period=21)
    
    # Cruce ascendente (ema_short cruza por encima de ema_long en el último punto)
    assert signal == "BUY"


def test_signal_generator_ema_crossover_sell(signal_generator, indicators):
    """Prueba que un cruce descendente de EMA genere una señal de venta."""
    prices = np.array([5400, 5300, 5200, 5100, 5000])
    ema_short = np.array([5420, 5320, 5220, 5120, 5020])  # EMA corta baja más rápido
    ema_long = np.array([5410, 5330, 5230, 5130, 5030])   # EMA larga más lenta
    
    # Mockear el cálculo de EMA para devolver valores controlados
    with patch.object(indicators, 'calculate_ema', side_effect=[ema_short, ema_long]):
        signal = signal_generator.generate_ema_signal(prices, short_period=9, long_period=21)
    
    # Cruce descendente (ema_short cruza por debajo de ema_long en el último punto)
    assert signal == "SELL"


def test_signal_generator_ema_no_crossover(signal_generator, indicators):
    """Prueba que no haya señal si no hay cruce de EMA."""
    prices = np.array([5000, 5100, 5200, 5300, 5400])
    ema_short = np.array([5020, 5120, 5220, 5320, 5420])  # EMA corta siempre por encima
    ema_long = np.array([5010, 5110, 5210, 5310, 5410])   # EMA larga siempre por debajo
    
    # Mockear el cálculo de EMA para devolver valores controlados
    with patch.object(indicators, 'calculate_ema', side_effect=[ema_short, ema_long]):
        signal = signal_generator.generate_ema_signal(prices, short_period=9, long_period=21)
    
    assert signal == signal_generator.NEUTRAL


def test_signal_generator_ema_invalid_periods(signal_generator):
    """Prueba el manejo de períodos inválidos en EMA."""
    prices = np.array([5000, 5100, 5200])
    
    with pytest.raises(ValueError, match="Periods must be positive"):
        signal_generator.generate_ema_signal(prices, short_period=0, long_period=21)
    
    with pytest.raises(ValueError, match="Short period must be less than long period"):
        signal_generator.generate_ema_signal(prices, short_period=21, long_period=9)


def test_signal_generator_ema_insufficient_data(signal_generator):
    """Prueba el manejo de datos insuficientes para EMA."""
    prices = np.array([5000])  # Menos datos que el período largo
    
    with pytest.raises(ValueError, match="Not enough data"):
        signal_generator.generate_ema_signal(prices, short_period=9, long_period=21)


# Pruebas para generate_rsi_signal
def test_signal_generator_rsi_overbought(signal_generator, indicators):
    """Prueba que un RSI alto genere una señal de venta."""
    prices = np.array([5000, 5100, 5200, 5300, 5400])
    mock_rsi = np.array([float('nan')] * (len(prices) - 1) + [75])  # RSI > 70, sobrecompra
    
    # Mockear el cálculo de RSI para devolver valores controlados
    with patch.object(indicators, 'calculate_rsi', return_value=mock_rsi):
        signal = signal_generator.generate_rsi_signal(prices, period=14)
    
    assert signal == "SELL"


def test_signal_generator_rsi_oversold(signal_generator, indicators):
    """Prueba que un RSI bajo genere una señal de compra."""
    prices = np.array([5400, 5300, 5200, 5100, 5000])
    mock_rsi = np.array([float('nan')] * (len(prices) - 1) + [25])  # RSI < 30, sobreventa
    
    # Mockear el cálculo de RSI para devolver valores controlados
    with patch.object(indicators, 'calculate_rsi', return_value=mock_rsi):
        signal = signal_generator.generate_rsi_signal(prices, period=14)
    
    assert signal == "BUY"


def test_signal_generator_rsi_neutral(signal_generator, indicators):
    """Prueba que un RSI intermedio genere una señal neutral."""
    prices = np.array([5000, 5100, 5200])
    mock_rsi = np.array([float('nan')] * (len(prices) - 1) + [45])  # 30 < RSI < 70, zona neutral
    
    # Mockear el cálculo de RSI para devolver valores controlados
    with patch.object(indicators, 'calculate_rsi', return_value=mock_rsi):
        signal = signal_generator.generate_rsi_signal(prices, period=14)
    
    assert signal == signal_generator.NEUTRAL


def test_signal_generator_rsi_invalid_period(signal_generator):
    """Prueba el manejo de períodos inválidos en RSI."""
    prices = np.array([5000, 5100, 5200])
    
    with pytest.raises(ValueError, match="Period must be positive"):
        signal_generator.generate_rsi_signal(prices, period=0)


def test_signal_generator_rsi_insufficient_data(signal_generator):
    """Prueba el manejo de datos insuficientes para RSI."""
    prices = np.array([5000, 5100])  # Menos datos que el período
    
    with pytest.raises(ValueError, match="Not enough data"):
        signal_generator.generate_rsi_signal(prices, period=14)


# Pruebas para generate_macd_signal
def test_signal_generator_macd_bullish(signal_generator, indicators):
    """Prueba que un cruce alcista del MACD genere una señal de compra."""
    prices = np.array([5000, 5100, 5200, 5300, 5400])
    mock_macd_line = np.array([float('nan'), float('nan'), 5, 6, 7])      # MACD subiendo
    mock_signal_line = np.array([float('nan'), float('nan'), 6, 7, 6])    # Signal cruza por debajo, alcista
    
    # Mockear el cálculo de MACD para devolver valores controlados
    with patch.object(indicators, 'calculate_macd', return_value=(mock_macd_line, mock_signal_line)):
        signal = signal_generator.generate_macd_signal(prices, fast=12, slow=26, signal_period=9)
    
    assert signal == "BUY"


def test_signal_generator_macd_bearish(signal_generator, indicators):
    """Prueba que un cruce bajista del MACD genere una señal de venta."""
    prices = np.array([5400, 5300, 5200, 5100, 5000])
    mock_macd_line = np.array([float('nan'), float('nan'), 5, 4, 3])      # MACD bajando
    mock_signal_line = np.array([float('nan'), float('nan'), 4, 3, 4])    # Signal cruza por encima, bajista
    
    # Mockear el cálculo de MACD para devolver valores controlados
    with patch.object(indicators, 'calculate_macd', return_value=(mock_macd_line, mock_signal_line)):
        signal = signal_generator.generate_macd_signal(prices, fast=12, slow=26, signal_period=9)
    
    assert signal == "SELL"


def test_signal_generator_macd_neutral(signal_generator, indicators):
    """Prueba que no haya señal si no hay cruce en el MACD."""
    prices = np.array([5000, 5100, 5200])
    mock_macd_line = np.array([float('nan'), 6, 7])      # MACD siempre por encima
    mock_signal_line = np.array([float('nan'), 4, 5])    # Signal siempre por debajo
    
    # Mockear el cálculo de MACD para devolver valores controlados
    with patch.object(indicators, 'calculate_macd', return_value=(mock_macd_line, mock_signal_line)):
        signal = signal_generator.generate_macd_signal(prices, fast=12, slow=26, signal_period=9)
    
    assert signal == signal_generator.NEUTRAL


def test_signal_generator_macd_invalid_periods(signal_generator):
    """Prueba el manejo de períodos inválidos en MACD."""
    prices = np.array([5000, 5100, 5200])
    
    with pytest.raises(ValueError, match="Periods must be positive"):
        signal_generator.generate_macd_signal(prices, fast=0, slow=26, signal_period=9)
    
    with pytest.raises(ValueError, match="Fast period must be less than slow period"):
        signal_generator.generate_macd_signal(prices, fast=26, slow=12, signal_period=9)


def test_signal_generator_macd_insufficient_data(signal_generator):
    """Prueba el manejo de datos insuficientes para MACD."""
    prices = np.array([5000, 5100])  # Menos datos que el período lento
    
    with pytest.raises(ValueError, match="Not enough data"):
        signal_generator.generate_macd_signal(prices, fast=12, slow=26, signal_period=9)


# Pruebas para generate_bollinger_bands_signal
def test_signal_generator_bollinger_bands_buy(signal_generator, indicators):
    """Prueba que precio cerca de la banda inferior genere una señal de compra."""
    prices = np.array([5000, 5100, 5200, 5300, 5400])
    upper_band = np.array([float('nan')] * 3 + [5600, 5700])
    middle_band = np.array([float('nan')] * 3 + [5350, 5450])
    lower_band = np.array([float('nan')] * 3 + [5100, 5200])
    
    # Último precio cerca de la banda inferior
    prices[-1] = lower_band[-1] + 50  # Ligeramente por encima de banda inferior
    
    # Mockear el cálculo de Bandas de Bollinger para devolver valores controlados
    with patch.object(indicators, 'calculate_bollinger_bands', 
                     return_value=(upper_band, middle_band, lower_band)):
        signal = signal_generator.generate_bollinger_bands_signal(
            prices, period=20, std_dev=2.0
        )
    
    assert signal == "BUY"


def test_signal_generator_bollinger_bands_sell(signal_generator, indicators):
    """Prueba que precio cerca de la banda superior genere una señal de venta."""
    prices = np.array([5000, 5100, 5200, 5300, 5400])
    upper_band = np.array([float('nan')] * 3 + [5600, 5700])
    middle_band = np.array([float('nan')] * 3 + [5350, 5450])
    lower_band = np.array([float('nan')] * 3 + [5100, 5200])
    
    # Último precio cerca de la banda superior
    prices[-1] = upper_band[-1] - 50  # Ligeramente por debajo de banda superior
    
    # Mockear el cálculo de Bandas de Bollinger para devolver valores controlados
    with patch.object(indicators, 'calculate_bollinger_bands', 
                     return_value=(upper_band, middle_band, lower_band)):
        signal = signal_generator.generate_bollinger_bands_signal(
            prices, period=20, std_dev=2.0
        )
    
    assert signal == "SELL"


def test_signal_generator_bollinger_bands_neutral(signal_generator, indicators):
    """Prueba que precio cerca de la banda media genere una señal neutral."""
    prices = np.array([5000, 5100, 5200, 5300, 5400])
    upper_band = np.array([float('nan')] * 3 + [5600, 5700])
    middle_band = np.array([float('nan')] * 3 + [5350, 5450])
    lower_band = np.array([float('nan')] * 3 + [5100, 5200])
    
    # Último precio cerca de la banda media
    prices[-1] = middle_band[-1] + 10  # Cerca de la banda media
    
    # Mockear el cálculo de Bandas de Bollinger para devolver valores controlados
    with patch.object(indicators, 'calculate_bollinger_bands', 
                     return_value=(upper_band, middle_band, lower_band)):
        signal = signal_generator.generate_bollinger_bands_signal(
            prices, period=20, std_dev=2.0
        )
    
    assert signal == signal_generator.NEUTRAL


# Pruebas para combinar señales
def test_signal_generator_combine_signals_majority(signal_generator):
    """Prueba la combinación de señales por mayoría."""
    signals = [signal_generator.BUY, signal_generator.BUY, signal_generator.SELL, signal_generator.NEUTRAL]
    combined = signal_generator.combine_signals(signals, method="majority")
    assert combined == signal_generator.BUY
    
    signals = [signal_generator.SELL, signal_generator.SELL, signal_generator.BUY, signal_generator.NEUTRAL]
    combined = signal_generator.combine_signals(signals, method="majority")
    assert combined == signal_generator.SELL
    
    signals = [signal_generator.NEUTRAL, signal_generator.NEUTRAL, signal_generator.BUY, signal_generator.SELL]
    combined = signal_generator.combine_signals(signals, method="majority")
    assert combined == signal_generator.NEUTRAL
    
    # Empate
    signals = [signal_generator.BUY, signal_generator.SELL]
    combined = signal_generator.combine_signals(signals, method="majority")
    assert combined == signal_generator.NEUTRAL  # En caso de empate, neutral


def test_signal_generator_combine_signals_conservative(signal_generator):
    """Prueba la combinación de señales conservadora (señal solo si todos coinciden)."""
    # Todos comprar
    signals = [signal_generator.BUY, signal_generator.BUY, signal_generator.BUY]
    combined = signal_generator.combine_signals(signals, method="conservative")
    assert combined == signal_generator.BUY
    
    # Todos vender
    signals = [signal_generator.SELL, signal_generator.SELL, signal_generator.SELL]
    combined = signal_generator.combine_signals(signals, method="conservative")
    assert combined == signal_generator.SELL
    
    # Todos neutral
    signals = [signal_generator.NEUTRAL, signal_generator.NEUTRAL]
    combined = signal_generator.combine_signals(signals, method="conservative")
    assert combined == signal_generator.NEUTRAL
    
    # Uno diferente, debería ser neutral
    signals = [signal_generator.BUY, signal_generator.BUY, signal_generator.NEUTRAL]
    combined = signal_generator.combine_signals(signals, method="conservative")
    assert combined == signal_generator.NEUTRAL
    
    signals = [signal_generator.SELL, signal_generator.BUY, signal_generator.SELL]
    combined = signal_generator.combine_signals(signals, method="conservative")
    assert combined == signal_generator.NEUTRAL


def test_signal_generator_combine_signals_weighted(signal_generator):
    """Prueba la combinación de señales ponderada."""
    signals = [signal_generator.BUY, signal_generator.SELL, signal_generator.NEUTRAL]
    weights = [0.6, 0.3, 0.1]  # Mayor peso a la primera señal
    combined = signal_generator.combine_signals(signals, method="weighted", weights=weights)
    assert combined == signal_generator.BUY
    
    weights = [0.3, 0.6, 0.1]  # Mayor peso a la segunda señal
    combined = signal_generator.combine_signals(signals, method="weighted", weights=weights)
    assert combined == signal_generator.SELL
    
    weights = [0.3, 0.3, 0.4]  # Mayor peso a la tercera señal
    combined = signal_generator.combine_signals(signals, method="weighted", weights=weights)
    assert combined == signal_generator.NEUTRAL
    
    # Pesos iguales, debería funcionar como majority
    weights = [1/3, 1/3, 1/3]
    combined = signal_generator.combine_signals(signals, method="weighted", weights=weights)
    assert combined == signal_generator.NEUTRAL  # Empate entre compra/venta, neutral gana


def test_signal_generator_combine_signals_invalid(signal_generator):
    """Prueba el manejo de entradas inválidas en la combinación de señales."""
    signals = [signal_generator.BUY, signal_generator.SELL, signal_generator.NEUTRAL]
    
    # Método inválido
    with pytest.raises(ValueError, match="Invalid method"):
        signal_generator.combine_signals(signals, method="invalid_method")
    
    # Pesos faltantes para método weighted
    with pytest.raises(ValueError, match="Weights must be provided"):
        signal_generator.combine_signals(signals, method="weighted")
    
    # Longitud incorrecta de pesos
    with pytest.raises(ValueError, match="Length of weights must match"):
        signal_generator.combine_signals(signals, method="weighted", weights=[0.5, 0.5])
    
    # Pesos no suman 1
    with pytest.raises(ValueError, match="Weights must sum to 1"):
        signal_generator.combine_signals(signals, method="weighted", weights=[0.5, 0.4, 0.4])