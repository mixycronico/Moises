"""
Tests básicos para el Data Manager.

Este módulo prueba las funcionalidades básicas del Data Manager,
incluyendo la obtención de datos históricos, procesamiento de datos
en tiempo real y almacenamiento en caché.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from genesis.data.manager import DataManager
from genesis.data.providers.base import DataProvider
from genesis.data.cache import DataCache


class MockDataProvider(DataProvider):
    """Proveedor de datos simulado para pruebas."""
    
    def __init__(self, name="mock_provider"):
        """Inicializar proveedor con datos simulados."""
        super().__init__(name)
        self.available_symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        self.available_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Datos OHLCV simulados
        self.ohlcv_data = {
            "BTC/USDT": {
                "1h": self._generate_mock_ohlcv(24),  # 24 horas de datos
                "1d": self._generate_mock_ohlcv(30)   # 30 días de datos
            },
            "ETH/USDT": {
                "1h": self._generate_mock_ohlcv(24),
                "1d": self._generate_mock_ohlcv(30)
            }
        }
        
        # Datos de ticker simulados
        self.ticker_data = {
            "BTC/USDT": {"last": 50000, "bid": 49900, "ask": 50100, "volume": 100},
            "ETH/USDT": {"last": 3000, "bid": 2990, "ask": 3010, "volume": 200},
            "XRP/USDT": {"last": 0.5, "bid": 0.49, "ask": 0.51, "volume": 1000000}
        }
    
    def _generate_mock_ohlcv(self, n_candles, base_price=50000):
        """Generar datos OHLCV simulados."""
        data = []
        timestamp = 1609459200000  # 1 de enero de 2021 00:00:00 UTC
        
        for i in range(n_candles):
            # Timestamp, Open, High, Low, Close, Volume
            open_price = base_price + i * 100  # Precio ascendente
            close_price = open_price + 50      # Cierre ligeramente más alto
            high_price = close_price + 20      # Alto ligeramente más alto que el cierre
            low_price = open_price - 20        # Bajo ligeramente más bajo que la apertura
            volume = 10 + i                    # Volumen creciente
            
            data.append([
                timestamp + i * 3600000,  # +1 hora por cada vela
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            ])
        
        return data
    
    async def get_historical_ohlcv(self, symbol, timeframe, since=None, limit=None):
        """Obtener datos históricos OHLCV."""
        if symbol not in self.ohlcv_data or timeframe not in self.ohlcv_data[symbol]:
            return []
        
        data = self.ohlcv_data[symbol][timeframe]
        
        # Aplicar limit si se especifica
        if limit is not None and limit > 0:
            data = data[-limit:]
            
        return data
    
    async def get_ticker(self, symbol):
        """Obtener datos de ticker."""
        if symbol not in self.ticker_data:
            return None
            
        return self.ticker_data[symbol]
    
    async def get_available_symbols(self):
        """Obtener símbolos disponibles."""
        return self.available_symbols
    
    async def get_available_timeframes(self):
        """Obtener marcos temporales disponibles."""
        return self.available_timeframes


@pytest.fixture
def mock_provider():
    """Proporcionar un proveedor de datos simulado."""
    return MockDataProvider()


@pytest.fixture
def data_cache():
    """Proporcionar una caché de datos."""
    return DataCache()


@pytest.fixture
def data_manager(mock_provider, data_cache):
    """Proporcionar un gestor de datos con proveedor simulado."""
    manager = DataManager()
    manager.register_provider(mock_provider)
    manager.set_cache(data_cache)
    return manager


@pytest.mark.asyncio
async def test_data_manager_initialization(data_manager):
    """Probar inicialización básica del Data Manager."""
    # Verificar que el manager se crea correctamente
    assert data_manager is not None
    
    # Verificar que el proveedor está registrado
    assert len(data_manager.providers) == 1
    assert "mock_provider" in data_manager.providers
    
    # Verificar que la caché está configurada
    assert data_manager.cache is not None


@pytest.mark.asyncio
async def test_get_historical_data_basic(data_manager):
    """Probar obtención básica de datos históricos."""
    # Obtener datos históricos
    data = await data_manager.get_historical_data("BTC/USDT", "1h", limit=10)
    
    # Verificar estructura de datos
    assert len(data) == 10
    assert len(data[0]) == 6  # Timestamp, Open, High, Low, Close, Volume
    
    # Verificar que los datos son coherentes
    for candle in data:
        timestamp, open_price, high, low, close, volume = candle
        assert isinstance(timestamp, (int, float))
        assert isinstance(open_price, (int, float))
        assert isinstance(high, (int, float))
        assert isinstance(low, (int, float))
        assert isinstance(close, (int, float))
        assert isinstance(volume, (int, float))
        
        # Verificar que high >= open, close, low y low <= open, close
        assert high >= open_price
        assert high >= close
        assert low <= open_price
        assert low <= close


@pytest.mark.asyncio
async def test_get_ticker_data(data_manager):
    """Probar obtención de datos de ticker."""
    # Obtener ticker para BTC/USDT
    ticker = await data_manager.get_ticker("BTC/USDT")
    
    # Verificar estructura
    assert "last" in ticker
    assert "bid" in ticker
    assert "ask" in ticker
    assert "volume" in ticker
    
    # Verificar coherencia
    assert ticker["bid"] <= ticker["last"] <= ticker["ask"]


@pytest.mark.asyncio
async def test_get_available_symbols(data_manager):
    """Probar obtención de símbolos disponibles."""
    symbols = await data_manager.get_available_symbols()
    
    # Verificar que se devuelven los símbolos esperados
    assert "BTC/USDT" in symbols
    assert "ETH/USDT" in symbols
    assert "XRP/USDT" in symbols


@pytest.mark.asyncio
async def test_get_available_timeframes(data_manager):
    """Probar obtención de marcos temporales disponibles."""
    timeframes = await data_manager.get_available_timeframes()
    
    # Verificar que se devuelven los timeframes esperados
    assert "1m" in timeframes
    assert "1h" in timeframes
    assert "1d" in timeframes


@pytest.mark.asyncio
async def test_cache_functionality(data_manager, mock_provider):
    """Probar funcionamiento de la caché."""
    # Espiar el método get_historical_ohlcv del proveedor
    with patch.object(mock_provider, 'get_historical_ohlcv', wraps=mock_provider.get_historical_ohlcv) as spy:
        # Primera llamada (debería ir al proveedor)
        data1 = await data_manager.get_historical_data("BTC/USDT", "1h", limit=10)
        assert spy.call_count == 1
        
        # Segunda llamada con los mismos parámetros (debería usar la caché)
        data2 = await data_manager.get_historical_data("BTC/USDT", "1h", limit=10)
        assert spy.call_count == 1  # No se llama de nuevo al proveedor
        
        # Verificar que los datos son idénticos
        assert data1 == data2


@pytest.mark.asyncio
async def test_data_transformation(data_manager):
    """Probar transformación de datos a diferentes formatos."""
    # Obtener datos históricos
    ohlcv_data = await data_manager.get_historical_data("BTC/USDT", "1h", limit=10)
    
    # Convertir a DataFrame
    df = data_manager.ohlcv_to_dataframe(ohlcv_data)
    
    # Verificar estructura del DataFrame
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    
    # Verificar que los datos coinciden
    assert len(df) == len(ohlcv_data)
    for i, candle in enumerate(ohlcv_data):
        timestamp, open_price, high, low, close, volume = candle
        assert df.iloc[i]["open"] == open_price
        assert df.iloc[i]["close"] == close
    
    # Extraer solo precios de cierre
    close_prices = data_manager.extract_close_prices(ohlcv_data)
    assert len(close_prices) == len(ohlcv_data)
    assert all(isinstance(price, (int, float)) for price in close_prices)


@pytest.mark.asyncio
async def test_multiple_providers(data_manager):
    """Probar funcionamiento con múltiples proveedores de datos."""
    # Crear un segundo proveedor
    second_provider = MockDataProvider(name="second_provider")
    
    # Modificar algunos datos para distinguirlo
    second_provider.ticker_data["BTC/USDT"]["last"] = 51000
    
    # Registrar el segundo proveedor
    data_manager.register_provider(second_provider)
    
    # Por defecto, debería usar el primer proveedor registrado
    ticker = await data_manager.get_ticker("BTC/USDT")
    assert ticker["last"] == 50000
    
    # Especificar explícitamente el segundo proveedor
    ticker = await data_manager.get_ticker("BTC/USDT", provider_name="second_provider")
    assert ticker["last"] == 51000


@pytest.mark.asyncio
async def test_error_handling(data_manager):
    """Probar manejo de errores en obtención de datos."""
    # Intentar obtener datos para un símbolo no disponible
    with pytest.raises(ValueError, match="No data available"):
        await data_manager.get_historical_data("INVALID/PAIR", "1h")
    
    # Intentar obtener datos para un timeframe no disponible
    with pytest.raises(ValueError, match="No data available"):
        await data_manager.get_historical_data("BTC/USDT", "invalid_timeframe")
    
    # Intentar usar un proveedor no registrado
    with pytest.raises(ValueError, match="Provider not found"):
        await data_manager.get_ticker("BTC/USDT", provider_name="non_existent_provider")
"""