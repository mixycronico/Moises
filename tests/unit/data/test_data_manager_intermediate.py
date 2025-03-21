"""
Tests intermedios para el Data Manager.

Este módulo prueba funcionalidades intermedias del Data Manager,
incluyendo integración con múltiples fuentes de datos, manejo de
datos en tiempo real y estrategias de agregación de datos.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from genesis.data.manager import DataManager
from genesis.data.providers.base import DataProvider
from genesis.data.providers.websocket import WebSocketProvider
from genesis.data.cache import DataCache
from genesis.core.event_bus import EventBus


class RealtimeDataProvider(WebSocketProvider):
    """Proveedor de datos en tiempo real simulado para pruebas."""
    
    def __init__(self, name="realtime_provider"):
        """Inicializar proveedor con simulación de websocket."""
        super().__init__(name)
        self.subscriptions = {}
        self.is_connected = False
        self.event_callbacks = {}
        self.data_callbacks = {}
    
    async def connect(self):
        """Simular conexión al websocket."""
        self.is_connected = True
        return True
    
    async def disconnect(self):
        """Simular desconexión del websocket."""
        self.is_connected = False
        self.subscriptions = {}
        return True
    
    async def subscribe(self, channel, symbol, callback):
        """Simular suscripción a un canal de datos."""
        key = f"{channel}:{symbol}"
        self.subscriptions[key] = callback
        return True
    
    async def unsubscribe(self, channel, symbol):
        """Simular cancelación de suscripción."""
        key = f"{channel}:{symbol}"
        if key in self.subscriptions:
            del self.subscriptions[key]
        return True
    
    async def send_update(self, channel, symbol, data):
        """Método de prueba para simular datos entrantes."""
        key = f"{channel}:{symbol}"
        if key in self.subscriptions:
            callback = self.subscriptions[key]
            await callback(data)
            return True
        return False


class MockHistoricalProvider(DataProvider):
    """Proveedor de datos históricos simulado para pruebas."""
    
    def __init__(self, name="historical_provider"):
        """Inicializar proveedor con datos simulados."""
        super().__init__(name)
        self.available_symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        self.available_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Datos OHLCV simulados para diferentes periodos
        self.ohlcv_data = {}
        
        # Inicializar datos para cada símbolo y timeframe
        for symbol in self.available_symbols:
            self.ohlcv_data[symbol] = {}
            base_price = 50000 if symbol == "BTC/USDT" else 3000 if symbol == "ETH/USDT" else 0.5
            
            for timeframe in self.available_timeframes:
                n_candles = 100
                self.ohlcv_data[symbol][timeframe] = self._generate_price_data(
                    n_candles=n_candles,
                    base_price=base_price,
                    volatility=0.01
                )
    
    def _generate_price_data(self, n_candles, base_price, volatility):
        """Generar datos de precios con tendencia y volatilidad."""
        data = []
        timestamp = 1609459200000  # 1 de enero de 2021 00:00:00 UTC
        current_price = base_price
        
        for i in range(n_candles):
            # Calcular precios simulando movimientos de mercado
            price_change = current_price * (np.random.normal(0, volatility) + 0.0002)  # Ligera tendencia alcista
            open_price = current_price
            close_price = current_price + price_change
            current_price = close_price
            
            # Calcular high y low
            price_range = abs(close_price - open_price)
            high_price = max(open_price, close_price) + np.random.random() * price_range
            low_price = min(open_price, close_price) - np.random.random() * price_range
            
            # Calcular volumen
            volume = np.random.normal(1000, 200)
            
            # Agregar vela
            data.append([
                timestamp + i * self._get_timeframe_ms(self.available_timeframes[0]),
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            ])
        
        return data
    
    def _get_timeframe_ms(self, timeframe):
        """Convertir timeframe a milisegundos."""
        multiplier = int(timeframe[:-1])
        unit = timeframe[-1]
        
        if unit == 'm':
            return multiplier * 60 * 1000
        elif unit == 'h':
            return multiplier * 60 * 60 * 1000
        elif unit == 'd':
            return multiplier * 24 * 60 * 60 * 1000
        else:
            return 60 * 1000  # Default: 1 minuto
    
    async def get_historical_ohlcv(self, symbol, timeframe, since=None, limit=None):
        """Obtener datos históricos OHLCV."""
        if symbol not in self.ohlcv_data or timeframe not in self.ohlcv_data[symbol]:
            return []
        
        data = self.ohlcv_data[symbol][timeframe]
        
        # Aplicar limit si se especifica
        if limit is not None and limit > 0:
            data = data[-limit:]
            
        return data
    
    async def get_available_symbols(self):
        """Obtener símbolos disponibles."""
        return self.available_symbols
    
    async def get_available_timeframes(self):
        """Obtener marcos temporales disponibles."""
        return self.available_timeframes


@pytest.fixture
def event_bus():
    """Proporcionar un bus de eventos para pruebas."""
    return EventBus()


@pytest.fixture
def realtime_provider():
    """Proporcionar un proveedor de datos en tiempo real simulado."""
    return RealtimeDataProvider()


@pytest.fixture
def historical_provider():
    """Proporcionar un proveedor de datos históricos simulado."""
    return MockHistoricalProvider()


@pytest.fixture
def data_cache():
    """Proporcionar una caché de datos."""
    return DataCache()


@pytest.fixture
def data_manager(event_bus, realtime_provider, historical_provider, data_cache):
    """Proporcionar un gestor de datos con múltiples proveedores."""
    manager = DataManager(event_bus=event_bus)
    manager.register_provider(historical_provider)
    manager.register_provider(realtime_provider)
    manager.set_cache(data_cache)
    manager.set_default_provider("historical_provider")
    return manager


@pytest.mark.asyncio
async def test_data_manager_with_multiple_providers(data_manager):
    """Probar funcionamiento con múltiples proveedores de datos."""
    # Verificar que ambos proveedores están registrados
    assert "historical_provider" in data_manager.providers
    assert "realtime_provider" in data_manager.providers
    
    # Verificar que el proveedor por defecto está configurado
    assert data_manager.default_provider == "historical_provider"


@pytest.mark.asyncio
async def test_realtime_data_subscription(data_manager, realtime_provider):
    """Probar suscripción a datos en tiempo real."""
    # Crear un callback para recibir datos
    received_data = []
    
    async def data_callback(data):
        received_data.append(data)
    
    # Suscribirse a un canal
    subscription_id = await data_manager.subscribe_to_ticker("BTC/USDT", data_callback, provider_name="realtime_provider")
    
    # Verificar que la suscripción se creó correctamente
    assert subscription_id is not None
    assert "ticker:BTC/USDT" in realtime_provider.subscriptions
    
    # Simular datos entrantes
    ticker_data = {"symbol": "BTC/USDT", "last": 50000, "timestamp": time.time()}
    await realtime_provider.send_update("ticker", "BTC/USDT", ticker_data)
    
    # Verificar que el callback recibió los datos
    assert len(received_data) == 1
    assert received_data[0] == ticker_data
    
    # Cancelar la suscripción
    await data_manager.unsubscribe(subscription_id, provider_name="realtime_provider")
    
    # Verificar que la suscripción se canceló
    assert "ticker:BTC/USDT" not in realtime_provider.subscriptions


@pytest.mark.asyncio
async def test_data_manager_aggregation(data_manager, historical_provider):
    """Probar agregación de datos de diferentes timeframes."""
    # Obtener datos de 1 minuto
    minute_data = await data_manager.get_historical_data(
        "BTC/USDT", "1m", limit=60, provider_name="historical_provider"
    )
    
    # Agregar a timeframe de 5 minutos (12 velas)
    five_min_data = data_manager.aggregate_ohlcv(minute_data, 5)
    
    # Verificar cantidad de velas
    assert len(five_min_data) == 12
    
    # Verificar que la agregación es correcta
    # Cada vela de 5 minutos debe tener:
    # - Open igual al Open de la primera vela de 1 minuto
    # - Close igual al Close de la última vela de 1 minuto
    # - High igual al máximo de todos los High
    # - Low igual al mínimo de todos los Low
    # - Volume igual a la suma de todos los volúmenes
    
    # Verificar la primera vela agregada
    first_5min = five_min_data[0]
    first_5_1min = minute_data[0:5]
    
    assert first_5min[1] == first_5_1min[0][1]  # Open
    assert first_5min[4] == first_5_1min[-1][4]  # Close
    assert first_5min[2] == max(candle[2] for candle in first_5_1min)  # High
    assert first_5min[3] == min(candle[3] for candle in first_5_1min)  # Low
    assert first_5min[5] == sum(candle[5] for candle in first_5_1min)  # Volume


@pytest.mark.asyncio
async def test_data_manager_missing_data_handling(data_manager, historical_provider):
    """Probar manejo de datos faltantes."""
    # Crear un conjunto de datos con huecos
    symbol = "BTC/USDT"
    timeframe = "1h"
    data = historical_provider.ohlcv_data[symbol][timeframe][:10]  # Primeras 10 velas
    
    # Añadir un hueco (saltar 2 velas)
    next_data = historical_provider.ohlcv_data[symbol][timeframe][12:22]  # Velas 12-21
    data.extend(next_data)
    
    # Reemplazar los datos en el proveedor
    historical_provider.ohlcv_data[symbol][timeframe] = data
    
    # Intentar obtener datos con interpolación
    interpolated_data = await data_manager.get_historical_data_with_interpolation(
        symbol, timeframe, provider_name="historical_provider"
    )
    
    # Verificar que se rellenaron los huecos
    assert len(interpolated_data) == len(data)
    
    # Verificar que los timestamps son consecutivos
    for i in range(1, len(interpolated_data)):
        current_ts = interpolated_data[i][0]
        prev_ts = interpolated_data[i-1][0]
        expected_diff = 3600000  # 1 hora en ms
        assert current_ts - prev_ts == expected_diff


@pytest.mark.asyncio
async def test_data_manager_event_emission(data_manager, event_bus):
    """Probar emisión de eventos cuando se obtienen nuevos datos."""
    # Espiar el método emit del event_bus
    with patch.object(event_bus, 'emit') as mock_emit:
        # Configurar el mock para que sea awaitable
        mock_emit.return_value = asyncio.Future()
        mock_emit.return_value.set_result(None)
        
        # Obtener datos históricos (debería emitir un evento)
        await data_manager.get_historical_data("BTC/USDT", "1h", limit=10)
        
        # Verificar que se emitió el evento correcto
        mock_emit.assert_called_once()
        # El primer argumento debería ser el tipo de evento
        assert mock_emit.call_args[0][0] == "data_updated"
        # El segundo argumento debería ser los datos
        assert "symbol" in mock_emit.call_args[0][1]
        assert mock_emit.call_args[0][1]["symbol"] == "BTC/USDT"
        assert "timeframe" in mock_emit.call_args[0][1]
        assert mock_emit.call_args[0][1]["timeframe"] == "1h"


@pytest.mark.asyncio
async def test_data_manager_backfill(data_manager, historical_provider):
    """Probar la funcionalidad de backfill para completar datos históricos."""
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Modificar datos en el proveedor para simular datos parciales
    original_data = historical_provider.ohlcv_data[symbol][timeframe]
    historical_provider.ohlcv_data[symbol][timeframe] = original_data[-20:]  # Solo últimas 20 velas
    
    # Realizar backfill
    backfilled_data = await data_manager.backfill_historical_data(
        symbol, timeframe, 
        limit=50,  # Solicitar 50 velas
        provider_name="historical_provider"
    )
    
    # Verificar que se solicitaron más datos para completar
    assert len(backfilled_data) > 20
    assert len(backfilled_data) <= 50  # No más de las solicitadas
    
    # Restaurar datos originales
    historical_provider.ohlcv_data[symbol][timeframe] = original_data


@pytest.mark.asyncio
async def test_data_manager_market_status_detection(data_manager):
    """Probar detección de estado del mercado (abierto/cerrado)."""
    # Este test es más específico para mercados tradicionales, pero también aplicable
    # Para crypto simularemos periodos de baja actividad/mantenimiento
    
    # Obtener datos recientes
    symbol = "BTC/USDT"
    recent_data = await data_manager.get_recent_data(symbol, timeframe="1m", limit=60)
    
    # Verificar si el mercado está activo basado en volumen y volatilidad
    is_active = data_manager.detect_market_activity(recent_data)
    
    # Para crypto, normalmente siempre activo
    assert is_active is True
    
    # Simular periodo de baja actividad (volumen casi cero)
    low_activity_data = []
    for candle in recent_data:
        # Copiar la vela pero con volumen muy bajo
        new_candle = list(candle)
        new_candle[5] = 0.001  # Volumen casi cero
        low_activity_data.append(new_candle)
    
    # Verificar detección de baja actividad
    is_active = data_manager.detect_market_activity(low_activity_data)
    assert is_active is False


@pytest.mark.asyncio
async def test_data_manager_concurrent_requests(data_manager):
    """Probar manejo de múltiples solicitudes concurrentes."""
    # Crear varias solicitudes concurrentes
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    timeframes = ["1m", "5m", "1h"]
    
    requests = []
    for symbol in symbols:
        for timeframe in timeframes:
            requests.append(data_manager.get_historical_data(
                symbol, timeframe, limit=10
            ))
    
    # Ejecutar todas las solicitudes concurrentemente
    start_time = time.time()
    results = await asyncio.gather(*requests)
    end_time = time.time()
    
    # Verificar que todas las solicitudes se completaron
    assert len(results) == len(symbols) * len(timeframes)
    
    # Verificar que cada resultado tiene la estructura correcta
    for result in results:
        assert len(result) == 10  # Solicitamos 10 velas
        assert len(result[0]) == 6  # Estructura OHLCV
    
    # La ejecución concurrente debería ser más rápida que la secuencial
    # (difícil de probar con precisión, pero verificamos que no sea absurdamente lento)
    execution_time = end_time - start_time
    assert execution_time < len(requests) * 0.5  # Estimación generosa
"""