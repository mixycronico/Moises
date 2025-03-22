"""
Tests avanzados para el Data Manager.

Este módulo prueba funcionalidades avanzadas del Data Manager,
incluyendo manejo de alta concurrencia, tolerancia a fallos,
recuperación ante caídas de proveedores, manejo de datos a gran
escala y optimización de rendimiento.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
import time
import random
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from genesis.data.manager import DataManager
from genesis.data.providers.base import DataProvider
from genesis.data.providers.websocket import WebSocketProvider
from genesis.data.providers.rest import RESTProvider
from genesis.data.cache import DataCache
from genesis.data.storage import DataStorage
from genesis.core.event_bus import EventBus
from genesis.core.component import Component


class UnreliableDataProvider(DataProvider):
    """Proveedor de datos que simula fallos intermitentes para pruebas de resiliencia."""
    
    def __init__(self, name, failure_rate=0.3, latency_mean=0.05, latency_std=0.02):
        """
        Inicializa el proveedor no confiable.
        
        Args:
            name: Nombre del proveedor
            failure_rate: Tasa de fallos (0-1)
            latency_mean: Latencia media en segundos
            latency_std: Desviación estándar de la latencia
        """
        super().__init__(name)
        self.failure_rate = failure_rate
        self.latency_mean = latency_mean
        self.latency_std = latency_std
        self.available = True
    
    async def get_market_data(self, symbol, timeframe, limit=100):
        """Simula obtención de datos de mercado con posibles fallos."""
        # Simular latencia variable
        latency = np.random.normal(self.latency_mean, self.latency_std)
        latency = max(0.001, latency)  # Evitar latencias negativas
        await asyncio.sleep(latency)
        
        # Simular fallo aleatorio
        if random.random() < self.failure_rate:
            raise Exception(f"Simulated failure in {self.name}")
        
        # Si no hay fallo, devolver datos aleatorios
        if not self.available:
            raise Exception(f"Provider {self.name} is unavailable")
            
        # Generar datos de ejemplo
        return self._generate_sample_data(limit)
        
    def _generate_sample_data(self, limit):
        """Genera datos OHLCV aleatorios para pruebas."""
        np.random.seed(int(time.time()))  # Diferente semilla cada vez
        
        # Crear timestamps
        now = int(time.time() * 1000)
        timestamps = np.array([now - (i * 60 * 1000) for i in range(limit)])[::-1]
        
        # Crear precios que siguen un camino aleatorio
        base_price = 1000.0
        price_changes = np.random.normal(0, 20, limit).cumsum()
        closes = base_price + price_changes
        
        # Crear OHLCV
        data = []
        for i in range(limit):
            volatility = np.random.uniform(5, 15)
            open_price = closes[i] - np.random.uniform(-volatility, volatility)
            high_price = max(open_price, closes[i]) + np.random.uniform(0, volatility)
            low_price = min(open_price, closes[i]) - np.random.uniform(0, volatility)
            volume = np.random.uniform(10, 1000)
            
            data.append([
                timestamps[i],
                open_price,
                high_price,
                low_price,
                closes[i],
                volume
            ])
            
        return np.array(data)


@pytest.fixture
def unreliable_providers():
    """Crear un conjunto de proveedores no confiables para pruebas."""
    return [
        UnreliableDataProvider(f"provider_{i}", 
                            failure_rate=random.uniform(0.1, 0.5),
                            latency_mean=random.uniform(0.01, 0.2),
                            latency_std=random.uniform(0.005, 0.05))
        for i in range(5)
    ]


@pytest.fixture
def event_bus():
    """Proporcionar un bus de eventos para pruebas."""
    return EventBus()


@pytest.fixture
def data_cache():
    """Proporcionar una caché de datos para pruebas."""
    return DataCache(max_size=1000)


@pytest.fixture
def data_storage():
    """Proporcionar un almacenamiento de datos para pruebas."""
    mock_storage = Mock(spec=DataStorage)
    mock_storage.store_market_data = AsyncMock()
    mock_storage.retrieve_market_data = AsyncMock(return_value=None)
    return mock_storage


@pytest.fixture
def data_manager(event_bus, data_cache, data_storage, unreliable_providers):
    """Proporcionar un gestor de datos con proveedores no confiables."""
    manager = DataManager(
        event_bus=event_bus,
        cache=data_cache,
        storage=data_storage
    )
    
    # Registrar proveedores no confiables
    for provider in unreliable_providers:
        manager.register_provider(provider)
        
    return manager


@pytest.mark.asyncio
async def test_data_manager_fault_tolerance(data_manager, unreliable_providers):
    """Probar la tolerancia a fallos del gestor de datos con proveedores no confiables."""
    # Configurar un símbolo y timeframe para pruebas
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Intentar obtener datos a pesar de proveedores no confiables
    data = await data_manager.get_market_data(symbol, timeframe, limit=50)
    
    # Debería haber obtenido datos válidos a pesar de los fallos
    assert data is not None
    assert len(data) > 0
    
    # Verificar el contador de fallos
    for provider in unreliable_providers:
        provider_stats = data_manager.get_provider_stats(provider.name)
        # Si el proveedor falló, debería estar registrado
        if provider_stats.get("failure_count", 0) > 0:
            assert provider_stats.get("last_failure_time") is not None


@pytest.mark.asyncio
async def test_data_manager_provider_failover(data_manager, unreliable_providers):
    """Probar la conmutación por error entre proveedores cuando uno falla."""
    # Hacer que el primer proveedor esté completamente no disponible
    unreliable_providers[0].available = False
    
    # Configurar un símbolo y timeframe para pruebas
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Espiar el método get_market_data de los proveedores
    provider_spies = []
    for provider in unreliable_providers:
        spy = AsyncMock(wraps=provider.get_market_data)
        provider.get_market_data = spy
        provider_spies.append(spy)
    
    # Intentar obtener datos
    data = await data_manager.get_market_data(symbol, timeframe, limit=50)
    
    # Verificar que se intentó con el primer proveedor y luego con otros
    provider_spies[0].assert_called_once()
    
    # Al menos un proveedor más debería haber sido llamado como respaldo
    other_provider_called = False
    for i in range(1, len(provider_spies)):
        if provider_spies[i].called:
            other_provider_called = True
            break
    
    assert other_provider_called, "No se intentó con proveedores de respaldo"
    
    # Debería haber obtenido datos válidos a pesar del fallo
    assert data is not None
    assert len(data) > 0


@pytest.mark.asyncio
async def test_data_manager_concurrent_requests(data_manager):
    """Probar el rendimiento del gestor de datos bajo múltiples solicitudes concurrentes."""
    # Configurar símbolos y timeframes para pruebas
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "DOT/USDT"]
    timeframes = ["1m", "5m", "15m", "1h", "4h"]
    
    # Crear muchas solicitudes concurrentes
    num_requests = 50
    requests = []
    
    for _ in range(num_requests):
        symbol = random.choice(symbols)
        timeframe = random.choice(timeframes)
        limit = random.randint(10, 100)
        
        requests.append(data_manager.get_market_data(symbol, timeframe, limit=limit))
    
    # Ejecutar todas las solicitudes concurrentemente
    start_time = time.time()
    results = await asyncio.gather(*requests, return_exceptions=True)
    end_time = time.time()
    
    # Calcular estadísticas
    total_time = end_time - start_time
    successful_requests = sum(1 for r in results if not isinstance(r, Exception))
    failed_requests = sum(1 for r in results if isinstance(r, Exception))
    
    # Verificar que se manejaron todas las solicitudes
    assert len(results) == num_requests
    
    # Debe haber al menos algunas solicitudes exitosas
    assert successful_requests > 0
    
    # Verificar que el tiempo total es razonable (depende del sistema de pruebas)
    # En un sistema bien optimizado con alta concurrencia, debería ser mucho menos 
    # que la suma de tiempos individuales
    assert total_time < (num_requests * 0.2)  # Asumiendo un límite superior razonable


@pytest.mark.asyncio
async def test_data_manager_backpressure_handling(data_manager, unreliable_providers):
    """Probar el manejo de contrapresión cuando hay demasiadas solicitudes."""
    # Configurar símbolos y timeframes para pruebas
    symbol = "BTC/USDT"
    timeframe = "1m"
    
    # Hacer que todos los proveedores sean muy lentos
    for provider in unreliable_providers:
        provider.latency_mean = 0.5  # Medio segundo por solicitud
    
    # Establecer límite de concurrencia en el gestor de datos
    data_manager.max_concurrent_requests = 5
    
    # Crear muchas solicitudes concurrentes (más que el límite de concurrencia)
    num_requests = 20
    requests = []
    
    for _ in range(num_requests):
        requests.append(data_manager.get_market_data(symbol, timeframe, limit=10))
    
    # Medir tiempo de ejecución
    start_time = time.time()
    results = await asyncio.gather(*requests, return_exceptions=True)
    end_time = time.time()
    
    # Verificar que todas las solicitudes se completaron
    assert len(results) == num_requests
    
    # Verificar que el tiempo total refleja el procesamiento por lotes
    # Debería ser aproximadamente (num_requests / concurrencia) * tiempo_medio_solicitud
    expected_time = (num_requests / data_manager.max_concurrent_requests) * 0.5
    actual_time = end_time - start_time
    
    # Permitir cierta variabilidad, pero el tiempo debe estar en el rango esperado
    assert actual_time > expected_time * 0.5, f"Tiempo demasiado corto: {actual_time}s vs esperado {expected_time}s"
    assert actual_time < expected_time * 2.0, f"Tiempo demasiado largo: {actual_time}s vs esperado {expected_time}s"


@pytest.mark.asyncio
async def test_data_manager_cache_performance(data_manager, unreliable_providers):
    """Probar el rendimiento de la caché en solicitudes repetidas."""
    # Configurar un símbolo y timeframe para pruebas
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Espiar el método get_market_data de los proveedores
    provider_spies = []
    for provider in unreliable_providers:
        spy = AsyncMock(wraps=provider.get_market_data)
        provider.get_market_data = spy
        provider_spies.append(spy)
    
    # Primera solicitud (no en caché)
    start_time = time.time()
    data1 = await data_manager.get_market_data(symbol, timeframe, limit=50)
    first_request_time = time.time() - start_time
    
    # Reiniciar contadores de llamadas
    for spy in provider_spies:
        spy.reset_mock()
    
    # Segunda solicitud (debería estar en caché)
    start_time = time.time()
    data2 = await data_manager.get_market_data(symbol, timeframe, limit=50)
    second_request_time = time.time() - start_time
    
    # Verificar que la segunda solicitud fue más rápida (de la caché)
    assert second_request_time < first_request_time * 0.5, "La solicitud en caché no fue significativamente más rápida"
    
    # Verificar que los proveedores no fueron llamados en la segunda solicitud
    for spy in provider_spies:
        spy.assert_not_called()
    
    # Verificar que los datos son los mismos
    np.testing.assert_array_equal(data1, data2)


@pytest.mark.asyncio
async def test_data_manager_provider_prioritization(data_manager, unreliable_providers):
    """Probar la priorización de proveedores basada en rendimiento histórico."""
    # Configurar un símbolo y timeframe para pruebas
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Simular estadísticas de rendimiento para los proveedores
    for i, provider in enumerate(unreliable_providers):
        # El primer proveedor tiene el mejor rendimiento
        if i == 0:
            data_manager.provider_stats[provider.name] = {
                "success_count": 100,
                "failure_count": 5,
                "avg_response_time": 0.05
            }
        # El segundo proveedor tiene rendimiento medio
        elif i == 1:
            data_manager.provider_stats[provider.name] = {
                "success_count": 80,
                "failure_count": 20,
                "avg_response_time": 0.1
            }
        # Los demás tienen peor rendimiento
        else:
            data_manager.provider_stats[provider.name] = {
                "success_count": 50,
                "failure_count": 50,
                "avg_response_time": 0.2
            }
    
    # Espiar el método get_market_data de los proveedores
    provider_spies = []
    for provider in unreliable_providers:
        spy = AsyncMock(wraps=provider.get_market_data)
        provider.get_market_data = spy
        provider_spies.append(spy)
        
    # Asegurarnos de que todos los proveedores funcionan para esta prueba
    for provider in unreliable_providers:
        provider.failure_rate = 0
        provider.available = True
    
    # Obtener datos
    await data_manager.get_market_data(symbol, timeframe, limit=50)
    
    # Verificar que se intentó primero con el proveedor de mejor rendimiento
    # y no con los demás ya que el primero no falló
    provider_spies[0].assert_called_once()
    
    for i in range(1, len(provider_spies)):
        provider_spies[i].assert_not_called()


@pytest.mark.asyncio
async def test_data_manager_recovery_from_failure(data_manager, unreliable_providers):
    """Probar la recuperación después de fallos de proveedores."""
    # Configurar un símbolo y timeframe para pruebas
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Hacer que todos los proveedores fallen inicialmente
    for provider in unreliable_providers:
        provider.failure_rate = 1.0  # 100% de fallos
    
    # La primera solicitud debería fallar completamente
    with pytest.raises(Exception):
        await data_manager.get_market_data(symbol, timeframe, limit=50)
    
    # Ahora hacer que algunos proveedores estén disponibles
    for i, provider in enumerate(unreliable_providers):
        if i < 2:  # Los dos primeros proveedores se recuperan
            provider.failure_rate = 0.0  # 0% de fallos
    
    # La segunda solicitud debería tener éxito
    data = await data_manager.get_market_data(symbol, timeframe, limit=50)
    
    # Verificar que se obtuvieron datos válidos
    assert data is not None
    assert len(data) > 0


@pytest.mark.asyncio
async def test_data_manager_storage_integration(data_manager, data_storage, unreliable_providers):
    """Probar la integración con almacenamiento persistente cuando la caché falla."""
    # Configurar un símbolo y timeframe para pruebas
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Hacer que los proveedores fallen
    for provider in unreliable_providers:
        provider.failure_rate = 1.0  # 100% de fallos
    
    # Simular datos almacenados
    stored_data = np.array([
        [1614556800000, 100.0, 105.0, 95.0, 103.0, 1000.0],
        [1614560400000, 103.0, 108.0, 98.0, 106.0, 1200.0],
        [1614564000000, 106.0, 110.0, 101.0, 109.0, 1100.0]
    ])
    data_storage.retrieve_market_data.return_value = stored_data
    
    # La solicitud debería recuperar datos del almacenamiento
    data = await data_manager.get_market_data(symbol, timeframe, limit=50)
    
    # Verificar que se llamó al método de recuperación del almacenamiento
    data_storage.retrieve_market_data.assert_called_once_with(symbol, timeframe, limit=50)
    
    # Verificar que se obtuvieron los datos almacenados
    assert data is not None
    np.testing.assert_array_equal(data, stored_data)


@pytest.mark.asyncio
async def test_data_manager_data_transformation(data_manager, unreliable_providers):
    """Probar transformaciones de datos avanzadas."""
    # Configurar un símbolo y timeframe para pruebas
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Asegurarse de que al menos un proveedor funcione
    unreliable_providers[0].failure_rate = 0.0
    
    # Definir una función de transformación personalizada
    def custom_transform(data):
        # Añadir una columna "typical_price" (TP) = (High + Low + Close) / 3
        if len(data) == 0:
            return data
            
        high = data[:, 2]
        low = data[:, 3]
        close = data[:, 4]
        
        typical_price = (high + low + close) / 3
        
        # Crear nuevo array con la columna adicional
        result = np.zeros((data.shape[0], data.shape[1] + 1))
        result[:, :-1] = data
        result[:, -1] = typical_price
        
        return result
    
    # Registrar la transformación
    transform_name = "typical_price"
    data_manager.register_transformation(transform_name, custom_transform)
    
    # Obtener datos con la transformación
    data = await data_manager.get_market_data(
        symbol, 
        timeframe, 
        limit=50,
        transformations=[transform_name]
    )
    
    # Verificar que los datos tienen la columna adicional
    assert data is not None
    assert data.shape[1] == 7  # 6 columnas originales + 1 calculada
    
    # Verificar que la columna calculada es correcta
    for i in range(len(data)):
        high = data[i, 2]
        low = data[i, 3]
        close = data[i, 4]
        expected_tp = (high + low + close) / 3
        
        assert abs(data[i, 6] - expected_tp) < 1e-6


@pytest.mark.asyncio
async def test_data_manager_real_time_updates(data_manager, event_bus):
    """Probar actualizaciones en tiempo real a través del bus de eventos."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Crear un receptor para eventos
    events_received = []
    
    async def event_listener(event_type, data, source):
        if event_type == "market_data_update" and data.get("symbol") == symbol:
            events_received.append(data)
    
    # Registrar el receptor
    event_bus.register_listener(event_listener)
    
    # Simular una actualización de datos en tiempo real
    market_update = {
        "symbol": symbol,
        "price": 45000.0,
        "volume": 1.5,
        "timestamp": int(time.time() * 1000)
    }
    
    # Emitir evento de actualización
    await event_bus.emit("market_data_update", market_update, "test")
    
    # Esperar un poco para que el evento se procese
    await asyncio.sleep(0.1)
    
    # Verificar que se recibió el evento
    assert len(events_received) == 1
    assert events_received[0]["symbol"] == symbol
    assert events_received[0]["price"] == 45000.0


@pytest.mark.asyncio
async def test_data_manager_websocket_integration(data_manager, event_bus):
    """Probar la integración de WebSocket con el gestor de datos."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Mockear un proveedor WebSocket
    mock_ws_provider = Mock(spec=WebSocketProvider)
    mock_ws_provider.name = "mock_websocket"
    mock_ws_provider.subscribe = AsyncMock()
    mock_ws_provider.unsubscribe = AsyncMock()
    
    # Registrar el proveedor WebSocket
    data_manager.register_websocket_provider(mock_ws_provider)
    
    # Suscribir a actualizaciones en tiempo real
    await data_manager.subscribe_to_ticker(symbol)
    
    # Verificar que se llamó al método subscribe del proveedor
    mock_ws_provider.subscribe.assert_called_once()
    assert symbol in mock_ws_provider.subscribe.call_args[0]
    
    # Simular una actualización de ticker
    ticker_update = {
        "symbol": symbol,
        "bid": 44000.0,
        "ask": 44100.0,
        "last": 44050.0,
        "volume": 2.5,
        "timestamp": int(time.time() * 1000)
    }
    
    # Configurar un receptor de eventos
    events_received = []
    
    async def event_listener(event_type, data, source):
        if event_type == "ticker_update" and data.get("symbol") == symbol:
            events_received.append(data)
    
    # Registrar el receptor
    event_bus.register_listener(event_listener)
    
    # Simular recepción de datos por WebSocket
    callback = mock_ws_provider.subscribe.call_args[1]["callback"]
    await callback("ticker_update", ticker_update)
    
    # Esperar un poco para que el evento se procese
    await asyncio.sleep(0.1)
    
    # Verificar que se recibió el evento
    assert len(events_received) == 1
    assert events_received[0]["symbol"] == symbol
    assert events_received[0]["last"] == 44050.0
    
    # Desuscribir
    await data_manager.unsubscribe_from_ticker(symbol)
    
    # Verificar que se llamó al método unsubscribe del proveedor
    mock_ws_provider.unsubscribe.assert_called_once()
    assert symbol in mock_ws_provider.unsubscribe.call_args[0]


@pytest.mark.asyncio
async def test_data_manager_load_balance(data_manager, unreliable_providers):
    """Probar el balanceo de carga entre proveedores."""
    # Configurar símbolos para pruebas
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    timeframe = "1h"
    
    # Hacer que todos los proveedores tengan baja tasa de fallos
    for provider in unreliable_providers:
        provider.failure_rate = 0.1
    
    # Espiar el método get_market_data de los proveedores
    provider_spies = []
    for provider in unreliable_providers:
        spy = AsyncMock(wraps=provider.get_market_data)
        provider.get_market_data = spy
        provider_spies.append(spy)
    
    # Activar balanceo de carga en el gestor de datos
    data_manager.enable_load_balancing = True
    
    # Realizar múltiples solicitudes
    num_requests = 20
    requests = []
    
    for _ in range(num_requests):
        symbol = random.choice(symbols)
        requests.append(data_manager.get_market_data(symbol, timeframe, limit=10))
    
    # Ejecutar todas las solicitudes
    await asyncio.gather(*requests, return_exceptions=True)
    
    # Verificar que se utilizaron múltiples proveedores
    providers_used = sum(1 for spy in provider_spies if spy.call_count > 0)
    assert providers_used > 1, "No se utilizaron múltiples proveedores para balanceo de carga"
    
    # Verificar que las llamadas se distribuyeron entre los proveedores
    calls_per_provider = [spy.call_count for spy in provider_spies]
    variance = np.var(calls_per_provider)
    
    # La varianza no debería ser demasiado alta si la carga está bien equilibrada
    assert variance < (num_requests / 2), f"Varianza demasiado alta: {variance}"


@pytest.mark.asyncio
async def test_data_manager_data_validation(data_manager, unreliable_providers):
    """Probar la validación de datos antes de su uso."""
    # Configurar un símbolo y timeframe para pruebas
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Espiar el método get_market_data del primer proveedor
    provider = unreliable_providers[0]
    original_get_market_data = provider.get_market_data
    
    # Hacer que a veces devuelva datos inválidos (con NaN o negativos)
    async def get_corrupted_data(*args, **kwargs):
        data = await original_get_market_data(*args, **kwargs)
        
        # 50% de probabilidad de corrupción
        if random.random() < 0.5:
            # Corromper algunos datos (precios negativos, NaN)
            corrupted_indices = random.sample(range(len(data)), k=len(data)//4)
            for idx in corrupted_indices:
                # Diferentes tipos de corrupción
                if random.random() < 0.5:
                    # Precios negativos
                    data[idx, random.randint(1, 4)] = -random.random() * 100
                else:
                    # NaN
                    data[idx, random.randint(1, 4)] = np.nan
                    
        return data
    
    provider.get_market_data = get_corrupted_data
    
    # Activar validación estricta en el gestor de datos
    data_manager.validate_data = True
    
    # Realizar múltiples solicitudes para aumentar probabilidad de datos corruptos
    num_requests = 10
    valid_results = 0
    
    for _ in range(num_requests):
        try:
            data = await data_manager.get_market_data(symbol, timeframe, limit=20)
            
            # Verificar que los datos son válidos
            assert data is not None
            assert len(data) > 0
            assert not np.isnan(data).any(), "Hay valores NaN en los datos"
            assert (data[:, 1:5] >= 0).all(), "Hay precios negativos en los datos"
            
            valid_results += 1
        except Exception:
            # Algunas solicitudes pueden fallar si todos los proveedores devuelven datos corruptos
            pass
    
    # Debería haber al menos algunos resultados válidos
    assert valid_results > 0, "No se obtuvieron resultados válidos"
    
    # Restaurar el método original
    provider.get_market_data = original_get_market_data
"""