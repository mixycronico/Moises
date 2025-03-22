"""
Pruebas de integración para el sistema Genesis completo.

Este módulo contiene pruebas que verifican la correcta integración de todos
los componentes del sistema Genesis trabajando conjuntamente, simulando
escenarios completos de trading desde la recepción de datos de mercado
hasta la ejecución de operaciones y el análisis de rendimiento.
"""

import pytest
import asyncio
import logging
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus
from genesis.core.config import Settings
from genesis.core.component import Component

from genesis.exchange.manager import ExchangeManager
from genesis.exchange.ccxt_exchange import CCXTExchange

from genesis.data.manager import DataManager
from genesis.data.cache import DataCache
from genesis.data.providers.base import DataProvider

from genesis.analysis.indicators import Indicators
from genesis.analysis.signal_generator import SignalGenerator

from genesis.strategy.base import Strategy
from genesis.strategy.rsi import RSIStrategy
from genesis.strategy.bollinger_bands import BollingerBandsStrategy
from genesis.strategy.moving_average import MovingAverageStrategy
from genesis.strategy.orchestrator import StrategyOrchestrator

from genesis.risk.risk_manager import RiskManager
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator

from genesis.execution.manager import ExecutionManager
from genesis.execution.order import Order, OrderType, OrderSide, OrderStatus

from genesis.performance.analyzer import PerformanceAnalyzer
from genesis.db.manager import DatabaseManager


class MockDataProvider(DataProvider):
    """Proveedor de datos simulado para pruebas de integración."""
    
    def __init__(self, name="mock_provider"):
        """Inicializar proveedor simulado."""
        super().__init__(name)
        self._data = {}
    
    async def get_market_data(self, symbol, timeframe, limit=100):
        """Obtener datos de mercado simulados."""
        if (symbol, timeframe) not in self._data:
            self._data[(symbol, timeframe)] = self._generate_data(limit)
        
        return self._data[(symbol, timeframe)]
    
    def _generate_data(self, limit):
        """Generar datos OHLCV simulados."""
        np.random.seed(int(time.time()))
        
        # Generar timestamps
        now = int(time.time() * 1000)
        timestamps = np.array([now - (i * 60 * 1000) for i in range(limit)])[::-1]
        
        # Generar precios
        base_price = 50000.0
        volatility = 100.0
        
        closes = np.random.normal(0, 1, limit).cumsum() * volatility + base_price
        opens = closes + np.random.normal(0, volatility/10, limit)
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, volatility/5, limit))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, volatility/5, limit))
        volumes = np.abs(np.random.normal(10, 5, limit))
        
        # Asegurar que los precios sean positivos
        opens = np.maximum(opens, 1)
        highs = np.maximum(highs, opens)
        lows = np.minimum(lows, opens)
        lows = np.maximum(lows, 1)
        closes = np.maximum(closes, 1)
        
        # Crear array OHLCV
        data = np.column_stack((timestamps, opens, highs, lows, closes, volumes))
        
        return data


class MockExchange(CCXTExchange):
    """Exchange simulado para pruebas de integración."""
    
    def __init__(self, name="mock_exchange", exchange_id="mock", **kwargs):
        """Inicializar exchange simulado."""
        super().__init__(name, exchange_id, **kwargs)
        
        self._balances = {
            "USDT": 100000.0,
            "BTC": 2.0,
            "ETH": 30.0
        }
        
        self._market_data = {}
        self._orders = {}
        self._positions = {}
        self._next_order_id = 1
    
    async def fetch_balance(self):
        """Obtener saldo simulado."""
        return {
            "free": self._balances.copy(),
            "used": {k: 0.0 for k in self._balances},
            "total": self._balances.copy()
        }
    
    async def fetch_ticker(self, symbol):
        """Obtener ticker simulado."""
        base_price = 50000.0 if "BTC" in symbol else 3000.0
        return {
            "symbol": symbol,
            "bid": base_price * 0.999,
            "ask": base_price * 1.001,
            "last": base_price,
            "volume": random.uniform(10, 100),
            "timestamp": int(time.time() * 1000)
        }
    
    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        """Crear orden simulada."""
        order_id = str(self._next_order_id)
        self._next_order_id += 1
        
        ticker = await self.fetch_ticker(symbol)
        
        # Determinar precio de ejecución
        if type == "market":
            exec_price = ticker["ask"] if side == "buy" else ticker["bid"]
        else:
            exec_price = price
        
        # Calcular costo y comisión
        cost = amount * exec_price
        fee = cost * 0.001  # 0.1% de comisión
        
        # Crear orden
        order = {
            "id": order_id,
            "symbol": symbol,
            "type": type,
            "side": side,
            "price": exec_price,
            "amount": amount,
            "cost": cost,
            "fee": {"cost": fee, "currency": symbol.split("/")[1]},
            "filled": amount,
            "status": "closed",
            "timestamp": int(time.time() * 1000)
        }
        
        # Guardar orden
        self._orders[order_id] = order
        
        # Actualizar balances simulados
        self._update_balances(symbol, side, amount, exec_price, fee)
        
        return order
    
    async def fetch_order(self, id, symbol=None):
        """Obtener información de orden simulada."""
        if id not in self._orders:
            raise Exception(f"Order {id} not found")
        
        return self._orders[id]
    
    def _update_balances(self, symbol, side, amount, price, fee):
        """Actualizar saldos después de una operación."""
        base, quote = symbol.split('/')
        
        if side == "buy":
            # Reducir quote, aumentar base
            self._balances[quote] -= amount * price + fee
            self._balances[base] += amount
        else:
            # Reducir base, aumentar quote
            self._balances[base] -= amount
            self._balances[quote] += amount * price - fee


@pytest.fixture
def event_bus():
    """Proporcionar un bus de eventos para pruebas."""
    return EventBus()


@pytest.fixture
def settings():
    """Proporcionar configuración para pruebas."""
    config = Settings()
    config.set("trading.default_risk_percentage", 2.0)
    config.set("trading.max_drawdown_percentage", 15.0)
    config.set("exchange.default", "mock_exchange")
    config.set("strategy.default", "rsi_strategy")
    config.set("trading.symbols", ["BTC/USDT", "ETH/USDT"])
    config.set("data.default_timeframe", "1h")
    
    return config


@pytest.fixture
def mock_data_provider():
    """Proporcionar un proveedor de datos simulado."""
    return MockDataProvider()


@pytest.fixture
def data_manager(event_bus, mock_data_provider):
    """Proporcionar un gestor de datos configurado."""
    manager = DataManager(event_bus=event_bus, cache=DataCache())
    manager.register_provider(mock_data_provider)
    return manager


@pytest.fixture
def indicators():
    """Proporcionar indicadores técnicos."""
    return Indicators()


@pytest.fixture
def signal_generator(indicators):
    """Proporcionar generador de señales."""
    return SignalGenerator(indicators=indicators)


@pytest.fixture
def position_sizer():
    """Proporcionar calculador de tamaño de posición."""
    sizer = PositionSizer(default_risk_percentage=2.0)
    sizer.set_account_balance(100000.0)
    return sizer


@pytest.fixture
def stop_loss_calculator():
    """Proporcionar calculador de stop-loss."""
    return StopLossCalculator()


@pytest.fixture
def risk_manager(event_bus, position_sizer, stop_loss_calculator):
    """Proporcionar gestor de riesgos."""
    manager = RiskManager(event_bus=event_bus)
    manager._position_sizer = position_sizer
    manager._stop_loss_calculator = stop_loss_calculator
    return manager


@pytest.fixture
def mock_exchange():
    """Proporcionar un exchange simulado."""
    return MockExchange()


@pytest.fixture
def exchange_manager(event_bus, mock_exchange):
    """Proporcionar un gestor de exchanges."""
    manager = ExchangeManager(event_bus=event_bus)
    manager.register_exchange(mock_exchange)
    return manager


@pytest.fixture
def execution_manager(event_bus, exchange_manager, risk_manager):
    """Proporcionar un gestor de ejecución."""
    return ExecutionManager(
        event_bus=event_bus,
        exchange_manager=exchange_manager,
        risk_manager=risk_manager
    )


@pytest.fixture
def rsi_strategy(signal_generator):
    """Proporcionar una estrategia RSI."""
    return RSIStrategy(
        name="rsi_strategy",
        signal_generator=signal_generator,
        parameters={"period": 14, "overbought": 70, "oversold": 30}
    )


@pytest.fixture
def bollinger_strategy(signal_generator):
    """Proporcionar una estrategia de Bandas de Bollinger."""
    return BollingerBandsStrategy(
        name="bollinger_strategy",
        signal_generator=signal_generator,
        parameters={"period": 20, "deviation": 2.0}
    )


@pytest.fixture
def ma_strategy(signal_generator):
    """Proporcionar una estrategia de Medias Móviles."""
    return MovingAverageStrategy(
        name="ma_strategy",
        signal_generator=signal_generator,
        parameters={"fast_period": 10, "slow_period": 30}
    )


@pytest.fixture
def strategy_orchestrator(event_bus, rsi_strategy, bollinger_strategy, ma_strategy):
    """Proporcionar un orquestador de estrategias."""
    orchestrator = StrategyOrchestrator(event_bus=event_bus)
    orchestrator.register_strategy(rsi_strategy)
    orchestrator.register_strategy(bollinger_strategy)
    orchestrator.register_strategy(ma_strategy)
    return orchestrator


@pytest.fixture
def performance_analyzer(event_bus):
    """Proporcionar un analizador de rendimiento."""
    return PerformanceAnalyzer(event_bus=event_bus)


@pytest.fixture
def database_manager():
    """Proporcionar un gestor de base de datos simulado."""
    manager = Mock(spec=DatabaseManager)
    manager.start = AsyncMock()
    manager.stop = AsyncMock()
    manager.store_trade = AsyncMock()
    manager.store_signal = AsyncMock()
    manager.store_performance_metrics = AsyncMock()
    return manager


@pytest.fixture
async def trading_engine(
    event_bus, 
    settings,
    data_manager,
    exchange_manager,
    risk_manager,
    execution_manager,
    strategy_orchestrator,
    performance_analyzer,
    database_manager
):
    """Proporcionar un motor de trading completamente configurado."""
    engine = Engine()
    
    # Registrar componentes
    engine.register_component(event_bus)
    engine.register_component(data_manager)
    engine.register_component(exchange_manager)
    engine.register_component(risk_manager)
    engine.register_component(execution_manager)
    engine.register_component(strategy_orchestrator)
    engine.register_component(performance_analyzer)
    engine.register_component(database_manager)
    
    # Iniciar el motor
    await engine.start()
    
    yield engine
    
    # Detener el motor al finalizar
    await engine.stop()


@pytest.mark.asyncio
async def test_full_trading_cycle(
    trading_engine,
    event_bus,
    data_manager,
    strategy_orchestrator,
    exchange_manager,
    risk_manager,
    execution_manager
):
    """
    Prueba el ciclo completo de trading desde datos hasta ejecución.
    
    Esta prueba simula una integración completa:
    1. Recepción de datos de mercado
    2. Generación de señales de estrategia
    3. Gestión de riesgos
    4. Ejecución de órdenes
    5. Actualización de rendimiento
    """
    # Configurar un símbolo y timeframe para la prueba
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Registrar eventos recibidos para validación
    received_events = {}
    
    async def capture_events(event_type, data, source):
        if event_type not in received_events:
            received_events[event_type] = []
        received_events[event_type].append((data, source))
    
    # Registrar receptor de eventos
    listener_id = event_bus.add_listener(capture_events)
    
    try:
        # 1. Simular la recepción de datos de mercado
        market_data = await data_manager.get_market_data(symbol, timeframe)
        assert market_data is not None, "No se obtuvieron datos de mercado"
        
        # Emitir evento de nuevos datos de mercado
        await event_bus.emit(
            "market_data.updated",
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": market_data
            },
            source="test"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # 2. Verificar que se generaron señales a partir de los datos
        assert "strategy.signal" in received_events, "No se generaron señales de estrategia"
        
        # Extraer la señal generada
        signal_events = received_events["strategy.signal"]
        assert len(signal_events) > 0, "No hay eventos de señal"
        
        # La primera señal generada
        signal_data, signal_source = signal_events[0]
        
        # 3. Verificar que la señal se validó por el gestor de riesgos
        assert "signal.validated" in received_events, "No se validaron las señales"
        
        # Extraer la señal validada
        validated_events = received_events["signal.validated"]
        assert len(validated_events) > 0, "No hay eventos de validación"
        
        validated_data, validated_source = validated_events[0]
        
        # Verificar que la validación incluyó un tamaño de posición
        assert "position_size" in validated_data, "No se calculó el tamaño de posición"
        
        # 4. Verificar que se ejecutó una orden a partir de la señal
        # Podría no haberse ejecutado si la señal era "hold"
        if signal_data.get("signal") in ["buy", "sell"]:
            assert "trade.opened" in received_events, "No se abrió ninguna operación"
            
            # Extraer la orden ejecutada
            trade_events = received_events["trade.opened"]
            assert len(trade_events) > 0, "No hay eventos de operación"
            
            trade_data, trade_source = trade_events[0]
            
            # Verificar que la orden tiene los datos esperados
            assert "symbol" in trade_data, "La orden no tiene símbolo"
            assert "side" in trade_data, "La orden no tiene lado (compra/venta)"
            assert "price" in trade_data, "La orden no tiene precio"
            assert "order_id" in trade_data, "La orden no tiene ID"
            
            # 5. Verificar que se calculó un stop-loss
            assert "trade.stop_loss_set" in received_events, "No se estableció stop-loss"
            
            # Extraer el stop-loss calculado
            stop_loss_events = received_events["trade.stop_loss_set"]
            assert len(stop_loss_events) > 0, "No hay eventos de stop-loss"
            
            stop_loss_data, stop_loss_source = stop_loss_events[0]
            
            # Verificar que el stop-loss tiene los datos esperados
            assert "symbol" in stop_loss_data, "El stop-loss no tiene símbolo"
            assert "price" in stop_loss_data, "El stop-loss no tiene precio"
            
            # Verificar que el stop-loss es coherente con el lado de la operación
            if trade_data["side"] == "buy":
                # Para compras el stop debe estar por debajo del precio de entrada
                assert stop_loss_data["price"] < trade_data["price"], \
                    "Stop-loss para compra no está por debajo del precio de entrada"
            else:
                # Para ventas el stop debe estar por encima del precio de entrada
                assert stop_loss_data["price"] > trade_data["price"], \
                    "Stop-loss para venta no está por encima del precio de entrada"
    
    finally:
        # Limpiar el receptor de eventos
        event_bus.remove_listener(listener_id)


@pytest.mark.asyncio
async def test_multi_strategy_integration(
    trading_engine,
    event_bus,
    data_manager,
    strategy_orchestrator,
    exchange_manager,
    risk_manager,
    execution_manager,
    rsi_strategy,
    bollinger_strategy,
    ma_strategy
):
    """
    Prueba la integración con múltiples estrategias activas.
    
    Verifica que:
    1. Múltiples estrategias procesen los mismos datos
    2. El orquestador combine adecuadamente las señales
    3. El sistema maneje correctamente señales potencialmente contradictorias
    """
    # Símbolos y timeframes para la prueba
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Rastrear señales generadas por cada estrategia
    strategy_signals = {
        "rsi_strategy": None,
        "bollinger_strategy": None,
        "ma_strategy": None
    }
    
    # Rastrear la señal combinada
    combined_signal = None
    
    # Capturador de eventos para verificar señales individuales
    async def capture_strategy_signal(event_type, data, source):
        if event_type == "strategy.signal" and source in strategy_signals:
            strategy_signals[source] = data
    
    # Capturador para señal combinada
    async def capture_combined_signal(event_type, data, source):
        nonlocal combined_signal
        if event_type == "strategy.combined_signal":
            combined_signal = data
    
    # Registrar receptores
    listener1 = event_bus.add_listener(capture_strategy_signal)
    listener2 = event_bus.add_listener(capture_combined_signal)
    
    try:
        # Obtener datos de mercado
        market_data = await data_manager.get_market_data(symbol, timeframe)
        assert market_data is not None, "No se obtuvieron datos de mercado"
        
        # Forzar cada estrategia a generar una señal con los mismos datos
        for strategy_name, strategy in [
            ("rsi_strategy", rsi_strategy),
            ("bollinger_strategy", bollinger_strategy),
            ("ma_strategy", ma_strategy)
        ]:
            signal = await strategy.generate_signal(symbol, market_data)
            assert signal is not None, f"Estrategia {strategy_name} no generó señal"
            
            # Emitir evento de señal de estrategia
            await event_bus.emit(
                "strategy.signal",
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "signal": signal,
                    "metadata": {
                        "strategy": strategy_name
                    }
                },
                source=strategy_name
            )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # Verificar que todas las estrategias generaron señales
        for strategy_name in strategy_signals:
            assert strategy_signals[strategy_name] is not None, \
                f"No se recibió señal de {strategy_name}"
        
        # Verificar que se generó una señal combinada
        assert combined_signal is not None, "No se generó señal combinada"
        
        # Verificar que la señal combinada tiene los metadatos de todas las estrategias
        assert "contributing_strategies" in combined_signal, \
            "La señal combinada no indica las estrategias contribuyentes"
        assert len(combined_signal["contributing_strategies"]) > 0, \
            "No hay estrategias contribuyentes en la señal combinada"
        
        # Verificar que la señal combinada tiene un valor final
        assert "signal" in combined_signal, "La señal combinada no tiene valor final"
        assert combined_signal["signal"] in ["buy", "sell", "hold"], \
            "Señal combinada no tiene un valor válido"
        
        # Verificar que la señal combinada tiene una fuerza
        assert "strength" in combined_signal, "La señal combinada no tiene fuerza"
        assert 0 <= combined_signal["strength"] <= 1, \
            "La fuerza de la señal combinada no está en el rango [0,1]"
    
    finally:
        # Limpiar receptores
        event_bus.remove_listener(listener1)
        event_bus.remove_listener(listener2)


@pytest.mark.asyncio
async def test_risk_management_integration(
    trading_engine,
    event_bus,
    data_manager,
    risk_manager,
    exchange_manager,
    execution_manager
):
    """
    Prueba la integración de la gestión de riesgos con el resto del sistema.
    
    Verifica:
    1. Validación de señales
    2. Cálculo de tamaño de posición
    3. Establecimiento de stop-loss
    4. Manejo de drawdown
    """
    # Símbolos y datos para la prueba
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Rastrear eventos para verificación
    risk_events = {}
    
    # Capturador de eventos de riesgo
    async def capture_risk_events(event_type, data, source):
        if event_type.startswith("risk.") or event_type.startswith("signal."):
            if event_type not in risk_events:
                risk_events[event_type] = []
            risk_events[event_type].append((data, source))
    
    # Registrar receptor
    listener = event_bus.add_listener(capture_risk_events)
    
    try:
        # 1. Probar validación de señal
        await event_bus.emit(
            "strategy.signal",
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "signal": "buy",
                "price": 50000.0,
                "metadata": {"strategy": "test_strategy"}
            },
            source="test_strategy"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # Verificar validación de señal
        assert "signal.validated" in risk_events, "No se validó la señal"
        valid_signal_data, valid_signal_source = risk_events["signal.validated"][0]
        
        # Verificar cálculo de tamaño de posición
        assert "position_size" in valid_signal_data, "No se calculó tamaño de posición"
        assert valid_signal_data["position_size"] > 0, "Tamaño de posición no válido"
        
        # 2. Probar establecimiento de stop-loss después de operación
        await event_bus.emit(
            "trade.opened",
            {
                "symbol": symbol,
                "side": "buy",
                "price": 50000.0,
                "amount": 1.0,
                "order_id": "test123",
                "exchange": "mock_exchange"
            },
            source="execution_manager"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # Verificar stop-loss
        assert "trade.stop_loss_set" in risk_events, "No se estableció stop-loss"
        stop_loss_data, stop_loss_source = risk_events["trade.stop_loss_set"][0]
        
        assert "price" in stop_loss_data, "Stop-loss no tiene precio"
        assert stop_loss_data["price"] < 50000.0, "Stop-loss no está por debajo del precio de entrada"
        
        # 3. Probar actualización de métricas después de cerrar operación
        await event_bus.emit(
            "trade.closed",
            {
                "symbol": symbol,
                "profit": 2000.0,  # $2000 de beneficio
                "profit_percentage": 4.0,  # 4%
                "trade_id": "test123",
                "exchange": "mock_exchange"
            },
            source="execution_manager"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # Verificar actualización de métricas
        assert "risk.metrics_updated" in risk_events, "No se actualizaron métricas de riesgo"
        metrics_data, metrics_source = risk_events["risk.metrics_updated"][0]
        
        assert "symbol" in metrics_data, "Métricas no tienen símbolo"
        assert metrics_data["symbol"] == symbol, "Símbolo incorrecto en métricas"
        assert "metrics" in metrics_data, "No hay datos de métricas"
        
        # 4. Probar manejo de drawdown
        # Simular un drawdown significativo
        await event_bus.emit(
            "portfolio.drawdown",
            {
                "drawdown_percentage": 18.0,  # 18% de drawdown (por encima del límite)
                "current_equity": 82000.0,
                "peak_equity": 100000.0
            },
            source="performance_analyzer"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # Verificar si se tomaron medidas para el drawdown
        # Dependiendo de la implementación, podría haber diferentes respuestas
        # Por ejemplo, activación de modo de protección o ajuste de parámetros
        found_drawdown_response = False
        for event_type in risk_events:
            if "drawdown" in event_type or "protection" in event_type:
                found_drawdown_response = True
                break
        
        assert found_drawdown_response, "No se detectó respuesta al drawdown"
    
    finally:
        # Limpiar receptor
        event_bus.remove_listener(listener)


@pytest.mark.asyncio
async def test_execution_integration(
    trading_engine,
    event_bus,
    exchange_manager,
    execution_manager,
    mock_exchange
):
    """
    Prueba la integración del módulo de ejecución con el resto del sistema.
    
    Verifica:
    1. Ejecución de órdenes
    2. Seguimiento de posiciones
    3. Manejo de errores
    4. Flujo completo de órdenes
    """
    # Símbolos para la prueba
    symbol = "BTC/USDT"
    
    # Rastrear eventos para verificación
    execution_events = {}
    
    # Capturador de eventos de ejecución
    async def capture_execution_events(event_type, data, source):
        if event_type.startswith("trade.") or event_type.startswith("order."):
            if event_type not in execution_events:
                execution_events[event_type] = []
            execution_events[event_type].append((data, source))
    
    # Registrar receptor
    listener = event_bus.add_listener(capture_execution_events)
    
    try:
        # 1. Probar ejecución de orden de mercado
        await event_bus.emit(
            "signal.validated",
            {
                "symbol": symbol,
                "signal": "buy",
                "price": 50000.0,
                "position_size": 0.5,  # 0.5 BTC
                "metadata": {"strategy": "test_strategy"}
            },
            source="risk_manager"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # Verificar que se creó una orden
        assert "trade.opened" in execution_events, "No se abrió ninguna operación"
        trade_data, trade_source = execution_events["trade.opened"][0]
        
        # Verificar detalles de la orden
        assert "symbol" in trade_data, "La orden no tiene símbolo"
        assert trade_data["symbol"] == symbol, "Símbolo incorrecto en la orden"
        assert "side" in trade_data, "La orden no tiene lado (compra/venta)"
        assert trade_data["side"] == "buy", "Lado incorrecto en la orden"
        assert "amount" in trade_data, "La orden no tiene cantidad"
        assert abs(trade_data["amount"] - 0.5) < 0.01, "Cantidad incorrecta en la orden"
        
        # 2. Probar cierre de posición
        order_id = trade_data.get("order_id")
        assert order_id is not None, "La orden no tiene ID"
        
        # Emisión de señal de venta para cerrar la posición
        await event_bus.emit(
            "signal.validated",
            {
                "symbol": symbol,
                "signal": "sell",
                "price": 52000.0,  # Precio más alto (beneficio)
                "position_size": 0.5,  # La misma cantidad para cerrar
                "metadata": {"strategy": "test_strategy"}
            },
            source="risk_manager"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # Verificar que se cerró la posición
        assert len(execution_events.get("trade.opened", [])) >= 2, "No se ejecutó la orden de cierre"
        
        # Verificar evento de cierre
        if "trade.closed" in execution_events:
            close_data, close_source = execution_events["trade.closed"][0]
            
            # Verificar detalles del cierre
            assert "symbol" in close_data, "El cierre no tiene símbolo"
            assert close_data["symbol"] == symbol, "Símbolo incorrecto en el cierre"
            assert "profit" in close_data, "El cierre no tiene beneficio"
            # Debería haber beneficio dado que vendimos a un precio mayor
            assert close_data["profit"] > 0, "No hay beneficio en el cierre"
        
        # 3. Probar manejo de error de ejecución
        # Forzar un error haciendo que el exchange falle
        with patch.object(mock_exchange, 'create_order', side_effect=Exception("Test error")):
            # Emisión de señal que debería fallar
            await event_bus.emit(
                "signal.validated",
                {
                    "symbol": symbol,
                    "signal": "buy",
                    "price": 50000.0,
                    "position_size": 10.0,  # Cantidad grande para distinguir
                    "metadata": {"strategy": "test_strategy"}
                },
                source="risk_manager"
            )
            
            # Esperar a que se procesen los eventos
            await asyncio.sleep(0.5)
            
            # Verificar que se emitió evento de error
            assert "order.failed" in execution_events, "No se registró el fallo de la orden"
            error_data, error_source = execution_events["order.failed"][0]
            
            # Verificar detalles del error
            assert "symbol" in error_data, "El error no tiene símbolo"
            assert error_data["symbol"] == symbol, "Símbolo incorrecto en el error"
            assert "error" in error_data, "No se incluyó mensaje de error"
            assert "Test error" in str(error_data["error"]), "Mensaje de error incorrecto"
    
    finally:
        # Limpiar receptor
        event_bus.remove_listener(listener)


@pytest.mark.asyncio
async def test_performance_analysis_integration(
    trading_engine,
    event_bus,
    performance_analyzer
):
    """
    Prueba la integración del módulo de análisis de rendimiento.
    
    Verifica:
    1. Registro y seguimiento de operaciones
    2. Cálculo de métricas (ROI, drawdown, etc.)
    3. Generación de informes de rendimiento
    4. Detección de drawdown
    """
    # Rastrear eventos para verificación
    performance_events = {}
    
    # Capturador de eventos de rendimiento
    async def capture_performance_events(event_type, data, source):
        if event_type.startswith("performance.") or event_type.startswith("portfolio."):
            if event_type not in performance_events:
                performance_events[event_type] = []
            performance_events[event_type].append((data, source))
    
    # Registrar receptor
    listener = event_bus.add_listener(capture_performance_events)
    
    try:
        # 1. Registrar una serie de operaciones para análisis
        # Operación 1: Beneficio
        await event_bus.emit(
            "trade.closed",
            {
                "symbol": "BTC/USDT",
                "side": "buy",
                "entry_price": 45000.0,
                "exit_price": 48000.0,
                "amount": 1.0,
                "profit": 3000.0,
                "profit_percentage": 6.67,
                "fees": 100.0,
                "duration_minutes": 120,
                "trade_id": "trade1",
                "strategy": "rsi_strategy"
            },
            source="execution_manager"
        )
        
        # Operación 2: Pérdida
        await event_bus.emit(
            "trade.closed",
            {
                "symbol": "ETH/USDT",
                "side": "buy",
                "entry_price": 3000.0,
                "exit_price": 2850.0,
                "amount": 10.0,
                "profit": -1500.0,
                "profit_percentage": -5.0,
                "fees": 80.0,
                "duration_minutes": 60,
                "trade_id": "trade2",
                "strategy": "bollinger_strategy"
            },
            source="execution_manager"
        )
        
        # Operación 3: Beneficio grande
        await event_bus.emit(
            "trade.closed",
            {
                "symbol": "BTC/USDT",
                "side": "buy",
                "entry_price": 46000.0,
                "exit_price": 52000.0,
                "amount": 2.0,
                "profit": 12000.0,
                "profit_percentage": 13.04,
                "fees": 150.0,
                "duration_minutes": 240,
                "trade_id": "trade3",
                "strategy": "ma_strategy"
            },
            source="execution_manager"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # 2. Solicitar cálculo de métricas
        await event_bus.emit(
            "performance.calculate_metrics",
            {
                "period": "all"  # Calcular métricas para todo el período
            },
            source="test"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # Verificar que se calcularon métricas
        assert "performance.metrics_calculated" in performance_events, "No se calcularon métricas"
        metrics_data, metrics_source = performance_events["performance.metrics_calculated"][0]
        
        # Verificar métricas calculadas
        assert "metrics" in metrics_data, "No hay datos de métricas"
        metrics = metrics_data["metrics"]
        
        assert "total_profit" in metrics, "No se calculó beneficio total"
        assert metrics["total_profit"] == 3000.0 + (-1500.0) + 12000.0, "Beneficio total incorrecto"
        
        assert "win_rate" in metrics, "No se calculó tasa de éxito"
        assert metrics["win_rate"] == 2/3 * 100, "Tasa de éxito incorrecta"  # 2 de 3 operaciones ganadoras
        
        assert "total_trades" in metrics, "No se calculó número de operaciones"
        assert metrics["total_trades"] == 3, "Número de operaciones incorrecto"
        
        # 3. Verificar análisis por estrategia
        assert "performance.strategy_analysis" in performance_events, \
            "No se realizó análisis por estrategia"
        
        strategy_data, strategy_source = performance_events["performance.strategy_analysis"][0]
        
        assert "strategies" in strategy_data, "No hay datos de estrategias"
        strategies = strategy_data["strategies"]
        
        # Debería haber datos para las tres estrategias
        assert len(strategies) == 3, "Número incorrecto de estrategias analizadas"
        
        # Verificar análisis por símbolo
        assert "performance.symbol_analysis" in performance_events, \
            "No se realizó análisis por símbolo"
            
        symbol_data, symbol_source = performance_events["performance.symbol_analysis"][0]
        
        assert "symbols" in symbol_data, "No hay datos de símbolos"
        symbols = symbol_data["symbols"]
        
        # Debería haber datos para dos símbolos
        assert len(symbols) == 2, "Número incorrecto de símbolos analizados"
        
        # 4. Verificar detección de drawdown
        assert "portfolio.drawdown" in performance_events, "No se calculó drawdown"
        drawdown_data, drawdown_source = performance_events["portfolio.drawdown"][0]
        
        assert "drawdown_percentage" in drawdown_data, "No hay porcentaje de drawdown"
        assert "current_equity" in drawdown_data, "No hay equidad actual"
        assert "peak_equity" in drawdown_data, "No hay equidad máxima"
    
    finally:
        # Limpiar receptor
        event_bus.remove_listener(listener)


@pytest.mark.asyncio
async def test_data_flow_and_reactivity(
    trading_engine,
    event_bus,
    data_manager,
    strategy_orchestrator,
    risk_manager,
    execution_manager
):
    """
    Prueba el flujo de datos y la reactividad del sistema.
    
    Verifica:
    1. Respuesta a eventos en tiempo real
    2. Propagación correcta de eventos a través del sistema
    3. Manejo de cambios súbitos en datos
    """
    # Símbolos y timeframes para la prueba
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Rastrear eventos secuencialmente para verificar flujo
    event_sequence = []
    
    # Capturador que registra el orden de los eventos
    async def capture_event_sequence(event_type, data, source):
        # Solo rastrear eventos relevantes para el flujo
        if any(event_type.startswith(prefix) for prefix in [
            "market_data.", "strategy.", "signal.", "risk.", "trade.", "order."
        ]):
            event_sequence.append((event_type, source))
    
    # Registrar receptor
    listener = event_bus.add_listener(capture_event_sequence)
    
    try:
        # 1. Simular actualización de datos de mercado con movimiento importante
        market_data = await data_manager.get_market_data(symbol, timeframe)
        
        # Modificar los últimos datos para simular un movimiento importante
        last_index = market_data.shape[0] - 1
        
        # Guardar precio anterior
        prev_close = market_data[last_index, 4]
        
        # Aumentar significativamente (10%) el último precio para generar señal clara
        market_data[last_index, 4] = prev_close * 1.10
        
        # Emitir actualización de datos
        await event_bus.emit(
            "market_data.updated",
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": market_data
            },
            source="test"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(1.0)
        
        # 2. Verificar que los eventos ocurrieron en el orden correcto
        expected_flow = [
            "market_data.updated",       # Datos actualizados
            "strategy.signal",           # Estrategia genera señal
            "strategy.combined_signal",  # Señales combinadas
            "signal.validated",          # Señal validada por gestión de riesgos
            "trade.opened"               # Se abre una operación
        ]
        
        found_sequence = True
        last_found_idx = -1
        
        for expected_event in expected_flow:
            found = False
            for i in range(last_found_idx + 1, len(event_sequence)):
                if event_sequence[i][0] == expected_event:
                    last_found_idx = i
                    found = True
                    break
            
            if not found:
                found_sequence = False
                break
        
        assert found_sequence, f"No se encontró la secuencia esperada de eventos. Secuencia actual: {event_sequence}"
        
        # 3. Simular otro cambio de mercado que debería generar cierre
        # Modificar los últimos datos para simular un movimiento opuesto
        market_data[last_index, 4] = prev_close * 0.90  # Caída del 10%
        
        # Emitir actualización de datos
        await event_bus.emit(
            "market_data.updated",
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": market_data
            },
            source="test"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(1.0)
        
        # Verificar que hay eventos relevantes para cierre
        close_events = [
            evt for evt in event_sequence 
            if evt[0] == "trade.closed" or evt[0] == "signal.validated" and "sell" in str(evt)
        ]
        
        assert len(close_events) > 0, "No se detectaron eventos de cierre tras el cambio de mercado"
    
    finally:
        # Limpiar receptor
        event_bus.remove_listener(listener)


@pytest.mark.asyncio
async def test_error_handling_and_resilience(
    trading_engine,
    event_bus,
    data_manager,
    strategy_orchestrator,
    risk_manager,
    execution_manager,
    exchange_manager,
    mock_exchange
):
    """
    Prueba el manejo de errores y la resiliencia del sistema.
    
    Verifica:
    1. Recuperación ante fallos de componentes
    2. Manejo de errores de red/exchange
    3. Comportamiento con datos malformados
    4. Propagación adecuada de errores
    """
    # Rastrear eventos de error
    error_events = []
    
    # Capturador que registra eventos de error
    async def capture_error_events(event_type, data, source):
        if "error" in event_type.lower() or "failed" in event_type.lower():
            error_events.append((event_type, data, source))
    
    # Registrar receptor
    listener = event_bus.add_listener(capture_error_events)
    
    try:
        # 1. Probar recuperación ante error en exchange
        # Forzar error en el exchange
        with patch.object(mock_exchange, 'create_order', side_effect=Exception("Simulated network error")):
            # Intentar ejecutar una orden
            await event_bus.emit(
                "signal.validated",
                {
                    "symbol": "BTC/USDT",
                    "signal": "buy",
                    "price": 50000.0,
                    "position_size": 1.0,
                    "metadata": {"strategy": "test_strategy"}
                },
                source="risk_manager"
            )
            
            # Esperar a que se procesen los eventos
            await asyncio.sleep(0.5)
            
            # Verificar que se registró el error
            exchange_errors = [e for e in error_events if "order" in e[0].lower()]
            assert len(exchange_errors) > 0, "No se registró error de exchange"
            
            # Verificar que el sistema sigue funcionando
            # Probar con otra operación simple sin el patch
        
        # El mock debería funcionar ahora
        await event_bus.emit(
            "signal.validated",
            {
                "symbol": "ETH/USDT",
                "signal": "buy",
                "price": 3000.0,
                "position_size": 1.0,
                "metadata": {"strategy": "test_strategy"}
            },
            source="risk_manager"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # 2. Probar manejo de datos malformados
        # Enviar datos de mercado malformados
        malformed_data = np.ones((10, 2))  # Solo 2 columnas en lugar de 6
        
        await event_bus.emit(
            "market_data.updated",
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "data": malformed_data
            },
            source="test"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # Verificar que se registró el error
        data_errors = [e for e in error_events if "data" in e[0].lower() or "strategy" in e[0].lower()]
        assert len(data_errors) > 0, "No se registró error de datos malformados"
        
        # 3. Probar señal con símbolo no soportado
        await event_bus.emit(
            "signal.validated",
            {
                "symbol": "INVALID/PAIR",
                "signal": "buy",
                "price": 100.0,
                "position_size": 1.0,
                "metadata": {"strategy": "test_strategy"}
            },
            source="risk_manager"
        )
        
        # Esperar a que se procesen los eventos
        await asyncio.sleep(0.5)
        
        # Verificar que se registró el error
        symbol_errors = [e for e in error_events if "symbol" in str(e[1]).lower() or "invalid" in str(e[1]).lower()]
        assert len(symbol_errors) > 0, "No se registró error de símbolo inválido"
        
        # 4. Verificar que el sistema sigue funcionando después de los errores
        # Verificar que el motor sigue ejecutándose
        assert trading_engine.is_running(), "El motor se detuvo debido a los errores"
        
        # Verificar que podemos seguir obteniendo datos
        data = await data_manager.get_market_data("BTC/USDT", "1h")
        assert data is not None, "No se pueden obtener datos después de los errores"
    
    finally:
        # Limpiar receptor
        event_bus.remove_listener(listener)


@pytest.mark.asyncio
async def test_database_integration(
    trading_engine,
    event_bus,
    database_manager
):
    """
    Prueba la integración con la base de datos.
    
    Verifica:
    1. Almacenamiento de señales
    2. Almacenamiento de operaciones
    3. Almacenamiento de métricas de rendimiento
    4. Persistencia de datos entre reinicios
    """
    # Verificar que el gestor de base de datos se inició
    assert database_manager.start.called, "El gestor de base de datos no se inició"
    
    # 1. Enviar eventos que deberían almacenarse en la base de datos
    # Señal de estrategia
    await event_bus.emit(
        "strategy.signal",
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "signal": "buy",
            "price": 50000.0,
            "metadata": {"strategy": "test_strategy"}
        },
        source="test_strategy"
    )
    
    # Operación abierta
    await event_bus.emit(
        "trade.opened",
        {
            "symbol": "BTC/USDT",
            "side": "buy",
            "price": 50000.0,
            "amount": 1.0,
            "order_id": "db_test_order_1",
            "exchange": "mock_exchange",
            "timestamp": int(time.time() * 1000)
        },
        source="execution_manager"
    )
    
    # Operación cerrada
    await event_bus.emit(
        "trade.closed",
        {
            "symbol": "BTC/USDT",
            "side": "buy",
            "entry_price": 50000.0,
            "exit_price": 52000.0,
            "amount": 1.0,
            "profit": 2000.0,
            "profit_percentage": 4.0,
            "fees": 50.0,
            "duration_minutes": 60,
            "trade_id": "db_test_order_1",
            "strategy": "test_strategy",
            "timestamp": int(time.time() * 1000)
        },
        source="execution_manager"
    )
    
    # Métricas de rendimiento
    await event_bus.emit(
        "performance.metrics_calculated",
        {
            "metrics": {
                "total_profit": 2000.0,
                "win_rate": 100.0,
                "total_trades": 1,
                "average_profit": 2000.0,
                "average_duration_minutes": 60
            },
            "period": "all",
            "timestamp": int(time.time() * 1000)
        },
        source="performance_analyzer"
    )
    
    # Esperar a que se procesen los eventos
    await asyncio.sleep(0.5)
    
    # 2. Verificar que se llamaron los métodos de almacenamiento
    # Almacenamiento de señal
    database_manager.store_signal.assert_called_at_least_once()
    
    # Almacenamiento de operación
    database_manager.store_trade.assert_called_at_least_once()
    
    # Almacenamiento de métricas
    database_manager.store_performance_metrics.assert_called_at_least_once()
    
    # 3. Verificar parámetros de llamadas
    # Verificar operación
    trade_calls = database_manager.store_trade.call_args_list
    found_trade_call = False
    
    for call in trade_calls:
        args, kwargs = call
        trade_data = args[0] if args else kwargs.get("trade_data")
        if trade_data and trade_data.get("trade_id") == "db_test_order_1":
            found_trade_call = True
            assert trade_data["profit"] == 2000.0, "Datos de beneficio incorrectos"
            assert trade_data["symbol"] == "BTC/USDT", "Datos de símbolo incorrectos"
            break
    
    assert found_trade_call, "No se encontró la llamada para almacenar la operación de prueba"
    
    # 4. Simular reinicio del sistema y verificar carga de datos
    # Detener el motor
    await trading_engine.stop()
    
    # Verificar que se llamó a stop en el gestor de base de datos
    assert database_manager.stop.called, "El gestor de base de datos no se detuvo"
    
    # Reiniciar el motor
    database_manager.start.reset_mock()
    await trading_engine.start()
    
    # Verificar que se reinició el gestor de base de datos
    assert database_manager.start.called, "El gestor de base de datos no se reinició"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])