"""
Test de integración completo del sistema Genesis.

Este test prueba el flujo completo del sistema de trading, desde la obtención de datos,
generación de señales, decisiones de riesgo, hasta la ejecución de operaciones.
"""

import pytest
import asyncio
import logging
import time
from unittest.mock import Mock, patch, AsyncMock

from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus
from genesis.core.component import Component
from genesis.data.manager import DataManager
from genesis.data.providers.base import DataProvider
from genesis.analysis.indicators import TechnicalIndicators
from genesis.analysis.signal_generator import SignalGenerator
from genesis.strategies.manager import StrategyManager
from genesis.strategies.base import Strategy
from genesis.strategies.moving_average import MovingAverageStrategy
from genesis.strategies.rsi import RSIStrategy
from genesis.risk.manager import RiskManager
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator
from genesis.execution.manager import ExecutionManager
from genesis.execution.order import Order, OrderType, OrderSide, OrderStatus
from genesis.exchanges.manager import ExchangeManager


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDataProvider(DataProvider):
    """Proveedor de datos simulado para pruebas."""
    
    def __init__(self, name="mock_provider"):
        """Inicializar proveedor con datos simulados."""
        super().__init__(name)
        self.available_symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        self.available_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Datos OHLCV simulados - tendencia alcista para generar señales de compra
        self.ohlcv_data = {
            "BTC/USDT": {
                "1h": self._generate_uptrend_data(50000, 24),  # 24 horas de datos con tendencia alcista
                "1d": self._generate_uptrend_data(50000, 30)   # 30 días de datos con tendencia alcista
            },
            "ETH/USDT": {
                "1h": self._generate_uptrend_data(3000, 24),
                "1d": self._generate_uptrend_data(3000, 30)
            }
        }
        
        # Datos de ticker simulados
        self.ticker_data = {
            "BTC/USDT": {"last": 50000, "bid": 49900, "ask": 50100, "volume": 100},
            "ETH/USDT": {"last": 3000, "bid": 2990, "ask": 3010, "volume": 200},
            "XRP/USDT": {"last": 0.5, "bid": 0.49, "ask": 0.51, "volume": 1000000}
        }
    
    def _generate_uptrend_data(self, start_price, n_candles):
        """Generar datos OHLCV con tendencia alcista."""
        data = []
        timestamp = 1609459200000  # 1 de enero de 2021 00:00:00 UTC
        current_price = start_price
        
        # Tendencia alcista
        for i in range(n_candles):
            # Incrementar el precio
            if i < n_candles // 2:
                # Primera mitad: ligera tendencia alcista
                price_change = current_price * 0.005  # +0.5% por vela
            else:
                # Segunda mitad: tendencia alcista más fuerte
                price_change = current_price * 0.01   # +1% por vela
            
            open_price = current_price
            close_price = current_price + price_change
            current_price = close_price
            
            # Calcular high y low
            high_price = close_price * 1.005  # 0.5% por encima del cierre
            low_price = open_price * 0.995    # 0.5% por debajo de la apertura
            
            # Calcular volumen (creciente con el precio)
            volume = 100 + (i * 10)
            
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


class MockExchange:
    """Exchange simulado para pruebas."""
    
    def __init__(self, name="mock_exchange"):
        """Inicializar exchange con datos simulados."""
        self.name = name
        self.orders = {}
        self.next_order_id = 1000
        
        # Simular tickers
        self.tickers = {
            "BTC/USDT": {"bid": 49000, "ask": 51000, "last": 50000},
            "ETH/USDT": {"bid": 2900, "ask": 3100, "last": 3000}
        }
        
        # Simular balances
        self.balances = {
            "BTC": {"free": 1.0, "used": 0.5, "total": 1.5},
            "ETH": {"free": 10.0, "used": 5.0, "total": 15.0},
            "USDT": {"free": 50000, "used": 10000, "total": 60000}
        }
    
    async def fetch_ticker(self, symbol):
        """Obtener ticker para un símbolo."""
        if symbol not in self.tickers:
            raise ValueError(f"Symbol {symbol} not found")
        return self.tickers[symbol]
    
    async def fetch_balance(self):
        """Obtener balance de la cuenta."""
        return self.balances
    
    async def create_limit_order(self, symbol, side, amount, price):
        """Crear una orden limitada."""
        order_id = str(self.next_order_id)
        self.next_order_id += 1
        
        # Crear orden
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": "limit",
            "price": price,
            "amount": amount,
            "filled": 0,
            "remaining": amount,
            "status": "open",
            "timestamp": int(time.time() * 1000)
        }
        
        # Guardar orden
        self.orders[order_id] = order
        
        # Simular actualización del balance
        self._update_balance_for_order(order)
        
        return order
    
    async def create_market_order(self, symbol, side, amount):
        """Crear una orden de mercado."""
        order_id = str(self.next_order_id)
        self.next_order_id += 1
        
        # Obtener precio actual
        ticker = self.tickers.get(symbol, {"bid": 0, "ask": 0})
        price = ticker["bid"] if side == "sell" else ticker["ask"]
        
        # Crear orden
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": "market",
            "price": price,
            "amount": amount,
            "filled": amount,  # Mercado se llena inmediatamente
            "remaining": 0,
            "status": "closed",
            "timestamp": int(time.time() * 1000)
        }
        
        # Guardar orden
        self.orders[order_id] = order
        
        # Simular actualización del balance
        self._update_balance_for_filled_order(order)
        
        return order
    
    async def cancel_order(self, order_id):
        """Cancelar una orden."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        if order["status"] == "closed":
            raise ValueError(f"Cannot cancel closed order {order_id}")
        
        # Actualizar estado
        order["status"] = "canceled"
        
        # Simular liberación de fondos
        self._release_funds_for_canceled_order(order)
        
        return order
    
    async def fetch_order(self, order_id):
        """Obtener detalles de una orden."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        return self.orders[order_id]
    
    async def fetch_open_orders(self, symbol=None):
        """Obtener órdenes abiertas."""
        open_orders = []
        for order in self.orders.values():
            if order["status"] == "open":
                if symbol is None or order["symbol"] == symbol:
                    open_orders.append(order)
        return open_orders
    
    def _update_balance_for_order(self, order):
        """Actualizar balance para reflejar la creación de orden."""
        symbol = order["symbol"]
        base, quote = symbol.split('/')
        
        if order["side"] == "buy":
            # Bloquear fondos en moneda quote
            cost = order["price"] * order["amount"]
            if quote in self.balances:
                self.balances[quote]["free"] -= cost
                self.balances[quote]["used"] += cost
        else:  # sell
            # Bloquear fondos en moneda base
            amount = order["amount"]
            if base in self.balances:
                self.balances[base]["free"] -= amount
                self.balances[base]["used"] += amount
    
    def _update_balance_for_filled_order(self, order):
        """Actualizar balance para reflejar orden completada."""
        symbol = order["symbol"]
        base, quote = symbol.split('/')
        
        if order["side"] == "buy":
            # Añadir moneda base, restar moneda quote
            cost = order["price"] * order["amount"]
            if base in self.balances:
                self.balances[base]["free"] += order["amount"]
                self.balances[base]["total"] += order["amount"]
            if quote in self.balances:
                self.balances[quote]["used"] -= cost
                self.balances[quote]["total"] -= cost
        else:  # sell
            # Añadir moneda quote, restar moneda base
            cost = order["price"] * order["amount"]
            if quote in self.balances:
                self.balances[quote]["free"] += cost
                self.balances[quote]["total"] += cost
            if base in self.balances:
                self.balances[base]["used"] -= order["amount"]
                self.balances[base]["total"] -= order["amount"]
    
    def _release_funds_for_canceled_order(self, order):
        """Liberar fondos bloqueados para una orden cancelada."""
        symbol = order["symbol"]
        base, quote = symbol.split('/')
        
        if order["side"] == "buy":
            # Liberar fondos en moneda quote
            cost = order["price"] * order["remaining"]
            if quote in self.balances:
                self.balances[quote]["free"] += cost
                self.balances[quote]["used"] -= cost
        else:  # sell
            # Liberar fondos en moneda base
            amount = order["remaining"]
            if base in self.balances:
                self.balances[base]["free"] += amount
                self.balances[base]["used"] -= amount


@pytest.fixture
def mock_data_provider():
    """Proporcionar un proveedor de datos simulado."""
    return MockDataProvider()


@pytest.fixture
def mock_exchange():
    """Proporcionar un exchange simulado."""
    return MockExchange()


@pytest.fixture
def event_bus():
    """Proporcionar un bus de eventos."""
    return EventBus()


@pytest.fixture
def trading_system(event_bus, mock_data_provider, mock_exchange):
    """Proporcionar un sistema de trading completo para pruebas."""
    # Crear componentes
    data_manager = DataManager()
    data_manager.register_provider(mock_data_provider)
    
    exchange_manager = ExchangeManager()
    exchange_manager.exchanges = {"mock_exchange": mock_exchange}
    
    indicators = TechnicalIndicators()
    signal_generator = SignalGenerator(indicators)
    
    strategy_manager = StrategyManager()
    
    # Registrar estrategias
    ma_strategy = MovingAverageStrategy(
        parameters={"fast_period": 5, "slow_period": 10}
    )
    rsi_strategy = RSIStrategy(
        parameters={"period": 14, "overbought": 70, "oversold": 30}
    )
    
    strategy_manager.register_strategy(ma_strategy)
    strategy_manager.register_strategy(rsi_strategy)
    
    # Componentes de riesgo
    position_sizer = PositionSizer()
    stop_loss_calculator = StopLossCalculator()
    risk_manager = RiskManager(position_sizer, stop_loss_calculator)
    
    # Gestor de ejecución
    execution_manager = ExecutionManager(event_bus, exchange_manager)
    
    # Vincular componentes para que puedan comunicarse
    components = {
        "data_manager": data_manager,
        "exchange_manager": exchange_manager,
        "indicators": indicators,
        "signal_generator": signal_generator,
        "strategy_manager": strategy_manager,
        "risk_manager": risk_manager,
        "execution_manager": execution_manager
    }
    
    # Crear motor y registrar componentes
    engine = Engine(event_bus=event_bus)
    
    # Configurar eventos entre componentes
    # Esta es una simplificación para pruebas; en un sistema real,
    # estos enlaces serían creados por los componentes en su startup
    
    return {
        "engine": engine,
        "components": components
    }


@pytest.mark.asyncio
async def test_full_trading_workflow(trading_system):
    """Probar flujo completo desde datos hasta ejecución."""
    engine = trading_system["engine"]
    components = trading_system["components"]
    
    # Extraer componentes para comodidad
    data_manager = components["data_manager"]
    strategy_manager = components["strategy_manager"]
    signal_generator = components["signal_generator"]
    risk_manager = components["risk_manager"]
    execution_manager = components["execution_manager"]
    
    # 1. Obtener datos históricos
    symbol = "BTC/USDT"
    timeframe = "1h"
    historical_data = await data_manager.get_historical_data(symbol, timeframe, limit=20)
    
    assert len(historical_data) > 0, "No se obtuvieron datos históricos"
    
    # 2. Generar señales a partir de los datos
    # Utilizar la estrategia de medias móviles
    ma_strategy = strategy_manager.get_strategy("MovingAverageStrategy")
    signal = await ma_strategy.generate_signal(symbol, historical_data)
    
    assert "signal" in signal, "La señal no contiene el campo 'signal'"
    
    # 3. Evaluar la señal con la gestión de riesgos
    # Configurar risk_manager para el test
    risk_manager.set_account_balance(50000)  # 50,000 USDT
    risk_manager.set_risk_percentage(1)     # 1% de riesgo por operación
    
    # Evaluar el riesgo de la operación
    risk_assessment = risk_manager.evaluate_trade(
        symbol=symbol,
        side=signal["signal"],
        current_price=historical_data[-1][4],  # Último precio de cierre
        stop_loss_price=historical_data[-1][4] * 0.95  # 5% por debajo del precio actual
    )
    
    assert "max_position_size" in risk_assessment, "La evaluación de riesgo no contiene el tamaño máximo de posición"
    
    # 4. Ejecutar la orden basada en la señal y evaluación de riesgo
    if signal["signal"] == "buy" and risk_assessment["approved"]:
        # Calcular cantidad a comprar
        price = historical_data[-1][4]  # Último precio de cierre
        amount = risk_assessment["max_position_size"] / price
        
        # Ejecutar orden
        order = await execution_manager.create_order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=amount,
            exchange_name="mock_exchange"
        )
        
        assert order is not None, "La orden no se creó correctamente"
        assert order.status == OrderStatus.FILLED, "La orden no se completó"
        
        # Verificar que se creó una posición
        assert symbol in execution_manager.positions, "No se creó una posición para BTC/USDT"
        position = execution_manager.positions[symbol]
        
        # 5. Configurar stop loss basado en el riesgo
        stop_loss_price = price * 0.95  # 5% por debajo del precio de entrada
        
        stop_loss_id = await execution_manager.create_stop_loss(
            position_id=position.id,
            price=stop_loss_price,
            exchange_name="mock_exchange"
        )
        
        assert stop_loss_id in execution_manager.conditional_orders, "No se creó la orden de stop loss"
    
    # Verificar flujo completo de eventos
    # (En un sistema real verificaríamos la ejecución exacta de todos los componentes)
    
    return True


@pytest.mark.asyncio
async def test_system_integration_with_events(trading_system):
    """Probar integración del sistema usando el bus de eventos."""
    engine = trading_system["engine"]
    components = trading_system["components"]
    event_bus = engine.event_bus
    
    # Registrar componentes en el motor
    for name, component in components.items():
        if isinstance(component, Component):
            engine.register_component(component)
        else:
            # Para componentes que no heredan de Component, creamos un wrapper
            class ComponentWrapper(Component):
                def __init__(self, name, wrapped):
                    super().__init__(name)
                    self.wrapped = wrapped
                
                async def start(self):
                    return True
                
                async def stop(self):
                    return True
                
                async def handle_event(self, event_type, data, source):
                    # Procesar evento según el tipo
                    if hasattr(self.wrapped, f"handle_{event_type}"):
                        handler = getattr(self.wrapped, f"handle_{event_type}")
                        return await handler(data, source)
                    return None
            
            wrapper = ComponentWrapper(name, component)
            engine.register_component(wrapper)
    
    # Iniciar el motor
    await engine.start()
    
    # Emitir evento de solicitud de datos
    await event_bus.emit(
        "request_market_data",
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "limit": 20
        },
        "test_component"
    )
    
    # En un sistema real, estos eventos serían manejados por los componentes
    # Para la prueba, simplemente verificamos que el bus de eventos funciona
    
    # Detener el motor
    await engine.stop()
    
    return True


@pytest.mark.asyncio
async def test_strategy_performance(trading_system):
    """Probar rendimiento de las estrategias en datos históricos."""
    components = trading_system["components"]
    data_manager = components["data_manager"]
    strategy_manager = components["strategy_manager"]
    
    # Obtener datos históricos para backtesting
    symbol = "BTC/USDT"
    timeframe = "1h"
    historical_data = await data_manager.get_historical_data(symbol, timeframe)
    
    # Probar cada estrategia
    results = {}
    
    for strategy_name in strategy_manager.strategies:
        strategy = strategy_manager.get_strategy(strategy_name)
        
        # Señales generadas
        signals = []
        
        # Recorrer los datos históricos punto por punto
        for i in range(30, len(historical_data)):
            # Usar una ventana de datos para cada punto
            window = historical_data[i-30:i]
            
            # Generar señal
            signal = await strategy.generate_signal(symbol, window)
            signals.append({
                "timestamp": window[-1][0],
                "price": window[-1][4],
                "signal": signal
            })
        
        # Contabilizar señales
        buy_signals = sum(1 for s in signals if s["signal"]["signal"] == "buy")
        sell_signals = sum(1 for s in signals if s["signal"]["signal"] == "sell")
        hold_signals = sum(1 for s in signals if s["signal"]["signal"] == "hold")
        
        results[strategy_name] = {
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
            "total_signals": len(signals)
        }
    
    # Verificar que se generaron señales
    for strategy_name, result in results.items():
        assert result["total_signals"] > 0, f"No se generaron señales para {strategy_name}"
    
    return results
"""