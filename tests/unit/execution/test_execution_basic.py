"""
Tests básicos para el módulo de ejecución.

Este módulo prueba las funcionalidades básicas del módulo de ejecución,
incluyendo la colocación de órdenes, seguimiento de posiciones y
manejo de errores.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from genesis.execution.manager import ExecutionManager
from genesis.execution.order import Order, OrderType, OrderSide, OrderStatus
from genesis.execution.position import Position
from genesis.exchanges.manager import ExchangeManager
from genesis.core.event_bus import EventBus


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
        self._update_balance_for_order(order)
        
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
def event_bus():
    """Proporcionar un bus de eventos para pruebas."""
    return EventBus()


@pytest.fixture
def mock_exchange():
    """Proporcionar un exchange simulado para pruebas."""
    return MockExchange()


@pytest.fixture
def exchange_manager(mock_exchange):
    """Proporcionar un gestor de exchanges con exchange simulado."""
    manager = ExchangeManager()
    manager.exchanges = {"mock_exchange": mock_exchange}
    return manager


@pytest.fixture
def execution_manager(event_bus, exchange_manager):
    """Proporcionar un gestor de ejecución."""
    return ExecutionManager(event_bus, exchange_manager)


@pytest.mark.asyncio
async def test_execution_manager_initialization(execution_manager):
    """Probar inicialización del gestor de ejecución."""
    # Verificar que se inicializa correctamente
    assert execution_manager is not None
    assert execution_manager.event_bus is not None
    assert execution_manager.exchange_manager is not None
    
    # Verificar que las colecciones están vacías
    assert len(execution_manager.orders) == 0
    assert len(execution_manager.positions) == 0


@pytest.mark.asyncio
async def test_create_limit_order(execution_manager, mock_exchange):
    """Probar creación de orden limitada."""
    # Crear una orden limitada
    order = await execution_manager.create_order(
        symbol="BTC/USDT",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        amount=0.1,
        price=50000,
        exchange_name="mock_exchange"
    )
    
    # Verificar que la orden se creó correctamente
    assert order is not None
    assert order.symbol == "BTC/USDT"
    assert order.order_type == OrderType.LIMIT
    assert order.side == OrderSide.BUY
    assert order.amount == 0.1
    assert order.price == 50000
    assert order.status == OrderStatus.OPEN
    
    # Verificar que se guardó en el gestor
    assert order.id in execution_manager.orders
    
    # Verificar que se creó en el exchange
    exchange_order = await mock_exchange.fetch_order(order.exchange_order_id)
    assert exchange_order is not None
    assert exchange_order["symbol"] == "BTC/USDT"
    assert exchange_order["side"] == "buy"
    assert exchange_order["amount"] == 0.1
    assert exchange_order["price"] == 50000


@pytest.mark.asyncio
async def test_create_market_order(execution_manager, mock_exchange):
    """Probar creación de orden de mercado."""
    # Crear una orden de mercado
    order = await execution_manager.create_order(
        symbol="ETH/USDT",
        order_type=OrderType.MARKET,
        side=OrderSide.SELL,
        amount=1.0,
        exchange_name="mock_exchange"
    )
    
    # Verificar que la orden se creó correctamente
    assert order is not None
    assert order.symbol == "ETH/USDT"
    assert order.order_type == OrderType.MARKET
    assert order.side == OrderSide.SELL
    assert order.amount == 1.0
    assert order.status == OrderStatus.FILLED  # Las órdenes de mercado se completan inmediatamente
    
    # Verificar que se guardó en el gestor
    assert order.id in execution_manager.orders
    
    # Verificar que se creó en el exchange
    exchange_order = await mock_exchange.fetch_order(order.exchange_order_id)
    assert exchange_order is not None
    assert exchange_order["symbol"] == "ETH/USDT"
    assert exchange_order["side"] == "sell"
    assert exchange_order["amount"] == 1.0
    assert exchange_order["type"] == "market"
    assert exchange_order["status"] == "closed"


@pytest.mark.asyncio
async def test_cancel_order(execution_manager, mock_exchange):
    """Probar cancelación de orden."""
    # Crear una orden para cancelar
    order = await execution_manager.create_order(
        symbol="BTC/USDT",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        amount=0.1,
        price=50000,
        exchange_name="mock_exchange"
    )
    
    # Cancelar la orden
    canceled = await execution_manager.cancel_order(order.id)
    
    # Verificar que la orden se canceló correctamente
    assert canceled is True
    assert execution_manager.orders[order.id].status == OrderStatus.CANCELED
    
    # Verificar en el exchange
    exchange_order = await mock_exchange.fetch_order(order.exchange_order_id)
    assert exchange_order["status"] == "canceled"


@pytest.mark.asyncio
async def test_get_order_status(execution_manager, mock_exchange):
    """Probar obtención de estado de orden."""
    # Crear una orden
    order = await execution_manager.create_order(
        symbol="BTC/USDT",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        amount=0.1,
        price=50000,
        exchange_name="mock_exchange"
    )
    
    # Verificar estado inicial
    assert order.status == OrderStatus.OPEN
    
    # Obtener estado actualizado
    updated_order = await execution_manager.get_order(order.id)
    
    # Verificar que es la misma orden
    assert updated_order.id == order.id
    assert updated_order.symbol == order.symbol
    assert updated_order.side == order.side


@pytest.mark.asyncio
async def test_manage_position(execution_manager, mock_exchange):
    """Probar gestión de posiciones."""
    # Crear una orden de compra
    buy_order = await execution_manager.create_order(
        symbol="BTC/USDT",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        amount=0.1,
        exchange_name="mock_exchange"
    )
    
    # Verificar que se creó una posición
    assert "BTC/USDT" in execution_manager.positions
    position = execution_manager.positions["BTC/USDT"]
    
    # Verificar datos de la posición
    assert position.symbol == "BTC/USDT"
    assert position.amount == 0.1
    assert position.entry_price > 0
    assert position.is_long is True
    
    # Crear una orden de venta parcial
    sell_order = await execution_manager.create_order(
        symbol="BTC/USDT",
        order_type=OrderType.MARKET,
        side=OrderSide.SELL,
        amount=0.05,  # Vender la mitad
        exchange_name="mock_exchange"
    )
    
    # Verificar actualización de la posición
    position = execution_manager.positions["BTC/USDT"]
    assert position.amount == 0.05  # 0.1 - 0.05
    
    # Cerrar la posición completamente
    close_order = await execution_manager.create_order(
        symbol="BTC/USDT",
        order_type=OrderType.MARKET,
        side=OrderSide.SELL,
        amount=0.05,  # Vender el resto
        exchange_name="mock_exchange"
    )
    
    # Verificar que la posición se cerró
    assert "BTC/USDT" not in execution_manager.positions


@pytest.mark.asyncio
async def test_execution_error_handling(execution_manager, mock_exchange):
    """Probar manejo de errores en ejecución."""
    # Intentar crear orden con símbolo inválido
    with pytest.raises(ValueError):
        await execution_manager.create_order(
            symbol="INVALID/PAIR",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=0.1,
            exchange_name="mock_exchange"
        )
    
    # Crear una orden y luego intentar cancelarla dos veces
    order = await execution_manager.create_order(
        symbol="BTC/USDT",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        amount=0.1,
        price=50000,
        exchange_name="mock_exchange"
    )
    
    # Primera cancelación (debería funcionar)
    assert await execution_manager.cancel_order(order.id) is True
    
    # Segunda cancelación (debería fallar porque la orden ya está cancelada)
    assert await execution_manager.cancel_order(order.id) is False


@pytest.mark.asyncio
async def test_get_open_orders(execution_manager, mock_exchange):
    """Probar obtención de órdenes abiertas."""
    # Crear varias órdenes límite
    orders = []
    for i in range(3):
        order = await execution_manager.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=0.1,
            price=49000 + i * 100,  # Precios diferentes
            exchange_name="mock_exchange"
        )
        orders.append(order)
    
    # Cancelar una orden
    await execution_manager.cancel_order(orders[0].id)
    
    # Obtener órdenes abiertas
    open_orders = await execution_manager.get_open_orders(symbol="BTC/USDT")
    
    # Verificar que solo hay dos órdenes abiertas
    assert len(open_orders) == 2
    
    # Verificar que son las órdenes correctas
    open_order_ids = [order.id for order in open_orders]
    assert orders[1].id in open_order_ids
    assert orders[2].id in open_order_ids
    assert orders[0].id not in open_order_ids  # Esta fue cancelada


@pytest.mark.asyncio
async def test_get_positions(execution_manager, mock_exchange):
    """Probar obtención de posiciones abiertas."""
    # Crear posiciones en diferentes símbolos
    await execution_manager.create_order(
        symbol="BTC/USDT",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        amount=0.1,
        exchange_name="mock_exchange"
    )
    
    await execution_manager.create_order(
        symbol="ETH/USDT",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        amount=1.0,
        exchange_name="mock_exchange"
    )
    
    # Verificar posiciones
    positions = execution_manager.get_positions()
    
    # Debería haber dos posiciones
    assert len(positions) == 2
    
    # Verificar detalles
    btc_position = next(p for p in positions if p.symbol == "BTC/USDT")
    eth_position = next(p for p in positions if p.symbol == "ETH/USDT")
    
    assert btc_position.amount == 0.1
    assert eth_position.amount == 1.0
    assert btc_position.is_long
    assert eth_position.is_long
"""