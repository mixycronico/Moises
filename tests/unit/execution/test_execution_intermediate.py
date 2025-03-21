"""
Tests intermedios para el módulo de ejecución.

Este módulo prueba funcionalidades más complejas del módulo de ejecución,
incluyendo órdenes condicionadas, división de órdenes y manejo de fallos.
"""

import pytest
import asyncio
import time
import random
from unittest.mock import Mock, patch, AsyncMock

from genesis.execution.manager import ExecutionManager
from genesis.execution.order import Order, OrderType, OrderSide, OrderStatus
from genesis.execution.position import Position
from genesis.execution.strategies import (
    ExecutionStrategy, 
    SimpleExecutionStrategy,
    TwapExecutionStrategy,
    IcebergExecutionStrategy
)
from genesis.exchanges.manager import ExchangeManager
from genesis.core.event_bus import EventBus


class DelayedMockExchange:
    """Exchange simulado con retrasos y fallas aleatorias para pruebas."""
    
    def __init__(self, name="delayed_mock_exchange", delay_range=(0.01, 0.1), error_rate=0.1):
        """Inicializar exchange con datos simulados."""
        self.name = name
        self.orders = {}
        self.next_order_id = 1000
        self.delay_range = delay_range
        self.error_rate = error_rate
        
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
        
        # Historial de órdenes para seguimiento
        self.order_history = []
    
    async def _simulate_delay(self):
        """Simular retraso en la red."""
        delay = random.uniform(*self.delay_range)
        await asyncio.sleep(delay)
        
        # Simular error aleatorio
        if random.random() < self.error_rate:
            raise Exception("Simulated exchange error")
    
    async def _simulate_price_movement(self):
        """Simular movimiento de precios."""
        for symbol in self.tickers:
            price = self.tickers[symbol]["last"]
            # Movimiento aleatorio de hasta +/- 0.5%
            change = price * random.uniform(-0.005, 0.005)
            new_price = price + change
            
            # Actualizar ticker
            self.tickers[symbol]["last"] = new_price
            self.tickers[symbol]["bid"] = new_price * 0.998
            self.tickers[symbol]["ask"] = new_price * 1.002
    
    async def fetch_ticker(self, symbol):
        """Obtener ticker para un símbolo."""
        await self._simulate_delay()
        await self._simulate_price_movement()
        
        if symbol not in self.tickers:
            raise ValueError(f"Symbol {symbol} not found")
        return self.tickers[symbol]
    
    async def fetch_balance(self):
        """Obtener balance de la cuenta."""
        await self._simulate_delay()
        return self.balances
    
    async def create_limit_order(self, symbol, side, amount, price):
        """Crear una orden limitada."""
        await self._simulate_delay()
        
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
        self.order_history.append({
            "action": "create",
            "order": order.copy(),
            "timestamp": time.time()
        })
        
        # Simular actualización del balance
        self._update_balance_for_order(order)
        
        # Simular ejecución parcial aleatoria
        await self._simulate_partial_execution(order_id)
        
        return order
    
    async def create_market_order(self, symbol, side, amount):
        """Crear una orden de mercado."""
        await self._simulate_delay()
        
        order_id = str(self.next_order_id)
        self.next_order_id += 1
        
        # Obtener precio actual
        ticker = self.tickers.get(symbol, {"bid": 0, "ask": 0})
        price = ticker["bid"] if side == "sell" else ticker["ask"]
        
        # Simular slippage
        if side == "buy":
            price *= (1 + random.uniform(0, 0.005))  # 0-0.5% peor
        else:
            price *= (1 - random.uniform(0, 0.005))  # 0-0.5% peor
        
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
        self.order_history.append({
            "action": "create",
            "order": order.copy(),
            "timestamp": time.time()
        })
        
        # Simular actualización del balance
        self._update_balance_for_filled_order(order)
        
        return order
    
    async def cancel_order(self, order_id):
        """Cancelar una orden."""
        await self._simulate_delay()
        
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        if order["status"] == "closed":
            raise ValueError(f"Cannot cancel closed order {order_id}")
        
        # Actualizar estado
        order["status"] = "canceled"
        self.order_history.append({
            "action": "cancel",
            "order": order.copy(),
            "timestamp": time.time()
        })
        
        # Simular liberación de fondos
        self._release_funds_for_canceled_order(order)
        
        return order
    
    async def fetch_order(self, order_id):
        """Obtener detalles de una orden."""
        await self._simulate_delay()
        
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        # Simular actualización de estado
        await self._simulate_order_update(order_id)
        
        return self.orders[order_id]
    
    async def fetch_open_orders(self, symbol=None):
        """Obtener órdenes abiertas."""
        await self._simulate_delay()
        
        # Actualizar órdenes antes de devolver
        for order_id in list(self.orders.keys()):
            await self._simulate_order_update(order_id)
        
        open_orders = []
        for order in self.orders.values():
            if order["status"] == "open":
                if symbol is None or order["symbol"] == symbol:
                    open_orders.append(order)
        return open_orders
    
    async def _simulate_partial_execution(self, order_id):
        """Simular ejecución parcial de una orden limitada."""
        if random.random() < 0.3:  # 30% de probabilidad de ejecución parcial
            order = self.orders[order_id]
            if order["status"] == "open" and order["type"] == "limit":
                # Simular ejecución parcial de hasta 75% de la orden
                max_fill = order["amount"] * 0.75
                partial_fill = random.uniform(0, max_fill)
                
                # Actualizar orden
                order["filled"] += partial_fill
                order["remaining"] -= partial_fill
                
                # Si se llenó completamente, cambiar estado
                if order["remaining"] < 0.00001:
                    order["status"] = "closed"
                    order["remaining"] = 0
                    order["filled"] = order["amount"]
                
                # Registrar actualización
                self.order_history.append({
                    "action": "update",
                    "order": order.copy(),
                    "timestamp": time.time()
                })
                
                # Actualizar balance
                self._update_balance_for_partial_fill(order, partial_fill)
    
    async def _simulate_order_update(self, order_id):
        """Simular actualización de estado de orden."""
        if order_id in self.orders:
            order = self.orders[order_id]
            
            # Solo actualizar órdenes abiertas
            if order["status"] == "open":
                # Probabilidad de actualización
                if random.random() < 0.2:  # 20% de probabilidad de actualización
                    await self._simulate_partial_execution(order_id)
    
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
    
    def _update_balance_for_partial_fill(self, order, partial_fill):
        """Actualizar balance para reflejar llenado parcial."""
        symbol = order["symbol"]
        base, quote = symbol.split('/')
        
        if order["side"] == "buy":
            # Añadir moneda base, disminuir moneda quote bloqueada
            cost = order["price"] * partial_fill
            if base in self.balances:
                self.balances[base]["free"] += partial_fill
                self.balances[base]["total"] += partial_fill
            if quote in self.balances:
                self.balances[quote]["used"] -= cost
                self.balances[quote]["total"] -= cost
        else:  # sell
            # Añadir moneda quote, disminuir moneda base bloqueada
            cost = order["price"] * partial_fill
            if quote in self.balances:
                self.balances[quote]["free"] += cost
                self.balances[quote]["total"] += cost
            if base in self.balances:
                self.balances[base]["used"] -= partial_fill
                self.balances[base]["total"] -= partial_fill
    
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
    return DelayedMockExchange()


@pytest.fixture
def exchange_manager(mock_exchange):
    """Proporcionar un gestor de exchanges con exchange simulado."""
    manager = ExchangeManager()
    manager.exchanges = {"mock_exchange": mock_exchange}
    return manager


@pytest.fixture
def execution_manager(event_bus, exchange_manager):
    """Proporcionar un gestor de ejecución."""
    manager = ExecutionManager(event_bus, exchange_manager)
    return manager


@pytest.mark.asyncio
async def test_retry_mechanism(execution_manager, mock_exchange):
    """Probar mecanismo de reintento para órdenes fallidas."""
    # Aumentar la tasa de error para esta prueba
    mock_exchange.error_rate = 0.8  # 80% de probabilidad de error
    
    # Configurar el gestor de ejecución para reintentar
    execution_manager.max_retries = 5
    execution_manager.retry_delay = 0.01
    
    # Crear una orden que probablemente fallará en el primer intento
    order = await execution_manager.create_order(
        symbol="BTC/USDT",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        amount=0.1,
        price=50000,
        exchange_name="mock_exchange"
    )
    
    # Verificar que la orden se creó eventualmente
    assert order is not None
    assert order.symbol == "BTC/USDT"
    
    # Restaurar tasa de error
    mock_exchange.error_rate = 0.1


@pytest.mark.asyncio
async def test_conditional_order(execution_manager, mock_exchange):
    """Probar órdenes condicionadas."""
    # Crear una orden condicionada a que el precio baje a cierto nivel
    condition = {
        "type": "price",
        "symbol": "BTC/USDT",
        "operator": "<",
        "value": 49500  # Ejecutar cuando BTC/USDT baje de 49500
    }
    
    # Registrar la orden condicionada
    order_params = {
        "symbol": "BTC/USDT",
        "order_type": OrderType.MARKET,
        "side": OrderSide.BUY,
        "amount": 0.1,
        "exchange_name": "mock_exchange"
    }
    
    conditional_id = await execution_manager.register_conditional_order(
        condition=condition,
        order_params=order_params
    )
    
    # Verificar que se registró
    assert conditional_id in execution_manager.conditional_orders
    
    # Simular que el precio baja por debajo del umbral
    mock_exchange.tickers["BTC/USDT"]["last"] = 49400
    mock_exchange.tickers["BTC/USDT"]["bid"] = 49350
    mock_exchange.tickers["BTC/USDT"]["ask"] = 49450
    
    # Activar el mecanismo de verificación de condiciones
    await execution_manager.check_conditional_orders()
    
    # Verificar que la orden se ejecutó
    # La orden debería haberse creado y la condicional eliminada
    assert conditional_id not in execution_manager.conditional_orders
    
    # Debería haber una orden nueva correspondiente a la condición
    order_found = False
    for order_id, order in execution_manager.orders.items():
        if (order.symbol == "BTC/USDT" and 
            order.order_type == OrderType.MARKET and
            order.side == OrderSide.BUY and
            order.amount == 0.1):
            order_found = True
            break
    
    assert order_found


@pytest.mark.asyncio
async def test_strategy_twap_execution(execution_manager, mock_exchange):
    """Probar estrategia de ejecución TWAP (Time-Weighted Average Price)."""
    # Crear una estrategia TWAP
    twap_strategy = TwapExecutionStrategy(
        num_slices=5,
        time_interval=0.05  # Intervalo corto para pruebas
    )
    
    # Ejecutar una orden grande usando TWAP
    total_amount = 0.5  # BTC
    
    # Iniciar ejecución TWAP
    execution_id = await execution_manager.execute_with_strategy(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=total_amount,
        strategy=twap_strategy,
        exchange_name="mock_exchange"
    )
    
    # Esperar a que termine la ejecución
    await asyncio.sleep(0.5)  # (5 slices * 0.05s interval) + margen
    
    # Verificar resultados
    execution_result = execution_manager.get_strategy_execution_result(execution_id)
    
    # Verificar que la estrategia está completada
    assert execution_result["status"] == "completed"
    
    # Verificar que se ejecutaron todas las órdenes
    assert len(execution_result["orders"]) == 5
    
    # Verificar que se ejecutó aproximadamente la cantidad correcta
    total_executed = sum(order.filled_amount for order in execution_result["orders"])
    assert abs(total_executed - total_amount) < 0.001
    
    # Verificar que las órdenes se ejecutaron a lo largo del tiempo
    timestamps = [order.timestamp for order in execution_result["orders"]]
    timestamps.sort()
    
    # Verificar que hay al menos algo de espaciado entre las órdenes
    for i in range(1, len(timestamps)):
        assert timestamps[i] - timestamps[i-1] > 0.01


@pytest.mark.asyncio
async def test_strategy_iceberg_execution(execution_manager, mock_exchange):
    """Probar estrategia de ejecución Iceberg (ocultar el volumen real)."""
    # Crear una estrategia Iceberg
    iceberg_strategy = IcebergExecutionStrategy(
        visible_size=0.02,  # Mostrar solo 0.02 BTC por orden
        price_offset=0.001  # Ligero offset en precio para mantener órdenes en rango
    )
    
    # Ejecutar una orden grande usando Iceberg
    total_amount = 0.1  # BTC
    
    # Iniciar ejecución Iceberg
    execution_id = await execution_manager.execute_with_strategy(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=total_amount,
        price=50000,  # Precio límite
        strategy=iceberg_strategy,
        exchange_name="mock_exchange"
    )
    
    # Esperar a que termine la ejecución
    # Para cada orden visible, el sistema colocará una nueva después de que se llene
    await asyncio.sleep(0.5)  # Tiempo suficiente para que se ejecute la estrategia
    
    # Verificar resultados
    execution_result = execution_manager.get_strategy_execution_result(execution_id)
    
    # Verificar que la estrategia está completada o en progreso
    assert execution_result["status"] in ["completed", "in_progress"]
    
    # Verificar que se ejecutaron varias órdenes pequeñas
    assert len(execution_result["orders"]) >= 2
    
    # Verificar que cada orden visible es pequeña
    for order in execution_result["orders"]:
        assert order.amount <= 0.02 + 0.001  # Tamaño visible + tolerancia
    
    # Verificar que el total ejecutado se acerca al objetivo
    total_executed = sum(order.filled_amount for order in execution_result["orders"])
    
    # La estrategia puede estar completada o en progreso
    if execution_result["status"] == "completed":
        assert abs(total_executed - total_amount) < 0.001
    else:
        assert total_executed < total_amount


@pytest.mark.asyncio
async def test_concurrent_order_execution(execution_manager, mock_exchange):
    """Probar ejecución concurrente de múltiples órdenes."""
    # Crear varias órdenes concurrentemente
    num_orders = 10
    order_futures = []
    
    for i in range(num_orders):
        future = execution_manager.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            amount=0.01,
            price=50000 - (i * 100) if i % 2 == 0 else 50000 + (i * 100),
            exchange_name="mock_exchange"
        )
        order_futures.append(future)
    
    # Esperar a que todas las órdenes se ejecuten
    orders = await asyncio.gather(*order_futures, return_exceptions=True)
    
    # Verificar resultados
    successful_orders = [order for order in orders if not isinstance(order, Exception)]
    
    # Debería haber al menos algunas órdenes exitosas
    assert len(successful_orders) > 0
    
    # Verificar que las órdenes exitosas están en el gestor
    for order in successful_orders:
        assert order.id in execution_manager.orders


@pytest.mark.asyncio
async def test_stop_loss_take_profit(execution_manager, mock_exchange):
    """Probar órdenes de stop loss y take profit."""
    # Crear una posición
    buy_order = await execution_manager.create_order(
        symbol="BTC/USDT",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        amount=0.1,
        exchange_name="mock_exchange"
    )
    
    assert "BTC/USDT" in execution_manager.positions
    position = execution_manager.positions["BTC/USDT"]
    entry_price = position.entry_price
    
    # Configurar stop loss (5% por debajo de entrada)
    stop_loss_price = entry_price * 0.95
    
    stop_loss_id = await execution_manager.create_stop_loss(
        position_id=position.id,
        price=stop_loss_price,
        exchange_name="mock_exchange"
    )
    
    # Configurar take profit (10% por encima de entrada)
    take_profit_price = entry_price * 1.10
    
    take_profit_id = await execution_manager.create_take_profit(
        position_id=position.id,
        price=take_profit_price,
        exchange_name="mock_exchange"
    )
    
    # Verificar que se crearon las órdenes condicionadas
    assert stop_loss_id in execution_manager.conditional_orders
    assert take_profit_id in execution_manager.conditional_orders
    
    # Simular que el precio sube por encima del take profit
    mock_exchange.tickers["BTC/USDT"]["last"] = take_profit_price * 1.01
    mock_exchange.tickers["BTC/USDT"]["bid"] = take_profit_price
    mock_exchange.tickers["BTC/USDT"]["ask"] = take_profit_price * 1.02
    
    # Activar verificación de condiciones
    await execution_manager.check_conditional_orders()
    
    # El take profit debería haberse ejecutado y el stop loss cancelado
    assert take_profit_id not in execution_manager.conditional_orders
    assert stop_loss_id not in execution_manager.conditional_orders
    
    # La posición debería haberse cerrado
    assert "BTC/USDT" not in execution_manager.positions


@pytest.mark.asyncio
async def test_execution_recovery(execution_manager, mock_exchange):
    """Probar recuperación de estado de ejecución después de fallos."""
    # Crear algunas órdenes
    orders = []
    for i in range(3):
        order = await execution_manager.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=0.01,
            price=49000 + (i * 100),
            exchange_name="mock_exchange"
        )
        orders.append(order)
    
    # Simular un fallo del sistema guardando el estado actual
    state_data = execution_manager.save_state()
    
    # Crear un nuevo gestor y restaurar estado
    new_manager = ExecutionManager(execution_manager.event_bus, execution_manager.exchange_manager)
    new_manager.load_state(state_data)
    
    # Verificar que todas las órdenes se restauraron
    for order in orders:
        assert order.id in new_manager.orders
        restored_order = new_manager.orders[order.id]
        assert restored_order.symbol == order.symbol
        assert restored_order.order_type == order.order_type
        assert restored_order.side == order.side
        assert restored_order.amount == order.amount
        assert restored_order.price == order.price
    
    # Verificar que el nuevo gestor puede seguir operando
    # Cancelar una orden en el nuevo gestor
    await new_manager.cancel_order(orders[0].id)
    
    # Verificar cancleación
    assert new_manager.orders[orders[0].id].status == OrderStatus.CANCELED
"""