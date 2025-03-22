"""
Tests avanzados para el módulo de ejecución.

Este módulo prueba funcionalidades avanzadas del módulo de ejecución,
incluyendo alta concurrencia, manejo de órdenes complejas, condiciones
extremas de mercado, algoritmos sofisticados de ejecución y
optimización de costos de trading.
"""

import pytest
import asyncio
import time
import random
import threading
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from genesis.execution.manager import ExecutionManager
from genesis.execution.order import Order, OrderType, OrderSide, OrderStatus
from genesis.execution.position import Position
from genesis.execution.strategies import (
    ExecutionStrategy, 
    SimpleExecutionStrategy,
    TwapExecutionStrategy,
    VwapExecutionStrategy,
    IcebergExecutionStrategy,
    SmartExecutionStrategy,
    AdaptiveExecutionStrategy
)
from genesis.core.event_bus import EventBus
from genesis.exchange.manager import ExchangeManager
from genesis.exchange.exchange import Exchange
from genesis.risk.manager import RiskManager


class MockExchange(Exchange):
    """Exchange simulado que emula condiciones complejas de mercado."""
    
    def __init__(self, name, exchange_id="mock", 
                 latency_ms=50, failure_rate=0.1, 
                 spread_bps=10, 
                 volatility=0.05,
                 slippage_model="normal"):
        """
        Inicializa el exchange simulado.
        
        Args:
            name: Nombre del exchange
            exchange_id: ID del exchange
            latency_ms: Latencia media en ms
            failure_rate: Tasa de fallos (0-1)
            spread_bps: Spread en puntos básicos
            volatility: Volatilidad del precio
            slippage_model: Modelo de slippage ("normal", "extreme", "none")
        """
        super().__init__(name, exchange_id)
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.spread_bps = spread_bps
        self.volatility = volatility
        self.slippage_model = slippage_model
        
        # Estado interno
        self._market_data = {}
        self._order_books = {}
        self._position = {}
        self._balances = {"USDT": 100000.0, "BTC": 5.0, "ETH": 50.0}
        self._orders = {}
        self._next_order_id = 1
        
        # Simulación de condiciones de mercado
        self._base_prices = {
            "BTC/USDT": 45000.0,
            "ETH/USDT": 3000.0,
            "XRP/USDT": 0.5,
            "ADA/USDT": 1.2,
            "DOT/USDT": 30.0
        }
        
        # Iniciar simulación de mercado
        self._last_update = time.time()
        self._initialize_order_books()
    
    def _initialize_order_books(self):
        """Inicializa libros de órdenes simulados para símbolos comunes."""
        for symbol, base_price in self._base_prices.items():
            # Crear un libro de órdenes básico con spread
            spread = base_price * (self.spread_bps / 10000)
            bid_price = base_price - spread / 2
            ask_price = base_price + spread / 2
            
            # Generar algunas órdenes alrededor del precio medio
            bids = []
            asks = []
            
            # Generar 10 niveles de profundidad
            for i in range(10):
                # Precio desciende para bids, aumenta para asks
                bid_level_price = bid_price - i * spread * 0.5
                ask_level_price = ask_price + i * spread * 0.5
                
                # Volumen aleatorio con más volumen cerca del precio medio
                bid_volume = random.uniform(0.1, 3.0) * (1.0 - i * 0.08)
                ask_volume = random.uniform(0.1, 3.0) * (1.0 - i * 0.08)
                
                bids.append([bid_level_price, bid_volume])
                asks.append([ask_level_price, ask_volume])
            
            self._order_books[symbol] = {
                "bids": bids,
                "asks": asks,
                "timestamp": int(time.time() * 1000)
            }
            
            # Inicializar datos de mercado
            self._market_data[symbol] = {
                "price": base_price,
                "bid": bid_price,
                "ask": ask_price,
                "volume_24h": random.uniform(1000, 10000),
                "timestamp": int(time.time() * 1000)
            }
    
    def _update_market(self, symbol):
        """Actualiza precios del mercado simulado con volatilidad."""
        now = time.time()
        elapsed_time = now - self._last_update
        
        if symbol not in self._base_prices:
            return
            
        base_price = self._base_prices[symbol]
        
        # Actualizar precio con camino aleatorio
        price_change = np.random.normal(0, self.volatility * np.sqrt(elapsed_time)) * base_price
        new_price = base_price + price_change
        
        # Asegurar que el precio no baje de 0
        new_price = max(0.000001, new_price)
        
        # Actualizar precio base
        self._base_prices[symbol] = new_price
        
        # Actualizar libro de órdenes
        self._initialize_order_books()  # Simplificado, regenera todo el libro
        
        self._last_update = now
    
    async def get_ticker(self, symbol):
        """Obtiene datos de ticker para un símbolo."""
        await self._simulate_latency()
        self._update_market(symbol)
        
        if symbol not in self._market_data:
            raise Exception(f"Symbol {symbol} not found")
            
        return self._market_data[symbol]
    
    async def get_order_book(self, symbol, limit=10):
        """Obtiene el libro de órdenes para un símbolo."""
        await self._simulate_latency()
        self._update_market(symbol)
        
        if symbol not in self._order_books:
            raise Exception(f"Order book for {symbol} not found")
            
        book = self._order_books[symbol]
        
        # Limitar la profundidad devuelta
        return {
            "bids": book["bids"][:limit],
            "asks": book["asks"][:limit],
            "timestamp": int(time.time() * 1000)
        }
    
    async def get_balance(self, currency=None):
        """Obtiene el saldo de una moneda o todas las monedas."""
        await self._simulate_latency()
        
        if self._should_fail():
            raise Exception("Simulated balance retrieval failure")
            
        if currency:
            return {currency: self._balances.get(currency, 0.0)}
        else:
            return self._balances
    
    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        """Crea una orden en el exchange simulado."""
        await self._simulate_latency()
        
        if self._should_fail():
            raise Exception("Simulated order creation failure")
            
        # Actualizar mercado
        self._update_market(symbol)
        
        # Obtener datos actuales de mercado
        ticker = self._market_data.get(symbol)
        if not ticker:
            raise Exception(f"Symbol {symbol} not found")
            
        # Determinar precio de ejecución
        exec_price = None
        
        if type == "market":
            # Para órdenes de mercado, simular slippage
            base_price = ticker["ask"] if side == "buy" else ticker["bid"]
            exec_price = self._apply_slippage(base_price, side)
        elif type == "limit":
            # Para órdenes límite, usar el precio especificado
            if price is None:
                raise Exception("Price is required for limit orders")
            exec_price = price
        else:
            raise Exception(f"Unsupported order type: {type}")
        
        # Calcular costos
        cost = amount * exec_price
        fee = cost * 0.001  # 0.1% fee
        
        # Verificar fondos suficientes
        base, quote = symbol.split('/')
        
        if side == "buy":
            if self._balances.get(quote, 0) < cost + fee:
                raise Exception(f"Insufficient {quote} balance")
        else:  # sell
            if self._balances.get(base, 0) < amount:
                raise Exception(f"Insufficient {base} balance")
        
        # Crear la orden
        order_id = str(self._next_order_id)
        self._next_order_id += 1
        
        # Determinar si la orden se ejecuta inmediatamente
        is_filled = False
        filled = 0.0
        
        if type == "market":
            # Las órdenes de mercado se ejecutan inmediatamente
            is_filled = True
            filled = amount
            
            # Actualizar balances
            if side == "buy":
                self._balances[quote] = self._balances.get(quote, 0) - cost - fee
                self._balances[base] = self._balances.get(base, 0) + amount
            else:  # sell
                self._balances[base] = self._balances.get(base, 0) - amount
                self._balances[quote] = self._balances.get(quote, 0) + cost - fee
        elif type == "limit":
            # Determinar si la orden límite se ejecutaría inmediatamente
            if side == "buy" and price >= ticker["ask"]:
                is_filled = True
                filled = amount
                
                # Actualizar balances
                self._balances[quote] = self._balances.get(quote, 0) - cost - fee
                self._balances[base] = self._balances.get(base, 0) + amount
            elif side == "sell" and price <= ticker["bid"]:
                is_filled = True
                filled = amount
                
                # Actualizar balances
                self._balances[base] = self._balances.get(base, 0) - amount
                self._balances[quote] = self._balances.get(quote, 0) + cost - fee
        
        # Crear objeto de orden
        order = {
            "id": order_id,
            "symbol": symbol,
            "type": type,
            "side": side,
            "price": exec_price,
            "amount": amount,
            "cost": cost,
            "fee": {"cost": fee, "currency": quote},
            "filled": filled,
            "status": "closed" if is_filled else "open",
            "timestamp": int(time.time() * 1000)
        }
        
        # Guardar la orden
        self._orders[order_id] = order
        
        return order
    
    async def cancel_order(self, id, symbol=None, params={}):
        """Cancela una orden existente."""
        await self._simulate_latency()
        
        if self._should_fail():
            raise Exception("Simulated order cancellation failure")
            
        if id not in self._orders:
            raise Exception(f"Order {id} not found")
            
        order = self._orders[id]
        
        # Solo se pueden cancelar órdenes abiertas
        if order["status"] != "open":
            raise Exception(f"Cannot cancel order with status {order['status']}")
            
        # Actualizar estado
        order["status"] = "canceled"
        
        return order
    
    async def fetch_order(self, id, symbol=None, params={}):
        """Obtiene información de una orden."""
        await self._simulate_latency()
        
        if self._should_fail():
            raise Exception("Simulated order fetch failure")
            
        if id not in self._orders:
            raise Exception(f"Order {id} not found")
            
        return self._orders[id]
    
    async def fetch_orders(self, symbol=None, since=None, limit=None, params={}):
        """Obtiene todas las órdenes."""
        await self._simulate_latency()
        
        if self._should_fail():
            raise Exception("Simulated orders fetch failure")
            
        orders = list(self._orders.values())
        
        # Filtrar por símbolo si se especifica
        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]
            
        # Filtrar por fecha si se especifica
        if since:
            orders = [o for o in orders if o["timestamp"] >= since]
            
        # Limitar resultados si se especifica
        if limit:
            orders = orders[:limit]
            
        return orders
    
    async def fetch_open_orders(self, symbol=None, since=None, limit=None, params={}):
        """Obtiene órdenes abiertas."""
        await self._simulate_latency()
        
        if self._should_fail():
            raise Exception("Simulated open orders fetch failure")
            
        orders = [o for o in self._orders.values() if o["status"] == "open"]
        
        # Filtrar por símbolo si se especifica
        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]
            
        # Filtrar por fecha si se especifica
        if since:
            orders = [o for o in orders if o["timestamp"] >= since]
            
        # Limitar resultados si se especifica
        if limit:
            orders = orders[:limit]
            
        return orders
    
    async def _simulate_latency(self):
        """Simula latencia de red."""
        # Convertir ms a segundos y añadir variabilidad
        latency = self.latency_ms / 1000
        jitter = random.uniform(-0.5, 1.0) * latency * 0.2
        await asyncio.sleep(latency + jitter)
    
    def _should_fail(self):
        """Determina si una operación debería fallar basado en la tasa de fallos."""
        return random.random() < self.failure_rate
    
    def _apply_slippage(self, base_price, side):
        """Aplica slippage al precio según el modelo elegido."""
        if self.slippage_model == "none":
            return base_price
            
        # Calcular slippage base
        base_slippage_pct = 0.0
        
        if self.slippage_model == "normal":
            # Distribución normal centrada en 0
            base_slippage_pct = np.random.normal(0, 0.001)  # 0.1% std dev
        elif self.slippage_model == "extreme":
            # Distribución con cola pesada
            if random.random() < 0.05:  # 5% de probabilidad de slippage extremo
                base_slippage_pct = np.random.gamma(2, 0.005)  # Más slippage positivo
            else:
                base_slippage_pct = np.random.normal(0, 0.001)
        
        # Ajustar dirección del slippage según el lado de la orden
        if side == "buy":
            # Para compras, el slippage positivo es desfavorable (precio más alto)
            slippage_pct = abs(base_slippage_pct)
        else:  # sell
            # Para ventas, el slippage negativo es desfavorable (precio más bajo)
            slippage_pct = -abs(base_slippage_pct)
        
        # Aplicar slippage
        return base_price * (1 + slippage_pct)


@pytest.fixture
def event_bus():
    """Proporciona un bus de eventos para pruebas."""
    return EventBus()


@pytest.fixture
def mock_exchanges():
    """Proporciona exchanges simulados con diferentes características."""
    return [
        MockExchange(
            name="reliable_exchange",
            exchange_id="reliable",
            latency_ms=30,
            failure_rate=0.05,
            spread_bps=8,
            volatility=0.03,
            slippage_model="normal"
        ),
        MockExchange(
            name="volatile_exchange",
            exchange_id="volatile",
            latency_ms=50,
            failure_rate=0.1,
            spread_bps=15,
            volatility=0.08,
            slippage_model="extreme"
        ),
        MockExchange(
            name="high_latency_exchange",
            exchange_id="slow",
            latency_ms=200,
            failure_rate=0.2,
            spread_bps=12,
            volatility=0.04,
            slippage_model="normal"
        )
    ]


@pytest.fixture
def exchange_manager(mock_exchanges, event_bus):
    """Proporciona un gestor de exchanges para pruebas."""
    manager = ExchangeManager(event_bus=event_bus)
    
    # Registrar exchanges
    for exchange in mock_exchanges:
        manager.register_exchange(exchange)
    
    return manager


@pytest.fixture
def risk_manager(event_bus):
    """Proporciona un gestor de riesgos para pruebas."""
    risk_manager = Mock(spec=RiskManager)
    risk_manager.validate_order = AsyncMock(return_value=True)
    risk_manager.calculate_position_size = AsyncMock(return_value=1.0)
    risk_manager.calculate_stop_loss = AsyncMock(return_value=40000.0)
    
    return risk_manager


@pytest.fixture
def execution_manager(exchange_manager, risk_manager, event_bus):
    """Proporciona un gestor de ejecución para pruebas."""
    return ExecutionManager(
        exchange_manager=exchange_manager,
        risk_manager=risk_manager,
        event_bus=event_bus
    )


@pytest.fixture
def complex_market_conditions(mock_exchanges):
    """Configura condiciones complejas de mercado para pruebas."""
    # Configuración específica para pruebas
    for exchange in mock_exchanges:
        # Aumentar volatilidad y spread en todos los exchanges
        exchange.volatility *= 2
        exchange.spread_bps *= 1.5
    
    # Simular un evento extremo en el primer exchange
    reliable_exchange = mock_exchanges[0]
    for symbol, price in reliable_exchange._base_prices.items():
        # Caída brusca del 15%
        reliable_exchange._base_prices[symbol] = price * 0.85
    reliable_exchange._initialize_order_books()
    
    return mock_exchanges


@pytest.mark.asyncio
async def test_execution_manager_high_frequency_trading(execution_manager):
    """Prueba la capacidad del gestor para manejar operaciones de alta frecuencia."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Configurar estrategia de ejecución simple
    strategy = SimpleExecutionStrategy()
    
    # Número alto de órdenes para simular HFT
    num_orders = 50
    
    # Parámetros de órdenes aleatorias
    sides = [OrderSide.BUY, OrderSide.SELL]
    
    # Crear órdenes
    orders = []
    for i in range(num_orders):
        order = Order(
            symbol=symbol,
            side=random.choice(sides),
            amount=random.uniform(0.01, 0.1),
            type=OrderType.MARKET
        )
        orders.append(order)
    
    # Medir tiempo de ejecución
    start_time = time.time()
    
    # Ejecutar órdenes concurrentemente
    tasks = []
    for order in orders:
        task = execution_manager.execute_order(order, strategy=strategy)
        tasks.append(task)
    
    # Esperar resultados, capturando excepciones
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Contar órdenes exitosas y fallidas
    successful_orders = [r for r in results if not isinstance(r, Exception)]
    failed_orders = [r for r in results if isinstance(r, Exception)]
    
    # En un entorno HFT, algunas órdenes pueden fallar, pero la mayoría deberían tener éxito
    success_rate = len(successful_orders) / num_orders
    
    # Imprimir estadísticas para debug
    print(f"HFT Test: {len(successful_orders)} successful, {len(failed_orders)} failed")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Total execution time: {execution_time:.3f}s")
    print(f"Average time per order: {execution_time / num_orders * 1000:.2f}ms")
    
    # Verificar métricas
    assert len(successful_orders) > 0, "No se ejecutó ninguna orden correctamente"
    assert success_rate > 0.7, f"Tasa de éxito demasiado baja: {success_rate:.2%}"
    
    # Verificar tiempo de ejecución (debería ser eficiente con alta concurrencia)
    assert execution_time < num_orders * 0.05, f"Ejecución demasiado lenta: {execution_time:.3f}s"
    
    # Verificar que las órdenes exitosas tienen estado correcto
    for result in successful_orders:
        assert result["status"] == "closed" or result["status"] == "open"


@pytest.mark.asyncio
async def test_advanced_execution_strategies(execution_manager):
    """Prueba estrategias avanzadas de ejecución para minimizar impacto de mercado."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Crear orden grande que se dividirá
    order = Order(
        symbol=symbol,
        side=OrderSide.BUY,
        amount=5.0,  # Orden grande de 5 BTC
        type=OrderType.LIMIT,
        price=45000.0
    )
    
    # Probar diferentes estrategias de ejecución
    strategies = [
        TwapExecutionStrategy(
            time_window_minutes=10,
            num_slices=5
        ),
        VwapExecutionStrategy(
            num_slices=5
        ),
        IcebergExecutionStrategy(
            visible_size=0.5,
            min_size=0.1
        ),
        SmartExecutionStrategy(
            aggression=0.5
        )
    ]
    
    execution_results = {}
    
    # Ejecutar la orden con cada estrategia
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        
        # Espiar el método execute para rastrear las sub-órdenes
        with patch.object(strategy, 'execute', wraps=strategy.execute) as mock_execute:
            # Ejecutar la orden
            result = await execution_manager.execute_order(order, strategy=strategy)
            
            # Guardar resultados y métricas
            execution_results[strategy_name] = {
                "result": result,
                "sub_orders": mock_execute.call_count,
            }
            
            # Verificar que la estrategia generó múltiples sub-órdenes
            assert mock_execute.call_count > 1, f"La estrategia {strategy_name} no dividió la orden"
    
    # Verificar TWAP
    assert execution_results["TwapExecutionStrategy"]["sub_orders"] == 5, "TWAP no creó el número correcto de sub-órdenes"
    
    # Verificar Iceberg
    assert execution_results["IcebergExecutionStrategy"]["sub_orders"] >= 5, "Iceberg no creó suficientes sub-órdenes"


@pytest.mark.asyncio
async def test_adaptive_execution_strategy(execution_manager, complex_market_conditions):
    """Prueba la estrategia de ejecución adaptativa que se ajusta a condiciones de mercado."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Crear una estrategia adaptativa que se ajuste a la volatilidad
    strategy = AdaptiveExecutionStrategy(
        base_aggression=0.5,
        volatility_sensitivity=0.8,
        min_time_between_orders_ms=100
    )
    
    # Espiar métodos internos para verificar adaptación
    with patch.object(strategy, '_adjust_parameters') as mock_adjust:
        # Crear una orden grande
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            amount=2.0,
            type=OrderType.LIMIT,
            price=45000.0
        )
        
        # Ejecutar la orden
        result = await execution_manager.execute_order(order, strategy=strategy)
        
        # Verificar que la estrategia se adaptó al mercado
        assert mock_adjust.called, "La estrategia no se adaptó a las condiciones del mercado"
        
        # Verificar que la orden se ejecutó (puede ser parcialmente)
        assert result is not None
        
        # En condiciones extremas, la orden puede no completarse totalmente
        if isinstance(result, list):
            # Verificar que al menos algunas sub-órdenes se ejecutaron
            executed = [r for r in result if r["status"] == "closed"]
            assert len(executed) > 0, "Ninguna sub-orden se ejecutó"
        else:
            assert result["status"] in ["closed", "open"], f"Estado inesperado: {result['status']}"


@pytest.mark.asyncio
async def test_order_cascading(execution_manager):
    """Prueba la creación de órdenes en cascada (entrada, take profit, stop loss)."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Crear orden principal
    main_order = Order(
        symbol=symbol,
        side=OrderSide.BUY,
        amount=1.0,
        type=OrderType.LIMIT,
        price=45000.0
    )
    
    # Configurar órdenes vinculadas
    stop_loss = Order(
        symbol=symbol,
        side=OrderSide.SELL,
        amount=1.0,
        type=OrderType.STOP_LOSS,
        price=43000.0
    )
    
    take_profit = Order(
        symbol=symbol,
        side=OrderSide.SELL,
        amount=1.0,
        type=OrderType.LIMIT,
        price=47000.0
    )
    
    # Registrar la orden principal con sus órdenes vinculadas
    result = await execution_manager.execute_order_cascade(
        main_order=main_order,
        stop_loss=stop_loss,
        take_profit=take_profit
    )
    
    # Verificar que se crearon las tres órdenes
    assert "main_order" in result, "No se creó la orden principal"
    assert "stop_loss" in result, "No se creó la orden de stop loss"
    assert "take_profit" in result, "No se creó la orden de take profit"
    
    # Verificar que las órdenes tienen los IDs vinculados correctamente
    assert "linked_orders" in result["main_order"], "La orden principal no tiene órdenes vinculadas"
    assert len(result["main_order"]["linked_orders"]) == 2, "La orden principal no tiene las 2 órdenes vinculadas"


@pytest.mark.asyncio
async def test_execution_under_market_stress(execution_manager, complex_market_conditions):
    """Prueba la ejecución de órdenes bajo condiciones extremas de mercado."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Crear varias órdenes con diferentes parámetros
    orders = [
        Order(symbol=symbol, side=OrderSide.BUY, amount=0.5, type=OrderType.MARKET),
        Order(symbol=symbol, side=OrderSide.SELL, amount=0.2, type=OrderType.LIMIT, price=43000.0),
        Order(symbol=symbol, side=OrderSide.BUY, amount=0.3, type=OrderType.LIMIT, price=42000.0),
        Order(symbol=symbol, side=OrderSide.SELL, amount=0.4, type=OrderType.STOP_LOSS, price=41000.0)
    ]
    
    # Intentar ejecutar todas las órdenes
    results = []
    for order in orders:
        try:
            result = await execution_manager.execute_order(order)
            results.append((order, result, None))
        except Exception as e:
            results.append((order, None, e))
    
    # Verificar resultados
    successful = [(o, r) for o, r, e in results if r is not None]
    failed = [(o, e) for o, r, e in results if e is not None]
    
    # En condiciones extremas, es aceptable que algunas órdenes fallen
    # pero deberíamos tener al menos algunas exitosas
    assert len(successful) > 0, "Todas las órdenes fallaron en condiciones extremas"
    
    # Verificar que las órdenes de mercado tienen más probabilidad de éxito
    market_orders_success = [
        (o, r) for o, r in successful 
        if o.type == OrderType.MARKET
    ]
    
    # Al menos algunas órdenes de mercado deberían tener éxito
    market_orders = [o for o in orders if o.type == OrderType.MARKET]
    if market_orders:
        assert len(market_orders_success) > 0, "Todas las órdenes de mercado fallaron"


@pytest.mark.asyncio
async def test_execution_cost_optimization(execution_manager):
    """Prueba la optimización de costos de ejecución."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Crear una orden grande que se ejecutará con diferentes estrategias
    order_amount = 3.0
    order = Order(
        symbol=symbol,
        side=OrderSide.BUY,
        amount=order_amount,
        type=OrderType.LIMIT,
        price=45000.0
    )
    
    # Comparar estrategias desde el punto de vista de costos
    # Estrategia simple (una sola orden)
    simple_strategy = SimpleExecutionStrategy()
    
    # Estrategia TWAP (divide la orden en partes iguales a lo largo del tiempo)
    twap_strategy = TwapExecutionStrategy(
        time_window_minutes=5,
        num_slices=3
    )
    
    # Estrategia que intenta optimizar costos
    cost_optimized_strategy = SmartExecutionStrategy(
        aggression=0.3,  # Baja agresividad para minimizar impacto
        optimize_for="cost"
    )
    
    # Ejecutar con cada estrategia y medir costos
    strategies = [
        ("Simple", simple_strategy),
        ("TWAP", twap_strategy),
        ("CostOptimized", cost_optimized_strategy)
    ]
    
    results = {}
    
    for name, strategy in strategies:
        # Ejecutar la orden
        result = await execution_manager.execute_order(order, strategy=strategy)
        
        # Calcular costo total (precio + comisiones)
        if isinstance(result, list):
            # Para estrategias que dividen en múltiples órdenes
            total_cost = sum(r["cost"] for r in result if r["status"] == "closed")
            total_fee = sum(r["fee"]["cost"] for r in result if r["status"] == "closed")
            total_filled = sum(r["filled"] for r in result if r["status"] == "closed")
        else:
            # Para estrategias que ejecutan en una sola orden
            total_cost = result["cost"] if result["status"] == "closed" else 0
            total_fee = result["fee"]["cost"] if result["status"] == "closed" else 0
            total_filled = result["filled"] if result["status"] == "closed" else 0
        
        # Solo considerar el costo por unidad para comparación justa
        if total_filled > 0:
            cost_per_unit = (total_cost + total_fee) / total_filled
        else:
            cost_per_unit = float('inf')
        
        results[name] = {
            "total_cost": total_cost,
            "total_fee": total_fee,
            "total_filled": total_filled,
            "cost_per_unit": cost_per_unit
        }
    
    # Verificar que todas las estrategias ejecutaron alguna cantidad
    for name, data in results.items():
        assert data["total_filled"] > 0, f"Estrategia {name} no ejecutó ninguna cantidad"
    
    # La estrategia optimizada para costos debería tener mejor precio por unidad
    if results["CostOptimized"]["total_filled"] > order_amount * 0.5:
        assert results["CostOptimized"]["cost_per_unit"] <= results["Simple"]["cost_per_unit"], \
            "La estrategia optimizada para costos no logró mejor precio que la simple"


@pytest.mark.asyncio
async def test_execution_resilience(execution_manager):
    """Prueba la resiliencia del gestor de ejecución ante fallos."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Crear una orden
    order = Order(
        symbol=symbol,
        side=OrderSide.BUY,
        amount=1.0,
        type=OrderType.MARKET
    )
    
    # Hacer que los exchanges fallen constantemente
    for exchange in execution_manager.exchange_manager._exchanges.values():
        if hasattr(exchange, "failure_rate"):
            exchange.failure_rate = 0.8  # 80% de fallos
    
    # Activar el modo resiliente del gestor
    execution_manager.max_retries = 5
    execution_manager.retry_delay_ms = 100
    
    # Ejecutar la orden - debería intentar varias veces a pesar de los fallos
    result = await execution_manager.execute_order(order)
    
    # Verificar que se obtuvo un resultado a pesar de los fallos
    assert result is not None, "No se obtuvo resultado a pesar de los reintentos"
    
    # Si todos los exchanges fallaron demasiadas veces, podría no ejecutarse
    # pero al menos debería haber intentado
    if result.get("status") != "closed":
        assert execution_manager.last_retry_count > 0, "No se intentaron reintentos"


@pytest.mark.asyncio
async def test_execution_consistency(execution_manager):
    """Prueba la consistencia en la ejecución, verificando idempotencia de operaciones."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Crear una orden con un ID cliente específico
    client_order_id = "test_idempotence_123"
    
    order = Order(
        symbol=symbol,
        side=OrderSide.BUY,
        amount=0.1,
        type=OrderType.LIMIT,
        price=45000.0,
        client_order_id=client_order_id
    )
    
    # Ejecutar la orden
    result1 = await execution_manager.execute_order(order)
    
    # Volver a ejecutar la misma orden (con el mismo ID cliente)
    # Esto debería ser idempotente y no crear una nueva orden
    result2 = await execution_manager.execute_order(order)
    
    # Verificar que los resultados son coherentes
    assert result1["id"] == result2["id"], "Se creó una nueva orden en lugar de referenciar la existente"
    
    # Modificar ligeramente la orden pero manteniendo el mismo ID
    modified_order = Order(
        symbol=symbol,
        side=OrderSide.BUY,
        amount=0.2,  # Cantidad diferente
        type=OrderType.LIMIT,
        price=45500.0,  # Precio diferente
        client_order_id=client_order_id
    )
    
    # Ejecutar la orden modificada - debería rechazarla o no crear una nueva
    with pytest.raises(Exception) as excinfo:
        await execution_manager.execute_order(modified_order, require_consistency=True)
    
    assert "consistency" in str(excinfo.value).lower() or "conflict" in str(excinfo.value).lower(), \
        "No se detectó inconsistencia al ejecutar una orden modificada con el mismo ID"


@pytest.mark.asyncio
async def test_position_management(execution_manager):
    """Prueba la gestión de posiciones integrada con la ejecución de órdenes."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Crear órdenes para abrir y modificar una posición
    open_order = Order(
        symbol=symbol,
        side=OrderSide.BUY,
        amount=1.0,
        type=OrderType.MARKET
    )
    
    # Ejecutar la orden de apertura
    open_result = await execution_manager.execute_order(open_order)
    
    # Verificar que se abrió la posición
    position = execution_manager.get_position(symbol)
    assert position is not None, "No se creó la posición"
    assert position.side == OrderSide.BUY, f"Lado incorrecto: {position.side}"
    assert position.amount == 1.0, f"Cantidad incorrecta: {position.amount}"
    
    # Añadir a la posición
    add_order = Order(
        symbol=symbol,
        side=OrderSide.BUY,
        amount=0.5,
        type=OrderType.MARKET
    )
    
    # Ejecutar la orden de adición
    add_result = await execution_manager.execute_order(add_order)
    
    # Verificar que se actualizó la posición
    position = execution_manager.get_position(symbol)
    assert position.amount == 1.5, f"Cantidad incorrecta después de añadir: {position.amount}"
    
    # Cerrar parcialmente la posición
    partial_close_order = Order(
        symbol=symbol,
        side=OrderSide.SELL,
        amount=0.5,
        type=OrderType.MARKET
    )
    
    # Ejecutar la orden de cierre parcial
    partial_close_result = await execution_manager.execute_order(partial_close_order)
    
    # Verificar que se actualizó la posición
    position = execution_manager.get_position(symbol)
    assert position.amount == 1.0, f"Cantidad incorrecta después de cierre parcial: {position.amount}"
    
    # Cerrar completamente la posición
    close_order = Order(
        symbol=symbol,
        side=OrderSide.SELL,
        amount=1.0,
        type=OrderType.MARKET
    )
    
    # Ejecutar la orden de cierre
    close_result = await execution_manager.execute_order(close_order)
    
    # Verificar que se cerró la posición
    position = execution_manager.get_position(symbol)
    assert position is None or position.amount == 0, "La posición no se cerró correctamente"


@pytest.mark.asyncio
async def test_execution_backpressure(execution_manager):
    """Prueba el manejo de contrapresión cuando hay demasiadas órdenes concurrentes."""
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Establecer límite de concurrencia bajo para la prueba
    execution_manager.max_concurrent_orders = 5
    
    # Simular alta latencia en los exchanges para que las órdenes tarden en procesarse
    for exchange in execution_manager.exchange_manager._exchanges.values():
        if hasattr(exchange, "latency_ms"):
            exchange.latency_ms = 300  # 300ms de latencia
    
    # Crear muchas órdenes concurrentes (más que el límite)
    num_orders = 20
    orders = []
    
    for i in range(num_orders):
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            amount=0.01,
            type=OrderType.MARKET
        )
        orders.append(order)
    
    # Medir tiempo de ejecución
    start_time = time.time()
    
    # Ejecutar todas las órdenes concurrentemente
    tasks = [execution_manager.execute_order(order) for order in orders]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Contar órdenes exitosas y fallidas
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    
    # Verificar que al menos algunas órdenes tuvieron éxito
    assert len(successful) > 0, "Todas las órdenes fallaron"
    
    # El tiempo total debería reflejar el procesamiento por lotes debido al límite de concurrencia
    # Fórmula aproximada: (num_orders / concurrencia) * tiempo_medio_orden
    expected_time = (num_orders / execution_manager.max_concurrent_orders) * 0.3
    
    # Verificar que el tiempo está en un rango razonable (con margen para variabilidad)
    assert total_time > expected_time * 0.5, f"Tiempo demasiado corto: {total_time}s vs esperado {expected_time}s"
    assert total_time < expected_time * 2.0, f"Tiempo demasiado largo: {total_time}s vs esperado {expected_time}s"


@pytest.mark.asyncio
async def test_execution_performance_benchmark(execution_manager):
    """Prueba el rendimiento del sistema de ejecución bajo carga sostenida."""
    # Esta prueba mide métricas de rendimiento del sistema de ejecución
    
    # Configurar un símbolo para pruebas
    symbol = "BTC/USDT"
    
    # Restablecer condiciones para esta prueba
    for exchange in execution_manager.exchange_manager._exchanges.values():
        if hasattr(exchange, "latency_ms"):
            exchange.latency_ms = 50  # Latencia normal
        if hasattr(exchange, "failure_rate"):
            exchange.failure_rate = 0.1  # Tasa de fallos normal
    
    # Números de órdenes a probar
    batch_sizes = [1, 5, 10, 20, 50]
    
    # Resultados
    benchmark_results = {}
    
    for batch_size in batch_sizes:
        # Crear las órdenes
        orders = []
        for i in range(batch_size):
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                amount=0.01,
                type=OrderType.MARKET
            )
            orders.append(order)
        
        # Medir tiempo de ejecución
        start_time = time.time()
        
        # Ejecutar todas las órdenes concurrentemente
        tasks = [execution_manager.execute_order(order) for order in orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calcular métricas
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful) / batch_size if batch_size > 0 else 0
        throughput = batch_size / total_time if total_time > 0 else 0
        latency_per_order = total_time / batch_size * 1000 if batch_size > 0 else 0
        
        # Guardar resultados
        benchmark_results[batch_size] = {
            "total_time": total_time,
            "success_rate": success_rate,
            "throughput": throughput,
            "latency_per_order": latency_per_order
        }
        
        # Imprimir resultados para debug
        print(f"Benchmark batch_size={batch_size}:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Throughput: {throughput:.2f} orders/s")
        print(f"  Latency per order: {latency_per_order:.2f}ms")
    
    # Verificar métricas de rendimiento
    # El éxito no debería degradarse significativamente con más órdenes
    for batch_size in batch_sizes[1:]:
        current_success = benchmark_results[batch_size]["success_rate"]
        baseline_success = benchmark_results[1]["success_rate"]
        
        # Permitir hasta un 20% de degradación en tasas de éxito
        assert current_success >= baseline_success * 0.8, \
            f"Degradación excesiva en tasa de éxito para batch_size={batch_size}: {current_success:.2%} vs {baseline_success:.2%}"
    
    # El throughput debería escalarse razonablemente
    # (no necesariamente lineal, pero debería crecer)
    baseline_throughput = benchmark_results[1]["throughput"]
    max_throughput = max(res["throughput"] for res in benchmark_results.values())
    
    assert max_throughput > baseline_throughput, \
        f"El throughput no mejoró con la concurrencia: {max_throughput:.2f} vs {baseline_throughput:.2f}"