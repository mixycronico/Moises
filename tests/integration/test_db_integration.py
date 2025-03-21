"""
Pruebas de integración para la base de datos del sistema Genesis.

Este módulo prueba la integración entre el sistema y la base de datos,
incluyendo consultas complejas, actualizaciones concurrentes y manipulación
de datos a través de los repositorios.
"""

import pytest
import os
import asyncio
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from genesis.db.enhanced_models import (
    Base, User, ApiKey, Exchange, Symbol, Candle, 
    Trade, Balance, Strategy, Signal, Role, TradeEvent
)


@pytest.fixture
def memory_db_url():
    """URL de base de datos en memoria para pruebas."""
    return 'sqlite:///:memory:'


@pytest.fixture
def db_engine(memory_db_url):
    """Crear un motor de base de datos para pruebas."""
    engine = create_engine(memory_db_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Crear una sesión de base de datos para las pruebas."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def setup_test_data(db_session):
    """Configurar datos de prueba básicos."""
    # Crear usuarios
    admin = User(
        username="admin", 
        email="admin@example.com", 
        password_hash="hash", 
        is_admin=True
    )
    trader = User(
        username="trader", 
        email="trader@example.com", 
        password_hash="hash"
    )
    
    # Crear exchanges
    binance = Exchange(name="binance", description="Binance Exchange")
    ftx = Exchange(name="ftx", description="FTX Exchange")
    
    # Añadir a la sesión
    db_session.add_all([admin, trader, binance, ftx])
    db_session.commit()
    
    # Crear símbolos
    btc_binance = Symbol(
        exchange_id=binance.id,
        name="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        price_precision=2,
        quantity_precision=6
    )
    eth_binance = Symbol(
        exchange_id=binance.id,
        name="ETHUSDT",
        base_asset="ETH",
        quote_asset="USDT",
        price_precision=2,
        quantity_precision=5
    )
    
    # Crear estrategias
    macd_strat = Strategy(
        name="macd_cross",
        description="MACD Crossover strategy",
        type="trend_following",
        parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9}
    )
    
    # Añadir a la sesión
    db_session.add_all([btc_binance, eth_binance, macd_strat])
    db_session.commit()
    
    # Devolver datos creados para uso en pruebas
    return {
        "users": {"admin": admin, "trader": trader},
        "exchanges": {"binance": binance, "ftx": ftx},
        "symbols": {"btc_binance": btc_binance, "eth_binance": eth_binance},
        "strategies": {"macd": macd_strat}
    }


@pytest.mark.integration
@pytest.mark.db
def test_complex_query_performance(db_session, setup_test_data):
    """Probar consultas complejas y su rendimiento."""
    test_data = setup_test_data
    
    # Crear velas para un período
    now = datetime.datetime.utcnow()
    candles = []
    
    # 100 velas de 1 minuto para BTC
    for i in range(100):
        candle_time = now - datetime.timedelta(minutes=i)
        candles.append(
            Candle(
                exchange_id=test_data["exchanges"]["binance"].id,
                symbol_id=test_data["symbols"]["btc_binance"].id,
                exchange="binance",
                symbol="BTCUSDT",
                timeframe="1m",
                timestamp=candle_time,
                open=40000 + i % 10,
                high=40100 + i % 20,
                low=39900 - i % 15,
                close=40050 + i % 25,
                volume=10 + i % 5
            )
        )
    
    # 100 velas de 1 minuto para ETH
    for i in range(100):
        candle_time = now - datetime.timedelta(minutes=i)
        candles.append(
            Candle(
                exchange_id=test_data["exchanges"]["binance"].id,
                symbol_id=test_data["symbols"]["eth_binance"].id,
                exchange="binance",
                symbol="ETHUSDT",
                timeframe="1m",
                timestamp=candle_time,
                open=3000 + i % 10,
                high=3050 + i % 15,
                low=2950 - i % 10,
                close=3025 + i % 20,
                volume=50 + i % 10
            )
        )
    
    db_session.add_all(candles)
    db_session.commit()
    
    # Consulta 1: Obtener últimas N velas para un símbolo específico
    import time
    start_time = time.time()
    
    last_10_btc = (db_session.query(Candle)
                   .filter(Candle.symbol == "BTCUSDT")
                   .filter(Candle.timeframe == "1m")
                   .order_by(Candle.timestamp.desc())
                   .limit(10)
                   .all())
    
    query_time = time.time() - start_time
    print(f"Query time for last 10 BTC candles: {query_time:.6f} seconds")
    
    assert len(last_10_btc) == 10
    assert last_10_btc[0].timestamp > last_10_btc[1].timestamp  # Ordenamiento correcto
    
    # Consulta 2: Obtener velas en un rango de tiempo específico
    one_hour_ago = now - datetime.timedelta(hours=1)
    
    btc_range = (db_session.query(Candle)
                 .filter(Candle.symbol == "BTCUSDT")
                 .filter(Candle.timeframe == "1m")
                 .filter(Candle.timestamp >= one_hour_ago)
                 .filter(Candle.timestamp <= now)
                 .order_by(Candle.timestamp)
                 .all())
    
    assert len(btc_range) <= 60  # Máximo 60 velas de 1 minuto en 1 hora
    
    # Consulta 3: Obtener máximo, mínimo y promedio para un período
    from sqlalchemy import func
    
    agg_data = (db_session.query(
                   func.max(Candle.high).label('max_price'),
                   func.min(Candle.low).label('min_price'),
                   func.avg(Candle.close).label('avg_close'),
                   func.sum(Candle.volume).label('total_volume')
               )
               .filter(Candle.symbol == "BTCUSDT")
               .filter(Candle.timeframe == "1m")
               .filter(Candle.timestamp >= one_hour_ago)
               .one())
    
    assert agg_data.max_price is not None
    assert agg_data.min_price is not None
    assert agg_data.avg_close is not None
    assert agg_data.total_volume is not None


@pytest.mark.integration
@pytest.mark.db
def test_trade_events_sourcing(db_session, setup_test_data):
    """Probar el patrón de event sourcing para operaciones de trading."""
    test_data = setup_test_data
    user = test_data["users"]["trader"]
    exchange = test_data["exchanges"]["binance"]
    symbol = test_data["symbols"]["btc_binance"]
    strategy = test_data["strategies"]["macd"]
    
    # Eventos de trading para una operación
    trade_id = "T12345"
    correlation_id = "550e8400-e29b-41d4-a716-446655440000"
    
    # Evento 1: Creación de orden
    create_event = TradeEvent(
        trade_id=trade_id,
        correlation_id=correlation_id,
        user_id=user.id,
        event_type="order_created",
        timestamp=datetime.datetime.utcnow() - datetime.timedelta(minutes=30),
        sequence=1,
        data={
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "type": "limit",
            "side": "buy",
            "amount": 0.1,
            "price": 40000,
            "strategy_id": strategy.id
        }
    )
    
    # Evento 2: Orden parcialmente ejecutada
    partial_event = TradeEvent(
        trade_id=trade_id,
        correlation_id=correlation_id,
        user_id=user.id,
        event_type="order_partially_filled",
        timestamp=datetime.datetime.utcnow() - datetime.timedelta(minutes=29),
        sequence=2,
        data={
            "filled_amount": 0.05,
            "price": 40001,
            "fee": 0.001 * 40001 * 0.05,
            "fee_currency": "USDT"
        }
    )
    
    # Evento 3: Orden completamente ejecutada
    complete_event = TradeEvent(
        trade_id=trade_id,
        correlation_id=correlation_id,
        user_id=user.id,
        event_type="order_filled",
        timestamp=datetime.datetime.utcnow() - datetime.timedelta(minutes=28),
        sequence=3,
        data={
            "filled_amount": 0.05,
            "price": 40002,
            "fee": 0.001 * 40002 * 0.05,
            "fee_currency": "USDT",
            "total_filled": 0.1,
            "avg_price": 40001.5
        }
    )
    
    # Guardar eventos
    db_session.add_all([create_event, partial_event, complete_event])
    db_session.commit()
    
    # Probar reconstrucción del estado actual
    trade_events = (db_session.query(TradeEvent)
                    .filter(TradeEvent.trade_id == trade_id)
                    .order_by(TradeEvent.sequence)
                    .all())
    
    assert len(trade_events) == 3
    
    # Reconstruir estado
    trade_state = {
        "trade_id": trade_id,
        "status": "pending",
        "filled_amount": 0,
        "avg_price": 0
    }
    
    for event in trade_events:
        if event.event_type == "order_created":
            trade_state.update({
                "exchange": event.data["exchange"],
                "symbol": event.data["symbol"],
                "side": event.data["side"],
                "type": event.data["type"],
                "amount": event.data["amount"],
                "price": event.data["price"],
                "entry_time": event.timestamp
            })
        elif event.event_type == "order_partially_filled":
            trade_state["filled_amount"] += event.data["filled_amount"]
            # Calcular precio promedio ponderado
            if trade_state["avg_price"] == 0:
                trade_state["avg_price"] = event.data["price"]
            else:
                trade_state["avg_price"] = (
                    (trade_state["avg_price"] * (trade_state["filled_amount"] - event.data["filled_amount"]) +
                     event.data["price"] * event.data["filled_amount"]) / trade_state["filled_amount"]
                )
        elif event.event_type == "order_filled":
            trade_state["filled_amount"] += event.data["filled_amount"]
            trade_state["avg_price"] = event.data["avg_price"]
            trade_state["status"] = "filled"
            trade_state["exit_time"] = event.timestamp
    
    # Verificar estado reconstruido
    assert trade_state["status"] == "filled"
    assert trade_state["filled_amount"] == 0.1
    assert trade_state["avg_price"] == 40001.5
    assert trade_state["symbol"] == "BTCUSDT"
    assert trade_state["side"] == "buy"


@pytest.mark.integration
@pytest.mark.db
def test_balance_tracking(db_session, setup_test_data):
    """Probar el seguimiento de saldos a lo largo del tiempo."""
    test_data = setup_test_data
    user = test_data["users"]["trader"]
    exchange = test_data["exchanges"]["binance"]
    
    # Crear saldos para diferentes momentos
    now = datetime.datetime.utcnow()
    
    # Día 1
    day1 = now - datetime.timedelta(days=2)
    balance1 = Balance(
        user_id=user.id,
        exchange_id=exchange.id,
        timestamp=day1,
        asset="BTC",
        free=1.0,
        used=0.0,
        total=1.0,
        equivalent_usd=40000.0
    )
    balance2 = Balance(
        user_id=user.id,
        exchange_id=exchange.id,
        timestamp=day1,
        asset="USDT",
        free=10000.0,
        used=0.0,
        total=10000.0,
        equivalent_usd=10000.0
    )
    
    # Día 2
    day2 = now - datetime.timedelta(days=1)
    balance3 = Balance(
        user_id=user.id,
        exchange_id=exchange.id,
        timestamp=day2,
        asset="BTC",
        free=0.9,
        used=0.0,
        total=0.9,
        equivalent_usd=37800.0
    )
    balance4 = Balance(
        user_id=user.id,
        exchange_id=exchange.id,
        timestamp=day2,
        asset="USDT",
        free=14000.0,
        used=0.0,
        total=14000.0,
        equivalent_usd=14000.0
    )
    
    # Día 3 (hoy)
    balance5 = Balance(
        user_id=user.id,
        exchange_id=exchange.id,
        timestamp=now,
        asset="BTC",
        free=1.1,
        used=0.0,
        total=1.1,
        equivalent_usd=44000.0
    )
    balance6 = Balance(
        user_id=user.id,
        exchange_id=exchange.id,
        timestamp=now,
        asset="USDT",
        free=8000.0,
        used=0.0,
        total=8000.0,
        equivalent_usd=8000.0
    )
    
    db_session.add_all([balance1, balance2, balance3, balance4, balance5, balance6])
    db_session.commit()
    
    # Consulta 1: Obtener saldo actual por activo
    current_balances = (db_session.query(Balance)
                        .filter(Balance.user_id == user.id)
                        .filter(Balance.exchange_id == exchange.id)
                        .filter(Balance.timestamp == now)
                        .all())
    
    assert len(current_balances) == 2
    
    # Calcular valor total en USD
    total_usd = sum(balance.equivalent_usd for balance in current_balances)
    assert total_usd == 52000.0  # 44000 + 8000
    
    # Consulta 2: Analizar cambio de valor en el tiempo
    from sqlalchemy import func
    
    daily_totals = (
        db_session.query(
            func.date(Balance.timestamp).label('date'),
            func.sum(Balance.equivalent_usd).label('total_usd')
        )
        .filter(Balance.user_id == user.id)
        .filter(Balance.exchange_id == exchange.id)
        .group_by(func.date(Balance.timestamp))
        .order_by(func.date(Balance.timestamp))
        .all()
    )
    
    assert len(daily_totals) == 3
    
    # Verificar cambio día a día
    assert daily_totals[0].total_usd == 50000.0  # Día 1: 40000 + 10000
    assert daily_totals[1].total_usd == 51800.0  # Día 2: 37800 + 14000
    assert daily_totals[2].total_usd == 52000.0  # Día 3: 44000 + 8000
    
    # Calcular rendimiento
    performance = (daily_totals[-1].total_usd - daily_totals[0].total_usd) / daily_totals[0].total_usd * 100
    assert performance == 4.0  # 4% de crecimiento