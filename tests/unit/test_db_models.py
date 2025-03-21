"""
Pruebas unitarias para los modelos de base de datos avanzados.

Este módulo prueba la estructura, relaciones y validaciones de los modelos
de base de datos optimizados para alto rendimiento en el sistema Genesis.
"""

import pytest
import datetime
from unittest.mock import patch
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.schema import Table
from sqlalchemy.exc import IntegrityError
from genesis.db.enhanced_models import (
    Base, User, ApiKey, Exchange, Symbol, Candle, 
    Trade, Balance, Strategy, Signal, Role, UserRole
)


@pytest.fixture
def memory_engine():
    """Crear un motor de base de datos en memoria para las pruebas."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(memory_engine):
    """Crear una sesión de base de datos para las pruebas."""
    Session = sessionmaker(bind=memory_engine)
    session = Session()
    yield session
    session.close()


def test_user_role_relationship(db_session):
    """Verificar que las relaciones entre usuarios y roles funcionan correctamente."""
    # Crear roles
    admin_role = Role(name="admin", description="Administrador del sistema", 
                      permissions={"system": ["manage", "view"], "trading": ["execute", "view"]})
    trader_role = Role(name="trader", description="Trader", 
                       permissions={"trading": ["execute", "view"]})
    db_session.add_all([admin_role, trader_role])
    db_session.commit()
    
    # Crear usuario con roles
    user = User(
        username="test_user",
        email="test@example.com",
        password_hash="hashed_password",
        first_name="Test",
        last_name="User"
    )
    user.roles = [admin_role, trader_role]
    db_session.add(user)
    db_session.commit()
    
    # Verificar la asignación de roles
    db_session.refresh(user)
    assert len(user.roles) == 2
    role_names = [role.name for role in user.roles]
    assert "admin" in role_names
    assert "trader" in role_names
    
    # Verificar la relación inversa (usuarios por rol)
    assert len(admin_role.users) == 1
    assert admin_role.users[0].username == "test_user"


def test_exchange_symbol_relationship(db_session):
    """Verificar que las relaciones entre exchanges y símbolos funcionan correctamente."""
    # Crear un exchange
    exchange = Exchange(
        name="binance",
        description="Binance Exchange",
        api_base_url="https://api.binance.com",
        websocket_url="wss://stream.binance.com:9443/ws"
    )
    db_session.add(exchange)
    db_session.commit()
    
    # Crear símbolos para el exchange
    btc_symbol = Symbol(
        exchange_id=exchange.id,
        name="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        price_precision=2,
        quantity_precision=6
    )
    eth_symbol = Symbol(
        exchange_id=exchange.id,
        name="ETHUSDT",
        base_asset="ETH",
        quote_asset="USDT",
        price_precision=2,
        quantity_precision=5
    )
    db_session.add_all([btc_symbol, eth_symbol])
    db_session.commit()
    
    # Verificar la relación exchange -> symbols
    db_session.refresh(exchange)
    assert len(exchange.symbols) == 2
    symbol_names = [symbol.name for symbol in exchange.symbols]
    assert "BTCUSDT" in symbol_names
    assert "ETHUSDT" in symbol_names
    
    # Verificar la relación symbol -> exchange
    db_session.refresh(btc_symbol)
    assert btc_symbol.exchange.name == "binance"


def test_user_api_key_relationship(db_session):
    """Verificar que las relaciones entre usuarios y API keys funcionan correctamente."""
    # Crear usuario
    user = User(
        username="api_user",
        email="api@example.com",
        password_hash="hashed_password"
    )
    db_session.add(user)
    
    # Crear exchange
    exchange = Exchange(name="kraken", description="Kraken Exchange")
    db_session.add(exchange)
    db_session.commit()
    
    # Crear API key para el usuario
    api_key = ApiKey(
        user_id=user.id,
        exchange_id=exchange.id,
        description="Trading API key",
        api_key="public_key_example",
        api_secret="encrypted_secret_example",
        permissions={"trade": True, "withdraw": False}
    )
    db_session.add(api_key)
    db_session.commit()
    
    # Verificar la relación user -> api_keys
    db_session.refresh(user)
    assert len(user.api_keys) == 1
    assert user.api_keys[0].api_key == "public_key_example"
    
    # Verificar la relación api_key -> user
    db_session.refresh(api_key)
    assert api_key.user.username == "api_user"
    
    # Verificar la relación api_key -> exchange
    assert api_key.exchange.name == "kraken"


def test_candle_data_integrity(db_session):
    """Verificar la inserción y recuperación de datos de velas (OHLCV)."""
    # Crear exchange y símbolo
    exchange = Exchange(name="ftx", description="FTX Exchange")
    db_session.add(exchange)
    db_session.commit()
    
    symbol = Symbol(
        exchange_id=exchange.id,
        name="BTCUSD",
        base_asset="BTC",
        quote_asset="USD"
    )
    db_session.add(symbol)
    db_session.commit()
    
    # Crear datos de velas para diferentes timeframes
    now = datetime.datetime.utcnow()
    candles = [
        Candle(
            exchange_id=exchange.id,
            symbol_id=symbol.id,
            exchange="ftx",
            symbol="BTCUSD",
            timeframe=timeframe,
            timestamp=now - datetime.timedelta(minutes=i*int(timeframe[:-1])),
            open=40000 + i,
            high=40100 + i,
            low=39900 + i,
            close=40050 + i,
            volume=10 + i
        )
        for i, timeframe in enumerate(["1m", "5m", "15m", "1h"])
    ]
    db_session.add_all(candles)
    db_session.commit()
    
    # Verificar la consulta de velas por timeframe
    one_min_candles = db_session.query(Candle).filter(Candle.timeframe == "1m").all()
    assert len(one_min_candles) == 1
    assert one_min_candles[0].open == 40000
    
    # Verificar la consulta de velas por símbolo
    btc_candles = db_session.query(Candle).filter(Candle.symbol == "BTCUSD").all()
    assert len(btc_candles) == 4
    
    # Verificar el ordenamiento por timestamp
    ordered_candles = db_session.query(Candle).order_by(Candle.timestamp.desc()).all()
    assert ordered_candles[0].timeframe == "1m"  # La más reciente


def test_trades_and_signals(db_session):
    """Verificar el registro de operaciones (trades) y señales."""
    # Configuración básica
    user = User(username="trader1", email="trader1@example.com", password_hash="hash")
    exchange = Exchange(name="bitmex", description="BitMEX Exchange")
    db_session.add_all([user, exchange])
    db_session.commit()
    
    symbol = Symbol(
        exchange_id=exchange.id,
        name="BTCUSD",
        base_asset="BTC",
        quote_asset="USD"
    )
    strategy = Strategy(
        name="macd_cross",
        description="MACD Crossover strategy",
        type="trend_following",
        parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9}
    )
    db_session.add_all([symbol, strategy])
    db_session.commit()
    
    # Crear una señal
    signal = Signal(
        strategy_id=strategy.id,
        symbol_id=symbol.id,
        timestamp=datetime.datetime.utcnow(),
        signal_type="buy",
        timeframe="1h",
        price=39500.0,
        quantity=1.0,
        confidence=0.85,
        indicators={"macd": 0.25, "rsi": 32}
    )
    db_session.add(signal)
    db_session.commit()
    
    # Crear una operación basada en la señal
    trade = Trade(
        user_id=user.id,
        trade_id="T123456",
        exchange_id=exchange.id,
        symbol_id=symbol.id,
        exchange="bitmex",
        symbol="BTCUSD",
        side="buy",
        type="market",
        amount=1.0,
        price=39505.0,  # Ligero slippage
        fee=0.001 * 39505.0,  # 0.1% fee
        fee_currency="USD",
        total=39505.0,
        status="closed",
        strategy_id=strategy.id,
        execution_time=120.5,  # ms
        entry_time=datetime.datetime.utcnow(),
        exit_time=datetime.datetime.utcnow() + datetime.timedelta(hours=2),
        profit_loss=495.0,
        profit_loss_pct=1.25,
        trade_metadata={"signal_id": signal.id, "note": "Test trade"}
    )
    db_session.add(trade)
    db_session.commit()
    
    # Verificar la relación trade -> strategy
    db_session.refresh(trade)
    assert trade.strategy.name == "macd_cross"
    
    # Verificar la relación trade -> user
    assert trade.user.username == "trader1"


def test_strategy_performance(db_session):
    """Verificar el registro y consulta de rendimiento de estrategias."""
    # Crear una estrategia
    strategy = Strategy(
        name="bollinger_breakout",
        description="Bollinger Bands Breakout strategy",
        type="breakout",
        parameters={"period": 20, "stddev": 2}
    )
    db_session.add(strategy)
    db_session.commit()
    
    # Añadir métricas de rendimiento para diferentes símbolos
    from genesis.db.enhanced_models import StrategyPerformance
    
    performances = [
        StrategyPerformance(
            strategy_id=strategy.id,
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime.datetime(2024, 1, 1),
            end_date=datetime.datetime(2024, 3, 1),
            total_trades=120,
            winning_trades=75,
            losing_trades=45,
            win_rate=0.625,
            profit_loss=15250.0,
            profit_loss_pct=15.25,
            max_drawdown=3800.0,
            max_drawdown_pct=3.8,
            sharpe_ratio=1.85,
            sortino_ratio=2.3,
            calmar_ratio=4.01,
            volatility=0.028,
            metrics={"avg_trade_duration": 18.5, "profit_factor": 2.1}
        ),
        StrategyPerformance(
            strategy_id=strategy.id,
            symbol="ETHUSDT",
            timeframe="1h",
            start_date=datetime.datetime(2024, 1, 1),
            end_date=datetime.datetime(2024, 3, 1),
            total_trades=105,
            winning_trades=60,
            losing_trades=45,
            win_rate=0.571,
            profit_loss=9800.0,
            profit_loss_pct=9.8,
            max_drawdown=2900.0,
            max_drawdown_pct=2.9,
            sharpe_ratio=1.55,
            sortino_ratio=1.9,
            calmar_ratio=3.38,
            volatility=0.032,
            metrics={"avg_trade_duration": 16.2, "profit_factor": 1.75}
        )
    ]
    db_session.add_all(performances)
    db_session.commit()
    
    # Verificar la relación strategy -> performances
    db_session.refresh(strategy)
    assert len(strategy.performances) == 2
    
    # Verificar los datos de rendimiento
    btc_perf = next(p for p in strategy.performances if p.symbol == "BTCUSDT")
    assert btc_perf.win_rate == 0.625
    assert btc_perf.sharpe_ratio == 1.85
    
    # Verificar consultas más complejas
    best_perf = (db_session.query(StrategyPerformance)
                 .filter(StrategyPerformance.strategy_id == strategy.id)
                 .order_by(StrategyPerformance.sharpe_ratio.desc())
                 .first())
    assert best_perf.symbol == "BTCUSDT"