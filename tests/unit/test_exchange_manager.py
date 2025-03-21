"""
Pruebas unitarias para el gestor de exchanges.

Este módulo prueba las funcionalidades del gestor de exchanges,
que se encarga de seleccionar el mejor exchange y ejecutar operaciones.
"""

import pytest
from unittest.mock import Mock, patch

from genesis.exchanges.manager import ExchangeManager
from genesis.exchanges.exchange_selector import ExchangeSelector


# Fixture para exchanges simulados
@pytest.fixture
def mock_exchanges():
    """Fixture que proporciona exchanges simulados con condiciones variables."""
    exchanges = {f"exchange_{i}": Mock() for i in range(1, 15)}
    for i, exchange in enumerate(exchanges.values(), start=1):
        exchange.get_conditions.return_value = {
            "latency": i * 10,      # Latencia aumenta con el índice (10, 20, ..., 140)
            "fees": i * 0.01,       # Fees aumentan con el índice (0.01, 0.02, ..., 0.14)
            "volume": 1000000 / i   # Volumen disminuye con el índice (1000000, 500000, ..., ~71428)
        }
        exchange.execute_trade.return_value = {"status": "success", "order_id": f"ORDER{i}"}
        exchange.fetch_balance.return_value = {
            "BTC": {"free": 1.0, "used": 0.5, "total": 1.5},
            "USDT": {"free": 10000, "used": 5000, "total": 15000}
        }
    return exchanges


@pytest.fixture
def exchange_selector():
    """Fixture que proporciona un selector de exchanges."""
    return ExchangeSelector()


@pytest.fixture
def exchange_manager(mock_exchanges, exchange_selector):
    """Fixture que proporciona una instancia de ExchangeManager con exchanges simulados."""
    manager = ExchangeManager()
    manager.exchanges = mock_exchanges
    manager.selector = exchange_selector
    return manager


# Pruebas principales
def test_exchange_manager_initialization():
    """Prueba la inicialización básica del gestor de exchanges."""
    # Sin exchanges
    manager = ExchangeManager()
    assert manager.exchanges == {}
    assert isinstance(manager.selector, ExchangeSelector)
    
    # Con lista de exchanges
    manager = ExchangeManager(exchanges=["binance", "kraken", "coinbase"])
    assert len(manager.exchanges) == 3
    assert "binance" in manager.exchanges
    assert "kraken" in manager.exchanges
    assert "coinbase" in manager.exchanges


def test_exchange_manager_best_exchange_selection(exchange_manager, mock_exchanges, exchange_selector):
    """Prueba que el sistema seleccione correctamente el mejor exchange."""
    # Mockear el selector para que devuelva un exchange específico
    exchange_selector.get_best_exchange = Mock(return_value="exchange_1")
    
    best_exchange = exchange_manager.get_best_exchange("BTC/USDT")
    
    # Verificar que el selector fue llamado correctamente y se devolvió el resultado esperado
    assert best_exchange == "exchange_1"
    exchange_selector.get_best_exchange.assert_called_once_with("BTC/USDT")


def test_exchange_manager_execute_trade_best_exchange(exchange_manager, mock_exchanges, exchange_selector):
    """Prueba que execute_trade use el mejor exchange y llame a execute_trade solo en él."""
    # Mockear el selector para que devuelva un exchange específico
    exchange_selector.get_best_exchange = Mock(return_value="exchange_3")
    
    result = exchange_manager.execute_trade("BTC/USDT", "buy", 0.01, price=40000)
    
    # Verificar el resultado y que se llamó al método correcto
    assert result["status"] == "success"
    assert result["order_id"] == "ORDER3"
    mock_exchanges["exchange_3"].execute_trade.assert_called_once_with("BTC/USDT", "buy", 0.01, price=40000)
    
    # Verificar que otros exchanges no fueron llamados
    for key, exchange in mock_exchanges.items():
        if key != "exchange_3":
            exchange.execute_trade.assert_not_called()


def test_exchange_manager_no_exchanges(exchange_selector):
    """Prueba el comportamiento cuando no hay exchanges disponibles."""
    empty_manager = ExchangeManager()
    empty_manager.selector = exchange_selector
    
    # Mockear el selector para que lance una excepción
    exchange_selector.get_best_exchange = Mock(side_effect=ValueError("No exchanges available"))
    
    with pytest.raises(ValueError, match="No exchanges available"):
        empty_manager.get_best_exchange("BTC/USDT")


def test_exchange_manager_unavailable_exchange(exchange_manager, exchange_selector):
    """Prueba el manejo cuando el mejor exchange seleccionado no está en el diccionario."""
    # Mockear el selector para que devuelva un exchange que no existe
    exchange_selector.get_best_exchange = Mock(return_value="exchange_99")
    
    with pytest.raises(ValueError, match="Exchange not available"):
        exchange_manager.execute_trade("BTC/USDT", "buy", 0.01)


def test_exchange_manager_trade_failure(exchange_manager, mock_exchanges, exchange_selector):
    """Prueba el manejo de un fallo en execute_trade del mejor exchange."""
    # Mockear el selector para que devuelva un exchange específico
    exchange_selector.get_best_exchange = Mock(return_value="exchange_5")
    
    # Configurar el mock para que devuelva un error
    mock_exchanges["exchange_5"].execute_trade.return_value = {
        "status": "error", 
        "message": "Insufficient funds"
    }
    
    result = exchange_manager.execute_trade("BTC/USDT", "buy", 0.01)
    
    assert result["status"] == "error"
    assert result["message"] == "Insufficient funds"
    mock_exchanges["exchange_5"].execute_trade.assert_called_once()


def test_exchange_manager_invalid_trade_parameters(exchange_manager):
    """Prueba el manejo de parámetros inválidos en execute_trade."""
    with pytest.raises(ValueError, match="Amount must be positive"):
        exchange_manager.execute_trade("BTC/USDT", "buy", -0.01)
    
    with pytest.raises(ValueError, match="Invalid side"):
        exchange_manager.execute_trade("BTC/USDT", "invalid_side", 0.01)
    
    with pytest.raises(ValueError, match="Symbol cannot be empty"):
        exchange_manager.execute_trade("", "buy", 0.01)


def test_exchange_manager_conditions_variation(exchange_manager, mock_exchanges, exchange_selector):
    """Prueba el selector con diferentes condiciones de exchanges."""
    # Verificar varias condiciones de exchanges a través del selector
    test_cases = [
        # Caso 1: Exchange 1 tiene mejores condiciones en todos los aspectos
        {
            "conditions": {
                "exchange_1": {"latency": 10, "fees": 0.001, "volume": 1000000},
                "exchange_2": {"latency": 50, "fees": 0.005, "volume": 500000}
            },
            "expected": "exchange_1"
        },
        # Caso 2: Exchange 2 tiene mejor latencia, Exchange 1 mejor en el resto
        {
            "conditions": {
                "exchange_1": {"latency": 30, "fees": 0.001, "volume": 1000000},
                "exchange_2": {"latency": 10, "fees": 0.005, "volume": 500000}
            },
            "expected": "exchange_1"  # Asumiendo que el algoritmo prioriza fees y volumen
        },
        # Caso 3: Exchange 3 tiene volumen extremadamente alto
        {
            "conditions": {
                "exchange_1": {"latency": 10, "fees": 0.001, "volume": 1000000},
                "exchange_3": {"latency": 100, "fees": 0.01, "volume": 10000000}
            },
            "expected": "exchange_1"  # Asumiendo que latencia y fees tienen más peso
        }
    ]
    
    for i, case in enumerate(test_cases):
        # Configurar mocks para cada caso
        for ex_name, conditions in case["conditions"].items():
            mock_exchanges[ex_name].get_conditions.return_value = conditions
        
        # Configurar selector para que llame al método real que analiza las condiciones
        def side_effect_get_best(symbol):
            return exchange_selector._evaluate_exchanges(
                symbol, list(case["conditions"].keys()), mock_exchanges
            )
        
        exchange_selector.get_best_exchange = Mock(side_effect=side_effect_get_best)
        
        # Ejecutar la prueba
        result = exchange_manager.get_best_exchange("BTC/USDT")
        
        # Verificar resultado (modificar según la lógica real de selección)
        assert result == case["expected"], f"Caso {i+1} falló"


def test_exchange_manager_get_balance(exchange_manager, mock_exchanges):
    """Prueba la obtención de saldo de un exchange específico."""
    # Obtener saldo de un exchange específico
    balance = exchange_manager.get_balance("exchange_2")
    
    # Verificar que se llamó al método correcto y se obtuvo el resultado esperado
    mock_exchanges["exchange_2"].fetch_balance.assert_called_once()
    assert balance["BTC"]["total"] == 1.5
    assert balance["USDT"]["free"] == 10000


def test_exchange_manager_get_balance_all_exchanges(exchange_manager, mock_exchanges):
    """Prueba la obtención de saldo de todos los exchanges."""
    # Obtener saldo de todos los exchanges
    balances = exchange_manager.get_all_balances()
    
    # Verificar que se llamó al método para cada exchange
    for exchange in mock_exchanges.values():
        exchange.fetch_balance.assert_called_once()
    
    # Verificar resultado
    assert len(balances) == len(mock_exchanges)
    for exchange_name in mock_exchanges.keys():
        assert exchange_name in balances
        assert balances[exchange_name]["BTC"]["total"] == 1.5
        assert balances[exchange_name]["USDT"]["free"] == 10000


def test_exchange_manager_get_ticker(exchange_manager, mock_exchanges):
    """Prueba la obtención de ticker para un símbolo en un exchange específico."""
    # Configurar mock para obtener ticker
    mock_ticker = {
        "symbol": "BTC/USDT",
        "bid": 40000,
        "ask": 40050,
        "last": 40025,
        "volume": 1000,
        "timestamp": 1625097600000
    }
    mock_exchanges["exchange_4"].fetch_ticker.return_value = mock_ticker
    
    # Obtener ticker
    ticker = exchange_manager.get_ticker("BTC/USDT", "exchange_4")
    
    # Verificar que se llamó al método correcto y se obtuvo el resultado esperado
    mock_exchanges["exchange_4"].fetch_ticker.assert_called_once_with("BTC/USDT")
    assert ticker == mock_ticker


def test_exchange_manager_get_order_book(exchange_manager, mock_exchanges):
    """Prueba la obtención del libro de órdenes para un símbolo en un exchange específico."""
    # Configurar mock para obtener libro de órdenes
    mock_order_book = {
        "bids": [[40000, 1.5], [39900, 2.0], [39800, 2.5]],
        "asks": [[40100, 1.0], [40200, 1.5], [40300, 2.0]],
        "timestamp": 1625097600000
    }
    mock_exchanges["exchange_3"].fetch_order_book.return_value = mock_order_book
    
    # Obtener libro de órdenes
    order_book = exchange_manager.get_order_book("BTC/USDT", "exchange_3", limit=5)
    
    # Verificar que se llamó al método correcto y se obtuvo el resultado esperado
    mock_exchanges["exchange_3"].fetch_order_book.assert_called_once_with("BTC/USDT", limit=5)
    assert order_book == mock_order_book


def test_exchange_manager_calculate_spread(exchange_manager, mock_exchanges):
    """Prueba el cálculo del spread para un símbolo en un exchange específico."""
    # Configurar mock para obtener ticker
    mock_ticker = {
        "symbol": "BTC/USDT",
        "bid": 40000,
        "ask": 40100,
        "last": 40050,
        "volume": 1000,
        "timestamp": 1625097600000
    }
    mock_exchanges["exchange_2"].fetch_ticker.return_value = mock_ticker
    
    # Calcular spread
    spread = exchange_manager.calculate_spread("BTC/USDT", "exchange_2")
    
    # Verificar que se llamó al método correcto y se obtuvo el resultado esperado
    mock_exchanges["exchange_2"].fetch_ticker.assert_called_once_with("BTC/USDT")
    
    # El spread debería ser (ask - bid) / ((ask + bid) / 2) * 100
    expected_spread = (40100 - 40000) / ((40100 + 40000) / 2) * 100  # ~0.25%
    assert spread == pytest.approx(expected_spread, abs=1e-10)


def test_exchange_manager_compare_prices(exchange_manager, mock_exchanges, exchange_selector):
    """Prueba la comparación de precios entre varios exchanges."""
    # Configurar mocks para obtener tickers en diferentes exchanges
    tickers = {
        "exchange_1": {"symbol": "BTC/USDT", "bid": 40000, "ask": 40100},
        "exchange_2": {"symbol": "BTC/USDT", "bid": 40050, "ask": 40150},
        "exchange_3": {"symbol": "BTC/USDT", "bid": 39950, "ask": 40050}
    }
    
    for exchange_name, ticker in tickers.items():
        mock_exchanges[exchange_name].fetch_ticker.return_value = ticker
    
    # Comparar precios
    comparison = exchange_manager.compare_prices("BTC/USDT", ["exchange_1", "exchange_2", "exchange_3"])
    
    # Verificar que se llamó al método correcto para cada exchange
    for exchange_name in tickers.keys():
        mock_exchanges[exchange_name].fetch_ticker.assert_called_once_with("BTC/USDT")
    
    # Verificar resultados
    assert len(comparison) == 3
    for exchange_name in tickers.keys():
        assert exchange_name in comparison
        assert comparison[exchange_name]["bid"] == tickers[exchange_name]["bid"]
        assert comparison[exchange_name]["ask"] == tickers[exchange_name]["ask"]
    
    # Verificar exchange con mejor precio de compra (mayor bid)
    assert comparison["best_bid_exchange"] == "exchange_2"
    assert comparison["best_bid"] == 40050
    
    # Verificar exchange con mejor precio de venta (menor ask)
    assert comparison["best_ask_exchange"] == "exchange_3"
    assert comparison["best_ask"] == 40050
    
    # Verificar el spread entre el mejor bid y el mejor ask
    assert comparison["cross_exchange_spread"] == pytest.approx(0.0, abs=1e-10)