"""
Pruebas unitarias para componentes básicos del sistema Genesis.

Este módulo prueba funcionalidades básicas como configuración, logging,
seguridad y gestión de exchanges.
"""

import pytest
from unittest.mock import Mock, patch

# Importar componentes del sistema
from genesis.config import Config
from genesis.utils.logger import setup_logging
from genesis.security.manager import SecurityUtils
from genesis.exchanges.manager import ExchangeManager


def test_config_load_and_set():
    """Prueba la carga y el establecimiento de configuraciones."""
    config = Config()
    default_value = config.get("log_level")
    assert default_value == "INFO"

    # Cambiar y verificar
    config.set("log_level", "DEBUG")
    assert config.get("log_level") == "DEBUG"


def test_logger_setup_logging():
    """Prueba que se configure el logging correctamente."""
    # Esto no prueba un resultado directo sino que asegura que el método no lanza excepciones.
    try:
        logger = setup_logging("test_logger")
        assert logger is not None
    except Exception:
        pytest.fail("Logger setup failed unexpectedly")


def test_security_hashing_and_verification():
    """Prueba el hashing y la verificación de contraseñas."""
    password = "securepassword"
    hashed = SecurityUtils.hash_password(password)
    assert SecurityUtils.verify_password(password, hashed)

    wrong_password = "wrongpassword"
    assert not SecurityUtils.verify_password(wrong_password, hashed)


def test_exchange_manager_get_best_exchange():
    """Prueba la selección del mejor exchange."""
    # Mockear el selector de exchanges
    mock_selector = Mock()
    mock_selector.get_best_exchange.return_value = "binance"
    
    # Crear exchange manager con el selector mockeado
    exchange_manager = ExchangeManager()
    exchange_manager.selector = mock_selector
    
    # Probar selección de mejor exchange
    best_exchange = exchange_manager.get_best_exchange("BTC/USDT")
    assert best_exchange == "binance"
    
    # Verificar que se llamó al selector con los parámetros correctos
    mock_selector.get_best_exchange.assert_called_once_with("BTC/USDT")


def test_exchange_manager_execute_trade():
    """Prueba la ejecución de una operación en el mejor exchange."""
    # Crear un mock para el exchange
    mock_exchange = Mock()
    mock_exchange.execute_trade.return_value = {"status": "success", "order_id": "12345"}
    
    # Mockear el método get_exchange para que devuelva nuestro mock
    with patch.object(ExchangeManager, 'get_exchange', return_value=mock_exchange):
        # Crear exchange manager
        exchange_manager = ExchangeManager()
        
        # Probar ejecución de trade
        result = exchange_manager.execute_trade("BTC/USDT", "buy", 0.01, price=40000)
        
        # Verificar resultado
        assert result["status"] == "success"
        assert result["order_id"] == "12345"
        
        # Verificar que se llamó al método execute_trade del exchange con los parámetros correctos
        mock_exchange.execute_trade.assert_called_once_with("BTC/USDT", "buy", 0.01, price=40000)


def test_exchange_manager_get_market_data():
    """Prueba la obtención de datos de mercado."""
    # Crear un mock para el exchange
    mock_exchange = Mock()
    mock_data = {
        "symbol": "BTC/USDT",
        "last": 40000,
        "bid": 39990,
        "ask": 40010,
        "volume": 100
    }
    mock_exchange.fetch_ticker.return_value = mock_data
    
    # Mockear el método get_exchange para que devuelva nuestro mock
    with patch.object(ExchangeManager, 'get_exchange', return_value=mock_exchange):
        # Crear exchange manager
        exchange_manager = ExchangeManager()
        
        # Probar obtención de datos
        ticker = exchange_manager.get_ticker("BTC/USDT")
        
        # Verificar resultado
        assert ticker["symbol"] == "BTC/USDT"
        assert ticker["last"] == 40000
        
        # Verificar que se llamó al método fetch_ticker del exchange con los parámetros correctos
        mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")


def test_exchange_manager_get_balance():
    """Prueba la obtención de saldo en un exchange."""
    # Crear un mock para el exchange
    mock_exchange = Mock()
    mock_balance = {
        "BTC": {"free": 1.0, "used": 0.5, "total": 1.5},
        "USDT": {"free": 10000, "used": 5000, "total": 15000}
    }
    mock_exchange.fetch_balance.return_value = mock_balance
    
    # Mockear el método get_exchange para que devuelva nuestro mock
    with patch.object(ExchangeManager, 'get_exchange', return_value=mock_exchange):
        # Crear exchange manager
        exchange_manager = ExchangeManager()
        
        # Probar obtención de saldo
        balance = exchange_manager.get_balance("binance")
        
        # Verificar resultado
        assert balance["BTC"]["total"] == 1.5
        assert balance["USDT"]["free"] == 10000
        
        # Verificar que se llamó al método fetch_balance del exchange
        mock_exchange.fetch_balance.assert_called_once()