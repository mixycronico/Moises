"""
Pruebas unitarias para componentes básicos del sistema Genesis.

Este módulo prueba funcionalidades básicas como configuración, logging,
seguridad y gestión de exchanges.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch

# Importar componentes del sistema
from genesis.config import Settings
from genesis.utils.logger import setup_logging
from genesis.security.manager import SecurityUtils
from genesis.exchanges.manager import ExchangeManager


# Fixtures para reutilizar instancias en las pruebas
@pytest.fixture
def config():
    """Fixture que proporciona una instancia de Settings."""
    from genesis.config.settings import Settings
    return Settings()


@pytest.fixture
def exchange_manager():
    """Fixture que proporciona una instancia de ExchangeManager con exchanges simulados."""
    # Crear configuraciones de ejemplo para exchanges
    exchange_configs = {
        "binance": {
            "api_key": "test_api_key_binance",
            "api_secret": "test_api_secret_binance"
        },
        "kucoin": {
            "api_key": "test_api_key_kucoin",
            "api_secret": "test_api_secret_kucoin"
        }
    }
    
    # Instanciar el ExchangeManager con la configuración
    manager = ExchangeManager(exchange_configs=exchange_configs)
    
    # Para pruebas, simulamos la inicialización directa de algunos exchanges
    manager.exchanges = {
        "binance": Mock(),
        "kucoin": Mock()
    }
    
    # También simulamos el selector
    manager.selector = Mock()
    
    return manager


@pytest.fixture
def mock_logger():
    """Fixture que simula el comportamiento del logging."""
    with patch('genesis.utils.logger.logging') as mock_logging:
        yield mock_logging


# Pruebas para Config
def test_config_load_and_set(config):
    """Prueba la carga y el establecimiento de configuraciones."""
    # Valor por defecto
    assert config.get("log_level") == "INFO"

    # Cambiar y verificar
    config.set("log_level", "DEBUG")
    assert config.get("log_level") == "DEBUG"


def test_config_get_non_existent_key(config):
    """Prueba el comportamiento con una clave inexistente."""
    assert config.get("non_existent_key") is None
    assert config.get("non_existent_key", default="DEFAULT") == "DEFAULT"


def test_config_set_invalid_type(config):
    """Prueba el manejo de tipos no válidos al establecer configuraciones."""
    with pytest.raises(TypeError):
        config.set("log_level", 123)  # Suponiendo que Config valida tipos


# Pruebas para Logger
def test_logger_setup_logging(mock_logger):
    """Prueba que se configure el logging correctamente."""
    try:
        logger = setup_logging("test_logger")
        assert logger is not None
        mock_logger.getLogger.assert_called_with("test_logger")
    except Exception:
        pytest.fail("Logger setup failed unexpectedly")


def test_logger_setup_logging_with_custom_level(mock_logger, config):
    """Prueba la configuración del logger con un nivel personalizado."""
    config.set("log_level", "DEBUG")
    setup_logging("test_logger", level="DEBUG")
    mock_logger.basicConfig.assert_called_once_with(level="DEBUG")


def test_logger_setup_logging_failure(mock_logger):
    """Prueba el manejo de errores al configurar el logging."""
    mock_logger.basicConfig.side_effect = Exception("Logging setup failed")
    with pytest.raises(Exception, match="Logging setup failed"):
        setup_logging("test_logger")


# Pruebas para Security
def test_security_hashing_and_verification():
    """Prueba el hashing y la verificación de contraseñas."""
    password = "securepassword"
    hashed = SecurityUtils.hash_password(password)
    
    # Verificación correcta
    assert SecurityUtils.verify_password(password, hashed)
    
    # Verificación incorrecta
    wrong_password = "wrongpassword"
    assert not SecurityUtils.verify_password(wrong_password, hashed)


def test_security_hash_password_empty():
    """Prueba el manejo de contraseñas vacías."""
    with pytest.raises(ValueError, match="Password cannot be empty"):
        SecurityUtils.hash_password("")


def test_security_verify_password_invalid_hash():
    """Prueba la verificación con un hash inválido."""
    password = "securepassword"
    invalid_hash = "invalid_hash_format"
    assert not SecurityUtils.verify_password(password, invalid_hash)


# Pruebas para ExchangeManager
def test_exchange_manager_initialization(exchange_manager):
    """Prueba la inicialización de ExchangeManager."""
    assert "binance" in exchange_manager.exchanges
    assert "kucoin" in exchange_manager.exchanges
    assert exchange_manager.selector is not None


@pytest.mark.asyncio
async def test_exchange_manager_get_best_exchange(exchange_manager):
    """Prueba la selección del mejor exchange."""
    # Mockear el selector de exchanges con una coroutine como return_value
    mock_selector = Mock()
    
    # Crear una coroutine que devuelva "binance"
    async def mock_get_best_exchange(trading_pair):
        return "binance"
    
    mock_selector.get_best_exchange = Mock(side_effect=mock_get_best_exchange)
    exchange_manager.selector = mock_selector
    
    # Probar selección de mejor exchange
    best_exchange = await exchange_manager.get_best_exchange("BTC/USDT")
    assert best_exchange == "binance"
    
    # Verificar que se llamó al selector con los parámetros correctos
    mock_selector.get_best_exchange.assert_called_once_with("BTC/USDT")


@pytest.mark.asyncio
async def test_exchange_manager_get_best_exchange_no_exchanges():
    """Prueba el comportamiento cuando no hay exchanges disponibles."""
    # Crear un exchange_manager con una configuración mínima
    exchange_manager = ExchangeManager(exchange_configs={})
    exchange_manager.exchanges = {}
    
    # También necesitamos mockear el selector
    mock_selector = Mock()
    
    # Crear una coroutine que lance una excepción
    async def mock_get_best_exchange(trading_pair):
        raise ValueError("No exchanges available")
        
    mock_selector.get_best_exchange = Mock(side_effect=mock_get_best_exchange)
    exchange_manager.selector = mock_selector
    
    with pytest.raises(ValueError, match="No exchanges available"):
        await exchange_manager.get_best_exchange("BTC/USDT")


@pytest.mark.asyncio
async def test_exchange_manager_execute_trade(exchange_manager):
    """Prueba la ejecución de una operación en el mejor exchange."""
    # Crear un mock para el exchange
    mock_exchange = Mock()
    mock_exchange.place_order.return_value = {"status": "ok", "order_id": "12345"}
    
    # Mockear el método get_best_exchange para que devuelva un nombre de exchange
    async def mock_get_best_exchange(trading_pair):
        return "binance"
        
    # Mockear el método emit_event para que no intente usar el bus de eventos
    async def mock_emit_event(event_type, data):
        pass  # No hacemos nada, solo evitamos la excepción
    
    # Establecemos el mock en el diccionario de exchanges
    exchange_manager.exchanges["binance"] = mock_exchange
    
    # Parcheamos los métodos necesarios
    with patch.object(exchange_manager, 'get_best_exchange', side_effect=mock_get_best_exchange):
        with patch.object(exchange_manager, 'emit_event', side_effect=mock_emit_event):
            # Probar ejecución de trade
            result = await exchange_manager.execute_trade("BTC/USDT", "buy", 0.01, price=40000)
            
            # Verificar resultado
            assert result["status"] == "ok"
            assert result["order_id"] == "12345"
            
            # Verificar que se llamó al método place_order del exchange con los parámetros correctos
            mock_exchange.place_order.assert_called_once_with("BTC/USDT", "buy", 0.01, 40000)


@pytest.mark.asyncio
async def test_exchange_manager_execute_trade_invalid_symbol(exchange_manager):
    """Prueba el manejo de símbolos inválidos."""
    # Crear un mock para el exchange que falla con símbolos inválidos
    mock_exchange = Mock()
    mock_exchange.place_order.side_effect = ValueError("Invalid symbol")
    
    # Mockear el método get_best_exchange para que devuelva un nombre de exchange
    async def mock_get_best_exchange(trading_pair):
        return "binance"
        
    # Mockear el método emit_event para que no intente usar el bus de eventos
    async def mock_emit_event(event_type, data):
        pass  # No hacemos nada, solo evitamos la excepción
    
    # Establecemos el mock en el diccionario de exchanges
    exchange_manager.exchanges["binance"] = mock_exchange
    
    # Parcheamos los métodos necesarios
    with patch.object(exchange_manager, 'get_best_exchange', side_effect=mock_get_best_exchange):
        with patch.object(exchange_manager, 'emit_event', side_effect=mock_emit_event):
            # Probar ejecución de trade con símbolo inválido
            with pytest.raises(ValueError, match="Invalid symbol"):
                await exchange_manager.execute_trade("INVALID/SYMBOL", "buy", 0.01)


@pytest.mark.asyncio
async def test_exchange_manager_execute_trade_exchange_unavailable(exchange_manager):
    """Prueba el comportamiento cuando el exchange no está disponible."""
    # Mockear get_best_exchange para que devuelva None
    async def mock_get_best_exchange(trading_pair):
        return None
    
    # Mockear el método emit_event para que no intente usar el bus de eventos
    async def mock_emit_event(event_type, data):
        pass  # No hacemos nada, solo evitamos la excepción
    
    # Parcheamos los métodos necesarios
    with patch.object(exchange_manager, 'get_best_exchange', side_effect=mock_get_best_exchange):
        with patch.object(exchange_manager, 'emit_event', side_effect=mock_emit_event):
            # Probar ejecución de trade con exchange no disponible
            result = await exchange_manager.execute_trade("BTC/USDT", "buy", 0.01)
            assert result["status"] == "error"
            assert "No suitable exchange found" in result["message"]


@pytest.mark.asyncio
async def test_exchange_manager_get_market_data(exchange_manager):
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

    # Configurar el mock para devolver mock_data al ser llamado con fetch_ticker
    # Esta vez usamos __call__ para simular que es callable directamente
    mock_exchange.fetch_ticker = lambda symbol: mock_data
    
    # Mockear el método get_best_exchange para que devuelva un nombre de exchange
    async def mock_get_best_exchange(trading_pair):
        return "binance"
        
    # Mockear el método emit_event para que no intente usar el bus de eventos
    async def mock_emit_event(event_type, data):
        pass  # No hacemos nada, solo evitamos la excepción
    
    # Establecemos el mock en el diccionario de exchanges
    exchange_manager.exchanges["binance"] = mock_exchange
    
    # Configuramos un future asincrono para el loop.run_in_executor
    future = asyncio.Future()
    future.set_result(mock_data)
    
    # Usamos run_in_executor para convertir la llamada síncrona a asíncrona
    # Reemplazamos esto con un mock
    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor.return_value = future
        
        # Parcheamos get_best_exchange y emit_event
        with patch.object(exchange_manager, 'get_best_exchange', side_effect=mock_get_best_exchange):
            with patch.object(exchange_manager, 'emit_event', side_effect=mock_emit_event):
                # Probar obtención de datos
                ticker = await exchange_manager.get_ticker("BTC/USDT")
                
                # Verificar resultado
                assert ticker["symbol"] == "BTC/USDT"
                assert ticker["last"] == 40000
                
                # No podemos verificar la llamada directamente porque estamos
                # usando una lambda en lugar de un Mock


@pytest.mark.asyncio
async def test_exchange_manager_get_balance(exchange_manager):
    """Prueba la obtención de saldo en un exchange."""
    # Crear un mock para el exchange
    mock_exchange = Mock()
    mock_balance = {
        "BTC": {"free": 1.0, "used": 0.5, "total": 1.5},
        "USDT": {"free": 10000, "used": 5000, "total": 15000}
    }
    
    # Configurar el mock para devolver mock_balance sin argumentos
    # Esta vez usamos un callable simple en lugar de Mock
    mock_exchange.fetch_balance = lambda: mock_balance
    
    # Mockear el método emit_event para que no intente usar el bus de eventos
    async def mock_emit_event(event_type, data):
        pass  # No hacemos nada, solo evitamos la excepción
    
    # Establecemos el mock en el diccionario de exchanges
    exchange_manager.exchanges["binance"] = mock_exchange
    
    # Configuramos un future asincrono para el loop.run_in_executor
    future = asyncio.Future()
    future.set_result(mock_balance)
    
    # Usamos run_in_executor para convertir la llamada síncrona a asíncrona
    # Reemplazamos esto con un mock
    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor.return_value = future
        
        # Parcheamos emit_event
        with patch.object(exchange_manager, 'emit_event', side_effect=mock_emit_event):
            # Probar obtención de saldo
            balance = await exchange_manager.get_balance("binance")
            
            # Verificar resultado
            assert balance["BTC"]["total"] == 1.5
            assert balance["USDT"]["free"] == 10000
            
            # No podemos verificar la llamada directamente porque estamos
            # usando una lambda en lugar de un Mock


# Pruebas de integración
def test_config_and_logger_integration(config, mock_logger):
    """Prueba la integración entre Config y Logger."""
    # Para la prueba, configuramos explícitamente el nivel en Settings
    config.set("log_level", "WARNING")
    
    # Mockeamos settings.get para que devuelva "WARNING"
    with patch('genesis.config.settings.settings.get', return_value="WARNING"):
        setup_logging("test_logger")
        mock_logger.basicConfig.assert_called_once_with(level="WARNING")


def test_security_and_exchange_manager_integration(exchange_manager):
    """Prueba la integración entre Security y ExchangeManager (simulada)."""
    password = "securepassword"
    hashed = SecurityUtils.hash_password(password)
    
    # Simulamos un método que usa credenciales seguras para autenticar un exchange
    mock_client = Mock()
    mock_client.authenticate.return_value = True
    exchange_manager.exchanges["binance"] = mock_client

    with patch.object(exchange_manager, 'get_best_exchange', return_value="binance"):
        exchange_manager.exchanges["binance"].authenticate(hashed)
        mock_client.authenticate.assert_called_once_with(hashed)