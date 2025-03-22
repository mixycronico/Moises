"""
Configuración centralizada para las pruebas de pytest.

Este archivo contiene fixtures y configuraciones compartidas
entre diferentes módulos de prueba.
"""

import os
import pytest
import logging
from unittest.mock import MagicMock, patch


# Mockear el setup_logging para evitar problemas de inicialización del logger
@pytest.fixture
def mock_logger():
    """
    Mockear la función setup_logging y los loggers utilizados en las pruebas.
    
    Esto evita problemas de inicialización en las pruebas que utilizan componentes
    que intentan configurar el logger internamente.
    """
    # Importamos explícitamente desde genesis.utils.logger para hacer el patching
    from genesis.utils.logger import setup_logging
    
    # Mock de logging módulo completo
    mock_logger = MagicMock()
    
    # Constantes necesarias para las pruebas
    mock_logger.DEBUG = logging.DEBUG       # Valor numérico: 10
    mock_logger.INFO = logging.INFO         # Valor numérico: 20
    mock_logger.WARNING = logging.WARNING   # Valor numérico: 30
    mock_logger.ERROR = logging.ERROR       # Valor numérico: 40
    mock_logger.CRITICAL = logging.CRITICAL # Valor numérico: 50
    
    # Métodos comunes
    mock_logger.info = MagicMock()
    mock_logger.error = MagicMock()
    mock_logger.warning = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.getLogger = MagicMock(return_value=mock_logger)
    
    # Usar valores numéricos directamente para facilitar las comparaciones en tests
    def mock_basicConfig(**kwargs):
        if 'level' in kwargs:
            # Convertir niveles de string a números
            if kwargs['level'] == 'DEBUG':
                kwargs['level'] = 10  # logging.DEBUG
            elif kwargs['level'] == 'INFO':
                kwargs['level'] = 20  # logging.INFO
            elif kwargs['level'] == 'WARNING':
                kwargs['level'] = 30  # logging.WARNING
            elif kwargs['level'] == 'ERROR':
                kwargs['level'] = 40  # logging.ERROR
            elif kwargs['level'] == 'CRITICAL':
                kwargs['level'] = 50  # logging.CRITICAL
        return None
        
    mock_logger.basicConfig = MagicMock(side_effect=mock_basicConfig)
    
    # Patch de logging y setup_logging
    with patch('genesis.utils.logger.logging', mock_logger), \
         patch('genesis.utils.logger.setup_logging', return_value=mock_logger):
        yield mock_logger


# Configurar logging básico para pruebas
@pytest.fixture(autouse=True)
def configure_logging():
    """Configurar logging para todas las pruebas."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    yield
    # Limpiar handlers para evitar duplicación de logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)


# Fixture para simular variables de entorno
@pytest.fixture
def env_setup():
    """Configurar variables de entorno para pruebas."""
    old_env = {}
    
    # Guardar valores originales
    for key in ["DATABASE_URL", "LOG_LEVEL", "API_KEYS"]:
        old_env[key] = os.environ.get(key)
    
    # Establecer valores de prueba
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["API_KEYS"] = "test_key1,test_key2"
    
    yield
    
    # Restaurar valores originales
    for key, value in old_env.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value


# Marcador asyncio para pruebas asíncronas
def pytest_configure(config):
    """Configurar marcadores personalizados para pytest."""
    config.addinivalue_line(
        "markers", "asyncio: marca pruebas asíncronas que utilizan asyncio"
    )
    
    # Configurar variables de entorno para todas las pruebas
    # Esta es la manera recomendada de configurar variables de entorno en lugar
    # de usar la sección "env" en pytest.ini, que está obsoleta
    if not os.environ.get("TEST_DB_URL"):
        os.environ["TEST_DB_URL"] = "sqlite:///:memory:"
    if not os.environ.get("LOG_LEVEL"):
        os.environ["LOG_LEVEL"] = "INFO"
    if not os.environ.get("TEST_MODE"):
        os.environ["TEST_MODE"] = "True"