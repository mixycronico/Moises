"""
Configuración centralizada para las pruebas de pytest.

Este archivo contiene fixtures y configuraciones compartidas
entre diferentes módulos de prueba.
"""

import os
import pytest
import logging


# Configurar logging para pruebas
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