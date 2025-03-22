"""
Configuración global para tests del sistema Genesis.

Este módulo configura el entorno de pruebas, incluyendo:
- Configuración de logging
- Configuración de variables de entorno
- Fixtures comunes
- Ajuste de PYTHONPATH para importaciones
"""

import os
import sys
import logging
import pytest

# Agregar el directorio raíz al PYTHONPATH para permitir importaciones desde genesis
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    print(f"Añadido {root_dir} al PYTHONPATH")

# Configuración de logging para pruebas
def configure_logging():
    """Configurar logging para todas las pruebas."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Configurar marcadores personalizados para pytest
def pytest_configure(config):
    """Configurar marcadores personalizados para pytest."""
    config.addinivalue_line("markers", "slow: marca pruebas que tardan mucho tiempo")
    config.addinivalue_line("markers", "integration: marca pruebas de integración")
    config.addinivalue_line("markers", "core: marca pruebas del módulo core")

# Configurar variables de entorno para pruebas
@pytest.fixture(scope="session", autouse=True)
def env_setup():
    """Configurar variables de entorno para pruebas."""
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    return os.environ

# Ejecutar configuración inicial
configure_logging()