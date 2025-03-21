"""
Módulo API para el sistema Genesis.

Este módulo proporciona interfaces para la comunicación con sistemas externos,
incluyendo API REST, WebSockets, y aplicación web.
"""

from genesis.api.init_api import init_api
from genesis.api.swagger import init_swagger
from genesis.api.logger import initialize_logging

__all__ = ["init_api", "init_swagger", "initialize_logging"]