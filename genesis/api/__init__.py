"""
Módulo de API para el sistema Genesis.

Este módulo proporciona la interfaz de programación de aplicaciones (API)
para interactuar con el sistema Genesis desde aplicaciones externas.
"""

from genesis.api.endpoints import create_routes
from genesis.api.init_api import init_api
from genesis.api.rest import RestAPI
from genesis.api.server import APIServer
from genesis.api.swagger import init_swagger

__all__ = [
    "create_routes", 
    "init_api", 
    "RestAPI", 
    "APIServer", 
    "init_swagger"
]