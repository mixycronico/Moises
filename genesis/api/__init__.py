"""
Módulo de API para el Sistema Genesis.

Este módulo proporciona los endpoints de API para el Sistema Genesis,
incluyendo interfaces para componentes como Aetherion, Buddha y Gabriel.
"""

# Importar controladores
from genesis.api.aetherion_controller import register_routes as register_aetherion_routes

def register_all_routes(app):
    """
    Registrar todas las rutas de API en la aplicación Flask.
    
    Args:
        app: Aplicación Flask
    """
    # Registrar rutas de Aetherion
    register_aetherion_routes(app)