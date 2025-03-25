"""
Módulo API para el Sistema Genesis.

Este módulo proporciona interfaces API para interactuar con los componentes
del Sistema Genesis, incluyendo Aetherion, integraciones y demás módulos.
"""

import logging
from typing import Dict, Any, Optional

# Configurar logging
logger = logging.getLogger(__name__)

# Exportar componentes importantes
from genesis.api.aetherion_controller import get_aetherion_controller, get_aetherion_blueprint

def initialize_api() -> bool:
    """
    Inicializar todos los componentes API.
    
    Returns:
        True si todos los componentes se inicializaron correctamente
    """
    try:
        # Inicializar controlador Aetherion
        controller = get_aetherion_controller()
        
        if not controller.is_initialized:
            logger.warning("Controlador Aetherion no inicializado automáticamente. Intentando inicializar manualmente...")
            if not controller.initialize():
                logger.error("No se pudo inicializar el controlador Aetherion")
                return False
        
        logger.info("API inicializada correctamente")
        return True
    except Exception as e:
        logger.error(f"Error al inicializar API: {e}")
        return False

def register_api_routes(app) -> bool:
    """
    Registrar todas las rutas API en la aplicación Flask.
    
    Args:
        app: Aplicación Flask
        
    Returns:
        True si se registraron correctamente
    """
    try:
        # Registrar blueprint de Aetherion
        app.register_blueprint(get_aetherion_blueprint())
        
        logger.info("Rutas API registradas correctamente")
        return True
    except Exception as e:
        logger.error(f"Error al registrar rutas API: {e}")
        return False