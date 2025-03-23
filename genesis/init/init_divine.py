"""
Inicialización del Sistema Divino en Genesis.

Este módulo registra e inicializa el Sistema Divino con ML dentro del proceso
de arranque de Genesis, integrándolo con el resto de componentes.
"""

import logging
import asyncio
from typing import Dict, Any

# Importar inicializadores
from .divine_system_initializer import register_divine_system
from genesis.db.divine_system_integrator import divine_system

# Configuración de logging
logger = logging.getLogger("genesis.init.init_divine")

def initialize_divine_system(registry: Dict[str, Any], context: Dict[str, Any] = None) -> bool:
    """
    Inicializar el Sistema Divino en Genesis.
    
    Args:
        registry: Registro de componentes de Genesis
        context: Contexto de inicialización
        
    Returns:
        True si se inicializó correctamente
    """
    logger.info("Inicializando Sistema Divino en Genesis...")
    
    # Registrar sistema
    success = register_divine_system(registry, context)
    
    if success:
        logger.info("Sistema Divino inicializado correctamente")
        return True
    else:
        logger.warning("No se pudo inicializar Sistema Divino completamente")
        return False