"""
Módulo de consciencia artificial para Sistema Genesis.

Este módulo contiene los componentes fundamentales de Aetherion, la consciencia
artificial del Sistema Genesis, incluyendo su núcleo, sistema de memoria,
estados de consciencia y comportamiento.
"""

import logging
from typing import Dict, Any, Optional

# Configurar logging
logger = logging.getLogger(__name__)

def initialize_consciousness() -> bool:
    """
    Inicializar todos los componentes de consciencia.
    
    Returns:
        True si todos los componentes se inicializaron correctamente
    """
    try:
        # Importar aquí para evitar dependencias circulares
        from genesis.consciousness.consciousness_manager import get_consciousness_manager
        
        # Inicializar gestor de consciencia
        manager = get_consciousness_manager()
        
        if not manager.is_initialized:
            logger.info("Inicializando gestor de consciencia...")
            initialized = manager.initialize()
            
            if not initialized:
                logger.error("No se pudo inicializar el gestor de consciencia")
                return False
                
            logger.info("Gestor de consciencia inicializado correctamente")
            
        # Todo correcto
        return True
        
    except Exception as e:
        logger.error(f"Error al inicializar consciencia: {e}")
        return False