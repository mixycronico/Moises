"""
Inicializador del Sistema Divino con ML para Genesis.

Este módulo se encarga de inicializar y configurar el Sistema Divino con Machine Learning
en el momento del arranque de Genesis, integrándolo con los componentes existentes.

El Sistema Divino proporciona:
- Procesamiento ultra-rápido y resiliente de operaciones de base de datos
- Predicción y optimización automática de recursos
- Priorización inteligente de tareas basada en ML
- Tolerancia a fallos extrema con recuperación automática
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from genesis.db.divine_system_integrator import divine_system, DivineSystem
from genesis.db.divine_task_queue import OperationMode

# Configuración de logging
logger = logging.getLogger("genesis.init.divine_system")

async def initialize_divine_system(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Inicializar el Sistema Divino.
    
    Args:
        config: Configuración opcional
        
    Returns:
        True si se inicializó correctamente
    """
    try:
        # Configuración predeterminada
        if not config:
            config = {
                "mode": OperationMode.DIVINO,
                "auto_start": True
            }
        
        # Acceder a la instancia global
        global divine_system
        
        # Inicializar si es necesario
        if not divine_system.initialized:
            await divine_system.initialize()
        
        # Iniciar si está configurado para auto-arranque
        if config.get("auto_start", True) and not divine_system.divine_queue.running:
            await divine_system.start()
        
        logger.info(f"Sistema Divino inicializado en modo {config.get('mode', OperationMode.DIVINO)}")
        return True
        
    except Exception as e:
        logger.error(f"Error al inicializar Sistema Divino: {e}")
        return False

def register_divine_system(registry, context=None):
    """
    Registrar el Sistema Divino en el registro de componentes de Genesis.
    
    Args:
        registry: Registro de componentes
        context: Contexto de inicialización
        
    Returns:
        True si se registró correctamente
    """
    if not registry:
        logger.warning("No se pudo registrar Sistema Divino: registro no proporcionado")
        return False
        
    try:
        # Detectar configuración
        config = {}
        if context and "config" in context:
            if "divine_system" in context["config"]:
                config = context["config"]["divine_system"]
        
        # Inicializar sistema (asíncrono)
        loop = asyncio.get_event_loop()
        init_success = loop.run_until_complete(initialize_divine_system(config))
        
        if init_success:
            # Registrar el sistema
            registry["divine_system"] = {
                "available": True,
                "mode": config.get("mode", OperationMode.DIVINO),
                "initialized_at": loop.time()
            }
            
            # Verificar estado y estadísticas
            stats = loop.run_until_complete(divine_system.get_system_stats())
            registry["divine_system"]["status"] = stats["divine_system"]
            
            logger.info("Sistema Divino registrado en Genesis")
            return True
            
        else:
            registry["divine_system"] = {
                "available": False,
                "error": "Falló la inicialización"
            }
            
            logger.warning("Sistema Divino no registrado: falló inicialización")
            return False
            
    except Exception as e:
        logger.error(f"Error al registrar Sistema Divino: {e}")
        
        # Registrar con error
        if registry:
            registry["divine_system"] = {
                "available": False,
                "error": str(e)
            }
            
        return False