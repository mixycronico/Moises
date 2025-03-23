"""
Inicializador del Sistema Divino de Base de Datos para Genesis.

Este módulo se encarga de inicializar y configurar el Sistema Divino de Base de Datos,
que implementa una arquitectura híbrida Redis + RabbitMQ para procesamiento de tareas
de base de datos con máxima eficiencia y confiabilidad.

La inicialización se realiza de forma transparente, permitiendo que Genesis
funcione correctamente incluso si Redis o RabbitMQ no están disponibles.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional

from genesis.db.divine_integrator import (
    initialize_divine_system,
    OperationMode
)

# Configuración de logging
logger = logging.getLogger("genesis.init.divine_db")

async def initialize_divine_database_system(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Inicializar el Sistema Divino de Base de Datos.
    
    Args:
        config: Configuración opcional
        
    Returns:
        True si se inicializó correctamente
    """
    try:
        # Configuración predeterminada
        mode = OperationMode.DIVINO
        auto_start = True
        
        # Usar configuración proporcionada si existe
        if config:
            mode = config.get("mode", mode)
            auto_start = config.get("auto_start", auto_start)
        
        # Inicializar sistema
        await initialize_divine_system(mode=mode, auto_start=auto_start)
        
        logger.info(f"Sistema Divino de Base de Datos inicializado en modo {mode}")
        return True
        
    except Exception as e:
        logger.error(f"Error al inicializar Sistema Divino de Base de Datos: {e}")
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
            if "divine_db" in context["config"]:
                config = context["config"]["divine_db"]
        
        # Inicializar sistema (asíncrono)
        # Nota: En Genesis debemos ejecutar esto en un bucle de eventos existente
        #       o crear uno nuevo si es necesario
        loop = asyncio.get_event_loop()
        init_success = loop.run_until_complete(initialize_divine_database_system(config))
        
        if init_success:
            # Registrar el sistema
            registry["divine_db_system"] = {
                "available": True,
                "mode": config.get("mode", OperationMode.DIVINO),
                "initialized_at": loop.time()
            }
            
            logger.info("Sistema Divino de Base de Datos registrado en Genesis")
            return True
        else:
            registry["divine_db_system"] = {
                "available": False,
                "error": "Falló la inicialización"
            }
            
            logger.warning("Sistema Divino de Base de Datos no registrado: falló inicialización")
            return False
            
    except Exception as e:
        logger.error(f"Error al registrar Sistema Divino de Base de Datos: {e}")
        
        # Registrar con error
        if registry:
            registry["divine_db_system"] = {
                "available": False,
                "error": str(e)
            }
            
        return False