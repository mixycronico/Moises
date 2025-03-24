"""
Main entry point for the Genesis trading system.

Este módulo inicializa y arranca el sistema, configurando todos los componentes
y proporcionando el punto de entrada principal para la operación.
También expone la aplicación Flask para Gunicorn, incluyendo la API REST
para integración con componentes cloud externos.
"""

import os
import sys
import logging
import asyncio
from flask import Flask, jsonify, request
from app import app as flask_app

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("genesis.main")

# Intentar importar componentes cloud
try:
    from genesis.cloud import (
        circuit_breaker_factory, 
        checkpoint_manager,
        load_balancer_manager,
        create_cloud_api
    )
    HAS_CLOUD_COMPONENTS = True
except ImportError:
    logger.warning("Componentes cloud no disponibles")
    HAS_CLOUD_COMPONENTS = False

# Función asincrónica para inicialización
async def init_cloud_components():
    """Inicializar componentes cloud."""
    if not HAS_CLOUD_COMPONENTS:
        return False
    
    try:
        logger.info("Inicializando componentes cloud...")
        
        # Inicializar checkpoint manager
        if checkpoint_manager and hasattr(checkpoint_manager, "initialize"):
            await checkpoint_manager.initialize(storage_type="LOCAL_FILE")
        
        # Inicializar circuit breaker factory
        # (ya se inicializa automáticamente)
        
        # Inicializar load balancer manager
        if load_balancer_manager and hasattr(load_balancer_manager, "initialize"):
            await load_balancer_manager.initialize()
        
        # Crear API REST
        api = create_cloud_api(flask_app, url_prefix="/api/cloud", enable_swagger=True)
        
        logger.info("Componentes cloud inicializados correctamente")
        return True
        
    except Exception as e:
        logger.exception(f"Error al inicializar componentes cloud: {e}")
        return False

# Función sincrónica para inicialización
def init_components():
    """Inicializar todos los componentes."""
    # Inicializar componentes cloud de forma asincrónica
    if HAS_CLOUD_COMPONENTS:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(init_cloud_components())
        finally:
            loop.close()

# Ejecutar inicialización
init_components()

# Crear endpoints de estado específicos para cloud
@flask_app.route("/api/cloud-status")
def cloud_status():
    """Verificar estado de componentes cloud."""
    if not HAS_CLOUD_COMPONENTS:
        return jsonify({
            "available": False,
            "message": "Componentes cloud no disponibles"
        })
    
    try:
        # Obtener estado de components
        components = {
            "circuit_breaker": {
                "available": circuit_breaker_factory is not None,
                "count": len(circuit_breaker_factory._circuit_breakers) if hasattr(circuit_breaker_factory, "_circuit_breakers") else 0
            },
            "checkpoint_manager": {
                "available": checkpoint_manager is not None,
                "initialized": hasattr(checkpoint_manager, "initialized") and checkpoint_manager.initialized
            },
            "load_balancer": {
                "available": load_balancer_manager is not None,
                "count": len(load_balancer_manager._balancers) if hasattr(load_balancer_manager, "_balancers") else 0
            },
        }
        
        return jsonify({
            "available": True,
            "components": components
        })
        
    except Exception as e:
        return jsonify({
            "available": True,
            "error": str(e)
        })

# Importar la aplicación Flask para Gunicorn
app = flask_app

# Código para correr la aplicación Flask en desarrollo
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
