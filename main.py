"""
Main entry point for the Genesis trading system.

Este módulo inicializa y arranca el sistema, configurando todos los componentes
y proporcionando el punto de entrada principal para la operación.
También expone la aplicación Flask para Gunicorn, incluyendo la API REST
para integración con componentes cloud externos.

El sistema incluye:
- Orquestador Seraphim con motor de comportamiento humano Gabriel
- Sistema de escalabilidad de capital adaptativo
- Principio "todos ganamos o todos perdemos" en todas las operaciones
- Integración con APIs externas
"""

import os
import sys
import logging
import asyncio
from flask import Flask, jsonify, request
# Cambiar temporalmente a la aplicación web en lugar de la API
# from app import app as flask_app
from website.app import app as flask_app

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

# Intentar importar componentes Seraphim
try:
    from genesis.init.seraphim_initializer import (
        initialize_seraphim_strategy,
        get_seraphim_strategy,
        integrate_with_genesis
    )
    HAS_SERAPHIM = True
except ImportError:
    logger.warning("Componentes Seraphim no disponibles")
    HAS_SERAPHIM = False

# Función para inicializar Seraphim
async def init_seraphim_components():
    """Inicializar componentes Seraphim."""
    if not HAS_SERAPHIM:
        return False
    
    try:
        logger.info("Inicializando estrategia Seraphim...")
        
        # Importar componentes necesarios
        from genesis.accounting.capital_scaling import CapitalScalingManager
        
        # Obtener gestor de capital si está disponible
        capital_manager = None
        try:
            from genesis.init.scaling_initializer import get_scaling_manager
            capital_manager = get_scaling_manager()
        except ImportError:
            logger.warning("Gestor de capital no disponible, se creará uno nuevo")
        
        # Integrar con Genesis
        success = await integrate_with_genesis(
            scaling_manager=capital_manager,
            capital_base=10000.0,
            symbols=["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT"]
        )
        
        if success:
            logger.info("Estrategia Seraphim integrada correctamente")
        else:
            logger.error("No se pudo integrar la estrategia Seraphim")
        
        return success
        
    except Exception as e:
        logger.exception(f"Error al inicializar componentes Seraphim: {e}")
        return False

# Función sincrónica para inicialización
def init_components():
    """Inicializar todos los componentes."""
    # Inicializar componentes de forma asincrónica
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Inicializar componentes cloud si están disponibles
        if HAS_CLOUD_COMPONENTS:
            loop.run_until_complete(init_cloud_components())
        
        # Inicializar componentes Seraphim si están disponibles
        if HAS_SERAPHIM:
            loop.run_until_complete(init_seraphim_components())
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

# Crear endpoint para estado de Seraphim
@flask_app.route("/api/seraphim-status")
def seraphim_status():
    """Verificar estado de la estrategia Seraphim."""
    if not HAS_SERAPHIM:
        return jsonify({
            "available": False,
            "message": "Estrategia Seraphim no disponible"
        })
    
    try:
        # Obtener estrategia
        from genesis.init.seraphim_initializer import get_seraphim_strategy
        strategy = get_seraphim_strategy()
        
        if not strategy:
            return jsonify({
                "available": True,
                "initialized": False,
                "message": "Estrategia Seraphim no inicializada"
            })
        
        # Obtener estado asíncrono
        def get_status():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(strategy.get_system_status())
            finally:
                loop.close()
        
        status = get_status()
        
        return jsonify({
            "available": True,
            "initialized": True,
            "status": status
        })
        
    except Exception as e:
        return jsonify({
            "available": True,
            "error": str(e)
        })

# Crear endpoint para capital de Seraphim
@flask_app.route("/api/seraphim-portfolio")
def seraphim_portfolio():
    """Obtener estado del portafolio de Seraphim."""
    if not HAS_SERAPHIM:
        return jsonify({
            "available": False,
            "message": "Estrategia Seraphim no disponible"
        })
    
    try:
        # Obtener estrategia
        from genesis.init.seraphim_initializer import get_seraphim_strategy
        strategy = get_seraphim_strategy()
        
        if not strategy:
            return jsonify({
                "available": True,
                "initialized": False,
                "message": "Estrategia Seraphim no inicializada"
            })
        
        # Obtener portafolio asíncrono
        def get_portfolio():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(strategy.get_portfolio_status())
            finally:
                loop.close()
        
        portfolio = get_portfolio()
        
        return jsonify({
            "available": True,
            "initialized": True,
            "portfolio": portfolio
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
