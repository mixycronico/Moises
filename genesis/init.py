"""
Inicialización del Sistema Genesis con integración de estrategia.

Este módulo inicializa todos los componentes del Sistema Genesis,
incluyendo la base de datos, el clasificador de criptomonedas,
el gestor de riesgo adaptativo y el rastreador de rendimiento.

Asegura la carga correcta de todos los módulos y su correcta interconexión.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
import os
import json

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("genesis")

# Importar componentes del sistema
from genesis.db.transcendental_database import db, initialize_database
from genesis.analysis.transcendental_crypto_classifier import crypto_classifier, initialize_classifier
from genesis.risk.adaptive_risk_manager import risk_manager, initialize_risk_manager
from genesis.analytics.transcendental_performance_tracker import performance_tracker, initialize_performance_tracker

async def initialize_genesis_system(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Inicializar todos los componentes del Sistema Genesis.
    
    Args:
        config_path: Ruta opcional a archivo de configuración
        
    Returns:
        Diccionario con resultados de inicialización
    """
    logger.info("Iniciando Sistema Genesis con integración de estrategia...")
    
    # Cargar configuración si se proporciona
    config = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuración cargada desde {config_path}")
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
    
    # Inicializar componentes en orden correcto
    results = {}
    
    # 1. Base de datos (primero siempre)
    try:
        logger.info("Inicializando base de datos transcendental...")
        db_result = await initialize_database()
        results["database"] = {
            "status": "success" if db_result else "error",
            "message": "Base de datos inicializada correctamente" if db_result else "Error al inicializar base de datos"
        }
    except Exception as e:
        logger.error(f"Error crítico al inicializar base de datos: {e}")
        results["database"] = {
            "status": "error",
            "message": f"Error crítico: {str(e)}"
        }
        return results  # Terminar si falla la base de datos
    
    # 2. Clasificador de criptomonedas
    try:
        logger.info("Inicializando clasificador transcendental...")
        # Configurar con parámetros si están disponibles
        if "classifier" in config:
            if "capital_inicial" in config["classifier"]:
                crypto_classifier.capital_inicial = config["classifier"]["capital_inicial"]
                crypto_classifier.capital_actual = config["classifier"]["capital_inicial"]
            if "exchanges" in config["classifier"]:
                crypto_classifier.exchanges = config["classifier"]["exchanges"]
        
        classifier_result = await initialize_classifier()
        results["classifier"] = {
            "status": "success",
            "message": "Clasificador inicializado correctamente",
            "capital_inicial": crypto_classifier.capital_inicial
        }
    except Exception as e:
        logger.error(f"Error al inicializar clasificador: {e}")
        results["classifier"] = {
            "status": "error",
            "message": f"Error: {str(e)}"
        }
    
    # 3. Gestor de riesgo
    try:
        logger.info("Inicializando gestor de riesgo adaptativo...")
        # Configurar con parámetros si están disponibles
        if "risk_manager" in config:
            if "capital_inicial" in config["risk_manager"]:
                risk_manager.capital_inicial = config["risk_manager"]["capital_inicial"]
                risk_manager.capital_actual = config["risk_manager"]["capital_inicial"]
            if "max_drawdown_permitido" in config["risk_manager"]:
                risk_manager.max_drawdown_permitido = config["risk_manager"]["max_drawdown_permitido"]
            if "volatilidad_base" in config["risk_manager"]:
                risk_manager.volatilidad_base = config["risk_manager"]["volatilidad_base"]
        
        risk_result = await initialize_risk_manager()
        results["risk_manager"] = {
            "status": "success",
            "message": "Gestor de riesgo inicializado correctamente",
            "capital_inicial": risk_manager.capital_inicial
        }
    except Exception as e:
        logger.error(f"Error al inicializar gestor de riesgo: {e}")
        results["risk_manager"] = {
            "status": "error",
            "message": f"Error: {str(e)}"
        }
    
    # 4. Rastreador de rendimiento
    try:
        logger.info("Inicializando rastreador de rendimiento...")
        # Configurar con parámetros si están disponibles
        if "performance_tracker" in config:
            if "capital_inicial" in config["performance_tracker"]:
                performance_tracker.capital_inicial = config["performance_tracker"]["capital_inicial"]
                performance_tracker.metricas["capital_actual"] = config["performance_tracker"]["capital_inicial"]
        
        perf_result = await initialize_performance_tracker()
        results["performance_tracker"] = {
            "status": "success",
            "message": "Rastreador de rendimiento inicializado correctamente",
            "capital_inicial": performance_tracker.capital_inicial
        }
    except Exception as e:
        logger.error(f"Error al inicializar rastreador de rendimiento: {e}")
        results["performance_tracker"] = {
            "status": "error",
            "message": f"Error: {str(e)}"
        }
    
    # Verificar si todos los componentes se iniciaron correctamente
    all_success = all(results[component]["status"] == "success" for component in results)
    
    if all_success:
        logger.info("Sistema Genesis inicializado completamente con éxito")
        results["overall_status"] = "success"
    else:
        failed_components = [c for c in results if results[c]["status"] == "error"]
        logger.warning(f"Sistema Genesis inicializado parcialmente. Componentes fallidos: {failed_components}")
        results["overall_status"] = "partial"
        results["failed_components"] = failed_components
    
    return results

# Función para ejecución directa
async def main():
    """Función principal para inicialización desde línea de comandos."""
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    results = await initialize_genesis_system(config_path)
    
    if results["overall_status"] == "success":
        logger.info("Sistema Genesis listo para operar")
        
        # Ejecutar una simulación de clasificación para verificar funcionalidad
        if "--test" in sys.argv:
            logger.info("Ejecutando simulación de clasificación...")
            try:
                resultado = await crypto_classifier.simular_clasificacion_completa(n_cryptos=10)
                logger.info(f"Simulación exitosa. Hot cryptos: {[c['symbol'] for c in resultado['hot_cryptos']]}")
            except Exception as e:
                logger.error(f"Error en simulación: {e}")
    else:
        logger.error("Error al inicializar Sistema Genesis. Revisando logs para detalles.")
    
    return results

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())