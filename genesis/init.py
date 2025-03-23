"""
Módulo de inicialización para el Sistema Genesis.

Este módulo proporciona las funciones necesarias para inicializar y configurar
todos los componentes del sistema, incluyendo la base de datos, clasificador,
gestor de riesgo y seguidor de rendimiento.
"""
import json
import logging
import os
import asyncio
from typing import Dict, Any, Optional

# Configuración de logging central
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('genesis.init')

# Importar componentes principales
from genesis.db.base import db_manager
from genesis.db.transcendental_database import transcendental_db
from genesis.analysis.transcendental_crypto_classifier import classifier, initialize_classifier
from genesis.risk.adaptive_risk_manager import risk_manager, initialize_risk_manager
from genesis.analytics.transcendental_performance_tracker import performance_tracker, initialize_performance_tracker

async def initialize_system(config_path: str = "genesis_config.json") -> Dict[str, Any]:
    """
    Inicializar todos los componentes del sistema Genesis.
    
    Args:
        config_path: Ruta al archivo de configuración JSON
        
    Returns:
        Diccionario con estado de inicialización
    """
    logger.info(f"Iniciando sistema Genesis con configuración: {config_path}")
    
    # Cargar configuración
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error al cargar configuración: {str(e)}")
        config = {
            "system": {"name": "Genesis", "version": "1.0.0", "environment": "development"},
            "classifier": {"capital_inicial": 10000.0},
            "risk_manager": {"capital_inicial": 10000.0},
            "performance_tracker": {"capital_inicial": 10000.0}
        }
    
    # Estado de inicialización
    init_status = {
        "database": False,
        "classifier": False,
        "risk_manager": False,
        "performance_tracker": False,
        "mensaje": ""
    }
    
    try:
        # Inicializar base de datos
        await initialize_database(config.get("database", {}))
        init_status["database"] = True
        
        # Inicializar clasificador
        await initialize_classifier(
            config.get("classifier", {}).get("capital_inicial", 10000.0),
            config.get("classifier", {})
        )
        init_status["classifier"] = True
        
        # Inicializar gestor de riesgo
        await initialize_risk_manager(
            config.get("risk_manager", {}).get("capital_inicial", 10000.0),
            config.get("risk_manager", {})
        )
        init_status["risk_manager"] = True
        
        # Inicializar seguidor de rendimiento
        try:
            await initialize_performance_tracker(
                config.get("performance_tracker", {}).get("capital_inicial", 10000.0),
                config.get("performance_tracker", {})
            )
            init_status["performance_tracker"] = True
        except Exception as e:
            logger.warning(f"Fallo en la inicialización del performance tracker: {str(e)}")
            # Continuar con la inicialización aunque falle este componente
        
        # Estado exitoso
        init_status["mensaje"] = "Sistema Genesis inicializado correctamente"
        logger.info("Sistema Genesis inicializado correctamente")
        
    except Exception as e:
        init_status["mensaje"] = f"Error durante la inicialización: {str(e)}"
        logger.error(f"Error durante la inicialización: {str(e)}")
    
    return init_status

async def initialize_database(config: Dict[str, Any] = None) -> None:
    """
    Inicializar la base de datos y sus componentes.
    
    Args:
        config: Configuración para la base de datos
    """
    if config is None:
        config = {}
    
    # Configurar pool de conexiones
    pool_size = config.get("pool_size", 20)
    max_overflow = config.get("max_overflow", 40)
    pool_recycle = config.get("pool_recycle", 300)
    
    # Inicializar DatabaseManager
    db_manager.pool_size = pool_size
    db_manager.max_overflow = max_overflow
    db_manager.pool_recycle = pool_recycle
    
    # Configurar motor de base de datos
    db_manager.setup()
    
    # Crear tablas si es necesario
    await db_manager.create_all_tables()
    
    logger.info("Base de datos inicializada")

async def get_system_status() -> Dict[str, Any]:
    """
    Obtener estado actual de todos los componentes del sistema.
    
    Returns:
        Diccionario con estado de cada componente
    """
    # Recopilar estado de todos los componentes
    try:
        db_status = await db_manager.get_connection_stats()
    except Exception as e:
        db_status = {"error": str(e)}
    
    try:
        cache_status = transcendental_db.get_stats()
    except Exception as e:
        cache_status = {"error": str(e)}
    
    # Obtener información del clasificador
    try:
        classifier_hot_cryptos = await classifier.get_hot_cryptos()
        classifier_info = {
            "capital_actual": classifier.current_capital,
            "cryptos_hot": len(classifier_hot_cryptos),
            "ultima_clasificacion": classifier.last_classification_time.isoformat()
        }
    except Exception as e:
        classifier_info = {"error": str(e)}
    
    # Obtener información del gestor de riesgo
    try:
        risk_stats = await risk_manager.obtener_estadisticas()
        risk_info = {
            "capital_actual": risk_manager.capital_actual,
            "nivel_proteccion": risk_manager.nivel_proteccion,
            "modo_trascendental": risk_manager.modo_trascendental
        }
    except Exception as e:
        risk_info = {"error": str(e)}
    
    # Obtener información del seguidor de rendimiento
    try:
        performance_info = await performance_tracker.obtener_resumen_rendimiento()
        performance_status = {
            "capital_actual": performance_tracker.capital_actual,
            "rendimiento_total": performance_tracker.metricas.get("rendimiento_total", 0),
            **performance_info
        }
    except Exception as e:
        logger.warning(f"Error al obtener información del performance tracker: {str(e)}")
        performance_status = {"error": str(e)}
    
    # Compilar estado completo
    status = {
        "timestamp": str(asyncio.get_event_loop().time()),
        "database": db_status,
        "cache": cache_status,
        "classifier": classifier_info,
        "risk_manager": risk_info,
        "performance_tracker": performance_status
    }
    
    return status

async def actualizar_capital_sistema(nuevo_capital: float) -> Dict[str, Any]:
    """
    Actualizar el capital en todos los componentes del sistema.
    
    Esta función mantiene sincronizado el capital entre el clasificador,
    gestor de riesgo y seguidor de rendimiento.
    
    Args:
        nuevo_capital: Nuevo monto de capital
        
    Returns:
        Diccionario con resultados de la actualización
    """
    resultados = {}
    
    if nuevo_capital <= 0:
        logger.warning(f"Capital inválido en actualizar_capital_sistema: {nuevo_capital}")
        resultados["error"] = "El capital debe ser un valor positivo"
        return resultados
    
    capital_anterior = 0
    
    # Actualizar capital en clasificador
    try:
        resultado_clasificador = await classifier.update_capital(nuevo_capital)
        resultados["classifier"] = resultado_clasificador
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Error al actualizar capital en classifier: {error_msg}")
        resultados["classifier"] = {"error": error_msg}
    
    # Actualizar capital en gestor de riesgo
    try:
        resultado_risk = await risk_manager.actualizar_capital(nuevo_capital)
        resultados["risk_manager"] = resultado_risk
        
        # Obtener capital anterior del risk manager si está disponible
        if isinstance(resultado_risk, dict) and "capital_anterior" in resultado_risk:
            capital_anterior = resultado_risk["capital_anterior"]
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Error al actualizar capital en risk_manager: {error_msg}")
        resultados["risk_manager"] = {"error": error_msg}
    
    # Actualizar capital en seguidor de rendimiento
    try:
        resultado_performance = await performance_tracker.actualizar_capital(
            nuevo_capital, "actualizacion_sistema"
        )
        resultados["performance_tracker"] = resultado_performance
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Error al actualizar capital en performance_tracker: {error_msg}")
        resultados["performance_tracker"] = {"error": error_msg}
    
    # Resultado global
    resultados["capital_anterior"] = capital_anterior  # Usando el valor capturado
    resultados["capital_nuevo"] = nuevo_capital
    
    # Calcular cambio porcentual con verificación de división por cero
    if capital_anterior > 0:
        resultados["cambio_porcentual"] = ((nuevo_capital / capital_anterior) - 1) * 100
    else:
        resultados["cambio_porcentual"] = 0
        logger.warning("No se pudo calcular cambio porcentual: capital_anterior es cero o negativo")
    
    return resultados

async def activar_modo_trascendental(modo: str = "SINGULARITY_V4") -> Dict[str, Any]:
    """
    Activar un modo trascendental específico en todos los componentes.
    
    Args:
        modo: Modo trascendental a activar 
              ("SINGULARITY_V4", "LIGHT", "DARK_MATTER", "DIVINE", "BIG_BANG", "INTERDIMENSIONAL")
        
    Returns:
        Diccionario con resultados de la activación
    """
    resultados = {}
    
    # Validar modo trascendental
    modos_validos = ["SINGULARITY_V4", "LIGHT", "DARK_MATTER", "DIVINE", "BIG_BANG", "INTERDIMENSIONAL"]
    if modo not in modos_validos:
        logger.warning(f"Modo trascendental inválido: {modo}. Usando SINGULARITY_V4 por defecto.")
        modo = "SINGULARITY_V4"
    
    logger.info(f"Activando modo trascendental: {modo}")
    
    # Activar en gestor de riesgo
    try:
        resultado_risk = await risk_manager.activar_modo_trascendental(modo)
        resultados["risk_manager"] = resultado_risk
        logger.info(f"Modo {modo} activado en risk_manager")
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Error al activar modo {modo} en risk_manager: {error_msg}")
        resultados["risk_manager"] = {"error": error_msg}
    
    # Activar en seguidor de rendimiento
    try:
        resultado_performance = await performance_tracker.activar_modo_trascendental(modo)
        resultados["performance_tracker"] = resultado_performance
        logger.info(f"Modo {modo} activado en performance_tracker")
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Error al activar modo {modo} en performance_tracker: {error_msg}")
        resultados["performance_tracker"] = {"error": error_msg}
    
    # Resultado global
    resultados["modo"] = modo
    resultados["timestamp"] = asyncio.get_event_loop().time()
    resultados["status"] = "success"
    
    # Verificar si hubo errores en algún componente
    if any("error" in resultado for componente, resultado in resultados.items() 
           if componente not in ["modo", "timestamp", "status"]):
        resultados["status"] = "partial_success"
        logger.warning(f"Activación del modo {modo} completada parcialmente con errores")
    else:
        logger.info(f"Activación del modo {modo} completada exitosamente en todos los componentes")
    
    return resultados