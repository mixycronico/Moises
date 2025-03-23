"""
Inicializador del sistema Genesis.

Este módulo coordina la inicialización de todos los componentes del sistema Genesis,
incluyendo base de datos, motor de eventos, análisis, trading, etc.
"""

import logging
import asyncio
from typing import Dict, List, Any, Tuple, Optional

from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.init.init_scaling import initialize_scaling_system
from genesis.analysis.transcendental_crypto_classifier import initialize_classifier
from genesis.risk.adaptive_risk_manager import initialize_risk_manager
from genesis.analytics.transcendental_performance_tracker import initialize_performance_tracker

# Configurar logging
logger = logging.getLogger("genesis.init")

async def initialize_system(
    db: Optional[TranscendentalDatabase] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Inicializar todo el sistema Genesis.
    
    Esta función coordina la inicialización de todos los componentes.
    
    Args:
        db: Conexión a la base de datos transcendental (opcional)
        config: Configuración adicional
        
    Returns:
        Tupla (éxito, componentes)
    """
    try:
        logger.info("Iniciando inicialización del sistema Genesis")
        
        config = config or {}
        components = {}
        
        # 1. Inicializar componentes base
        # 1.1 Inicializar clasificador
        crypto_classifier = await initialize_classifier(db, config.get("classifier_config"))
        if crypto_classifier:
            components["crypto_classifier"] = crypto_classifier
            logger.info("Clasificador de criptomonedas inicializado")
        else:
            logger.warning("Fallo en la inicialización del clasificador de criptomonedas")
        
        # 1.2 Inicializar risk manager
        risk_manager = await initialize_risk_manager(db, config.get("risk_config"))
        if risk_manager:
            components["risk_manager"] = risk_manager
            logger.info("Risk manager inicializado")
        else:
            logger.warning("Fallo en la inicialización del risk manager")
        
        # 1.3 Inicializar performance tracker
        performance_tracker = await initialize_performance_tracker(db, config.get("performance_config"))
        if performance_tracker:
            components["performance_tracker"] = performance_tracker
            logger.info("Performance tracker inicializado")
        else:
            logger.warning("Fallo en la inicialización del performance tracker")
        
        # 2. Inicializar sistema de escalabilidad adaptativa
        scaling_success, scaling_components = await initialize_scaling_system(db, config.get("scaling_config"))
        if scaling_success:
            components.update(scaling_components)
            logger.info("Sistema de escalabilidad adaptativa inicializado")
        else:
            logger.warning("Fallo en la inicialización del sistema de escalabilidad adaptativa")
        
        # 3. Enlazar componentes entre sí
        # 3.1 Si existe la estrategia adaptativa, enlazarla con risk manager
        if "adaptive_strategy" in components and "risk_manager" in components:
            components["adaptive_strategy"].risk_manager = components["risk_manager"]
            logger.info("Estrategia adaptativa enlazada con risk manager")
        
        # 3.2 Si existe el scaling_manager, enlazarlo con performance_tracker
        if "scaling_manager" in components and "performance_tracker" in components:
            # Proporcionar acceso al performance tracker para métricas 
            if hasattr(components["scaling_manager"], "register_metrics_provider"):
                components["scaling_manager"].register_metrics_provider(
                    components["performance_tracker"]
                )
                logger.info("Scaling manager enlazado con performance tracker")
        
        # Verificar si la inicialización fue exitosa (al menos algunos componentes iniciados)
        success = len(components) > 0
        
        if success:
            logger.info("Sistema Genesis inicializado correctamente")
        else:
            logger.error("Fallo en la inicialización del sistema Genesis")
        
        return success, components
    
    except Exception as e:
        logger.error(f"Error en la inicialización del sistema Genesis: {str(e)}")
        return False, {}