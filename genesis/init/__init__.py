"""
Inicializador del sistema Genesis.

Este módulo coordina la inicialización de todos los componentes del sistema Genesis,
incluyendo base de datos, motor de eventos, análisis, trading, etc.
"""

import logging
import asyncio
import json
import os
import time
from typing import Dict, List, Any, Tuple, Optional

from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.init.init_scaling import initialize_scaling_system
from genesis.analysis.transcendental_crypto_classifier import initialize_classifier
from genesis.risk.adaptive_risk_manager import initialize_risk_manager
from genesis.analytics.transcendental_performance_tracker import initialize_performance_tracker
from genesis.strategies.orchestrator import StrategyOrchestrator
from genesis.strategies.adaptive_scaling_strategy import AdaptiveScalingStrategy

# Configurar logging
logger = logging.getLogger("genesis.init")

async def initialize_system(config_path: str) -> Dict[str, Any]:
    """
    Inicializar todo el sistema Genesis.
    
    Esta función coordina la inicialización de todos los componentes
    usando la configuración del archivo especificado.
    
    Args:
        config_path: Ruta al archivo de configuración JSON
        
    Returns:
        Dict con resultados de inicialización con formato {"componente": True/False, "mensaje": "..."}
    """
    try:
        logger.info("Iniciando inicialización del sistema Genesis")
        
        # Cargar configuración del archivo
        if not os.path.exists(config_path):
            logger.error(f"Archivo de configuración no encontrado: {config_path}")
            return {"mensaje": f"Archivo de configuración no encontrado: {config_path}"}
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
            return {"mensaje": f"Error al cargar configuración: {e}"}
        
        # Extraer valores principales de configuración 
        initial_capital = config.get("scaling_config", {}).get("initial_capital", 10000.0)
        logger.info(f"Capital inicial configurado: {initial_capital}")
        
        results = {"mensaje": ""}
        
        # 1. Inicializar componentes base
        # 1.1 Inicializar clasificador con capital numérico
        classifier_config = config.get("classifier_config", {})
        crypto_classifier = await initialize_classifier(
            initial_capital=float(initial_capital), 
            config=classifier_config or {}
        )
        
        if crypto_classifier:
            results["crypto_classifier"] = crypto_classifier
            results["crypto_classifier_status"] = True
            logger.info("Clasificador de criptomonedas inicializado")
        else:
            results["crypto_classifier_status"] = False
            logger.warning("Fallo en la inicialización del clasificador de criptomonedas")
        
        # 1.2 Inicializar risk manager con capital numérico
        risk_config = config.get("risk_config", {})
        risk_manager = await initialize_risk_manager(
            capital_inicial=float(initial_capital), 
            config=risk_config or {}
        )
        
        if risk_manager:
            results["risk_manager"] = risk_manager
            results["risk_manager_status"] = True
            logger.info("Risk manager inicializado")
        else:
            results["risk_manager_status"] = False
            logger.warning("Fallo en la inicialización del risk manager")
        
        # 1.3 Inicializar performance tracker con capital numérico
        performance_config = config.get("performance_config", {})
        performance_tracker = await initialize_performance_tracker(
            capital_inicial=float(initial_capital), 
            config=performance_config or {}
        )
        
        if performance_tracker:
            results["performance_tracker"] = performance_tracker
            results["performance_tracker_status"] = True
            logger.info("Performance tracker inicializado")
        else:
            results["performance_tracker_status"] = False
            logger.warning("Fallo en la inicialización del performance tracker")
        
        # 2. Inicializar sistema de escalabilidad adaptativa
        scaling_config = config.get("scaling_config", {})
        try:
            scaling_success, scaling_components = await initialize_scaling_system(
                db=None,  # La base de datos se maneja internamente 
                config=scaling_config
            )
            
            if scaling_success and isinstance(scaling_components, dict):
                # Guardar estado de éxito
                results["scaling_system_status"] = True
                
                # Guardar componentes especiales para enlazar
                for key, component in scaling_components.items():
                    component_key = f"{key}_status"
                    results[component_key] = True
                    results[key] = component
                
                logger.info("Sistema de escalabilidad adaptativa inicializado")
            else:
                results["scaling_system_status"] = False
                logger.warning("Fallo en la inicialización del sistema de escalabilidad adaptativa")
        except Exception as e:
            results["scaling_system_status"] = False
            logger.warning(f"Error en la inicialización del sistema de escalabilidad: {e}")
        
        # 3. Inicializar orquestador de estrategias
        orchestrator_config = config.get("orchestrator_config", {})
        try:
            # Inicializar orquestrador
            orchestrator = StrategyOrchestrator()
            await orchestrator.start()
            
            results["orchestrator"] = orchestrator
            results["orchestrator_status"] = True
            logger.info("Orquestador de estrategias inicializado")
            
            # Configurar orquestador según la configuración
            if orchestrator_config:
                # Establecer threshold de rendimiento
                min_threshold = orchestrator_config.get("min_performance_threshold")
                if min_threshold is not None:
                    orchestrator.min_performance_threshold = float(min_threshold)
                
                # Establecer cooldown
                eval_cooldown = orchestrator_config.get("eval_cooldown")
                if eval_cooldown is not None:
                    orchestrator.eval_cooldown = int(eval_cooldown)
                
                # Establecer max_failures
                max_failures = orchestrator_config.get("max_eval_failures")
                if max_failures is not None:
                    orchestrator.max_eval_failures = int(max_failures)
                
                # Registrar estrategias
                strategies_config = orchestrator_config.get("strategies", {})
                
                # Registrar estrategia adaptativa si existe en los componentes
                if "adaptive_strategy" in results and "adaptive_scaling" in strategies_config:
                    adaptive_strategy = results["adaptive_strategy"]
                    strategy_config = strategies_config["adaptive_scaling"]
                    
                    if strategy_config.get("enabled", True):
                        orchestrator.register_strategy(
                            "adaptive_scaling", 
                            adaptive_strategy, 
                            strategy_config.get("config", {})
                        )
                        logger.info("Estrategia adaptativa registrada en el orquestador")
                
                # Establecer estrategia por defecto
                default_strategy = orchestrator_config.get("default_strategy")
                if default_strategy and default_strategy in orchestrator.strategies:
                    orchestrator.force_change_strategy(default_strategy)
                    logger.info(f"Estrategia por defecto establecida: {default_strategy}")
            
        except Exception as e:
            results["orchestrator_status"] = False
            logger.warning(f"Error en la inicialización del orquestador de estrategias: {e}")
            
        # 4. Enlazar componentes entre sí si es posible
        try:
            # 4.1 Si existe la estrategia adaptativa, enlazarla con risk manager
            if risk_manager and "adaptive_strategy" in results:
                adaptive_strategy = results.get("adaptive_strategy")
                if hasattr(adaptive_strategy, "risk_manager"):
                    adaptive_strategy.risk_manager = risk_manager
                    logger.info("Estrategia adaptativa enlazada con risk manager")
            
            # 4.2 Si existe el scaling_manager, enlazarlo con performance_tracker
            if performance_tracker and "scaling_manager" in results:
                scaling_manager = results.get("scaling_manager")
                if hasattr(scaling_manager, "register_metrics_provider"):
                    scaling_manager.register_metrics_provider(performance_tracker)
                    logger.info("Scaling manager enlazado con performance tracker")
                    
            # 4.3 Si existe el orquestador y performance_tracker, registrar como proveedor de métricas
            if "orchestrator" in results and performance_tracker:
                orchestrator = results.get("orchestrator")
                # Registrar evento de métricas de rendimiento
                await orchestrator.emit_event("strategy.performance_update", {
                    "strategy_name": "adaptive_scaling",
                    "score": 0.85,  # Valor inicial de rendimiento
                    "timestamp": time.time()
                })
                logger.info("Performance inicial registrado en orquestador")
                
            # 4.4 Si existe la estrategia adaptativa y el scaling_manager, enlazarlos
            if "adaptive_strategy" in results and "scaling_manager" in results:
                adaptive_strategy = results.get("adaptive_strategy")
                scaling_manager = results.get("scaling_manager")
                if hasattr(adaptive_strategy, "scaling_manager"):
                    adaptive_strategy.scaling_manager = scaling_manager
                    logger.info("Estrategia adaptativa enlazada con scaling manager")
                
        except Exception as e:
            logger.warning(f"Error al enlazar componentes: {e}")
        
        # Verificar si la inicialización fue exitosa (al menos algunos componentes iniciados)
        status_keys = [k for k in results if k.endswith("_status")]
        success_count = sum(1 for k in status_keys if results[k] is True)
        
        if success_count > 0:
            results["mensaje"] = "Sistema Genesis inicializado correctamente"
            logger.info(f"Sistema Genesis inicializado correctamente con {success_count} componentes")
        else:
            results["mensaje"] = "Fallo en la inicialización del sistema Genesis"
            logger.error("Fallo en la inicialización del sistema Genesis")
        
        return results
    
    except Exception as e:
        error_msg = f"Error en la inicialización del sistema Genesis: {str(e)}"
        logger.error(error_msg)
        return {"mensaje": error_msg}