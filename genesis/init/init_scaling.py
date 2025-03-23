"""
Inicializador del sistema de escalabilidad adaptativa para Genesis.

Este módulo proporciona las funciones necesarias para inicializar
y enlazar el sistema de escalabilidad adaptativa con el resto del
sistema Genesis.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.init.scaling_initializer import ScalingInitializer
from genesis.accounting.predictive_scaling import PredictiveScalingEngine
from genesis.accounting.balance_manager import CapitalScalingManager
from genesis.strategies.adaptive_scaling_strategy import AdaptiveScalingStrategy

# Configurar logging
logger = logging.getLogger("genesis.init.init_scaling")

async def initialize_scaling_system(
    db: Optional[TranscendentalDatabase] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Inicializar todo el sistema de escalabilidad adaptativa.
    
    Esta función realiza:
    1. Configuración de la base de datos
    2. Inicialización de tablas
    3. Creación de componentes necesarios
    4. Enlace con el sistema principal de Genesis
    
    Args:
        db: Conexión a la base de datos transcendental
        config: Configuración adicional
        
    Returns:
        Tupla (éxito, componentes)
    """
    try:
        logger.info("Iniciando inicialización del sistema de escalabilidad adaptativa")
        
        # 1. Crear inicializador de escalabilidad
        initializer = ScalingInitializer(db, config)
        
        # 2. Inicializar sistema completo
        success, components = await initializer.initialize()
        
        if not success:
            logger.error("Fallo en la inicialización del sistema de escalabilidad")
            return False, {}
        
        # 3. Extraer componentes principales
        engine = components.get("engine")
        scaling_manager = components.get("scaling_manager")
        adaptive_strategy = components.get("adaptive_strategy")
        
        # 4. Verificar que todo esté inicializado
        if not engine or not scaling_manager or not adaptive_strategy:
            logger.error("Componentes faltantes en la inicialización")
            return False, {}
        
        logger.info("Sistema de escalabilidad adaptativa inicializado correctamente")
        return True, components
        
    except Exception as e:
        logger.error(f"Error inicializando sistema de escalabilidad: {str(e)}")
        return False, {}

async def load_scaling_system(
    db: Optional[TranscendentalDatabase] = None,
    components: Dict[str, Any] = None,
    symbols: List[str] = None,
    initial_capital: float = 10000.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Cargar y configurar el sistema de escalabilidad adaptativa.
    
    Esta función configura los componentes existentes o crea nuevos
    si es necesario. Útil cuando solo necesitamos cargar sin inicializar
    toda la base de datos.
    
    Args:
        db: Conexión a la base de datos transcendental
        components: Componentes existentes para reutilizar
        symbols: Lista de símbolos a operar
        initial_capital: Capital inicial
        
    Returns:
        Tupla (éxito, componentes)
    """
    try:
        logger.info("Cargando sistema de escalabilidad adaptativa")
        
        components = components or {}
        symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
        
        # Reutilizar componentes o crear nuevos
        engine = components.get("engine")
        scaling_manager = components.get("scaling_manager")
        adaptive_strategy = components.get("adaptive_strategy")
        
        # 1. Crear motor predictivo si no existe
        if not engine:
            engine = PredictiveScalingEngine(
                config={
                    "default_model_type": "polynomial",
                    "cache_ttl": 300,
                    "auto_train": True,
                    "confidence_threshold": 0.6
                }
            )
        
        # 2. Crear scaling manager si no existe
        if not scaling_manager:
            scaling_manager = CapitalScalingManager(
                engine=engine,
                initial_capital=initial_capital,
                min_efficiency=0.5
            )
            
            # Vincular base de datos
            if hasattr(scaling_manager, 'db'):
                scaling_manager.db = db
        
        # 3. Crear estrategia adaptativa si no existe
        if not adaptive_strategy:
            adaptive_strategy = AdaptiveScalingStrategy(
                name="Estrategia de Escalabilidad Adaptativa",
                symbols=symbols,
                config={"initial_capital": initial_capital},
                db=db
            )
            
            # Vincular componentes
            if hasattr(adaptive_strategy, 'engine'):
                adaptive_strategy.engine = engine
            
            if hasattr(adaptive_strategy, 'scaling_manager'):
                adaptive_strategy.scaling_manager = scaling_manager
            
            # Inicializar
            await adaptive_strategy.initialize()
        
        logger.info("Sistema de escalabilidad adaptativa cargado correctamente")
        
        return True, {
            "engine": engine,
            "scaling_manager": scaling_manager,
            "adaptive_strategy": adaptive_strategy
        }
        
    except Exception as e:
        logger.error(f"Error cargando sistema de escalabilidad: {str(e)}")
        return False, {}

async def optimize_capital_allocation(
    engine: Optional[PredictiveScalingEngine] = None,
    symbols: List[str] = None,
    total_capital: float = 10000.0,
    min_efficiency: float = 0.5
) -> Dict[str, float]:
    """
    Optimizar asignación de capital utilizando el motor predictivo.
    
    Esta función permite realizar una optimización directa sin necesidad
    de inicializar todo el sistema.
    
    Args:
        engine: Motor predictivo (opcional, se creará uno temporal si no se proporciona)
        symbols: Lista de símbolos a considerar
        total_capital: Capital total a asignar
        min_efficiency: Eficiencia mínima aceptable (0-1)
        
    Returns:
        Diccionario con asignaciones por símbolo
    """
    logger.info(f"Optimizando asignación para ${total_capital:,.2f} entre {len(symbols or [])} símbolos")
    
    # Validar parámetros
    if not symbols or total_capital <= 0:
        logger.warning("Parámetros inválidos para optimización")
        return {}
    
    # Crear motor temporal si no se proporciona
    local_engine = None
    if not engine:
        logger.info("Creando motor predictivo temporal")
        local_engine = PredictiveScalingEngine()
        engine = local_engine
    
    try:
        # Ejecutar optimización
        allocations = await engine.optimize_allocation(
            symbols=symbols,
            total_capital=total_capital,
            min_efficiency=min_efficiency
        )
        
        # Registrar resultados
        symbols_used = len([a for a in allocations.values() if a > 0])
        logger.info(f"Optimización completada: {symbols_used}/{len(symbols)} símbolos utilizados")
        
        return allocations
        
    except Exception as e:
        logger.error(f"Error en optimización: {str(e)}")
        return {}
    
    finally:
        # Limpiar si creamos un motor temporal
        if local_engine:
            logger.debug("Limpiando motor predictivo temporal")
            # No hay método explícito de limpieza, confiar en Python
            del local_engine

# Versión simplificada para integraciones rápidas
async def quick_optimize(
    symbols: List[str],
    total_capital: float
) -> Dict[str, float]:
    """
    Versión simplificada de optimización para integraciones rápidas.
    
    Args:
        symbols: Lista de símbolos
        total_capital: Capital total
        
    Returns:
        Asignaciones optimizadas
    """
    return await optimize_capital_allocation(
        engine=None,
        symbols=symbols,
        total_capital=total_capital,
        min_efficiency=0.4  # Umbral más permisivo
    )