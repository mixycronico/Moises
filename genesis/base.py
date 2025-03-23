"""
Módulo base para el Sistema Genesis.

Este módulo proporciona las funcionalidades y estructuras base
que son usadas por todos los componentes del sistema.
"""
import logging
import os
import json
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, TypeVar

# Configuración de logging central
logger = logging.getLogger("genesis.base")

# Constantes globales
DEFAULT_CONFIG_PATH = "genesis_config.json"
CONFIG_SCHEMA_VERSION = "2.0.0"

# Modos transcendentales disponibles
TRASCENDENTAL_MODES = [
    "SINGULARITY_V4",  # Modo por defecto, capacidad extrema
    "LIGHT",           # Modo Luz
    "DARK_MATTER",     # Modo Materia Oscura
    "DIVINE",          # Modo Divino
    "BIG_BANG",        # Modo Big Bang (regeneración)
    "INTERDIMENSIONAL" # Modo Interdimensional (múltiples dimensiones)
]

# Indicadores de rendimiento
PERFORMANCE_INDICATORS = [
    "rendimiento_total",       # Rendimiento acumulado
    "rendimiento_anualizado",  # Rendimiento anualizado
    "sharpe_ratio",            # Sharpe ratio (exceso retorno / volatilidad)
    "max_drawdown",            # Máxima caída desde máximo
    "win_rate",                # Tasa de aciertos
    "expectancy",              # Expectativa matemática
    "recovery_factor"          # Factor de recuperación
]

def get_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Obtener configuración del sistema desde archivo JSON.
    
    Args:
        path: Ruta al archivo de configuración (opcional)
        
    Returns:
        Diccionario con configuración
    """
    config_path = path or DEFAULT_CONFIG_PATH
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
    
    # Configuración por defecto
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "trascendental_mode": "SINGULARITY_V4",
        "capital_inicial": 10000.0,
        "max_symbols": 5,
        "database_config": {
            "pool_size": 20,
            "max_overflow": 40,
            "pool_recycle": 300
        },
        "scaling_config": {
            "initial_capital": 10000.0,
            "min_efficiency": 0.5,
            "default_model_type": "polynomial"
        }
    }

def save_config(config: Dict[str, Any], path: Optional[str] = None) -> bool:
    """
    Guardar configuración en archivo JSON.
    
    Args:
        config: Configuración a guardar
        path: Ruta al archivo de configuración (opcional)
        
    Returns:
        True si se guardó correctamente
    """
    config_path = path or DEFAULT_CONFIG_PATH
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error al guardar configuración: {str(e)}")
        return False

def validate_mode(mode: str) -> str:
    """
    Validar modo trascendental.
    
    Args:
        mode: Modo a validar
        
    Returns:
        Modo validado o modo por defecto si no es válido
    """
    if mode in TRASCENDENTAL_MODES:
        return mode
    
    logger.warning(f"Modo trascendental '{mode}' no válido, usando SINGULARITY_V4")
    return "SINGULARITY_V4"

def format_currency(amount: float) -> str:
    """
    Formatear valor como moneda.
    
    Args:
        amount: Cantidad a formatear
        
    Returns:
        Cadena formateada
    """
    return f"${amount:,.2f}"

class GenesisSingleton:
    """
    Base para implementar singletons en el sistema.
    
    Esta clase permite implementar el patrón singleton para
    componentes que deben tener una única instancia en todo el sistema.
    """
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(GenesisSingleton, cls).__new__(cls)
        return cls._instances[cls]

class GenesisComponent:
    """
    Base para componentes del sistema Genesis.
    
    Esta clase proporciona funcionalidades comunes para todos los
    componentes del sistema, como gestión de estado y telemetría.
    """
    def __init__(self, component_id: str, mode: str = "SINGULARITY_V4"):
        """
        Inicializar componente.
        
        Args:
            component_id: Identificador único del componente
            mode: Modo trascendental
        """
        self.component_id = component_id
        self.mode = validate_mode(mode)
        self.creation_time = time.time()
        self.last_update = self.creation_time
        self.operation_count = 0
        self.error_count = 0
        self.metrics: Dict[str, float] = {}
        
        logger.debug(f"Componente {component_id} inicializado en modo {mode}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del componente.
        
        Returns:
            Diccionario con estadísticas
        """
        uptime = time.time() - self.creation_time
        
        return {
            "component_id": self.component_id,
            "mode": self.mode,
            "uptime": uptime,
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.operation_count),
            "metrics": self.metrics
        }
    
    def register_operation(self, success: bool = True) -> None:
        """
        Registrar operación.
        
        Args:
            success: Si la operación fue exitosa
        """
        self.operation_count += 1
        if not success:
            self.error_count += 1
        self.last_update = time.time()
    
    def update_metric(self, key: str, value: float) -> None:
        """
        Actualizar métrica.
        
        Args:
            key: Nombre de la métrica
            value: Valor
        """
        self.metrics[key] = value
        self.last_update = time.time()

# Configuración de logging por defecto
def setup_logging(level: int = logging.INFO) -> None:
    """
    Configurar logging para el sistema.
    
    Args:
        level: Nivel de logging
    """
    root_logger = logging.getLogger("genesis")
    
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    
    root_logger.setLevel(level)
    logger.debug("Logging configurado con nivel %s", 
                logging.getLevelName(level))