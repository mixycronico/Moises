"""
Utilidades de configuración para el sistema Genesis.

Este módulo proporciona funciones para acceder a configuraciones
de diferentes componentes del sistema.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

# Configuraciones por defecto
DEFAULT_CONFIGS = {
    'scaling': {
        'capital_base': 10000.0,
        'efficiency_threshold': 0.85,
        'max_symbols_small': 5,
        'max_symbols_large': 15,
        'volatility_adjustment': 1.0,
        'correlation_limit': 0.7,
        'capital_protection_level': 0.95,
        'db_persistence_enabled': True,
        'monitoring_enabled': True,
        'redis_cache_enabled': True,
        'max_capital': 1000000.0
    },
    'database': {
        'pool_size': 20,
        'max_overflow': 40,
        'pool_timeout': 60,
        'pool_recycle': 3600,
        'echo': False
    },
    'risk': {
        'max_drawdown': 0.2,
        'daily_var': 0.02,
        'position_size_limit': 0.25,
        'correlation_threshold': 0.7,
        'risk_free_rate': 0.02
    }
}

# Ruta al archivo de configuración
CONFIG_FILE = os.environ.get('GENESIS_CONFIG_FILE', 'genesis_config.json')

# Caché de configuraciones cargadas
_config_cache: Dict[str, Dict[str, Any]] = {}

def get_config(section: str) -> Dict[str, Any]:
    """
    Obtener la configuración para una sección específica.
    
    Carga la configuración desde el archivo si está disponible,
    o utiliza los valores por defecto.
    
    Args:
        section: Nombre de la sección (scaling, database, risk, etc.)
        
    Returns:
        Diccionario con la configuración
    """
    # Si ya está en caché, devolverla
    if section in _config_cache:
        return _config_cache[section].copy()
    
    # Intentar cargar el archivo de configuración
    loaded_config = {}
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                
            logging.info(f"Configuración cargada desde {CONFIG_FILE}")
    except Exception as e:
        logging.warning(f"Error cargando archivo de configuración: {str(e)}")
        
    # Obtener sección específica del archivo (si existe)
    section_config = loaded_config.get(section, {})
    
    # Combinar con valores por defecto
    default_section = DEFAULT_CONFIGS.get(section, {})
    final_config = {**default_section, **section_config}
    
    # Almacenar en caché
    _config_cache[section] = final_config
    
    return final_config.copy()

def save_config(section: str, config: Dict[str, Any]) -> bool:
    """
    Guardar una configuración específica.
    
    Args:
        section: Nombre de la sección
        config: Configuración a guardar
        
    Returns:
        True si se guardó correctamente
    """
    # Intentar cargar configuración existente
    full_config = {}
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                full_config = json.load(f)
    except Exception:
        pass
    
    # Actualizar sección
    full_config[section] = config
    
    # Actualizar caché
    _config_cache[section] = config.copy()
    
    # Guardar archivo
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(full_config, f, indent=4)
        
        logging.info(f"Configuración guardada en {CONFIG_FILE}")
        return True
    except Exception as e:
        logging.error(f"Error guardando configuración: {str(e)}")
        return False

def update_config(section: str, updates: Dict[str, Any]) -> bool:
    """
    Actualizar parcialmente una configuración.
    
    Args:
        section: Nombre de la sección
        updates: Actualizaciones a aplicar
        
    Returns:
        True si se actualizó correctamente
    """
    current = get_config(section)
    current.update(updates)
    return save_config(section, current)

def get_scaling_factor(capital: float) -> float:
    """
    Calcular factor de escala basado en el capital.
    
    El factor de escala se utiliza para ajustar diversos parámetros
    en función del capital disponible.
    
    Args:
        capital: Capital disponible
        
    Returns:
        Factor de escala (1.0 para capital base)
    """
    config = get_config('scaling')
    capital_base = config.get('capital_base', 10000.0)
    
    if capital <= capital_base:
        return 1.0
    
    # Escala logarítmica suavizada
    import math
    return 1.0 + math.log(capital / capital_base, 10) * 0.5