"""
Módulo de configuración para la integración de DeepSeek en el Sistema Genesis.

Este módulo proporciona funciones para gestionar la configuración y el estado
de la integración con DeepSeek, incluyendo activación/desactivación,
configuración de parámetros y gestión de API keys.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# Constantes
CONFIG_FILE = "deepseek_config.json"
DEFAULT_CONFIG = {
    "enabled": True,
    "model_version": "deepseek-coder-33b-instruct",
    "intelligence_factor": 1.0,
    "max_tokens": 4096,
    "temperature": 0.7,
    "cache_enabled": True,
    "cache_ttl": 3600,  # 1 hora
    "last_updated": datetime.now().isoformat()
}

# Variable global para el estado (se puede modificar desde otros módulos)
_deepseek_state = {
    "enabled": True,
    "initialized": False,
    "api_key_available": bool(os.environ.get("DEEPSEEK_API_KEY")),
    "config": DEFAULT_CONFIG.copy(),
    "stats": {
        "requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "last_request_time": None
    }
}

def load_config() -> Dict[str, Any]:
    """
    Cargar configuración de DeepSeek desde archivo.
    
    Returns:
        Configuración actual
    """
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                logger.info(f"Configuración de DeepSeek cargada desde {CONFIG_FILE}")
                
                # Asegurar que siempre tenga todas las claves necesarias
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                
                _deepseek_state["config"] = config
                return config
        else:
            logger.info(f"Archivo de configuración {CONFIG_FILE} no encontrado, usando valores por defecto")
            return save_config(_deepseek_state["config"])
    except Exception as e:
        logger.error(f"Error al cargar configuración de DeepSeek: {str(e)}")
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Guardar configuración de DeepSeek en archivo.
    
    Args:
        config: Configuración a guardar (opcional, usa la actual si no se proporciona)
        
    Returns:
        Configuración guardada
    """
    if config is None:
        config = _deepseek_state["config"]
    
    try:
        # Actualizar timestamp
        config["last_updated"] = datetime.now().isoformat()
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
            logger.info(f"Configuración de DeepSeek guardada en {CONFIG_FILE}")
        
        _deepseek_state["config"] = config
        return config
    except Exception as e:
        logger.error(f"Error al guardar configuración de DeepSeek: {str(e)}")
        return config

def is_enabled() -> bool:
    """
    Verificar si DeepSeek está habilitado.
    
    Returns:
        True si DeepSeek está habilitado, False en caso contrario
    """
    return _deepseek_state["enabled"] and _deepseek_state["api_key_available"]

def enable() -> bool:
    """
    Activar DeepSeek.
    
    Returns:
        True si se activó correctamente, False en caso contrario
    """
    _deepseek_state["enabled"] = True
    _deepseek_state["config"]["enabled"] = True
    save_config()
    logger.info("DeepSeek activado")
    return True

def disable() -> bool:
    """
    Desactivar DeepSeek.
    
    Returns:
        True si se desactivó correctamente, False en caso contrario
    """
    _deepseek_state["enabled"] = False
    _deepseek_state["config"]["enabled"] = False
    save_config()
    logger.info("DeepSeek desactivado")
    return True

def toggle() -> bool:
    """
    Alternar estado (activar/desactivar) de DeepSeek.
    
    Returns:
        Nuevo estado (True=activado, False=desactivado)
    """
    new_state = not _deepseek_state["enabled"]
    if new_state:
        enable()
    else:
        disable()
    return new_state

def set_intelligence_factor(factor: float) -> float:
    """
    Establecer factor de inteligencia para DeepSeek.
    
    Args:
        factor: Factor de inteligencia (entre 0.1 y 10.0)
        
    Returns:
        Valor actualizado
    """
    factor = max(0.1, min(factor, 10.0))  # Limitar entre 0.1 y 10.0
    _deepseek_state["config"]["intelligence_factor"] = factor
    save_config()
    logger.info(f"Factor de inteligencia de DeepSeek establecido a {factor}")
    return factor

def get_config() -> Dict[str, Any]:
    """
    Obtener configuración actual.
    
    Returns:
        Configuración actual
    """
    return _deepseek_state["config"]

def update_config(new_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Actualizar configuración.
    
    Args:
        new_config: Nueva configuración parcial
        
    Returns:
        Configuración actualizada
    """
    # Actualizar solo las claves proporcionadas
    for key, value in new_config.items():
        if key in _deepseek_state["config"]:
            _deepseek_state["config"][key] = value
    
    # Si se actualiza enabled, actualizar también el estado global
    if "enabled" in new_config:
        _deepseek_state["enabled"] = new_config["enabled"]
    
    save_config()
    logger.info(f"Configuración de DeepSeek actualizada: {new_config.keys()}")
    return _deepseek_state["config"]

def get_state() -> Dict[str, Any]:
    """
    Obtener estado completo de DeepSeek.
    
    Returns:
        Estado actual
    """
    # Actualizar estado del API key por si ha cambiado
    _deepseek_state["api_key_available"] = bool(os.environ.get("DEEPSEEK_API_KEY"))
    
    return {
        "enabled": _deepseek_state["enabled"],
        "initialized": _deepseek_state["initialized"],
        "api_key_available": _deepseek_state["api_key_available"],
        "config": _deepseek_state["config"],
        "stats": _deepseek_state["stats"]
    }

def update_stats(
    requests: int = 0,
    successful_requests: int = 0,
    failed_requests: int = 0
) -> Dict[str, Any]:
    """
    Actualizar estadísticas de uso.
    
    Args:
        requests: Número de solicitudes a incrementar
        successful_requests: Número de solicitudes exitosas a incrementar
        failed_requests: Número de solicitudes fallidas a incrementar
        
    Returns:
        Estadísticas actualizadas
    """
    _deepseek_state["stats"]["requests"] += requests
    _deepseek_state["stats"]["successful_requests"] += successful_requests
    _deepseek_state["stats"]["failed_requests"] += failed_requests
    
    if requests > 0:
        _deepseek_state["stats"]["last_request_time"] = datetime.now().isoformat()
    
    return _deepseek_state["stats"]

def set_initialized(state: bool = True) -> None:
    """
    Establecer estado de inicialización.
    
    Args:
        state: Nuevo estado de inicialización
    """
    _deepseek_state["initialized"] = state
    logger.info(f"Estado de inicialización de DeepSeek establecido a {state}")

# Inicializar al importar
load_config()
logger.info(f"Módulo de configuración DeepSeek inicializado. Estado actual: {'Activado' if is_enabled() else 'Desactivado'}")