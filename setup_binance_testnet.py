#!/usr/bin/env python3
"""
Script de configuración para Binance Testnet.

Este script guía al usuario para configurar sus claves API de Binance Testnet,
almacenándolas como variables de entorno para su uso con el Sistema Genesis.

Instrucciones para obtener claves de la Testnet de Binance:
1. Crear una cuenta en Binance (si aún no tienes una)
2. Visitar https://testnet.binance.vision/
3. Iniciar sesión con tu cuenta de GitHub o de Binance
4. Generar nueva API key y Secret key
5. Copiar estas claves al ser solicitadas por este script
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, Optional, Tuple
import getpass

# Archivo de configuración por defecto
DEFAULT_CONFIG_FILE = os.path.expanduser("~/.genesis_binance_testnet")

def read_existing_config(config_file: str) -> Dict[str, str]:
    """
    Leer configuración existente si está disponible.
    
    Args:
        config_file: Ruta al archivo de configuración
        
    Returns:
        Dict con configuración existente, o dict vacío si no existe
    """
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error al leer configuración existente: {e}")
    
    return {}

def write_config(config: Dict[str, str], config_file: str) -> bool:
    """
    Escribir configuración a archivo.
    
    Args:
        config: Dict con configuración a escribir
        config_file: Ruta al archivo de configuración
        
    Returns:
        True si se escribió correctamente
    """
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
        
        # Escribir configuración
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        # Establecer permisos restrictivos (solo lectura/escritura para el usuario)
        os.chmod(config_file, 0o600)
        
        return True
        
    except Exception as e:
        print(f"Error al escribir configuración: {e}")
        return False

def prompt_for_keys() -> Tuple[str, str]:
    """
    Solicitar claves API al usuario.
    
    Returns:
        Tupla (api_key, api_secret)
    """
    print("\n=== Configuración de Binance Testnet ===")
    print("Necesitas obtener tus claves API de Binance Testnet.")
    print("Instrucciones:")
    print("1. Visita https://testnet.binance.vision/")
    print("2. Inicia sesión con tu cuenta de GitHub")
    print("3. Genera nueva API key y Secret key")
    print("4. Copia estas claves a continuación\n")
    
    api_key = input("API Key: ").strip()
    api_secret = getpass.getpass("API Secret: ").strip()
    
    return api_key, api_secret

def setup_environment_vars(config: Dict[str, str]) -> None:
    """
    Configurar variables de entorno a partir de la configuración.
    
    Args:
        config: Dict con configuración
    """
    # Establecer variables de entorno
    os.environ["BINANCE_TESTNET_API_KEY"] = config.get("api_key", "")
    os.environ["BINANCE_TESTNET_API_SECRET"] = config.get("api_secret", "")
    
    print("\nVariables de entorno establecidas para la sesión actual.")

def create_activation_script(config: Dict[str, str], script_file: str) -> bool:
    """
    Crear script de activación para cargar variables de entorno.
    
    Args:
        config: Dict con configuración
        script_file: Ruta al script de activación
        
    Returns:
        True si se creó correctamente
    """
    try:
        script_content = f"""#!/bin/sh
# Script de activación para variables de entorno de Binance Testnet
# Generado por {os.path.basename(__file__)}

# Exportar variables de entorno
export BINANCE_TESTNET_API_KEY="{config.get('api_key', '')}"
export BINANCE_TESTNET_API_SECRET="{config.get('api_secret', '')}"

echo "Variables de entorno de Binance Testnet activadas."
"""
        
        # Escribir script
        with open(script_file, "w") as f:
            f.write(script_content)
        
        # Establecer permisos de ejecución
        os.chmod(script_file, 0o700)
        
        return True
        
    except Exception as e:
        print(f"Error al crear script de activación: {e}")
        return False

def main():
    """Función principal."""
    # Crear parser de argumentos
    parser = argparse.ArgumentParser(description="Configuración de Binance Testnet")
    parser.add_argument(
        "--config-file", type=str, default=DEFAULT_CONFIG_FILE,
        help=f"Archivo de configuración (default: {DEFAULT_CONFIG_FILE})"
    )
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Leer configuración existente
    config = read_existing_config(args.config_file)
    
    # Verificar si ya hay configuración
    has_existing_config = bool(config.get("api_key")) and bool(config.get("api_secret"))
    
    if has_existing_config:
        print(f"Se encontró configuración existente en {args.config_file}")
        choice = input("¿Deseas reconfigurar? (s/N): ").strip().lower()
        
        if choice != "s":
            print("Usando configuración existente.")
            setup_environment_vars(config)
            return
    
    # Solicitar claves API
    api_key, api_secret = prompt_for_keys()
    
    # Verificar que se ingresaron ambas claves
    if not api_key or not api_secret:
        print("Error: Debes ingresar ambas claves.")
        return
    
    # Actualizar configuración
    config["api_key"] = api_key
    config["api_secret"] = api_secret
    
    # Escribir configuración
    if write_config(config, args.config_file):
        print(f"\nConfiguración guardada en {args.config_file}")
    
    # Configurar variables de entorno
    setup_environment_vars(config)
    
    # Crear script de activación
    activation_script = os.path.expanduser("~/activate_binance_testnet.sh")
    if create_activation_script(config, activation_script):
        print(f"\nScript de activación creado en {activation_script}")
        print("Para cargar las variables en una nueva sesión, ejecuta:")
        print(f"  source {activation_script}")
    
    # Mostrar instrucciones para probar la conexión
    print("\nPara probar la conexión a Binance Testnet, ejecuta:")
    print("  python run_binance_testnet_demo.py")

if __name__ == "__main__":
    main()