#!/usr/bin/env python3
"""
Script para construir la aplicación React.
Este script ejecuta los comandos necesarios para construir la aplicación React
y prepararla para ser servida por Flask.
"""

import os
import subprocess
import sys
import shutil
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, cwd=None):
    """Ejecutar un comando y mostrar su salida."""
    logger.info(f"Ejecutando: {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=cwd
        )
        
        for line in process.stdout:
            print(line.strip())
        
        for line in process.stderr:
            print(line.strip(), file=sys.stderr)
            
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"El comando falló con código de salida {process.returncode}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error al ejecutar el comando: {e}")
        return False

def build_client():
    """Construir la aplicación cliente React."""
    client_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'client')
    
    logger.info("Instalando dependencias del cliente...")
    if not run_command("npm install", cwd=client_dir):
        return False
    
    logger.info("Construyendo la aplicación React...")
    if not run_command("npm run build", cwd=client_dir):
        return False
    
    logger.info("La aplicación React ha sido construida con éxito.")
    return True

def main():
    """Función principal."""
    if build_client():
        logger.info("Construcción completada exitosamente.")
        logger.info("La aplicación está lista para ser servida por Flask.")
        return 0
    else:
        logger.error("La construcción ha fallado.")
        return 1

if __name__ == "__main__":
    sys.exit(main())