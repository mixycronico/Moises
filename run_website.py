"""
Script para ejecutar la interfaz web del Sistema Genesis.
Ejecuta la aplicación en el puerto 5001 para evitar conflictos.
"""

import subprocess
import sys
import os

def main():
    """Ejecuta la aplicación web de Genesis."""
    # Establecer directorio de trabajo
    os.chdir("website")
    
    # Ejecutar aplicación Flask
    subprocess.run([sys.executable, "app.py"], check=True)

if __name__ == "__main__":
    main()