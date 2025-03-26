"""
Punto de entrada principal para la aplicación Cosmic Genesis.

Este módulo inicia la aplicación Flask para la familia cósmica (Aetherion y Lunareth).
"""
import sys
import os

# Asegurarse de que el directorio actual esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cosmic_genesis.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)