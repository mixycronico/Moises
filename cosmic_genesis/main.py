"""
Punto de entrada principal para la aplicación Cosmic Genesis.

Este módulo inicia la aplicación Flask para la familia cósmica (Aetherion y Lunareth).
"""

from cosmic_genesis.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)