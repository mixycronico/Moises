"""
Punto de entrada principal para la aplicación web del Sistema Genesis.

Este módulo proporciona la integración entre la aplicación web y
los componentes backend del Sistema Genesis.
"""

import os
import sys
import logging
from flask import Flask

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('genesis_website')

# Agregar proto_genesis al path de Python
sys.path.append(os.path.abspath('proto_genesis'))

# Importar app de proto_genesis
try:
    from proto_genesis.app import app
except ImportError:
    # Si falla, crear una app Flask básica
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return "Sistema Genesis en mantenimiento. Por favor, intente más tarde."

# Punto de entrada para Gunicorn
# No eliminar - esta línea es requerida para que el servidor funcione correctamente

if __name__ == "__main__":
    logger.info("Iniciando servidor web del Sistema Genesis...")
    app.run(host="0.0.0.0", port=5000, debug=True)