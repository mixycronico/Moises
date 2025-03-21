"""
Inicialización de la API REST para el sistema Genesis.

Este módulo proporciona funciones para inicializar y configurar
la API REST del sistema Genesis en una aplicación Flask.
"""

import logging
from flask import Flask, Blueprint, jsonify
from genesis.api.logger import initialize_logging

def init_api(app: Flask) -> None:
    """
    Inicializar la API REST.
    
    Args:
        app: Aplicación Flask
    """
    # Configurar logging
    initialize_logging(app)
    
    # Crear blueprint para la API
    api_blueprint = Blueprint('api', __name__, url_prefix='/api/v1')
    
    # Ruta de verificación para la API
    @api_blueprint.route('/ping', methods=['GET'])
    def ping():
        return jsonify({
            'status': 'success',
            'message': 'Genesis API está funcionando correctamente',
            'version': '1.0.0'
        })
    
    # Registrar blueprint
    app.register_blueprint(api_blueprint)
    
    logging.getLogger('api').info('API REST inicializada correctamente')
    
    return api_blueprint