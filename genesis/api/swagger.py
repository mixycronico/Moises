"""
Configuración de Swagger UI para la API REST del sistema Genesis.

Este módulo proporciona funcionalidades para generar documentación
OpenAPI/Swagger automáticamente para la API REST.
"""

import logging
from flask import Flask, Blueprint, jsonify, render_template_string

def init_swagger(app: Flask) -> None:
    """
    Inicializar la documentación Swagger para la API.
    
    Args:
        app: Aplicación Flask
    """
    # Crear blueprint para la documentación
    swagger_blueprint = Blueprint('swagger', __name__, url_prefix='/api/docs')
    
    # OpenAPI spec básica
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Genesis Trading System API",
            "description": "API para el sistema de trading algorítmico Genesis",
            "version": "1.0.0",
            "contact": {
                "name": "Genesis Team"
            }
        },
        "servers": [
            {
                "url": "/api/v1",
                "description": "API Genesis v1"
            }
        ],
        "paths": {
            "/ping": {
                "get": {
                    "summary": "Verificar estado de la API",
                    "description": "Endpoint para verificar que la API está funcionando correctamente",
                    "responses": {
                        "200": {
                            "description": "API funcionando correctamente",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {
                                                "type": "string",
                                                "example": "success"
                                            },
                                            "message": {
                                                "type": "string",
                                                "example": "Genesis API está funcionando correctamente"
                                            },
                                            "version": {
                                                "type": "string",
                                                "example": "1.0.0"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    # Ruta para acceder a la especificación OpenAPI
    @swagger_blueprint.route('/spec', methods=['GET'])
    def get_spec():
        return jsonify(openapi_spec)
    
    # Ruta para la interfaz Swagger UI
    swagger_ui_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Genesis API - Swagger UI</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3/swagger-ui.css">
        <style>
            html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
            *, *:before, *:after { box-sizing: inherit; }
            body { margin: 0; background: #fafafa; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@3/swagger-ui-bundle.js"></script>
        <script>
            window.onload = function() {
                const ui = SwaggerUIBundle({
                    url: "/api/docs/spec",
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIBundle.SwaggerUIStandalonePreset
                    ],
                    layout: "BaseLayout",
                    withCredentials: true
                });
                window.ui = ui;
            };
        </script>
    </body>
    </html>
    """
    
    @swagger_blueprint.route('/', methods=['GET'])
    def swagger_ui():
        return render_template_string(swagger_ui_html)
    
    logging.getLogger('api').info('Swagger UI inicializado en /api/docs')
    
    # Registrar blueprint
    app.register_blueprint(swagger_blueprint)
    
    return swagger_blueprint