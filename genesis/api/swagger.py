"""
Configuración de Swagger UI para la API REST.

Este módulo proporciona funcionalidades para exponer la documentación 
de la API REST utilizando Swagger UI.
"""

import os
from flask import Blueprint, render_template, send_from_directory


def init_swagger(app):
    """
    Inicializar documentación Swagger.
    
    Args:
        app: Aplicación Flask
    """
    # Crear Blueprint para Swagger
    swagger_bp = Blueprint(
        'swagger_ui',
        __name__,
        url_prefix='/api/docs',
        template_folder='templates'
    )
    
    # Ruta para la interfaz Swagger
    @swagger_bp.route('/', methods=['GET'])
    def swagger_ui():
        """Interfaz Swagger UI."""
        # Redirect para la API de FastAPI que ya incluye Swagger UI
        return render_template(
            'swagger.html',
            api_url='/openapi.json'
        )
    
    # Registrar el blueprint
    app.register_blueprint(swagger_bp)
    
    # Asegurar que existe directorio de templates
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    
    # Crear template básico de Swagger UI si no existe
    swagger_template_path = os.path.join(os.path.dirname(__file__), 'templates', 'swagger.html')
    if not os.path.exists(swagger_template_path):
        with open(swagger_template_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genesis API - Swagger UI</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css">
    <style>
        body { margin: 0; }
        .swagger-ui .topbar { background-color: #2C3E50; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: "{{ api_url }}",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ],
                layout: "BaseLayout",
                supportedSubmitMethods: ["get", "post", "put", "delete", "patch"],
            });
            window.ui = ui;
        };
    </script>
</body>
</html>
            """)
    
    return swagger_bp