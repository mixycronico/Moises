"""
Main entry point for the Genesis trading system.

Este módulo inicializa y arranca el sistema, configurando todos los componentes
y proporcionando el punto de entrada principal para la operación.
También expone la aplicación Flask para Gunicorn.
"""

# Versión simplificada - solo exponer la aplicación para Gunicorn
# Importar la aplicación Flask para Gunicorn
from app import app


# Código para correr la aplicación Flask en desarrollo
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
