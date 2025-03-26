"""
Punto de entrada principal para la aplicación web del Sistema Genesis.

Este módulo inicia el servidor Flask con todos los componentes y rutas necesarias.
"""
from website.app import app

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)