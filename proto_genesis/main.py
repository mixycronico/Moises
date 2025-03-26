"""
Punto de entrada principal para la aplicación web de Proto Genesis.
"""

from app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)