import os
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.security import generate_password_hash, check_password_hash

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Clase base para modelos SQLAlchemy
class Base(DeclarativeBase):
    pass

# Inicializar SQLAlchemy
db = SQLAlchemy(model_class=Base)

# Crear la aplicación Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("SESSION_SECRET", "genesis_secret_key_development")

# Configurar la conexión a la base de datos
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Inicializar la base de datos
db.init_app(app)

# Importar los modelos y las rutas después de inicializar db para evitar problemas de importación circular
with app.app_context():
    import models
    from routes import register_routes
    from auth_routes import register_auth_routes
    from aetherion_routes import register_aetherion_routes
    
    # Registrar todas las rutas
    register_routes(app)
    register_auth_routes(app)
    register_aetherion_routes(app)
    
    # Crear tablas si no existen
    db.create_all()
    logger.info("Base de datos inicializada correctamente")

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Manejador de errores 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Manejador de errores 500
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)