import os
import logging
import sys
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta

# Logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar db desde extensions (ya no desde SQLAlchemy directo)
from extensions import db

# Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("SESSION_SECRET", "genesis_secret_key_development")
CORS(app)

# Config DB
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Inicializar DB
db.init_app(app)

# Importar modelos (después de inicializar db y app)
try:
    from models import User, Investor, Transaction, Loan, Bonus, Commission
    logger.info("Modelos importados correctamente.")
except ImportError as e:
    logger.error(f"Error al importar modelos: {e}")

# ... (todas las rutas y lógica igual que ya tienes) ...
# Puedes dejar todo lo demás como ya está.

# Inicializar la app y la base de datos
with app.app_context():
    db.create_all()

    # Crear usuario creador si no existe
    creator = User.query.filter_by(username='mixycronico').first()
    if not creator:
        creator = User(
            username='mixycronico',
            email='mixycronico@aol.com',
            password_hash=generate_password_hash('Jesus@2@7'),
            first_name='Moises',
            last_name='Alvarenga',
            is_admin=True
        )
        db.session.add(creator)
        db.session.add(Investor(
            user=creator,
            balance=1000000.0,
            capital=900000.0,
            earnings=100000.0,
            risk_level='high',
            category='platinum'
        ))
        try:
            db.session.commit()
            logger.info("Usuario creador creado correctamente.")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error al crear usuario: {e}")

    # Inicializar familia cósmica
    try:
        from cosmic_family import initialize_cosmic_family
        initialize_cosmic_family()
    except Exception as e:
        logger.error(f"Error al inicializar familia cósmica: {e}")

# Registrar rutas de comisiones
try:
    from commission_routes import register_commission_routes
    register_commission_routes(app)
except Exception as e:
    logger.error(f"Error registrando rutas de comisiones: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
