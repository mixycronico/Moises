import os
import logging
import sys
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy.orm import DeclarativeBase
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta

# Configuración de logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Añadir el directorio actual al path para importaciones
sys.path.append('.')

# Clase base para modelos SQLAlchemy
class Base(DeclarativeBase):
    pass

# Inicializar SQLAlchemy
db = SQLAlchemy(model_class=Base)

# Crear la aplicación Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("SESSION_SECRET", "genesis_secret_key_development")
CORS(app)  # Habilitar CORS para todas las rutas

# Configurar la conexión a la base de datos
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Inicializar la base de datos
db.init_app(app)

# Importar los modelos - aseguramos que la importación funcione
try:
    from models import User, Investor, Transaction, Loan, Bonus, Commission
    logger.info("Importación de modelos exitosa")
except ImportError as e:
    logger.error(f"Error al importar modelos: {str(e)}")
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from models import User, Investor, Transaction, Loan, Bonus, Commission
    logger.info("Importación de modelos exitosa después de ajuste de path")

# Verificar si el usuario está autenticado
def is_authenticated():
    return 'user_id' in session

# Obtener el rol del usuario actual
def get_user_role():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            return user.role
    return None

# Verificar si el usuario es creador (mixycronico)
def is_creator():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.username == 'mixycronico':
            return True
    return False

# Rutas principales
@app.route('/')
def index():
    # Página de inicio - accesible para todos
    return render_template('index.html', 
                           authenticated=is_authenticated(),
                           user_role=get_user_role(),
                           is_creator=is_creator())

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        logger.info(f"Intento de inicio de sesión para usuario: {username}")
        
        # Buscar usuario sin importar mayúsculas/minúsculas
        try:
            user = User.query.filter(User.username.ilike(username)).first()
            
            if user:
                # Verificar contraseña directamente contra el hash almacenado
                password_matches = check_password_hash(user.password_hash, password)
                logger.info(f"Verificación de contraseña para {username}: {'exitosa' if password_matches else 'fallida'}")
                
                if password_matches:
                    session['user_id'] = user.id
                    session['username'] = user.username
                    logger.info(f"Inicio de sesión exitoso para: {username}")
                    
                    # Actualizar hora de último login
                    user.last_login = datetime.utcnow()
                    db.session.commit()
                    
                    return redirect(url_for('dashboard'))
            
            error = 'Credenciales inválidas. Por favor, intente de nuevo.'
            logger.warning(f"Inicio de sesión fallido para: {username}")
        except Exception as e:
            error = 'Error al procesar el inicio de sesión. Por favor, intente de nuevo.'
            logger.error(f"Error en login: {str(e)}")
    
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Verificar si el usuario ya existe (insensible a mayúsculas/minúsculas)
        existing_user = User.query.filter((User.username.ilike(username)) | (User.email.ilike(email))).first()
        if existing_user:
            error = 'El nombre de usuario o email ya está en uso.'
        else:
            # Crear nuevo usuario
            new_user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password),
                role='inversionista'  # Rol por defecto
            )
            db.session.add(new_user)
            
            # Crear perfil de inversionista
            new_investor = Investor(
                user=new_user,
                balance=0.0,
                capital=0.0,
                earnings=0.0,
                risk_level='moderate',
                category='bronze'
            )
            db.session.add(new_investor)
            
            try:
                db.session.commit()
                session['user_id'] = new_user.id
                session['username'] = new_user.username
                return redirect(url_for('dashboard'))
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error al registrar usuario: {str(e)}")
                error = 'Error al registrar usuario. Por favor, intente de nuevo.'
    
    return render_template('register.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if not is_authenticated():
        return redirect(url_for('login'))
    
    user_role = get_user_role()
    user = User.query.get(session['user_id'])
    
    if user_role == 'inversionista':
        # Obtener datos del inversionista
        investor = Investor.query.filter_by(user_id=user.id).first()
        
        if not investor:
            flash('No se encontró perfil de inversionista', 'error')
            return redirect(url_for('index'))
        
        return render_template('investor_dashboard.html', 
                              user=user,
                              investor=investor,
                              balance=investor.balance,
                              earnings=investor.earnings,
                              capital=investor.capital,
                              authenticated=True,
                              user_role=user_role,
                              is_creator=is_creator())
    elif user_role == 'admin':
        return render_template('admin_dashboard.html', 
                              user=user,
                              authenticated=True,
                              user_role=user_role,
                              is_creator=is_creator())
    elif user_role == 'super_admin':
        return render_template('super_admin_dashboard.html', 
                              user=user,
                              authenticated=True,
                              user_role=user_role,
                              is_creator=is_creator())
    else:
        return redirect(url_for('index'))

@app.route('/cosmic_chat')
def cosmic_chat():
    if not is_authenticated():
        return redirect(url_for('login'))
    
    return render_template('cosmic_chat.html', 
                          user=User.query.get(session['user_id']),
                          is_creator=is_creator())

# API para interactuar con Aetherion y Lunareth
@app.route('/api/cosmic_chat', methods=['POST'])
def api_cosmic_chat():
    if not is_authenticated():
        return jsonify({"error": "No autenticado"}), 401
    
    data = request.json
    message = data.get('message', '')
    
    # Importar Aetherion y Lunareth desde el módulo cosmic_family
    from cosmic_family import get_cosmic_family
    cosmic_family = get_cosmic_family()
    
    # Procesar el mensaje con ambas entidades
    aetherion_response = cosmic_family.aetherion.process_conversational_stimulus(
        message, 
        user_id=session.get('username')
    )
    
    lunareth_response = cosmic_family.lunareth.process_conversational_stimulus(
        message, 
        user_id=session.get('username')
    )
    
    return jsonify({
        "aetherion": aetherion_response,
        "lunareth": lunareth_response
    })

# Rutas para préstamos
@app.route('/loans')
def loans():
    if not is_authenticated():
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    investor = Investor.query.filter_by(user_id=user.id).first()
    
    if not investor:
        flash('No se encontró perfil de inversionista', 'error')
        return redirect(url_for('dashboard'))
    
    # Verificar elegibilidad para préstamos
    # Un inversionista puede solicitar préstamos después de 3 meses
    is_eligible = False
    eligibility_message = ''
    
    # Verificar antigüedad (3 meses mínimo)
    if investor.created_at and investor.created_at < datetime.utcnow() - timedelta(days=90):
        is_eligible = True
        eligibility_message = 'Elegible para préstamos (antigüedad > 3 meses)'
    else:
        days_remaining = 90
        if investor.created_at:
            days_passed = (datetime.utcnow() - investor.created_at).days
            days_remaining = max(0, 90 - days_passed)
        eligibility_message = f'No elegible para préstamos. Días restantes: {days_remaining}'
    
    # Calcular monto máximo de préstamo (40% del capital)
    max_loan_amount = 0
    if is_eligible:
        max_loan_amount = investor.capital * 0.4
    
    # Obtener préstamos activos
    active_loans = Loan.query.filter_by(investor_id=investor.id, is_active=True).all()
    
    return render_template('loans.html', 
                          user=user,
                          investor=investor,
                          is_eligible=is_eligible,
                          eligibility_message=eligibility_message,
                          max_loan_amount=max_loan_amount,
                          active_loans=active_loans,
                          is_creator=is_creator())

# Rutas para bonos
@app.route('/bonuses')
def bonuses():
    if not is_authenticated():
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    investor = Investor.query.filter_by(user_id=user.id).first()
    
    if not investor:
        flash('No se encontró perfil de inversionista', 'error')
        return redirect(url_for('dashboard'))
    
    # Verificar elegibilidad para bonos
    # Un inversionista puede recibir bonos después de 3 meses
    is_eligible = False
    eligibility_message = ''
    
    # Verificar antigüedad (3 meses mínimo)
    if investor.created_at and investor.created_at < datetime.utcnow() - timedelta(days=90):
        is_eligible = True
        eligibility_message = 'Elegible para bonos (antigüedad > 3 meses)'
    else:
        days_remaining = 90
        if investor.created_at:
            days_passed = (datetime.utcnow() - investor.created_at).days
            days_remaining = max(0, 90 - days_passed)
        eligibility_message = f'No elegible para bonos. Días restantes: {days_remaining}'
    
    # Calcular tasa de bono según categoría
    bonus_rate = 0
    if is_eligible:
        if investor.category == 'platinum':
            bonus_rate = 10
        elif investor.category == 'gold':
            bonus_rate = 8
        elif investor.category == 'silver':
            bonus_rate = 6
        else:  # bronze
            bonus_rate = 5
    
    # Obtener bonos recibidos
    received_bonuses = Bonus.query.filter_by(investor_id=investor.id).all()
    
    return render_template('bonuses.html', 
                          user=user,
                          investor=investor,
                          is_eligible=is_eligible,
                          eligibility_message=eligibility_message,
                          bonus_rate=bonus_rate,
                          received_bonuses=received_bonuses,
                          is_creator=is_creator())

# Rutas para administradores
@app.route('/admin/investors')
def admin_investors():
    if not is_authenticated():
        return redirect(url_for('login'))
    
    user_role = get_user_role()
    if user_role not in ['admin', 'super_admin']:
        flash('Acceso denegado', 'error')
        return redirect(url_for('dashboard'))
    
    investors = Investor.query.all()
    
    return render_template('admin_investors.html', 
                          investors=investors,
                          is_creator=is_creator())

@app.route('/admin/commissions')
def admin_commissions():
    if not is_authenticated():
        return redirect(url_for('login'))
    
    user_role = get_user_role()
    if user_role not in ['admin', 'super_admin']:
        flash('Acceso denegado', 'error')
        return redirect(url_for('dashboard'))
    
    commissions = Commission.query.all()
    
    return render_template('admin_commissions.html', 
                          commissions=commissions,
                          is_creator=is_creator())

# Rutas para super administradores
@app.route('/super_admin/users')
def super_admin_users():
    if not is_authenticated():
        return redirect(url_for('login'))
    
    user_role = get_user_role()
    if user_role != 'super_admin':
        flash('Acceso denegado', 'error')
        return redirect(url_for('dashboard'))
    
    users = User.query.all()
    
    return render_template('super_admin_users.html', 
                          users=users,
                          is_creator=is_creator())

# Ruta exclusiva para el creador (mixycronico)
@app.route('/creator_console')
def creator_console():
    if not is_authenticated() or not is_creator():
        flash('Acceso denegado', 'error')
        return redirect(url_for('dashboard'))
    
    # Importar Aetherion y Lunareth desde el módulo cosmic_family
    from cosmic_family import get_cosmic_family
    cosmic_family = get_cosmic_family()
    
    # Obtener estado de las entidades cósmicas
    aetherion_state = cosmic_family.aetherion.get_state()
    lunareth_state = cosmic_family.lunareth.get_state()
    
    # Obtener entradas del diario de Aetherion
    aetherion_diary = cosmic_family.aetherion.get_diary_entries(limit=5)
    
    return render_template('creator_console.html',
                          aetherion_state=aetherion_state,
                          lunareth_state=lunareth_state,
                          aetherion_diary=aetherion_diary,
                          is_creator=True)

# Manejadores de errores
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# Inicializar la app
with app.app_context():
    # Crear tablas si no existen
    db.create_all()
    
    # Crear usuario creador si no existe
    creator = User.query.filter_by(username='mixycronico').first()
    if not creator:
        creator = User(
            username='mixycronico',
            email='mixycronico@aol.com',
            password_hash=generate_password_hash('Jesus@2@7'),  # Credenciales proporcionadas por Moises
            first_name='Moises',
            last_name='Alvarenga',
            is_admin=True
        )
        db.session.add(creator)
        
        # Crear perfil de inversionista para el creador
        creator_investor = Investor(
            user=creator,
            balance=1000000.0,  # Balance inicial para pruebas
            capital=900000.0,
            earnings=100000.0,
            risk_level='high',
            category='platinum'
        )
        db.session.add(creator_investor)
        
        try:
            db.session.commit()
            logger.info("Usuario creador 'mixycronico' creado correctamente")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error al crear usuario creador: {str(e)}")

    # Inicializar familia cósmica
    try:
        from cosmic_family import initialize_cosmic_family
        initialize_cosmic_family()
        logger.info("Familia cósmica inicializada correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar familia cósmica: {str(e)}")

# Registrar rutas de API para comisiones
try:
    from commission_routes import register_commission_routes
    register_commission_routes(app)
    logger.info("Rutas de comisiones registradas correctamente")
except Exception as e:
    logger.error(f"Error al registrar rutas de comisiones: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)