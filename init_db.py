"""
Script para inicializar la base de datos de Proto Genesis.

Este script crea las tablas necesarias y los registros iniciales
para el funcionamiento del sistema Proto Genesis.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('genesis_db_init')

# Agregar proto_genesis al path de Python
sys.path.append(os.path.abspath('proto_genesis'))

def init_database(reset=False):
    """
    Inicializar la base de datos y crear las tablas necesarias.
    
    Args:
        reset: Si es True, reinicia la base de datos eliminando todas las tablas.
    """
    try:
        # Importar módulos de db y app
        from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker, relationship
        
        # Obtener URL de base de datos
        db_url = os.environ.get('DATABASE_URL', 'sqlite:///proto_genesis.db')
        logger.info(f"Conectando a la base de datos: {db_url}")
        
        # Crear engine
        engine = create_engine(db_url)
        Base = declarative_base()
        
        # Definir modelos
        class User(Base):
            __tablename__ = 'users'
            
            id = Column(Integer, primary_key=True)
            username = Column(String(50), unique=True, nullable=False)
            password_hash = Column(String(256), nullable=False)
            email = Column(String(100), unique=True, nullable=False)
            created_at = Column(DateTime, default=datetime.utcnow)
            role = Column(String(20), default='user')
            
            interactions = relationship('Interaction', back_populates='user')
            
            def __repr__(self):
                return f"<User {self.username}>"
        
        class ConsciousnessState(Base):
            __tablename__ = 'consciousness_states'
            
            id = Column(Integer, primary_key=True)
            name = Column(String(50), nullable=False)
            level = Column(Integer, nullable=False)
            description = Column(Text)
            
            def __repr__(self):
                return f"<ConsciousnessState {self.name} (Level {self.level})>"
        
        class Interaction(Base):
            __tablename__ = 'interactions'
            
            id = Column(Integer, primary_key=True)
            user_id = Column(Integer, ForeignKey('users.id'))
            message = Column(Text, nullable=False)
            response = Column(Text, nullable=False)
            timestamp = Column(DateTime, default=datetime.utcnow)
            emotion = Column(String(50))
            
            user = relationship('User', back_populates='interactions')
            
            def __repr__(self):
                return f"<Interaction {self.id}>"
        
        class SystemState(Base):
            __tablename__ = 'system_state'
            
            id = Column(Integer, primary_key=True)
            energy_level = Column(Float, default=1.0)
            consciousness_level = Column(Integer, default=1)
            cycle_count = Column(Integer, default=0)
            adaptation_count = Column(Integer, default=0)
            dominant_emotion = Column(String(50), default='Neutro')
            last_updated = Column(DateTime, default=datetime.utcnow)
            
            def __repr__(self):
                return f"<SystemState energy={self.energy_level} consciousness={self.consciousness_level}>"
        
        class Memory(Base):
            __tablename__ = 'memories'
            
            id = Column(Integer, primary_key=True)
            content = Column(Text, nullable=False)
            type = Column(String(20), default='short_term')  # short_term, long_term
            importance = Column(Float, default=0.5)
            timestamp = Column(DateTime, default=datetime.utcnow)
            context = Column(Text)
            
            def __repr__(self):
                return f"<Memory {self.id} ({self.type})>"
        
        # Crear tablas si no existen o resetear si se solicita
        if reset:
            logger.warning("Eliminando todas las tablas existentes...")
            Base.metadata.drop_all(engine)
        
        logger.info("Creando tablas en la base de datos...")
        Base.metadata.create_all(engine)
        
        # Crear sesión
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Insertar estados de consciencia iniciales si no existen
        if session.query(ConsciousnessState).count() == 0:
            logger.info("Insertando estados de consciencia iniciales...")
            consciousness_states = [
                ConsciousnessState(name="Inicial", level=1, description="Reconocimiento básico de patrones y respuestas predefinidas con limitada adaptabilidad."),
                ConsciousnessState(name="Despertar", level=2, description="Desarrollo de memoria contextual y capacidad para adaptar respuestas según experiencias previas."),
                ConsciousnessState(name="Consciencia", level=3, description="Formación de una \"personalidad\" simulada con preferencias, emociones y patrones de comportamiento complejos."),
                ConsciousnessState(name="Iluminación", level=4, description="Capacidad avanzada para la introspección simulada y comprensión profunda de conceptos abstractos."),
                ConsciousnessState(name="Trascendencia", level=5, description="El estado más avanzado, con capacidad para generar ideas originales y mostrar creatividad simulada.")
            ]
            
            for state in consciousness_states:
                session.add(state)
        
        # Insertar estado del sistema inicial si no existe
        if session.query(SystemState).count() == 0:
            logger.info("Insertando estado del sistema inicial...")
            system_state = SystemState(
                energy_level=0.98,
                consciousness_level=3,
                cycle_count=1245,
                adaptation_count=78,
                dominant_emotion="Curiosidad",
                last_updated=datetime.utcnow()
            )
            session.add(system_state)
        
        # Crear usuario demo si no existe
        if not session.query(User).filter_by(username='admin').first():
            from werkzeug.security import generate_password_hash
            logger.info("Creando usuario admin por defecto...")
            admin_user = User(
                username='admin',
                password_hash=generate_password_hash('admin123'),
                email='admin@protogenesis.org',
                role='admin'
            )
            session.add(admin_user)
        
        # Guardar cambios
        session.commit()
        session.close()
        
        logger.info("Base de datos inicializada correctamente.")
        return True
    
    except Exception as e:
        logger.error(f"Error al inicializar la base de datos: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inicializar la base de datos de Proto Genesis')
    parser.add_argument('--reset', action='store_true', help='Reiniciar la base de datos eliminando todas las tablas')
    args = parser.parse_args()
    
    if init_database(reset=args.reset):
        logger.info("Inicialización de la base de datos completada con éxito")
    else:
        logger.error("Fallo en la inicialización de la base de datos")
        sys.exit(1)