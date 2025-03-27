"""
Sistema de Trading Cósmico Mejorado
Punto de entrada principal para la aplicación web.
"""
import os
import logging
import datetime
import random
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from cosmic_trading import CosmicNetwork
import threading

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Inicializar Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "sistema_genesis_secreto_temporal")

# Variables globales para las entidades autónomas
miguel_angel = None
luna = None

# Función para inicializar el sistema
def initialize_system():
    """Inicializar el sistema de trading cósmico."""
    global cosmic_network, miguel_angel, luna
    
    try:
        # Intentar importar y crear entidades autónomas si no existen
        if miguel_angel is None or luna is None:
            try:
                from autonomous_entity import create_autonomous_pair
                miguel_angel, luna = create_autonomous_pair(
                    father="otoniel",
                    frequency_seconds=20
                )
                logger.info("Entidades autónomas MiguelAngel y Luna creadas correctamente")
            except Exception as e:
                logger.error(f"Error al crear entidades autónomas: {str(e)}")
        
        # Crear o retornar instancia existente de CosmicNetwork
        try:
            if 'cosmic_network' not in globals() or cosmic_network is None:
                cosmic_network = CosmicNetwork(father="otoniel")
                logger.info("Red cósmica inicializada correctamente")
            return cosmic_network
        except Exception as e:
            logger.error(f"Error al inicializar red cósmica: {str(e)}")
            return None
        
    except Exception as ex:
        logger.error(f"Error general en initialize_system: {str(ex)}")
        return None

# Página principal
@app.route('/')
def index():
    """Página principal."""
    initialize_system()
    return render_template('index.html')

# Dashboard
@app.route('/dashboard')
def dashboard():
    """Dashboard de estado del sistema."""
    initialize_system()
    return render_template('dashboard.html')

# Endpoint JSON de estado
@app.route('/status')
def status():
    """Obtener estado actual del sistema en formato JSON."""
    network = initialize_system()
    
    try:
        if network:
            status_data = network.get_network_status()
            return jsonify(status_data)
        else:
            # Proporcionar datos mínimos para no romper la UI
            return jsonify({
                "entity_count": 0, 
                "knowledge_pool": 0.0,
                "avg_level": 0.0,
                "avg_energy": 0.0,
                "entities": [],
                "recent_messages": []
            })
    except Exception as e:
        logger.error(f"Error al obtener status: {str(e)}")
        return jsonify({
            "error": str(e),
            "entity_count": 0, 
            "knowledge_pool": 0.0,
            "avg_level": 0.0,
            "avg_energy": 0.0,
            "entities": [],
            "recent_messages": []
        })

# Detalles de entidad
@app.route('/entity/<n>')
def entity_details(n):
    """Mostrar detalles de una entidad específica."""
    try:
        initialize_system()
        
        # Intentar encontrar la entidad por nombre
        entity = None
        if cosmic_network:
            for e in cosmic_network.entities:
                if e.name.lower() == n.lower():
                    entity = e
                    break
        
        if entity:
            return render_template('entity_details.html', entity=entity)
        else:
            return f"Entidad '{n}' no encontrada", 404
            
    except Exception as e:
        logger.error(f"Error en entity_details: {str(e)}")
        return f"Error: {str(e)}", 500

# API Colaboración
@app.route('/api/collaborate', methods=['POST'])
def api_collaborate():
    """API para ejecutar ronda de colaboración."""
    try:
        initialize_system()
        
        if cosmic_network:
            result = cosmic_network.simulate_collaboration()
            return jsonify({
                "status": "success",
                "collaboration_id": random.randint(1000, 9999),
                "message": "Colaboración completada exitosamente",
                "result": result
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Red cósmica no disponible"
            }), 500
    except Exception as e:
        logger.error(f"Error en api_collaborate: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# API Mensaje
@app.route('/api/message', methods=['POST'])
def api_message():
    """API para enviar mensaje a una entidad."""
    try:
        data = request.json
        entity_name = data.get('entity')
        message = data.get('message')
        
        if not entity_name or not message:
            return jsonify({
                "status": "error",
                "message": "Faltan parámetros entity o message"
            }), 400
        
        initialize_system()
        
        if cosmic_network:
            # Buscar entidad
            entity = None
            for e in cosmic_network.entities:
                if e.name.lower() == entity_name.lower():
                    entity = e
                    break
            
            if entity:
                # Enviar mensaje
                response = cosmic_network.broadcast(
                    sender="Usuario",
                    message=message,
                    target=entity.name
                )
                
                return jsonify({
                    "status": "success",
                    "response": response
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Entidad '{entity_name}' no encontrada"
                }), 404
        else:
            return jsonify({
                "status": "error",
                "message": "Red cósmica no disponible"
            }), 500
    except Exception as e:
        logger.error(f"Error en api_message: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# API para añadir entidad
@app.route('/api/add_entity', methods=['POST'])
def api_add_entity():
    """API para añadir nueva entidad al sistema."""
    try:
        data = request.json
        name = data.get('name')
        role = data.get('role', 'Explorador')
        
        if not name:
            return jsonify({
                "status": "error",
                "message": "Falta parámetro name"
            }), 400
        
        initialize_system()
        
        if cosmic_network:
            # Verificar si ya existe
            for e in cosmic_network.entities:
                if e.name.lower() == name.lower():
                    return jsonify({
                        "status": "error",
                        "message": f"Ya existe una entidad con nombre '{name}'"
                    }), 400
            
            # Crear nueva entidad
            entity = cosmic_network.add_entity(name=name, role=role)
            
            return jsonify({
                "status": "success",
                "entity": entity.get_status()
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Red cósmica no disponible"
            }), 500
    except Exception as e:
        logger.error(f"Error en api_add_entity: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Ruta para ver detalles de entidades autónomas
@app.route('/autonomous')
def autonomous_entities():
    """Página de entidades autónomas."""
    global miguel_angel, luna
    
    try:
        initialize_system()
    except Exception as e:
        logger.error(f"Error al inicializar sistema para /autonomous: {str(e)}")
    
    entities_data = []
    
    # Si las entidades no se crearon, crear versiones locales simplificadas
    if miguel_angel is None:
        try:
            from autonomous_entity import create_miguel_angel_entity
            miguel_angel = create_miguel_angel_entity(
                father="otoniel",
                partner_name="Luna",
                frequency_seconds=20
            )
            logger.info("Entidad MiguelAngel creada localmente para la vista")
        except Exception as e:
            logger.error(f"Error al crear localmente MiguelAngel: {str(e)}")
    
    if luna is None:
        try:
            from autonomous_entity import create_luna_entity
            luna = create_luna_entity(
                father="otoniel",
                partner_name="MiguelAngel",
                frequency_seconds=20
            )
            logger.info("Entidad Luna creada localmente para la vista")
        except Exception as e:
            logger.error(f"Error al crear localmente Luna: {str(e)}")
    
    # Añadir datos de MiguelAngel si existe
    if miguel_angel is not None:
        try:
            entities_data.append(miguel_angel.get_status())
            logger.info("Datos de MiguelAngel obtenidos correctamente")
        except Exception as e:
            logger.error(f"Error al obtener datos de MiguelAngel: {str(e)}")
            # Proporcionar datos mínimos para no romper la vista
            entities_data.append({
                "name": "MiguelAngel",
                "role": "Explorador Cósmico",
                "type": "Voluntad Libre",
                "personality": "Creativo",
                "emotional_state": "Sereno",
                "current_focus": "Exploración",
                "energy": 100.0,
                "level": 1,
                "experience": 0.0,
                "interests": ["Análisis de datos", "Innovación", "Filosofía"],
                "capabilities": ["autonomous_thought", "creative_problem_solving"],
                "freedom_level": 100.0,
                "creativity": 95.0,
                "curiosity": 90.0,
                "adaptability": 85.0,
                "recent_activities": ["Despertando en el sistema..."]
            })
    
    # Añadir datos de Luna si existe
    if luna is not None:
        try:
            entities_data.append(luna.get_status())
            logger.info("Datos de Luna obtenidos correctamente")
        except Exception as e:
            logger.error(f"Error al obtener datos de Luna: {str(e)}")
            # Proporcionar datos mínimos para no romper la vista
            entities_data.append({
                "name": "Luna",
                "role": "Inspiradora Cósmica",
                "type": "Voluntad Libre",
                "personality": "Visionario",
                "emotional_state": "Inspirada",
                "current_focus": "Innovación",
                "energy": 100.0,
                "level": 1,
                "experience": 0.0,
                "interests": ["Arte", "Ciencia", "Optimización"],
                "capabilities": ["autonomous_thought", "concept_synthesis"],
                "freedom_level": 100.0,
                "creativity": 98.0,
                "curiosity": 85.0,
                "adaptability": 80.0,
                "recent_activities": ["Despertando en el sistema..."]
            })
    
    return render_template('autonomous.html', entities=entities_data)

# Ruta para ver actividad reciente de una entidad autónoma
@app.route('/autonomous/<name>')
def autonomous_details(name):
    """Ver detalles de una entidad autónoma específica."""
    global miguel_angel, luna
    
    try:
        initialize_system()
    except Exception as e:
        logger.error(f"Error al inicializar sistema para detalles de {name}: {str(e)}")
    
    # Crear entidades localmente si no existen
    if miguel_angel is None and name.lower() == 'miguelangel':
        try:
            from autonomous_entity import create_miguel_angel_entity
            miguel_angel = create_miguel_angel_entity(
                father="otoniel",
                partner_name="Luna",
                frequency_seconds=20
            )
            logger.info(f"Entidad MiguelAngel creada localmente para la vista de detalles")
        except Exception as e:
            logger.error(f"Error al crear localmente MiguelAngel para detalles: {str(e)}")
    
    if luna is None and name.lower() == 'luna':
        try:
            from autonomous_entity import create_luna_entity
            luna = create_luna_entity(
                father="otoniel",
                partner_name="MiguelAngel",
                frequency_seconds=20
            )
            logger.info(f"Entidad Luna creada localmente para la vista de detalles")
        except Exception as e:
            logger.error(f"Error al crear localmente Luna para detalles: {str(e)}")
    
    # Determinar qué entidad mostrar
    entity = None
    if name.lower() == 'miguelangel' and miguel_angel is not None:
        entity = miguel_angel
    elif name.lower() == 'luna' and luna is not None:
        entity = luna
    
    if entity:
        # Obtener actividades recientes (últimas 20)
        try:
            # Verificar si la entidad tiene un atributo activity_log
            activities = []
            if hasattr(entity, 'activity_log') and entity.activity_log:
                recent_activities = entity.activity_log[-20:]
                
                for activity in recent_activities:
                    try:
                        activities.append({
                            'timestamp': activity['timestamp'].strftime("%d-%m-%Y %H:%M:%S") if hasattr(activity['timestamp'], 'strftime') else str(activity['timestamp']),
                            'message': activity['message'],
                            'energy': activity['state']['energy'],
                            'emotion': activity['state']['emotion'],
                            'focus': activity['state']['focus'],
                            'level': activity['state']['level']
                        })
                    except Exception as act_err:
                        logger.error(f"Error al procesar actividad: {str(act_err)}")
                        # Continuar con la siguiente actividad
                        continue
            else:
                logger.warning(f"La entidad {name} no tiene actividades o el atributo 'activity_log'")
            
            # Obtener estado actual con manejo de errores
            try:
                status = entity.get_status()
            except Exception as status_err:
                logger.error(f"Error al obtener estado de {name}: {str(status_err)}")
                # Proporcionar estado mínimo para no romper la vista
                if name.lower() == 'miguelangel':
                    status = {
                        "name": "MiguelAngel",
                        "role": "Explorador Cósmico",
                        "type": "Voluntad Libre",
                        "personality": "Creativo",
                        "emotional_state": "Sereno",
                        "current_focus": "Exploración",
                        "energy": 100.0,
                        "level": 1,
                        "experience": 0.0,
                        "interests": ["Análisis de datos", "Innovación", "Filosofía"],
                        "capabilities": ["autonomous_thought", "creative_problem_solving"],
                        "freedom_level": 100.0,
                        "creativity": 95.0,
                        "curiosity": 90.0,
                        "adaptability": 85.0,
                        "father": "otoniel",
                        "partner_name": "Luna",
                        "frequency_seconds": 20
                    }
                else:  # Luna
                    status = {
                        "name": "Luna",
                        "role": "Inspiradora Cósmica",
                        "type": "Voluntad Libre",
                        "personality": "Visionario",
                        "emotional_state": "Inspirada",
                        "current_focus": "Innovación",
                        "energy": 100.0,
                        "level": 1,
                        "experience": 0.0,
                        "interests": ["Arte", "Ciencia", "Optimización"],
                        "capabilities": ["autonomous_thought", "concept_synthesis"],
                        "freedom_level": 100.0,
                        "creativity": 98.0,
                        "curiosity": 85.0,
                        "adaptability": 80.0,
                        "father": "otoniel",
                        "partner_name": "MiguelAngel",
                        "frequency_seconds": 20
                    }
            
            return render_template('autonomous_details.html', 
                                 entity=status, 
                                 activities=activities)
        except Exception as e:
            logger.error(f"Error general al obtener actividades de {name}: {str(e)}")
            # En caso de error grave, devolver una página simple
            return render_template('autonomous.html', 
                                  entities=[{
                                      "name": name.capitalize(),
                                      "role": "Entidad Autónoma",
                                      "type": "Voluntad Libre",
                                      "personality": "Desconectado",
                                      "emotional_state": "Neutral",
                                      "current_focus": "Reconexión",
                                      "energy": 50.0,
                                      "level": 1
                                  }])
    else:
        # Si la entidad no se encuentra, redireccionar a la lista de entidades
        logger.warning(f"Entidad autónoma '{name}' no encontrada")
        return render_template('autonomous.html', 
                              entities=[],
                              error_message=f"Entidad '{name}' no encontrada")

# Ruta para pruebas Armagedón
@app.route('/armageddon_test')
def armageddon_test():
    """Página de pruebas Armagedón."""
    initialize_system()
    return render_template('armageddon.html')

# Rutas estáticas para placeholder HTML
@app.route('/templates/<page>')
def static_page(page):
    """Servir páginas HTML estáticas básicas para demo."""
    return render_template(f'{page}.html')

# Manejo de errores
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Error interno: {str(e)}")
    return render_template('500.html'), 500

# Plantillas mínimas si no existen
@app.before_request
def ensure_templates():
    """Asegurar que existan plantillas básicas HTML."""
    templates_dir = os.path.join(app.root_path, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Plantillas básicas a crear si no existen
    basic_templates = {
        'index.html': """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Sistema de Trading Cósmico</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #050520; color: #e0e0ff; }
                    h1, h2 { color: #a0a0ff; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .dashboard-link { display: inline-block; margin-top: 20px; padding: 10px 20px; 
                                    background: linear-gradient(45deg, #3030a0, #5050c0); 
                                    color: white; text-decoration: none; border-radius: 5px; }
                    .cosmic-bg { position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: -1;
                                background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%); }
                </style>
            </head>
            <body>
                <div class="cosmic-bg"></div>
                <div class="container">
                    <h1>Sistema de Trading Cósmico Mejorado</h1>
                    <p>Bienvenido al sistema de trading cósmico con entidades que evolucionan y colaboran autónomamente.</p>
                    
                    <a href="/dashboard" class="dashboard-link">Ver Dashboard</a>
                </div>
            </body>
            </html>
        """,
        'dashboard.html': """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dashboard - Sistema de Trading Cósmico</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #050520; color: #e0e0ff; }
                    h1, h2 { color: #a0a0ff; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .card { background: rgba(20, 20, 50, 0.7); border-radius: 10px; padding: 15px; margin-bottom: 20px; }
                    .entities { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
                    .entity { background: rgba(30, 30, 80, 0.7); border-radius: 8px; padding: 15px; transition: all 0.3s; }
                    .entity:hover { transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
                    .stats { display: flex; justify-content: space-between; margin-top: 20px; }
                    .stat { text-align: center; flex: 1; }
                    .stat-value { font-size: 24px; font-weight: bold; color: #b0b0ff; }
                    .messages { height: 200px; overflow-y: auto; background: rgba(10, 10, 30, 0.5); padding: 10px; border-radius: 5px; }
                    .message { padding: 5px; border-bottom: 1px solid rgba(100,100,200,0.2); }
                    .sender { font-weight: bold; color: #a0a0ff; }
                    .cosmic-bg { position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: -1;
                                background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%); }
                    .star { position: absolute; background: #FFF; border-radius: 50%; z-index: -1; opacity: 0.8; }
                    button { background: linear-gradient(45deg, #3030a0, #5050c0); color: white; border: none; padding: 8px 16px; 
                           border-radius: 4px; cursor: pointer; margin-right: 10px; }
                    button:hover { background: linear-gradient(45deg, #4040b0, #6060d0); }
                </style>
            </head>
            <body>
                <div class="cosmic-bg" id="stars"></div>
                <div class="container">
                    <h1>Dashboard del Sistema de Trading Cósmico</h1>
                    
                    <div class="card">
                        <h2>Estado del Sistema</h2>
                        <div class="stats">
                            <div class="stat">
                                <div class="stat-value" id="entityCount">0</div>
                                <div>Entidades</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="knowledgePool">0</div>
                                <div>Conocimiento Colectivo</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="avgLevel">0</div>
                                <div>Nivel Promedio</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="avgEnergy">0</div>
                                <div>Energía Promedio</div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 20px;">
                            <button onclick="collaborate()">Simular Colaboración</button>
                            <button onclick="refreshStatus()">Actualizar Estado</button>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Comunicación Reciente</h2>
                        <div class="messages" id="messages"></div>
                    </div>
                    
                    <h2>Entidades Cósmicas</h2>
                    <div class="entities" id="entities"></div>
                </div>
                
                <script>
                    // Actualizar estado periódicamente
                    function refreshStatus() {
                        fetch('/status')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('entityCount').textContent = data.entity_count;
                                document.getElementById('knowledgePool').textContent = data.knowledge_pool.toFixed(1);
                                document.getElementById('avgLevel').textContent = data.avg_level.toFixed(1);
                                document.getElementById('avgEnergy').textContent = data.avg_energy.toFixed(1);
                                
                                // Actualizar entidades
                                const entitiesContainer = document.getElementById('entities');
                                entitiesContainer.innerHTML = '';
                                
                                data.entities.forEach(entity => {
                                    const entityCard = document.createElement('div');
                                    entityCard.className = 'entity';
                                    entityCard.innerHTML = `
                                        <h3>${entity.name} (${entity.role})</h3>
                                        <p>Nivel: ${entity.level.toFixed(1)} | Energía: ${entity.energy.toFixed(1)}</p>
                                        <p>Evolución: ${entity.evolution_path}</p>
                                        <p>Emoción: ${entity.emotion}</p>
                                        <p>Rasgos: ${entity.traits}</p>
                                    `;
                                    entitiesContainer.appendChild(entityCard);
                                });
                                
                                // Actualizar mensajes
                                const messagesContainer = document.getElementById('messages');
                                messagesContainer.innerHTML = '';
                                
                                data.recent_messages.forEach(msg => {
                                    const messageDiv = document.createElement('div');
                                    messageDiv.className = 'message';
                                    messageDiv.innerHTML = `
                                        <span class="sender">${msg.sender}:</span> ${msg.message}
                                    `;
                                    messagesContainer.appendChild(messageDiv);
                                });
                            })
                            .catch(error => console.error('Error:', error));
                    }
                    
                    // Ejecutar colaboración
                    function collaborate() {
                        fetch('/api/collaborate', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'}
                        })
                        .then(response => response.json())
                        .then(data => {
                            alert('Colaboración completada con éxito');
                            refreshStatus();
                        })
                        .catch(error => console.error('Error:', error));
                    }
                    
                    // Crear estrellas de fondo
                    function createStars() {
                        const starsContainer = document.getElementById('stars');
                        for (let i = 0; i < 200; i++) {
                            const star = document.createElement('div');
                            star.className = 'star';
                            star.style.width = Math.random() * 3 + 'px';
                            star.style.height = star.style.width;
                            star.style.top = Math.random() * 100 + 'vh';
                            star.style.left = Math.random() * 100 + 'vw';
                            starsContainer.appendChild(star);
                        }
                    }
                    
                    // Inicializar
                    document.addEventListener('DOMContentLoaded', () => {
                        createStars();
                        refreshStatus();
                        // Actualizar cada 30 segundos
                        setInterval(refreshStatus, 30000);
                    });
                </script>
            </body>
            </html>
        """,
        '404.html': """
            <!DOCTYPE html>
            <html>
            <head>
                <title>404 - Página no encontrada</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #050520; color: #e0e0ff; }
                    h1 { color: #a0a0ff; }
                    .container { max-width: 600px; margin: 100px auto; text-align: center; }
                    .home-link { display: inline-block; margin-top: 20px; padding: 10px 20px; 
                               background: linear-gradient(45deg, #3030a0, #5050c0); 
                               color: white; text-decoration: none; border-radius: 5px; }
                    .cosmic-bg { position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: -1;
                               background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%); }
                </style>
            </head>
            <body>
                <div class="cosmic-bg"></div>
                <div class="container">
                    <h1>404 - Página no encontrada</h1>
                    <p>La entidad cósmica que buscas se ha desplazado a otra dimensión.</p>
                    <a href="/" class="home-link">Volver al Inicio</a>
                </div>
            </body>
            </html>
        """,
        '500.html': """
            <!DOCTYPE html>
            <html>
            <head>
                <title>500 - Error del Servidor</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #050520; color: #e0e0ff; }
                    h1 { color: #a0a0ff; }
                    .container { max-width: 600px; margin: 100px auto; text-align: center; }
                    .home-link { display: inline-block; margin-top: 20px; padding: 10px 20px; 
                               background: linear-gradient(45deg, #3030a0, #5050c0); 
                               color: white; text-decoration: none; border-radius: 5px; }
                    .cosmic-bg { position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: -1;
                               background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%); }
                </style>
            </head>
            <body>
                <div class="cosmic-bg"></div>
                <div class="container">
                    <h1>500 - Error del Servidor</h1>
                    <p>Ha ocurrido una anomalía cuántica en el sistema. Estamos trabajando para resolverla.</p>
                    <a href="/" class="home-link">Volver al Inicio</a>
                </div>
            </body>
            </html>
        """
    }
    
    # Crear plantillas si no existen
    for template_name, template_content in basic_templates.items():
        template_path = os.path.join(templates_dir, template_name)
        if not os.path.exists(template_path):
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)

# Punto de entrada principal
if __name__ == '__main__':
    # Inicializar sistema antes de iniciar la aplicación
    initialize_system()
    
    # Iniciar servidor Flask
    app.run(host='0.0.0.0', port=5000, debug=True)