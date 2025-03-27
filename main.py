"""
Sistema de Trading Cósmico Mejorado
Punto de entrada principal para la aplicación web.
"""

import os
import logging
from flask import Flask, render_template, jsonify, request, redirect, url_for
from enhanced_simple_cosmic_trader import initialize_enhanced_trading
from armageddon_api import register_armageddon_routes

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "cosmic_secret_key")

# Registrar rutas de Armagedón
register_armageddon_routes(app)

# Variables globales
cosmic_network = None
aetherion = None
lunareth = None

def initialize_system():
    """Inicializar el sistema de trading cósmico."""
    global cosmic_network, aetherion, lunareth
    
    if cosmic_network is None:
        logger.info("Inicializando Sistema de Trading Cósmico...")
        cosmic_network, aetherion, lunareth = initialize_enhanced_trading(
            father_name="otoniel",
            include_extended_entities=True
        )
        logger.info(f"Sistema inicializado con {len(cosmic_network.entities)} entidades")

# Rutas de la aplicación
@app.route('/')
def index():
    """Página principal."""
    initialize_system()
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard de estado del sistema."""
    initialize_system()
    return render_template('dashboard.html')

@app.route('/status')
def status():
    """Obtener estado actual del sistema en formato JSON."""
    initialize_system()
    return jsonify(cosmic_network.get_network_status())

@app.route("/entity/<name>")
def entity_details(n):
    """Mostrar detalles de una entidad específica."""
    initialize_system()
    
    # Buscar entidad por nombre
    entity = next((e for e in cosmic_network.entities if e.name == n), None)
    
    if entity:
        return render_template('entity.html', entity=entity.get_status())
    else:
        return "Entidad no encontrada", 404

@app.route('/api/collaborate', methods=['POST'])
def api_collaborate():
    """API para ejecutar ronda de colaboración."""
    initialize_system()
    
    results = cosmic_network.simulate_collaboration()
    return jsonify({
        "status": "success",
        "results": results
    })

@app.route('/api/message', methods=['POST'])
def api_message():
    """API para enviar mensaje a una entidad."""
    initialize_system()
    
    data = request.json
    if not data or 'sender' not in data or 'message' not in data:
        return jsonify({"status": "error", "message": "Datos incompletos"}), 400
    
    sender = next((e for e in cosmic_network.entities if e.name == data['sender']), None)
    
    if not sender:
        return jsonify({"status": "error", "message": "Entidad no encontrada"}), 404
    
    # Enviar mensaje a la red
    msg = sender.generate_message("luz", data['message'])
    cosmic_network.broadcast(sender.name, msg)
    
    return jsonify({
        "status": "success",
        "message": msg
    })

@app.route('/api/add_entity', methods=['POST'])
def api_add_entity():
    """API para añadir nueva entidad al sistema."""
    initialize_system()
    
    from enhanced_simple_cosmic_trader import EnhancedSpeculatorEntity, EnhancedStrategistEntity
    
    data = request.json
    if not data or 'name' not in data or 'type' not in data:
        return jsonify({"status": "error", "message": "Datos incompletos"}), 400
    
    # Verificar que el nombre no exista
    if any(e.name == data['name'] for e in cosmic_network.entities):
        return jsonify({"status": "error", "message": "El nombre ya existe"}), 400
    
    # Crear entidad según tipo
    if data['type'].lower() == 'speculator':
        entity = EnhancedSpeculatorEntity(data['name'], "Speculator")
    elif data['type'].lower() == 'strategist':
        entity = EnhancedStrategistEntity(data['name'], "Strategist")
    else:
        return jsonify({"status": "error", "message": "Tipo de entidad no válido"}), 400
    
    # Añadir a la red
    cosmic_network.add_entity(entity)
    
    return jsonify({
        "status": "success",
        "entity": entity.get_status()
    })

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
                        // Actualizar cada 5 segundos
                        setInterval(refreshStatus, 5000);
                    });
                </script>
            </body>
            </html>
        """
    }
    
    # Crear archivos de plantilla si no existen
    for name, content in basic_templates.items():
        file_path = os.path.join(templates_dir, name)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(content.strip())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
