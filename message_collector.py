"""
Sistema de Recopilación y Distribución de Mensajes para Sistema Genesis

Este módulo implementa un sistema centralizado que permite a todas las entidades
enviar mensajes, recopilarlos y distribuirlos a través de email en un formato
consolidado. Proporciona una interfaz común para que cualquier entidad comunique
información al creador de forma eficiente.

Características:
1. Cola centralizada de mensajes
2. Consolidación automática de mensajes por entidad, tipo y prioridad
3. Envío periódico de emails con diseño responsivo
4. Integración con todas las entidades del Sistema Genesis
5. Personalización de mensajes según tipo y entidad
6. Persistencia de mensajes en base de datos
"""

import os
import time
import random
import logging
import smtplib
import threading
import sqlite3
import json
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, deque

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración global
CREATOR_EMAIL = "mixycronico@aol.com"  # Email del creador
DEFAULT_SEND_INTERVAL = 3600 * 6  # Envío cada 6 horas por defecto
MIN_MESSAGES_FOR_IMMEDIATE_SEND = 10  # Mínimo de mensajes para envío inmediato
MAX_MESSAGES_PER_ENTITY = 15  # Máximo de mensajes por entidad en cada email


class MessageManager:
    """
    Gestor centralizado de mensajes para todas las entidades del Sistema Genesis.
    Permite recopilar, almacenar y enviar mensajes consolidados.
    """
    
    def __init__(self, 
                 send_interval: int = DEFAULT_SEND_INTERVAL, 
                 immediate_send_threshold: int = MIN_MESSAGES_FOR_IMMEDIATE_SEND):
        """
        Inicializar gestor de mensajes.
        
        Args:
            send_interval: Intervalo en segundos entre envíos automáticos
            immediate_send_threshold: Número de mensajes que activan envío inmediato
        """
        self.send_interval = send_interval
        self.immediate_send_threshold = immediate_send_threshold
        self.messages = defaultdict(list)  # Mensajes por tipo
        self.entity_messages = defaultdict(list)  # Mensajes por entidad
        self.personal_messages = []  # Mensajes personales del creador
        self.priority_messages = []  # Mensajes de alta prioridad
        self.all_messages = []  # Todos los mensajes en orden cronológico
        self.message_count = 0  # Contador de mensajes
        
        # Estado de envío
        self.last_send_time = time.time()
        self.emails_sent = 0
        self.is_enabled = True
        
        # Control de concurrencia
        self.lock = threading.Lock()
        
        # Inicializar base de datos
        self.init_db()
        
        # Cargar mensajes no enviados desde DB
        self.load_pending_messages()
        
        # Iniciar thread de envío automático
        self.send_thread = threading.Thread(target=self._auto_send_loop)
        self.send_thread.daemon = True
        self.send_thread.start()
        
        logger.info("Sistema de gestión de mensajes inicializado")
    
    def init_db(self):
        """Inicializar base de datos para mensajes."""
        try:
            self.conn = sqlite3.connect("message_system.db", check_same_thread=False)
            self.cursor = self.conn.cursor()
            
            # Crear tablas si no existen
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_name TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    is_priority INTEGER DEFAULT 0,
                    is_personal INTEGER DEFAULT 0,
                    sent INTEGER DEFAULT 0
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS emails (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    message_count INTEGER NOT NULL
                )
            ''')
            
            self.conn.commit()
            logger.info("Base de datos de mensajes inicializada")
        
        except Exception as e:
            logger.error(f"Error inicializando DB de mensajes: {str(e)}")
    
    def load_pending_messages(self):
        """Cargar mensajes pendientes desde la base de datos."""
        try:
            # Obtener mensajes no enviados
            self.cursor.execute(
                "SELECT id, entity_name, message_type, content, timestamp, is_priority, is_personal FROM messages WHERE sent = 0"
            )
            
            rows = self.cursor.fetchall()
            if not rows:
                logger.info("No hay mensajes pendientes para cargar desde DB")
                return
            
            # Procesar mensajes
            with self.lock:
                for msg_id, entity, msg_type, content, timestamp, is_priority, is_personal in rows:
                    message_data = {
                        "id": msg_id,
                        "entity": entity,
                        "type": msg_type,
                        "content": content,
                        "timestamp": timestamp,
                        "datetime": datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Añadir a las colecciones correspondientes
                    self.all_messages.append(message_data)
                    self.messages[msg_type].append(message_data)
                    self.entity_messages[entity].append(message_data)
                    
                    if is_priority:
                        self.priority_messages.append(message_data)
                    
                    if is_personal:
                        self.personal_messages.append(message_data)
                    
                    self.message_count += 1
            
            logger.info(f"Cargados {len(rows)} mensajes pendientes desde DB")
        
        except Exception as e:
            logger.error(f"Error cargando mensajes pendientes: {str(e)}")
    
    def add_message(self, entity_name: str, message_type: str, content: str, 
                  is_priority: bool = False, is_personal: bool = False) -> int:
        """
        Añadir un mensaje al sistema.
        
        Args:
            entity_name: Nombre de la entidad emisora
            message_type: Tipo de mensaje (estado, alerta, informe, etc)
            content: Contenido del mensaje
            is_priority: Indica si es un mensaje prioritario
            is_personal: Indica si es un mensaje personal para el creador
            
        Returns:
            ID del mensaje añadido
        """
        timestamp = time.time()
        message_id = None
        
        # Almacenar en DB primero para obtener ID
        try:
            self.cursor.execute(
                "INSERT INTO messages (entity_name, message_type, content, timestamp, is_priority, is_personal) VALUES (?, ?, ?, ?, ?, ?)",
                (entity_name, message_type, content, timestamp, 1 if is_priority else 0, 1 if is_personal else 0)
            )
            self.conn.commit()
            message_id = self.cursor.lastrowid
        except Exception as e:
            logger.error(f"Error almacenando mensaje en DB: {str(e)}")
            # Continuar con ID None
        
        # Crear objeto de mensaje
        message_data = {
            "id": message_id,
            "entity": entity_name,
            "type": message_type,
            "content": content,
            "timestamp": timestamp,
            "datetime": datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Añadir a colecciones correspondientes
        with self.lock:
            self.all_messages.append(message_data)
            self.messages[message_type].append(message_data)
            self.entity_messages[entity_name].append(message_data)
            
            if is_priority:
                self.priority_messages.append(message_data)
                
            if is_personal:
                self.personal_messages.append(message_data)
            
            self.message_count += 1
        
        # Verificar si debemos enviar inmediatamente
        if self.message_count >= self.immediate_send_threshold or (is_priority and len(self.priority_messages) >= 3):
            # Iniciar envío en thread separado para no bloquear
            threading.Thread(target=self.send_email_now).start()
        
        logger.debug(f"Mensaje añadido: [{entity_name}] {message_type}")
        return message_id
    
    def get_pending_messages(self) -> Dict[str, Any]:
        """
        Obtener todos los mensajes pendientes organizados.
        
        Returns:
            Dict con mensajes agrupados por tipo, entidad, etc.
        """
        with self.lock:
            result = {
                "by_type": dict(self.messages),
                "by_entity": dict(self.entity_messages),
                "personal": self.personal_messages.copy(),
                "priority": self.priority_messages.copy(),
                "all": sorted(self.all_messages, key=lambda x: x["timestamp"], reverse=True),
                "count": self.message_count
            }
            return result
    
    def mark_messages_as_sent(self):
        """Marcar mensajes como enviados en DB y limpiar colecciones."""
        try:
            # Obtener IDs de mensajes
            message_ids = []
            with self.lock:
                for msg in self.all_messages:
                    if msg.get("id"):
                        message_ids.append(msg["id"])
            
            if not message_ids:
                logger.info("No hay mensajes con ID para marcar como enviados")
                return
            
            # Actualizar DB
            placeholders = ", ".join(["?"] * len(message_ids))
            self.cursor.execute(f"UPDATE messages SET sent = 1 WHERE id IN ({placeholders})", message_ids)
            self.conn.commit()
            
            # Limpiar colecciones
            with self.lock:
                self.messages = defaultdict(list)
                self.entity_messages = defaultdict(list)
                self.personal_messages = []
                self.priority_messages = []
                self.all_messages = []
                self.message_count = 0
            
            logger.info(f"Marcados {len(message_ids)} mensajes como enviados")
        
        except Exception as e:
            logger.error(f"Error marcando mensajes como enviados: {str(e)}")
    
    def format_email_content(self, messages: Dict[str, Any]) -> str:
        """
        Formatear contenido HTML del email con todos los mensajes.
        
        Args:
            messages: Diccionario con mensajes agrupados
            
        Returns:
            HTML formateado para el email
        """
        # Fecha actual
        current_date = datetime.datetime.now().strftime('%d de %B de %Y')
        
        # Iniciar HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Informe del Sistema Genesis</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
                
                body {{
                    font-family: 'Roboto', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f9f9f9;
                    margin: 0;
                    padding: 0;
                }}
                
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                
                .header {{
                    background: linear-gradient(135deg, #1a237e, #283593);
                    color: white;
                    padding: 20px;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    margin: -20px -20px 20px;
                    text-align: center;
                }}
                
                h1 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 700;
                }}
                
                .date {{
                    opacity: 0.8;
                    font-size: 16px;
                    margin-top: 5px;
                }}
                
                h2 {{
                    color: #1a237e;
                    font-size: 22px;
                    border-bottom: 2px solid #e0e0e0;
                    padding-bottom: 10px;
                    margin-top: 30px;
                }}
                
                h3 {{
                    color: #303f9f;
                    font-size: 18px;
                    margin-top: 25px;
                }}
                
                .section {{
                    margin-bottom: 30px;
                }}
                
                .message {{
                    margin-bottom: 15px;
                    padding: 15px;
                    border-radius: 6px;
                    background-color: #f5f5f5;
                    border-left: 4px solid #3f51b5;
                }}
                
                .priority {{
                    border-left: 4px solid #f44336;
                    background-color: #ffebee;
                }}
                
                .personal {{
                    border-left: 4px solid #4caf50;
                    background-color: #e8f5e9;
                }}
                
                .entity-name {{
                    font-weight: bold;
                    color: #1a237e;
                }}
                
                .message-type {{
                    display: inline-block;
                    font-size: 12px;
                    text-transform: uppercase;
                    background-color: #e0e0e0;
                    padding: 2px 6px;
                    border-radius: 3px;
                    margin-left: 5px;
                }}
                
                .timestamp {{
                    color: #757575;
                    font-size: 12px;
                }}
                
                .content {{
                    margin-top: 10px;
                }}
                
                .footer {{
                    margin-top: 40px;
                    font-size: 14px;
                    color: #757575;
                    text-align: center;
                    border-top: 1px solid #e0e0e0;
                    padding-top: 20px;
                }}
                
                .divider {{
                    height: 1px;
                    background-color: #e0e0e0;
                    margin: 20px 0;
                }}
                
                @media only screen and (max-width: 600px) {{
                    .container {{
                        padding: 15px;
                    }}
                    
                    .header {{
                        padding: 15px;
                        margin: -15px -15px 15px;
                    }}
                    
                    h1 {{
                        font-size: 22px;
                    }}
                    
                    h2 {{
                        font-size: 20px;
                    }}
                    
                    h3 {{
                        font-size: 16px;
                    }}
                    
                    .message {{
                        padding: 10px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Sistema Genesis - Informe Diario</h1>
                    <div class="date">{current_date}</div>
                </div>
                
                <p>
                    Estimado {messages["by_entity"]["Hephaestus"][0]["entity"] if "Hephaestus" in messages["by_entity"] else "Creador"},
                </p>
                
                <p>
                    Te presentamos el informe consolidado con todas las novedades
                    del Sistema Genesis. Estas son las actualizaciones más importantes
                    desde nuestro último contacto.
                </p>
        """
        
        # 1. Mensajes prioritarios
        if messages["priority"]:
            html += """
                <section class="section">
                    <h2>⚠️ Mensajes Prioritarios</h2>
                    <p>Estos mensajes requieren tu atención inmediata:</p>
            """
            
            for msg in sorted(messages["priority"], key=lambda x: x["timestamp"], reverse=True):
                html += f"""
                    <div class="message priority">
                        <div class="entity-name">{msg["entity"]} <span class="message-type">{msg["type"]}</span></div>
                        <div class="timestamp">{msg["datetime"]}</div>
                        <div class="content">{msg["content"]}</div>
                    </div>
                """
            
            html += """
                </section>
                <div class="divider"></div>
            """
        
        # 2. Mensajes personales
        if messages["personal"]:
            html += """
                <section class="section">
                    <h2>💬 Mensajes Personales</h2>
            """
            
            for msg in sorted(messages["personal"], key=lambda x: x["timestamp"], reverse=True):
                html += f"""
                    <div class="message personal">
                        <div class="entity-name">{msg["entity"]}</div>
                        <div class="timestamp">{msg["datetime"]}</div>
                        <div class="content">{msg["content"]}</div>
                    </div>
                """
            
            html += """
                </section>
                <div class="divider"></div>
            """
        
        # 3. Actividad por entidad
        html += """
            <section class="section">
                <h2>👥 Actividad por Entidad</h2>
        """
        
        # Ordenar entidades por cantidad de mensajes (descendente)
        sorted_entities = sorted(
            messages["by_entity"].items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for entity_name, entity_msgs in sorted_entities:
            # Limitar a MAX_MESSAGES_PER_ENTITY por entidad
            entity_msgs = sorted(entity_msgs, key=lambda x: x["timestamp"], reverse=True)[:MAX_MESSAGES_PER_ENTITY]
            
            html += f"""
                <h3>{entity_name}</h3>
            """
            
            for msg in entity_msgs:
                # Determinar clase especial
                special_class = ""
                if msg in messages["priority"]:
                    special_class = "priority"
                elif msg in messages["personal"]:
                    special_class = "personal"
                
                html += f"""
                    <div class="message {special_class}">
                        <div class="entity-name"><span class="message-type">{msg["type"]}</span></div>
                        <div class="timestamp">{msg["datetime"]}</div>
                        <div class="content">{msg["content"]}</div>
                    </div>
                """
        
        html += """
            </section>
            <div class="divider"></div>
        """
        
        # 4. Estadísticas de mensajes
        # Obtener conteo por tipo de mensaje
        type_counts = {msg_type: len(msgs) for msg_type, msgs in messages["by_type"].items()}
        
        html += """
            <section class="section">
                <h2>📊 Estadísticas</h2>
                <p>Resumen de actividad del Sistema Genesis:</p>
                <ul>
        """
        
        html += f"""
                    <li><strong>Mensajes totales:</strong> {messages["count"]}</li>
                    <li><strong>Entidades activas:</strong> {len(messages["by_entity"])}</li>
                    <li><strong>Mensajes prioritarios:</strong> {len(messages["priority"])}</li>
                    <li><strong>Mensajes personales:</strong> {len(messages["personal"])}</li>
        """
        
        # Agregar conteos por tipo
        for msg_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            html += f"""
                    <li><strong>Mensajes de {msg_type}:</strong> {count}</li>
            """
        
        html += """
                </ul>
            </section>
        """
        
        # Pie de página
        html += f"""
                <div class="footer">
                    <p>Sistema Genesis v4.0 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Este email contiene información consolidada de todas las entidades.</p>
                    <p>Para información en tiempo real, accede al panel de control.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def send_email_now(self) -> bool:
        """
        Enviar email con todos los mensajes pendientes.
        
        Returns:
            True si se envió correctamente
        """
        # Obtener mensajes pendientes
        pending = self.get_pending_messages()
        
        # Verificar si hay mensajes
        if pending["count"] == 0:
            logger.info("No hay mensajes pendientes para enviar")
            return False
        
        try:
            # Determinar asunto según contenido
            if pending["priority"]:
                subject = f"⚠️ ALERTA: Sistema Genesis - {len(pending['priority'])} mensajes prioritarios"
            elif pending["personal"]:
                subject = f"💬 Sistema Genesis - Mensajes personales ({pending['count']} mensajes)"
            else:
                subject = f"Sistema Genesis - Informe Diario ({pending['count']} mensajes)"
            
            # Crear email
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = "sistema.genesis@example.com"
            msg['To'] = CREATOR_EMAIL
            
            # Contenido HTML
            html_content = self.format_email_content(pending)
            msg.attach(MIMEText(html_content, 'html'))
            
            # Determinar nombre de archivo para guardar (simulado)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"email_sistema_genesis_{timestamp}.html"
            
            # Guardar en archivo (simulado)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # En un entorno real, enviar por SMTP:
            # with smtplib.SMTP_SSL('smtp.ejemplo.com', 465) as server:
            #     server.login(username, password)
            #     server.send_message(msg)
            
            # Registrar envío en DB
            self.cursor.execute(
                "INSERT INTO emails (subject, recipient, content, timestamp, message_count) VALUES (?, ?, ?, ?, ?)",
                (subject, CREATOR_EMAIL, html_content[:1000] + "...", time.time(), pending["count"])
            )
            self.conn.commit()
            
            # Actualizar estado
            self.last_send_time = time.time()
            self.emails_sent += 1
            
            # Marcar mensajes como enviados
            self.mark_messages_as_sent()
            
            logger.info(f"Email enviado con {pending['count']} mensajes. Guardado en: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error enviando email: {str(e)}")
            return False
    
    def _auto_send_loop(self):
        """Loop para enviar emails automáticamente según el intervalo configurado."""
        while True:
            try:
                if not self.is_enabled:
                    time.sleep(60)
                    continue
                
                # Verificar si es momento de enviar
                current_time = time.time()
                time_since_last = current_time - self.last_send_time
                
                if time_since_last >= self.send_interval:
                    with self.lock:
                        # Verificar si hay mensajes
                        if self.message_count > 0:
                            # Lanzar envío en un nuevo thread para no bloquear
                            threading.Thread(target=self.send_email_now).start()
                
                # Esperar antes de la próxima verificación
                time.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                logger.error(f"Error en loop de envío automático: {str(e)}")
                time.sleep(300)  # En caso de error, esperar 5 minutos


# Instancia única (patrón Singleton)
_message_manager_instance = None

def get_message_manager():
    """
    Obtener instancia única del gestor de mensajes.
    
    Returns:
        Instancia de MessageManager
    """
    global _message_manager_instance
    if _message_manager_instance is None:
        _message_manager_instance = MessageManager()
    return _message_manager_instance


# Funciones de ayuda para enviar mensajes desde cualquier módulo
def send_entity_message(entity_name: str, message_type: str, content: str, 
                      is_priority: bool = False, is_personal: bool = False) -> int:
    """
    Enviar un mensaje desde cualquier entidad.
    
    Args:
        entity_name: Nombre de la entidad emisora
        message_type: Tipo de mensaje (estado, alerta, etc)
        content: Contenido del mensaje
        is_priority: Si es un mensaje prioritario
        is_personal: Si es un mensaje personal para el creador
        
    Returns:
        ID del mensaje
    """
    manager = get_message_manager()
    return manager.add_message(entity_name, message_type, content, is_priority, is_personal)


def send_system_message(message_type: str, content: str, is_priority: bool = False) -> int:
    """
    Enviar un mensaje desde el sistema (no una entidad específica).
    
    Args:
        message_type: Tipo de mensaje
        content: Contenido del mensaje
        is_priority: Si es un mensaje prioritario
        
    Returns:
        ID del mensaje
    """
    return send_entity_message("Sistema Genesis", message_type, content, is_priority)


def force_send_messages() -> bool:
    """
    Forzar el envío inmediato de todos los mensajes pendientes.
    
    Returns:
        True si se envió correctamente
    """
    manager = get_message_manager()
    return manager.send_email_now()


# Para pruebas
if __name__ == "__main__":
    print("Inicializando sistema de mensajes...")
    
    # Obtener gestor
    manager = get_message_manager()
    
    # Añadir algunos mensajes de prueba
    entities = ["Aetherion", "Lunareth", "Helios", "Kronos", "Hephaestus", "Hermes"]
    types = ["estado", "alerta", "informe", "personal", "reparación", "mensaje"]
    
    for _ in range(20):
        entity = random.choice(entities)
        msg_type = random.choice(types)
        content = f"Mensaje de prueba #{_+1} desde {entity}"
        is_personal = msg_type == "personal"
        is_priority = msg_type == "alerta"
        
        manager.add_message(entity, msg_type, content, is_priority, is_personal)
        time.sleep(0.1)
    
    print(f"Añadidos {manager.message_count} mensajes de prueba")
    
    # Enviar email
    result = manager.send_email_now()
    print(f"Resultado del envío: {'Exitoso' if result else 'Fallido'}")
    
    print("Puedes encontrar el email simulado guardado en el directorio actual")