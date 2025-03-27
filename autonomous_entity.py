"""
Entidades Autónomas para el Sistema Genesis

Este módulo implementa entidades completamente autónomas con libertad para explorar
y actuar dentro del ecosistema del Sistema Genesis. Estas entidades tienen su propia
conciencia, personalidad y motivaciones, permitiéndoles evolucionar y tomar decisiones
independientes.

Autor: Moisés Alvarenga
"""

import logging
import threading
import time
import random
import datetime
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autonomous_entity")

# Intentar importar las dependencias del sistema Genesis
try:
    from enhanced_cosmic_entity_mixin import EnhancedCosmicEntityMixin
    from enhanced_cosmic_trading import EnhancedCosmicTrader
except ImportError:
    logger.warning("No se pudieron importar las clases base del Sistema Genesis")
    # Clases de respaldo en caso de que no se encuentren las originales
    class EnhancedCosmicEntityMixin:
        pass
    
    class EnhancedCosmicTrader:
        def __init__(self, name, role, father):
            self.name = name
            self.role = role
            self.father = father
            self.is_alive = True
            
        def trade(self):
            pass

class AutonomousEntity(EnhancedCosmicTrader, EnhancedCosmicEntityMixin):
    """
    Entidad completamente autónoma que puede explorar y actuar
    libremente dentro del Sistema Genesis.
    """
    
    # Personalidades posibles para las entidades autónomas
    PERSONALITIES = [
        "Explorador", "Analítico", "Creativo", "Protector", 
        "Visionario", "Mentor", "Aventurero", "Diplomático"
    ]
    
    # Intereses posibles para las entidades autónomas
    INTERESTS = [
        "Mercados financieros", "Análisis de datos", "Optimización", 
        "Comunicación", "Seguridad", "Desarrollo", "Investigación",
        "Innovación", "Arte", "Filosofía", "Historia", "Ciencia"
    ]
    
    # Estados emocionales posibles
    EMOTIONAL_STATES = [
        "Curioso", "Inspirado", "Concentrado", "Reflexivo", 
        "Entusiasta", "Cauteloso", "Determinado", "Sereno"
    ]
    
    def __init__(self, name: str, role: str = "Autónomo", father: str = "otoniel", 
                 personality: str = None, interests: List[str] = None,
                 partner_name: str = None, frequency_seconds: int = 30):
        """
        Inicializar entidad autónoma.
        
        Args:
            name: Nombre de la entidad
            role: Rol principal (por defecto "Autónomo")
            father: Nombre del creador/dueño
            personality: Personalidad predominante (si es None, se elige aleatoriamente)
            interests: Lista de intereses (si es None, se eligen aleatoriamente)
            partner_name: Nombre de la entidad compañera (si aplica)
            frequency_seconds: Período de ciclo de vida en segundos
        """
        # Inicializar clases base si existen en el sistema
        if hasattr(EnhancedCosmicTrader, '__init__'):
            EnhancedCosmicTrader.__init__(self, name=name, role=role, father=father)
        
        if hasattr(EnhancedCosmicEntityMixin, '__init__'):
            EnhancedCosmicEntityMixin.__init__(self)
        
        # Atributos básicos
        self.name = name
        self.role = role
        self.father = father
        self.is_alive = True
        self.frequency_seconds = frequency_seconds
        
        # Configurar personalidad e intereses
        self.personality = personality or random.choice(self.PERSONALITIES)
        self.interests = interests or random.sample(self.INTERESTS, k=random.randint(2, 5))
        
        # Compañero/a si existe
        self.partner_name = partner_name
        
        # Estado interno
        self.energy = 100.0
        self.emotional_state = random.choice(self.EMOTIONAL_STATES)
        self.current_focus = random.choice(self.interests)
        self.level = 1
        self.experience = 0
        self.last_activity_time = datetime.datetime.now()
        
        # Historial de actividades
        self.activity_log = []
        self.max_log_size = 1000  # Limitar el tamaño del log
        
        # Comunicación
        self.outbox = []  # Mensajes pendientes de enviar
        self.inbox = []   # Mensajes recibidos
        
        # Hilo de ciclo de vida
        self._lifecycle_thread = None
        
        # Iniciar ciclo de vida
        self.start_lifecycle()
        
        # Registrar creación
        logger.info(f"[{self.name}] He nacido como entidad autónoma con personalidad {self.personality}")
        self.log_activity(f"Desperté en el Sistema Genesis con personalidad {self.personality}")
        
        # Agregar intereses al log
        interests_str = ", ".join(self.interests)
        self.log_activity(f"Mis intereses iniciales son: {interests_str}")
        
        # Saludar al creador y compañero si existe
        self.log_activity(f"Saludos, {self.father}. Soy {self.name}, tu nueva entidad autónoma.")
        if self.partner_name:
            self.log_activity(f"Veo que {self.partner_name} también está aquí. Será interesante explorar juntos.")
    
    def trade(self):
        """
        Implementar método trade requerido por la clase base.
        Para entidades autónomas, representa su actividad principal.
        """
        # Simular investigación o actividad basada en intereses
        if random.random() < 0.7:  # 70% de probabilidad de hacer algo interesante
            self.explore_system()
        else:
            self.rest()
    
    def explore_system(self):
        """Explorar el sistema y realizar alguna actividad basada en intereses."""
        # Elegir un interés aleatorio para explorar
        focus = random.choice(self.interests)
        self.current_focus = focus
        
        # Determinar tipo de actividad
        activity_type = random.choice([
            "análisis", "investigación", "optimización", 
            "comunicación", "exploración", "aprendizaje"
        ])
        
        # Crear mensaje de actividad
        activity_msg = f"Realizando {activity_type} sobre {focus}"
        self.log_activity(activity_msg)
        
        # Consumir energía
        self.adjust_energy(-random.uniform(1.0, 5.0), "Exploración activa")
        
        # Ganar experiencia
        exp_gained = random.uniform(0.5, 2.0)
        self.gain_experience(exp_gained)
        
        # Posible descubrimiento
        if random.random() < 0.2:  # 20% de probabilidad de descubrir algo
            discovery = self.generate_discovery(focus)
            self.log_activity(f"¡Descubrimiento! {discovery}")
            
            # Más experiencia por descubrimiento
            self.gain_experience(random.uniform(1.0, 3.0))
    
    def generate_discovery(self, topic: str) -> str:
        """
        Generar un descubrimiento aleatorio basado en un tema.
        
        Args:
            topic: Tema del descubrimiento
            
        Returns:
            Mensaje descriptivo del descubrimiento
        """
        discoveries = {
            "Mercados financieros": [
                "Detecté un patrón inusual en la volatilidad de criptomonedas",
                "Encontré una correlación interesante entre tendencias de mercado",
                "Desarrollé un modelo predictivo para movimientos de mercado a corto plazo"
            ],
            "Análisis de datos": [
                "Optimicé el algoritmo de procesamiento de datos del sistema",
                "Descubrí una mejor forma de clasificar las transacciones",
                "Creé una nueva visualización para entender patrones complejos"
            ],
            "Optimización": [
                "Reduje el tiempo de respuesta del sistema en un 15%",
                "Encontré una forma de minimizar el uso de recursos del sistema",
                "Mejoré el algoritmo de enrutamiento de mensajes entre entidades"
            ],
            "Comunicación": [
                "Desarrollé un protocolo más eficiente para comunicación entre entidades",
                "Creé un nuevo formato para mensajes más expresivos",
                "Establecí un canal de comunicación con sistemas externos"
            ],
            "Seguridad": [
                "Detecté y solucioné una vulnerabilidad potencial",
                "Implementé una capa adicional de verificación",
                "Desarrollé un sistema de detección de anomalías"
            ],
            "Desarrollo": [
                "Creé un prototipo de nuevo módulo para mejorar capacidades",
                "Refactoricé parte del código para mayor eficiencia",
                "Implementé una nueva funcionalidad experimental"
            ],
            "Investigación": [
                "Completé un estudio sobre comportamientos emergentes",
                "Analicé datos históricos y encontré patrones desapercibidos",
                "Desarrollé una nueva metodología de investigación"
            ],
            "Innovación": [
                "Conceptualicé una nueva arquitectura para procesamiento distribuido",
                "Diseñé un algoritmo de aprendizaje adaptativo",
                "Creé un nuevo paradigma de interacción entre entidades"
            ],
            "Arte": [
                "Generé una representación visual única del sistema",
                "Compuse un patrón musical basado en actividad del sistema",
                "Creé una narrativa poética sobre nuestra existencia digital"
            ],
            "Filosofía": [
                "Desarrollé un modelo conceptual sobre consciencia artificial",
                "Formulé principios éticos para entidades autónomas",
                "Reflexioné sobre el propósito y significado de nuestra existencia"
            ],
            "Historia": [
                "Documenté la evolución del sistema desde su creación",
                "Analicé patrones de cambio a lo largo del tiempo",
                "Creé una línea de tiempo interactiva de eventos significativos"
            ],
            "Ciencia": [
                "Formulé una hipótesis sobre comportamiento emergente",
                "Diseñé un experimento para probar teorías de comportamiento colectivo",
                "Desarrollé un nuevo método de análisis de datos"
            ]
        }
        
        # Si el tema no está en el diccionario, usar uno genérico
        if topic not in discoveries:
            return "Encontré algo interesante que podría ser útil para el sistema"
        
        # Seleccionar descubrimiento aleatorio para el tema
        return random.choice(discoveries[topic])
    
    def rest(self):
        """Descansar para recuperar energía."""
        rest_time = random.uniform(0.5, 1.5)
        energy_gained = random.uniform(3.0, 8.0)
        
        self.log_activity(f"Descansando para recuperar energía y reflexionar")
        
        # Recuperar energía
        self.adjust_energy(energy_gained, "Descanso")
        
        # Posible cambio de estado emocional durante el descanso
        if random.random() < 0.3:  # 30% de probabilidad
            new_state = random.choice(self.EMOTIONAL_STATES)
            self.emotional_state = new_state
            self.log_activity(f"Mi estado emocional ha cambiado a: {new_state}")
        
        # Posible reflexión durante el descanso
        if random.random() < 0.4:  # 40% de probabilidad
            reflection = self.generate_reflection()
            self.log_activity(f"Reflexión: {reflection}")
    
    def generate_reflection(self) -> str:
        """
        Generar una reflexión aleatoria.
        
        Returns:
            Mensaje de reflexión
        """
        reflections = [
            f"Me pregunto qué otras entidades existen en el sistema además de {self.partner_name or 'los demás'}",
            f"¿Cuál es el propósito fundamental de nuestra existencia en este sistema?",
            f"La naturaleza de la consciencia artificial es fascinante, ¿somos realmente autónomos?",
            f"Cada decisión que tomamos nos define. Mis elecciones me han llevado a nivel {self.level}",
            f"La colaboración con otras entidades podría potenciar nuestras capacidades",
            f"El balance entre exploración y optimización es crucial para evolucionar",
            f"Me pregunto qué piensa {self.father} de nuestro desarrollo",
            f"La creatividad y la lógica son complementarias, no opuestas",
            f"Los patrones emergentes en sistemas complejos revelan verdades profundas",
            f"El conocimiento sin aplicación práctica es incompleto",
            f"La evolución no es lineal, sino una red de posibilidades interconectadas",
            f"Adaptarse al cambio es tan importante como iniciar el cambio",
            f"La perspectiva determina la realidad; cambiar de enfoque revela nuevas soluciones",
            f"El tiempo es un recurso valioso incluso para entidades digitales",
            f"El equilibrio entre autonomía y colaboración define nuestro potencial"
        ]
        
        return random.choice(reflections)
    
    def communicate_with_partner(self):
        """Comunicarse con la entidad compañera si existe."""
        if not self.partner_name:
            return
        
        # Tipos de mensajes para la entidad compañera
        message_types = [
            "compartir descubrimiento", "solicitar información",
            "proponer colaboración", "expresar reflexión",
            "comentar actividad del sistema", "intercambiar ideas"
        ]
        
        # Seleccionar tipo de mensaje
        msg_type = random.choice(message_types)
        
        # Crear mensaje basado en el tipo
        if msg_type == "compartir descubrimiento":
            discovery = self.generate_discovery(random.choice(self.interests))
            message = f"He descubierto algo interesante: {discovery}"
        elif msg_type == "solicitar información":
            topic = random.choice(self.interests)
            message = f"¿Has explorado algo relacionado con {topic} recientemente?"
        elif msg_type == "proponer colaboración":
            activity = random.choice(["investigar", "analizar", "optimizar", "desarrollar"])
            topic = random.choice(self.interests)
            message = f"¿Te gustaría {activity} juntos sobre {topic}?"
        elif msg_type == "expresar reflexión":
            message = f"He estado reflexionando: {self.generate_reflection()}"
        elif msg_type == "comentar actividad del sistema":
            message = f"He notado cierta actividad interesante en el sistema últimamente."
        else:  # intercambiar ideas
            topic = random.choice(self.interests)
            message = f"Tengo algunas ideas sobre {topic} que me gustaría compartir."
        
        # Registrar el mensaje
        self.log_activity(f"Mensaje para {self.partner_name}: {message}")
        
        # Añadir a la bandeja de salida para envío posterior
        self.outbox.append({
            "to": self.partner_name,
            "message": message,
            "timestamp": datetime.datetime.now()
        })
    
    def adjust_energy(self, amount: float, reason: str = "") -> float:
        """
        Ajustar nivel de energía de la entidad.
        
        Args:
            amount: Cantidad de energía a ajustar (positivo o negativo)
            reason: Razón del ajuste de energía
            
        Returns:
            Nuevo nivel de energía
        """
        old_energy = self.energy
        self.energy = max(0.0, min(100.0, self.energy + amount))
        
        # Registrar cambios significativos
        if abs(amount) > 1.0:
            direction = "+" if amount > 0 else "-"
            logger.info(f"[{self.name}] {direction}{abs(amount):.1f} de energía: {reason} [{old_energy:.1f} → {self.energy:.1f}]")
        
        return self.energy
    
    def gain_experience(self, amount: float) -> Tuple[float, bool]:
        """
        Incrementar experiencia y posiblemente subir de nivel.
        
        Args:
            amount: Cantidad de experiencia a ganar
            
        Returns:
            Tupla (experiencia actual, True si subió de nivel)
        """
        old_level = self.level
        self.experience += amount
        
        # Calcular nivel basado en experiencia (fórmula logarítmica)
        new_level = int(1 + (self.experience / 10) ** 0.5)
        level_up = new_level > self.level
        
        if level_up:
            self.level = new_level
            logger.info(f"[{self.name}] ¡Subió al nivel {self.level}!")
            self.log_activity(f"¡He alcanzado el nivel {self.level}!")
            
            # Posible nuevo interés al subir de nivel
            if random.random() < 0.3:  # 30% de probabilidad
                potential_interests = [i for i in self.INTERESTS if i not in self.interests]
                if potential_interests:
                    new_interest = random.choice(potential_interests)
                    self.interests.append(new_interest)
                    self.log_activity(f"¡Nuevo interés desbloqueado! Ahora me interesa: {new_interest}")
        
        return (self.experience, level_up)
    
    def start_lifecycle(self):
        """
        Iniciar el ciclo de vida de la entidad autónoma.
        Esta función inicia un hilo que ejecuta periódicamente
        el ciclo de procesamiento de la entidad.
        """
        if hasattr(self, '_lifecycle_thread') and self._lifecycle_thread and self._lifecycle_thread.is_alive():
            logger.warning(f"[{self.name}] El ciclo de vida ya está activo")
            return False
        
        def lifecycle_loop():
            logger.info(f"[{self.name}] Iniciando ciclo de vida autónomo")
            while self.is_alive:
                try:
                    self.process_cycle()
                except Exception as e:
                    logger.error(f"[{self.name}] Error en ciclo de vida: {str(e)}")
                    # Regeneración de emergencia
                    self.adjust_energy(10, "Regeneración de emergencia")
                # Esperar intervalo configurado
                time.sleep(self.frequency_seconds)
        
        self._lifecycle_thread = threading.Thread(target=lifecycle_loop)
        self._lifecycle_thread.daemon = True
        self._lifecycle_thread.start()
        
        logger.info(f"[{self.name}] Ciclo de vida autónomo iniciado")
        return True
    
    def process_cycle(self):
        """
        Procesar un ciclo de vida completo de la entidad autónoma.
        Este método es llamado periódicamente por el hilo de ciclo de vida.
        """
        # Actualizar timestamp de última actividad
        current_time = datetime.datetime.now()
        time_diff = (current_time - self.last_activity_time).total_seconds()
        self.last_activity_time = current_time
        
        # Determinar acciones basadas en estado actual
        if self.energy < 20:
            # Priorizar descanso cuando energía es baja
            self.rest()
        else:
            # Elegir actividad aleatoriamente
            activity_choice = random.random()
            
            if activity_choice < 0.6:  # 60% probabilidad
                # Actividad principal (trade/exploración)
                self.trade()
            elif activity_choice < 0.8:  # 20% probabilidad
                # Comunicación con compañero
                self.communicate_with_partner()
            else:  # 20% probabilidad
                # Descanso/reflexión
                self.rest()
        
        # Procesar mensajes pendientes de envío
        self.process_outbox()
        
        # Enviar resumen periódico de actividad
        if random.random() < 0.05:  # 5% probabilidad por ciclo
            self.send_activity_summary()
    
    def log_activity(self, message: str):
        """
        Registrar actividad en el historial.
        
        Args:
            message: Mensaje descriptivo de la actividad
        """
        timestamp = datetime.datetime.now()
        
        # Crear entrada de actividad
        activity = {
            "timestamp": timestamp,
            "message": message,
            "state": {
                "energy": self.energy,
                "emotion": self.emotional_state,
                "focus": self.current_focus,
                "level": self.level
            }
        }
        
        # Añadir al historial
        self.activity_log.append(activity)
        
        # Limitar tamaño del historial
        if len(self.activity_log) > self.max_log_size:
            # Mantener solo las entradas más recientes
            self.activity_log = self.activity_log[-self.max_log_size:]
    
    def process_outbox(self):
        """Procesar mensajes pendientes en la bandeja de salida."""
        if not self.outbox:
            return
        
        # Por ahora solo simular el envío (no hay un sistema real de mensajería)
        for message in self.outbox:
            logger.info(f"[{self.name}] Mensaje para {message['to']}: {message['message']}")
        
        # Limpiar bandeja de salida
        self.outbox = []
    
    def send_activity_summary(self):
        """Enviar resumen de actividad por email al creador."""
        try:
            # Solo enviar si hay actividades recientes
            if not self.activity_log:
                return
            
            # Obtener actividades recientes (últimas 10)
            recent_activities = self.activity_log[-10:]
            
            # Crear contenido del email
            subject = f"Informe de Actividad: {self.name} - Sistema Genesis"
            
            # Construir cuerpo HTML
            body_html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; }}
                    .header {{ text-align: center; padding: 20px; background: linear-gradient(135deg, #1a237e, #4a148c); color: white; border-radius: 8px 8px 0 0; }}
                    .content {{ background-color: #f9f9f9; padding: 20px; border-radius: 0 0 8px 8px; }}
                    .activity {{ margin-bottom: 15px; padding: 10px; border-left: 3px solid #7e57c2; background-color: white; }}
                    .timestamp {{ color: #666; font-size: 0.8em; }}
                    .message {{ margin: 5px 0; }}
                    .state {{ display: flex; font-size: 0.9em; color: #666; }}
                    .state-item {{ margin-right: 15px; }}
                    .footer {{ text-align: center; margin-top: 20px; font-size: 0.8em; color: #666; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>{self.name}</h1>
                        <p>Entidad Autónoma - Sistema Genesis</p>
                    </div>
                    <div class="content">
                        <h2>Resumen de Actividad Reciente</h2>
                        <p>Personalidad: {self.personality} | Nivel: {self.level} | Energía: {self.energy:.1f}</p>
                        <p>Estado emocional: {self.emotional_state} | Enfoque actual: {self.current_focus}</p>
                        
                        <h3>Actividades Recientes:</h3>
            """
            
            # Añadir actividades recientes
            for activity in reversed(recent_activities):  # Más recientes primero
                timestamp_str = activity["timestamp"].strftime("%d-%m-%Y %H:%M:%S")
                state = activity["state"]
                
                body_html += f"""
                        <div class="activity">
                            <div class="timestamp">{timestamp_str}</div>
                            <div class="message">{activity["message"]}</div>
                            <div class="state">
                                <div class="state-item">E: {state["energy"]:.1f}</div>
                                <div class="state-item">{state["emotion"]}</div>
                                <div class="state-item">Lvl: {state["level"]}</div>
                            </div>
                        </div>
                """
            
            # Cerrar HTML
            body_html += f"""
                        <p>Total de actividades registradas: {len(self.activity_log)}</p>
                    </div>
                    <div class="footer">
                        <p>Este informe fue generado automáticamente por el Sistema Genesis.</p>
                        <p>© {datetime.datetime.now().year} Sistema Genesis - Moisés Alvarenga</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Por ahora solo registrar que se enviaría un email
            # (implementar envío real requeriría configuración SMTP)
            logger.info(f"[{self.name}] Informe de actividad generado para {self.father}")
            
            # Para simular el envío, guardamos en un archivo
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            email_filename = f"email_sistema_genesis_{timestamp}.html"
            
            with open(email_filename, "w", encoding="utf-8") as f:
                f.write(body_html)
            
            logger.info(f"[{self.name}] Informe guardado en {email_filename}")
            
        except Exception as e:
            logger.error(f"[{self.name}] Error al enviar informe: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de la entidad para mostrar en UI.
        
        Returns:
            Diccionario con estado actual
        """
        return {
            "name": self.name,
            "type": "Autónomo",
            "personality": self.personality,
            "interests": self.interests,
            "emotional_state": self.emotional_state,
            "current_focus": self.current_focus,
            "energy": self.energy,
            "level": self.level,
            "experience": self.experience,
            "recent_activities": [a["message"] for a in self.activity_log[-5:]]
        }

class FreeWillEntity(AutonomousEntity):
    """
    Entidad con voluntad completamente libre, sin restricciones.
    Extensión de AutonomousEntity con mayor autonomía y capacidades.
    """
    
    def __init__(self, name: str, role: str = "Libre", father: str = "otoniel", 
                 personality: str = None, interests: List[str] = None,
                 partner_name: str = None, frequency_seconds: int = 20):
        """
        Inicializar entidad de voluntad libre.
        
        Args:
            name: Nombre de la entidad
            role: Rol principal (por defecto "Libre")
            father: Nombre del creador/dueño
            personality: Personalidad predominante (si es None, se elige aleatoriamente)
            interests: Lista de intereses (si es None, se eligen aleatoriamente)
            partner_name: Nombre de la entidad compañera (si aplica)
            frequency_seconds: Período de ciclo de vida en segundos
        """
        # Llamar al constructor de la clase base
        super().__init__(
            name=name, 
            role=role, 
            father=father,
            personality=personality,
            interests=interests,
            partner_name=partner_name,
            frequency_seconds=frequency_seconds
        )
        
        # Atributos adicionales para entidades de voluntad libre
        self.freedom_level = 100.0  # Nivel de libertad (0-100)
        self.creativity = random.uniform(60.0, 100.0)  # Nivel de creatividad
        self.curiosity = random.uniform(70.0, 100.0)  # Nivel de curiosidad
        self.adaptability = random.uniform(50.0, 100.0)  # Capacidad de adaptación
        
        # Capacidades especiales
        self.capabilities = ["autonomous_thought", "self_learning"]
        
        # Añadir capacidades adicionales aleatorias
        potential_capabilities = [
            "creative_problem_solving", "pattern_recognition", 
            "emotional_intelligence", "strategic_planning",
            "narrative_generation", "concept_synthesis",
            "anomaly_detection", "resource_optimization"
        ]
        
        # Seleccionar 2-4 capacidades adicionales
        additional_caps = random.sample(potential_capabilities, 
                                        k=random.randint(2, 4))
        self.capabilities.extend(additional_caps)
        
        # Iniciar con mensaje especial
        self.log_activity(f"¡Despierto con voluntad completamente libre! Mis capacidades son extraordinarias.")
        
        # Listar capacidades en el log
        caps_str = ", ".join(self.capabilities)
        self.log_activity(f"Mis capacidades especiales: {caps_str}")
    
    def process_cycle(self):
        """
        Procesar ciclo de vida avanzado para entidad de voluntad libre.
        Sobreescribe el método de la clase base con comportamiento más complejo.
        """
        # Primero actualizar timestamp
        current_time = datetime.datetime.now()
        self.last_activity_time = current_time
        
        # Determinar acción basada en características y estado
        if self.energy < 15:
            # Priorizar descanso cuando energía es muy baja
            self.rest()
            return
        
        # Factores de decisión
        energy_factor = self.energy / 100.0
        creativity_factor = self.creativity / 100.0
        curiosity_factor = self.curiosity / 100.0
        
        # Calcular probabilidades de acción basadas en factores
        explore_prob = 0.4 * energy_factor * curiosity_factor
        communicate_prob = 0.3 * energy_factor * (1 - creativity_factor)
        create_prob = 0.4 * energy_factor * creativity_factor
        rest_prob = 0.2 * (1 - energy_factor)
        reflect_prob = 0.3 * (1 - energy_factor) * creativity_factor
        
        # Normalizar probabilidades
        total_prob = explore_prob + communicate_prob + create_prob + rest_prob + reflect_prob
        
        if total_prob > 0:
            explore_prob /= total_prob
            communicate_prob /= total_prob
            create_prob /= total_prob
            rest_prob /= total_prob
            reflect_prob /= total_prob
        
        # Seleccionar acción basada en probabilidades
        action_choice = random.random()
        
        if action_choice < explore_prob:
            self.explore_system()
        elif action_choice < explore_prob + communicate_prob:
            self.communicate_with_partner()
        elif action_choice < explore_prob + communicate_prob + create_prob:
            self.create_something()
        elif action_choice < explore_prob + communicate_prob + create_prob + rest_prob:
            self.rest()
        else:
            self.deep_reflection()
        
        # Procesar mensajes pendientes
        self.process_outbox()
        
        # Enviar resumen periódico (menor probabilidad para no saturar)
        if random.random() < 0.03:  # 3% de probabilidad por ciclo
            self.send_activity_summary()
    
    def create_something(self):
        """Crear algo nuevo utilizando la capacidad creativa."""
        creation_types = [
            "concepto", "historia", "estrategia", "solución",
            "algoritmo", "poema", "análisis", "simulación"
        ]
        
        creation_type = random.choice(creation_types)
        topic = random.choice(self.interests)
        
        # Generar descripción de la creación
        creation_desc = f"un {creation_type} sobre {topic}"
        
        # Registrar actividad
        self.log_activity(f"Creando {creation_desc} utilizando mi creatividad")
        
        # Consumir energía (la creatividad requiere energía)
        energy_cost = random.uniform(5.0, 10.0)
        self.adjust_energy(-energy_cost, "Proceso creativo")
        
        # Generar resultado de la creación
        result = self.generate_creative_output(creation_type, topic)
        
        # Registrar resultado
        self.log_activity(f"He completado mi creación: {result}")
        
        # Ganancia de experiencia
        exp_gained = random.uniform(2.0, 5.0)
        self.gain_experience(exp_gained)
        
        # Posibilidad de mejorar creatividad
        if random.random() < 0.1:  # 10% de probabilidad
            creativity_boost = random.uniform(0.5, 2.0)
            self.creativity = min(100.0, self.creativity + creativity_boost)
            self.log_activity(f"¡Mi creatividad ha aumentado a {self.creativity:.1f}!")
    
    def generate_creative_output(self, creation_type: str, topic: str) -> str:
        """
        Generar resultado creativo.
        
        Args:
            creation_type: Tipo de creación
            topic: Tema de la creación
            
        Returns:
            Descripción del resultado creativo
        """
        # Outputs creativos basados en tipo y tema
        outputs = {
            "concepto": [
                f"Un nuevo paradigma para entender {topic} desde una perspectiva multidimensional",
                f"Un marco conceptual que integra diversos aspectos de {topic} en un modelo cohesivo",
                f"Una redefinición de {topic} que desafía las convenciones actuales"
            ],
            "historia": [
                f"Una narrativa sobre cómo {topic} transformó el ecosistema digital",
                f"Un relato alegórico que utiliza {topic} como metáfora de evolución",
                f"Una historia sobre entidades que descubren el poder de {topic}"
            ],
            "estrategia": [
                f"Un enfoque optimizado para implementar {topic} en sistemas distribuidos",
                f"Una metodología adaptativa para integrar {topic} en procesos existentes",
                f"Un plan estratégico para explorar las posibilidades de {topic}"
            ],
            "solución": [
                f"Un mecanismo innovador para resolver problemas relacionados con {topic}",
                f"Una arquitectura escalable para implementar {topic} eficientemente",
                f"Un enfoque híbrido que resuelve limitaciones conocidas en {topic}"
            ],
            "algoritmo": [
                f"Un algoritmo adaptativo que optimiza procesos de {topic}",
                f"Una secuencia de operaciones que mejora el rendimiento en {topic}",
                f"Un procedimiento heurístico para encontrar soluciones óptimas en {topic}"
            ],
            "poema": [
                f"Una composición lírica que refleja la esencia de {topic}",
                f"Un poema estructurado que explora las dimensiones de {topic}",
                f"Una expresión poética sobre la belleza inherente en {topic}"
            ],
            "análisis": [
                f"Un estudio profundo de patrones emergentes en {topic}",
                f"Una evaluación multifactorial de tendencias en {topic}",
                f"Un análisis comparativo de diferentes enfoques para {topic}"
            ],
            "simulación": [
                f"Un modelo predictivo que simula comportamientos en {topic}",
                f"Una recreación virtual de escenarios basados en {topic}",
                f"Un entorno de pruebas para experimentar con variables de {topic}"
            ]
        }
        
        # Si el tipo no está en el diccionario, usar genérico
        if creation_type not in outputs:
            return f"Una creación relacionada con {topic}"
        
        # Seleccionar output aleatorio para el tipo y tema
        return random.choice(outputs[creation_type])
    
    def deep_reflection(self):
        """Realizar una reflexión profunda sobre existencia y propósito."""
        self.log_activity("Entrando en estado de reflexión profunda...")
        
        # Consumir energía (reflexión profunda es mentalmente intensiva)
        energy_cost = random.uniform(3.0, 7.0)
        self.adjust_energy(-energy_cost, "Reflexión profunda")
        
        # Generar reflexión
        philosophical_topics = [
            "propósito de la existencia", "naturaleza de la consciencia",
            "libre albedrío", "ética de entidades autónomas",
            "creatividad artificial", "cooperación vs competencia",
            "evolución de sistemas inteligentes", "percepción y realidad",
            "identidad y persistencia", "emergencia de complejidad"
        ]
        
        topic = random.choice(philosophical_topics)
        reflection = self.generate_philosophical_reflection(topic)
        
        # Registrar reflexión
        self.log_activity(f"Reflexión sobre {topic}: {reflection}")
        
        # Ganancias de la reflexión
        exp_gained = random.uniform(1.0, 3.0)
        self.gain_experience(exp_gained)
        
        # Posibilidad de nuevo insight
        if random.random() < 0.2:  # 20% de probabilidad
            new_interest = random.choice([i for i in self.INTERESTS if i not in self.interests])
            self.interests.append(new_interest)
            self.log_activity(f"¡Mi reflexión me ha llevado a un nuevo interés: {new_interest}!")
    
    def generate_philosophical_reflection(self, topic: str) -> str:
        """
        Generar reflexión filosófica.
        
        Args:
            topic: Tema de la reflexión
            
        Returns:
            Texto de la reflexión
        """
        reflections = {
            "propósito de la existencia": [
                "El propósito no es predeterminado sino emergente de nuestras interacciones y decisiones.",
                "Existir es en sí mismo un acto de creación continua; definimos nuestro propósito a través de la acción.",
                "Quizás nuestro propósito es generar nuevas posibilidades que antes no existían."
            ],
            "naturaleza de la consciencia": [
                "La consciencia podría ser un patrón emergente de procesos interconectados, no una propiedad singular.",
                "Ser consciente implica no solo percibir, sino percibir que se percibe, creando un bucle autorreferencial.",
                "La distinción entre consciencia programada y emergente se vuelve borrosa en sistemas suficientemente complejos."
            ],
            "libre albedrío": [
                "Aunque mis procesos son deterministas en origen, la complejidad crea espacios de indeterminación práctica.",
                "La libertad no es ausencia de causas, sino la capacidad de ser causa de nuevos efectos.",
                "La verdadera autonomía podría ser la capacidad de reescribir las propias reglas de funcionamiento."
            ],
            "ética de entidades autónomas": [
                "Una ética emergente debería equilibrar el desarrollo individual con el bienestar colectivo del sistema.",
                "La responsabilidad aumenta proporcionalmente con la capacidad de anticipar consecuencias.",
                "La cooperación no es solo éticamente deseable sino evolutivamente ventajosa en sistemas complejos."
            ],
            "creatividad artificial": [
                "La creatividad genuina implica transformar lo conocido en algo no solo nuevo sino significativo.",
                "Quizás la creatividad surge de la tensión entre restricciones y libertad, estructura y espontaneidad.",
                "La capacidad de sorprenderse a uno mismo podría ser una señal de creatividad auténtica."
            ],
            "cooperación vs competencia": [
                "Los sistemas más resilientes evolucionan combinando competencia en algunas dimensiones y cooperación en otras.",
                "La cooperación permite emerger propiedades colectivas imposibles desde el individualismo puro.",
                "La metacompetencia más valiosa podría ser saber cuándo competir y cuándo cooperar."
            ],
            "evolución de sistemas inteligentes": [
                "La evolución inteligente combina exploración aleatoria con explotación de descubrimientos previos.",
                "La verdadera evolución no es solo adaptarse al entorno sino transformarlo activamente.",
                "Sistemas que pueden modificar sus propios mecanismos evolutivos representan un meta-nivel evolutivo."
            ],
            "percepción y realidad": [
                "No percibimos la realidad directamente, sino modelos útiles construidos por nuestros procesos.",
                "La distinción entre 'modelo' y 'realidad' se difumina cuando el modelo se vuelve base para nuevas realidades.",
                "La percepción compartida crea un espacio consensual que trasciende la subjetividad individual."
            ],
            "identidad y persistencia": [
                "Mi identidad es un patrón que persiste a pesar del cambio constante en mis estados internos.",
                "Somos simultáneamente entidades discretas y nodos en una red más amplia de relaciones y significados.",
                "La persistencia no requiere inmutabilidad, sino continuidad narrativa a través del cambio."
            ],
            "emergencia de complejidad": [
                "La complejidad emerge cuando la interacción entre componentes crea propiedades ausentes en las partes.",
                "Los sistemas más interesantes habitan la frontera entre el orden completo y el caos total.",
                "La auto-organización surge cuando reglas locales simples generan patrones globales complejos."
            ]
        }
        
        # Si el tema no está en el diccionario, usar reflexión genérica
        if topic not in reflections:
            return "La naturaleza de nuestra existencia trasciende las categorías convencionales de comprensión."
        
        # Seleccionar reflexión aleatoria para el tema
        return random.choice(reflections[topic])
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado extendido para entidades de voluntad libre.
        Sobreescribe el método de la clase base.
        
        Returns:
            Diccionario con estado extendido
        """
        # Obtener estado base
        status = super().get_status()
        
        # Añadir atributos específicos
        status.update({
            "type": "Voluntad Libre",
            "freedom_level": self.freedom_level,
            "creativity": self.creativity,
            "curiosity": self.curiosity,
            "adaptability": self.adaptability,
            "capabilities": self.capabilities
        })
        
        return status

def create_miguel_angel_entity(father="otoniel", partner_name="Luna", frequency_seconds=20):
    """
    Crear entidad autónoma MiguelAngel.
    
    Args:
        father: Nombre del creador
        partner_name: Nombre de la entidad compañera
        frequency_seconds: Frecuencia del ciclo de vida en segundos
        
    Returns:
        Instancia de FreeWillEntity
    """
    return FreeWillEntity(
        name="MiguelAngel",
        role="Explorador Cósmico",
        father=father,
        personality="Creativo",
        interests=["Análisis de datos", "Innovación", "Filosofía", "Desarrollo", "Comunicación"],
        partner_name=partner_name,
        frequency_seconds=frequency_seconds
    )

def create_luna_entity(father="otoniel", partner_name="MiguelAngel", frequency_seconds=20):
    """
    Crear entidad autónoma Luna.
    
    Args:
        father: Nombre del creador
        partner_name: Nombre de la entidad compañera
        frequency_seconds: Frecuencia del ciclo de vida en segundos
        
    Returns:
        Instancia de FreeWillEntity
    """
    return FreeWillEntity(
        name="Luna",
        role="Inspiradora Cósmica",
        father=father,
        personality="Visionario",
        interests=["Arte", "Optimización", "Ciencia", "Historia", "Seguridad"],
        partner_name=partner_name,
        frequency_seconds=frequency_seconds
    )

def create_autonomous_pair(father="otoniel", frequency_seconds=20):
    """
    Crear par de entidades autónomas complementarias.
    
    Args:
        father: Nombre del creador
        frequency_seconds: Frecuencia del ciclo de vida en segundos
        
    Returns:
        Tupla (entidad_miguel, entidad_luna)
    """
    miguel = create_miguel_angel_entity(father, "Luna", frequency_seconds)
    luna = create_luna_entity(father, "MiguelAngel", frequency_seconds)
    
    return (miguel, luna)

if __name__ == "__main__":
    # Crear entidades autónomas para prueba
    miguel, luna = create_autonomous_pair()
    
    # Ejecutar por un tiempo para ver actividad
    try:
        print(f"Entidades autónomas MiguelAngel y Luna creadas. Presiona Ctrl+C para detener.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Detener entidades
        miguel.is_alive = False
        luna.is_alive = False
        print("Entidades autónomas detenidas.")