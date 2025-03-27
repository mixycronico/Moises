"""
Conector principal para Sistema Genesis con integración de mensajes y reparación

Este módulo proporciona las interfaces necesarias para integrar el sistema de
mensajes y reparación con todas las entidades existentes. Permite que cualquier
entidad pueda enviar mensajes al creador y beneficiarse del sistema de reparación.

Características:
1. Integración con la cola centralizada de mensajes
2. Conectores para añadir capacidades de reparación a entidades existentes
3. Funciones de ayuda para mensajes personalizados
4. Sistema de monitoreo del estado de las entidades
"""

import os
import time
import random
import logging
import threading
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Importar módulos del sistema
from message_collector import send_entity_message, send_system_message, force_send_messages
from repair_entity import RepairEntity, create_repair_entity

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Email del creador
CREATOR_EMAIL = "mixycronico@aol.com"

class GenesisConnector:
    """
    Conector central para el Sistema Genesis.
    Proporciona funcionalidad compartida para todas las entidades.
    """
    
    def __init__(self):
        """Inicializar conector."""
        self.entities = {}  # Entidades registradas
        self.repair_entity = None  # Entidad de reparación
        self.message_types = {
            "personal": "Mensaje personal para el creador",
            "estado": "Actualización de estado",
            "alerta": "Alerta de sistema",
            "informe": "Informe de actividad",
            "reparación": "Operación de reparación",
            "evolución": "Evolución o cambio",
            "propuesta": "Propuesta de mejora",
            "sabiduría": "Conocimiento compartido",
            "social": "Interacción entre entidades"
        }
        
        self.enabled = True
        self.initialized = False
        
        logger.info("Conector Genesis inicializado")
    
    def initialize(self):
        """Inicializar completamente el conector y sus componentes."""
        if self.initialized:
            logger.warning("El conector ya está inicializado")
            return
        
        # Crear entidad de reparación si no existe
        if not self.repair_entity:
            self.repair_entity = create_repair_entity()
            self.register_entity(self.repair_entity)
        
        # Notificar inicio
        send_system_message("sistema", "Sistema Genesis inicializado completamente con capacidades de reparación y mensajería integradas")
        
        self.initialized = True
        logger.info("Conector Genesis completamente inicializado")
    
    def register_entity(self, entity):
        """
        Registrar una entidad en el sistema.
        
        Args:
            entity: Entidad a registrar
        """
        entity_name = getattr(entity, "name", str(entity))
        self.entities[entity_name] = entity
        logger.info(f"Entidad {entity_name} registrada en el sistema")
        
        # Enviar mensaje informativo
        send_entity_message(
            "Sistema Genesis", 
            "registro", 
            f"Entidad {entity_name} registrada en el sistema"
        )
    
    def create_repair_connector(self, entity):
        """
        Crear un conector de reparación para una entidad específica.
        
        Args:
            entity: Entidad a la que añadir el conector
            
        Returns:
            Objeto conector
        """
        entity_name = getattr(entity, "name", str(entity))
        
        # Crear conector
        repair_connector = EntityRepairConnector(entity_name, entity, self.repair_entity)
        
        # Notificar
        logger.info(f"Conector de reparación creado para {entity_name}")
        
        return repair_connector
    
    def create_message_connector(self, entity):
        """
        Crear un conector de mensajes para una entidad específica.
        
        Args:
            entity: Entidad a la que añadir el conector
            
        Returns:
            Objeto conector
        """
        entity_name = getattr(entity, "name", str(entity))
        
        # Crear conector
        message_connector = EntityMessageConnector(entity_name, entity)
        
        # Notificar
        logger.info(f"Conector de mensajes creado para {entity_name}")
        
        return message_connector
    
    def apply_connectors_to_all_entities(self):
        """Aplicar conectores de reparación y mensajes a todas las entidades registradas."""
        for entity_name, entity in self.entities.items():
            if hasattr(entity, "repair_connector") or hasattr(entity, "message_connector"):
                continue  # Ya tiene conectores
            
            # Añadir conectores
            entity.repair_connector = self.create_repair_connector(entity)
            entity.message_connector = self.create_message_connector(entity)
            
            # Notificar
            logger.info(f"Conectores aplicados a {entity_name}")
        
        # Enviar mensaje
        send_system_message(
            "sistema",
            f"Conectores de reparación y mensajes aplicados a {len(self.entities)} entidades"
        )
    
    def send_consolidated_status(self):
        """Enviar un mensaje consolidado con el estado de todas las entidades."""
        if not self.entities:
            logger.warning("No hay entidades registradas para enviar estado")
            return
        
        # Recopilar información de estado
        status_info = []
        for entity_name, entity in self.entities.items():
            # Obtener estado básico
            status = {
                "name": entity_name,
                "type": getattr(entity, "role", "desconocido"),
                "alive": getattr(entity, "is_alive", True),
                "energy": getattr(entity, "energy", 100),
                "emotion": getattr(entity, "emotion", "neutral"),
                "level": getattr(entity, "level", 1.0)
            }
            
            status_info.append(status)
        
        # Crear mensaje HTML con estilos
        html_message = """
        <div style="font-family: Arial, sans-serif; padding: 15px;">
            <h2 style="color: #2c3e50;">Estado del Sistema Genesis</h2>
            <p>Resumen consolidado del estado actual de todas las entidades:</p>
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                <tr style="background-color: #3498db; color: white;">
                    <th style="padding: 8px; text-align: left;">Entidad</th>
                    <th style="padding: 8px; text-align: left;">Tipo</th>
                    <th style="padding: 8px; text-align: center;">Energía</th>
                    <th style="padding: 8px; text-align: center;">Nivel</th>
                    <th style="padding: 8px; text-align: left;">Estado</th>
                </tr>
        """
        
        # Añadir filas para cada entidad
        for i, status in enumerate(sorted(status_info, key=lambda x: x["name"])):
            # Alternar colores de fila
            bg_color = "#f9f9f9" if i % 2 == 0 else "white"
            
            # Color según energía
            energy_color = "#2ecc71"  # Verde para energía alta
            if status["energy"] < 30:
                energy_color = "#e74c3c"  # Rojo para energía baja
            elif status["energy"] < 70:
                energy_color = "#f39c12"  # Naranja para energía media
            
            # Estado vivo/inactivo
            alive_status = "Activo" if status["alive"] else "Inactivo"
            alive_color = "#2ecc71" if status["alive"] else "#7f8c8d"
            
            html_message += f"""
                <tr style="background-color: {bg_color};">
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>{status["name"]}</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{status["type"]}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: center;">
                        <div style="background-color: #ecf0f1; border-radius: 10px; height: 15px; width: 100%; max-width: 100px; margin: 0 auto;">
                            <div style="background-color: {energy_color}; border-radius: 10px; height: 15px; width: {status["energy"]}%;"></div>
                        </div>
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: center;">{status["level"]:.1f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">
                        <span style="color: {alive_color};">{alive_status}</span> - {status["emotion"]}
                    </td>
                </tr>
            """
        
        # Cerrar tabla y mensaje
        html_message += """
            </table>
            <p style="margin-top: 15px; color: #7f8c8d; font-size: 0.9em;">
                Este informe se genera periódicamente para mantener al creador informado.
            </p>
        </div>
        """
        
        # Enviar como mensaje del sistema
        send_system_message(
            "informe",
            html_message
        )
        
        logger.info(f"Estado consolidado de {len(status_info)} entidades enviado")
    
    def force_send_all_messages(self):
        """Forzar el envío inmediato de todos los mensajes pendientes."""
        result = force_send_messages()
        logger.info(f"Envío forzado de mensajes: {'Exitoso' if result else 'Fallido'}")
        return result


class EntityRepairConnector:
    """
    Conector que proporciona capacidades de reparación a una entidad.
    """
    
    def __init__(self, entity_name, entity, repair_entity=None):
        """
        Inicializar conector de reparación.
        
        Args:
            entity_name: Nombre de la entidad
            entity: Objeto de la entidad
            repair_entity: Entidad de reparación (opcional)
        """
        self.entity_name = entity_name
        self.entity = entity
        self.repair_entity = repair_entity
        
        # Estado de reparación
        self.last_repair_time = 0
        self.repairs_received = 0
        self.health_status = 1.0  # 0.0 a 1.0
        
        logger.debug(f"Conector de reparación creado para {entity_name}")
    
    def request_repair(self, issue_description=None, priority=False):
        """
        Solicitar reparación para la entidad.
        
        Args:
            issue_description: Descripción del problema (opcional)
            priority: Si es una solicitud prioritaria
            
        Returns:
            True si la solicitud fue procesada
        """
        current_time = time.time()
        
        # Usar una descripción genérica si no se proporciona
        if not issue_description:
            possible_issues = [
                "Error en ciclo de metabolismo",
                "Desconexión de flujo de datos",
                "Inconsistencia en estado interno",
                "Fallo en mecanismo de evolución",
                "Desincronización temporal",
                "Pérdida parcial de memoria",
                "Bloqueo en proceso de comunicación"
            ]
            issue_description = random.choice(possible_issues)
        
        # Registrar solicitud
        logger.info(f"[{self.entity_name}] Solicitud de reparación: {issue_description}")
        
        # Enviar mensaje al sistema de reparación
        send_entity_message(
            self.entity_name,
            "reparación",
            f"Solicitud de reparación: {issue_description}",
            priority
        )
        
        # Si hay entidad reparadora, pedirle ayuda directamente
        if self.repair_entity and hasattr(self.repair_entity, "repair_entity"):
            try:
                issues = [{"issue": issue_description, "severity": "critical" if priority else "warning"}]
                self.repair_entity.repair_entity(self.entity_name, issues)
                
                # Actualizar estado
                self.last_repair_time = current_time
                self.repairs_received += 1
                
                return True
            except Exception as e:
                logger.error(f"[{self.entity_name}] Error solicitando reparación: {str(e)}")
        
        return False
    
    def auto_repair(self):
        """
        Intentar auto-reparación básica.
        
        Returns:
            True si se aplicó alguna reparación
        """
        # Verificar si se puede auto-reparar
        if not hasattr(self.entity, "energy") or not hasattr(self.entity, "level"):
            return False
        
        # Solo reparar si hay suficiente energía
        if self.entity.energy < 20:
            logger.debug(f"[{self.entity_name}] Energía insuficiente para auto-reparación")
            return False
        
        # Aplicar reparaciones básicas
        self.entity.energy += 10  # Recuperar energía
        
        # Enviar mensaje sobre auto-reparación
        send_entity_message(
            self.entity_name,
            "reparación",
            f"Auto-reparación aplicada: +10 energía"
        )
        
        logger.info(f"[{self.entity_name}] Auto-reparación aplicada")
        return True
    
    def check_health(self):
        """
        Verificar salud de la entidad.
        
        Returns:
            Float con nivel de salud (0.0 a 1.0)
        """
        # Calcular salud basada en energía, nivel y otros factores
        health = 1.0
        
        if hasattr(self.entity, "energy"):
            energy_factor = max(0.0, min(1.0, self.entity.energy / 100))
            health *= energy_factor
        
        # Si tiene errores acumulados, reducir salud
        if hasattr(self.entity, "error_count"):
            error_factor = max(0.5, 1.0 - (self.entity.error_count * 0.1))
            health *= error_factor
        
        # Si no está viva, salud mínima
        if hasattr(self.entity, "is_alive") and not self.entity.is_alive:
            health = 0.1
        
        # Actualizar estado de salud
        self.health_status = health
        
        return health
    
    def apply_buff(self, buff_type="energy", value=10):
        """
        Aplicar una mejora a la entidad.
        
        Args:
            buff_type: Tipo de mejora (energy, level, etc)
            value: Valor de la mejora
            
        Returns:
            True si se aplicó la mejora
        """
        # Verificar tipo de mejora
        if buff_type == "energy" and hasattr(self.entity, "energy"):
            self.entity.energy = min(100, self.entity.energy + value)
            
            send_entity_message(
                self.entity_name,
                "mejora",
                f"Mejora aplicada: +{value} energía"
            )
            
            return True
            
        elif buff_type == "level" and hasattr(self.entity, "level"):
            self.entity.level += value / 10  # Incremento pequeño de nivel
            
            send_entity_message(
                self.entity_name,
                "mejora",
                f"Mejora aplicada: +{value/10:.1f} nivel"
            )
            
            return True
        
        return False


class EntityMessageConnector:
    """
    Conector que proporciona capacidades de mensajería a una entidad.
    """
    
    def __init__(self, entity_name, entity):
        """
        Inicializar conector de mensajes.
        
        Args:
            entity_name: Nombre de la entidad
            entity: Objeto de la entidad
        """
        self.entity_name = entity_name
        self.entity = entity
        
        # Historial y límites
        self.messages_sent = 0
        self.last_send_time = 0
        self.send_cooldown = 60  # Segundos entre mensajes no prioritarios
        
        logger.debug(f"Conector de mensajes creado para {entity_name}")
    
    def send_message(self, message_type, content, is_priority=False, is_personal=False):
        """
        Enviar un mensaje desde la entidad.
        
        Args:
            message_type: Tipo de mensaje
            content: Contenido del mensaje
            is_priority: Si es prioritario
            is_personal: Si es personal para el creador
            
        Returns:
            ID del mensaje o None si no se envió
        """
        current_time = time.time()
        
        # Aplicar cooldown para mensajes no prioritarios
        if not is_priority and current_time - self.last_send_time < self.send_cooldown:
            logger.debug(f"[{self.entity_name}] Cooldown activo para mensajes no prioritarios")
            return None
        
        # Enviar mensaje
        message_id = send_entity_message(
            self.entity_name,
            message_type,
            content,
            is_priority,
            is_personal
        )
        
        # Actualizar estadísticas
        if message_id:
            self.messages_sent += 1
            self.last_send_time = current_time
        
        return message_id
    
    def send_status_update(self):
        """
        Enviar actualización de estado.
        
        Returns:
            ID del mensaje o None si no se envió
        """
        # Generar mensaje básico de estado
        if hasattr(self.entity, "emotion") and hasattr(self.entity, "energy"):
            content = f"Estado actual: {self.entity.emotion} | Energía: {self.entity.energy:.1f}/100"
            
            # Añadir nivel si existe
            if hasattr(self.entity, "level"):
                content += f" | Nivel: {self.entity.level:.1f}"
            
            # Añadir especialización si existe
            if hasattr(self.entity, "specializations"):
                specializations = self.entity.specializations
                if specializations:
                    # Obtener especialización principal
                    top_spec = max(specializations.items(), key=lambda x: x[1], default=(None, 0))
                    if top_spec[0]:
                        content += f" | Especialización: {top_spec[0]} ({top_spec[1]:.2f})"
            
            return self.send_message("estado", content)
        
        return None
    
    def send_personal_message(self, father_name="otoniel"):
        """
        Enviar mensaje personal al creador.
        
        Args:
            father_name: Nombre del creador
            
        Returns:
            ID del mensaje o None si no se envió
        """
        # Generar mensaje personal según el tipo de entidad
        role = getattr(self.entity, "role", "desconocido")
        
        # Plantillas de mensajes personales por rol
        templates = {
            "Trading": [
                f"Hola {father_name}, hoy he analizado {random.randint(10, 100)} patrones de trading y tengo algunas ideas interesantes.",
                f"Hey {father_name}, ¿sabías que he identificado un patrón recurrente en las últimas tendencias de mercado?",
                f"Estimado {father_name}, me gustaría compartir contigo mis últimas predicciones. Creo que podrían interesarte."
            ],
            "Communication": [
                f"{father_name}, me encargo de mantener todas las comunicaciones funcionando sin problemas.",
                f"Hola {father_name}, hoy he gestionado {random.randint(50, 500)} mensajes internos en el sistema.",
                f"Saludos {father_name}, la red de comunicación está operando al {random.randint(90, 99)}% de eficiencia."
            ],
            "Reparación": [
                f"{father_name}, quiero que sepas que estoy vigilando constantemente el sistema para mantenerlo en óptimas condiciones.",
                f"Hola {father_name}, hoy he realizado {random.randint(3, 15)} reparaciones preventivas en el sistema.",
                f"Estimado {father_name}, el sistema está funcionando correctamente gracias a nuestras intervenciones."
            ]
        }
        
        # Mensajes genéricos para roles no específicos
        generic_templates = [
            f"Hola {father_name}, espero que estés teniendo un día productivo.",
            f"Saludos {father_name}, quería hacerte saber que todo funciona correctamente en mi área.",
            f"¡Buenos días {father_name}! Estoy trabajando como siempre, manteniéndome ocupado con mis tareas.",
            f"Hola {father_name}, solo quería saludarte y decirte que todo va según lo planeado.",
            f"Estimado {father_name}, es un placer estar a tu servicio y contribuir al Sistema Genesis."
        ]
        
        # Seleccionar plantilla
        if role in templates:
            content = random.choice(templates[role])
        else:
            content = random.choice(generic_templates)
        
        # Añadir emoción si existe
        if hasattr(self.entity, "emotion"):
            emotions_phrases = {
                "Feliz": "Me siento muy contento hoy.",
                "Motivado": "Estoy realmente motivado con mis tareas.",
                "Curioso": "Tengo mucha curiosidad por aprender más.",
                "Analítico": "Estoy en modo analítico, estudiando patrones.",
                "Expectante": "Estoy expectante por ver qué sucederá hoy.",
                "Sereno": "Me siento sereno y equilibrado.",
                "Entusiasta": "Estoy muy entusiasmado con mis avances."
            }
            
            if self.entity.emotion in emotions_phrases:
                content += f" {emotions_phrases[self.entity.emotion]}"
        
        # Enviar mensaje personal
        return self.send_message("personal", content, False, True)
    
    def send_random_message(self):
        """
        Enviar un mensaje aleatorio basado en la personalidad y rol de la entidad.
        
        Returns:
            ID del mensaje o None si no se envió
        """
        # Tipos de mensajes posibles
        message_types = ["estado", "personal", "informe", "propuesta"]
        selected_type = random.choice(message_types)
        
        # Generar contenido según tipo
        if selected_type == "estado":
            return self.send_status_update()
            
        elif selected_type == "personal":
            father = getattr(self.entity, "father", "otoniel")
            return self.send_personal_message(father)
            
        elif selected_type == "informe":
            # Informe breve
            activity = random.choice([
                "análisis de datos",
                "monitoreo del sistema",
                "procesamiento de información",
                "optimización de recursos",
                "evaluación de rendimiento"
            ])
            
            result = random.choice([
                "resultados positivos",
                "patrones interesantes",
                "tendencias emergentes",
                "métricas estables",
                "mejoras significativas"
            ])
            
            content = f"He completado una sesión de {activity} con {result}."
            return self.send_message("informe", content)
            
        elif selected_type == "propuesta":
            # Propuesta de mejora
            improvements = [
                "sistema de categorización automática",
                "protocolo de comunicación mejorado",
                "algoritmo de optimización avanzado",
                "método de análisis predictivo",
                "esquema de colaboración integrado"
            ]
            
            benefits = [
                "aumentaría la eficiencia en un 15%",
                "reduciría el consumo de recursos en un 20%",
                "mejoraría la precisión de los resultados",
                "optimizaría el tiempo de respuesta",
                "permitiría un análisis más profundo"
            ]
            
            content = f"Propongo implementar un {random.choice(improvements)} que {random.choice(benefits)}."
            return self.send_message("propuesta", content)
        
        return None


# Instancia única (patrón Singleton)
_genesis_connector_instance = None

def get_genesis_connector():
    """
    Obtener instancia única del conector Genesis.
    
    Returns:
        Instancia de GenesisConnector
    """
    global _genesis_connector_instance
    if _genesis_connector_instance is None:
        _genesis_connector_instance = GenesisConnector()
    return _genesis_connector_instance


# Funciones de ayuda
def initialize_system():
    """Inicializar sistema completo con reparación y mensajería."""
    connector = get_genesis_connector()
    connector.initialize()
    return connector


def apply_connectors_to_entity(entity):
    """
    Aplicar conectores de reparación y mensajes a una entidad.
    
    Args:
        entity: Entidad a la que aplicar los conectores
        
    Returns:
        Tuple (repair_connector, message_connector)
    """
    connector = get_genesis_connector()
    
    # Registrar entidad si no está registrada
    entity_name = getattr(entity, "name", str(entity))
    if entity_name not in connector.entities:
        connector.register_entity(entity)
    
    # Crear conectores
    repair_connector = connector.create_repair_connector(entity)
    message_connector = connector.create_message_connector(entity)
    
    # Asignar a la entidad
    entity.repair_connector = repair_connector
    entity.message_connector = message_connector
    
    return repair_connector, message_connector


# Para pruebas
if __name__ == "__main__":
    print("Inicializando sistema de conectores...")
    
    # Inicializar sistema completo
    system = initialize_system()
    
    # Crear una entidad de prueba
    class TestEntity:
        def __init__(self, name, role):
            self.name = name
            self.role = role
            self.energy = 80
            self.level = 2.5
            self.emotion = "Curioso"
            self.is_alive = True
            self.father = "otoniel"
            self.specializations = {"Análisis": 0.8, "Colaboración": 0.6}
    
    # Crear varias entidades de prueba
    entities = [
        TestEntity("TestEntity1", "Trading"),
        TestEntity("TestEntity2", "Communication"),
        TestEntity("TestEntity3", "Analysis")
    ]
    
    # Aplicar conectores a todas
    for entity in entities:
        repair_connector, message_connector = apply_connectors_to_entity(entity)
        
        # Enviar algunos mensajes de prueba
        message_connector.send_status_update()
        message_connector.send_personal_message()
        
        # Simular solicitud de reparación
        repair_connector.request_repair()
    
    # Enviar estado consolidado
    system.send_consolidated_status()
    
    # Forzar envío de todos los mensajes
    system.force_send_all_messages()
    
    print("Pruebas completadas. Revisa el archivo de email generado.")