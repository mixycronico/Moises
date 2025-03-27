"""
Implementación de la Red Cósmica para el Sistema Genesis.

La Red Cósmica permite la comunicación entre todas las entidades 
y proporciona servicios de intercambio de conocimiento, colaboración
y mecanismos de resiliencia distribuida.
"""

import time
import random
import logging
import threading
from typing import Dict, List, Any, Set, Optional, Union
from collections import deque

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CosmicNetwork:
    """
    Red de comunicación y compartición de conocimiento para entidades del Sistema Genesis.
    Actúa como sistema nervioso central del colectivo de entidades.
    """
    
    def __init__(self, max_messages=1000, broadcast_energy_cost=0.1):
        """
        Inicializar red cósmica.
        
        Args:
            max_messages: Número máximo de mensajes en cola
            broadcast_energy_cost: Costo de energía para transmitir mensajes
        """
        # Entidades conectadas
        self.entities = {}  # {nombre_entidad: objeto_entidad}
        self.entity_roles = {}  # {nombre_entidad: rol}
        
        # Sistema de mensajería
        self.messages = deque(maxlen=max_messages)
        self.broadcast_energy_cost = broadcast_energy_cost
        
        # Conocimiento colectivo
        self.knowledge_pool = 0.0
        self.collective_insights = []
        self.recent_collaborations = []
        
        # Métricas
        self.network_health = 100.0
        self.metrics = {
            "messages_sent": 0,
            "knowledge_shared": 0.0,
            "collaborations": 0,
            "entity_count": 0
        }
        
        # Sistema activo
        self.active = True
        
        # Log
        logger.info("Red Cósmica inicializada")
    
    def register_entity(self, entity, role=None):
        """
        Registrar una entidad en la red.
        
        Args:
            entity: Objeto de entidad
            role: Rol de la entidad en la red (opcional)
            
        Returns:
            True si se registró correctamente, False en caso contrario
        """
        if not hasattr(entity, 'name') or not entity.name:
            logger.error("Entidad sin nombre no puede ser registrada en la red")
            return False
        
        # Registrar entidad
        self.entities[entity.name] = entity
        
        # Establecer rol si se proporciona
        if role:
            self.entity_roles[entity.name] = role
        elif hasattr(entity, 'role') and entity.role:
            self.entity_roles[entity.name] = entity.role
        else:
            self.entity_roles[entity.name] = "Miembro"
        
        # Asignar red a la entidad
        if hasattr(entity, 'network'):
            entity.network = self
        
        # Actualizar métricas
        self.metrics["entity_count"] = len(self.entities)
        
        logger.info(f"Entidad {entity.name} ({self.entity_roles[entity.name]}) registrada en la Red Cósmica")
        return True
    
    def unregister_entity(self, entity_name):
        """
        Eliminar una entidad de la red.
        
        Args:
            entity_name: Nombre de la entidad a eliminar
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        if entity_name in self.entities:
            # Eliminar entidad y su rol
            entity = self.entities.pop(entity_name)
            self.entity_roles.pop(entity_name, None)
            
            # Actualizar métricas
            self.metrics["entity_count"] = len(self.entities)
            
            logger.info(f"Entidad {entity_name} eliminada de la Red Cósmica")
            return True
        else:
            logger.warning(f"Entidad {entity_name} no encontrada en la Red Cósmica")
            return False
    
    def broadcast(self, sender_name, message):
        """
        Transmitir un mensaje a todas las entidades de la red.
        
        Args:
            sender_name: Nombre de la entidad que envía el mensaje
            message: Mensaje a transmitir
            
        Returns:
            Número de entidades que recibieron el mensaje
        """
        if sender_name not in self.entities:
            logger.warning(f"Emisor {sender_name} no registrado en la Red Cósmica")
            return 0
        
        # Obtener entidad emisora
        sender = self.entities[sender_name]
        
        # Verificar si tiene energía para transmitir
        if hasattr(sender, 'energy') and sender.energy < self.broadcast_energy_cost:
            logger.warning(f"{sender_name} no tiene suficiente energía para transmitir mensaje")
            return 0
        
        # Consumir energía si la entidad tiene ese atributo
        if hasattr(sender, 'energy'):
            sender.energy -= self.broadcast_energy_cost
        
        # Almacenar mensaje
        timestamp = time.time()
        full_message = {
            "timestamp": timestamp,
            "sender": sender_name,
            "sender_role": self.entity_roles.get(sender_name, "Desconocido"),
            "content": message
        }
        self.messages.append(full_message)
        
        # Actualizar métricas
        self.metrics["messages_sent"] += 1
        
        # Mensaje especial para conocimiento colectivo
        if isinstance(message, dict) and message.get("type") == "knowledge":
            knowledge_value = message.get("value", 0.0)
            self.knowledge_pool += knowledge_value
            self.metrics["knowledge_shared"] += knowledge_value
            
            # Registrar insight importante
            if knowledge_value > 2.0:
                insight = {
                    "timestamp": timestamp,
                    "entity": sender_name,
                    "content": message.get("content", ""),
                    "value": knowledge_value
                }
                self.collective_insights.append(insight)
        
        # Número de receptores
        return len(self.entities) - 1
    
    def simulate_collaboration(self, entities=None):
        """
        Simular colaboración entre entidades para generar conocimiento emergente.
        
        Args:
            entities: Lista de nombres de entidades participantes (opcional)
            
        Returns:
            Resultado de la colaboración
        """
        # Si no se especifican entidades, seleccionar aleatoriamente 2-4
        if not entities:
            available = list(self.entities.keys())
            if len(available) < 2:
                return {"success": False, "reason": "Insuficientes entidades para colaborar"}
            
            num_collaborators = min(len(available), random.randint(2, 4))
            entities = random.sample(available, num_collaborators)
        
        # Verificar si todas las entidades existen
        for entity_name in entities:
            if entity_name not in self.entities:
                return {"success": False, "reason": f"Entidad {entity_name} no registrada"}
        
        # Calcular contribución de conocimiento y energía
        total_knowledge = 0.0
        total_energy_cost = 0.0
        
        # Contribución de cada entidad
        contributions = {}
        for entity_name in entities:
            entity = self.entities[entity_name]
            
            # Obtener conocimiento y nivel si están disponibles
            knowledge = getattr(entity, 'knowledge', 0.0)
            level = getattr(entity, 'level', 1.0)
            
            # Contribución basada en conocimiento y nivel
            contribution = knowledge * 0.1 * (level ** 0.5)
            contributions[entity_name] = contribution
            total_knowledge += contribution
            
            # Consumir energía
            energy_cost = 0.5 + (contribution * 0.1)
            if hasattr(entity, 'energy'):
                entity.energy = max(0.0, entity.energy - energy_cost)
            
            total_energy_cost += energy_cost
        
        # Calcular conocimiento emergente (con factor de sinergia)
        synergy_factor = 1.0 + (len(entities) * 0.1)
        emergent_knowledge = total_knowledge * synergy_factor
        
        # Distribuir beneficios a las entidades participantes
        for entity_name in entities:
            entity = self.entities[entity_name]
            
            # Cantidad proporcional a la contribución
            contribution_ratio = contributions[entity_name] / total_knowledge if total_knowledge > 0 else 1.0 / len(entities)
            knowledge_gain = emergent_knowledge * contribution_ratio * 0.5
            
            # Aplicar ganancia
            if hasattr(entity, 'knowledge'):
                entity.knowledge += knowledge_gain
            
            # Experiencia adicional
            if hasattr(entity, 'experience'):
                entity.experience += random.uniform(0.1, 0.5)
            
            # Posibilidad de evolución acelerada
            if hasattr(entity, 'evolve') and callable(entity.evolve) and random.random() < 0.2:
                entity.evolve()
        
        # Registrar colaboración
        collaboration = {
            "timestamp": time.time(),
            "entities": entities,
            "knowledge_generated": emergent_knowledge,
            "energy_cost": total_energy_cost
        }
        self.recent_collaborations.append(collaboration)
        
        # Actualizar métricas
        self.metrics["collaborations"] += 1
        
        # Añadir al conocimiento colectivo
        self.knowledge_pool += emergent_knowledge * 0.3
        
        # Registrar en log
        logger.info(f"Colaboración entre {', '.join(entities)} generó {emergent_knowledge:.2f} de conocimiento")
        
        return {
            "success": True,
            "knowledge_generated": emergent_knowledge,
            "energy_cost": total_energy_cost,
            "participants": entities,
            "synergy_factor": synergy_factor
        }
    
    def request_help(self, entity, help_type):
        """
        Solicitar ayuda a la red para una entidad en dificultades.
        
        Args:
            entity: Entidad que necesita ayuda
            help_type: Tipo de ayuda requerida ("luz"=energía, "sombra"=protección, etc.)
            
        Returns:
            Resultado de la solicitud
        """
        if not hasattr(entity, 'name') or entity.name not in self.entities:
            logger.warning(f"Entidad no registrada solicita ayuda: {getattr(entity, 'name', 'desconocido')}")
            return {"success": False, "reason": "Entidad no registrada"}
        
        # Tipo de ayuda
        if help_type == "luz":
            # Ayuda energética
            helpers = []
            total_energy = 0.0
            
            # Buscar entidades con energía de sobra
            for name, other_entity in self.entities.items():
                if name != entity.name and hasattr(other_entity, 'energy') and other_entity.energy > 50.0:
                    # Cantidad de energía que puede donar (25% del excedente sobre 50)
                    donation = (other_entity.energy - 50.0) * 0.25
                    
                    # Aplicar donación
                    other_entity.energy -= donation
                    total_energy += donation
                    helpers.append(name)
                    
                    # Log de donación
                    logger.info(f"{name} dona {donation:.2f} de energía a {entity.name}")
            
            # Aplicar energía recibida
            if hasattr(entity, 'energy'):
                entity.energy += total_energy
                
                # Log de recepción
                logger.info(f"{entity.name} recibe {total_energy:.2f} de energía de la red")
            
            return {
                "success": True,
                "help_type": "luz",
                "amount": total_energy,
                "helpers": helpers
            }
            
        elif help_type == "sombra":
            # Implementar protección (pendiente)
            return {"success": False, "reason": "Tipo de ayuda no implementado"}
            
        else:
            return {"success": False, "reason": "Tipo de ayuda desconocido"}
    
    def get_network_status(self):
        """
        Obtener estado actual de la red cósmica.
        
        Returns:
            Diccionario con estado actual
        """
        return {
            "active": self.active,
            "entities": len(self.entities),
            "knowledge_pool": self.knowledge_pool,
            "network_health": self.network_health,
            "messages_count": len(self.messages),
            "metrics": self.metrics,
            "recent_insights": self.collective_insights[-5:] if self.collective_insights else []
        }

def create_cosmic_network():
    """
    Crear y configurar una red cósmica.
    
    Returns:
        Instancia de CosmicNetwork
    """
    return CosmicNetwork()