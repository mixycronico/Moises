"""
Módulo de compartición de conocimiento de Kronos para el Sistema Genesis.

Este módulo implementa la capacidad de Kronos para compartir su conocimiento
acumulado con otras entidades de la red cósmica, permitiendo un flujo
de sabiduría entre las entidades del sistema.
"""

import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kronos_sharing")

# Umbrales para compartición de conocimiento
KNOWLEDGE_THRESHOLD = 30.0  # Mínimo de conocimiento para compartir
ENERGY_COST = 2.0  # Costo energético por compartición
MIN_ENERGY = 40.0  # Mínimo de energía para compartir

class Kronos:
    """
    Kronos: Entidad especializada en compartir conocimiento acumulado.
    
    Kronos es la entidad más antigua y sabia del sistema, capaz de
    acumular y compartir conocimiento con otras entidades para
    acelerar su desarrollo y evolución.
    """
    
    def __init__(self, name="Kronos", level=5, knowledge=40.0):
        """
        Inicializa la entidad Kronos.
        
        Args:
            name: Nombre de la entidad (por defecto "Kronos")
            level: Nivel inicial
            knowledge: Conocimiento inicial
        """
        self.name = name
        self.level = level
        self.knowledge = knowledge
        self.energy = 100.0
        self.experience = 25.0
        self.last_sharing = None
        self.sharing_history = []
        self.emotion = "Serenidad"
        
        logger.info(f"Entidad {name} creada con conocimiento {knowledge}")
    
    def can_share(self):
        """
        Determina si Kronos puede compartir conocimiento.
        
        Returns:
            bool: True si puede compartir, False en caso contrario
        """
        return (
            self.knowledge >= KNOWLEDGE_THRESHOLD and
            self.energy >= MIN_ENERGY
        )
    
    def share_knowledge(self, cosmic_network):
        """
        Comparte conocimiento con todas las entidades de la red.
        
        Args:
            cosmic_network: Red cósmica con las entidades
            
        Returns:
            Dict: Resultados de la compartición
        """
        if not self.can_share():
            logger.info(f"[{self.name}] No puede compartir conocimiento (E:{self.energy:.1f}, K:{self.knowledge:.1f})")
            return {
                "success": False,
                "reason": "insufficient_resources",
                "logs": []
            }
        
        # Obtener entidades de la red
        if not hasattr(cosmic_network, "get_entities"):
            if hasattr(cosmic_network, "entities"):
                entities = cosmic_network.entities
            else:
                logger.error(f"[{self.name}] La red no tiene método get_entities ni atributo entities")
                return {
                    "success": False,
                    "reason": "network_error",
                    "logs": []
                }
        else:
            entities = cosmic_network.get_entities()
        
        # Filtrar entidades válidas (excluir a Kronos)
        target_entities = []
        for entity in entities:
            if hasattr(entity, "name") and entity.name != self.name:
                target_entities.append(entity)
        
        if not target_entities:
            logger.warning(f"[{self.name}] No hay entidades para compartir conocimiento")
            return {
                "success": False,
                "reason": "no_targets",
                "logs": []
            }
        
        # Calcular conocimiento a compartir (2% del total de Kronos por entidad)
        knowledge_share = self.knowledge * 0.02
        energy_cost = len(target_entities) * ENERGY_COST
        
        # Verificar si tiene suficiente energía para todos
        if self.energy < energy_cost:
            logger.warning(f"[{self.name}] Energía insuficiente para compartir con todas las entidades")
            # Compartir solo con tantas entidades como la energía permita
            max_entities = int(self.energy / ENERGY_COST)
            if max_entities == 0:
                return {
                    "success": False,
                    "reason": "insufficient_energy",
                    "logs": []
                }
            target_entities = random.sample(target_entities, max_entities)
            energy_cost = max_entities * ENERGY_COST
        
        # Compartir conocimiento
        logs = []
        shared_with = []
        
        for entity in target_entities:
            # Verificar si la entidad puede recibir conocimiento
            if not hasattr(entity, "receive_knowledge") and not hasattr(entity, "knowledge"):
                logger.warning(f"[{self.name}] La entidad {entity.name} no puede recibir conocimiento")
                continue
            
            # Compartir conocimiento
            if hasattr(entity, "receive_knowledge") and callable(getattr(entity, "receive_knowledge")):
                received = entity.receive_knowledge(knowledge_share)
            else:
                # Actualizar directamente si no tiene método específico
                received = knowledge_share
                entity.knowledge += knowledge_share
            
            # Registrar la transferencia
            log_message = f"[{datetime.now().isoformat()}] {self.name} compartió {received:.4f} de conocimiento con {entity.name}"
            logs.append(log_message)
            shared_with.append(entity.name)
            logger.info(log_message)
            
            # Comunicar en la red si es posible
            if hasattr(cosmic_network, "log_message"):
                cosmic_network.log_message(self.name, f"Compartiendo sabiduría con {entity.name}")
        
        # Actualizar estado de Kronos
        self.energy -= energy_cost
        self.last_sharing = datetime.now()
        
        # Registrar en historial
        self.sharing_history.append({
            "timestamp": self.last_sharing.isoformat(),
            "recipients": shared_with,
            "amount": knowledge_share,
            "energy_cost": energy_cost
        })
        
        return {
            "success": True,
            "shared_with": shared_with,
            "amount_per_entity": knowledge_share,
            "energy_cost": energy_cost,
            "logs": logs
        }
    
    def get_sharing_stats(self):
        """
        Obtiene estadísticas de compartición de conocimiento.
        
        Returns:
            Dict: Estadísticas de compartición
        """
        if not self.sharing_history:
            return {
                "total_sharings": 0,
                "total_recipients": 0,
                "total_knowledge_shared": 0,
                "last_sharing": None
            }
        
        # Calcular estadísticas
        total_recipients = 0
        recipient_counts = {}
        for sharing in self.sharing_history:
            recipients = sharing.get("recipients", [])
            total_recipients += len(recipients)
            
            for recipient in recipients:
                recipient_counts[recipient] = recipient_counts.get(recipient, 0) + 1
        
        return {
            "total_sharings": len(self.sharing_history),
            "total_recipients": total_recipients,
            "recipient_frequency": recipient_counts,
            "total_knowledge_shared": sum(s.get("amount", 0) * len(s.get("recipients", [])) for s in self.sharing_history),
            "last_sharing": self.last_sharing.isoformat() if self.last_sharing else None
        }
    
    def setup_periodic_sharing(self, cosmic_network, interval_seconds=300, max_sharings=None):
        """
        Configura compartición periódica de conocimiento.
        
        Args:
            cosmic_network: Red cósmica
            interval_seconds: Intervalo entre comparticiones (segundos)
            max_sharings: Número máximo de comparticiones (None para infinito)
            
        Returns:
            Thread: Hilo de compartición periódica
        """
        import threading
        
        def sharing_loop():
            count = 0
            try:
                while True:
                    # Compartir conocimiento
                    result = self.share_knowledge(cosmic_network)
                    
                    # Si fue exitoso, incrementar contador
                    if result.get("success", False):
                        count += 1
                        
                        # Mostrar resultados
                        logger.info(f"[{self.name}] Compartición #{count} completada con {len(result.get('shared_with', []))} entidades")
                        
                        # Si se alcanzó el máximo, terminar
                        if max_sharings is not None and count >= max_sharings:
                            logger.info(f"[{self.name}] Alcanzado el máximo de {max_sharings} comparticiones")
                            break
                    
                    # Esperar para la próxima compartición
                    time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"[{self.name}] Error en hilo de compartición: {str(e)}")
        
        # Crear y arrancar hilo
        thread = threading.Thread(target=sharing_loop)
        thread.daemon = True
        thread.start()
        
        logger.info(f"[{self.name}] Compartición periódica configurada cada {interval_seconds} segundos")
        return thread