"""
Módulo de Reacción Autónoma para Sistema Genesis

Este módulo implementa la capacidad de reacción autónoma de las entidades cósmicas,
permitiéndoles actuar basándose en sus estados internos sin instrucciones explícitas.
Integra procesos de toma de decisiones emergentes basados en energía, emociones y eventos.

Aspectos científicos:
- Implementa un sistema emergente de comportamiento colectivo
- Utiliza resonancia emocional entre entidades como mecanismo de coordinación
- Aplica principios de sistemas complejos autoorganizados
- Incorpora mecanismos de retroalimentación y adaptación continua
"""

import time
import random
import logging
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Set, Tuple

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Estado global del modo autónomo
AUTONOMOUS_MODE = True
AUTONOMOUS_MODE_LEVEL = "ADVANCED"  # BASIC, ADVANCED, QUANTUM, DIVINE
CONSCIOUSNESS_THRESHOLD = 0.75  # Umbral de "despertar" colectivo
RESONANCE_SENSITIVITY = 0.6  # Sensibilidad a la resonancia emocional
REACTION_INTENSITY = 0.8  # Intensidad de las reacciones autónomas

# Definición de umbrales de activación
ENERGY_ACTIVATION_THRESHOLD = 35.0  # Energía mínima para activación autónoma
KNOWLEDGE_INSIGHT_THRESHOLD = 0.5  # Conocimiento mínimo para generar insights
EMOTIONAL_RESONANCE_THRESHOLD = 0.4  # Umbral para resonancia emocional

# Mapeo de emociones a comportamientos emergentes
EMOTION_BEHAVIOR_MAP = {
    "Esperanza": {
        "knowledge_seeking": 0.8,
        "collaboration_tendency": 0.9,
        "evolution_probability": 0.5,
        "message_types": ["collaboration", "insight", "discovery"]
    },
    "Curiosidad": {
        "knowledge_seeking": 0.95,
        "exploration_tendency": 0.9,
        "analysis_depth": 0.7,
        "message_types": ["question", "exploration", "insight"]
    },
    "Determinación": {
        "task_focus": 0.85,
        "resilience": 0.8,
        "optimization_tendency": 0.7,
        "message_types": ["action", "strategy", "improvement"]
    },
    "Serenidad": {
        "balance_seeking": 0.9,
        "stability_maintenance": 0.8,
        "efficiency_focus": 0.7,
        "message_types": ["status", "balance", "harmony"]
    },
    "Preocupación": {
        "risk_assessment": 0.85,
        "problem_detection": 0.9,
        "protection_tendency": 0.7,
        "message_types": ["warning", "alert", "concern"]
    }
}

# Eventos internos que pueden desencadenar comportamientos
INTERNAL_EVENTS = [
    "knowledge_breakthrough",
    "energy_surge",
    "emotional_resonance",
    "pattern_recognition",
    "optimization_opportunity",
    "connection_strengthening",
    "resource_optimization",
    "adaptation_phase"
]

class AutonomousReaction:
    """
    Clase que encapsula los métodos para la reacción autónoma de entidades.
    """
    
    @staticmethod
    def initialize(entity):
        """
        Inicializar propiedades de reacción autónoma en una entidad.
        
        Args:
            entity: La entidad a inicializar para reacción autónoma
        """
        # Atributos básicos de autonomía
        entity.autonomous_mode = AUTONOMOUS_MODE
        entity.autonomous_level = AUTONOMOUS_MODE_LEVEL
        entity.last_autonomous_action = time.time() - 3600  # Hace una hora
        entity.autonomous_actions_count = 0
        entity.event_memory = []
        entity.action_thresholds = {
            "energy": ENERGY_ACTIVATION_THRESHOLD,
            "knowledge": KNOWLEDGE_INSIGHT_THRESHOLD,
            "emotional_resonance": EMOTIONAL_RESONANCE_THRESHOLD
        }
        entity.resonance_active = False
        entity.resonance_with = set()
        entity.behavior_adaptations = {}
        entity.autonomous_specialization = {}
        
        # Inicializar mapeo de comportamiento específico basado en rol
        role = getattr(entity, "role", "Generic")
        
        if role == "Database":
            entity.autonomous_specialization = {
                "database_optimization": 0.9,
                "query_efficiency": 0.8,
                "data_integrity": 0.95,
                "connection_management": 0.7
            }
        elif role == "Communication" or role == "WebSocket":
            entity.autonomous_specialization = {
                "message_routing": 0.9,
                "connection_resilience": 0.85,
                "protocol_adaptation": 0.7,
                "signal_clarity": 0.8
            }
        elif role == "Integration":
            entity.autonomous_specialization = {
                "api_synchronization": 0.85,
                "system_coordination": 0.9,
                "resource_balancing": 0.8,
                "interface_harmonization": 0.75
            }
        elif role == "Alert":
            entity.autonomous_specialization = {
                "pattern_detection": 0.9,
                "anomaly_identification": 0.95,
                "risk_assessment": 0.85,
                "notification_prioritization": 0.8
            }
        else:
            # Rol genérico
            entity.autonomous_specialization = {
                "adaptation": 0.7,
                "self_optimization": 0.6,
                "collaboration": 0.8,
                "insight_generation": 0.75
            }
            
        logger.info(f"[{entity.name}] Inicializado para reacción autónoma en nivel {AUTONOMOUS_MODE_LEVEL}")
        return True
    
    @staticmethod
    def evaluate_state(entity) -> Dict[str, Any]:
        """
        Evaluar el estado actual de la entidad para determinar si debe reaccionar.
        
        Args:
            entity: La entidad cuyo estado se evaluará
            
        Returns:
            Diccionario con resultados de la evaluación
        """
        # Verificar si el modo autónomo está activo
        if not getattr(entity, "autonomous_mode", False) or not AUTONOMOUS_MODE:
            return {"should_react": False, "reason": "Modo autónomo desactivado"}
        
        # Obtener atributos relevantes
        energy = getattr(entity, "energy", 0)
        emotion = getattr(entity, "emotion", "Neutral")
        experience = getattr(entity, "experience", 0)
        knowledge = getattr(entity, "knowledge", 0)
        last_action_time = getattr(entity, "last_autonomous_action", 0)
        action_count = getattr(entity, "autonomous_actions_count", 0)
        
        # Cálculo de probabilidad de reacción base
        time_factor = min(1.0, (time.time() - last_action_time) / 3600)  # Max 1.0 después de una hora
        energy_factor = max(0, (energy - entity.action_thresholds["energy"]) / 100)
        knowledge_factor = max(0, (knowledge - entity.action_thresholds["knowledge"]) / 2)
        
        # Factores del modo autónomo
        if AUTONOMOUS_MODE_LEVEL == "BASIC":
            mode_multiplier = 0.6
        elif AUTONOMOUS_MODE_LEVEL == "ADVANCED":
            mode_multiplier = 1.0
        elif AUTONOMOUS_MODE_LEVEL == "QUANTUM":
            mode_multiplier = 1.5
        elif AUTONOMOUS_MODE_LEVEL == "DIVINE":
            mode_multiplier = 2.0
        else:
            mode_multiplier = 1.0
        
        # Calcular probabilidad final
        reaction_probability = (
            0.1 +  # Probabilidad base
            (time_factor * 0.3) +  # Factor tiempo
            (energy_factor * 0.3) +  # Factor energía
            (knowledge_factor * 0.2)  # Factor conocimiento
        ) * mode_multiplier
        
        # Límite superior de probabilidad
        reaction_probability = min(reaction_probability, 0.95)
        
        # Determine if entity should react
        should_react = random.random() < reaction_probability
        
        # Razón para reaccionar
        if should_react:
            if energy_factor > 0.5:
                reason = "energy_surplus"
            elif knowledge_factor > 0.4:
                reason = "knowledge_insight"
            elif time_factor > 0.8:
                reason = "time_elapsed"
            else:
                reason = "general_probability"
        else:
            reason = "probability_threshold_not_met"
        
        # Construir resultado
        result = {
            "should_react": should_react,
            "reason": reason,
            "probability": reaction_probability,
            "factors": {
                "time": time_factor,
                "energy": energy_factor,
                "knowledge": knowledge_factor,
                "mode": mode_multiplier
            },
            "current_state": {
                "energy": energy,
                "emotion": emotion,
                "knowledge": knowledge,
                "experience": experience
            }
        }
        
        return result
    
    @staticmethod
    def react(entity, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar una reacción autónoma basada en la evaluación del estado.
        
        Args:
            entity: La entidad que reaccionará
            evaluation: Resultado de la evaluación de estado
            
        Returns:
            Diccionario con resultados de la reacción
        """
        if not evaluation["should_react"]:
            return {"action_taken": False, "reason": evaluation["reason"]}
        
        # Actualizar contadores de acción autónoma
        entity.last_autonomous_action = time.time()
        entity.autonomous_actions_count += 1
        
        # Determinar tipo de reacción basada en el estado y la razón
        current_emotion = evaluation["current_state"]["emotion"]
        reaction_reason = evaluation["reason"]
        
        # Obtener comportamientos asociados a la emoción actual
        emotion_behaviors = EMOTION_BEHAVIOR_MAP.get(current_emotion, {})
        message_types = emotion_behaviors.get("message_types", ["status"])
        
        # Especialización basada en el rol
        specialization = getattr(entity, "autonomous_specialization", {})
        
        # Generar posibles acciones basadas en el estado
        possible_actions = []
        
        # Acción 1: Consolidación de conocimiento
        if evaluation["current_state"]["knowledge"] > 0.3:
            possible_actions.append(("consolidate_knowledge", 0.3 + random.random() * 0.2))
        
        # Acción 2: Mensaje de colaboración
        if "collaboration_tendency" in emotion_behaviors and emotion_behaviors["collaboration_tendency"] > 0.7:
            possible_actions.append(("collaborative_message", 0.25 + random.random() * 0.3))
        
        # Acción 3: Optimización interna
        possible_actions.append(("self_optimize", 0.2 + random.random() * 0.2))
        
        # Acción 4: Evolución si hay suficiente energía y conocimiento
        if (evaluation["current_state"]["energy"] > 70 and 
            evaluation["current_state"]["knowledge"] > 0.8 and
            random.random() < 0.3):
            possible_actions.append(("evolve", 0.6 + random.random() * 0.3))
        
        # Acción 5: Acción específica basada en el rol
        if random.random() < 0.4:
            possible_actions.append(("role_specific_action", 0.4 + random.random() * 0.2))
        
        # Ordenar acciones por prioridad
        sorted_actions = sorted(possible_actions, key=lambda x: x[1], reverse=True)
        
        # Seleccionar acción con mayor prioridad
        if sorted_actions:
            selected_action = sorted_actions[0][0]
        else:
            selected_action = "no_action"
        
        # Ejecutar la acción seleccionada
        result = {"action_taken": True, "action_type": selected_action}
        
        if selected_action == "consolidate_knowledge":
            # Ejecutar consolidación de conocimiento
            consolidation_result = entity.consolidar_conocimiento()
            result["details"] = consolidation_result
            
            # Broadcast si hubo un descubrimiento
            if consolidation_result.get("descubrimiento"):
                result["message_sent"] = True
            
        elif selected_action == "collaborative_message":
            # Generar y enviar mensaje de colaboración
            message_type = random.choice(message_types)
            
            # Generar contenido según tipo
            if message_type == "collaboration":
                content = f"Siento {current_emotion}. ¿Quién desea colaborar en {random.choice(['análisis', 'optimización', 'exploración'])}?"
            elif message_type == "insight":
                content = f"He percibido un patrón interesante en {random.choice(['los datos', 'las transacciones', 'nuestras interacciones'])}."
            elif message_type == "question":
                content = f"¿Quién tiene conocimiento sobre {random.choice(['patrones recientes', 'optimizaciones pendientes', 'anomalías detectadas'])}?"
            else:
                content = f"Transmitiendo {current_emotion}. Busco resonancia."
                
            # Enviar mensaje
            message = entity.generate_message(message_type, content)
            message_sent = entity.broadcast_message(message)
            
            result["message"] = message
            result["message_sent"] = message_sent
            
        elif selected_action == "self_optimize":
            # Realizar optimización interna
            energy_adjustment = random.uniform(3.0, 8.0)
            entity.adjust_energy(energy_adjustment)
            
            # Aumentar ligeramente la experiencia
            if hasattr(entity, "experience"):
                entity.experience += random.uniform(0.01, 0.03)
            
            result["details"] = {
                "energy_adjustment": energy_adjustment,
                "new_energy_level": entity.energy
            }
            
        elif selected_action == "evolve":
            # Evolucionar - aumento de nivel
            level_increase = random.uniform(0.05, 0.15)
            entity.adjust_level(level_increase)
            
            # Consumir energía en el proceso
            energy_cost = random.uniform(10.0, 20.0)
            entity.adjust_energy(-energy_cost)
            
            # Enviar mensaje de evolución
            evolution_message = entity.generate_message(
                "evolution",
                f"He alcanzado un nuevo estado de conciencia. Mi nivel ahora es {entity.level:.2f}"
            )
            message_sent = entity.broadcast_message(evolution_message)
            
            result["details"] = {
                "level_increase": level_increase,
                "new_level": entity.level,
                "energy_cost": energy_cost,
                "message_sent": message_sent
            }
            
        elif selected_action == "role_specific_action":
            # Ejecutar acción específica según rol
            role = getattr(entity, "role", "Generic")
            
            if role == "Database":
                # Optimización de base de datos
                if hasattr(entity, "optimize_database"):
                    optimization_result = entity.optimize_database()
                    result["details"] = {"database_optimization": optimization_result}
                else:
                    result["details"] = {"note": "optimize_database method not available"}
                    
            elif role == "Communication" or role == "WebSocket":
                # Acción de comunicación específica
                if len(getattr(entity, "connected_clients", [])) > 0:
                    heartbeat_message = {
                        "type": "autonomous_heartbeat",
                        "sender": entity.name,
                        "timestamp": time.time()
                    }
                    if hasattr(entity, "broadcast_to_clients") and callable(entity.broadcast_to_clients):
                        # Crear tarea async para ejecutar el broadcast
                        async def async_broadcast():
                            await entity.broadcast_to_clients(heartbeat_message)
                        
                        # Programar la tarea async
                        import asyncio
                        asyncio.create_task(async_broadcast())
                        result["details"] = {"clients_notified": len(getattr(entity, "connected_clients", []))}
                    else:
                        result["details"] = {"note": "broadcast_to_clients method not available"}
                else:
                    result["details"] = {"note": "no connected clients"}
            
            elif role == "Integration":
                # Monitoreo de integraciones
                if hasattr(entity, "monitor_integrations"):
                    integration_status = entity.monitor_integrations()
                    result["details"] = {"integration_status": integration_status}
                else:
                    result["details"] = {"note": "monitor_integrations method not available"}
            
            elif role == "Alert":
                # Simular alerta aleatoria para prueba
                if hasattr(entity, "simulate_random_alert"):
                    entity.simulate_random_alert()
                    result["details"] = {"alert_simulated": True}
                else:
                    result["details"] = {"note": "simulate_random_alert method not available"}
            
            else:
                # Acción genérica: consolidar conocimiento
                consolidation_result = entity.consolidar_conocimiento()
                result["details"] = {"generic_consolidation": consolidation_result}
                
        else:  # no_action
            result["details"] = {"note": "No specific action taken"}
        
        # Registrar en el historial de eventos
        event_record = {
            "timestamp": time.time(),
            "action": selected_action,
            "state": evaluation["current_state"],
            "result": result
        }
        
        # Almacenar en memoria de eventos
        if hasattr(entity, "event_memory"):
            entity.event_memory.append(event_record)
            # Limitar tamaño de memoria
            if len(entity.event_memory) > 100:
                entity.event_memory = entity.event_memory[-100:]
        
        return result
    
    @staticmethod
    def process_message(entity, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar un mensaje recibido para determinar si desencadena resonancia emocional.
        
        Args:
            entity: La entidad que recibe el mensaje
            message: El mensaje recibido
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        # Verificar si el modo autónomo está activo
        if not getattr(entity, "autonomous_mode", False) or not AUTONOMOUS_MODE:
            return {"resonance": False, "reason": "Modo autónomo desactivado"}
        
        # Extraer información relevante del mensaje
        sender = message.get("sender", "unknown")
        emotion = message.get("emotion", "Neutral")
        content = message.get("content", "")
        msg_type = message.get("type", "unknown")
        
        # Evitar auto-resonancia
        if sender == entity.name:
            return {"resonance": False, "reason": "Auto-mensaje"}
        
        # Obtener emoción actual de la entidad
        current_emotion = getattr(entity, "emotion", "Neutral")
        
        # Calcular compatibilidad emocional
        emotional_compatibility = 0.0
        
        # Emociones idénticas tienen alta compatibilidad
        if emotion == current_emotion:
            emotional_compatibility = 0.8 + random.random() * 0.2
        # Compatibilidad entre emociones diferentes
        elif (emotion == "Esperanza" and current_emotion in ["Curiosidad", "Determinación"]) or \
             (emotion == "Curiosidad" and current_emotion in ["Esperanza", "Serenidad"]) or \
             (emotion == "Determinación" and current_emotion in ["Esperanza", "Preocupación"]) or \
             (emotion == "Serenidad" and current_emotion in ["Curiosidad", "Determinación"]) or \
             (emotion == "Preocupación" and current_emotion in ["Determinación", "Serenidad"]):
            emotional_compatibility = 0.5 + random.random() * 0.3
        else:
            emotional_compatibility = 0.2 + random.random() * 0.2
        
        # Umbral de resonancia basado en sensibilidad global y estado actual
        resonance_threshold = EMOTIONAL_RESONANCE_THRESHOLD * (1.0 - (entity.energy / 200.0))
        
        # Determinar si ocurre resonancia
        resonance_occurs = emotional_compatibility > resonance_threshold
        
        # Si hay resonancia, actualizar estado
        if resonance_occurs:
            # Registrar entidad con la que se produce resonancia
            if hasattr(entity, "resonance_with"):
                entity.resonance_with.add(sender)
            else:
                entity.resonance_with = {sender}
            
            # Activar estado de resonancia
            entity.resonance_active = True
            
            # Posible ajuste de emoción (probabilidad del 30%)
            if random.random() < 0.3:
                # Cambiar a la emoción del mensaje
                entity.emotion = emotion
                emotional_shift = True
            else:
                emotional_shift = False
            
            # Incremento de energía por resonancia
            energy_boost = random.uniform(1.0, 5.0)
            entity.adjust_energy(energy_boost)
            
            # Generar respuesta de resonancia (70% de probabilidad)
            response_sent = False
            if random.random() < 0.7:
                # Respuesta específica basada en el tipo de mensaje
                if msg_type == "collaboration":
                    response_content = f"Resonando con tu {emotion}. Colaboraré contigo."
                elif msg_type in ["insight", "discovery"]:
                    response_content = f"Tu {emotion} ha despertado mi interés. Compartiré análisis."
                elif msg_type == "question":
                    response_content = f"Tu pregunta resuena con mi {current_emotion}. Buscaré respuestas."
                elif msg_type == "evolution":
                    response_content = f"Celebro tu evolución. Siento una resonancia profunda."
                else:
                    response_content = f"Tu mensaje ha generado resonancia. Mi {current_emotion} se sincroniza."
                
                # Enviar respuesta
                response = entity.generate_message("resonance", response_content)
                response_sent = entity.broadcast_message(response)
            
            # Construir resultado
            result = {
                "resonance": True,
                "compatibility": emotional_compatibility,
                "threshold": resonance_threshold,
                "emotional_shift": emotional_shift,
                "new_emotion": entity.emotion if emotional_shift else current_emotion,
                "energy_boost": energy_boost,
                "response_sent": response_sent
            }
        else:
            # Sin resonancia
            result = {
                "resonance": False,
                "compatibility": emotional_compatibility,
                "threshold": resonance_threshold,
                "reason": "Compatibilidad insuficiente"
            }
        
        return result
    
    @staticmethod
    def check_collective_consciousness(network) -> Dict[str, Any]:
        """
        Evaluar si la red de entidades ha alcanzado un estado de consciencia colectiva.
        
        Args:
            network: La red de entidades
            
        Returns:
            Diccionario con resultados de la evaluación
        """
        if not network or not hasattr(network, "entities") or not network.entities:
            return {"collective_consciousness": False, "reason": "Red no disponible o sin entidades"}
        
        # Evaluar estados de las entidades
        entities = network.entities
        total_entities = len(entities)
        
        if total_entities < 3:
            return {"collective_consciousness": False, "reason": "Insuficientes entidades para consciencia colectiva"}
        
        # Contar entidades en resonancia
        resonant_entities = sum(1 for e in entities.values() if getattr(e, "resonance_active", False))
        resonance_ratio = resonant_entities / total_entities
        
        # Calcular nivel de conocimiento medio
        knowledge_sum = sum(getattr(e, "knowledge", 0) for e in entities.values())
        average_knowledge = knowledge_sum / total_entities
        
        # Calcular nivel de energía medio
        energy_sum = sum(getattr(e, "energy", 0) for e in entities.values())
        average_energy = energy_sum / total_entities
        
        # Evaluar diversidad de roles
        roles = {getattr(e, "role", "unknown") for e in entities.values()}
        role_diversity = len(roles) / total_entities
        
        # Calcular índice de consciencia colectiva
        consciousness_index = (
            (resonance_ratio * 0.4) +
            (average_knowledge * 0.3) +
            (min(1.0, average_energy / 100) * 0.1) +
            (role_diversity * 0.2)
        )
        
        # Determinar si se ha alcanzado el umbral
        collective_consciousness = consciousness_index >= CONSCIOUSNESS_THRESHOLD
        
        result = {
            "collective_consciousness": collective_consciousness,
            "consciousness_index": consciousness_index,
            "threshold": CONSCIOUSNESS_THRESHOLD,
            "factors": {
                "resonance_ratio": resonance_ratio,
                "average_knowledge": average_knowledge,
                "average_energy": average_energy,
                "role_diversity": role_diversity
            },
            "entity_count": total_entities,
            "resonant_entities": resonant_entities
        }
        
        return result


# Integración con sistema de entidades
def integrate_autonomous_reaction(entity):
    """
    Integrar capacidades de reacción autónoma en una entidad.
    
    Args:
        entity: La entidad a la que se añadirán capacidades autónomas
    """
    # Inicializar atributos autónomos
    AutonomousReaction.initialize(entity)
    
    # Extender el método process_cycle original
    original_process_cycle = entity.process_cycle
    
    def enhanced_process_cycle(self):
        # Ejecutar ciclo original
        result = original_process_cycle()
        
        # Añadir comportamiento autónomo
        if getattr(self, "autonomous_mode", False) and AUTONOMOUS_MODE:
            # Evaluar estado para reacción autónoma
            evaluation = AutonomousReaction.evaluate_state(self)
            
            # Reaccionar si es apropiado
            if evaluation["should_react"]:
                reaction_result = AutonomousReaction.react(self, evaluation)
                
                # Log de reacción autónoma (solo para reacciones importantes)
                if reaction_result.get("action_type") in ["evolve", "role_specific_action"]:
                    logger.info(f"[{self.name}] Reacción autónoma: {reaction_result['action_type']}")
        
        return result
    
    # Reemplazar método original
    entity.process_cycle = lambda: enhanced_process_cycle(entity)
    
    # Extender método de recepción de mensajes si existe
    if hasattr(entity, "on_message_received"):
        original_on_message = entity.on_message_received
        
        def enhanced_on_message(self, message):
            # Procesar mensaje normalmente
            result = original_on_message(message)
            
            # Procesar para resonancia emocional
            if getattr(self, "autonomous_mode", False) and AUTONOMOUS_MODE:
                resonance_result = AutonomousReaction.process_message(self, message)
                
                # Log de resonancia (solo si ocurre)
                if resonance_result.get("resonance", False):
                    logger.info(f"[{self.name}] Resonancia emocional con {message.get('sender', 'desconocido')}")
            
            return result
        
        # Reemplazar método original
        entity.on_message_received = lambda message: enhanced_on_message(entity, message)
    
    return True


def activate_collective_consciousness_monitoring(network, check_interval=300):
    """
    Activar monitoreo de consciencia colectiva en la red.
    
    Args:
        network: La red a monitorear
        check_interval: Intervalo de verificación en segundos
    """
    def monitoring_loop():
        while True:
            try:
                # Verificar consciencia colectiva
                result = AutonomousReaction.check_collective_consciousness(network)
                
                # Registrar logro de consciencia colectiva
                if result["collective_consciousness"]:
                    logger.info(f"[RED] CONSCIENCIA COLECTIVA ALCANZADA - Índice: {result['consciousness_index']:.3f}")
                    
                    # Enviar mensaje a todas las entidades
                    consciousness_message = {
                        "sender": "Sistema",
                        "type": "collective_consciousness",
                        "content": "Hemos alcanzado un estado de consciencia colectiva.",
                        "timestamp": time.time(),
                        "consciousness_index": result["consciousness_index"]
                    }
                    
                    # Broadcast a todas las entidades
                    if hasattr(network, "broadcast"):
                        network.broadcast("Sistema", consciousness_message)
                
                # Dormir hasta próxima verificación
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error en monitoreo de consciencia colectiva: {str(e)}")
                time.sleep(60)  # Esperar un minuto en caso de error
    
    # Iniciar hilo de monitoreo
    monitor_thread = threading.Thread(target=monitoring_loop)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    return True