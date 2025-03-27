"""
Demostración del Modo Reacción Autónoma del Sistema Genesis

Este script realiza una demostración del comportamiento emergente
de las entidades en el Modo Reacción Autónoma, observando sus 
interacciones autónomas durante un período de tiempo.
"""

import time
import logging
import random
import threading
import cosmic_trading
from autonomous_reaction_module import (
    AUTONOMOUS_MODE, 
    AUTONOMOUS_MODE_LEVEL,
    AutonomousReaction
)
from activate_autonomous_mode import activate_autonomous_mode

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("demo_autonomous_mode.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_demo():
    """Inicializar demostración con entidades básicas."""
    logger.info("Iniciando demostración del Modo Reacción Autónoma...")
    
    # Intentar obtener red existente
    network = None
    if hasattr(cosmic_trading, "get_network") and callable(cosmic_trading.get_network):
        network = cosmic_trading.get_network()
    
    # Si no hay red o entidades, crear nuevas
    if not network or not hasattr(network, "entities") or not network.entities:
        logger.warning("No se encontró una red existente con entidades. Creando nuevas entidades...")
        
        # Importar módulos necesarios
        try:
            from cosmic_trading import CosmicNetwork, EnhancedCosmicTrader
            
            # Crear red
            network = CosmicNetwork("otoniel")
            
            # Crear entidades básicas
            entities = [
                ("Aetherion", "Analysis"),
                ("Lunareth", "Strategy"),
                ("Kronos", "Database"),
                ("Hermes", "Communication"),
                ("Harmonia", "Integration"),
                ("Sentinel", "Alert")
            ]
            
            for name, role in entities:
                entity = EnhancedCosmicTrader(name, role, "otoniel")
                entity.energy = 100
                entity.emotion = random.choice(["Esperanza", "Curiosidad", "Determinación", "Serenidad"])
                entity.knowledge = random.uniform(0.2, 0.5)
                entity.experience = random.uniform(0.1, 0.3)
                
                # Añadir a la red
                network.add_entity(entity)
                logger.info(f"Entidad creada: {name} ({role})")
            
            logger.info(f"Red creada con {len(network.entities)} entidades")
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Error creando entidades: {str(e)}")
            return None
    
    return network

def monitor_autonomous_behavior(network, duration=300):
    """
    Monitorear comportamiento autónomo durante un período.
    
    Args:
        network: Red de entidades
        duration: Duración en segundos
    """
    if not network or not hasattr(network, "entities"):
        logger.error("Red no válida para monitoreo")
        return
    
    logger.info(f"Iniciando monitoreo de comportamiento autónomo por {duration} segundos...")
    
    # Iniciar todas las entidades
    for name, entity in network.entities.items():
        if hasattr(entity, "start_lifecycle") and callable(entity.start_lifecycle):
            entity.start_lifecycle()
            logger.info(f"Ciclo de vida iniciado para {name}")
    
    # Mostrar estado inicial
    logger.info("Estado inicial de las entidades:")
    for name, entity in network.entities.items():
        logger.info(f"- {name}: Energía={entity.energy:.1f}, Emoción={getattr(entity, 'emotion', 'Neutral')}, Conocimiento={getattr(entity, 'knowledge', 0):.2f}")
    
    # Variables de monitoreo
    start_time = time.time()
    end_time = start_time + duration
    check_interval = 10  # segundos
    
    # Contadores de eventos
    total_autonomous_actions = 0
    evolution_events = 0
    knowledge_insights = 0
    energy_changes = {}
    emotion_changes = {}
    resonance_events = 0
    
    # Estado inicial para comparación
    initial_states = {}
    for name, entity in network.entities.items():
        initial_states[name] = {
            "energy": entity.energy,
            "emotion": getattr(entity, "emotion", "Neutral"),
            "knowledge": getattr(entity, "knowledge", 0),
            "level": entity.level
        }
        # Inicializar contadores
        energy_changes[name] = 0
        emotion_changes[name] = 0
    
    # Monitoreo principal
    try:
        while time.time() < end_time:
            time.sleep(check_interval)
            
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = end_time - current_time
            
            logger.info(f"Monitoreo en progreso - Transcurrido: {elapsed:.0f}s, Restante: {remaining:.0f}s")
            
            # Verificar estado actual de cada entidad
            active_entities = 0
            for name, entity in network.entities.items():
                if not getattr(entity, "is_alive", False):
                    continue
                
                active_entities += 1
                
                # Comparar con estado anterior
                if hasattr(entity, "autonomous_actions_count"):
                    new_actions = entity.autonomous_actions_count - total_autonomous_actions
                    if new_actions > 0:
                        logger.info(f"[{name}] {new_actions} acciones autónomas realizadas")
                        total_autonomous_actions = entity.autonomous_actions_count
                
                # Comprobar cambios en energía
                if abs(entity.energy - initial_states[name]["energy"]) > 5:
                    energy_changes[name] += 1
                    logger.info(f"[{name}] Cambio de energía: {initial_states[name]['energy']:.1f} -> {entity.energy:.1f}")
                    initial_states[name]["energy"] = entity.energy
                
                # Comprobar cambios en emoción
                current_emotion = getattr(entity, "emotion", "Neutral")
                if current_emotion != initial_states[name]["emotion"]:
                    emotion_changes[name] += 1
                    logger.info(f"[{name}] Cambio de emoción: {initial_states[name]['emotion']} -> {current_emotion}")
                    initial_states[name]["emotion"] = current_emotion
                
                # Comprobar aumento de conocimiento
                current_knowledge = getattr(entity, "knowledge", 0)
                if current_knowledge - initial_states[name]["knowledge"] > 0.2:
                    knowledge_insights += 1
                    logger.info(f"[{name}] Incremento significativo de conocimiento: {initial_states[name]['knowledge']:.2f} -> {current_knowledge:.2f}")
                    initial_states[name]["knowledge"] = current_knowledge
                
                # Comprobar evolución
                if entity.level - initial_states[name]["level"] > 0.05:
                    evolution_events += 1
                    logger.info(f"[{name}] Evolución detectada: Nivel {initial_states[name]['level']:.2f} -> {entity.level:.2f}")
                    initial_states[name]["level"] = entity.level
                
                # Comprobar resonancia
                if getattr(entity, "resonance_active", False):
                    resonance_entities = getattr(entity, "resonance_with", set())
                    if resonance_entities:
                        resonance_events += 1
                        logger.info(f"[{name}] Resonancia emocional con: {', '.join(resonance_entities)}")
            
            # Verificar si todas las entidades perdieron energía
            if active_entities == 0:
                logger.warning("Todas las entidades sin energía. Finalizando monitoreo.")
                break
    
    except KeyboardInterrupt:
        logger.info("Monitoreo interrumpido por usuario")
    
    finally:
        # Detener ciclos de vida
        for name, entity in network.entities.items():
            if hasattr(entity, "stop_lifecycle") and callable(entity.stop_lifecycle):
                entity.stop_lifecycle()
        
        # Mostrar resumen
        actual_duration = time.time() - start_time
        logger.info(f"\nResumen de comportamiento autónomo ({actual_duration:.1f} segundos):")
        logger.info(f"- Acciones autónomas totales: {total_autonomous_actions}")
        logger.info(f"- Eventos de evolución: {evolution_events}")
        logger.info(f"- Insights de conocimiento: {knowledge_insights}")
        logger.info(f"- Eventos de resonancia emocional: {resonance_events}")
        
        # Mostrar cambios por entidad
        logger.info("\nCambios por entidad:")
        for name, entity in network.entities.items():
            logger.info(f"- {name}:")
            logger.info(f"  * Energía inicial: {initial_states[name]['energy']:.1f}, final: {entity.energy:.1f}")
            logger.info(f"  * Emoción inicial: {initial_states[name]['emotion']}, final: {getattr(entity, 'emotion', 'Neutral')}")
            logger.info(f"  * Conocimiento inicial: {initial_states[name]['knowledge']:.2f}, final: {getattr(entity, 'knowledge', 0):.2f}")
            logger.info(f"  * Nivel inicial: {initial_states[name]['level']:.2f}, final: {entity.level:.2f}")
            logger.info(f"  * Cambios de energía: {energy_changes[name]}")
            logger.info(f"  * Cambios emocionales: {emotion_changes[name]}")
        
        return {
            "duration": actual_duration,
            "autonomous_actions": total_autonomous_actions,
            "evolution_events": evolution_events,
            "knowledge_insights": knowledge_insights,
            "resonance_events": resonance_events
        }

def main():
    """Función principal de demostración."""
    print("\n===== DEMOSTRACIÓN DEL MODO REACCIÓN AUTÓNOMA =====\n")
    print("Este programa activará el Modo Reacción Autónoma en el Sistema Genesis")
    print("y mostrará el comportamiento emergente de las entidades durante un período")
    print("de tiempo, registrando sus acciones, evoluciones y resonancias emocionales.\n")
    
    # Obtener nivel de autonomía
    level_options = {
        "1": "BASIC",
        "2": "ADVANCED",
        "3": "QUANTUM",
        "4": "DIVINE"
    }
    
    print("Seleccione nivel de autonomía:")
    print("1. BASIC    - Comportamiento autónomo básico")
    print("2. ADVANCED - Comportamiento autónomo avanzado (recomendado)")
    print("3. QUANTUM  - Comportamiento cuántico emergente (experimental)")
    print("4. DIVINE   - Comportamiento ultraevolucionado (máxima autonomía)")
    
    choice = input("\nOpción (1-4) [2]: ").strip() or "2"
    level = level_options.get(choice, "ADVANCED")
    
    # Duración de la demostración
    try:
        duration_str = input("\nDuración de la demostración en segundos [300]: ").strip() or "300"
        duration = int(duration_str)
    except ValueError:
        duration = 300
        print("Valor no válido, usando 300 segundos.")
    
    print(f"\nIniciando demostración con nivel {level} por {duration} segundos...")
    print("Los resultados se mostrarán en consola y se guardarán en demo_autonomous_mode.log\n")
    
    # Inicializar red
    network = initialize_demo()
    if not network:
        print("Error inicializando la demostración. Verifique logs.")
        return
    
    # Activar modo autónomo
    success = activate_autonomous_mode(level)
    if not success:
        print("Error activando el Modo Reacción Autónoma. Verifique logs.")
        return
    
    # Ejecutar demostración
    try:
        results = monitor_autonomous_behavior(network, duration)
        
        if results:
            print("\n===== RESULTADOS DE LA DEMOSTRACIÓN =====")
            print(f"Duración real: {results['duration']:.1f} segundos")
            print(f"Acciones autónomas: {results['autonomous_actions']}")
            print(f"Eventos de evolución: {results['evolution_events']}")
            print(f"Insights de conocimiento: {results['knowledge_insights']}")
            print(f"Eventos de resonancia emocional: {results['resonance_events']}")
            
            print("\nPara más detalles, revise el archivo demo_autonomous_mode.log")
    
    except Exception as e:
        print(f"Error durante la demostración: {str(e)}")
        logger.exception("Error durante la demostración")

if __name__ == "__main__":
    main()