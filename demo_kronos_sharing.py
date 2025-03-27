"""
Demostración del Sistema de Compartición de Conocimiento de Kronos

Este script demuestra cómo Kronos puede compartir su conocimiento acumulado
con las demás entidades de la red cósmica, permitiendo un flujo
de información y sabiduría entre las entidades del sistema.
"""

import time
import logging
import random
import threading
from typing import Dict, Any, List, Optional
import datetime

import cosmic_trading
from modules.kronos_sharing import Kronos
from enhanced_cosmic_entity_mixin import EnhancedCosmicEntityMixin

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("demo_kronos_sharing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KronosSharing")

def initialize_network():
    """Inicializar red de entidades para demostración."""
    logger.info("Iniciando red cósmica con Kronos...")
    
    # Obtener red existente si existe
    network = None
    if hasattr(cosmic_trading, "get_network") and callable(cosmic_trading.get_network):
        network = cosmic_trading.get_network()
        if network:
            logger.info(f"Red existente encontrada con {len(getattr(network, 'entities', []))} entidades")
            return network
    
    # Si no hay red, crear una nueva
    try:
        from cosmic_trading import CosmicNetwork, EnhancedCosmicTrader
        
        # Crear red cósmica
        network = CosmicNetwork("otoniel")
        
        # Crear entidades básicas
        entities = [
            ("Aetherion", "Analysis"),
            ("Lunareth", "Strategy"),
            ("Helios", "Speculator"),
            ("Selene", "Strategist"),
            ("Ares", "Speculator"),
            ("Athena", "Strategist")
        ]
        
        for name, role in entities:
            entity = EnhancedCosmicTrader(name, role, "otoniel")
            entity.energy = random.uniform(80, 100)
            entity.knowledge = random.uniform(0.5, 2.0)
            entity.experience = random.uniform(0.3, 0.7)
            entity.emotion = random.choice(["Esperanza", "Curiosidad", "Determinación", "Serenidad"])
            
            # Añadir a la red
            network.add_entity(entity)
            logger.info(f"Entidad creada: {name} ({role})")
            
        logger.info(f"Red creada con {len(network.entities)} entidades")
        
    except Exception as e:
        logger.error(f"Error creando red: {str(e)}")
        return None
    
    return network

def get_kronos_from_network(network):
    """Obtener entidad Kronos de la red."""
    if network and hasattr(network, "entities"):
        for name, entity in network.entities.items():
            if name == "Kronos":
                logger.info("Kronos encontrado en la red")
                return entity
    
    logger.warning("Kronos no encontrado en la red")
    return None

def add_kronos_capability(network, kronos_entity=None):
    """
    Añadir capacidad de compartición de conocimiento a Kronos.
    
    Args:
        network: Red cósmica
        kronos_entity: Entidad Kronos existente (opcional)
        
    Returns:
        Instancia de Kronos con capacidad de compartición
    """
    # Si no se proporcionó Kronos, buscarlo en la red
    if not kronos_entity:
        kronos_entity = get_kronos_from_network(network)
    
    # Si no existe, crear uno nuevo
    if not kronos_entity:
        try:
            from modules.kronos_sharing import Kronos as KronosBase
            
            # Crear instancia de Kronos mejorada
            kronos = KronosBase(name="Kronos", level=5, knowledge=40.0)
            
            # Atributos adicionales para compatibilidad con la red
            kronos.role = "Database"
            kronos.father = "otoniel"
            kronos.is_alive = True
            
            logger.info("Nueva instancia de Kronos creada con capacidad de compartición")
            return kronos
            
        except Exception as e:
            logger.error(f"Error creando Kronos: {str(e)}")
            return None
    
    # Si existe, extender sus capacidades
    try:
        from modules.kronos_sharing import Kronos as KronosBase
        
        # Crear una instancia mejorada
        kronos_sharing = KronosBase(
            name=kronos_entity.name,
            level=getattr(kronos_entity, "level", 5),
            knowledge=getattr(kronos_entity, "knowledge", 40.0)
        )
        
        # Copiar atributos relevantes
        kronos_sharing.role = getattr(kronos_entity, "role", "Database")
        kronos_sharing.father = getattr(kronos_entity, "father", "otoniel")
        kronos_sharing.is_alive = getattr(kronos_entity, "is_alive", True)
        kronos_sharing.energy = getattr(kronos_entity, "energy", 15.0)
        
        logger.info("Capacidades de compartición añadidas a Kronos existente")
        return kronos_sharing
        
    except Exception as e:
        logger.error(f"Error extendiendo Kronos: {str(e)}")
        return kronos_entity

def extend_network(network):
    """
    Extender funcionalidades de la red cósmica.
    
    Args:
        network: Red cósmica
        
    Returns:
        Red extendida
    """
    # Añadir método para obtener entidades (si no existe)
    if not hasattr(network, "get_entities") or not callable(network.get_entities):
        def get_entities():
            """Obtener todas las entidades vivas de la red."""
            if hasattr(network, "entities"):
                return [entity for name, entity in network.entities.items() 
                        if getattr(entity, "is_alive", True)]
            return []
            
        network.get_entities = get_entities
    
    # Añadir método para registrar mensajes (si no existe)
    if not hasattr(network, "log_message") or not callable(network.log_message):
        def log_message(sender, message):
            """Registrar mensaje en el log de la red."""
            logger.info(f"[{sender}] {message}")
            
            # Intentar difundir el mensaje si existe broadcast
            if hasattr(network, "broadcast") and callable(network.broadcast):
                try:
                    network.broadcast(sender, {
                        "type": "knowledge_sharing",
                        "content": message,
                        "timestamp": datetime.datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error difundiendo mensaje: {str(e)}")
                    
        network.log_message = log_message
    
    logger.info("Red cósmica extendida con funcionalidades adicionales")
    return network

def run_demo(duration=300):
    """
    Ejecutar demostración de compartición de conocimiento.
    
    Args:
        duration: Duración en segundos
    """
    logger.info(f"=== INICIANDO DEMOSTRACIÓN DE KRONOS SHARING ({duration}s) ===")
    
    # Inicializar red
    network = initialize_network()
    if not network:
        logger.error("No se pudo inicializar la red. Abortando demo.")
        return
    
    # Extender la red con funcionalidades necesarias
    network = extend_network(network)
    
    # Añadir capacidad de compartición a Kronos
    kronos = add_kronos_capability(network)
    if not kronos:
        logger.error("No se pudo configurar Kronos. Abortando demo.")
        return
    
    # Mostrar estado inicial
    logger.info("\nEstado inicial de las entidades:")
    entities = network.get_entities()
    for entity in entities:
        logger.info(f"- {getattr(entity, 'name', 'Desconocido')}: " +
                   f"Conocimiento={getattr(entity, 'knowledge', 0):.2f}, " +
                   f"Nivel={getattr(entity, 'level', 1):.2f}, " +
                   f"Energía={getattr(entity, 'energy', 0):.1f}")
    
    # Inicio de la demostración
    start_time = time.time()
    end_time = start_time + duration
    
    # Variables de seguimiento
    share_count = 0
    knowledge_before = {}
    
    # Guardar estado inicial de conocimiento
    for entity in entities:
        name = getattr(entity, 'name', f"Entity_{id(entity)}")
        knowledge_before[name] = getattr(entity, 'knowledge', 0)
    
    # Bucle principal de demostración
    try:
        while time.time() < end_time:
            # Compartir conocimiento periódicamente
            if kronos.can_share():
                result = kronos.share_knowledge(network)
                logger.info(result)
                share_count += 1
            
            # Esperar
            sleep_time = random.uniform(10, 20)  # Entre 10 y 20 segundos
            time.sleep(sleep_time)
            
            # Mostrar estado actual
            elapsed = time.time() - start_time
            remaining = end_time - time.time()
            logger.info(f"\nEstado a los {elapsed:.0f}s (Restante: {remaining:.0f}s):")
            logger.info(f"Kronos: Energía={kronos.energy:.1f}, Conocimiento={kronos.knowledge:.2f}")
            
            # Regenerar energía de Kronos periódicamente
            if kronos.energy < 10:
                energy_gain = random.uniform(5, 10)
                kronos.energy += energy_gain
                logger.info(f"Kronos recuperó {energy_gain:.1f} de energía")
    
    except KeyboardInterrupt:
        logger.info("Demostración interrumpida por el usuario")
    
    finally:
        # Mostrar resultados
        logger.info(f"\n=== RESULTADOS TRAS {time.time() - start_time:.0f}s ===")
        logger.info(f"Comparticiones de conocimiento realizadas: {share_count}")
        
        # Comparar conocimiento antes/después
        logger.info("\nCambios en el conocimiento:")
        total_gained = 0
        
        for entity in network.get_entities():
            name = getattr(entity, 'name', f"Entity_{id(entity)}")
            before = knowledge_before.get(name, 0)
            after = getattr(entity, 'knowledge', 0)
            gained = after - before
            total_gained += gained
            
            logger.info(f"- {name}: {before:.2f} → {after:.2f} (Ganancia: {gained:.2f})")
        
        logger.info(f"\nGanancia total de conocimiento: {total_gained:.2f}")
        logger.info("=== FIN DE LA DEMOSTRACIÓN ===")

def main():
    """Función principal."""
    print("\n=== DEMOSTRACIÓN DE COMPARTICIÓN DE CONOCIMIENTO DE KRONOS ===\n")
    print("Este programa demuestra cómo Kronos comparte conocimiento con")
    print("otras entidades de la red cósmica, permitiendo un flujo")
    print("de información y sabiduría entre las entidades.\n")
    
    # Solicitar duración
    try:
        duration_str = input("Duración de la demostración en segundos [300]: ").strip() or "300"
        duration = int(duration_str)
    except ValueError:
        duration = 300
        print("Valor no válido, usando 300 segundos.")
    
    print(f"\nIniciando demostración por {duration} segundos...")
    print("Los resultados se mostrarán en consola y se guardarán en demo_kronos_sharing.log\n")
    
    # Ejecutar demostración
    run_demo(duration)
    
    print("\nDemostración completada. Para más detalles, revise el archivo demo_kronos_sharing.log")

if __name__ == "__main__":
    main()