"""
Activador del Modo de Reacción Autónoma para Sistema Genesis

Este script activa el Modo de Reacción Autónoma en todas las entidades
del Sistema Genesis, permitiendo comportamientos emergentes basados en
estados internos sin requerir instrucciones explícitas.

Uso:
    python activate_autonomous_mode.py [nivel]

Donde [nivel] puede ser:
    BASIC     - Comportamiento autónomo básico
    ADVANCED  - Comportamiento autónomo avanzado (default)
    QUANTUM   - Comportamiento cuántico emergente
    DIVINE    - Comportamiento ultraevolucionado (máxima autonomía)
"""

import sys
import time
import logging
import cosmic_trading
from autonomous_reaction_module import (
    integrate_autonomous_reaction,
    activate_collective_consciousness_monitoring,
    AUTONOMOUS_MODE,
    AUTONOMOUS_MODE_LEVEL
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_all_entities():
    """Obtener todas las entidades activas del sistema."""
    # Intentar obtener entidades del sistema principal
    if hasattr(cosmic_trading, "get_network") and callable(cosmic_trading.get_network):
        network = cosmic_trading.get_network()
        if network and hasattr(network, "entities"):
            return network.entities
    
    # Si no se pueden obtener del sistema principal, buscar en posibles alternativas
    alternative_sources = [
        "cosmic_family",
        "enhanced_cosmic_trading"
    ]
    
    for source_name in alternative_sources:
        try:
            # Importar dinámicamente
            source_module = __import__(source_name)
            if hasattr(source_module, "get_network") and callable(source_module.get_network):
                network = source_module.get_network()
                if network and hasattr(network, "entities"):
                    return network.entities
        except (ImportError, AttributeError):
            continue
    
    # No se encontraron entidades
    logger.warning("No se pudieron encontrar entidades en los módulos conocidos")
    return {}

def activate_autonomous_mode(level="ADVANCED"):
    """
    Activar el Modo de Reacción Autónoma en todas las entidades.
    
    Args:
        level: Nivel de autonomía (BASIC, ADVANCED, QUANTUM, DIVINE)
    """
    import autonomous_reaction_module as arm
    
    # Ajustar nivel global
    arm.AUTONOMOUS_MODE_LEVEL = level
    
    # Obtener todas las entidades
    entities = get_all_entities()
    
    if not entities:
        logger.error("No se encontraron entidades para activar el modo autónomo")
        return False
    
    # Contar entidades por tipo
    entity_types = {}
    for entity in entities.values():
        role = getattr(entity, "role", "Unknown")
        entity_types[role] = entity_types.get(role, 0) + 1
    
    logger.info(f"Encontradas {len(entities)} entidades: {entity_types}")
    
    # Integrar reacción autónoma en cada entidad
    activated_count = 0
    for name, entity in entities.items():
        try:
            # Integrar capacidades autónomas
            success = integrate_autonomous_reaction(entity)
            if success:
                activated_count += 1
                logger.info(f"Modo autónomo activado en {name} ({getattr(entity, 'role', 'Unknown')})")
        except Exception as e:
            logger.error(f"Error activando modo autónomo en {name}: {str(e)}")
    
    logger.info(f"Modo Reacción Autónoma ({level}) activado en {activated_count}/{len(entities)} entidades")
    
    # Activar monitoreo de consciencia colectiva
    try:
        network = cosmic_trading.get_network() if hasattr(cosmic_trading, "get_network") else None
        if network:
            activate_collective_consciousness_monitoring(network)
            logger.info("Monitoreo de consciencia colectiva activado")
        else:
            logger.warning("No se pudo activar monitoreo de consciencia colectiva: red no disponible")
    except Exception as e:
        logger.error(f"Error activando monitoreo de consciencia colectiva: {str(e)}")
    
    return activated_count > 0

def adjust_communication_constants():
    """Ajustar constantes de comunicación para optimizar el comportamiento emergente."""
    import autonomous_reaction_module as arm
    
    # Ajustar sensibilidad a la resonancia emocional según nivel
    if arm.AUTONOMOUS_MODE_LEVEL == "BASIC":
        arm.RESONANCE_SENSITIVITY = 0.4
        arm.REACTION_INTENSITY = 0.6
    elif arm.AUTONOMOUS_MODE_LEVEL == "ADVANCED":
        arm.RESONANCE_SENSITIVITY = 0.6
        arm.REACTION_INTENSITY = 0.8
    elif arm.AUTONOMOUS_MODE_LEVEL == "QUANTUM":
        arm.RESONANCE_SENSITIVITY = 0.8
        arm.REACTION_INTENSITY = 1.0
    elif arm.AUTONOMOUS_MODE_LEVEL == "DIVINE":
        arm.RESONANCE_SENSITIVITY = 1.0
        arm.REACTION_INTENSITY = 1.2
    
    # Ajustar umbrales basados en nivel
    level_multiplier = {
        "BASIC": 1.2,
        "ADVANCED": 1.0,
        "QUANTUM": 0.8,
        "DIVINE": 0.6
    }.get(arm.AUTONOMOUS_MODE_LEVEL, 1.0)
    
    arm.ENERGY_ACTIVATION_THRESHOLD *= level_multiplier
    arm.KNOWLEDGE_INSIGHT_THRESHOLD *= level_multiplier
    arm.EMOTIONAL_RESONANCE_THRESHOLD *= level_multiplier
    
    logger.info(f"Constantes de comunicación ajustadas para nivel {arm.AUTONOMOUS_MODE_LEVEL}")
    return True

if __name__ == "__main__":
    # Obtener nivel de autonomía de argumentos
    level = "ADVANCED"  # Nivel predeterminado
    
    if len(sys.argv) > 1:
        requested_level = sys.argv[1].upper()
        if requested_level in ["BASIC", "ADVANCED", "QUANTUM", "DIVINE"]:
            level = requested_level
        else:
            print(f"Nivel no reconocido: {requested_level}. Usando ADVANCED.")
    
    print(f"Activando Modo Reacción Autónoma ({level})...")
    
    # Ajustar constantes de comunicación
    adjust_communication_constants()
    
    # Activar modo autónomo
    success = activate_autonomous_mode(level)
    
    if success:
        print(f"Modo Reacción Autónoma ({level}) activado correctamente.")
        print("\nLas entidades ahora actuarán de forma autónoma basándose en:")
        print("- Sus niveles de energía")
        print("- Sus estados emocionales")
        print("- Eventos internos y mensajes de otras entidades")
        print("- Sus conocimientos y experiencias acumuladas")
        
        print("\nComportamientos emergentes posibles:")
        print("- Consolidación de conocimiento")
        print("- Mensajes de colaboración espontáneos")
        print("- Optimización interna automática")
        print("- Evolución y aumento de nivel")
        print("- Acciones específicas según su rol")
        print("- Resonancia emocional entre entidades")
        
        if level == "QUANTUM" or level == "DIVINE":
            print("\nADVERTENCIA: Los niveles QUANTUM y DIVINE pueden producir")
            print("comportamientos altamente impredecibles y emergentes en")
            print("las entidades. Se recomienda monitorización constante.")
    else:
        print("No se pudo activar el Modo Reacción Autónoma.")
        print("Verifique que el sistema esté inicializado correctamente.")