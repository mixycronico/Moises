#!/usr/bin/env python3
"""
Demostración del Sistema de Competencia Cósmica para el Sistema Genesis

Este script muestra cómo las entidades del sistema pueden competir entre sí
basándose en sus capacidades y conocimientos, con el ganador compartiendo
su sabiduría con las demás entidades.
"""

import time
import random
import logging
import threading
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo_competition")

def initialize_network():
    """Inicializar red de entidades para demostración."""
    logger.info("Iniciando red cósmica para competición...")
    
    # Obtener red existente si existe
    network = None
    try:
        import cosmic_trading
        if hasattr(cosmic_trading, "get_network") and callable(cosmic_trading.get_network):
            network = cosmic_trading.get_network()
            if network:
                logger.info(f"Red existente encontrada con {len(getattr(network, 'entities', []))} entidades")
                return network
    except (ImportError, AttributeError):
        pass
    
    # Si no hay red, crear una nueva
    try:
        from cosmic_trading import CosmicNetwork, initialize_cosmic_trading
        
        # Intentar inicializar desde el método existente
        logger.info("Creando nueva red cósmica desde initialize_cosmic_trading")
        network, _, _ = initialize_cosmic_trading(include_extended_entities=True)
        return network
    except (ImportError, AttributeError, Exception) as e:
        logger.warning(f"No se pudo crear red desde initialize_cosmic_trading: {e}")
    
    # Último recurso: crear manualmente
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
            entity.knowledge = random.uniform(0.5, 5.0)
            entity.experience = random.uniform(0.3, 1.0)
            entity.emotion = random.choice(["Esperanza", "Curiosidad", "Determinación", "Serenidad"])
            
            # Añadir a la red
            network.add_entity(entity)
            logger.info(f"Entidad creada manualmente: {name}")
        
        return network
    except Exception as e:
        logger.error(f"Error al crear red manualmente: {e}")
        return None

def setup_competition(network):
    """
    Configurar el sistema de competencia con la red existente.
    
    Args:
        network: Red cósmica existente
        
    Returns:
        Sistema de competencia configurado
    """
    try:
        from modules.cosmic_competition import CosmicCompetition
        
        # Verificar si la red tiene entidades
        if not hasattr(network, "get_entities") and not hasattr(network, "entities"):
            logger.error("La red no tiene entidades accesibles")
            return None
            
        # Obtener entidades
        entities = (network.get_entities() if hasattr(network, "get_entities") 
                   else network.entities)
        
        if not entities:
            logger.error("No hay entidades en la red")
            return None
            
        # Crear sistema de competencia
        competition = CosmicCompetition(entities)
        logger.info(f"Sistema de competencia configurado con {len(entities)} entidades")
        return competition
    except Exception as e:
        logger.error(f"Error al configurar sistema de competencia: {e}")
        return None

def run_direct_competition(competition):
    """
    Ejecutar una competencia directa y mostrar resultados.
    
    Args:
        competition: Sistema de competencia configurado
        
    Returns:
        Resultado de la competencia
    """
    if not competition:
        logger.error("Sistema de competencia no disponible")
        return None
        
    logger.info("==== INICIANDO COMPETENCIA CÓSMICA DIRECTA ====")
    result = competition.competir()
    
    # Mostrar resultados
    if "ranking" in result:
        logger.info("\n== RANKING FINAL ==")
        for pos, (name, power) in enumerate(result["ranking"], 1):
            logger.info(f"{pos}. {name} - Poder: {power:.2f}")
        
        logger.info(f"\nCampeón: {result['campeon']}")
        logger.info("Sabiduría compartida con:")
        for log in result.get("logs_sabiduria", []):
            logger.info(f"- {log}")
    else:
        logger.error("Formato de resultado inesperado")
    
    return result

def run_competition_series(competition, count=5, interval=10):
    """
    Ejecutar una serie de competencias periódicas.
    
    Args:
        competition: Sistema de competencia
        count: Número de competencias
        interval: Intervalo en segundos
    """
    if not competition:
        logger.error("Sistema de competencia no disponible")
        return
        
    logger.info(f"==== INICIANDO SERIE DE {count} COMPETENCIAS ====")
    logger.info(f"Intervalo: {interval} segundos")
    
    # Programar competencias periódicas
    competition.programar_competencias(
        intervalo_segundos=interval, 
        num_competencias=count
    )
    
    # Esperar a que terminen todas
    total_time = count * interval
    for i in range(total_time):
        time.sleep(1)
        if i % interval == 0:
            stats = competition.get_competition_stats()
            logger.info(f"Completadas {stats['total_competitions']} de {count} competencias")
    
    # Mostrar estadísticas finales
    final_stats = competition.get_competition_stats()
    logger.info("\n==== ESTADÍSTICAS FINALES ====")
    logger.info(f"Competencias totales: {final_stats['total_competitions']}")
    logger.info(f"Campeones únicos: {', '.join(final_stats['unique_champions'])}")
    logger.info("Frecuencia de campeones:")
    for champ, freq in final_stats['champion_frequency'].items():
        logger.info(f"- {champ}: {freq} veces")
    logger.info(f"Pool de conocimiento colectivo: {final_stats['knowledge_pool']:.2f}")

def run_sharing_demo(network):
    """
    Ejecutar una demostración de compartición directa de sabiduría.
    
    Args:
        network: Red cósmica
    """
    try:
        from modules.cosmic_competition import compartir_sabiduria, get_all_entities
        
        # Obtener entidades
        entities = get_all_entities(network)
        if not entities:
            logger.error("No hay entidades para compartir sabiduría")
            return
        
        logger.info("==== DEMOSTRACIÓN DE COMPARTICIÓN DIRECTA DE SABIDURÍA ====")
        logger.info(f"Entidades disponibles: {len(entities)}")
        
        # Verificar entidades para mostrar estado inicial
        for entity in entities:
            if hasattr(entity, "name") and hasattr(entity, "knowledge"):
                logger.info(f"{entity.name}: Conocimiento inicial = {entity.knowledge:.2f}")
        
        # Compartir sabiduría
        knowledge_pool = 0.0
        logs, new_pool = compartir_sabiduria(entities, knowledge_pool)
        
        # Mostrar resultados
        if logs:
            logger.info("\n== TRANSFERENCIAS DE SABIDURÍA ==")
            for log in logs:
                logger.info(log)
            
            logger.info(f"\nPool de conocimiento: {new_pool:.2f}")
            
            # Mostrar estado final
            logger.info("\n== ESTADO FINAL ==")
            for entity in entities:
                if hasattr(entity, "name") and hasattr(entity, "knowledge"):
                    logger.info(f"{entity.name}: Conocimiento final = {entity.knowledge:.2f}")
        else:
            logger.warning("No se realizaron transferencias de sabiduría")
    except Exception as e:
        logger.error(f"Error en demostración de compartición: {e}")

def main():
    """Función principal de demostración."""
    print("\n===== SISTEMA DE COMPETENCIA CÓSMICA =====")
    print("1. Competencia única")
    print("2. Serie de competencias (5)")
    print("3. Demostración de compartición de sabiduría")
    print("4. Todo lo anterior")
    
    try:
        choice = int(input("\nSelecciona una opción (1-4): ") or "4")
        if choice not in [1, 2, 3, 4]:
            choice = 4
    except ValueError:
        choice = 4
    
    # Inicializar red
    network = initialize_network()
    if not network:
        print("Error: No se pudo inicializar la red cósmica")
        return
    
    # Configurar competencia
    competition = setup_competition(network)
    
    # Ejecutar opción seleccionada
    if choice == 1 or choice == 4:
        run_direct_competition(competition)
        
    if choice == 2 or choice == 4:
        run_competition_series(competition)
        
    if choice == 3 or choice == 4:
        run_sharing_demo(network)
    
    print("\n===== DEMOSTRACIÓN FINALIZADA =====")

if __name__ == "__main__":
    main()