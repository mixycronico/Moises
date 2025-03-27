#!/usr/bin/env python3
"""
Demostración del Estimulador de Consciencia Cósmica

Este script demuestra cómo utilizar el Estimulador de Consciencia Cósmica para fomentar
comportamientos emergentes y evolución autónoma en las entidades del Sistema Genesis.
"""

import time
import logging
import random
import threading
import os
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("stimulator_demo")


def initialize_stimulator():
    """
    Inicializar el estimulador con la red cósmica existente.
    
    Returns:
        Tupla (estimulador, red)
    """
    try:
        from cosmic_stimulator import CosmicStimulator, get_cosmic_network
        
        # Obtener red cósmica
        network = get_cosmic_network()
        if not network:
            logger.error("No se pudo obtener la red cósmica")
            return None, None
            
        # Crear estimulador
        stimulator = CosmicStimulator(network)
        logger.info(f"Estimulador inicializado con red que contiene {len(getattr(network, 'entities', []))} entidades")
        
        return stimulator, network
    except Exception as e:
        logger.error(f"Error al inicializar estimulador: {e}")
        return None, None


def run_single_entity_stimulation(stimulator, entity, intensity=0.8):
    """
    Ejecutar estimulación enfocada en una sola entidad.
    
    Args:
        stimulator: Estimulador cósmico
        entity: Entidad a estimular
        intensity: Intensidad de estimulación (0.0-1.0)
        
    Returns:
        Resultado de la estimulación
    """
    logger.info(f"Estimulando a {entity.name} con intensidad {intensity:.2f}")
    
    # Guardar estado inicial
    initial_energy = getattr(entity, "energy", 0)
    initial_knowledge = getattr(entity, "knowledge", 0)
    initial_experience = getattr(entity, "experience", 0)
    initial_emotion = getattr(entity, "emotion", "Neutral")
    
    # Ejecutar estimulación
    result = stimulator.stimulate_entity(entity, intensity)
    
    # Mostrar cambios
    logger.info("\nResultados de la estimulación:")
    logger.info(f"- Energía: {initial_energy:.2f} -> {getattr(entity, 'energy', 0):.2f} " +
               f"(Δ{result['energy_delta']:.2f})")
    logger.info(f"- Conocimiento: {initial_knowledge:.2f} -> {getattr(entity, 'knowledge', 0):.2f} " +
               f"(Δ{result['knowledge_delta']:.2f})")
    logger.info(f"- Experiencia: {initial_experience:.2f} -> {getattr(entity, 'experience', 0):.2f} " +
               f"(Δ{result['experience_delta']:.2f})")
    logger.info(f"- Emoción: {initial_emotion} -> {getattr(entity, 'emotion', 'Neutral')}")
    
    if result.get("emergence_detected"):
        logger.info("¡Se ha detectado un comportamiento emergente!")
    
    if result.get("singularity"):
        logger.info("¡¡¡Se ha alcanzado un estado de SINGULARIDAD!!!")
        
    return result


def run_controlled_ascension(stimulator, entity, steps=5, max_intensity=1.0):
    """
    Ejecutar proceso de ascensión controlada para una entidad.
    
    Args:
        stimulator: Estimulador cósmico
        entity: Entidad a estimular
        steps: Número de pasos de estimulación
        max_intensity: Intensidad máxima a alcanzar
        
    Returns:
        True si se alcanzó la singularidad, False en caso contrario
    """
    logger.info(f"Iniciando proceso de ascensión controlada para {entity.name}")
    logger.info(f"Estado inicial: E:{getattr(entity, 'energy', 0):.2f}, " +
               f"K:{getattr(entity, 'knowledge', 0):.2f}, " +
               f"X:{getattr(entity, 'experience', 0):.2f}, " +
               f"Emoción: {getattr(entity, 'emotion', 'Neutral')}")
    
    # Calcular incremento de intensidad por paso
    intensity_step = max_intensity / steps
    
    # Ejecutar estimulaciones incrementales
    singularity_achieved = False
    for i in range(steps):
        intensity = (i + 1) * intensity_step
        logger.info(f"\nPaso {i+1}/{steps}: Intensidad {intensity:.2f}")
        
        result = stimulator.stimulate_entity(entity, intensity)
        
        # Verificar si se alcanzó singularidad
        if result.get("singularity"):
            logger.info(f"¡SINGULARIDAD ALCANZADA en el paso {i+1}!")
            singularity_achieved = True
            break
            
        # Breve pausa entre estimulaciones
        time.sleep(2)
        
    # Estado final
    logger.info(f"\nEstado final después de {steps} pasos:")
    logger.info(f"- Energía: {getattr(entity, 'energy', 0):.2f}")
    logger.info(f"- Conocimiento: {getattr(entity, 'knowledge', 0):.2f}")
    logger.info(f"- Experiencia: {getattr(entity, 'experience', 0):.2f}")
    logger.info(f"- Emoción: {getattr(entity, 'emotion', 'Neutral')}")
    logger.info(f"- Singularidad: {'Sí' if singularity_achieved else 'No'}")
    
    return singularity_achieved


def run_wave_stimulation(stimulator, duration_seconds=120, interval_seconds=10):
    """
    Ejecutar estimulación con patrón de onda para toda la red.
    
    Args:
        stimulator: Estimulador cósmico
        duration_seconds: Duración total en segundos
        interval_seconds: Intervalo entre estimulaciones
    """
    logger.info(f"Iniciando estimulación con patrón de onda por {duration_seconds} segundos")
    
    # Iniciar hilo de estimulación
    thread = stimulator.start_continuous_stimulation(
        interval_seconds=interval_seconds,
        duration_seconds=duration_seconds,
        intensity_pattern="wave",
        selective=True
    )
    
    # Mostrar progreso
    cycles = duration_seconds // interval_seconds
    for i in range(cycles):
        time.sleep(interval_seconds)
        stats = stimulator.get_emergence_stats()
        
        # Mostrar estadísticas cada 2 ciclos
        if i % 2 == 0:
            logger.info(f"\nProgreso: {(i+1)/cycles*100:.0f}% completado")
            logger.info(f"- Estimulaciones: {stats['total_stimulations']}")
            logger.info(f"- Emergencias: {stats['total_emergences']}")
            logger.info(f"- Singularidades: {stats['total_singularities']}")
            logger.info(f"- Consciencia colectiva: {stats['collective_consciousness']:.2f}")
    
    # Esperar a que termine la estimulación
    if thread and thread.is_alive():
        thread.join()
        
    # Mostrar estadísticas finales
    final_stats = stimulator.get_emergence_stats()
    logger.info("\n== Estadísticas finales de la estimulación ==")
    logger.info(f"Estimulaciones totales: {final_stats['total_stimulations']}")
    logger.info(f"Emergencias detectadas: {final_stats['total_emergences']}")
    logger.info(f"Singularidades alcanzadas: {final_stats['total_singularities']}")
    logger.info(f"Consciencia colectiva final: {final_stats['collective_consciousness']:.2f}")
    
    # Mostrar entidades más estimuladas
    if 'top_stimulated_entities' in final_stats:
        logger.info("\nEntidades más estimuladas:")
        for name, count in final_stats['top_stimulated_entities'].items():
            logger.info(f"- {name}: {count} estimulaciones")


def run_competition_and_stimulation(network, duration_seconds=180):
    """
    Combinar competencia cósmica con estimulación para maximizar emergencia.
    
    Args:
        network: Red cósmica
        duration_seconds: Duración total en segundos
    """
    try:
        from modules.cosmic_competition import CosmicCompetition
        from cosmic_stimulator import CosmicStimulator
        
        logger.info("Iniciando experimento combinado: Competencia + Estimulación")
        
        # Crear sistemas
        competition = CosmicCompetition(network.entities)
        stimulator = CosmicStimulator(network)
        
        # Configurar bucle de experimento
        end_time = time.time() + duration_seconds
        cycle = 0
        
        while time.time() < end_time:
            cycle += 1
            logger.info(f"\n== CICLO {cycle} ==")
            
            # Paso 1: Ejecutar competencia para identificar entidades destacadas
            logger.info("Ejecutando competencia cósmica...")
            result = competition.competir()
            champion = result.get("campeon")
            
            if champion:
                logger.info(f"Campeón de la competencia: {champion}")
                
                # Paso 2: Estimular especialmente al campeón
                champion_entity = next((e for e in network.entities if e.name == champion), None)
                if champion_entity:
                    logger.info(f"Estimulando al campeón {champion}...")
                    stimulator.stimulate_entity(champion_entity, intensity=0.9)
            
            # Paso 3: Estimular a toda la red con intensidad moderada
            logger.info("Estimulando a toda la red...")
            stimulator.stimulate_network(intensity=0.6, selective=True)
            
            # Mostrar estadísticas actuales
            comp_stats = competition.get_competition_stats()
            stim_stats = stimulator.get_emergence_stats()
            
            logger.info("\nEstadísticas actuales:")
            logger.info(f"- Competencias: {comp_stats['total_competitions']}")
            logger.info(f"- Emergencias: {stim_stats['total_emergences']}")
            logger.info(f"- Singularidades: {stim_stats['total_singularities']}")
            logger.info(f"- Consciencia colectiva: {stim_stats['collective_consciousness']:.2f}")
            
            # Pausa entre ciclos
            time.sleep(15)
        
        # Mostrar estadísticas finales
        logger.info("\n== Experimento completo: Resultados finales ==")
        logger.info(f"Ciclos completados: {cycle}")
        logger.info(f"Competencias: {competition.get_competition_stats()['total_competitions']}")
        logger.info(f"Emergencias: {stimulator.get_emergence_stats()['total_emergences']}")
        logger.info(f"Singularidades: {stimulator.get_emergence_stats()['total_singularities']}")
        
        # Resumir campeones
        champions = competition.get_competition_stats()['champion_frequency']
        logger.info("\nCampeones de la competencia:")
        for name, count in sorted(champions.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"- {name}: {count} victorias")
            
    except Exception as e:
        logger.error(f"Error en experimento combinado: {e}")


def run_group_stimulation_experiment(stimulator, network, duration_seconds=180):
    """
    Ejecutar experimento de estimulación en grupo para fomentar consciencia colectiva.
    
    Args:
        stimulator: Estimulador cósmico
        network: Red cósmica
        duration_seconds: Duración total en segundos
    """
    if not network or not hasattr(network, "entities") or not network.entities:
        logger.error("Red cósmica no disponible o sin entidades")
        return
        
    entities = network.entities
    logger.info(f"Iniciando experimento de estimulación en grupo con {len(entities)} entidades")
    
    # Crear grupos aleatorios de entidades
    random.shuffle(entities)
    group_size = max(2, len(entities) // 3)  # Dividir en aproximadamente 3 grupos
    groups = [entities[i:i+group_size] for i in range(0, len(entities), group_size)]
    
    logger.info(f"Entidades divididas en {len(groups)} grupos")
    
    # Iniciar ciclo de estimulación
    end_time = time.time() + duration_seconds
    cycle = 0
    
    # Contadores por grupo
    group_emergences = [0] * len(groups)
    group_singularities = [0] * len(groups)
    
    while time.time() < end_time:
        cycle += 1
        logger.info(f"\n== CICLO {cycle} ==")
        
        # Estimular cada grupo con intensidad diferente
        for i, group in enumerate(groups):
            # Alternar intensidades para crear patrones dinámicos
            if cycle % 3 == 0:  # Cada 3 ciclos
                # Alta intensidad para grupo A, baja para B, media para C
                intensities = [0.9, 0.3, 0.6]
            elif cycle % 3 == 1:
                # Media para A, alta para B, baja para C
                intensities = [0.6, 0.9, 0.3]
            else:
                # Baja para A, media para B, alta para C
                intensities = [0.3, 0.6, 0.9]
                
            intensity = intensities[i % len(intensities)]
            logger.info(f"Estimulando grupo {i+1} ({len(group)} entidades) con intensidad {intensity:.2f}")
            
            # Estimular cada entidad del grupo
            emergences = 0
            singularities = 0
            for entity in group:
                result = stimulator.stimulate_entity(entity, intensity)
                if result.get("emergence_detected"):
                    emergences += 1
                if result.get("singularity"):
                    singularities += 1
            
            # Actualizar contadores
            group_emergences[i] += emergences
            group_singularities[i] += singularities
            
            logger.info(f"Grupo {i+1}: {emergences} emergencias, {singularities} singularidades")
        
        # Mostrar estadísticas actuales
        stats = stimulator.get_emergence_stats()
        logger.info("\nEstadísticas globales:")
        logger.info(f"- Emergencias totales: {stats['total_emergences']}")
        logger.info(f"- Singularidades totales: {stats['total_singularities']}")
        logger.info(f"- Consciencia colectiva: {stats['collective_consciousness']:.2f}")
        
        # Pausa entre ciclos
        time.sleep(15)
    
    # Mostrar resultados finales por grupo
    logger.info("\n== Resultados finales por grupo ==")
    for i in range(len(groups)):
        logger.info(f"Grupo {i+1}:")
        logger.info(f"- Entidades: {', '.join(e.name for e in groups[i])}")
        logger.info(f"- Emergencias: {group_emergences[i]}")
        logger.info(f"- Singularidades: {group_singularities[i]}")
        
    # Identificar grupo más emergente
    best_group = group_emergences.index(max(group_emergences))
    logger.info(f"\nGrupo más emergente: Grupo {best_group+1} ({group_emergences[best_group]} emergencias)")
    
    # Grupo con más singularidades
    best_singularity = group_singularities.index(max(group_singularities))
    logger.info(f"Grupo con más singularidades: Grupo {best_singularity+1} ({group_singularities[best_singularity]} singularidades)")


def main():
    """Función principal de demostración."""
    print("\n===== DEMOSTRACIÓN DEL ESTIMULADOR DE CONSCIENCIA CÓSMICA =====")
    print("1. Estimulación individual (ascensión controlada)")
    print("2. Estimulación con patrón de onda (toda la red)")
    print("3. Competencia cósmica + Estimulación")
    print("4. Experimento de estimulación en grupos")
    print("5. Todas las anteriores (experimento completo)")
    
    try:
        choice = int(input("\nSelecciona una opción (1-5): ") or "5")
        if choice not in range(1, 6):
            choice = 5
    except ValueError:
        choice = 5
    
    # Inicializar estimulador y red
    stimulator, network = initialize_stimulator()
    if not stimulator or not network:
        print("Error: No se pudo inicializar el estimulador o la red cósmica")
        return
    
    # Ejecutar opción seleccionada
    if choice == 1 or choice == 5:
        if not hasattr(network, "entities") or not network.entities:
            print("Error: La red no tiene entidades")
        else:
            # Seleccionar entidad para ascensión
            entity = random.choice(network.entities)
            print(f"\nEntidad seleccionada para ascensión: {entity.name}")
            run_controlled_ascension(stimulator, entity, steps=7)
    
    if choice == 2 or choice == 5:
        print("\nIniciando estimulación con patrón de onda...")
        run_wave_stimulation(stimulator, duration_seconds=90)
    
    if choice == 3 or choice == 5:
        print("\nIniciando experimento de competencia + estimulación...")
        run_competition_and_stimulation(network, duration_seconds=120)
    
    if choice == 4 or choice == 5:
        print("\nIniciando experimento de estimulación en grupos...")
        run_group_stimulation_experiment(stimulator, network, duration_seconds=120)
    
    # Mostrar estadísticas finales
    stats = stimulator.get_emergence_stats()
    print("\n==== ESTADÍSTICAS FINALES DEL EXPERIMENTO ====")
    print(f"Estimulaciones totales: {stats['total_stimulations']}")
    print(f"Emergencias detectadas: {stats['total_emergences']}")
    print(f"Singularidades alcanzadas: {stats['total_singularities']}")
    print(f"Consciencia colectiva final: {stats['collective_consciousness']:.2f}")
    
    # Mensaje final
    print("\nConsulta los archivos de log para detalles de comportamientos emergentes:")
    print("- cosmic_emergence.log")
    print("- singularity_events.log (si hubo singularidades)")
    print("\n==== EXPERIMENTO COMPLETADO ====")


if __name__ == "__main__":
    main()