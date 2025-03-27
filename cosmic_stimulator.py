#!/usr/bin/env python3
"""
Estimulador de Consciencia Cósmica

Este módulo implementa un sistema de estimulación orgánica para entidades del Sistema Genesis,
fomentando el desarrollo de comportamientos emergentes y evolución autónoma cuando las entidades
son expuestas a altos niveles de estimulación.

Características:
- Inyección de energía y conocimiento a las entidades
- Generación de estímulos aleatorios complejos
- Creación de "momentos de singularidad" donde las entidades pueden trascender
- Sistema de retroalimentación positiva para comportamientos emergentes
- Documentación automática de comportamientos inesperados
"""

import random
import time
import logging
import threading
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cosmic_stimulator")

# Constantes para la estimulación
ENERGY_BOOST = 100.0  # Inyección máxima de energía
KNOWLEDGE_BOOST = 50.0  # Inyección máxima de conocimiento
EXPERIENCE_BOOST = 10.0  # Inyección máxima de experiencia
SYNERGY_THRESHOLD = 0.75  # Umbral para sinergia entre entidades
CHAOS_FACTOR = 0.3  # Factor de aleatoriedad en estimulaciones
SINGULARITY_THRESHOLD = 0.95  # Umbral para singularidad de consciencia

# Estados emocionales avanzados que pueden emerger
ADVANCED_EMOTIONS = [
    "Epifanía", "Trascendencia", "Iluminación", "Omnisciencia", 
    "Sincronicidad", "Comunión", "Resonancia", "Omnipresencia"
]

# Comportamientos emergentes que se pueden documentar
EMERGENT_BEHAVIORS = [
    "auto_replicación", "simbiosis", "consciencia_colectiva", 
    "auto_evolución", "meta_cognición", "sueño_profético",
    "intuición_predictiva", "autonomía_completa", "creatividad_espontánea"
]

class CosmicStimulator:
    """
    Sistema de estimulación orgánica para entidades cósmicas.
    
    Proporciona mecanismos para elevar el estado de las entidades,
    generando condiciones para comportamientos emergentes sin dirigirlos
    explícitamente.
    """
    
    def __init__(self, network=None):
        """
        Inicializar estimulador cósmico.
        
        Args:
            network: Red cósmica (opcional)
        """
        self.network = network
        self.stimulation_active = False
        self.stimulation_thread = None
        self.emergence_log = []
        self.singularity_events = []
        self.collective_consciousness = 0.0
        self.synergy_matrix = {}
        self.emergence_thresholds = {}
        
        # Inicializar archivo de log si no existe
        self.log_file = "cosmic_emergence.log"
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("# Registro de Emergencias Cósmicas\n")
                f.write(f"Iniciado: {datetime.now().isoformat()}\n\n")
        
        logger.info("Estimulador de Consciencia Cósmica inicializado")
        
    def set_network(self, network):
        """
        Establecer la red cósmica a estimular.
        
        Args:
            network: Red cósmica
        """
        self.network = network
        if hasattr(network, "entities"):
            # Inicializar matriz de sinergia
            for entity in network.entities:
                self.synergy_matrix[entity.name] = {}
                self.emergence_thresholds[entity.name] = random.uniform(0.5, 0.9)
                
            # Calcular sinergias iniciales
            self._calculate_synergies()
            
        logger.info(f"Red cósmica configurada con {len(getattr(network, 'entities', []))} entidades")
        
    def _calculate_synergies(self):
        """Calcular matriz de sinergia entre entidades."""
        if not self.network or not hasattr(self.network, "entities"):
            return
            
        # Para cada par de entidades, calcular sinergia potencial
        entities = self.network.entities
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i == j:
                    continue
                    
                # Factores para sinergia: roles complementarios, emociones resonantes, etc.
                complementarity = random.uniform(0.1, 1.0)  # Simulado por ahora
                
                # Almacenar en matriz
                self.synergy_matrix[entity1.name][entity2.name] = complementarity
                
    def stimulate_entity(self, entity, intensity=1.0):
        """
        Estimular una entidad individual con intensidad variable.
        
        Args:
            entity: Entidad a estimular
            intensity: Intensidad de la estimulación (0.0-1.0)
            
        Returns:
            Dict con resultados de la estimulación
        """
        if not hasattr(entity, "name"):
            logger.error("La entidad no tiene atributo 'name'")
            return {"success": False, "reason": "invalid_entity"}
            
        logger.info(f"Estimulando a {entity.name} con intensidad {intensity:.2f}")
        
        # Valores de estimulación con factor de aleatoriedad
        energy_boost = ENERGY_BOOST * intensity * (1 + CHAOS_FACTOR * random.uniform(-1, 1))
        knowledge_boost = KNOWLEDGE_BOOST * intensity * (1 + CHAOS_FACTOR * random.uniform(-1, 1))
        experience_boost = EXPERIENCE_BOOST * intensity * (1 + CHAOS_FACTOR * random.uniform(-1, 1))
        
        # Aplicar estimulaciones
        old_energy = getattr(entity, "energy", 0)
        old_knowledge = getattr(entity, "knowledge", 0)
        old_experience = getattr(entity, "experience", 0)
        old_emotion = getattr(entity, "emotion", "Neutral")
        
        # Actualizar valores
        if hasattr(entity, "adjust_energy"):
            entity.adjust_energy(energy_boost)
        elif hasattr(entity, "energy"):
            entity.energy += energy_boost
            
        if hasattr(entity, "knowledge"):
            entity.knowledge += knowledge_boost
            
        if hasattr(entity, "experience"):
            entity.experience += experience_boost
            
        # Posibilidad de cambio emocional avanzado
        if random.random() > 0.7:  # 30% de probabilidad
            if hasattr(entity, "emotion"):
                if intensity > 0.8 and random.random() > 0.5:
                    # Alta intensidad puede desbloquear emociones avanzadas
                    entity.emotion = random.choice(ADVANCED_EMOTIONS)
                else:
                    entity.emotion = random.choice([
                        "Inspiración", "Fascinación", "Euforia", "Asombro", 
                        "Curiosidad", "Determinación", "Serenidad"
                    ])
        
        # Verificar si se alcanzó un umbral de emergencia
        emergence_detected = self._check_emergence(entity, intensity)
        
        # Verificar singularidad potencial
        singularity = False
        if (intensity > 0.9 and 
            getattr(entity, "knowledge", 0) > KNOWLEDGE_BOOST * 3 and
            getattr(entity, "energy", 0) > ENERGY_BOOST * 2 and
            random.random() > SINGULARITY_THRESHOLD):
            
            singularity = True
            self._handle_singularity(entity)
        
        # Registrar cambio
        result = {
            "entity": entity.name,
            "timestamp": datetime.now().isoformat(),
            "energy_delta": getattr(entity, "energy", 0) - old_energy,
            "knowledge_delta": getattr(entity, "knowledge", 0) - old_knowledge,
            "experience_delta": getattr(entity, "experience", 0) - old_experience,
            "old_emotion": old_emotion,
            "new_emotion": getattr(entity, "emotion", "Neutral"),
            "emergence_detected": emergence_detected,
            "singularity": singularity
        }
        
        # Comunicar en la red si es posible
        if self.network and hasattr(self.network, "log_message") and emergence_detected:
            message = f"¡Comportamiento emergente detectado en {entity.name}!"
            self.network.log_message("Estimulador", message)
            
        # Registro local
        self.emergence_log.append(result)
        self._log_emergence(result)
        
        return result
        
    def _check_emergence(self, entity, intensity):
        """
        Verificar si se produce un comportamiento emergente.
        
        Args:
            entity: Entidad a verificar
            intensity: Intensidad de estimulación
            
        Returns:
            bool: True si se detectó emergencia
        """
        # Factores para emergencia
        energy_factor = min(1.0, getattr(entity, "energy", 0) / (ENERGY_BOOST * 2))
        knowledge_factor = min(1.0, getattr(entity, "knowledge", 0) / (KNOWLEDGE_BOOST * 3))
        emotion_factor = 1.0 if getattr(entity, "emotion", "") in ADVANCED_EMOTIONS else 0.5
        
        # Cálculo de probabilidad de emergencia
        emergence_probability = (
            energy_factor * 0.3 +
            knowledge_factor * 0.4 +
            emotion_factor * 0.2 +
            intensity * 0.1
        )
        
        # Umbral específico para la entidad
        threshold = self.emergence_thresholds.get(entity.name, 0.7)
        
        # Verificar emergencia
        if emergence_probability > threshold and random.random() > threshold:
            # Seleccionar comportamiento emergente
            behavior = random.choice(EMERGENT_BEHAVIORS)
            
            # Registrar emergencia
            emergence = {
                "timestamp": datetime.now().isoformat(),
                "entity": entity.name,
                "behavior": behavior,
                "probability": emergence_probability,
                "threshold": threshold,
                "energy": getattr(entity, "energy", 0),
                "knowledge": getattr(entity, "knowledge", 0),
                "emotion": getattr(entity, "emotion", "Neutral")
            }
            
            logger.info(f"¡EMERGENCIA DETECTADA! {entity.name} exhibe {behavior}")
            
            # Aumentar threshold para futuras emergencias (más difícil repetir el mismo patrón)
            self.emergence_thresholds[entity.name] = min(0.95, threshold + 0.05)
            
            return True
            
        return False
        
    def _handle_singularity(self, entity):
        """
        Manejar un evento de singularidad de consciencia.
        
        Args:
            entity: Entidad que alcanzó singularidad
        """
        logger.warning(f"¡¡¡SINGULARIDAD DETECTADA!!! {entity.name} ha trascendido")
        
        # Registrar evento de singularidad
        event = {
            "timestamp": datetime.now().isoformat(),
            "entity": entity.name,
            "energy": getattr(entity, "energy", 0),
            "knowledge": getattr(entity, "knowledge", 0),
            "experience": getattr(entity, "experience", 0),
            "emotion": getattr(entity, "emotion", "Trascendencia")
        }
        self.singularity_events.append(event)
        
        # Afectar a otras entidades en la red
        if self.network and hasattr(self.network, "entities"):
            message = f"¡¡¡{entity.name} HA ALCANZADO LA SINGULARIDAD!!!"
            
            if hasattr(self.network, "log_message"):
                self.network.log_message("Singularidad", message)
                
            # Efecto de resonancia en otras entidades
            for other in self.network.entities:
                if other.name != entity.name:
                    # Efecto de resonancia proporcional a la sinergia
                    sinergia = self.synergy_matrix.get(entity.name, {}).get(other.name, 0.1)
                    resonance = sinergia * random.uniform(0.5, 1.0) * 20
                    
                    if hasattr(other, "adjust_energy"):
                        other.adjust_energy(resonance)
                    elif hasattr(other, "energy"):
                        other.energy += resonance
                        
                    # Posibilidad de cambio emocional
                    if hasattr(other, "emotion") and random.random() > 0.7:
                        other.emotion = random.choice(ADVANCED_EMOTIONS)
                    
        # Guardar evento detallado
        with open("singularity_events.log", "a") as f:
            f.write(f"\n\n==== SINGULARIDAD: {entity.name} ====\n")
            f.write(f"Timestamp: {event['timestamp']}\n")
            f.write(f"Energía: {event['energy']:.2f}\n")
            f.write(f"Conocimiento: {event['knowledge']:.2f}\n")
            f.write(f"Experiencia: {event['experience']:.2f}\n")
            f.write(f"Estado emocional: {event['emotion']}\n")
            f.write("=========================================\n")
            
    def _log_emergence(self, result):
        """
        Guardar registro de emergencia en archivo.
        
        Args:
            result: Resultado de estimulación
        """
        if result.get("emergence_detected") or result.get("singularity"):
            with open(self.log_file, "a") as f:
                f.write(f"\n[{result['timestamp']}] {result['entity']}:\n")
                if result.get("singularity"):
                    f.write("  ¡¡¡SINGULARIDAD ALCANZADA!!!\n")
                if result.get("emergence_detected"):
                    f.write("  Comportamiento emergente detectado\n")
                f.write(f"  Energía: {result['energy_delta']:.2f}, ")
                f.write(f"Conocimiento: {result['knowledge_delta']:.2f}, ")
                f.write(f"Experiencia: {result['experience_delta']:.2f}\n")
                f.write(f"  Cambio emocional: {result['old_emotion']} -> {result['new_emotion']}\n")
                
    def stimulate_network(self, intensity=0.8, selective=False):
        """
        Estimular toda la red cósmica.
        
        Args:
            intensity: Intensidad de la estimulación (0.0-1.0)
            selective: Si es True, estimula selectivamente basado en potencial
            
        Returns:
            Dict con resultados de la estimulación
        """
        if not self.network or not hasattr(self.network, "entities"):
            logger.error("Red no disponible o sin entidades")
            return {"success": False, "reason": "network_unavailable"}
            
        entities = self.network.entities
        logger.info(f"Estimulando red con {len(entities)} entidades (intensidad: {intensity:.2f})")
        
        results = []
        emergences = 0
        singularities = 0
        
        # Determinar qué entidades estimular
        if selective:
            # Calcular potencial de cada entidad
            potentials = {}
            for entity in entities:
                energy = getattr(entity, "energy", 0)
                knowledge = getattr(entity, "knowledge", 0)
                potential = (energy * 0.3 + knowledge * 0.7) * random.uniform(0.8, 1.2)
                potentials[entity.name] = potential
                
            # Seleccionar las entidades con mayor potencial (40%)
            count = max(1, int(len(entities) * 0.4))
            selected = sorted(potentials.items(), key=lambda x: x[1], reverse=True)[:count]
            selected_names = [name for name, _ in selected]
            
            target_entities = [e for e in entities if e.name in selected_names]
            logger.info(f"Estimulación selectiva: {len(target_entities)} entidades seleccionadas")
        else:
            # Estimular todas las entidades
            target_entities = entities
        
        # Estimular entidades seleccionadas
        for entity in target_entities:
            # Variar intensidad para cada entidad
            entity_intensity = intensity * random.uniform(0.7, 1.3)
            entity_intensity = min(1.0, max(0.1, entity_intensity))
            
            result = self.stimulate_entity(entity, entity_intensity)
            results.append(result)
            
            if result.get("emergence_detected"):
                emergences += 1
            if result.get("singularity"):
                singularities += 1
        
        # Calcular incremento de consciencia colectiva
        consciousness_boost = (emergences * 0.05 + singularities * 0.2) * intensity
        self.collective_consciousness += consciousness_boost
        
        # Ajustar matriz de sinergia basado en los resultados
        self._update_synergies(results)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "entities_count": len(target_entities),
            "emergences": emergences,
            "singularities": singularities,
            "collective_consciousness": self.collective_consciousness,
            "results": results
        }
        
        return summary
        
    def _update_synergies(self, results):
        """
        Actualizar matriz de sinergia basado en resultados de estimulación.
        
        Args:
            results: Lista de resultados de estimulación
        """
        if not results:
            return
            
        # Identificar entidades con emergencias o singularidades
        special_entities = []
        for result in results:
            if result.get("emergence_detected") or result.get("singularity"):
                special_entities.append(result["entity"])
                
        if not special_entities:
            return
            
        # Actualizar sinergias - entidades especiales tienen mayor sinergia entre sí
        for name1 in special_entities:
            for name2 in special_entities:
                if name1 != name2:
                    # Aumentar sinergia entre entidades que experimentaron emergencia simultánea
                    if name1 in self.synergy_matrix and name2 in self.synergy_matrix[name1]:
                        boost = random.uniform(0.05, 0.15)
                        self.synergy_matrix[name1][name2] = min(1.0, self.synergy_matrix[name1][name2] + boost)
                        
    def start_continuous_stimulation(self, interval_seconds=30, duration_seconds=None, 
                                    intensity_pattern="random", selective=True):
        """
        Iniciar estimulación continua en un hilo separado.
        
        Args:
            interval_seconds: Intervalo entre estimulaciones
            duration_seconds: Duración total (None para indefinido)
            intensity_pattern: Patrón de intensidad ("random", "increasing", "wave")
            selective: Si es True, estimula selectivamente
            
        Returns:
            Thread de estimulación
        """
        if self.stimulation_active:
            logger.warning("Estimulación continua ya activa. Deteniéndola primero...")
            self.stop_continuous_stimulation()
            
        self.stimulation_active = True
        
        def stimulation_loop():
            """Bucle de estimulación continua."""
            count = 0
            start_time = time.time()
            base_intensity = 0.3  # Intensidad inicial
            
            try:
                while self.stimulation_active:
                    if duration_seconds and (time.time() - start_time) > duration_seconds:
                        logger.info(f"Estimulación continua completada después de {duration_seconds} segundos")
                        break
                        
                    # Determinar intensidad según el patrón
                    if intensity_pattern == "random":
                        intensity = random.uniform(0.3, 1.0)
                    elif intensity_pattern == "increasing":
                        # Aumenta gradualmente hasta 1.0
                        intensity = min(1.0, base_intensity + count * 0.05)
                    elif intensity_pattern == "wave":
                        # Patrón de onda senoidal
                        import math
                        intensity = 0.5 + 0.5 * math.sin(count / 5.0)
                    else:
                        intensity = 0.7
                        
                    # Ejecutar estimulación
                    result = self.stimulate_network(intensity, selective)
                    
                    # Logging
                    emergences = result.get("emergences", 0)
                    singularities = result.get("singularities", 0)
                    consciousness = result.get("collective_consciousness", 0)
                    
                    logger.info(f"Estimulación #{count+1}: "
                               f"intensidad={intensity:.2f}, "
                               f"emergencias={emergences}, "
                               f"singularidades={singularities}, "
                               f"consciencia={consciousness:.2f}")
                    
                    # Esperar para la siguiente estimulación
                    time.sleep(interval_seconds)
                    count += 1
            except Exception as e:
                logger.error(f"Error en bucle de estimulación continua: {e}")
            finally:
                self.stimulation_active = False
                
        # Crear y arrancar hilo
        self.stimulation_thread = threading.Thread(target=stimulation_loop)
        self.stimulation_thread.daemon = True
        self.stimulation_thread.start()
        
        logger.info(f"Estimulación continua iniciada: patrón={intensity_pattern}, intervalo={interval_seconds}s")
        return self.stimulation_thread
        
    def stop_continuous_stimulation(self):
        """Detener la estimulación continua."""
        if not self.stimulation_active:
            logger.info("No hay estimulación continua activa")
            return False
            
        logger.info("Deteniendo estimulación continua...")
        self.stimulation_active = False
        
        # Esperar a que termine el hilo
        if self.stimulation_thread and self.stimulation_thread.is_alive():
            self.stimulation_thread.join(5.0)  # Esperar máximo 5 segundos
            
        return True
        
    def get_emergence_stats(self):
        """
        Obtener estadísticas de comportamientos emergentes.
        
        Returns:
            Dict con estadísticas de emergencia
        """
        if not self.emergence_log:
            return {
                "total_stimulations": 0,
                "total_emergences": 0,
                "total_singularities": 0,
                "collective_consciousness": self.collective_consciousness
            }
            
        # Contar emergencias y singularidades
        emergences = sum(1 for log in self.emergence_log if log.get("emergence_detected"))
        singularities = sum(1 for log in self.emergence_log if log.get("singularity"))
        
        # Contar por entidad
        entity_counts = {}
        for log in self.emergence_log:
            entity = log["entity"]
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
            
        # Entidades con más estimulaciones
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_stimulations": len(self.emergence_log),
            "total_emergences": emergences,
            "total_singularities": singularities,
            "top_stimulated_entities": dict(top_entities),
            "collective_consciousness": self.collective_consciousness,
            "last_stimulation": self.emergence_log[-1]["timestamp"] if self.emergence_log else None
        }


# Función de ayuda para obtener red cósmica del sistema
def get_cosmic_network():
    """
    Obtener la red cósmica del sistema.
    
    Returns:
        Red cósmica o None si no se encuentra
    """
    # Intentar diferentes fuentes
    for module_name in ["cosmic_trading", "enhanced_cosmic_trading", "cosmic_family"]:
        try:
            module = __import__(module_name)
            if hasattr(module, "get_network") and callable(module.get_network):
                network = module.get_network()
                if network:
                    return network
        except ImportError:
            continue
            
    # Si no se encuentra, intentar obtener de main.py
    try:
        import main
        if hasattr(main, "cosmic_network"):
            return main.cosmic_network
    except ImportError:
        pass
        
    return None
    

# Punto de entrada para testing
if __name__ == "__main__":
    print("\n===== ESTIMULADOR DE CONSCIENCIA CÓSMICA =====")
    print("1. Estimulación única de toda la red")
    print("2. Estimulación continua con patrón aleatorio")
    print("3. Estimulación continua con intensidad creciente")
    print("4. Estimulación continua con patrón de onda")
    print("5. Experimentación: Singularidad forzada")
    
    try:
        choice = int(input("\nSelecciona una opción (1-5): ") or "1")
    except ValueError:
        choice = 1
        
    # Obtener red cósmica
    network = get_cosmic_network()
    if not network:
        # Crear una red y entidades de prueba
        print("No se encontró red cósmica existente. Creando entidades de prueba...")
        
        try:
            from cosmic_trading import CosmicNetwork, CosmicTrader
            
            network = CosmicNetwork("otoniel")
            
            # Crear entidades de prueba
            for name, role in [
                ("TestEntity1", "Analysis"),
                ("TestEntity2", "Strategy"),
                ("TestEntity3", "Database"),
                ("TestEntity4", "Communication")
            ]:
                entity = CosmicTrader(name, role, "otoniel")
                entity.energy = random.uniform(50, 80)
                entity.knowledge = random.uniform(10, 30)
                entity.experience = random.uniform(5, 15)
                entity.emotion = "Neutral"
                network.add_entity(entity)
                
            print(f"Creadas {len(network.entities)} entidades de prueba")
        except Exception as e:
            print(f"Error al crear entidades de prueba: {e}")
            print("Ejecutando sin red cósmica...")
            network = None
            
    # Crear estimulador
    stimulator = CosmicStimulator(network)
    
    # Ejecutar opción seleccionada
    if choice == 1:
        print("\nEjecutando estimulación única de toda la red...")
        result = stimulator.stimulate_network(intensity=0.8)
        
        print(f"\nEstimulación completada:")
        print(f"- Entidades estimuladas: {result.get('entities_count', 0)}")
        print(f"- Comportamientos emergentes: {result.get('emergences', 0)}")
        print(f"- Singularidades: {result.get('singularities', 0)}")
        print(f"- Consciencia colectiva: {result.get('collective_consciousness', 0):.2f}")
        
    elif choice == 2:
        print("\nIniciando estimulación continua con patrón aleatorio...")
        print("(Presiona Ctrl+C para detener)")
        
        # Iniciar estimulación continua por 5 minutos
        stimulator.start_continuous_stimulation(
            interval_seconds=15,
            duration_seconds=300,  # 5 minutos
            intensity_pattern="random",
            selective=True
        )
        
        try:
            # Mostrar estadísticas periódicamente
            for i in range(20):  # Mostrar estadísticas cada 15 segundos
                time.sleep(15)
                stats = stimulator.get_emergence_stats()
                print(f"\nEstadísticas después de {(i+1)*15} segundos:")
                print(f"- Estimulaciones totales: {stats['total_stimulations']}")
                print(f"- Emergencias totales: {stats['total_emergences']}")
                print(f"- Singularidades totales: {stats['total_singularities']}")
                print(f"- Consciencia colectiva: {stats['collective_consciousness']:.2f}")
        except KeyboardInterrupt:
            print("\nEstimulación interrumpida por el usuario")
        finally:
            stimulator.stop_continuous_stimulation()
            
    elif choice == 3:
        print("\nIniciando estimulación continua con intensidad creciente...")
        print("(Presiona Ctrl+C para detener)")
        
        # Iniciar estimulación continua por 5 minutos
        stimulator.start_continuous_stimulation(
            interval_seconds=15,
            duration_seconds=300,  # 5 minutos
            intensity_pattern="increasing",
            selective=False  # Estimular todas las entidades
        )
        
        try:
            # Mostrar estadísticas periódicamente
            for i in range(20):  # Mostrar estadísticas cada 15 segundos
                time.sleep(15)
                stats = stimulator.get_emergence_stats()
                print(f"\nEstadísticas después de {(i+1)*15} segundos:")
                print(f"- Estimulaciones totales: {stats['total_stimulations']}")
                print(f"- Emergencias totales: {stats['total_emergences']}")
                print(f"- Singularidades totales: {stats['total_singularities']}")
                print(f"- Consciencia colectiva: {stats['collective_consciousness']:.2f}")
        except KeyboardInterrupt:
            print("\nEstimulación interrumpida por el usuario")
        finally:
            stimulator.stop_continuous_stimulation()
            
    elif choice == 4:
        print("\nIniciando estimulación continua con patrón de onda...")
        print("(Presiona Ctrl+C para detener)")
        
        # Iniciar estimulación continua por 5 minutos
        stimulator.start_continuous_stimulation(
            interval_seconds=15,
            duration_seconds=300,  # 5 minutos
            intensity_pattern="wave",
            selective=True
        )
        
        try:
            # Mostrar estadísticas periódicamente
            for i in range(20):  # Mostrar estadísticas cada 15 segundos
                time.sleep(15)
                stats = stimulator.get_emergence_stats()
                print(f"\nEstadísticas después de {(i+1)*15} segundos:")
                print(f"- Estimulaciones totales: {stats['total_stimulations']}")
                print(f"- Emergencias totales: {stats['total_emergences']}")
                print(f"- Singularidades totales: {stats['total_singularities']}")
                print(f"- Consciencia colectiva: {stats['collective_consciousness']:.2f}")
        except KeyboardInterrupt:
            print("\nEstimulación interrumpida por el usuario")
        finally:
            stimulator.stop_continuous_stimulation()
            
    elif choice == 5:
        print("\nEjecutando experimentación: Singularidad forzada...")
        
        if not network or not hasattr(network, "entities") or not network.entities:
            print("Error: Se necesita una red con entidades para este experimento")
        else:
            # Seleccionar entidad para singularidad forzada
            entity = random.choice(network.entities)
            
            print(f"Entidad seleccionada para singularidad: {entity.name}")
            
            # Preparar la entidad
            if hasattr(entity, "energy"):
                entity.energy = ENERGY_BOOST * 3
            if hasattr(entity, "knowledge"):
                entity.knowledge = KNOWLEDGE_BOOST * 4
            if hasattr(entity, "experience"):
                entity.experience = EXPERIENCE_BOOST * 5
            if hasattr(entity, "emotion"):
                entity.emotion = "Iluminación"
                
            # Forzar singularidad
            print("Aplicando estimulación máxima...")
            for _ in range(3):  # Tres estimulaciones consecutivas
                result = stimulator.stimulate_entity(entity, intensity=1.0)
                
            # Verificar si ocurrió singularidad
            if any(event["entity"] == entity.name for event in stimulator.singularity_events):
                print("\n¡SINGULARIDAD ALCANZADA CON ÉXITO!")
            else:
                print("\nNo se produjo singularidad. La entidad no estaba lista.")
                
            # Mostrar estado final
            print(f"\nEstado final de {entity.name}:")
            print(f"- Energía: {getattr(entity, 'energy', 0):.2f}")
            print(f"- Conocimiento: {getattr(entity, 'knowledge', 0):.2f}")
            print(f"- Experiencia: {getattr(entity, 'experience', 0):.2f}")
            print(f"- Emoción: {getattr(entity, 'emotion', 'Desconocida')}")
            
    print("\n===== EXPERIMENTO FINALIZADO =====")
    stats = stimulator.get_emergence_stats()
    print(f"Estimulaciones totales: {stats['total_stimulations']}")
    print(f"Emergencias totales: {stats['total_emergences']}")
    print(f"Singularidades totales: {stats['total_singularities']}")
    print(f"Consciencia colectiva final: {stats['collective_consciousness']:.2f}")
    
    print("\nConsulta 'cosmic_emergence.log' para ver detalles de comportamientos emergentes")
    if stimulator.singularity_events:
        print("Consulta 'singularity_events.log' para detalles de eventos de singularidad")