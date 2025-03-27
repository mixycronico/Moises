"""
Módulo de Competencia Cósmica para el Sistema Genesis.

Este módulo implementa un mecanismo de competencia y compartición de sabiduría
entre las entidades del sistema, permitiendo que las entidades más avanzadas
compartan su conocimiento con las menos desarrolladas.
"""

import random
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cosmic_competition")

# Umbrales para compartir sabiduría
ENERGIA_MINIMA = 50
NIVEL_MINIMO = 3
CONOCIMIENTO_MINIMO = 20.0

def compartir_sabiduria(entities, knowledge_pool=0.0):
    """
    Función para compartir sabiduría entre entidades.
    
    Args:
        entities: Lista de entidades (diccionarios o objetos)
        knowledge_pool: Valor inicial del pool de conocimiento compartido
        
    Returns:
        Tupla (logs, nuevo_knowledge_pool)
    """
    logs = []
    
    # Convertir entidades a formato consistente si son objetos y no diccionarios
    entidades_normalizadas = []
    for e in entities:
        if isinstance(e, dict):
            entidades_normalizadas.append(e)
        else:
            # Si es objeto, convertir atributos relevantes a diccionario
            entidades_normalizadas.append({
                "name": getattr(e, "name", f"Entidad_{id(e)}"),
                "energy": getattr(e, "energy", 0),
                "level": getattr(e, "level", 1),
                "knowledge": getattr(e, "knowledge", 0),
                "_object": e  # Guardar referencia al objeto original
            })
    
    # Identificar sabios que pueden compartir conocimiento
    sabios = [e for e in entidades_normalizadas if e["energy"] > ENERGIA_MINIMA and 
              e["level"] >= NIVEL_MINIMO and e["knowledge"] >= CONOCIMIENTO_MINIMO]

    if not sabios:
        logger.info("No hay entidades con suficiente energía, nivel y conocimiento para compartir sabiduría.")
        return logs, knowledge_pool

    for sabio in sabios:
        # Identificar receptores (entidades con menos conocimiento)
        receptores = [e for e in entidades_normalizadas if e["name"] != sabio["name"] and 
                      e["knowledge"] < sabio["knowledge"]]
        
        if not receptores:
            logger.info(f"[{sabio['name']}] No hay entidades con menos conocimiento para recibir sabiduría.")
            continue

        for receptor in receptores:
            # Calcular transferencia de conocimiento (5% del conocimiento del sabio)
            transferencia = round(sabio["knowledge"] * 0.05, 4)
            
            # Aplicar transferencia al receptor
            if "_object" in receptor:
                # Si guardamos referencia al objeto original, actualizar su conocimiento
                if hasattr(receptor["_object"], "receive_knowledge") and callable(getattr(receptor["_object"], "receive_knowledge")):
                    receptor["_object"].receive_knowledge(transferencia)
                elif hasattr(receptor["_object"], "knowledge"):
                    receptor["_object"].knowledge += transferencia
            receptor["knowledge"] += transferencia
            
            # Actualizar energía del sabio
            if "_object" in sabio:
                if hasattr(sabio["_object"], "adjust_energy"):
                    sabio["_object"].adjust_energy(-0.5)
                elif hasattr(sabio["_object"], "energy"):
                    sabio["_object"].energy -= 0.5
            sabio["energy"] -= 0.5
            
            # Registrar la transferencia
            log = f"[{datetime.now().isoformat()}] {sabio['name']} compartió {transferencia} de sabiduría con {receptor['name']}"
            logs.append(log)
            logger.info(log)

            # Incrementar el pool colectivo (20% de la transferencia)
            knowledge_pool += transferencia * 0.2

    return logs, knowledge_pool


class CosmicCompetition:
    """
    Clase que implementa un sistema de competencia entre entidades cósmicas,
    con evaluación de capacidades y compartición de sabiduría.
    """
    
    def __init__(self, entities=None):
        """
        Inicializar competición cósmica.
        
        Args:
            entities: Lista de entidades (opcional)
        """
        self.entities = entities or []
        self.last_competition = None
        self.competition_history = []
        self.knowledge_pool = 0.0
        
    def add_entity(self, entity):
        """
        Añadir entidad a la competencia.
        
        Args:
            entity: Entidad a añadir
        """
        self.entities.append(entity)
        
    def set_entities(self, entities):
        """
        Establecer lista completa de entidades.
        
        Args:
            entities: Lista de entidades
        """
        self.entities = entities
        
    def evaluar_entidad(self, entidad):
        """
        Evaluar el poder general de una entidad.
        
        Args:
            entidad: Entidad a evaluar (objeto o diccionario)
            
        Returns:
            Puntuación de poder
        """
        # Si es diccionario, usar valores directamente
        if isinstance(entidad, dict):
            return (
                entidad.get("knowledge", 0) * 2 +
                entidad.get("experience", 0) * 1.5 +
                entidad.get("energy", 0) * 0.5 +
                entidad.get("level", 1)
            )
        
        # Si es objeto, usar atributos
        return (
            getattr(entidad, "knowledge", 0) * 2 +
            getattr(entidad, "experience", 0) * 1.5 +
            getattr(entidad, "energy", 0) * 0.5 +
            getattr(entidad, "level", 1)
        )
        
    def competir(self):
        """
        Ejecutar una ronda de competencia.
        
        Returns:
            Diccionario con resultados de la competencia
        """
        logger.info("== Comienza la Competencia Cósmica ==")
        puntuaciones = {}
        
        # Si no hay entidades, retornar resultados vacíos
        if not self.entities:
            logger.warning("No hay entidades para competir")
            return {
                "timestamp": datetime.now().isoformat(),
                "ranking": [],
                "campeon": None,
                "logs_sabiduria": []
            }
        
        # Evaluar cada entidad
        for entidad in self.entities:
            nombre = entidad["name"] if isinstance(entidad, dict) else getattr(entidad, "name", f"Entidad_{id(entidad)}")
            puntuaciones[nombre] = self.evaluar_entidad(entidad)

        # Ordenar por puntuación
        ranking = sorted(puntuaciones.items(), key=lambda x: x[1], reverse=True)

        logger.info("\n-- Resultados de la Competencia --")
        for i, (nombre, puntaje) in enumerate(ranking, 1):
            logger.info(f"{i}. {nombre} - Poder Cósmico: {round(puntaje, 3)}")

        # La mejor entidad comparte su sabiduría con otras
        campeon = ranking[0][0]
        logs_sabiduria = self.compartir_sabiduria_campeon(campeon)
        
        # Actualizar estado de la competencia
        self.last_competition = datetime.now()
        resultado = {
            "timestamp": self.last_competition.isoformat(),
            "ranking": ranking,
            "campeon": campeon,
            "logs_sabiduria": logs_sabiduria
        }
        
        self.competition_history.append(resultado)
        return resultado

    def compartir_sabiduria_campeon(self, nombre_ganador):
        """
        El campeón comparte su sabiduría con el resto de entidades.
        
        Args:
            nombre_ganador: Nombre de la entidad ganadora
            
        Returns:
            Lista de logs de compartición
        """
        # Encontrar entidad ganadora
        ganador = None
        for entidad in self.entities:
            if isinstance(entidad, dict) and entidad["name"] == nombre_ganador:
                ganador = entidad
                break
            elif hasattr(entidad, "name") and entidad.name == nombre_ganador:
                ganador = entidad
                break
                
        if not ganador:
            logger.error(f"No se encontró la entidad ganadora: {nombre_ganador}")
            return []
            
        logger.info(f"\n{nombre_ganador} comparte su sabiduría...")
        
        # Calcular sabiduría a compartir (25% del conocimiento del ganador)
        sabiduria = (ganador["knowledge"] if isinstance(ganador, dict) else getattr(ganador, "knowledge", 0)) * 0.25
        logs = []

        # Compartir con cada entidad
        for entidad in self.entities:
            # Saltar al ganador
            nombre_entidad = entidad["name"] if isinstance(entidad, dict) else getattr(entidad, "name", f"Entidad_{id(entidad)}")
            if nombre_entidad == nombre_ganador:
                continue
                
            # Actualizar conocimiento de la entidad
            if isinstance(entidad, dict):
                entidad["knowledge"] += sabiduria
            elif hasattr(entidad, "receive_knowledge") and callable(getattr(entidad, "receive_knowledge")):
                entidad.receive_knowledge(sabiduria)
            elif hasattr(entidad, "knowledge"):
                entidad.knowledge += sabiduria
                
            # Registrar transferencia
            log = f"{nombre_entidad} recibe {round(sabiduria, 4)} puntos de conocimiento."
            logs.append(log)
            logger.info(log)

        logger.info("\n== Sabiduría compartida con éxito ==")
        return logs
        
    def programar_competencias(self, intervalo_segundos=60, num_competencias=None):
        """
        Programar competencias periódicas.
        
        Args:
            intervalo_segundos: Tiempo entre competencias
            num_competencias: Número máximo de competencias (None para indefinido)
            
        Returns:
            Thread de competencias
        """
        import threading
        
        def ejecutar_competencias():
            count = 0
            try:
                while True:
                    self.competir()
                    count += 1
                    
                    # Si llegamos al número máximo, salir
                    if num_competencias is not None and count >= num_competencias:
                        logger.info(f"Completadas {count} competencias programadas")
                        break
                        
                    # Esperar para la siguiente competencia
                    time.sleep(intervalo_segundos)
            except Exception as e:
                logger.error(f"Error en hilo de competencias: {str(e)}")
                
        # Crear y arrancar hilo
        thread = threading.Thread(target=ejecutar_competencias)
        thread.daemon = True
        thread.start()
        
        return thread
        
    def get_competition_stats(self):
        """
        Obtener estadísticas de competiciones.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.competition_history:
            return {
                "total_competitions": 0,
                "unique_champions": [],
                "champion_frequency": {},
                "knowledge_pool": self.knowledge_pool
            }
            
        # Contar frecuencia de campeones
        champion_count = {}
        for comp in self.competition_history:
            champion = comp["campeon"]
            champion_count[champion] = champion_count.get(champion, 0) + 1
            
        return {
            "total_competitions": len(self.competition_history),
            "unique_champions": list(champion_count.keys()),
            "champion_frequency": champion_count,
            "knowledge_pool": self.knowledge_pool,
            "last_competition": self.last_competition.isoformat() if self.last_competition else None
        }


# Función de conveniencia para obtener todas las entidades del sistema
def get_all_entities(network=None):
    """
    Obtener todas las entidades del sistema.
    
    Args:
        network: Red cósmica (opcional)
        
    Returns:
        Lista de entidades
    """
    if network is None:
        # Intentar importar la red desde el sistema principal
        try:
            import cosmic_trading
            if hasattr(cosmic_trading, "get_network") and callable(cosmic_trading.get_network):
                network = cosmic_trading.get_network()
        except (ImportError, AttributeError):
            pass
            
    if network and hasattr(network, "entities"):
        # Si es diccionario de entidades
        if isinstance(network.entities, dict):
            return list(network.entities.values())
        # Si es lista de entidades
        elif isinstance(network.entities, list):
            return network.entities
            
    logger.warning("No se pudo obtener lista de entidades")
    return []


# Ejemplo de uso
if __name__ == "__main__":
    # Crear entidades de ejemplo
    entities = [
        {"name": "Entidad1", "energy": 100, "level": 5, "knowledge": 30, "experience": 20},
        {"name": "Entidad2", "energy": 80, "level": 4, "knowledge": 25, "experience": 15},
        {"name": "Entidad3", "energy": 60, "level": 3, "knowledge": 20, "experience": 10},
        {"name": "Entidad4", "energy": 40, "level": 2, "knowledge": 15, "experience": 5},
        {"name": "Entidad5", "energy": 20, "level": 1, "knowledge": 10, "experience": 2}
    ]
    
    # Crear competencia
    torneo = CosmicCompetition(entities)
    
    # Ejecutar una ronda de competencia
    resultado = torneo.competir()
    
    # Mostrar resultados
    print("\nResultados de la competencia:")
    for rank, (nombre, poder) in enumerate(resultado["ranking"], 1):
        print(f"{rank}. {nombre}: {poder:.2f}")
    
    print(f"\nCampeón: {resultado['campeon']}")
    print("\nTransferencias de sabiduría:")
    for log in resultado["logs_sabiduria"]:
        print(f"- {log}")
        
    # Compartir sabiduría entre entidades
    logs, knowledge_pool = compartir_sabiduria(entities)
    
    print("\nCompartición directa de sabiduría:")
    for log in logs:
        print(f"- {log}")
    
    print(f"\nPool de conocimiento colectivo: {knowledge_pool:.2f}")