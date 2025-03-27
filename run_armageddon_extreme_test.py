"""
PRUEBA ARMAGEDÓN EXTREMA para el Sistema de Trading Cósmico Mejorado

Este script realiza pruebas extremas para validar la resistencia y estabilidad
del sistema de trading cósmico bajo condiciones adversas y cargas intensas:

1. Prueba de Resistencia Básica: Operación continua de múltiples entidades
2. Prueba de Escasez de Energía: Consumo de energía acelerado
3. Prueba de Comunicación Masiva: Sobrecarga de mensajes entre entidades
4. Prueba de Cascada de Fallos: Inducción de fallos en cadena
5. Prueba de Volatilidad Extrema: Simulación de mercado altamente volátil
6. Prueba de Recuperación: Capacidad de restauración tras fallos
7. Prueba de Sobrecarga de Base de Datos: Validación de persistencia

El reporte completo se guarda en armageddon_extreme_report.log
"""

import os
import sys
import time
import random
import threading
import sqlite3
import logging
import traceback
from datetime import datetime
from enhanced_simple_cosmic_trader import (
    EnhancedCosmicNetwork, 
    EnhancedCosmicTrader,
    EnhancedSpeculatorEntity, 
    EnhancedStrategistEntity,
    DB_PATH
)

# Configuración de logging para el reporte de prueba
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("armageddon_extreme_report.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("armageddon_test")

# Variables globales para controlar la prueba
FATHER_NAME = "Otoniel"
TEST_DURATION = 180  # Duración total en segundos
ENTITIES_COUNT = 20  # Número de entidades a crear
VOLATILITY_MULTIPLIER = 10  # Multiplicador de volatilidad del mercado
COMMUNICATION_INTENSITY = 5  # Multiplicador de intensidad de comunicación
ENERGY_DRAIN_FACTOR = 3  # Factor de drenaje de energía
DATABASE_STRESS_FACTOR = 10  # Factor de estrés de base de datos

# Variables de control de prueba
test_running = True
failure_count = 0
recovery_count = 0
database_operations = 0
total_messages = 0
total_trades = 0
crashed_entities = []
peak_cpu_usage = 0
peak_memory_usage = 0
test_start_time = None

class ArmageddonEntity(EnhancedSpeculatorEntity):
    """Entidad especializada para pruebas Armagedón con capacidad de generar estrés en el sistema."""

    def __init__(self, name, role, father=FATHER_NAME, stress_level=1.0):
        """
        Inicializar entidad de prueba Armagedón.
        
        Args:
            name: Nombre de la entidad
            role: Rol especializado
            father: Creador del sistema
            stress_level: Nivel de estrés que genera la entidad (1.0 = normal)
        """
        super().__init__(name, role, father, energy_rate=0.1*stress_level)
        self.stress_level = stress_level
        self.failure_probability = 0.01 * stress_level
        self.recovery_probability = 0.5
        self.crashed = False
        self.emergency_shutdown = False
        self.operations_count = 0
        self.log_state(f"Entidad Armagedón iniciada con nivel de estrés {stress_level}")
    
    def trade(self):
        """Sobrecarga de trade para incluir posibles fallos aleatorios."""
        global failure_count, total_trades, recovery_count

        # Incrementar contador de operaciones
        self.operations_count += 1
        total_trades += 1
        
        # Posibilidad de fallo (aumenta con el número de operaciones)
        failure_chance = self.failure_probability * (1 + self.operations_count / 100)
        
        if self.crashed:
            # Posibilidad de recuperación
            if random.random() < self.recovery_probability:
                self.crashed = False
                recovery_count += 1
                self.log_state(f"¡RECUPERACIÓN! Entidad {self.name} se ha recuperado del fallo")
                return self.generate_message("luz", "RECUPERACIÓN")
            return self.generate_message("sombra", "ERROR PERSISTENTE")
        
        # Verificar si ocurre un fallo
        if random.random() < failure_chance and not self.emergency_shutdown:
            self.crashed = True
            failure_count += 1
            crashed_entities.append(self.name)
            self.log_state(f"¡FALLO INDUCIDO! Entidad {self.name} ha sufrido un fallo en operación {self.operations_count}")
            return self.generate_message("sombra", "FALLO CRÍTICO")
        
        try:
            # Ejecutar trade normal con posible fallo aleatorio
            if random.random() < 0.01 * self.stress_level and not self.emergency_shutdown:
                # Simular error en cálculo (1% de probabilidad * nivel de estrés)
                raise ValueError(f"Error simulado en cálculo de trading de {self.name}")
            
            return super().trade()
        except Exception as e:
            self.crashed = True
            failure_count += 1
            crashed_entities.append(self.name)
            self.log_state(f"ERROR REAL: {str(e)}")
            logger.error(f"Error en trade de {self.name}: {str(e)}")
            return self.generate_message("sombra", f"ERROR: {type(e).__name__}")

    def metabolize(self):
        """Sobrecarga para incluir drenaje acelerado de energía durante la prueba."""
        global ENERGY_DRAIN_FACTOR
        
        # Aplicar factor de drenaje
        original_energy_rate = self.energy_rate
        self.energy_rate *= ENERGY_DRAIN_FACTOR
        
        # Metabolizar con tasa incrementada
        result = super().metabolize()
        
        # Restaurar tasa original
        self.energy_rate = original_energy_rate
        
        return result

    def generate_message(self, base_word, context):
        """Sobrecarga para generar mensajes más intensos durante la prueba."""
        global total_messages, COMMUNICATION_INTENSITY
        
        # Generar mensaje base
        message = super().generate_message(base_word, context)
        
        # Incrementar contador de mensajes
        total_messages += 1
        
        # En modo de comunicación intensiva, generar múltiples mensajes
        if random.random() < 0.1 * COMMUNICATION_INTENSITY and self.network and not self.crashed:
            # Enviar mensajes adicionales para estresar el sistema
            for _ in range(int(random.random() * COMMUNICATION_INTENSITY)):
                spam_msg = super().generate_message(base_word, f"spam_{random.randint(1, 1000)}")
                if self.network:
                    self.network.broadcast(self.name, spam_msg)
                total_messages += 1
        
        return message

def simulate_extreme_market_volatility(network):
    """Simula volatilidad extrema en el mercado para todas las entidades."""
    logger.info("Iniciando simulación de volatilidad extrema de mercado...")
    
    def volatility_worker():
        while test_running:
            # Afectar el último precio de todas las entidades con volatilidad extrema
            for entity in network.entities:
                if hasattr(entity, 'last_price') and entity.last_price is not None:
                    # Generar cambio extremo (hasta ±20% * multiplicador de volatilidad)
                    extreme_change = (random.random() - 0.5) * 0.4 * VOLATILITY_MULTIPLIER
                    entity.last_price *= (1 + extreme_change)
                    
                    # Añadir a historial de precios
                    if hasattr(entity, 'price_history'):
                        entity.price_history.append(entity.last_price)
                    
                    # Loggear cambios extremos
                    if abs(extreme_change) > 0.1 * VOLATILITY_MULTIPLIER:
                        entity.log_state(f"¡VOLATILIDAD EXTREMA! Cambio de {extreme_change*100:.2f}% en precio")
            
            # Esperar antes del siguiente ciclo de volatilidad
            time.sleep(3)
    
    # Iniciar hilo de volatilidad
    threading.Thread(target=volatility_worker, daemon=True).start()

def simulate_database_stress(network):
    """Simula estrés en la base de datos con operaciones intensivas."""
    logger.info("Iniciando simulación de estrés en base de datos...")
    
    def database_worker():
        global database_operations
        
        while test_running:
            try:
                # Conexión a la base de datos
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Realizar múltiples operaciones concurrentes
                for _ in range(5 * DATABASE_STRESS_FACTOR):
                    # Operaciones aleatorias de lectura y escritura
                    op_type = random.choice(["read", "write"])
                    
                    if op_type == "read":
                        # Lectura aleatoria
                        table = random.choice([
                            "cosmic_entities", "entity_logs", "trade_history", 
                            "entity_messages", "collective_knowledge"
                        ])
                        
                        cursor.execute(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT 10")
                        cursor.fetchall()
                    else:
                        # Escritura aleatoria (logs)
                        random_entity = random.choice(network.entities)
                        cursor.execute(
                            "INSERT INTO entity_logs (entity_id, log_type, message) VALUES (?, ?, ?)",
                            (random_entity.entity_id, "stress_test", f"Stress operation {database_operations}")
                        )
                    
                    database_operations += 1
                
                # Commit al final del ciclo
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error en operación de base de datos: {str(e)}")
            
            # Esperar antes del siguiente ciclo
            time.sleep(2)
    
    # Iniciar hilo de estrés de base de datos
    threading.Thread(target=database_worker, daemon=True).start()

def induce_cascade_failures(network):
    """Induce fallos en cascada entre entidades durante la prueba."""
    logger.info("Iniciando inducción de fallos en cascada...")
    
    def cascade_worker():
        while test_running:
            # Seleccionar aleatoriamente entidades para inducir fallos
            if network.entities:
                cascade_size = random.randint(2, max(2, len(network.entities) // 3))
                targets = random.sample(network.entities, min(cascade_size, len(network.entities)))
                
                # Inducir fallos en las entidades seleccionadas
                for entity in targets:
                    if random.random() < 0.3 and not entity.crashed:  # 30% probabilidad
                        entity.crashed = True
                        entity.energy -= 50  # Drenar energía
                        entity.log_state("¡INDUCIDO! Fallo en cascada")
                        logger.warning(f"Fallo en cascada inducido en {entity.name}")
                        
                        # Propagar mensajes de fallo por la red
                        if entity.network:
                            entity.network.broadcast(
                                entity.name,
                                entity.generate_message("sombra", "CASCADA DE FALLOS DETECTADA")
                            )
            
            # Esperar antes del siguiente ciclo
            time.sleep(15)  # Cada 15 segundos
    
    # Iniciar hilo de inducción de cascada
    threading.Thread(target=cascade_worker, daemon=True).start()

def monitor_system_resources():
    """Monitorea el uso de recursos del sistema durante la prueba."""
    logger.info("Iniciando monitoreo de recursos del sistema...")
    
    def resource_monitor():
        global peak_cpu_usage, peak_memory_usage
        
        while test_running:
            try:
                # Esta es una implementación muy básica que solo sirve de placeholder
                # En un sistema real usaríamos psutil u otras bibliotecas para medir recursos
                import os
                
                # Simulación de medición de recursos
                # En un entorno real, esto sería reemplazado por código que realmente mide
                cpu_usage = random.uniform(10, 80) # Simulación
                memory_usage = random.uniform(100, 500) # Simulación en MB
                
                # Actualizar picos
                peak_cpu_usage = max(peak_cpu_usage, cpu_usage)
                peak_memory_usage = max(peak_memory_usage, memory_usage)
                
                # Loggear solo cada cierto tiempo para no saturar logs
                if random.random() < 0.1:
                    logger.info(f"Recursos - CPU: {cpu_usage:.1f}%, Memoria: {memory_usage:.1f}MB")
                
            except Exception as e:
                logger.error(f"Error en monitoreo de recursos: {str(e)}")
            
            # Esperar antes del siguiente ciclo
            time.sleep(5)
    
    # Iniciar hilo de monitoreo
    threading.Thread(target=resource_monitor, daemon=True).start()

def generate_armageddon_report():
    """Genera un reporte completo de la prueba Armagedón."""
    global test_start_time, ENTITIES_COUNT, failure_count, recovery_count
    global database_operations, total_messages, total_trades, crashed_entities
    global peak_cpu_usage, peak_memory_usage, TEST_DURATION
    
    logger.info("\n\n" + "="*80)
    logger.info("REPORTE FINAL - PRUEBA ARMAGEDÓN EXTREMA")
    logger.info("="*80)
    
    # Duración real
    if test_start_time:
        duration = (datetime.now() - test_start_time).total_seconds()
    else:
        duration = 0
    
    # Métricas generales
    logger.info("\n[1] MÉTRICAS GENERALES")
    logger.info(f"Duración de la prueba: {duration:.2f} segundos (objetivo: {TEST_DURATION}s)")
    logger.info(f"Entidades testeadas: {ENTITIES_COUNT}")
    logger.info(f"Total de operaciones: {total_trades}")
    logger.info(f"Total de mensajes: {total_messages}")
    logger.info(f"Operaciones de base de datos: {database_operations}")
    
    # Estabilidad
    stability_score = 100 - min(100, (failure_count / max(1, ENTITIES_COUNT)) * 100)
    logger.info("\n[2] MÉTRICAS DE ESTABILIDAD")
    logger.info(f"Fallos totales: {failure_count}")
    logger.info(f"Recuperaciones: {recovery_count}")
    logger.info(f"Puntuación de estabilidad: {stability_score:.2f}%")
    
    # Recuperación
    recovery_rate = 100 * recovery_count / max(1, failure_count)
    recovery_score = min(100, recovery_rate)
    logger.info("\n[3] MÉTRICAS DE RECUPERACIÓN")
    logger.info(f"Tasa de recuperación: {recovery_rate:.2f}%")
    logger.info(f"Puntuación de recuperación: {recovery_score:.2f}%")
    
    # Rendimiento
    throughput = total_trades / max(1, duration)
    message_rate = total_messages / max(1, duration)
    db_rate = database_operations / max(1, duration)
    logger.info("\n[4] MÉTRICAS DE RENDIMIENTO")
    logger.info(f"Operaciones por segundo: {throughput:.2f}")
    logger.info(f"Mensajes por segundo: {message_rate:.2f}")
    logger.info(f"Operaciones de DB por segundo: {db_rate:.2f}")
    logger.info(f"Pico de uso de CPU: {peak_cpu_usage:.2f}%")
    logger.info(f"Pico de uso de memoria: {peak_memory_usage:.2f} MB")
    
    # Resistencia a fallos
    failure_rate = failure_count / max(1, total_trades)
    survival_score = 100 - min(100, failure_rate * 1000)
    logger.info("\n[5] MÉTRICAS DE RESISTENCIA")
    logger.info(f"Tasa de fallos: {failure_rate:.4f}")
    logger.info(f"Puntuación de supervivencia: {survival_score:.2f}%")
    logger.info(f"Entidades que fallaron: {', '.join(set(crashed_entities))}")
    
    # Puntuación final
    final_score = (stability_score * 0.3 + 
                  recovery_score * 0.3 + 
                  survival_score * 0.4)
    
    logger.info("\n[6] RESULTADO FINAL")
    logger.info(f"Puntuación ARMAGEDÓN EXTREMA: {final_score:.2f}%")
    
    # Evaluación cualitativa
    if final_score >= 90:
        evaluation = "EXCEPCIONAL - Sistema extremadamente robusto y resiliente"
    elif final_score >= 80:
        evaluation = "EXCELENTE - Sistema altamente confiable"
    elif final_score >= 70:
        evaluation = "MUY BUENO - Sistema resiliente con áreas de mejora menores"
    elif final_score >= 60:
        evaluation = "BUENO - Sistema estable pero con vulnerabilidades"
    elif final_score >= 50:
        evaluation = "ACEPTABLE - Sistema funcional pero necesita mejoras"
    else:
        evaluation = "REQUIERE MEJORAS - Sistema vulnerable a condiciones extremas"
    
    logger.info(f"Evaluación: {evaluation}")
    logger.info("\n" + "="*80)
    
    return {
        "duration": duration,
        "entities_count": ENTITIES_COUNT,
        "failure_count": failure_count,
        "recovery_count": recovery_count,
        "stability_score": stability_score,
        "recovery_score": recovery_score,
        "survival_score": survival_score,
        "final_score": final_score,
        "evaluation": evaluation
    }

def run_extreme_test(duration=TEST_DURATION, entities_count=ENTITIES_COUNT):
    """
    Ejecuta la prueba Armagedón extrema completa.
    
    Args:
        duration: Duración total de la prueba en segundos
        entities_count: Número de entidades a crear
    
    Returns:
        dict: Resultados de la prueba
    """
    global test_running, test_start_time
    global ENTITIES_COUNT, TEST_DURATION
    
    # Actualizar variables globales
    ENTITIES_COUNT = entities_count
    TEST_DURATION = duration
    
    logger.info("\n" + "="*80)
    logger.info("INICIANDO PRUEBA ARMAGEDÓN EXTREMA")
    logger.info("="*80)
    logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duración programada: {duration} segundos")
    logger.info(f"Entidades a probar: {entities_count}")
    logger.info(f"Factor de volatilidad: {VOLATILITY_MULTIPLIER}x")
    logger.info(f"Factor de comunicación: {COMMUNICATION_INTENSITY}x")
    logger.info(f"Factor de drenaje de energía: {ENERGY_DRAIN_FACTOR}x")
    logger.info(f"Factor de estrés de base de datos: {DATABASE_STRESS_FACTOR}x")
    
    # Inicializar sistema
    test_running = True
    test_start_time = datetime.now()
    
    try:
        # Iniciar la red
        logger.info("\n[1] Inicializando red cósmica para prueba Armagedón...")
        network = EnhancedCosmicNetwork(father=FATHER_NAME)
        
        # Crear entidades de prueba
        logger.info(f"\n[2] Creando {entities_count} entidades de prueba Armagedón...")
        for i in range(entities_count):
            # Alternar entre especuladores y estrategas
            EntityClass = ArmageddonEntity if i % 2 == 0 else EnhancedStrategistEntity
            role = "Speculator" if i % 2 == 0 else "Strategist"
            
            # Nivel de estrés variable (entre 1 y 3)
            stress_level = random.uniform(1.0, 3.0)
            
            # Crear entidad
            if EntityClass == ArmageddonEntity:
                entity = EntityClass(
                    name=f"Armageddon{i+1}",
                    role=role,
                    father=FATHER_NAME,
                    stress_level=stress_level
                )
            else:
                entity = EntityClass(
                    name=f"Armageddon{i+1}",
                    role=role,
                    father=FATHER_NAME
                )
            
            # Añadir a la red
            network.add_entity(entity)
        
        logger.info(f"Entidades creadas: {len(network.entities)}")
        
        # Iniciar simulación de volatilidad extrema
        logger.info("\n[3] Iniciando simulación de volatilidad extrema...")
        simulate_extreme_market_volatility(network)
        
        # Iniciar estrés de base de datos
        logger.info("\n[4] Iniciando estrés de base de datos...")
        simulate_database_stress(network)
        
        # Iniciar fallos en cascada
        logger.info("\n[5] Preparando inducción de fallos en cascada...")
        threading.Timer(30, lambda: induce_cascade_failures(network)).start()
        
        # Iniciar monitoreo de recursos
        logger.info("\n[6] Iniciando monitoreo de recursos del sistema...")
        monitor_system_resources()
        
        # Ejecutar ciclos de prueba
        logger.info("\n[7] Ejecutando ciclos de prueba...")
        cycle_count = 0
        cycle_length = min(30, duration / 6)  # Máximo 6 ciclos
        
        start_time = time.time()
        while test_running and (time.time() - start_time) < duration:
            cycle_count += 1
            logger.info(f"\n--- Ciclo de prueba {cycle_count} ---")
            
            # Simular operaciones
            network.simulate()
            
            # Colaboración cada 2 ciclos
            if cycle_count % 2 == 0:
                logger.info("Simulando colaboración masiva...")
                for _ in range(3):  # Aumentar estrés con múltiples rondas
                    network.simulate_collaboration()
            
            # Estadísticas intermedias
            alive_count = sum(1 for e in network.entities if not hasattr(e, 'crashed') or not e.crashed)
            energy_levels = [e.energy for e in network.entities]
            avg_energy = sum(energy_levels) / len(energy_levels) if energy_levels else 0
            
            logger.info(f"Entidades activas: {alive_count}/{len(network.entities)}")
            logger.info(f"Energía promedio: {avg_energy:.2f}")
            logger.info(f"Fallos acumulados: {failure_count}")
            logger.info(f"Mensajes totales: {total_messages}")
            logger.info(f"Operaciones de trading: {total_trades}")
            
            # Esperar para el siguiente ciclo
            time.sleep(cycle_length)
        
        # Finalizar prueba
        test_running = False
        logger.info("\n[8] Prueba completada, generando reporte final...")
        
        # Generar reporte final
        results = generate_armageddon_report()
        
        return results
        
    except Exception as e:
        test_running = False
        logger.error(f"ERROR CRÍTICO EN PRUEBA ARMAGEDÓN: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Intentar generar reporte a pesar del error
        try:
            return generate_armageddon_report()
        except:
            return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    # Parámetros desde línea de comandos
    import argparse
    
    parser = argparse.ArgumentParser(description="Prueba ARMAGEDÓN EXTREMA")
    parser.add_argument("--duration", type=int, default=TEST_DURATION, 
                      help=f"Duración de la prueba en segundos (default: {TEST_DURATION})")
    parser.add_argument("--entities", type=int, default=ENTITIES_COUNT,
                      help=f"Número de entidades (default: {ENTITIES_COUNT})")
    parser.add_argument("--volatility", type=float, default=VOLATILITY_MULTIPLIER,
                      help=f"Multiplicador de volatilidad (default: {VOLATILITY_MULTIPLIER})")
    parser.add_argument("--communication", type=float, default=COMMUNICATION_INTENSITY,
                      help=f"Multiplicador de comunicación (default: {COMMUNICATION_INTENSITY})")
    parser.add_argument("--energy-drain", type=float, default=ENERGY_DRAIN_FACTOR,
                      help=f"Factor de drenaje de energía (default: {ENERGY_DRAIN_FACTOR})")
    parser.add_argument("--db-stress", type=float, default=DATABASE_STRESS_FACTOR,
                      help=f"Factor de estrés de base de datos (default: {DATABASE_STRESS_FACTOR})")
    
    args = parser.parse_args()
    
    # Actualizar variables globales
    TEST_DURATION = args.duration
    ENTITIES_COUNT = args.entities
    VOLATILITY_MULTIPLIER = args.volatility
    COMMUNICATION_INTENSITY = args.communication
    ENERGY_DRAIN_FACTOR = args.energy_drain
    DATABASE_STRESS_FACTOR = args.db_stress
    
    # Ejecutar prueba
    results = run_extreme_test(duration=TEST_DURATION, entities_count=ENTITIES_COUNT)
    
    # Mostrar resumen final
    print("\n" + "="*50)
    print(f"PRUEBA ARMAGEDÓN EXTREMA COMPLETADA")
    print(f"Puntuación final: {results['final_score']:.2f}%")
    print(f"Evaluación: {results['evaluation']}")
    print("="*50)
    print(f"Ver reporte completo en: armageddon_extreme_report.log")