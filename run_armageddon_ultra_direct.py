#!/usr/bin/env python3
"""
PRUEBA ARMAGEDÓN ULTRA DIRECTA para el Sistema de Trading Cósmico

Este script ejecuta directamente una versión extrema de las pruebas Armagedón,
forzando al sistema al máximo con:

1. Máximo número de entidades
2. Comunicación masiva entre entidades
3. Drenaje extremo de energía
4. Volatilidad de mercado al límite
5. Inducción de fallos en cascada
6. Pruebas de recuperación bajo presión
7. Estrés de base de datos intenso
8. Ejecución de operaciones a máxima velocidad

ADVERTENCIA: Este script es extremadamente intensivo en recursos y puede causar
inestabilidad en el sistema. Solo debe usarse en entornos de prueba controlados.
"""

import os
import sys
import time
import random
import threading
import logging
import traceback
from datetime import datetime
import concurrent.futures
from enhanced_simple_cosmic_trader import (
    EnhancedCosmicNetwork, 
    EnhancedCosmicTrader,
    EnhancedSpeculatorEntity, 
    EnhancedStrategistEntity,
    DB_PATH
)

# Ajustar configuración para prueba ultra extrema
MAX_ENTITIES = 50             # Número de entidades a crear
THREADS = 8                   # Número de hilos para procesamiento paralelo
TEST_DURATION = 60            # Duración en segundos
VOLATILITY_FACTOR = 30        # Multiplicador de volatilidad
ENERGY_DRAIN_FACTOR = 10      # Factor de drenaje de energía
COMMUNICATION_FACTOR = 15     # Factor de intensidad de comunicación
DATABASE_STRESS_FACTOR = 20   # Intensidad de operaciones de base de datos
FAILURE_PROBABILITY = 0.25    # Probabilidad de fallo inducido (0-1)

# Configuración de logging
LOG_FILE = "armageddon_ultra_direct_report.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("armageddon_ultra")

# Variables de control
test_running = True
total_operations = 0
total_failures = 0
total_recoveries = 0
operations_per_second = 0
last_operations = 0
last_time = time.time()

class UltraStressEntity(EnhancedSpeculatorEntity):
    """Entidad especializada para generar estrés extremo en el sistema."""
    
    def __init__(self, name, role, father="otoniel", stress_level=1.0):
        """Inicializar entidad con nivel de estrés aumentado."""
        super().__init__(name, role, father, energy_rate=0.2*stress_level)
        self.stress_level = stress_level
        self.failure_chance = FAILURE_PROBABILITY * stress_level
        self.operations = 0
        self.failures = 0
        self.recoveries = 0
        self.crashed = False
        
    def trade(self):
        """Operación de trading con posibilidad de fallo controlado."""
        global total_operations, total_failures, total_recoveries
        
        total_operations += 1
        self.operations += 1
        
        # Si estaba crasheada, intentar recuperarse
        if self.crashed:
            if random.random() < 0.4:  # 40% chance de recuperación
                self.crashed = False
                self.recoveries += 1
                total_recoveries += 1
                self.log_state(f"RECUPERADO después de fallo ({self.failures} fallos)")
                return self.generate_message("recuperación", "Sistema estabilizado")
            return self.generate_message("error", "Sistema aún inestable")
        
        # Posibilidad de fallo controlado
        if random.random() < self.failure_chance * (1 + self.operations / 1000):
            self.crashed = True
            self.failures += 1
            total_failures += 1
            self.log_state(f"FALLO INDUCIDO (op #{self.operations})")
            return self.generate_message("error", "Fallo crítico detectado")
        
        # Trading normal con volatilidad extrema
        try:
            # Simular volatilidad extrema
            if hasattr(self, 'last_price') and self.last_price is not None:
                extreme_change = (random.random() - 0.5) * 0.5 * VOLATILITY_FACTOR
                self.last_price *= (1 + extreme_change)
                if hasattr(self, 'price_history'):
                    self.price_history.append(self.last_price)
            
            # Drenaje acelerado de energía
            original_energy = self.energy
            self.energy -= random.uniform(0.1, 0.5) * ENERGY_DRAIN_FACTOR
            if self.energy < 0:
                self.energy = 0
                self.crashed = True
                self.failures += 1
                total_failures += 1
                self.log_state(f"COLAPSO POR ENERGÍA: {original_energy:.2f} -> 0")
                return self.generate_message("energía", "Colapso por agotamiento")
                
            return super().trade()
        except Exception as e:
            self.crashed = True
            self.failures += 1
            total_failures += 1
            logger.error(f"ERROR real en {self.name}: {str(e)}")
            return self.generate_message("excepción", str(e))
            
    def generate_message(self, base_word, context):
        """Sobrecarga para generar comunicación masiva."""
        global COMMUNICATION_FACTOR
        
        # Mensaje base
        message = super().generate_message(base_word, context)
        
        # Comunicación masiva si está activa la prueba y no está crasheada
        if random.random() < 0.15 * COMMUNICATION_FACTOR and self.network and not self.crashed:
            # Enviar varios mensajes adicionales para estresar la red
            for _ in range(int(random.random() * COMMUNICATION_FACTOR)):
                spam_word = random.choice(["estrés", "sobrecarga", "prueba", "límite", "armagedón"])
                spam_msg = super().generate_message(spam_word, f"comunicación_masiva_{random.randint(1,1000)}")
                if self.network:
                    try:
                        self.network.broadcast(self.name, spam_msg)
                    except Exception as e:
                        logger.error(f"Error en broadcast masivo: {str(e)}")
        
        return message
        
    def metabolize(self):
        """Sobrecarga para drenaje extremo de energía."""
        global ENERGY_DRAIN_FACTOR
        
        # Aplicar drenaje extremo
        original_rate = self.energy_rate
        self.energy_rate *= ENERGY_DRAIN_FACTOR
        
        # Proceso normal
        result = super().metabolize()
        
        # Restaurar rate
        self.energy_rate = original_rate
        
        return result

def create_ultra_stress_network(entity_count=MAX_ENTITIES):
    """Crear red de entidades para prueba de estrés extremo."""
    logger.info(f"Creando red de estrés con {entity_count} entidades...")
    
    # Crear red
    network = EnhancedCosmicNetwork("otoniel")
    
    # Crear entidades con niveles variados de estrés
    for i in range(entity_count):
        # Alternar entre tipos y niveles de estrés
        entity_type = "Speculator" if i % 2 == 0 else "Strategist"
        stress_level = random.uniform(0.5, 2.0)  # Variar niveles de estrés
        name = f"Stress{i+1}_{entity_type[:3]}"
        
        # Crear entidad según tipo
        if entity_type == "Speculator":
            entity = UltraStressEntity(name, entity_type, stress_level=stress_level)
        else:
            entity = UltraStressEntity(name, entity_type, stress_level=stress_level)
            
        # Añadir a la red
        network.add_entity(entity)
        logger.info(f"Creada entidad {name} (nivel estrés: {stress_level:.2f})")
    
    return network

def stress_database(db_path, intensity=DATABASE_STRESS_FACTOR):
    """Realizar operaciones intensivas de base de datos."""
    import sqlite3
    import time
    
    logger.info(f"Iniciando estrés de base de datos (factor: {intensity})...")
    
    def db_worker():
        operations = 0
        while test_running:
            try:
                # Conectar a DB
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Ejecutar varias operaciones
                for _ in range(int(random.random() * 10 * intensity)):
                    table = random.choice([
                        "cosmic_entities", "entity_logs", "trade_history", 
                        "entity_messages", "collective_knowledge"
                    ])
                    
                    # Aleatoriamente elegir operación
                    op = random.choice(["select", "insert", "update"])
                    
                    if op == "select":
                        cursor.execute(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT ?", 
                                      (int(random.random() * 20),))
                        cursor.fetchall()
                    elif op == "insert" and table == "entity_logs":
                        cursor.execute(
                            "INSERT INTO entity_logs (entity_id, log_type, message) VALUES (?, ?, ?)",
                            (random.randint(1, MAX_ENTITIES), "stress_test", f"DB stress op {operations}")
                        )
                    elif op == "update" and table == "cosmic_entities":
                        cursor.execute(
                            "UPDATE cosmic_entities SET level = level + 0.01 WHERE id = ?",
                            (random.randint(1, MAX_ENTITIES),)
                        )
                    
                    operations += 1
                
                # Commit y cerrar
                conn.commit()
                conn.close()
                
                # Pequeña pausa para no saturar completamente
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error en operación DB: {str(e)}")
    
    # Iniciar varios hilos de estrés de base de datos
    for i in range(min(4, THREADS)):
        t = threading.Thread(target=db_worker, daemon=True)
        t.start()
        logger.info(f"Iniciado hilo de estrés DB #{i+1}")

def run_trading_operations(network, concurrent=True):
    """Ejecutar operaciones de trading masivas."""
    logger.info(f"Iniciando operaciones de trading masivas (concurrencia: {concurrent})...")
    
    def entity_worker(entity):
        """Trabajo intensivo para una entidad."""
        while test_running and entity in network.entities:
            try:
                # Operación de trading
                result = entity.trade()
                
                # Simular colaboración ocasional
                if random.random() < 0.05:  # 5% de probabilidad
                    network.simulate_collaboration(subset_size=min(5, len(network.entities)))
                
                # Pequeña pausa para no saturar completamente
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error en operación de {entity.name}: {str(e)}")
    
    # Ejecutar en múltiples hilos si es concurrente
    if concurrent:
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = []
            for entity in network.entities:
                futures.append(executor.submit(entity_worker, entity))
    else:
        # Crear hilos manualmente
        threads = []
        for entity in network.entities:
            t = threading.Thread(target=entity_worker, args=(entity,), daemon=True)
            threads.append(t)
            t.start()

def monitor_performance():
    """Monitorear rendimiento durante la prueba."""
    global total_operations, last_operations, last_time, operations_per_second
    
    logger.info("Iniciando monitoreo de rendimiento...")
    
    def monitor_worker():
        global total_operations, last_operations, last_time, operations_per_second
        
        while test_running:
            try:
                current_time = time.time()
                time_diff = current_time - last_time
                
                if time_diff >= 1.0:  # Actualizar cada segundo
                    ops_diff = total_operations - last_operations
                    operations_per_second = ops_diff / time_diff
                    
                    logger.info(f"Rendimiento: {operations_per_second:.2f} ops/s | "
                               f"Total: {total_operations} | "
                               f"Fallos: {total_failures} | "
                               f"Recuperaciones: {total_recoveries}")
                    
                    last_operations = total_operations
                    last_time = current_time
                
                time.sleep(1.0)
            
            except Exception as e:
                logger.error(f"Error en monitoreo: {str(e)}")
    
    # Iniciar hilo de monitoreo
    t = threading.Thread(target=monitor_worker, daemon=True)
    t.start()

def chaos_monkey():
    """Simular eventos caóticos aleatorios en el sistema."""
    logger.info("Iniciando Chaos Monkey para eventos aleatorios...")
    
    def chaos_worker(network):
        while test_running:
            try:
                # Esperar un tiempo aleatorio
                time.sleep(random.uniform(5, 15))
                
                # Seleccionar evento caótico
                event = random.choice([
                    "kill_random", "spike_volatility", "energy_drain", 
                    "broadcast_storm", "reset_entity", "add_entity"
                ])
                
                if event == "kill_random" and network.entities:
                    # Matar aleatoriamente una entidad
                    target = random.choice(network.entities)
                    target.crashed = True
                    target.energy = 0
                    logger.warning(f"CHAOS: Entidad {target.name} eliminada")
                
                elif event == "spike_volatility":
                    # Generar spike de volatilidad extrema
                    for entity in network.entities:
                        if hasattr(entity, 'last_price') and entity.last_price is not None:
                            spike = random.choice([-0.9, -0.7, -0.5, 0.5, 0.7, 0.9])
                            entity.last_price *= (1 + spike)
                    logger.warning(f"CHAOS: Spike de volatilidad de {spike*100:.1f}%")
                
                elif event == "energy_drain":
                    # Drenar energía masivamente
                    for entity in network.entities:
                        entity.energy *= random.uniform(0.3, 0.8)
                    logger.warning("CHAOS: Drenaje masivo de energía")
                
                elif event == "broadcast_storm":
                    # Generar tormenta de mensajes
                    if network.entities:
                        sender = random.choice(network.entities)
                        for _ in range(50):
                            msg = sender.generate_message("caos", "tormenta_mensajes")
                            network.broadcast(sender.name, msg)
                        logger.warning(f"CHAOS: Tormenta de 50 mensajes de {sender.name}")
                
                elif event == "reset_entity" and network.entities:
                    # Resetear una entidad al estado inicial
                    target = random.choice(network.entities)
                    target.level = 1.0
                    target.energy = 100.0
                    target.crashed = False
                    logger.warning(f"CHAOS: Reset de {target.name} a estado inicial")
                
                elif event == "add_entity":
                    # Añadir entidad nueva durante ejecución
                    name = f"Chaos{random.randint(100, 999)}"
                    entity = UltraStressEntity(name, "Speculator", stress_level=2.0)
                    network.add_entity(entity)
                    logger.warning(f"CHAOS: Añadida nueva entidad {name}")
            
            except Exception as e:
                logger.error(f"Error en Chaos Monkey: {str(e)}")
    
    # Iniciar hilo de chaos monkey
    def start_chaos(network):
        t = threading.Thread(target=chaos_worker, args=(network,), daemon=True)
        t.start()
        logger.info("Chaos Monkey iniciado")
    
    return start_chaos

def generate_final_report():
    """Generar reporte final de la prueba."""
    global total_operations, total_failures, total_recoveries, operations_per_second
    
    logger.info("\n" + "="*80)
    logger.info("REPORTE FINAL - PRUEBA ARMAGEDÓN ULTRA DIRECTA")
    logger.info("="*80)
    
    # Métricas generales
    logger.info("\n[1] MÉTRICAS GENERALES")
    logger.info(f"Duración de la prueba: {TEST_DURATION} segundos")
    logger.info(f"Entidades probadas: {MAX_ENTITIES}")
    logger.info(f"Factor de volatilidad: {VOLATILITY_FACTOR}x")
    logger.info(f"Factor de drenaje de energía: {ENERGY_DRAIN_FACTOR}x")
    logger.info(f"Factor de comunicación: {COMMUNICATION_FACTOR}x")
    
    # Métricas de rendimiento
    logger.info("\n[2] MÉTRICAS DE RENDIMIENTO")
    logger.info(f"Total de operaciones: {total_operations}")
    logger.info(f"Operaciones por segundo: {operations_per_second:.2f}")
    logger.info(f"Operaciones por entidad: {total_operations/MAX_ENTITIES:.2f}")
    
    # Métricas de estabilidad
    failure_rate = total_failures / max(1, total_operations)
    logger.info("\n[3] MÉTRICAS DE ESTABILIDAD")
    logger.info(f"Total de fallos: {total_failures}")
    logger.info(f"Tasa de fallos: {failure_rate:.6f} ({failure_rate*100:.4f}%)")
    logger.info(f"Total de recuperaciones: {total_recoveries}")
    logger.info(f"Ratio recuperación/fallos: {total_recoveries/max(1,total_failures):.2f}")
    
    # Puntuación de resistencia
    resistance_score = max(0, 100 - (failure_rate * 10000))
    recovery_score = min(100, (total_recoveries / max(1, total_failures)) * 100)
    performance_factor = min(100, operations_per_second)
    
    final_score = (resistance_score * 0.5 + recovery_score * 0.3 + performance_factor * 0.2)
    
    logger.info("\n[4] PUNTUACIONES")
    logger.info(f"Resistencia a fallos: {resistance_score:.2f}/100")
    logger.info(f"Recuperación de fallos: {recovery_score:.2f}/100")
    logger.info(f"Factor de rendimiento: {performance_factor:.2f}/100")
    logger.info(f"PUNTUACIÓN FINAL: {final_score:.2f}/100")
    
    # Determinar resultado
    if final_score >= 90:
        result = "EXCEPCIONAL"
    elif final_score >= 75:
        result = "EXCELENTE"
    elif final_score >= 60:
        result = "BUENO"
    elif final_score >= 40:
        result = "ACEPTABLE"
    else:
        result = "NECESITA MEJORAS"
    
    logger.info(f"\nRESULTADO: {result}")
    logger.info("="*80)
    
    # Guardar también en archivo separado
    with open("armageddon_ultra_result.txt", "w") as f:
        f.write(f"PRUEBA ARMAGEDÓN ULTRA DIRECTA\n")
        f.write(f"PUNTUACIÓN FINAL: {final_score:.2f}/100\n")
        f.write(f"RESULTADO: {result}\n\n")
        f.write(f"Operaciones totales: {total_operations}\n")
        f.write(f"Tasa de fallos: {failure_rate*100:.4f}%\n")
        f.write(f"Rendimiento: {operations_per_second:.2f} ops/s\n")
    
    print(f"\nPrueba ARMAGEDÓN ULTRA DIRECTA completada con puntuación {final_score:.2f}/100 - {result}")
    print(f"Reporte detallado guardado en {LOG_FILE}")
    
def main():
    """Función principal."""
    global test_running, last_time
    
    print("\n" + "="*80)
    print("INICIANDO PRUEBA ARMAGEDÓN ULTRA DIRECTA")
    print("="*80)
    print(f"Configuración:")
    print(f"- Entidades: {MAX_ENTITIES}")
    print(f"- Duración: {TEST_DURATION} segundos")
    print(f"- Volatilidad: {VOLATILITY_FACTOR}x")
    print(f"- Drenaje de energía: {ENERGY_DRAIN_FACTOR}x")
    print(f"- Comunicación: {COMMUNICATION_FACTOR}x")
    print(f"- Estrés DB: {DATABASE_STRESS_FACTOR}x")
    print(f"- Probabilidad de fallos: {FAILURE_PROBABILITY*100:.1f}%")
    print("="*80)
    
    # Inicializar tiempo
    last_time = time.time()
    start_time = last_time
    
    try:
        # Crear red de prueba
        network = create_ultra_stress_network(MAX_ENTITIES)
        
        # Iniciar monitoreo de rendimiento
        monitor_performance()
        
        # Iniciar estrés de base de datos
        stress_database(DB_PATH, DATABASE_STRESS_FACTOR)
        
        # Iniciar chaos monkey
        chaos = chaos_monkey()
        chaos(network)
        
        # Iniciar operaciones de trading
        run_trading_operations(network, concurrent=True)
        
        # Ejecutar por duración especificada
        logger.info(f"Prueba iniciada - ejecutando por {TEST_DURATION} segundos...")
        time.sleep(TEST_DURATION)
        
        # Finalizar prueba
        test_running = False
        logger.info("Finalizando prueba...")
        time.sleep(2)  # Dar tiempo a que terminen los hilos
        
        # Generar reporte final
        generate_final_report()
        
    except KeyboardInterrupt:
        logger.info("Prueba interrumpida por usuario")
        test_running = False
    except Exception as e:
        logger.error(f"Error en prueba: {str(e)}")
        traceback.print_exc()
    finally:
        test_running = False

if __name__ == "__main__":
    main()