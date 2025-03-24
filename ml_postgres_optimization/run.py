#!/usr/bin/env python3
"""
Sistema de Optimización con ML para PostgreSQL - Script Principal

Este script integra todos los componentes del Sistema de Optimización ML,
permitiendo entrenar modelos, iniciar monitoreo en tiempo real y ejecutar pruebas.

Modos de ejecución:
1. Optimización con ML: Entrena y aplica modelos ML para optimizar parámetros PostgreSQL
2. Prueba Extendida: Ejecuta pruebas de 72+ horas para validar estabilidad
3. Configuración Inicial: Configura la base de datos PostgreSQL con las tablas y funciones necesarias
"""

import os
import sys
import logging
import argparse
import subprocess
import time
from ml_optimizer import MLPostgresOptimizer
import psycopg2

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ML_Postgres_Optimizer')

def setup_database():
    """Configura la base de datos ejecutando el script SQL."""
    try:
        # Conexión a PostgreSQL
        conn = psycopg2.connect(
            dbname=os.environ.get("POSTGRES_DB", "postgres"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", "5432")
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        # Ejecutar script de configuración
        with open('setup_db.sql', 'r') as f:
            setup_sql = f.read()
            cur.execute(setup_sql)
        
        logger.info("Base de datos configurada correctamente")
        return True
    except Exception as e:
        logger.error(f"Error configurando la base de datos: {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def run_optimizer(duration_hours=None):
    """Ejecuta el optimizador ML para PostgreSQL."""
    logger.info(f"Iniciando optimizador ML {'(duración: indefinida)' if duration_hours is None else f'(duración: {duration_hours} horas)'}")
    
    optimizer = MLPostgresOptimizer()
    try:
        # Entrenar modelos iniciales
        if not optimizer.train_models():
            logger.warning("No se pudieron entrenar los modelos iniciales. Intentando simular carga para generar datos.")
            simulate_load()
            time.sleep(2)  # Esperar a que se completen algunas inserciones
            optimizer.train_models()
        
        # Ejecutar monitoreo y optimización
        duration_seconds = duration_hours * 3600 if duration_hours else None
        optimizer.monitor_and_optimize(interval=10, duration=duration_seconds)
        return True
    except KeyboardInterrupt:
        logger.info("Optimizador interrumpido por el usuario")
        return True
    except Exception as e:
        logger.error(f"Error ejecutando optimizador: {e}")
        return False
    finally:
        optimizer.close()

def simulate_load():
    """Simula carga para generar datos de entrenamiento."""
    logger.info("Simulando carga en la base de datos para generar métricas iniciales...")
    try:
        # Conexión a PostgreSQL
        conn = psycopg2.connect(
            dbname=os.environ.get("POSTGRES_DB", "postgres"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", "5432")
        )
        cur = conn.cursor()
        
        # Ejecutar script de simulación de carga
        with open('simulate_load.sql', 'r') as f:
            load_sql = f.read()
            cur.execute(load_sql)
        
        logger.info("Simulación de carga completada")
        return True
    except Exception as e:
        logger.error(f"Error simulando carga: {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def run_extended_test(duration_hours=72):
    """Ejecuta la prueba extendida."""
    logger.info(f"Iniciando prueba extendida (duración: {duration_hours} horas)")
    from extended_test import ExtendedTestRunner, TEST_CONFIG
    
    # Configurar la duración
    config = TEST_CONFIG.copy()
    config['duration_hours'] = duration_hours
    
    # Crear y ejecutar la prueba
    runner = ExtendedTestRunner(config)
    success = runner.run()
    
    if success:
        logger.info("Prueba extendida completada con éxito")
    else:
        logger.warning("Prueba extendida completada, pero no cumplió con los criterios de éxito")
    
    return success

def main():
    """Función principal."""
    # Crear parser de argumentos
    parser = argparse.ArgumentParser(description='Sistema de Optimización ML para PostgreSQL')
    parser.add_argument('--mode', '-m', type=str, choices=['setup', 'optimize', 'test', 'all'], 
                      default='optimize', help='Modo de ejecución')
    parser.add_argument('--duration', '-d', type=float, default=None,
                      help='Duración en horas (para modos optimize y test)')
    parser.add_argument('--simulate', '-s', action='store_true',
                      help='Simular carga para generar datos de entrenamiento')
    
    args = parser.parse_args()
    
    # Ejecutar según el modo seleccionado
    if args.mode == 'setup' or args.mode == 'all':
        if not setup_database():
            logger.error("Fallo en configuración de base de datos")
            return 1
    
    if args.simulate or args.mode == 'all':
        simulate_load()
    
    if args.mode == 'optimize' or args.mode == 'all':
        if not run_optimizer(args.duration):
            logger.error("Fallo en optimizador ML")
            return 1
    
    if args.mode == 'test' or args.mode == 'all':
        duration = args.duration or 72
        if not run_extended_test(duration):
            logger.error("Fallo en prueba extendida")
            return 1
    
    logger.info("Ejecución completada con éxito")
    return 0

if __name__ == "__main__":
    sys.exit(main())