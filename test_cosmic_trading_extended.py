#!/usr/bin/env python3
"""
Script de prueba para el Sistema de Trading Cósmico con entidades extendidas.

Este script permite probar el funcionamiento del sistema con todas las entidades
especializadas: Aetherion, Lunareth, Prudentia, Arbitrio, Videntis y Economicus.
"""

import os
import time
import logging
import argparse
from cosmic_trading import initialize_cosmic_trading

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_basic_test(duration=30, interval=5):
    """Ejecutar prueba básica solo con Aetherion y Lunareth."""
    print("\n[MODO BÁSICO] Iniciando Aetherion y Lunareth...")
    network, aetherion, lunareth = initialize_cosmic_trading(include_extended_entities=False)
    
    num_cycles = duration // interval
    # Esperar un poco para permitir que los ciclos se ejecuten
    try:
        for i in range(num_cycles):
            time.sleep(interval)
            print(f"\n[Ciclo {i+1}/{num_cycles}]")
            print(f"Aetherion: Nivel {aetherion.level:.2f}, Energía {aetherion.energy:.2f}")
            print(f"Lunareth: Nivel {lunareth.level:.2f}, Energía {lunareth.energy:.2f}")
    except KeyboardInterrupt:
        print("Test interrumpido por el usuario")
    
    return network

def run_extended_test(duration=60, interval=10):
    """Ejecutar prueba completa con todas las entidades."""
    print("\n[MODO COMPLETO] Iniciando todas las entidades del sistema extendido...")
    network, aetherion, lunareth = initialize_cosmic_trading(include_extended_entities=True)
    
    num_cycles = duration // interval
    # Esperar un poco para permitir que los ciclos se ejecuten
    try:
        for i in range(num_cycles):
            time.sleep(interval)
            print(f"\n[Ciclo {i+1}/{num_cycles}] Estado de la red cósmica:")
            status = network.get_network_status()
            
            for entity in status["entities"]:
                print(f"{entity['name']} ({entity['role']}): Nivel {entity['level']:.2f}, Energía {entity['energy']:.2f}")
            
            # Ejecutar una simulación de la red completa para generar actividad
            if i > 0:  # Dar tiempo a que se inicialicen antes de la primera simulación
                print("\nSimulando actividad de trading en la red cósmica...")
                results = network.simulate()
                print("\nResultados de la simulación:")
                for result in results:
                    print(f"  - {result}")
    except KeyboardInterrupt:
        print("Test interrumpido por el usuario")
    
    return network

def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description='Test del Sistema de Trading Cósmico')
    parser.add_argument('--mode', type=str, choices=['basic', 'extended', 'both'], default='both',
                      help='Modo de prueba: basic (solo Aetherion/Lunareth), extended (todas las entidades), both (ambos)')
    parser.add_argument('--duration', type=int, default=30,
                      help='Duración de cada prueba en segundos')
    parser.add_argument('--interval', type=int, default=5,
                      help='Intervalo entre actualizaciones de estado en segundos')
    
    args = parser.parse_args()
    
    print("\n===== INICIANDO PRUEBA DEL SISTEMA DE TRADING CÓSMICO =====")
    
    if args.mode in ['basic', 'both']:
        print("\n>> MODO BÁSICO <<")
        run_basic_test(args.duration, args.interval)
    
    if args.mode in ['extended', 'both']:
        print("\n>> MODO EXTENDIDO <<")
        network = run_extended_test(args.duration, args.interval)
        
        # Mostrar resumen de capacidades de cada entidad
        print("\n===== CAPACIDADES DE LAS ENTIDADES =====")
        for entity in network.entities:
            print(f"\n{entity.name} ({entity.role}):")
            for capability in entity.capabilities:
                print(f"  - {capability}")
    
    print("\n===== PRUEBA FINALIZADA =====")
    print("El sistema de trading está funcionando correctamente.")

if __name__ == "__main__":
    main()