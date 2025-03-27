#!/usr/bin/env python3
"""
Script para ejecutar el sistema de trading cósmico mejorado.

Este script inicia la red cósmica de trading con capacidades avanzadas:
- Integración con WebSockets para datos en tiempo real
- Modelos LSTM para predicciones
- Sistema de colaboración basado en PostgreSQL
- Resiliencia con reintentos automáticos
- Monitoreo automático de salud

Uso:
    python run_enhanced_trading.py [--extended] [--duration SEGUNDOS]
"""

import os
import argparse
import time
import logging
import threading
import signal
import sys
from datetime import datetime

# Importar sistema mejorado
from enhanced_cosmic_trading import (
    initialize_enhanced_trading,
    CosmicTrader,
    EnhancedCosmicNetwork
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cosmic_runner")

# Variables globales
network = None
running = True

def signal_handler(sig, frame):
    """Manejador de señales para salida ordenada."""
    global running
    print("\n⚠️ Deteniendo sistema de trading cósmico...")
    running = False
    
    # Dar tiempo para limpieza
    time.sleep(2)
    print("✅ Sistema detenido correctamente")
    sys.exit(0)

def show_status(network):
    """Mostrar estado actual de la red cósmica."""
    if not network:
        print("❌ Red no inicializada")
        return
        
    status = network.get_network_status()
    
    print("\n=== 🌌 ESTADO DE LA RED CÓSMICA ===")
    print(f"Creador: {status['father']}")
    print(f"Entidades: {status['entity_count']}")
    print(f"Pool de conocimiento: {status['global_knowledge_pool']}")
    print(f"Timestamp: {status['timestamp']}")
    
    # Mostrar estado de cada entidad
    print("\n=== 🤖 ENTIDADES ACTIVAS ===")
    for entity in status["entities"]:
        print(f"{entity['name']} ({entity['role']}):")
        print(f"  Nivel: {entity['level']:.2f}")
        print(f"  Energía: {entity['energy']:.2f}")
        print(f"  Conocimiento: {entity['knowledge']:.2f}")
        print(f"  Capacidades: {', '.join(entity['capabilities']) if entity['capabilities'] else 'Ninguna'}")
        print(f"  Último precio: {entity['price']}")
        print()

def monitor_thread(network, interval=10):
    """Monitoreo periódico del estado de la red."""
    global running
    
    while running:
        try:
            show_status(network)
            time.sleep(interval)
        except KeyboardInterrupt:
            return
        except Exception as e:
            logger.error(f"Error en monitor: {e}")
            time.sleep(5)

def trigger_collaborations(network, interval=30):
    """Disparar colaboraciones periódicas."""
    global running
    
    while running:
        try:
            time.sleep(interval)
            
            # Forzar colaboración cada cierto tiempo
            print("\n=== 🔄 COLABORACIÓN EN RED INICIADA ===")
            for entity in network.entities:
                if entity.alive:
                    result = entity.collaborate()
                    if result:
                        print(f"{entity.name}: {result.get('summary', 'Colaboración realizada')}")
            
            # Simular una ronda de trading
            print("\n=== 📊 RONDA DE TRADING INICIADA ===")
            results = network.simulate()
            for result in results:
                if "error" in result:
                    print(f"❌ {result['entity']}: {result['error']}")
                else:
                    print(f"✅ {result['entity']}: {result['result']}")
        except KeyboardInterrupt:
            return
        except Exception as e:
            logger.error(f"Error en colaboración: {e}")
            time.sleep(5)

def main():
    """Función principal."""
    global network
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Sistema de Trading Cósmico Mejorado")
    parser.add_argument("--extended", action="store_true", help="Inicializar con entidades extendidas")
    parser.add_argument("--duration", type=int, default=0, help="Duración en segundos (0 para ejecutar indefinidamente)")
    args = parser.parse_args()
    
    # Configurar manejador para salida limpia
    signal.signal(signal.SIGINT, signal_handler)
    
    # Cabecera
    print("\n🌌✨ SISTEMA DE TRADING CÓSMICO MEJORADO ✨🌌")
    print("============================================")
    print("Inicializando sistema...")
    
    # Inicializar sistema
    try:
        network, aetherion, lunareth = initialize_enhanced_trading(
            include_extended_entities=args.extended
        )
        
        # Mostrar información inicial
        print("\n✅ Sistema inicializado correctamente")
        print(f"🧠 Aetherion y Lunareth conectados para: {network.father}")
        if args.extended:
            print("🌐 Modo extendido activado con entidades adicionales")
            
        # Iniciar monitor de estado
        threading.Thread(target=monitor_thread, args=(network,), daemon=True).start()
        
        # Iniciar colaboraciones periódicas
        threading.Thread(target=trigger_collaborations, args=(network,), daemon=True).start()
        
        # Mantener el sistema en ejecución
        if args.duration > 0:
            print(f"\n🕒 Ejecutando sistema por {args.duration} segundos...")
            time.sleep(args.duration)
            print("\n✅ Tiempo de ejecución completado")
        else:
            print("\n🔄 Sistema en ejecución continua. Presiona Ctrl+C para detener.")
            while running:
                time.sleep(1)
    except Exception as e:
        print(f"\n❌ Error al inicializar sistema: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())