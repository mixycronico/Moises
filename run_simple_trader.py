"""
Ejecutor independiente del Sistema Cósmico de Trading Simplificado.

Este script permite ejecutar y probar el sistema de trading cósmico
simplificado de forma independiente, sin necesidad de Flask ni otros
componentes externos.
"""

import time
import logging
import threading
from datetime import datetime
from simple_cosmic_trader import (
    initialize_simple_trading,
    SimpleSpeculatorEntity,
    SimpleStrategistEntity,
    SimpleCosmicNetwork
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_simple_trader")

def main():
    """Ejecutar prueba del sistema de trading cósmico simplificado."""
    print("\n===== INICIANDO SISTEMA DE TRADING CÓSMICO SIMPLIFICADO =====")
    print("Inicializando entidades principales (Aetherion y Lunareth)...")
    
    try:
        # Inicializar sistema básico
        network, aetherion, lunareth = initialize_simple_trading()
        
        print("\nSistema inicializado correctamente.")
        print(f"Entidades activas: {len(network.entities)}")
        
        # Mostrar información inicial
        print("\nInformación inicial de entidades:")
        for entity in network.entities:
            print(f"  - {entity.name} ({entity.role}): Nivel {entity.level:.2f}, Energía {entity.energy:.2f}")
        
        # Informar ciclos de vida
        print("\nLos ciclos de vida de las entidades se están ejecutando en hilos separados.")
        print("Cada 30 segundos, las entidades realizarán operaciones de trading automáticas.")
        
        # Bucle principal de interacción
        running = True
        while running:
            try:
                print("\n" + "="*50)
                print("MENÚ DEL SISTEMA DE TRADING CÓSMICO SIMPLIFICADO")
                print("="*50)
                print("1. Ver estado de la red")
                print("2. Simular ronda de colaboración")
                print("3. Ejecutar operaciones de trading")
                print("4. Ver precio actual de mercado")
                print("5. Ver predicción de precio")
                print("6. Ver capacidades de entidades")
                print("7. Salir")
                
                option = input("\nSeleccione una opción (1-7): ")
                
                if option == '1':
                    # Ver estado de la red
                    status = network.get_network_status()
                    print("\nESTADO DE LA RED CÓSMICA")
                    print(f"Propietario: {status['father']}")
                    print(f"Entidades activas: {status['entity_count']}")
                    print(f"Conocimiento colectivo: {status['knowledge_pool']:.2f}")
                    print("\nEstado de entidades:")
                    
                    for entity_status in status['entities']:
                        print(f"  - {entity_status['name']} ({entity_status['role']})")
                        print(f"    Nivel: {entity_status['level']:.2f}")
                        print(f"    Energía: {entity_status['energy']:.2f}")
                        print(f"    Conocimiento: {entity_status['knowledge']:.2f}")
                        print(f"    Capacidades: {', '.join(entity_status['capabilities']) if entity_status['capabilities'] else 'Ninguna'}")
                
                elif option == '2':
                    # Simular colaboración
                    print("\nEjecutando ronda de colaboración...")
                    results = network.simulate_collaboration()
                    
                    if results:
                        print("\nResultados de colaboración:")
                        for result in results:
                            print(f"- {result['entity']}: {result['message'] if 'message' in result else 'Colaboración realizada'}")
                            print(f"  Conocimiento ganado: {result['knowledge_gained']:.2f}")
                    else:
                        print("No se produjo colaboración en este ciclo.")
                
                elif option == '3':
                    # Ejecutar operaciones de trading
                    print("\nEjecutando operaciones de trading...")
                    results = network.simulate()
                    
                    if results:
                        print("\nResultados de trading:")
                        for result in results:
                            if 'error' in result:
                                print(f"- {result['entity']}: ERROR - {result['error']}")
                            else:
                                print(f"- {result['entity']}: {result['result']}")
                    else:
                        print("No se produjeron operaciones en este ciclo.")
                
                elif option == '4':
                    # Ver precio actual
                    price = aetherion.fetch_market_data()
                    print(f"\nPrecio actual de BTCUSD: ${price:.2f}")
                
                elif option == '5':
                    # Ver predicción
                    price = aetherion.fetch_market_data()
                    predicted = aetherion.predict_price()
                    if predicted:
                        change = ((predicted - price) / price) * 100
                        direction = "ALCISTA 📈" if change > 0 else "BAJISTA 📉"
                        print(f"\nPrecio actual de BTCUSD: ${price:.2f}")
                        print(f"Precio predicho: ${predicted:.2f}")
                        print(f"Cambio esperado: {change:.2f}% ({direction})")
                    else:
                        print("\nNo hay suficientes datos para realizar una predicción.")
                
                elif option == '6':
                    # Ver capacidades
                    print("\nCAPACIDADES DE LAS ENTIDADES")
                    for entity in network.entities:
                        print(f"\n{entity.name} ({entity.role}):")
                        if entity.capabilities:
                            for cap in entity.capabilities:
                                print(f"  - {cap}")
                        else:
                            print("  - Sin capacidades desbloqueadas")
                
                elif option == '7':
                    # Salir
                    running = False
                    print("\nFinalizando Sistema de Trading Cósmico Simplificado...")
                
                else:
                    print("\nOpción no válida. Por favor, seleccione una opción del 1 al 7.")
                    
                if running:
                    # Esperar un momento antes de mostrar el menú nuevamente
                    time.sleep(1)
                
            except KeyboardInterrupt:
                running = False
                print("\nPrograma interrumpido por el usuario.")
            except Exception as e:
                print(f"\nError: {e}")
                time.sleep(2)
        
    except Exception as e:
        print(f"\nError al inicializar el sistema: {e}")
    
    print("\n===== SISTEMA FINALIZADO =====")

if __name__ == "__main__":
    main()