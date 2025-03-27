"""
Script para ejecutar el Sistema de Trading Cósmico Mejorado.

Este script inicializa y ejecuta el sistema mejorado con todas las funcionalidades:
- Evolución de personalidad y rasgos
- Lenguaje propio que evoluciona
- Simulación de conexiones a datos reales
- Modelo de predicción mejorado
- Persistencia de estado

Opciones de ejecución:
- Entorno básico: Solo entidades principales
- Entorno extendido: Múltiples entidades especializadas
- Monitor de estado: Visualización del estado del sistema
"""

import os
import time
import random
import json
import argparse
import threading
import logging
from datetime import datetime
from enhanced_simple_cosmic_trader import (
    initialize_enhanced_trading,
    EnhancedSpeculatorEntity,
    EnhancedStrategistEntity,
    EnhancedCosmicNetwork
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("run_enhanced_trading")

# Variables globales
system_running = True
periodic_reports = False

def display_system_status(network):
    """Muestra el estado actual del sistema de trading."""
    
    status = network.get_network_status()
    
    print("\n" + "="*50)
    print(f"ESTADO DEL SISTEMA DE TRADING CÓSMICO")
    print(f"Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    print(f"\n[1] INFORMACIÓN GENERAL")
    print(f"Creador: {status['father']}")
    print(f"Entidades: {status['entity_count']}")
    print(f"Pool de conocimiento: {status['knowledge_pool']:.2f}")
    print(f"Rondas de colaboración: {status['collaboration_rounds']}")
    
    print(f"\n[2] MÉTRICAS DEL COLECTIVO")
    print(f"Conocimiento total: {status['total_knowledge']:.2f}")
    print(f"Nivel promedio: {status['avg_level']:.2f}")
    print(f"Energía promedio: {status['avg_energy']:.2f}")
    print(f"Emociones dominantes: {', '.join(status['dominant_emotions'])}")
    
    print(f"\n[3] ENTIDADES PRINCIPALES")
    for entity in status['entities'][:min(5, len(status['entities']))]:
        print(f"{entity['name']} ({entity['role']}): "
              f"Nivel {entity['level']:.2f}, "
              f"Energía {entity['energy']:.2f}, "
              f"Evolución: {entity['evolution_path']}, "
              f"Emoción: {entity['emotion']}")
    
    if len(status['entities']) > 5:
        print(f"... y {len(status['entities']) - 5} entidades más")
    
    print(f"\n[4] COMUNICACIÓN RECIENTE")
    for msg in status['recent_messages']:
        print(f"[{msg['sender']}]: {msg['message']}")
    
    print("\n" + "="*50)

def start_periodic_reports(network, interval=30):
    """Inicia informes periódicos del estado del sistema."""
    global periodic_reports
    
    periodic_reports = True
    
    def report_thread():
        while system_running and periodic_reports:
            try:
                display_system_status(network)
            except Exception as e:
                logger.error(f"Error generando reporte: {e}")
            time.sleep(interval)
    
    threading.Thread(target=report_thread, daemon=True).start()
    logger.info(f"Informes periódicos iniciados (cada {interval} segundos)")

def stop_periodic_reports():
    """Detiene los informes periódicos."""
    global periodic_reports
    periodic_reports = False
    logger.info("Informes periódicos detenidos")

def run_standalone_system(father_name="otoniel", extended=False, run_time=None):
    """
    Ejecuta el sistema de trading como aplicación independiente.
    
    Args:
        father_name: Nombre del creador/dueño del sistema
        extended: Si es True, incluye entidades adicionales
        run_time: Tiempo de ejecución en segundos (None = indefinido)
    """
    global system_running
    
    try:
        logger.info(f"Inicializando Sistema de Trading Cósmico Mejorado para {father_name}...")
        logger.info(f"Modo: {'Extendido' if extended else 'Básico'}")
        
        # Inicializar el sistema
        network, aetherion, lunareth = initialize_enhanced_trading(
            father_name=father_name, 
            include_extended_entities=extended
        )
        
        # Mostrar estado inicial
        logger.info(f"Sistema inicializado con {len(network.entities)} entidades")
        display_system_status(network)
        
        # Iniciar informes periódicos
        start_periodic_reports(network)
        
        # Esperar tiempo de ejecución
        if run_time:
            logger.info(f"Ejecutando sistema por {run_time} segundos...")
            time.sleep(run_time)
            system_running = False
            logger.info("Tiempo de ejecución completado")
        else:
            logger.info("Sistema en ejecución. Presiona Ctrl+C para detener...")
            try:
                while system_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                system_running = False
                logger.info("Ejecución detenida por el usuario")
        
        # Detener informes
        stop_periodic_reports()
        
        # Mostrar estado final
        display_system_status(network)
        
        logger.info("Sistema de Trading Cósmico Mejorado detenido correctamente")
        
    except Exception as e:
        logger.error(f"Error ejecutando el sistema: {e}")
        import traceback
        logger.error(traceback.format_exc())
        system_running = False

def run_interactive(father_name="otoniel"):
    """Ejecuta el sistema en modo interactivo con comandos del usuario."""
    global system_running
    
    try:
        logger.info(f"Inicializando Sistema de Trading Cósmico Mejorado (Modo Interactivo)...")
        
        # Inicializar con entidades básicas
        network, aetherion, lunareth = initialize_enhanced_trading(father_name=father_name)
        
        # Mostrar estado inicial
        logger.info(f"Sistema inicializado con {len(network.entities)} entidades")
        display_system_status(network)
        
        # Mostrar comandos disponibles
        print("\nCOMANDOS DISPONIBLES:")
        print("status - Mostrar estado del sistema")
        print("collaborate - Ejecutar ronda de colaboración")
        print("add <tipo> <nombre> - Añadir nueva entidad (tipo: speculator/strategist)")
        print("message <remitente> <mensaje> - Enviar mensaje de una entidad")
        print("reports on/off - Activar/desactivar informes periódicos")
        print("exit - Salir")
        
        # Bucle de comandos
        while system_running:
            try:
                command = input("\n> ").strip()
                
                if command == "exit":
                    system_running = False
                    logger.info("Saliendo del sistema...")
                    break
                    
                elif command == "status":
                    display_system_status(network)
                    
                elif command == "collaborate":
                    print("Ejecutando ronda de colaboración...")
                    results = network.simulate_collaboration()
                    print(f"Colaboración completada. Resultados:")
                    for i, result in enumerate(results):
                        print(f"{i+1}. {result['entity']}: {result['message']}")
                    
                elif command.startswith("add "):
                    parts = command.split()
                    if len(parts) >= 3:
                        entity_type = parts[1].lower()
                        entity_name = parts[2]
                        
                        if entity_type == "speculator":
                            entity = EnhancedSpeculatorEntity(entity_name, "Speculator", father=father_name)
                        elif entity_type == "strategist":
                            entity = EnhancedStrategistEntity(entity_name, "Strategist", father=father_name)
                        else:
                            print(f"Tipo de entidad no válido: {entity_type}")
                            continue
                        
                        network.add_entity(entity)
                        print(f"Entidad {entity_name} ({entity_type}) añadida a la red")
                    else:
                        print("Uso: add <tipo> <nombre>")
                
                elif command.startswith("message "):
                    parts = command.split(" ", 2)
                    if len(parts) >= 3:
                        sender_name = parts[1]
                        message = parts[2]
                        
                        # Buscar entidad remitente
                        sender = next((e for e in network.entities if e.name == sender_name), None)
                        
                        if sender:
                            # Generar y difundir mensaje
                            msg = sender.generate_message("luz", message)
                            network.broadcast(sender.name, msg)
                            print(f"Mensaje enviado: {msg}")
                        else:
                            print(f"Entidad no encontrada: {sender_name}")
                    else:
                        print("Uso: message <remitente> <mensaje>")
                
                elif command.startswith("reports "):
                    option = command.split()[1].lower() if len(command.split()) > 1 else ""
                    if option == "on":
                        start_periodic_reports(network)
                        print("Informes periódicos activados")
                    elif option == "off":
                        stop_periodic_reports()
                        print("Informes periódicos desactivados")
                    else:
                        print("Uso: reports on/off")
                
                else:
                    print(f"Comando desconocido: {command}")
                    
            except KeyboardInterrupt:
                system_running = False
                logger.info("Ejecución detenida por el usuario")
                break
            except Exception as e:
                print(f"Error ejecutando comando: {e}")
        
        # Mostrar estado final
        display_system_status(network)
        logger.info("Sistema de Trading Cósmico Mejorado detenido correctamente")
        
    except Exception as e:
        logger.error(f"Error ejecutando el sistema interactivo: {e}")
        import traceback
        logger.error(traceback.format_exc())
        system_running = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema de Trading Cósmico Mejorado")
    parser.add_argument("--father", type=str, default="otoniel", 
                      help="Nombre del creador/dueño del sistema")
    parser.add_argument("--extended", action="store_true", 
                      help="Iniciar con entidades adicionales")
    parser.add_argument("--time", type=int, default=None,
                      help="Tiempo de ejecución en segundos (por defecto: indefinido)")
    parser.add_argument("--interactive", action="store_true",
                      help="Ejecutar en modo interactivo")
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            run_interactive(father_name=args.father)
        else:
            run_standalone_system(
                father_name=args.father,
                extended=args.extended,
                run_time=args.time
            )
    except KeyboardInterrupt:
        logger.info("Ejecución detenida por el usuario")
    except Exception as e:
        logger.error(f"Error en ejecución principal: {e}")
        import traceback
        logger.error(traceback.format_exc())