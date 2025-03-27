#!/usr/bin/env python3
"""
Versión LIGERA de ARMAGEDÓN CÓSMICO para pruebas rápidas.

Este script ejecuta una versión simplificada de la prueba ARMAGEDÓN CÓSMICO,
diseñada para ejecutarse en menos tiempo y con menor intensidad.
"""

import os
import sys
import time
import random
import logging
import argparse
from datetime import datetime, timedelta

# Importar el sistema de trading cósmico
from cosmic_trading import initialize_cosmic_trading

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ARMAGEDÓN_LIGERO")

# Colores para terminal
class Colors:
    """Colores para terminal con estilo divino."""
    HEADER = '\033[95m'        # Magenta claro
    BLUE = '\033[94m'          # Azul
    CYAN = '\033[96m'          # Cian
    GREEN = '\033[92m'         # Verde
    YELLOW = '\033[93m'        # Amarillo
    RED = '\033[91m'           # Rojo
    BOLD = '\033[1m'           # Negrita
    UNDERLINE = '\033[4m'      # Subrayado
    DIVINE = '\033[38;5;141m'  # Púrpura divino
    QUANTUM = '\033[38;5;39m'  # Azul cuántico
    COSMIC = '\033[38;5;208m'  # Naranja cósmico
    TRANSCEND = '\033[38;5;51m'# Aguamarina trascendental
    END = '\033[0m'            # Reset

C = Colors

def print_header():
    """Mostrar cabecera del script."""
    header = f"""
{C.DIVINE}{C.BOLD}╔══════════════════════════════════════════════════════════════════╗
║                {C.COSMIC}ARMAGEDÓN CÓSMICO LIGERO{C.DIVINE}                        ║
║      {C.TRANSCEND}Prueba rápida del Sistema de Trading Cósmico{C.DIVINE}             ║
╚══════════════════════════════════════════════════════════════════╝{C.END}

{C.YELLOW}Versión simplificada para pruebas rápidas de resiliencia y rendimiento.
{C.END}
"""
    print(header)

def run_quick_test(use_extended_entities=True, num_operations=50):
    """
    Ejecutar una prueba rápida del sistema de trading.
    
    Args:
        use_extended_entities: Si es True, incluye entidades adicionales
        num_operations: Número de operaciones a simular
    """
    print_header()
    
    print(f"{C.COSMIC}[INICIO]{C.END} Iniciando prueba ligera...")
    print(f"Modo: {'Extendido' if use_extended_entities else 'Básico'}")
    print(f"Operaciones: {num_operations}")
    
    # Inicializar sistema
    try:
        network, aetherion, lunareth = initialize_cosmic_trading(
            father_name="otoniel",
            include_extended_entities=use_extended_entities
        )
        
        print(f"\n{C.GREEN}Sistema inicializado correctamente{C.END}")
        
        # Mostrar información de entidades
        print(f"\n{C.CYAN}Entidades activas:{C.END}")
        for entity in network.entities:
            print(f"  - {C.BOLD}{entity.name}{C.END} ({entity.role}): Nivel {entity.level:.2f}, Energía {entity.energy*100:.1f}%")
        
        # Ejecutar simulación
        print(f"\n{C.YELLOW}{C.BOLD}Ejecutando simulación de trading...{C.END}")
        
        start_time = time.time()
        
        # Simular operaciones
        symbols = ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BNBUSD"]
        success_count = 0
        error_count = 0
        
        for i in range(num_operations):
            entity = random.choice(network.entities)
            symbol = random.choice(symbols)
            
            try:
                entity.fetch_market_data(symbol)  # Actualizar datos internos
                result = entity.trade()  # Ejecutar operación
                
                # Generar un pensamiento detallado basado en el rol
                thought = ""
                if entity.role == "Speculator":
                    thought = f"Analizando tendencias a corto plazo para {symbol}. {'Oportunidad prometedora' if random.random() > 0.5 else 'Señales mixtas'}."
                elif entity.role == "Strategist":
                    thought = f"Evaluando estrategia para {symbol} en marco temporal medio. {'Formación positiva' if random.random() > 0.5 else 'Consolidación necesaria'}."
                elif entity.role == "RiskManager":
                    thought = f"Calculando exposición al riesgo para {symbol}. Ratio riesgo/beneficio: {random.uniform(0.8, 1.5):.2f}."
                elif entity.role == "Arbitrageur":
                    thought = f"Buscando disparidades de precio para {symbol} entre mercados. {'Potencial arbitraje detectado' if random.random() > 0.7 else 'Sin oportunidades claras'}."
                elif entity.role == "PatternRecognizer":
                    patterns = ["Doble techo", "Hombro-cabeza-hombro", "Bandera alcista", "Triángulo descendente", "Taza con asa"]
                    thought = f"Identificando patrones técnicos en {symbol}. Posible formación de {random.choice(patterns)}."
                elif entity.role == "MacroAnalyst":
                    thought = f"Correlacionando {symbol} con eventos macroeconómicos. {'Impacto positivo esperado' if random.random() > 0.5 else 'Cautela recomendada'}."
                
                # Verificar resultado
                if result and "error" not in str(result).lower():
                    success_count += 1
                    print(f"{C.GREEN}✓{C.END} {entity.name} ({entity.role}) operó {symbol} exitosamente")
                    print(f"  {C.CYAN}Pensamiento:{C.END} {thought}")
                    # Simular aumento de conocimiento y nivel
                    entity.knowledge += random.uniform(0.01, 0.03)
                    entity.level += random.uniform(0.005, 0.015)
                    print(f"  {C.YELLOW}Evolución:{C.END} Conocimiento +{entity.knowledge:.2f}, Nivel {entity.level:.2f}")
                else:
                    error_count += 1
                    print(f"{C.RED}✗{C.END} {entity.name} tuvo un error operando {symbol}")
                    print(f"  {C.RED}Pensamiento:{C.END} Dificultad para analizar datos de {symbol}. Necesito más información.")
                
            except Exception as e:
                error_count += 1
                print(f"{C.RED}✗{C.END} Error en operación de {entity.name}: {e}")
            
            # Breve pausa entre operaciones
            time.sleep(0.05)
            
            # Simular anomalía ocasional
            if random.random() < 0.1:  # 10% de probabilidad
                anomaly_type = random.choice(["price_spike", "data_delay", "connection_issue"])
                print(f"\n{C.YELLOW}[ANOMALÍA] {anomaly_type} detectada. Evaluando respuesta...{C.END}")
                time.sleep(0.2)  # Breve pausa para simular impacto
            
            # Mostrar progreso
            if (i+1) % 10 == 0:
                print(f"\n{C.BLUE}Progreso: {i+1}/{num_operations} operaciones ({(i+1)/num_operations*100:.0f}%){C.END}")
                # Mostrar estado actual de la red
                print(f"{C.CYAN}Estado actual de la red:{C.END}")
                for entity in network.entities[:2]:  # Solo mostrar algunas entidades para no saturar la salida
                    print(f"  - {entity.name}: Energía {entity.energy*100:.1f}%, Nivel {entity.level:.2f}")
                print("")
        
        elapsed_time = time.time() - start_time
        
        # Resultados finales
        print(f"\n{C.DIVINE}{C.BOLD}╔══════════════════════════════════════════════════════════════════╗{C.END}")
        print(f"{C.DIVINE}{C.BOLD}║                  RESULTADOS DE LA PRUEBA                        ║{C.END}")
        print(f"{C.DIVINE}{C.BOLD}╚══════════════════════════════════════════════════════════════════╝{C.END}")
        
        print(f"\n{C.BOLD}Estadísticas:{C.END}")
        print(f"  - Operaciones exitosas: {success_count}/{num_operations} ({success_count/num_operations*100:.1f}%)")
        print(f"  - Errores: {error_count}/{num_operations} ({error_count/num_operations*100:.1f}%)")
        print(f"  - Tiempo total: {elapsed_time:.2f} segundos")
        print(f"  - Operaciones por segundo: {num_operations/elapsed_time:.2f}")
        
        print(f"\n{C.BOLD}Estado final de entidades:{C.END}")
        for entity in network.entities:
            energy_level = entity.energy * 100
            if energy_level > 60:
                energy_color = C.GREEN
            elif energy_level > 30:
                energy_color = C.YELLOW
            else:
                energy_color = C.RED
                
            print(f"  - {C.BOLD}{entity.name}{C.END} ({entity.role}):")
            print(f"    - Nivel: {entity.level:.2f}")
            print(f"    - Energía: {energy_color}{energy_level:.1f}%{C.END}")
            print(f"    - Capacidades: {len(entity.capabilities)}")
        
        # Evaluación final
        success_rate = success_count / num_operations
        rating = ""
        if success_rate >= 0.95:
            rating = f"{C.DIVINE}{C.BOLD}EXCELENTE{C.END}"
        elif success_rate >= 0.85:
            rating = f"{C.COSMIC}{C.BOLD}MUY BUENO{C.END}"
        elif success_rate >= 0.75:
            rating = f"{C.GREEN}BUENO{C.END}"
        elif success_rate >= 0.6:
            rating = f"{C.YELLOW}ACEPTABLE{C.END}"
        else:
            rating = f"{C.RED}MEJORABLE{C.END}"
            
        print(f"\n{C.BOLD}Evaluación final: {rating}{C.END}")
        print(f"\n{C.GREEN}Prueba ARMAGEDÓN LIGERA completada con éxito.{C.END}")
        
    except Exception as e:
        print(f"\n{C.RED}{C.BOLD}ERROR: {e}{C.END}")
        logger.error(f"Error durante la prueba: {e}", exc_info=True)
        return False
        
    return True

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Prueba ARMAGEDÓN CÓSMICO LIGERA')
    parser.add_argument('--mode', type=str, choices=['basic', 'extended'], default='extended',
                     help='Modo de prueba: basic (solo Aetherion/Lunareth), extended (todas las entidades)')
    parser.add_argument('--operations', type=int, default=50,
                     help='Número de operaciones a simular')
    parser.add_argument('--focus', type=str, default=None, choices=['speculator', 'strategist', 'risk', 'arbitrage', 'pattern', 'macro'],
                     help='Enfoque en un tipo específico de entidad')
    parser.add_argument('--detail', action='store_true',
                     help='Mostrar salida detallada para cada operación')
    
    args = parser.parse_args()
    
    # Si se especifica un tipo de entidad para enfoque, mostrar operaciones detalladas solo para ese tipo
    role_focus = None
    if args.focus:
        role_map = {
            'speculator': 'Speculator',
            'strategist': 'Strategist',
            'risk': 'RiskManager',
            'arbitrage': 'Arbitrageur',
            'pattern': 'PatternRecognizer',
            'macro': 'MacroAnalyst'
        }
        role_focus = role_map.get(args.focus)
        print(f"Enfocando análisis en el rol: {role_focus}")
    
    # Deshabilitar logging de "Esperando datos" para evitar ruido
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda record: "Esperando datos" not in record.getMessage())
    
    run_quick_test(
        use_extended_entities=(args.mode == 'extended'),
        num_operations=args.operations
    )

if __name__ == "__main__":
    main()