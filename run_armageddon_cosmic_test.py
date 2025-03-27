#!/usr/bin/env python3
"""
Ejecutor para la Prueba ARMAGEDÓN CÓSMICO del Sistema Genesis de Trading.

Este script facilita la ejecución de la prueba ARMAGEDÓN CÓSMICO que evalúa
la resiliencia extrema del sistema de trading bajo condiciones catastróficas.

Uso:
  python run_armageddon_cosmic_test.py [-m {basic,extended}] [-d MINUTOS] [-c CICLOS] [-e EVENTOS]

Donde:
  -m, --mode      : Modo de prueba (basic: solo Aetherion/Lunareth, extended: todas las entidades)
  -d, --duration  : Duración máxima de la prueba en minutos
  -c, --cycles    : Número de ciclos de operaciones normales
  -e, --events    : Número de eventos catastróficos a simular

Ejemplos:
  python run_armageddon_cosmic_test.py
  python run_armageddon_cosmic_test.py --mode extended --duration 10 --cycles 5 --events 2
"""

import os
import sys
import argparse
from test_armageddon_cosmic_traders import ArmageddonTester

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

def print_header():
    """Mostrar cabecera del script."""
    header = f"""
{Colors.DIVINE}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════╗
║                   {Colors.COSMIC}ARMAGEDÓN CÓSMICO{Colors.DIVINE}                           ║
║       {Colors.TRANSCEND}La prueba definitiva para el Sistema Genesis de Trading{Colors.DIVINE}   ║
╚══════════════════════════════════════════════════════════════════╝{Colors.END}

{Colors.YELLOW}Este script ejecutará la prueba ARMAGEDÓN CÓSMICO para verificar 
la resiliencia absoluta del sistema de trading bajo condiciones extremas.
{Colors.END}

{Colors.RED}{Colors.BOLD}ADVERTENCIA: Esta prueba pondrá todo el sistema bajo estrés extremo.{Colors.END}
"""
    print(header)

def main():
    """Función principal."""
    print_header()
    
    parser = argparse.ArgumentParser(description='Ejecutor para la Prueba ARMAGEDÓN CÓSMICO')
    parser.add_argument('-m', '--mode', type=str, choices=['basic', 'extended'], default='extended',
                      help='Modo de prueba: basic (solo Aetherion/Lunareth) o extended (todas las entidades)')
    parser.add_argument('-d', '--duration', type=int, default=5,
                      help='Duración máxima de la prueba en minutos')
    parser.add_argument('-c', '--cycles', type=int, default=3,
                      help='Número de ciclos de operaciones normales')
    parser.add_argument('-e', '--events', type=int, default=1,
                      help='Número de eventos catastróficos a simular')
    parser.add_argument('-f', '--father', type=str, default="otoniel",
                      help='Nombre del creador/dueño del sistema')
    
    args = parser.parse_args()
    
    print(f"{Colors.CYAN}Configuración seleccionada:{Colors.END}")
    print(f"  - Modo: {args.mode}")
    print(f"  - Duración máxima: {args.duration} minutos")
    print(f"  - Ciclos: {args.cycles}")
    print(f"  - Eventos catastróficos: {args.events}")
    print(f"  - Creador: {args.father}")
    
    confirmation = input(f"\n{Colors.YELLOW}¿Iniciar la prueba ARMAGEDÓN CÓSMICO con esta configuración? (s/n): {Colors.END}")
    
    if confirmation.lower() not in ['s', 'si', 'y', 'yes']:
        print(f"\n{Colors.RED}Prueba cancelada por el usuario.{Colors.END}")
        return
    
    print(f"\n{Colors.GREEN}Iniciando prueba ARMAGEDÓN CÓSMICO...{Colors.END}\n")
    
    # Crear y ejecutar tester
    tester = ArmageddonTester(
        use_extended_entities=(args.mode == 'extended'),
        father_name=args.father
    )
    
    tester.run_armageddon_test(
        duration_minutes=args.duration,
        operation_cycles=args.cycles,
        catastrophic_events=args.events
    )

if __name__ == "__main__":
    main()