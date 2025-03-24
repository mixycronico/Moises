#!/usr/bin/env python3
"""
Ejecutor de la Prueba ARMAGEDÓN Ultra-Divina para Sistema Genesis.

Este script invoca la legendaria prueba ARMAGEDÓN, diseñada para evaluar
la resiliencia absoluta del Sistema Genesis mediante patrones de destrucción
extremos que ponen a prueba todos los componentes.

Uso:
  python ejecutar_armageddon.py [-i INTENSIDAD] [-p PATRÓN]

Donde:
  INTENSIDAD: NORMAL, DIVINO, ULTRA_DIVINO, COSMICO, TRANSCENDENTAL
  PATRÓN: Nombre de un patrón específico para prueba individual

Ejemplos:
  python ejecutar_armageddon.py -i DIVINO
  python ejecutar_armageddon.py -i ULTRA_DIVINO -p DEVASTADOR_TOTAL
"""

import sys
import asyncio
import argparse
import logging
import time
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("genesis.armageddon_executor")

# Definir colores para terminal
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

# Importar prueba ARMAGEDÓN
try:
    from genesis.cloud.armageddon_test import (
        ArmageddonPattern,
        ArmageddonIntensity,
        ArmageddonExecutor,
        run_armageddon_test
    )
    TEST_AVAILABLE = True
except ImportError:
    logger.error("No se pudo importar la prueba ARMAGEDÓN")
    TEST_AVAILABLE = False


def print_header():
    """Mostrar cabecera del script."""
    print(f"\n{Colors.DIVINE}{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.DIVINE}{Colors.BOLD}{'SISTEMA GENESIS MODO DIVINO: PRUEBA ARMAGEDÓN':^80}{Colors.END}")
    print(f"{Colors.DIVINE}{Colors.BOLD}{'=' * 80}{Colors.END}\n")
    
    print(f"{Colors.CYAN}Este script ejecuta la legendaria prueba ARMAGEDÓN para evaluar la resiliencia{Colors.END}")
    print(f"{Colors.CYAN}del Sistema Genesis ante condiciones catastróficas y patrones de destrucción.{Colors.END}")
    print()
    print(f"{Colors.YELLOW}ADVERTENCIA: Esta prueba induce deliberadamente condiciones extremas{Colors.END}")
    print(f"{Colors.YELLOW}en el sistema. No ejecutar en entornos de producción.{Colors.END}")
    print()
    print(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


async def main():
    """Función principal."""
    if not TEST_AVAILABLE:
        print(f"{Colors.RED}Error: La prueba ARMAGEDÓN no está disponible.{Colors.END}")
        print(f"{Colors.RED}Verifique que los módulos cloud estén correctamente instalados.{Colors.END}")
        return 1
    
    # Mostrar cabecera
    print_header()
    
    # Parsear argumentos
    parser = argparse.ArgumentParser(description="Ejecutor de la Prueba ARMAGEDÓN para Sistema Genesis")
    parser.add_argument(
        "--intensity", "-i",
        choices=["NORMAL", "DIVINO", "ULTRA_DIVINO", "COSMICO", "TRANSCENDENTAL"],
        default="DIVINO",
        help="Intensidad de la prueba"
    )
    parser.add_argument(
        "--pattern", "-p",
        help="Ejecutar solo un patrón específico"
    )
    parser.add_argument(
        "--wait", "-w",
        action="store_true",
        help="Esperar confirmación antes de empezar"
    )
    
    args = parser.parse_args()
    
    # Convertir intensidad
    intensity = getattr(ArmageddonIntensity, args.intensity, ArmageddonIntensity.DIVINO)
    
    # Mostrar configuración
    print(f"Intensidad seleccionada: {Colors.QUANTUM}{intensity.name}{Colors.END}")
    if args.pattern:
        print(f"Patrón seleccionado: {Colors.COSMIC}{args.pattern}{Colors.END}")
        # Verificar patrón
        try:
            pattern = getattr(ArmageddonPattern, args.pattern)
            print(f"Patrón reconocido: {Colors.GREEN}✓{Colors.END}")
        except AttributeError:
            print(f"{Colors.RED}Error: Patrón desconocido: {args.pattern}{Colors.END}")
            print(f"Patrones disponibles: {', '.join(p.name for p in ArmageddonPattern)}")
            return 1
    else:
        print(f"Ejecutando {Colors.BOLD}TODOS{Colors.END} los patrones de ataque")
    
    # Esperar confirmación si se solicita
    if args.wait:
        print(f"\n{Colors.YELLOW}Presione Enter para iniciar la prueba ARMAGEDÓN...{Colors.END}", end="")
        input()
    else:
        # Pequeña pausa para preparación
        print()
        for i in range(5, 0, -1):
            print(f"{Colors.YELLOW}Iniciando prueba en {i}...{Colors.END}", end="\r")
            await asyncio.sleep(1)
        print(f"{Colors.GREEN}¡Iniciando prueba ARMAGEDÓN!{Colors.END}          ")
    
    print()
    
    # Ejecutar prueba
    start_time = time.time()
    try:
        if args.pattern:
            # Ejecutar solo un patrón
            executor = ArmageddonExecutor(intensity)
            if not await executor.initialize():
                print(f"{Colors.RED}Error al inicializar ejecutor ARMAGEDÓN{Colors.END}")
                return 1
            
            pattern = getattr(ArmageddonPattern, args.pattern)
            await executor.run_test(pattern, intensity)
            
            # Mostrar resumen
            executor._print_summary()
            
        else:
            # Ejecutar todos los patrones
            results = await run_armageddon_test(intensity)
            if not results:
                print(f"{Colors.RED}Error al ejecutar prueba ARMAGEDÓN{Colors.END}")
                return 1
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Prueba interrumpida por el usuario{Colors.END}")
        return 1
    
    except Exception as e:
        print(f"\n{Colors.RED}Error durante la ejecución de la prueba: {e}{Colors.END}")
        return 1
    
    finally:
        # Mostrar tiempo total
        total_time = time.time() - start_time
        print(f"\nTiempo total de ejecución: {Colors.CYAN}{total_time:.2f} segundos{Colors.END}")
    
    return 0


if __name__ == "__main__":
    # Crear y ejecutar el bucle de eventos
    exit_code = asyncio.run(main())
    sys.exit(exit_code)