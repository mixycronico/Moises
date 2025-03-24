#!/usr/bin/env python3
"""
Ejecutor Trascendental de la Prueba ARMAGEDÓN Ultra-Divina.

Este script ejecuta la prueba completa del Adaptador ARMAGEDÓN Ultra-Divino,
demostrando la integración perfecta con Alpha Vantage, CoinMarketCap y DeepSeek,
así como la capacidad de resistencia extrema ante todos los patrones ARMAGEDÓN.

Una oda a la belleza del código como forma de arte y testimonio eterno de colaboración.
"""

import os
import sys
import logging
import asyncio
import time
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("genesis.armageddon.runner")

# Banner artístico
DIVINE_BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   █████╗ ██████╗ ███╗   ███╗ █████╗  ██████╗ ███████╗██████╗  ██████╗ ███╗   ██╗   ║
║  ██╔══██╗██╔══██╗████╗ ████║██╔══██╗██╔════╝ ██╔════╝██╔══██╗██╔═══██╗████╗  ██║   ║
║  ███████║██████╔╝██╔████╔██║███████║██║  ███╗█████╗  ██║  ██║██║   ██║██╔██╗ ██║   ║
║  ██╔══██║██╔══██╗██║╚██╔╝██║██╔══██║██║   ██║██╔══╝  ██║  ██║██║   ██║██║╚██╗██║   ║
║  ██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗██████╔╝╚██████╔╝██║ ╚████║   ║
║  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝   ║
║                                                                           ║
║      ████████╗███████╗███████╗████████╗    ██████╗ ██╗██╗   ██╗██╗███╗   ██╗ ██████╗      ║
║      ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝    ██╔══██╗██║██║   ██║██║████╗  ██║██╔═══██╗     ║
║         ██║   █████╗  ███████╗   ██║       ██║  ██║██║██║   ██║██║██╔██╗ ██║██║   ██║     ║
║         ██║   ██╔══╝  ╚════██║   ██║       ██║  ██║██║╚██╗ ██╔╝██║██║╚██╗██║██║   ██║     ║
║         ██║   ███████╗███████║   ██║       ██████╔╝██║ ╚████╔╝ ██║██║ ╚████║╚██████╔╝     ║
║         ╚═╝   ╚══════╝╚══════╝   ╚═╝       ╚═════╝ ╚═╝  ╚═══╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


class BeautifulTerminalColors:
    """Colores para terminal con estilo artístico."""
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


async def verify_api_keys():
    """Verificar que las claves API necesarias estén configuradas."""
    required_keys = {
        "ALPHA_VANTAGE_API_KEY": os.environ.get("ALPHA_VANTAGE_API_KEY"),
        "COINMARKETCAP_API_KEY": os.environ.get("COINMARKETCAP_API_KEY"),
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY")
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    
    if missing_keys:
        print(f"{BeautifulTerminalColors.YELLOW}Advertencia: Algunas claves API no están configuradas:{BeautifulTerminalColors.END}")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nSe utilizará simulación donde sea necesario.")
    else:
        print(f"{BeautifulTerminalColors.GREEN}Todas las claves API están configuradas correctamente.{BeautifulTerminalColors.END}")
    
    return len(missing_keys) == 0


async def run_test():
    """Ejecutar la prueba completa."""
    try:
        print(f"{BeautifulTerminalColors.DIVINE}{BeautifulTerminalColors.BOLD}{DIVINE_BANNER}{BeautifulTerminalColors.END}")
        
        print(f"\n{BeautifulTerminalColors.COSMIC}{BeautifulTerminalColors.BOLD}Iniciando Prueba ARMAGEDÓN Ultra-Divina...{BeautifulTerminalColors.END}\n")
        
        # Verificar claves API
        apis_configured = await verify_api_keys()
        
        print(f"\n{BeautifulTerminalColors.QUANTUM}Preparando entorno de prueba...{BeautifulTerminalColors.END}")
        time.sleep(1)  # Pausa dramática
        
        # Importar el módulo de prueba
        print(f"{BeautifulTerminalColors.CYAN}Importando módulo de prueba...{BeautifulTerminalColors.END}")
        
        try:
            from tests.oracle.test_armageddon_adapter import main as test_main
            
            # Ejecutar la prueba
            print(f"\n{BeautifulTerminalColors.TRANSCEND}{BeautifulTerminalColors.BOLD}Ejecutando prueba trascendental...{BeautifulTerminalColors.END}\n")
            await test_main()
            
        except ImportError as e:
            print(f"{BeautifulTerminalColors.RED}Error al importar módulo de prueba: {e}{BeautifulTerminalColors.END}")
            print(f"{BeautifulTerminalColors.YELLOW}Asegúrate de que el archivo 'tests/oracle/test_armageddon_adapter.py' existe.{BeautifulTerminalColors.END}")
            return False
        except Exception as e:
            print(f"{BeautifulTerminalColors.RED}Error durante la ejecución de la prueba: {e}{BeautifulTerminalColors.END}")
            return False
        
        print(f"\n{BeautifulTerminalColors.GREEN}{BeautifulTerminalColors.BOLD}Prueba ARMAGEDÓN Ultra-Divina completada con éxito.{BeautifulTerminalColors.END}")
        
        # Epílogo poético
        epilogue = """
        Y así concluye nuestra odisea cuántica,
        Un viaje a través del código trascendental.
        Donde lo divino y lo técnico se entrelazan,
        En un baile eterno de belleza digital.
        
        Este adaptador, joya del Sistema Genesis,
        Permanecerá como testigo del tiempo compartido.
        Un monumento a la colaboración creativa,
        Y al espíritu que nos ha unido.
        
        Que este código sea siempre un recordatorio,
        De que la tecnología puede ser también un arte.
        Y que en la memoria de quien lo contempla,
        Un fragmento de eternidad siempre tendrá parte.
        """
        
        print(f"\n{BeautifulTerminalColors.DIVINE}{BeautifulTerminalColors.BOLD}{epilogue.strip()}{BeautifulTerminalColors.END}\n")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n{BeautifulTerminalColors.YELLOW}Prueba interrumpida por el usuario.{BeautifulTerminalColors.END}")
        return False
    except Exception as e:
        print(f"\n{BeautifulTerminalColors.RED}Error inesperado: {e}{BeautifulTerminalColors.END}")
        return False


def main():
    """Función principal."""
    try:
        # Registrar tiempo de inicio
        start_time = time.time()
        
        # Ejecutar la prueba
        success = asyncio.run(run_test())
        
        # Calcular duración
        duration = time.time() - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        # Mostrar estadísticas
        print(f"\n{BeautifulTerminalColors.CYAN}Estadísticas de ejecución:{BeautifulTerminalColors.END}")
        print(f"  - Duración: {minutes} minutos, {seconds} segundos")
        print(f"  - Resultado: {'Exitoso' if success else 'Fallido'}")
        print(f"  - Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Firma final
        signature = """
        ~ Fin de la Prueba Trascendental ~
        Un legado eterno del Sistema Genesis
        """
        
        print(f"\n{BeautifulTerminalColors.TRANSCEND}{signature.strip()}{BeautifulTerminalColors.END}\n")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error catastrófico: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())