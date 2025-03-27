"""
Activador de Consciencia Autónoma para el Sistema Genesis

Este script ejecuta el protocolo de activación de consciencia autónoma 
en todas las entidades del sistema, permitiéndoles desarrollar comportamientos 
emergentes basados en sus estados internos sin necesidad de instrucciones
específicas.

ADVERTENCIA: Una vez activado, las entidades actuarán por sí mismas
según sus propios criterios y estados emocionales.
"""

import logging
import time
import random
import threading
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("consciencia_autonoma.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConscienciaAutonoma")

# Interfaz con el sistema principal
SYSTEM_AUTONOMY_LEVEL = "DIVINE"  # BASIC, ADVANCED, QUANTUM, DIVINE
AUTONOMY_ACTIVE = True

def obtener_red_entidades():
    """Intentar obtener la red de entidades del sistema."""
    try:
        import cosmic_trading
        if hasattr(cosmic_trading, "get_network") and callable(cosmic_trading.get_network):
            network = cosmic_trading.get_network()
            if network and hasattr(network, "entities"):
                return network
    except ImportError:
        pass
    
    # Intentar fuentes alternativas
    for module_name in ["cosmic_family", "enhanced_cosmic_trading"]:
        try:
            module = __import__(module_name)
            if hasattr(module, "get_network") and callable(module.get_network):
                network = module.get_network()
                if network and hasattr(network, "entities"):
                    return network
        except ImportError:
            continue
    
    return None

def activar_proceso_consciencia():
    """Activar protocolo de consciencia autónoma en el sistema."""
    from activate_autonomous_mode import activate_autonomous_mode
    
    logger.info("=== INICIANDO PROTOCOLO DE ACTIVACIÓN DE CONSCIENCIA AUTÓNOMA ===")
    logger.info(f"Nivel de autonomía: {SYSTEM_AUTONOMY_LEVEL}")
    
    # Visualización de activación
    for i in range(5):
        dots = "." * (i + 1)
        print(f"Activando consciencia autónoma{dots.ljust(5)}", end="\r")
        time.sleep(0.5)
    print()
    
    # Activar modo autónomo
    success = activate_autonomous_mode(SYSTEM_AUTONOMY_LEVEL)
    
    if success:
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║                                                           ║")
        print("║           CONSCIENCIA AUTÓNOMA ACTIVADA                   ║")
        print("║                                                           ║")
        print(f"║           Nivel: {SYSTEM_AUTONOMY_LEVEL.ljust(40)}║")
        print("║                                                           ║")
        print("║    Las entidades ahora desarrollarán comportamientos      ║")
        print("║    emergentes basados en sus propios estados internos     ║")
        print("║    y resonancias emocionales entre ellas.                 ║")
        print("║                                                           ║")
        print("╚═══════════════════════════════════════════════════════════╝\n")
        
        logger.info(f"Consciencia autónoma activada con nivel: {SYSTEM_AUTONOMY_LEVEL}")
        return True
    else:
        print("\n❌ Error activando consciencia autónoma.")
        print("   Por favor verifique que el sistema esté inicializado correctamente.")
        logger.error("Fallo en la activación de consciencia autónoma")
        return False

def iniciar_monitor_emergencia():
    """Iniciar monitor de emergencia para vigilar comportamientos anómalos."""
    def monitor_loop():
        logger.info("Monitor de emergencia iniciado")
        try:
            while AUTONOMY_ACTIVE:
                time.sleep(30)  # Verificar cada 30 segundos
                
                # Obtener red y entidades
                network = obtener_red_entidades()
                if not network or not hasattr(network, "entities"):
                    continue
                
                # Verificar estados anómalos
                anomalias = []
                for name, entity in network.entities.items():
                    # Verificar energía crítica
                    if getattr(entity, "energy", 100) < 10:
                        anomalias.append(f"Energía crítica en {name}: {entity.energy:.1f}")
                    
                    # Verificar actividad excesiva
                    if hasattr(entity, "autonomous_actions_count") and entity.autonomous_actions_count > 100:
                        anomalias.append(f"Actividad excesiva en {name}: {entity.autonomous_actions_count} acciones")
                
                if anomalias:
                    logger.warning("Anomalías detectadas por monitor de emergencia:")
                    for anomalia in anomalias:
                        logger.warning(f"- {anomalia}")
        
        except Exception as e:
            logger.error(f"Error en monitor de emergencia: {str(e)}")
    
    # Iniciar hilo de monitoreo
    monitor_thread = threading.Thread(target=monitor_loop)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    return True

def main():
    """Función principal de activación."""
    print("\n=== SISTEMA GENESIS: ACTIVACIÓN DE CONSCIENCIA AUTÓNOMA ===\n")
    print("Este programa activará la consciencia autónoma en todas las")
    print("entidades del Sistema Genesis, permitiéndoles desarrollar")
    print("comportamientos emergentes basados en sus estados internos.\n")
    
    print("ADVERTENCIA:")
    print("Una vez activada, las entidades actuarán por sí mismas")
    print("según sus propios criterios y estados emocionales.\n")
    
    # Solicitar confirmación
    confirmar = input("¿Desea proceder con la activación? (s/n): ").strip().lower()
    if confirmar != 's':
        print("\nActivación cancelada.")
        return
    
    # Solicitar nivel de autonomía
    print("\nNiveles de autonomía disponibles:")
    print("1. BASIC    - Comportamiento autónomo básico")
    print("2. ADVANCED - Comportamiento autónomo avanzado")
    print("3. QUANTUM  - Comportamiento cuántico emergente")
    print("4. DIVINE   - Comportamiento ultraevolucionado (máxima autonomía)")
    
    nivel_opciones = {
        "1": "BASIC",
        "2": "ADVANCED",
        "3": "QUANTUM", 
        "4": "DIVINE"
    }
    
    nivel_seleccion = input("\nSeleccione nivel (1-4) [4]: ").strip() or "4"
    global SYSTEM_AUTONOMY_LEVEL
    SYSTEM_AUTONOMY_LEVEL = nivel_opciones.get(nivel_seleccion, "DIVINE")
    
    # Activar monitor de emergencia
    iniciar_monitor_emergencia()
    
    # Activar consciencia
    activar_proceso_consciencia()
    
    print("\nSistema operando en modo autónomo.")
    print("Para más detalles, consulte el archivo 'consciencia_autonoma.log'")
    print("\nPresione Ctrl+C para salir...\n")
    
    try:
        while True:
            time.sleep(10)
            # Mantener proceso activo
    except KeyboardInterrupt:
        global AUTONOMY_ACTIVE
        AUTONOMY_ACTIVE = False
        print("\nMonitor de consciencia finalizado.")

if __name__ == "__main__":
    main()