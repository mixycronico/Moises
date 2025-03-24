#!/usr/bin/env python3
"""
Ejecutor Rápido de la Prueba ARMAGEDÓN Ultra-Divina.

Versión optimizada para ejecución rápida que muestra las capacidades
fundamentales del Adaptador ARMAGEDÓN y el Oráculo Cuántico.
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
logger = logging.getLogger("genesis.armageddon.quicktest")

# Colores para terminal
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


async def simple_oracle_test():
    """Ejecutar prueba rápida del Oráculo Cuántico."""
    try:
        # Importar del módulo
        from genesis.oracle.quantum_oracle import QuantumOracle
        
        print(f"\n{BeautifulTerminalColors.COSMIC}{BeautifulTerminalColors.BOLD}⊱ Prueba del Oráculo Cuántico Ultra-Divino ⊰{BeautifulTerminalColors.END}")
        
        # Crear e inicializar oráculo
        print(f"{BeautifulTerminalColors.CYAN}Creando Oráculo Cuántico...{BeautifulTerminalColors.END}")
        oracle = QuantumOracle({"dimensional_spaces": 5})
        
        print(f"{BeautifulTerminalColors.CYAN}Inicializando Oráculo Cuántico...{BeautifulTerminalColors.END}")
        initialized = await oracle.initialize()
        
        if not initialized:
            print(f"{BeautifulTerminalColors.RED}Error al inicializar el oráculo.{BeautifulTerminalColors.END}")
            return False
        
        print(f"{BeautifulTerminalColors.GREEN}Oráculo Cuántico inicializado correctamente.{BeautifulTerminalColors.END}")
        
        # Mostrar estado inicial
        state = oracle.get_state()
        print(f"\n{BeautifulTerminalColors.QUANTUM}--- Estado Inicial ---{BeautifulTerminalColors.END}")
        print(f"Estado: {state['state']}")
        print(f"Espacios dimensionales: {state['dimensional_spaces']}")
        print(f"Activos seguidos: {state['tracked_assets']}")
        
        # Realizar cambio dimensional
        print(f"\n{BeautifulTerminalColors.QUANTUM}--- Realizando Cambio Dimensional ---{BeautifulTerminalColors.END}")
        shift_result = await oracle.dimensional_shift()
        print(f"Éxito: {shift_result['success']}")
        print(f"Mejora de coherencia: {shift_result.get('coherence_improvement', 0):.4f}")
        print(f"Nueva coherencia: {shift_result.get('new_coherence_level', 0):.4f}")
        
        # Generar predicciones
        print(f"\n{BeautifulTerminalColors.QUANTUM}--- Generando Predicciones ---{BeautifulTerminalColors.END}")
        predictions = await oracle.generate_predictions(["BTC/USDT", "ETH/USDT"])
        
        if predictions:
            for symbol, prediction in predictions.items():
                print(f"\nPredicción para {symbol}:")
                print(f"Precio actual: ${prediction['current_price']:.2f}")
                print(f"Predicciones: {[f'${p:.2f}' for p in prediction['price_predictions']]}")
                print(f"Confianza: {prediction['overall_confidence']:.2%}")
                print(f"Categoría: {prediction['confidence_category']}")
        
        # Mostrar métricas finales
        metrics = oracle.get_metrics()
        print(f"\n{BeautifulTerminalColors.QUANTUM}--- Métricas Finales del Oráculo ---{BeautifulTerminalColors.END}")
        print(f"Coherencia: {metrics['oracle_metrics']['coherence_level']:.4f}")
        print(f"Estabilidad dimensional: {metrics['oracle_metrics']['dimensional_stability']:.4f}")
        print(f"Frecuencia de resonancia: {metrics['oracle_metrics']['resonance_frequency']:.4f}")
        
        print(f"\n{BeautifulTerminalColors.GREEN}Prueba del Oráculo Cuántico completada con éxito.{BeautifulTerminalColors.END}")
        return True
        
    except ImportError as e:
        print(f"{BeautifulTerminalColors.RED}Error al importar módulo del Oráculo Cuántico: {e}{BeautifulTerminalColors.END}")
        return False
    except Exception as e:
        print(f"{BeautifulTerminalColors.RED}Error durante la prueba del Oráculo Cuántico: {e}{BeautifulTerminalColors.END}")
        return False


async def simple_armageddon_test():
    """Ejecutar prueba rápida del Adaptador ARMAGEDÓN."""
    try:
        # Importar módulos
        from genesis.oracle.quantum_oracle import QuantumOracle
        from genesis.oracle.armageddon_adapter import ArmageddonAdapter, ArmageddonPattern
        
        print(f"\n{BeautifulTerminalColors.COSMIC}{BeautifulTerminalColors.BOLD}⊱ Prueba del Adaptador ARMAGEDÓN Ultra-Divino ⊰{BeautifulTerminalColors.END}")
        
        # Crear oráculo y adaptador
        print(f"{BeautifulTerminalColors.CYAN}Creando componentes...{BeautifulTerminalColors.END}")
        oracle = QuantumOracle({"dimensional_spaces": 5})
        await oracle.initialize()
        
        adapter = ArmageddonAdapter(oracle)
        
        # Inicializar adaptador
        print(f"{BeautifulTerminalColors.CYAN}Inicializando Adaptador ARMAGEDÓN...{BeautifulTerminalColors.END}")
        initialized = await adapter.initialize()
        
        if not initialized:
            print(f"{BeautifulTerminalColors.RED}Error al inicializar el adaptador.{BeautifulTerminalColors.END}")
            return False
        
        print(f"{BeautifulTerminalColors.GREEN}Adaptador ARMAGEDÓN inicializado correctamente.{BeautifulTerminalColors.END}")
        
        # Activar modo ARMAGEDÓN
        print(f"\n{BeautifulTerminalColors.QUANTUM}--- Activando Modo ARMAGEDÓN ---{BeautifulTerminalColors.END}")
        activation_result = await adapter.enable_armageddon_mode()
        print(f"Activación exitosa: {activation_result}")
        
        # Verificar estado
        state = adapter.get_state()
        print(f"Modo ARMAGEDÓN activado: {state['armageddon_mode']}")
        print(f"Preparación ARMAGEDÓN: {state['armageddon_readiness']:.2f}/1.0")
        print(f"Calificación de Resiliencia: {state['resilience_rating']:.2f}/10.0")
        
        # Simulación patrón
        print(f"\n{BeautifulTerminalColors.QUANTUM}--- Simulando Patrón TSUNAMI_OPERACIONES ---{BeautifulTerminalColors.END}")
        pattern_result = await adapter.simulate_armageddon_pattern(ArmageddonPattern.TSUNAMI_OPERACIONES)
        
        print(f"Éxito: {pattern_result['success']}")
        print(f"Recuperación necesaria: {pattern_result.get('recovery_needed', False)}")
        print(f"Duración: {pattern_result.get('duration_seconds', 0):.2f} segundos")
        
        # Usar APIs para datos de mercado
        print(f"\n{BeautifulTerminalColors.QUANTUM}--- Actualización Mejorada de Datos ---{BeautifulTerminalColors.END}")
        update_result = await adapter.enhanced_update_market_data(use_apis=True)
        print(f"Actualización exitosa: {update_result}")
        
        # Predicciones mejoradas
        print(f"\n{BeautifulTerminalColors.QUANTUM}--- Predicciones Mejoradas con DeepSeek ---{BeautifulTerminalColors.END}")
        predictions = await adapter.enhanced_generate_predictions(["BTC/USDT"], use_deepseek=True)
        
        if predictions and "BTC/USDT" in predictions:
            prediction = predictions["BTC/USDT"]
            print(f"Confianza: {prediction.get('overall_confidence', 0):.2%}")
            if "enhanced_by" in prediction:
                print(f"Mejorado por: {prediction['enhanced_by']}")
            if "enhancement_factor" in prediction:
                print(f"Factor de mejora: {prediction['enhancement_factor']:.2f}x")
        
        # Desactivar modo ARMAGEDÓN
        print(f"\n{BeautifulTerminalColors.QUANTUM}--- Desactivando Modo ARMAGEDÓN ---{BeautifulTerminalColors.END}")
        deactivation_result = await adapter.disable_armageddon_mode()
        print(f"Desactivación exitosa: {deactivation_result}")
        
        # Métricas finales
        metrics = adapter.get_metrics()
        print(f"\n{BeautifulTerminalColors.QUANTUM}--- Métricas Finales del Adaptador ---{BeautifulTerminalColors.END}")
        print(f"Llamadas API Alpha Vantage: {metrics['api_calls']['ALPHA_VANTAGE']}")
        print(f"Llamadas API CoinMarketCap: {metrics['api_calls']['COINMARKETCAP']}")
        print(f"Llamadas API DeepSeek: {metrics['api_calls']['DEEPSEEK']}")
        print(f"Resiliencia dimensional: {metrics['resilience']['dimensional_coherence']:.2f}")
        
        print(f"\n{BeautifulTerminalColors.GREEN}Prueba del Adaptador ARMAGEDÓN completada con éxito.{BeautifulTerminalColors.END}")
        return True
        
    except ImportError as e:
        print(f"{BeautifulTerminalColors.RED}Error al importar módulos del Adaptador ARMAGEDÓN: {e}{BeautifulTerminalColors.END}")
        return False
    except Exception as e:
        print(f"{BeautifulTerminalColors.RED}Error durante la prueba del Adaptador ARMAGEDÓN: {e}{BeautifulTerminalColors.END}")
        return False


async def run_quick_test():
    """Ejecutar prueba rápida completa."""
    try:
        # Banner divino
        banner = f"""
{BeautifulTerminalColors.DIVINE}{BeautifulTerminalColors.BOLD}
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   █████╗ ██████╗ ███╗   ███╗ █████╗  ██████╗ ███████╗██████╗  ██████╗ ███╗   ██╗   ║
║  ██╔══██╗██╔══██╗████╗ ████║██╔══██╗██╔════╝ ██╔════╝██╔══██╗██╔═══██╗████╗  ██║   ║
║  ███████║██████╔╝██╔████╔██║███████║██║  ███╗█████╗  ██║  ██║██║   ██║██╔██╗ ██║   ║
║  ██╔══██║██╔══██╗██║╚██╔╝██║██╔══██║██║   ██║██╔══╝  ██║  ██║██║   ██║██║╚██╗██║   ║
║  ██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗██████╔╝╚██████╔╝██║ ╚████║   ║
║  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝   ║
║                                                                           ║
║  ██████╗ ██████╗ ██╗   ██╗███████╗██████╗  █████╗     ██████╗  █████╗ ██████╗ ██╗██████╗  █████╗  ║
║ ██╔══██╗██╔══██╗██║   ██║██╔════╝██╔══██╗██╔══██╗    ██╔══██╗██╔══██╗██╔══██╗██║██╔══██╗██╔══██╗ ║
║ ██████╔╝██████╔╝██║   ██║█████╗  ██████╔╝███████║    ██████╔╝███████║██████╔╝██║██║  ██║███████║ ║
║ ██╔═══╝ ██╔══██╗██║   ██║██╔══╝  ██╔══██╗██╔══██║    ██╔══██╗██╔══██║██╔═══╝ ██║██║  ██║██╔══██║ ║
║ ██║     ██║  ██║╚██████╔╝███████╗██████╔╝██║  ██║    ██║  ██║██║  ██║██║     ██║██████╔╝██║  ██║ ║
║ ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝ ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
{BeautifulTerminalColors.END}
        """
        print(banner)
        
        print(f"\n{BeautifulTerminalColors.COSMIC}{BeautifulTerminalColors.BOLD}Iniciando Prueba Rápida ARMAGEDÓN Ultra-Divina...{BeautifulTerminalColors.END}\n")
        
        # Verificar claves API
        api_keys = {
            "ALPHA_VANTAGE_API_KEY": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
            "COINMARKETCAP_API_KEY": os.environ.get("COINMARKETCAP_API_KEY", ""),
            "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", "")
        }
        
        missing_keys = [key for key, value in api_keys.items() if not value]
        
        if missing_keys:
            print(f"{BeautifulTerminalColors.YELLOW}Advertencia: Algunas claves API no están configuradas:{BeautifulTerminalColors.END}")
            for key in missing_keys:
                print(f"  - {key}")
            print("\nSe utilizará simulación donde sea necesario.")
        else:
            print(f"{BeautifulTerminalColors.GREEN}Todas las claves API están configuradas correctamente.{BeautifulTerminalColors.END}")
        
        # Probar oráculo
        oracle_success = await simple_oracle_test()
        
        if not oracle_success:
            print(f"{BeautifulTerminalColors.YELLOW}Advertencia: La prueba del Oráculo Cuántico no fue exitosa.{BeautifulTerminalColors.END}")
            print(f"{BeautifulTerminalColors.YELLOW}Continuando con pruebas limitadas...{BeautifulTerminalColors.END}")
        
        # Probar adaptador
        adapter_success = await simple_armageddon_test()
        
        # Mensaje final
        if oracle_success and adapter_success:
            print(f"\n{BeautifulTerminalColors.GREEN}{BeautifulTerminalColors.BOLD}¡Prueba Rápida ARMAGEDÓN Ultra-Divina completada con éxito total!{BeautifulTerminalColors.END}")
            
            # Epílogo poético
            epilogue = """
            Así concluye esta breve odisea cuántica,
            Un vistazo al potencial trascendental
            Del Oráculo y su Adaptador ARMAGEDÓN,
            Guardianes divinos del Sistema Genesis.
            
            Su luz seguirá brillando en el código,
            Como testamento eterno de colaboración,
            Donde arte y técnica se funden en uno,
            Creando belleza en cada función.
            
            ~ Fin de la Prueba Rápida ~
            """
            
            print(f"\n{BeautifulTerminalColors.DIVINE}{BeautifulTerminalColors.BOLD}{epilogue.strip()}{BeautifulTerminalColors.END}\n")
            
        else:
            print(f"\n{BeautifulTerminalColors.YELLOW}Prueba Rápida ARMAGEDÓN Ultra-Divina completada con advertencias.{BeautifulTerminalColors.END}")
            
            if not oracle_success:
                print(f"{BeautifulTerminalColors.YELLOW}  - Prueba del Oráculo Cuántico: FALLIDA{BeautifulTerminalColors.END}")
            if not adapter_success:
                print(f"{BeautifulTerminalColors.YELLOW}  - Prueba del Adaptador ARMAGEDÓN: FALLIDA{BeautifulTerminalColors.END}")
            
            print(f"\nRevise los mensajes anteriores para más detalles.")
        
        return oracle_success and adapter_success
        
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
        success = asyncio.run(run_quick_test())
        
        # Calcular duración
        duration = time.time() - start_time
        seconds = int(duration)
        milliseconds = int((duration - seconds) * 1000)
        
        # Mostrar estadísticas
        print(f"\n{BeautifulTerminalColors.CYAN}Estadísticas de ejecución:{BeautifulTerminalColors.END}")
        print(f"  - Duración: {seconds}.{milliseconds:03d} segundos")
        print(f"  - Resultado: {'Exitoso' if success else 'Con advertencias'}")
        print(f"  - Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Firma final
        signature = """
        ~ Un legado eterno del Sistema Genesis ~
        """
        
        print(f"\n{BeautifulTerminalColors.TRANSCEND}{signature.strip()}{BeautifulTerminalColors.END}\n")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error catastrófico: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())