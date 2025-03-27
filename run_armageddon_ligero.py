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
import re
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

def run_quick_test(use_extended_entities=True, num_operations=50, role_focus=None, detailed_output=False):
    """
    Ejecutar una prueba rápida del sistema de trading.
    
    Args:
        use_extended_entities: Si es True, incluye entidades adicionales
        num_operations: Número de operaciones a simular
        role_focus: Si está definido, solo muestra operaciones para ese rol
        detailed_output: Si es True, muestra salida detallada para cada operación
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
            # Seleccionar entidad según el filtro por rol si está activo
            if role_focus:
                candidates = [e for e in network.entities if e.role == role_focus]
                entity = random.choice(candidates) if candidates else random.choice(network.entities)
            else:
                entity = random.choice(network.entities)
            
            symbol = random.choice(symbols)
            # Mostrar detalles según configuración
            if detailed_output:
                print(f"\n{C.DIVINE}{C.BOLD}[OPERACIÓN {i+1}/{num_operations}]{C.END}")
                print(f"{C.CYAN}Entidad:{C.END} {entity.name} ({entity.role})")
                print(f"{C.CYAN}Símbolo:{C.END} {symbol}")
                print(f"{C.CYAN}Nivel actual:{C.END} {entity.level:.2f}")
                print(f"{C.CYAN}Energía:{C.END} {entity.energy*100:.1f}%")
            
            try:
                entity.fetch_market_data(symbol)  # Actualizar datos internos
                result = entity.trade()  # Ejecutar operación
                
                # Generar un pensamiento detallado basado en el rol
                thought = ""
                if entity.role == "Speculator":
                    price_direction = "alcista" if random.random() > 0.5 else "bajista"
                    timeframe = random.choice(["corto plazo", "intradía", "swing trading"])
                    confidence = random.randint(65, 95)
                    volatility = random.choice(["alta", "moderada", "baja"])
                    thought = f"Análisis de momentum: {symbol} muestra tendencia {price_direction} a {timeframe} con {confidence}% de confianza. " \
                             f"Volatilidad {volatility}. {'Estrategia: entrar agresivamente con stop ajustado' if price_direction == 'alcista' else 'Estrategia: esperar retroceso para posición corta'}."

                elif entity.role == "Strategist":
                    market_phase = random.choice(["acumulación", "tendencia alcista", "distribución", "tendencia bajista"])
                    strength = random.randint(1, 10)
                    time_horizon = random.choice(["días", "semanas", "meses"])
                    resistance = round(random.uniform(100, 500), 2)
                    support = round(resistance * 0.8, 2)
                    thought = f"Análisis estratégico: {symbol} en fase de {market_phase} con fuerza {strength}/10. " \
                             f"Horizonte de inversión: {time_horizon}. Soporte clave en {support}, resistencia en {resistance}. " \
                             f"Recomendación: {'acumular en soportes' if random.random() > 0.5 else 'reducir exposición en resistencias'}."
                             
                elif entity.role == "RiskManager":
                    risk_level = random.choice(["bajo", "moderado", "elevado", "extremo"])
                    risk_ratio = round(random.uniform(0.8, 2.5), 2)
                    max_position = random.randint(5, 20)
                    corr_btc = round(random.uniform(0.3, 0.9), 2)
                    thought = f"Evaluación de riesgo: {symbol} presenta riesgo {risk_level}. Ratio riesgo/beneficio: {risk_ratio}. " \
                             f"Exposición máxima recomendada: {max_position}% del capital. Correlación con BTC: {corr_btc}. " \
                             f"{'⚠️ Recomiendo reducir tamaño de posición' if risk_level in ['elevado', 'extremo'] else 'Parámetros de riesgo aceptables para operación'}."
                             
                elif entity.role == "Arbitrageur":
                    exchanges = random.sample(["Binance", "Kraken", "Coinbase", "Bitfinex", "Huobi"], 3)
                    price_diff = round(random.uniform(0.01, 2.5), 2)
                    opportunity = price_diff > 0.8
                    profit_potential = round(price_diff * random.uniform(10, 100), 2)
                    thought = f"Análisis de arbitraje: {symbol} muestra diferencia de {price_diff}% entre {exchanges[0]} y {exchanges[1]}. " \
                             f"{'✓ Oportunidad de arbitraje detectada' if opportunity else '✗ Diferencia insuficiente para arbitraje rentable'}. " \
                             f"{'Beneficio potencial: $' + str(profit_potential) if opportunity else 'Continuando monitoreo de diferencias entre ' + exchanges[2]}."
                             
                elif entity.role == "PatternRecognizer":
                    patterns = {
                        "Doble techo": "señal bajista, objetivo de precio inferior",
                        "Hombro-cabeza-hombro": "patrón de reversión bajista, objetivo proyectado a la baja",
                        "Bandera alcista": "continuación alcista, objetivo medido por mástil",
                        "Triángulo descendente": "compresión de volatilidad con sesgo bajista",
                        "Taza con asa": "formación alcista de consolidación y ruptura"
                    }
                    pattern_name = random.choice(list(patterns.keys()))
                    completion = random.randint(60, 100)
                    time_to_trigger = random.randint(1, 48)
                    thought = f"Reconocimiento de patrones: Identificado {pattern_name} en {symbol} ({completion}% formado). " \
                             f"{patterns[pattern_name]}. Probable activación en {time_to_trigger} horas. " \
                             f"Recomendación: {'Preparar entrada tras confirmación' if 'alcista' in patterns[pattern_name] else 'Cautela, posible reversión'}."
                             
                elif entity.role == "MacroAnalyst":
                    events = ["tasas de interés FED", "datos de inflación", "tensiones geopolíticas", "regulación cripto", "adopción institucional"]
                    primary_event = random.choice(events)
                    impact = random.choice(["fuertemente positivo", "ligeramente positivo", "neutral", "ligeramente negativo", "fuertemente negativo"])
                    correlation = round(random.uniform(-0.9, 0.9), 2)
                    thought = f"Análisis macroeconómico: {symbol} muestra correlación {correlation} con {primary_event}. " \
                             f"Impacto esperado: {impact}. " \
                             f"{'➤ Potencial aumento de volatilidad en próximos días' if abs(correlation) > 0.7 else '➤ Comportamiento principalmente técnico a corto plazo'}. " \
                             f"Recomendación: {'Establecer coberturas' if impact in ['ligeramente negativo', 'fuertemente negativo'] else 'Mantener exposición bajo monitoreo continuo'}."
                
                # Verificar resultado
                if result and "error" not in str(result).lower():
                    success_count += 1
                    print(f"{C.GREEN}✓{C.END} {entity.name} ({entity.role}) operó {symbol} exitosamente")
                    
                    # Dividir el pensamiento en múltiples líneas para evitar truncamiento
                    print(f"  {C.CYAN}Pensamiento:{C.END}")
                    
                    # Primero intentar dividir por frases
                    try:
                        sentences = re.split(r'([.!?])\s+', thought)
                        current_sentence = ""
                        
                        # Reagrupar las frases correctamente (el split separa los signos de puntuación)
                        for i in range(0, len(sentences), 2):
                            if i + 1 < len(sentences):
                                current_sentence = sentences[i] + sentences[i+1]
                            else:
                                current_sentence = sentences[i]
                            
                            # Imprimir cada frase en una línea separada
                            if current_sentence.strip():
                                print(f"    {current_sentence.strip()}")
                    except Exception as e:
                        # En caso de error, simplemente dividir por longitud manualmente
                        max_line_length = 30  # Línea extremadamente corta para evitar truncamiento
                        for i in range(0, len(thought), max_line_length):
                            segment = thought[i:i+max_line_length]
                            if segment.strip():
                                print(f"    {segment.strip()}")
                                
                        # Añadir mensaje informativo al final para verificar que no hay truncamiento
                        print(f"    [Longitud total: {len(thought)} caracteres]")
                    
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
        num_operations=args.operations,
        role_focus=role_focus,
        detailed_output=args.detail
    )

if __name__ == "__main__":
    main()