#!/usr/bin/env python3
"""
Script para ejecutar las pruebas de estrés del motor Genesis.

Este script ejecuta diferentes tipos de pruebas de estrés y genera
informes detallados de rendimiento y cuellos de botella.

Uso:
    python run_stress_tests.py [--all] [--gradual] [--isolation]
                             [--concurrency] [--priority] [--expansion]
                             [--report-file REPORT_FILE]
"""

import os
import sys
import argparse
import asyncio
import logging
import json
import time
from datetime import datetime
from pathlib import Path

# Configurar logging para el script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('stress_tests')

# Intentar importar las pruebas de estrés
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.unit.core.test_core_stress_tests import (
        test_gradual_load_increase,
        test_slow_component_isolation,
        test_high_concurrency,
        test_priority_under_pressure,
        test_dynamic_expansion_stress
    )
    from tests.unit.core.test_core_race_conditions import (
        test_resource_contention_and_deadlocks,
        test_concurrent_event_collisions
    )
    from tests.unit.core.test_core_peak_load_recovery import (
        test_peak_load_recovery,
        test_concurrent_load_distribution
    )
    from tests.utils.timeout_helpers import (
        emit_with_timeout,
        check_component_status,
        run_test_with_timing,
        cleanup_engine
    )
    # Importar los tipos de motor
    from genesis.core.engine_non_blocking import EngineNonBlocking
    from genesis.core.engine_priority_blocks import EnginePriorityBlocks
    from genesis.core.engine_dynamic_blocks import DynamicExpansionEngine
except ImportError as e:
    logger.error(f"Error al importar las pruebas: {e}")
    logger.error("Asegúrate de ejecutar este script desde la raíz del proyecto")
    sys.exit(1)


async def run_test(test_func, engine, name):
    """
    Ejecutar una prueba individual y capturar resultados.
    
    Args:
        test_func: Función de prueba a ejecutar
        engine: Instancia del motor a usar
        name: Nombre de la prueba para el informe
        
    Returns:
        Resultados de la prueba y tiempo de ejecución
    """
    logger.info(f"Ejecutando prueba: {name}")
    start_time = time.time()
    
    try:
        result = await test_func(engine)
        success = True
    except Exception as e:
        logger.error(f"Error en prueba {name}: {e}")
        result = {"error": str(e)}
        success = False
    
    elapsed = time.time() - start_time
    logger.info(f"Prueba {name} completada en {elapsed:.2f}s")
    
    return {
        "name": name,
        "success": success,
        "time": elapsed,
        "result": result
    }


async def run_all_tests(args):
    """
    Ejecutar todas las pruebas seleccionadas y generar informe.
    
    Args:
        args: Argumentos de línea de comandos
        
    Returns:
        Resultados completos de las pruebas
    """
    # Lista de pruebas a ejecutar
    test_queue = []
    
    # Prueba de carga gradual
    if args.all or args.gradual:
        engine = DynamicExpansionEngine()
        test_queue.append((test_gradual_load_increase, engine, "Aumento gradual de carga"))
    
    # Prueba de aislamiento de componentes lentos
    if args.all or args.isolation:
        engine = DynamicExpansionEngine()
        test_queue.append((test_slow_component_isolation, engine, "Aislamiento de componentes lentos"))
    
    # Prueba de alta concurrencia
    if args.all or args.concurrency:
        engine = DynamicExpansionEngine()
        test_queue.append((test_high_concurrency, engine, "Alta concurrencia"))
    
    # Prueba de prioridades bajo presión
    if args.all or args.priority:
        engine = EnginePriorityBlocks()
        test_queue.append((test_priority_under_pressure, engine, "Prioridades bajo presión"))
    
    # Prueba de expansión dinámica
    if args.all or args.expansion:
        engine = DynamicExpansionEngine()
        test_queue.append((test_dynamic_expansion_stress, engine, "Expansión dinámica acelerada"))
    
    # Prueba de detección de deadlocks
    if args.all or args.deadlock:
        engine = EngineNonBlocking()
        test_queue.append((test_resource_contention_and_deadlocks, engine, "Detección de deadlocks"))
    
    # Prueba de condiciones de carrera
    if args.all or args.race:
        engine = EngineNonBlocking()
        test_queue.append((test_concurrent_event_collisions, engine, "Condiciones de carrera"))
    
    # Prueba de recuperación de picos de carga
    if args.all or args.peak:
        engine = DynamicExpansionEngine()
        test_queue.append((test_peak_load_recovery, engine, "Recuperación de picos de carga"))
    
    # Prueba de distribución de carga concurrente
    if args.all or args.distribution:
        engine = DynamicExpansionEngine()
        test_queue.append((test_concurrent_load_distribution, engine, "Distribución de carga concurrente"))
    
    # Verificar que haya al menos una prueba seleccionada
    if not test_queue:
        logger.error("No se seleccionó ninguna prueba. Usa --all o una de las opciones específicas.")
        return None
    
    # Ejecutar las pruebas en secuencia
    results = []
    for test_func, engine, name in test_queue:
        result = await run_test(test_func, engine, name)
        results.append(result)
        
        # Dar tiempo entre pruebas para limpieza
        await asyncio.sleep(1)
    
    return results


def generate_report(results, report_file):
    """
    Generar un informe detallado de las pruebas.
    
    Args:
        results: Resultados de las pruebas
        report_file: Nombre del archivo para el informe
    """
    if not results:
        logger.error("No hay resultados para generar informe")
        return
    
    # Crear directorio de informes si no existe
    report_dir = os.path.dirname(report_file)
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Datos básicos del informe
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(results),
        "successful_tests": sum(1 for r in results if r["success"]),
        "total_time": sum(r["time"] for r in results),
        "results": results
    }
    
    # Guardar informe en formato JSON
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Informe guardado en {report_file}")
    
    # Mostrar resumen en consola
    logger.info("=== RESUMEN DE PRUEBAS DE ESTRÉS ===")
    logger.info(f"Pruebas ejecutadas: {report['total_tests']}")
    logger.info(f"Pruebas exitosas: {report['successful_tests']}")
    logger.info(f"Tiempo total: {report['total_time']:.2f}s")
    
    for result in results:
        status = "✓" if result["success"] else "✗"
        logger.info(f"{status} {result['name']}: {result['time']:.2f}s")
        
        # Mostrar hallazgos clave si están disponibles
        if result["success"] and isinstance(result["result"], dict):
            if "scaling_metrics" in result["result"]:
                metrics = result["result"]["scaling_metrics"]
                logger.info(f"  - Expansión máxima: {metrics.get('max_expansion_ratio', 'N/A')}x")
                
                if "blocks_throughput_correlation" in metrics:
                    corr = metrics["blocks_throughput_correlation"]
                    effectiveness = "Efectiva" if corr > 0.5 else "Limitada" if corr > 0 else "Inefectiva"
                    logger.info(f"  - Efectividad de escalado: {effectiveness} ({corr:.2f})")
            
            if "traffic_result" in result["result"]:
                traffic = result["result"]["traffic_result"]
                logger.info(f"  - Rendimiento: {traffic.get('events_per_second', 0):.2f} eventos/s")
                
                if "success_rate" in traffic:
                    logger.info(f"  - Tasa de éxito: {traffic['success_rate'] * 100:.1f}%")
            
            if "impact_factor" in result["result"]:
                logger.info(f"  - Factor de impacto de componentes lentos: {result['result']['impact_factor']:.2f}x")
            
            if "priority_ratios" in result["result"]:
                ratios = result["result"]["priority_ratios"]
                if "low_vs_high" in ratios:
                    logger.info(f"  - Ratio baja vs. alta prioridad: {ratios['low_vs_high']:.2f}x")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Ejecutar pruebas de estrés del motor Genesis")
    
    parser.add_argument("--all", action="store_true", help="Ejecutar todas las pruebas")
    parser.add_argument("--gradual", action="store_true", help="Ejecutar prueba de aumento gradual de carga")
    parser.add_argument("--isolation", action="store_true", help="Ejecutar prueba de aislamiento de componentes lentos")
    parser.add_argument("--concurrency", action="store_true", help="Ejecutar prueba de alta concurrencia")
    parser.add_argument("--priority", action="store_true", help="Ejecutar prueba de prioridades bajo presión")
    parser.add_argument("--expansion", action="store_true", help="Ejecutar prueba de expansión dinámica")
    parser.add_argument("--deadlock", action="store_true", help="Ejecutar prueba de detección de deadlocks")
    parser.add_argument("--race", action="store_true", help="Ejecutar prueba de condiciones de carrera")
    parser.add_argument("--peak", action="store_true", help="Ejecutar prueba de recuperación de picos de carga")
    parser.add_argument("--distribution", action="store_true", help="Ejecutar prueba de distribución de carga concurrente")
    
    report_path = os.path.join("logs", f"stress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    parser.add_argument("--report-file", type=str, default=report_path,
                        help="Ruta del archivo para el informe de resultados")
    
    args = parser.parse_args()
    
    # Ejecutar las pruebas
    try:
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(run_all_tests(args))
        
        if results:
            generate_report(results, args.report_file)
    except KeyboardInterrupt:
        logger.warning("Pruebas interrumpidas por el usuario")
    except Exception as e:
        logger.error(f"Error al ejecutar las pruebas: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())