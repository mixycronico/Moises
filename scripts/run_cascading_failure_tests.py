#!/usr/bin/env python3
"""
Script para ejecutar y validar las pruebas de fallos en cascada.

Este script ejecuta las pruebas de fallos en cascada implementadas y genera
un informe detallado de los resultados, identificando potenciales problemas
y verificando que los mecanismos de prevención de fallos en cascada funcionen correctamente.
"""

import asyncio
import logging
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/cascading_failure_tests.log', mode='w')
    ]
)

logger = logging.getLogger("cascade_tests")

# Asegurar que el módulo genesis esté en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importamos clases necesarias
from genesis.core.engine_non_blocking import EngineNonBlocking
from genesis.core.component import Component
from genesis.core.component_monitor import ComponentMonitor
from tests.utils.timeout_helpers import (
    emit_with_timeout,
    check_component_status,
    safe_get_response,
    cleanup_engine
)

# Importar pruebas específicas
try:
    from tests.unit.core.test_core_cascading_failures_fixed import (
        DependentComponent,
        test_cascading_failure_basic,
        test_cascading_failure_partial,
        test_cascading_failure_recovery
    )
except ImportError as e:
    logger.error(f"Error al importar las pruebas: {e}")
    logger.error("Asegúrate de que el archivo de pruebas existe y es accesible")
    sys.exit(1)

async def run_test_with_reporting(test_name: str, test_func, *args, **kwargs) -> Dict[str, Any]:
    """
    Ejecutar una prueba con medición de tiempo y generación de informe.
    
    Args:
        test_name: Nombre de la prueba
        test_func: Función de prueba a ejecutar
        *args, **kwargs: Argumentos para la función de prueba
        
    Returns:
        Diccionario con resultados de la prueba
    """
    logger.info(f"Iniciando prueba: {test_name}")
    start_time = time.time()
    
    result = {
        "test_name": test_name,
        "start_time": start_time,
        "success": False,
        "error": None,
        "duration": 0
    }
    
    try:
        await test_func(*args, **kwargs)
        result["success"] = True
    except Exception as e:
        logger.error(f"Error en prueba {test_name}: {e}")
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    finally:
        duration = time.time() - start_time
        result["duration"] = duration
        logger.info(f"Prueba {test_name} completada en {duration:.3f} segundos. Éxito: {result['success']}")
    
    return result

async def run_tests_with_monitor() -> List[Dict[str, Any]]:
    """
    Ejecutar todas las pruebas de fallos en cascada con un monitor de componentes.
    
    Returns:
        Lista de resultados de las pruebas
    """
    logger.info("Iniciando suite de pruebas de fallos en cascada con monitor")
    
    results = []
    
    # Ejecutar prueba básica con monitor
    engine = EngineNonBlocking(test_mode=True)
    
    # Registrar monitor de componentes
    monitor = ComponentMonitor(
        name="cascade_monitor",
        check_interval=1.0,
        max_failures=2,
        recovery_interval=5.0
    )
    await engine.register_component(monitor)
    
    # Crear componentes para la prueba
    comp_a = DependentComponent("comp_a")
    comp_b = DependentComponent("comp_b", dependencies=["comp_a"])
    comp_c = DependentComponent("comp_c", dependencies=["comp_b"])
    
    # Registrar componentes
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    await engine.register_component(comp_c)
    
    # Verificar estado inicial
    logger.info("Verificando estado inicial de componentes")
    monitor_status = await emit_with_timeout(
        engine, "check_status", {}, "cascade_monitor", timeout=1.0
    )
    logger.info(f"Estado del monitor: {monitor_status}")
    
    # Obtener informe de salud inicial
    health_report = await emit_with_timeout(
        engine, "get_health_report", {}, "cascade_monitor", timeout=1.0
    )
    logger.info(f"Informe de salud inicial: {health_report}")
    
    # FASE 1: Provocar fallo en componente A
    logger.info("FASE 1: Provocando fallo en componente A")
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=1.0
    )
    
    # Notificar a dependientes
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_a", "status": False}, "comp_b", timeout=1.0
    )
    
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_b", "status": False}, "comp_c", timeout=1.0
    )
    
    # Esperar a que el monitor detecte el fallo
    logger.info("Esperando a que el monitor detecte el fallo...")
    await asyncio.sleep(2.0)
    
    # Obtener informe de salud después del fallo
    health_report_after_failure = await emit_with_timeout(
        engine, "get_health_report", {}, "cascade_monitor", timeout=1.0
    )
    logger.info(f"Informe de salud después del fallo: {health_report_after_failure}")
    
    # Verificar componentes aislados
    isolated_components = safe_get_response(health_report_after_failure, "isolated_components", [])
    logger.info(f"Componentes aislados: {isolated_components}")
    
    # FASE 2: Recuperar componente A y verificar propagación de recuperación
    logger.info("FASE 2: Recuperando componente A")
    await emit_with_timeout(
        engine, "set_health", {"healthy": True}, "comp_a", timeout=1.0
    )
    
    # Intentar recuperar componentes aislados manualmente
    for component_id in isolated_components:
        logger.info(f"Intentando recuperar componente {component_id}")
        await emit_with_timeout(
            engine, "recover_component", {"component_id": component_id}, "cascade_monitor", timeout=1.0
        )
    
    # Notificar cambio de estado a dependientes
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_a", "status": True}, "comp_b", timeout=1.0
    )
    
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_b", "status": True}, "comp_c", timeout=1.0
    )
    
    # Esperar a que se propague la recuperación
    logger.info("Esperando a que se propague la recuperación...")
    await asyncio.sleep(1.0)
    
    # Verificar estado final
    health_report_final = await emit_with_timeout(
        engine, "get_health_report", {}, "cascade_monitor", timeout=1.0
    )
    logger.info(f"Informe de salud final: {health_report_final}")
    
    # Limpiar recursos
    logger.info("Limpiando recursos")
    await cleanup_engine(engine)
    
    # Ahora ejecutar las pruebas específicas
    logger.info("Ejecutando pruebas específicas de fallos en cascada")
    
    # Test 1: Prueba básica
    result_basic = await run_test_with_reporting(
        "test_cascading_failure_basic",
        test_cascading_failure_basic
    )
    results.append(result_basic)
    
    # Test 2: Prueba parcial
    result_partial = await run_test_with_reporting(
        "test_cascading_failure_partial",
        test_cascading_failure_partial
    )
    results.append(result_partial)
    
    # Test 3: Prueba de recuperación
    result_recovery = await run_test_with_reporting(
        "test_cascading_failure_recovery",
        test_cascading_failure_recovery
    )
    results.append(result_recovery)
    
    return results

def generate_report(results: List[Dict[str, Any]]) -> str:
    """
    Generar informe de resultados en formato markdown.
    
    Args:
        results: Lista de resultados de pruebas
        
    Returns:
        Informe en formato markdown
    """
    report = "# Informe de Pruebas de Fallos en Cascada\n\n"
    report += f"Generado el: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Resumen general
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    
    report += "## Resumen\n\n"
    report += f"- Total de pruebas ejecutadas: {total_tests}\n"
    report += f"- Pruebas exitosas: {successful_tests}\n"
    report += f"- Pruebas fallidas: {total_tests - successful_tests}\n"
    report += f"- Porcentaje de éxito: {(successful_tests / total_tests) * 100:.1f}%\n\n"
    
    # Tabla de resultados
    report += "## Resultados Detallados\n\n"
    report += "| Prueba | Estado | Duración (s) |\n"
    report += "|--------|--------|-------------|\n"
    
    for result in results:
        status = "✅ Éxito" if result["success"] else "❌ Fallo"
        report += f"| {result['test_name']} | {status} | {result['duration']:.3f} |\n"
    
    report += "\n"
    
    # Detalles de fallos
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        report += "## Detalles de Fallos\n\n"
        
        for i, result in enumerate(failed_tests, 1):
            report += f"### {i}. {result['test_name']}\n\n"
            report += f"**Error:** {result['error']}\n\n"
            report += "**Traza de error:**\n\n"
            report += f"```python\n{result.get('traceback', 'No hay traza disponible')}\n```\n\n"
    
    # Conclusiones y recomendaciones
    report += "## Conclusiones\n\n"
    
    if successful_tests == total_tests:
        report += (
            "Todas las pruebas se han ejecutado correctamente. "
            "Los mecanismos de prevención de fallos en cascada funcionan como se espera.\n\n"
        )
    elif successful_tests > 0:
        report += (
            "Algunas pruebas han fallado. "
            "Los mecanismos de prevención de fallos en cascada deben revisarse en detalle.\n\n"
        )
    else:
        report += (
            "Todas las pruebas han fallado. "
            "Los mecanismos de prevención de fallos en cascada no están funcionando correctamente.\n\n"
        )
    
    report += (
        "### Recomendaciones\n\n"
        "1. Asegurar que todos los componentes implementen correctamente el método `handle_event`\n"
        "2. Verificar que los componentes respondan a eventos de tipo `check_status`\n"
        "3. Utilizar el `ComponentMonitor` en entornos de producción\n"
        "4. Implementar timeouts en todas las operaciones asíncronas\n"
        "5. Utilizar funciones seguras como `safe_get_response` para acceder a respuestas\n"
    )
    
    return report

async def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Ejecutar pruebas de fallos en cascada")
    parser.add_argument("--output", help="Archivo de salida para el informe", default="docs/reporte_cascading_failures_test.md")
    args = parser.parse_args()
    
    try:
        # Crear directorio de logs si no existe
        Path("logs").mkdir(exist_ok=True)
        
        # Ejecutar las pruebas
        logger.info("Ejecutando pruebas de fallos en cascada")
        results = await run_tests_with_monitor()
        
        # Generar informe
        logger.info(f"Generando informe en {args.output}")
        report = generate_report(results)
        
        # Guardar informe
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(report)
        
        logger.info(f"Informe guardado en {output_path}")
        
        # Verificar si todas las pruebas pasaron
        all_passed = all(r["success"] for r in results)
        exit_code = 0 if all_passed else 1
        
        logger.info(f"Todas las pruebas pasaron: {all_passed}")
        return exit_code
        
    except Exception as e:
        logger.error(f"Error al ejecutar las pruebas: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)