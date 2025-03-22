#!/usr/bin/env python3
"""
Script para ejecutar todos los tests de prevención de fallos en cascada.

Este script ejecuta las pruebas específicas relacionadas con la prevención
de fallos en cascada en el sistema Genesis y genera un informe detallado.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/cascade_tests_all.log', mode='w')
    ]
)

logger = logging.getLogger("cascade_tests_all")

TEST_MODULES = [
    "tests/unit/core/test_component_monitor.py",
    "tests/unit/core/test_core_cascading_failures_fixed.py",
    "tests/unit/core/test_eventbus_deadlock_detection.py"
]

def run_tests(test_modules: List[str] = None) -> Dict[str, Any]:
    """
    Ejecutar todas las pruebas relacionadas con fallos en cascada.
    
    Args:
        test_modules: Lista opcional de módulos de prueba a ejecutar
        
    Returns:
        Resultados de las pruebas
    """
    if test_modules is None:
        test_modules = TEST_MODULES
        
    logger.info(f"Ejecutando {len(test_modules)} módulos de prueba")
    start_time = time.time()
    
    try:
        command = [
            "python", "-m", "pytest",
            *test_modules,
            "-v", "--junitxml=logs/cascade_tests_all.xml"
        ]
        
        logger.info(f"Ejecutando comando: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        # Extraer estadísticas básicas
        stdout = result.stdout
        stderr = result.stderr
        
        # Extraer número de pruebas pasadas, fallidas, etc.
        passed = 0
        failed = 0
        skipped = 0
        
        # Analizar la salida para obtener estadísticas
        for line in stdout.splitlines():
            if "passed" in line and "=" in line:
                parts = line.split("=")
                for part in parts:
                    part = part.strip()
                    if part.endswith(" passed"):
                        try:
                            passed = int(part.split(" ")[0])
                        except (ValueError, IndexError):
                            pass
                    elif part.endswith(" failed"):
                        try:
                            failed = int(part.split(" ")[0])
                        except (ValueError, IndexError):
                            pass
                    elif part.endswith(" skipped"):
                        try:
                            skipped = int(part.split(" ")[0])
                        except (ValueError, IndexError):
                            pass
        
        # Formato del resultado final
        return {
            "success": success,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration": duration,
            "stdout": stdout,
            "stderr": stderr
        }
        
    except Exception as e:
        logger.error(f"Error ejecutando pruebas: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "duration": time.time() - start_time
        }

def generate_report(results: Dict[str, Any]) -> str:
    """
    Generar un informe detallado de los resultados de las pruebas.
    
    Args:
        results: Resultados de las pruebas
        
    Returns:
        Informe en formato markdown
    """
    report = "# Informe Completo de Pruebas de Prevención de Fallos en Cascada\n\n"
    report += f"Generado el: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Estadísticas generales
    report += "## Resumen General\n\n"
    
    success = results.get("success", False)
    passed = results.get("passed", 0)
    failed = results.get("failed", 0)
    skipped = results.get("skipped", 0)
    total = passed + failed + skipped
    
    report += f"- Estado general: {'✅ ÉXITO' if success else '❌ FALLO'}\n"
    report += f"- Pruebas totales: {total}\n"
    report += f"- Pruebas exitosas: {passed} ({passed/total*100:.1f}% del total)\n"
    report += f"- Pruebas fallidas: {failed} ({failed/total*100:.1f}% del total)\n"
    report += f"- Pruebas omitidas: {skipped} ({skipped/total*100:.1f}% del total)\n"
    report += f"- Tiempo total: {results.get('duration', 0):.2f} segundos\n\n"
    
    # Módulos de prueba
    report += "## Módulos de Prueba Ejecutados\n\n"
    for module in TEST_MODULES:
        report += f"- `{module}`\n"
    report += "\n"
    
    # Detalles de ejecución
    report += "## Detalles de Ejecución\n\n"
    report += "### Salida Estándar\n\n"
    report += "```\n"
    report += results.get("stdout", "No hay salida estándar disponible")
    report += "\n```\n\n"
    
    if results.get("stderr"):
        report += "### Errores\n\n"
        report += "```\n"
        report += results.get("stderr")
        report += "\n```\n\n"
    
    # Conclusiones
    report += "## Conclusiones y Recomendaciones\n\n"
    
    if success:
        report += (
            "Las pruebas de prevención de fallos en cascada han sido exitosas, demostrando que:\n\n"
            "1. El `ComponentMonitor` funciona correctamente, detectando componentes problemáticos y aislándolos\n"
            "2. El mecanismo de notificación de cambios de estado a componentes dependientes funciona adecuadamente\n"
            "3. La recuperación automática de componentes funciona según lo esperado\n"
            "4. Los fallos se mantienen aislados y no se propagan a componentes no relacionados\n\n"
            "En general, el sistema demuestra buena resiliencia ante fallos y capacidad para prevenir fallos en cascada.\n"
        )
    else:
        report += (
            "Se han detectado problemas en las pruebas de prevención de fallos en cascada. Es necesario revisar:\n\n"
            "1. La implementación del `ComponentMonitor` y su capacidad para detectar y aislar componentes problemáticos\n"
            "2. El mecanismo de notificación a componentes dependientes\n"
            "3. Los timeouts y la gestión de componentes que no responden\n"
            "4. La lógica de recuperación automática\n\n"
            "Resolver estos problemas es prioritario para garantizar la estabilidad del sistema Genesis.\n"
        )
    
    # Pasos siguientes
    report += "## Pasos Siguientes\n\n"
    report += (
        "Para seguir mejorando la resiliencia del sistema ante fallos en cascada:\n\n"
        "1. **Mejoras en la monitorización**: Implementar métricas de estado para visualizar el estado del sistema en tiempo real\n"
        "2. **Circuito semi-abierto**: Implementar una fase intermedia en el patrón Circuit Breaker para permitir la recuperación gradual\n"
        "3. **Mejoras en las pruebas**: Añadir pruebas con generación dinámica de fallos aleatorios para simular escenarios más complejos\n"
        "4. **Documentación**: Mantener actualizada la documentación sobre la prevención de fallos en cascada\n"
    )
    
    return report

def main():
    """Función principal del script."""
    # Crear directorio de logs si no existe
    Path("logs").mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)
    
    logger.info("Iniciando ejecución de todas las pruebas de prevención de fallos en cascada")
    
    # Ejecutar pruebas
    results = run_tests()
    
    # Generar informe
    report = generate_report(results)
    
    # Guardar informe
    report_path = Path("docs/reporte_prevencion_fallos_cascada.md")
    
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Informe generado en {report_path}")
    
    # Imprimir resumen
    print("\n" + "-" * 80)
    print(f"Pruebas completadas: {'ÉXITO' if results.get('success', False) else 'FALLO'}")
    print(f"Pasadas: {results.get('passed', 0)}")
    print(f"Fallidas: {results.get('failed', 0)}")
    print(f"Omitidas: {results.get('skipped', 0)}")
    print(f"Tiempo total: {results.get('duration', 0):.2f} segundos")
    print(f"Informe completo: {report_path}")
    print("-" * 80 + "\n")
    
    # Ejecutar scripts individuales si se especifica
    if len(sys.argv) > 1 and sys.argv[1] == "--run-all-scripts":
        print("Ejecutando scripts individuales...")
        
        scripts = [
            "scripts/run_component_monitor_tests.py",
            "scripts/run_cascading_failure_tests.py"
        ]
        
        for script in scripts:
            print(f"\nEjecutando {script}...")
            subprocess.run(["python", script])
    
    # Código de salida
    return 0 if results.get("success", False) else 1

if __name__ == "__main__":
    sys.exit(main())