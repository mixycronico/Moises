#!/usr/bin/env python3
"""
Script para ejecutar las pruebas específicas del monitor de componentes.

Este script ejecuta las pruebas del ComponentMonitor y genera un informe
detallado de los resultados, verificando que el sistema de aislamiento
y recuperación de componentes funcione correctamente.
"""

import asyncio
import logging
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/component_monitor_tests.log', mode='w')
    ]
)

logger = logging.getLogger("component_monitor_tests")

def generate_report(test_results: Dict[str, Any]) -> str:
    """
    Generar un informe detallado de los resultados de las pruebas.
    
    Args:
        test_results: Resultados completos de las pruebas
        
    Returns:
        Informe en formato markdown
    """
    report = "# Informe de Pruebas del Monitor de Componentes\n\n"
    report += f"Generado el: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Estadísticas generales
    report += "## Resumen\n\n"
    
    successful = test_results.get("success", False)
    passed = test_results.get("passed", 0)
    failed = test_results.get("failed", 0)
    skipped = test_results.get("skipped", 0)
    total = passed + failed + skipped
    
    report += f"- Estado general: {'✅ ÉXITO' if successful else '❌ FALLO'}\n"
    report += f"- Pruebas exitosas: {passed}/{total} ({passed/total*100:.1f}%)\n"
    report += f"- Pruebas fallidas: {failed}/{total} ({failed/total*100:.1f}%)\n"
    report += f"- Pruebas omitidas: {skipped}/{total} ({skipped/total*100:.1f}%)\n"
    report += f"- Tiempo total: {test_results.get('duration', 0):.2f} segundos\n\n"
    
    # Detalles de las pruebas
    report += "## Detalles de las Pruebas\n\n"
    
    for test_name, test_result in test_results.get("tests", {}).items():
        status = "✅ PASÓ" if test_result.get("success", False) else "❌ FALLÓ"
        duration = test_result.get("duration", 0)
        
        report += f"### {test_name}\n\n"
        report += f"- Estado: {status}\n"
        report += f"- Duración: {duration:.3f} segundos\n"
        
        if not test_result.get("success", False):
            error = test_result.get("error", "No hay información de error disponible")
            report += f"- Error: ```\n{error}\n```\n"
            
            if "traceback" in test_result:
                report += f"- Traceback: ```python\n{test_result['traceback']}\n```\n"
        
        report += "\n"
    
    # Conclusiones y recomendaciones
    report += "## Conclusiones y Recomendaciones\n\n"
    
    if successful:
        report += (
            "El Monitor de Componentes está funcionando correctamente, mostrando capacidad para:\n\n"
            "- Detectar componentes no saludables\n"
            "- Aislar componentes problemáticos\n"
            "- Manejar componentes bloqueados o que no responden\n"
            "- Recuperar componentes automáticamente\n"
            "- Notificar a componentes dependientes sobre cambios de estado\n\n"
            "El sistema muestra una buena resiliencia y capacidad para prevenir fallos en cascada.\n"
        )
    else:
        report += (
            "Hay problemas en el funcionamiento del Monitor de Componentes. Se recomienda:\n\n"
            "1. Revisar la lógica de detección de componentes no saludables\n"
            "2. Verificar el mecanismo de aislamiento de componentes\n"
            "3. Mejorar el sistema de notificación a componentes dependientes\n"
            "4. Verificar la recuperación automática de componentes aislados\n"
            "5. Revisar la implementación de timeouts en las verificaciones de salud\n\n"
            "Es necesario solucionar estos problemas para garantizar la prevención efectiva de fallos en cascada.\n"
        )
    
    return report

def run_pytest() -> Dict[str, Any]:
    """
    Ejecutar pruebas específicas del monitor de componentes con pytest.
    
    Returns:
        Resultados de las pruebas
    """
    logger.info("Ejecutando pruebas del monitor de componentes")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [
                "python", "-m", "pytest", 
                "tests/unit/core/test_component_monitor.py", 
                "-v", "--junitxml=logs/component_monitor_tests.xml"
            ],
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
        
        # Extraer información detallada de cada prueba
        test_results = {}
        current_test = None
        current_error = []
        current_traceback = []
        in_traceback = False
        
        for line in stdout.splitlines() + stderr.splitlines():
            if line.startswith("test_"):
                # Nueva prueba
                parts = line.split(" ")
                test_name = parts[0]
                current_test = test_name
                
                if "PASSED" in line:
                    test_results[current_test] = {
                        "success": True,
                        "duration": 0.0
                    }
                elif "FAILED" in line:
                    test_results[current_test] = {
                        "success": False,
                        "duration": 0.0,
                        "error": "",
                        "traceback": ""
                    }
                    in_traceback = False
                    current_error = []
                    current_traceback = []
            
            # Extracción de duración
            if current_test and "(" in line and ")" in line and "s" in line:
                try:
                    time_str = line.split("(")[1].split(")")[0]
                    if time_str.endswith("s"):
                        test_results[current_test]["duration"] = float(time_str[:-1])
                except (ValueError, IndexError):
                    pass
            
            # Extracción de error y traceback
            if current_test and not test_results.get(current_test, {}).get("success", True):
                if "E       " in line:
                    # Error message
                    current_error.append(line.replace("E       ", ""))
                elif "_ _ _ " in line or line.startswith("    "):
                    # Entramos en traceback
                    in_traceback = True
                
                if in_traceback:
                    current_traceback.append(line)
                
                # Guardar error y traceback
                if current_error:
                    test_results[current_test]["error"] = "\n".join(current_error)
                if current_traceback:
                    test_results[current_test]["traceback"] = "\n".join(current_traceback)
        
        # Formato del resultado final
        return {
            "success": success,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration": duration,
            "tests": test_results,
            "stdout": stdout,
            "stderr": stderr
        }
        
    except Exception as e:
        logger.error(f"Error ejecutando pytest: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "duration": time.time() - start_time
        }

def main():
    """Función principal del script."""
    # Crear directorio de logs si no existe
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("Iniciando ejecución de pruebas del monitor de componentes")
    
    # Ejecutar pruebas
    test_results = run_pytest()
    
    # Generar informe
    report = generate_report(test_results)
    
    # Guardar informe
    report_path = Path("docs/reporte_monitor_componentes.md")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Informe generado en {report_path}")
    
    # Imprimir resumen
    print("\n" + "-" * 80)
    print(f"Pruebas completadas: {'ÉXITO' if test_results.get('success', False) else 'FALLO'}")
    print(f"Pasadas: {test_results.get('passed', 0)}")
    print(f"Fallidas: {test_results.get('failed', 0)}")
    print(f"Omitidas: {test_results.get('skipped', 0)}")
    print(f"Tiempo total: {test_results.get('duration', 0):.2f} segundos")
    print(f"Informe completo: {report_path}")
    print("-" * 80 + "\n")
    
    # Código de salida
    return 0 if test_results.get("success", False) else 1

if __name__ == "__main__":
    sys.exit(main())