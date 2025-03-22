"""
Script para generar documentación sobre condiciones de carrera y deadlocks en el sistema Genesis.

Este script analiza y documenta los resultados de las pruebas de detección de
condiciones de carrera y deadlocks, facilitando la identificación de posibles problemas
y mejorando la comprensión de comportamientos inesperados.
"""

import asyncio
import os
import sys
import json
import logging
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("genesis.tools.race_docs")

# Intentar importar componentes necesarios
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.unit.core.test_core_race_conditions import (
        test_resource_contention_and_deadlocks,
        test_concurrent_event_collisions,
        ResourceContenderComponent
    )
    from tests.utils.timeout_helpers import (
        emit_with_timeout,
        check_component_status,
        run_test_with_timing,
        cleanup_engine
    )
    from genesis.core.engine_non_blocking import EngineNonBlocking
    from genesis.core.component import Component
except ImportError as e:
    logger.error(f"Error al importar componentes necesarios: {e}")
    logger.error("Asegúrate de ejecutar este script desde la raíz del proyecto")
    sys.exit(1)


class DocumentationGenerator:
    """
    Generador de documentación para condiciones de carrera y deadlocks.
    
    Esta clase se encarga de ejecutar pruebas específicas para detectar
    condiciones de carrera y deadlocks, y generar documentación
    detallada basada en los resultados.
    """
    
    def __init__(self, output_dir="docs/race_conditions"):
        """
        Inicializar el generador de documentación.
        
        Args:
            output_dir: Directorio donde se guardará la documentación generada
        """
        self.output_dir = output_dir
        self.results = {}
        self.engine = None
        
        # Crear directorio de salida si no existe
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    async def run_tests(self):
        """
        Ejecutar las pruebas de condiciones de carrera y deadlocks.
        
        Returns:
            Resultados combinados de las pruebas
        """
        logger.info("Iniciando ejecución de pruebas de condiciones de carrera")
        
        # Crear una instancia del motor no bloqueante
        self.engine = EngineNonBlocking()
        
        # Ejecutar prueba de detección de deadlocks
        logger.info("Ejecutando prueba de detección de deadlocks")
        try:
            deadlock_result = await run_test_with_timing(
                self.engine,
                "test_resource_contention_and_deadlocks",
                lambda engine: test_resource_contention_and_deadlocks(engine)
            )
            self.results["deadlocks"] = deadlock_result
            logger.info(f"Prueba de deadlocks completada en {deadlock_result.get('test_duration', 0):.2f}s")
        except Exception as e:
            logger.error(f"Error en prueba de deadlocks: {e}")
        
        # Crear una nueva instancia del motor para la siguiente prueba
        await cleanup_engine(self.engine)
        self.engine = EngineNonBlocking()
        
        # Ejecutar prueba de colisiones de eventos
        logger.info("Ejecutando prueba de colisiones de eventos concurrentes")
        try:
            collision_result = await run_test_with_timing(
                self.engine,
                "test_concurrent_event_collisions",
                lambda engine: test_concurrent_event_collisions(engine)
            )
            self.results["collisions"] = collision_result
            logger.info(f"Prueba de colisiones completada en {collision_result.get('test_duration', 0):.2f}s")
        except Exception as e:
            logger.error(f"Error en prueba de colisiones: {e}")
        
        # Limpiar recursos
        await cleanup_engine(self.engine)
        
        return self.results
    
    def generate_documentation(self):
        """
        Generar documentación basada en los resultados de las pruebas.
        
        Returns:
            Ruta al archivo de documentación generado
        """
        if not self.results:
            logger.error("No hay resultados para generar documentación")
            return None
        
        # Nombre del archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_file = os.path.join(self.output_dir, f"race_conditions_report_{timestamp}.md")
        
        # Crear el contenido del archivo de documentación
        with open(doc_file, "w") as f:
            f.write("# Informe de Análisis de Condiciones de Carrera y Deadlocks\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Resumen Ejecutivo\n\n")
            
            # Resumen de resultados de deadlocks
            if "deadlocks" in self.results:
                deadlock_result = self.results["deadlocks"]
                deadlock_summary = deadlock_result.get("summary", {})
                f.write("### Detección de Deadlocks\n\n")
                f.write(f"- **Total de deadlocks detectados**: {deadlock_summary.get('total_deadlocks', 0)}\n")
                f.write(f"- **Recursos en contención**: {len(deadlock_summary.get('contended_resources', []))}\n")
                f.write(f"- **Tasa de éxito en adquisición de recursos**: {deadlock_summary.get('resource_success_rate', 0):.2%}\n")
                f.write(f"- **Tasa de éxito en procesamiento**: {deadlock_summary.get('dependency_success_rate', 0):.2%}\n")
                f.write(f"- **Tiempo de ejecución**: {deadlock_result.get('test_duration', 0):.2f}s\n\n")
            
            # Resumen de resultados de colisiones
            if "collisions" in self.results:
                collision_result = self.results["collisions"]
                collision_summary = collision_result.get("summary", {})
                f.write("### Colisiones de Eventos Concurrentes\n\n")
                f.write(f"- **Total de eventos procesados**: {collision_summary.get('total_events', 0)}\n")
                f.write(f"- **Tasa de éxito**: {collision_summary.get('success_rate', 0):.2%}\n")
                f.write(f"- **Tasa de timeouts**: {collision_summary.get('timeout_rate', 0):.2%}\n")
                f.write(f"- **Tasa de errores**: {collision_summary.get('error_rate', 0):.2%}\n")
                f.write(f"- **Rendimiento promedio**: {collision_summary.get('avg_events_per_second', 0):.2f} eventos/s\n")
                f.write(f"- **Tiempo de ejecución**: {collision_result.get('test_duration', 0):.2f}s\n\n")
            
            # Detalles de deadlocks
            if "deadlocks" in self.results:
                f.write("## Análisis Detallado de Deadlocks\n\n")
                
                # Detalles de contención de recursos
                deadlock_result = self.results["deadlocks"]
                f.write("### Patrones de Contención de Recursos\n\n")
                
                contention_results = deadlock_result.get("contention_results", [])
                if contention_results:
                    f.write("| Ronda | Adq. Exitosas | Adq. Fallidas | Recursos en Contención |\n")
                    f.write("|-------|--------------|---------------|------------------------|\n")
                    
                    for i, round_result in enumerate(contention_results):
                        successful = round_result.get("successful_acquisitions", 0)
                        failed = round_result.get("failed_acquisitions", 0)
                        contended = ", ".join(round_result.get("contended_resources", []))
                        
                        f.write(f"| {i+1}    | {successful}            | {failed}           | {contended} |\n")
                
                # Detalles de deadlocks detectados
                f.write("\n### Ciclos de Deadlock Detectados\n\n")
                
                deadlock_results = deadlock_result.get("deadlock_results", [])
                detected_deadlocks = []
                
                for round_result in deadlock_results:
                    deadlocks = round_result.get("deadlock_details", [])
                    detected_deadlocks.extend(deadlocks)
                
                if detected_deadlocks:
                    for i, deadlock in enumerate(detected_deadlocks):
                        f.write(f"#### Deadlock #{i+1}\n\n")
                        f.write(f"- **Componente inicial**: {deadlock.get('starting_component', 'Desconocido')}\n")
                        f.write(f"- **Ciclo**: {' -> '.join(deadlock.get('cycle', []))}\n")
                        f.write(f"- **Recursos involucrados**: {', '.join(deadlock.get('resources', []))}\n\n")
                else:
                    f.write("No se detectaron ciclos de deadlock durante las pruebas.\n\n")
                
                # Estadísticas de componentes
                f.write("### Estadísticas de Componentes\n\n")
                
                component_stats = deadlock_result.get("component_stats", {})
                if component_stats:
                    f.write("| Componente | Operaciones Procesadas | Errores | Recursos Retenidos | Deadlock Detectado |\n")
                    f.write("|------------|------------------------|---------|-------------------|-----------------|\n")
                    
                    for comp_name, stats in component_stats.items():
                        processing_count = stats.get("processing_count", 0)
                        error_count = stats.get("error_count", 0)
                        held_resources = ", ".join(stats.get("held_resources", []))
                        deadlock_detected = "Sí" if stats.get("deadlock_detected", False) else "No"
                        
                        f.write(f"| {comp_name} | {processing_count} | {error_count} | {held_resources} | {deadlock_detected} |\n")
            
            # Detalles de colisiones
            if "collisions" in self.results:
                f.write("\n## Análisis Detallado de Colisiones de Eventos\n\n")
                
                collision_result = self.results["collisions"]
                round_results = collision_result.get("round_results", [])
                
                if round_results:
                    f.write("### Resultados por Ronda\n\n")
                    f.write("| Ronda | Eventos | Exitosos | Timeouts | Errores | Eventos/s |\n")
                    f.write("|-------|---------|----------|----------|---------|----------|\n")
                    
                    for i, round_result in enumerate(round_results):
                        total = round_result.get("total_events", 0)
                        successful = round_result.get("successful", 0)
                        timeouts = round_result.get("timeouts", 0)
                        errors = round_result.get("errors", 0)
                        events_per_sec = round_result.get("events_per_second", 0)
                        
                        f.write(f"| {i+1} | {total} | {successful} | {timeouts} | {errors} | {events_per_sec:.2f} |\n")
                
                # Estadísticas de componentes
                f.write("\n### Estadísticas de Componentes en Colisiones\n\n")
                
                component_stats = collision_result.get("component_stats", {})
                if component_stats:
                    f.write("| Componente | Operaciones Procesadas | Errores | Recursos Retenidos |\n")
                    f.write("|------------|------------------------|---------|-------------------|\n")
                    
                    for comp_name, stats in component_stats.items():
                        processing_count = stats.get("processing_count", 0)
                        error_count = stats.get("error_count", 0)
                        held_resources = ", ".join(stats.get("held_resources", []))
                        
                        f.write(f"| {comp_name} | {processing_count} | {error_count} | {held_resources} |\n")
            
            # Conclusiones y recomendaciones
            f.write("\n## Conclusiones y Recomendaciones\n\n")
            
            # Conclusiones basadas en resultados de deadlocks
            if "deadlocks" in self.results:
                deadlock_result = self.results["deadlocks"]
                deadlock_summary = deadlock_result.get("summary", {})
                
                if deadlock_summary.get("total_deadlocks", 0) > 0:
                    f.write("### Aspectos Críticos Identificados\n\n")
                    f.write("1. **Presencia de deadlocks**: Se han detectado ciclos de dependencia que pueden provocar bloqueos en el sistema.\n")
                    f.write("   - **Recomendación**: Implementar un mecanismo de detección y resolución de deadlocks en tiempo de ejecución.\n")
                    f.write("   - **Recomendación**: Revisar el orden de adquisición de recursos en los componentes para establecer una jerarquía consistente.\n\n")
                
                resource_success = deadlock_summary.get("resource_success_rate", 0)
                if resource_success < 0.8:
                    f.write("2. **Baja tasa de éxito en adquisición de recursos**: Muchas solicitudes de recursos están fallando.\n")
                    f.write("   - **Recomendación**: Aumentar los timeouts para operaciones críticas.\n")
                    f.write("   - **Recomendación**: Implementar reintentos con backoff exponencial para adquisición de recursos.\n")
                    f.write("   - **Recomendación**: Revisar la estrategia de compartición de recursos entre componentes.\n\n")
                
                dependency_success = deadlock_summary.get("dependency_success_rate", 0)
                if dependency_success < 0.8:
                    f.write("3. **Problemas en procesamiento con dependencias**: Alto índice de fallos en operaciones que dependen de otros componentes.\n")
                    f.write("   - **Recomendación**: Reducir las dependencias circulares entre componentes.\n")
                    f.write("   - **Recomendación**: Mejorar el manejo de errores en cascada.\n\n")
            
            # Conclusiones basadas en resultados de colisiones
            if "collisions" in self.results:
                collision_result = self.results["collisions"]
                collision_summary = collision_result.get("summary", {})
                
                success_rate = collision_summary.get("success_rate", 0)
                if success_rate < 0.8:
                    f.write("4. **Baja tasa de éxito en eventos concurrentes**: Muchos eventos están fallando cuando hay alta concurrencia.\n")
                    f.write("   - **Recomendación**: Revisar la implementación de eventos concurrentes para identificar puntos de bloqueo.\n")
                    f.write("   - **Recomendación**: Considerar aumentar el paralelismo en el motor de eventos.\n\n")
                
                timeout_rate = collision_summary.get("timeout_rate", 0)
                if timeout_rate > 0.2:
                    f.write("5. **Alta tasa de timeouts**: Muchas operaciones están alcanzando el timeout configurado.\n")
                    f.write("   - **Recomendación**: Ajustar los timeouts según la carga del sistema.\n")
                    f.write("   - **Recomendación**: Implementar un mecanismo de prioridad para eventos críticos.\n\n")
                
                events_per_second = collision_summary.get("avg_events_per_second", 0)
                if events_per_second < 100:
                    f.write("6. **Bajo rendimiento en procesamiento de eventos**: El sistema está procesando menos eventos por segundo de lo esperado.\n")
                    f.write("   - **Recomendación**: Optimizar el rendimiento del motor de eventos.\n")
                    f.write("   - **Recomendación**: Considerar implementar un sistema de back-pressure.\n\n")
            
            # Recomendaciones generales
            f.write("### Recomendaciones Generales\n\n")
            f.write("1. **Implementar un mecanismo de recuperación automática**\n")
            f.write("   - Detectar automáticamente situaciones de bloqueo y reiniciar componentes afectados\n")
            f.write("   - Establecer timeouts adaptativos basados en la carga del sistema\n\n")
            
            f.write("2. **Mejorar la monitorización**\n")
            f.write("   - Registrar métricas detalladas sobre adquisición de recursos y tiempos de procesamiento\n")
            f.write("   - Implementar alertas para situaciones anómalas como deadlocks o alta contención\n\n")
            
            f.write("3. **Refinar la arquitectura**\n")
            f.write("   - Reducir dependencias circulares entre componentes\n")
            f.write("   - Considerar un diseño más orientado a mensajes que a recursos compartidos\n")
            f.write("   - Implementar patrones como Circuit Breaker para prevenir cascadas de fallos\n\n")
            
            f.write("4. **Pruebas continuas**\n")
            f.write("   - Incorporar estas pruebas al pipeline de CI/CD\n")
            f.write("   - Ejecutar periódicamente para detectar regresiones\n")
            f.write("   - Considerar técnicas de fuzzing para encontrar condiciones de carrera ocultas\n\n")
        
        logger.info(f"Documentación generada en {doc_file}")
        return doc_file


async def main():
    """Función principal."""
    try:
        logger.info("Iniciando generación de documentación sobre condiciones de carrera")
        
        generator = DocumentationGenerator()
        await generator.run_tests()
        doc_file = generator.generate_documentation()
        
        if doc_file:
            logger.info(f"Documentación generada exitosamente en: {doc_file}")
        else:
            logger.error("No se pudo generar la documentación")
            return 1
        
    except Exception as e:
        logger.error(f"Error durante la generación de documentación: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Ejecutar la función principal dentro de un event loop
    loop = asyncio.get_event_loop()
    sys.exit(loop.run_until_complete(main()))