"""
Script para generar un informe detallado de las pruebas de recuperación de carga.

Este script ejecuta las pruebas de recuperación de picos de carga y distribución
de carga concurrente, y genera un informe detallado con gráficos y métricas 
para evaluar el rendimiento del sistema en situaciones extremas.
"""

import asyncio
import os
import sys
import json
import logging
import random
import time
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("genesis.tools.load_recovery")

# Intentar importar componentes necesarios
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.unit.core.test_core_peak_load_recovery import (
        test_peak_load_recovery,
        test_concurrent_load_distribution,
        LoadGeneratorComponent,
        BurstMonitorComponent
    )
    from tests.utils.timeout_helpers import (
        emit_with_timeout,
        check_component_status,
        run_test_with_timing,
        cleanup_engine
    )
    from genesis.core.engine_dynamic_blocks import DynamicExpansionEngine
except ImportError as e:
    logger.error(f"Error al importar componentes necesarios: {e}")
    logger.error("Asegúrate de ejecutar este script desde la raíz del proyecto")
    sys.exit(1)


class LoadRecoveryReportGenerator:
    """
    Generador de informes de recuperación de carga para el sistema Genesis.
    
    Esta clase ejecuta pruebas de recuperación de picos de carga y distribución
    de carga concurrente, y genera un informe detallado con gráficos y métricas.
    """
    
    def __init__(self, output_dir="docs/load_recovery"):
        """
        Inicializar el generador de informes.
        
        Args:
            output_dir: Directorio donde se guardarán los informes generados
        """
        self.output_dir = output_dir
        self.results = {}
        
        # Crear directorio de salida si no existe
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # También crear subdirectorio para gráficos
        if not os.path.exists(os.path.join(output_dir, "graphs")):
            os.makedirs(os.path.join(output_dir, "graphs"))
    
    async def run_tests(self, iterations=1):
        """
        Ejecutar las pruebas de recuperación de carga.
        
        Args:
            iterations: Número de iteraciones para cada prueba
            
        Returns:
            Resultados combinados de las pruebas
        """
        logger.info(f"Iniciando ejecución de {iterations} iteraciones de pruebas de recuperación de carga")
        
        # Resultados por iteración
        peak_recovery_results = []
        distribution_results = []
        
        for i in range(iterations):
            logger.info(f"Ejecutando iteración {i+1}/{iterations}")
            
            # Test de recuperación de picos
            engine = DynamicExpansionEngine(
                min_concurrent_blocks=2,
                max_concurrent_blocks=8,
                expansion_threshold=0.7,
                scale_cooldown=2.0
            )
            
            try:
                logger.info("Ejecutando prueba de recuperación de picos de carga")
                result = await run_test_with_timing(
                    engine,
                    "test_peak_load_recovery",
                    test_peak_load_recovery
                )
                peak_recovery_results.append(result)
                logger.info(f"Prueba completada en {result.get('test_duration', 0):.2f}s")
            except Exception as e:
                logger.error(f"Error en prueba de recuperación de picos: {e}")
            
            # Limpiar engine
            await cleanup_engine(engine)
            
            # Test de distribución de carga
            engine = DynamicExpansionEngine(
                min_concurrent_blocks=2,
                max_concurrent_blocks=6,
                expansion_threshold=0.6,
                scale_cooldown=1.0
            )
            
            try:
                logger.info("Ejecutando prueba de distribución de carga concurrente")
                result = await run_test_with_timing(
                    engine,
                    "test_concurrent_load_distribution",
                    test_concurrent_load_distribution
                )
                distribution_results.append(result)
                logger.info(f"Prueba completada en {result.get('test_duration', 0):.2f}s")
            except Exception as e:
                logger.error(f"Error en prueba de distribución de carga: {e}")
            
            # Limpiar engine
            await cleanup_engine(engine)
        
        self.results = {
            "peak_recovery": peak_recovery_results,
            "distribution": distribution_results,
            "timestamp": datetime.now().isoformat(),
            "iterations": iterations
        }
        
        return self.results
    
    def generate_report(self):
        """
        Generar informe detallado con gráficos basado en los resultados.
        
        Returns:
            Ruta al archivo de informe generado
        """
        if not self.results:
            logger.error("No hay resultados para generar informe")
            return None
        
        # Nombre del archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"load_recovery_report_{timestamp}.md")
        
        # Generar gráficos
        graph_paths = self._generate_graphs(timestamp)
        
        # Crear el contenido del archivo de informe
        with open(report_file, "w") as f:
            f.write("# Informe de Pruebas de Recuperación de Carga\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Resumen Ejecutivo\n\n")
            
            # Resumen de prueba de recuperación de picos
            peak_results = self.results.get("peak_recovery", [])
            if peak_results:
                # Calcular promedio de ratios de recuperación
                recovery_ratios = [r.get("recovery_ratio", 0) for r in peak_results]
                avg_recovery = sum(recovery_ratios) / len(recovery_ratios) if recovery_ratios else 0
                
                f.write("### Recuperación de Picos de Carga\n\n")
                f.write(f"- **Promedio de ratio de recuperación**: {avg_recovery:.2f}x (1.0 = recuperación completa)\n")
                f.write(f"- **Mejor ratio de recuperación**: {max(recovery_ratios):.2f}x\n")
                f.write(f"- **Peor ratio de recuperación**: {min(recovery_ratios):.2f}x\n")
                f.write(f"- **Iteraciones exitosas**: {sum(1 for r in peak_results if r.get('recovery_ratio', 0) > 0.7)}/{len(peak_results)}\n\n")
                
                if "peak_recovery_graph" in graph_paths:
                    f.write(f"![Gráfico de Recuperación de Picos](graphs/peak_recovery_graph_{timestamp}.png)\n\n")
            
            # Resumen de prueba de distribución de carga
            dist_results = self.results.get("distribution", [])
            if dist_results:
                # Calcular coeficientes de variación promedio
                variation_coeffs = [r.get("distribution_stats", {}).get("overall", {}).get("variation_coefficient", 999) for r in dist_results]
                avg_variation = sum(variation_coeffs) / len(variation_coeffs) if variation_coeffs else 999
                
                # Calcular tasas de éxito promedio
                success_rates = [r.get("high_uniform_result", {}).get("success_count", 0) / 
                               r.get("high_uniform_result", {}).get("total_events", 1) for r in dist_results]
                avg_success = sum(success_rates) / len(success_rates) if success_rates else 0
                
                f.write("### Distribución de Carga Concurrente\n\n")
                f.write(f"- **Coeficiente de variación promedio**: {avg_variation:.2f} (menor = mejor distribución)\n")
                f.write(f"- **Tasa de éxito promedio en carga alta**: {avg_success:.2%}\n")
                f.write(f"- **Mejor distribución (menor variación)**: {min(variation_coeffs):.2f}\n")
                f.write(f"- **Iteraciones con buena distribución**: {sum(1 for v in variation_coeffs if v < 0.5)}/{len(variation_coeffs)}\n\n")
                
                if "distribution_graph" in graph_paths:
                    f.write(f"![Gráfico de Distribución de Carga](graphs/distribution_graph_{timestamp}.png)\n\n")
            
            # Análisis detallado de recuperación de picos
            if peak_results:
                f.write("## Análisis Detallado de Recuperación de Picos\n\n")
                
                for i, result in enumerate(peak_results):
                    f.write(f"### Iteración {i+1}\n\n")
                    
                    # Extraer datos detallados
                    baseline = result.get("baseline_metrics", {})
                    peak = result.get("peak_metrics", {})
                    recovery = result.get("recovery_metrics", {})
                    ratio = result.get("recovery_ratio", 0)
                    
                    # Baseline
                    f.write("#### Fase de Línea Base (carga normal)\n\n")
                    if "generators" in baseline:
                        f.write("| Generador | Eventos Procesados | Tiempo Promedio (ms) | Eventos/s |\n")
                        f.write("|-----------|-------------------|--------------------|----------|\n")
                        
                        for gen in baseline["generators"]:
                            name = gen.get("component", "Desconocido")
                            count = gen.get("processing_count", 0)
                            avg_time = gen.get("avg_process_time", 0) * 1000  # ms
                            events_per_sec = gen.get("events_per_second", 0)
                            
                            f.write(f"| {name} | {count} | {avg_time:.2f} | {events_per_sec:.2f} |\n")
                    
                    # Pico
                    f.write("\n#### Fase de Pico (carga extrema)\n\n")
                    if "generators" in peak:
                        f.write("| Generador | Eventos Procesados | Tiempo Promedio (ms) | Eventos/s |\n")
                        f.write("|-----------|-------------------|--------------------|----------|\n")
                        
                        for gen in peak["generators"]:
                            name = gen.get("component", "Desconocido")
                            count = gen.get("processing_count", 0)
                            avg_time = gen.get("avg_process_time", 0) * 1000  # ms
                            events_per_sec = gen.get("events_per_second", 0)
                            
                            f.write(f"| {name} | {count} | {avg_time:.2f} | {events_per_sec:.2f} |\n")
                    
                    # Recuperación
                    f.write("\n#### Fase de Recuperación\n\n")
                    if "generators" in recovery:
                        f.write("| Generador | Eventos Procesados | Tiempo Promedio (ms) | Eventos/s |\n")
                        f.write("|-----------|-------------------|--------------------|----------|\n")
                        
                        for gen in recovery["generators"]:
                            name = gen.get("component", "Desconocido")
                            count = gen.get("processing_count", 0)
                            avg_time = gen.get("avg_process_time", 0) * 1000  # ms
                            events_per_sec = gen.get("events_per_second", 0)
                            
                            f.write(f"| {name} | {count} | {avg_time:.2f} | {events_per_sec:.2f} |\n")
                    
                    # Métricas clave
                    f.write("\n#### Métricas Clave\n\n")
                    baseline_rate = baseline.get("success_rate", 0)
                    peak_rate = peak.get("success_rate", 0)
                    recovery_rate = recovery.get("success_rate", 0)
                    
                    f.write(f"- **Tasa de éxito en línea base**: {baseline_rate:.2%}\n")
                    f.write(f"- **Tasa de éxito en pico**: {peak_rate:.2%}\n")
                    f.write(f"- **Tasa de éxito en recuperación**: {recovery_rate:.2%}\n")
                    f.write(f"- **Ratio de recuperación**: {ratio:.2f}x\n\n")
            
            # Análisis detallado de distribución de carga
            if dist_results:
                f.write("## Análisis Detallado de Distribución de Carga\n\n")
                
                for i, result in enumerate(dist_results):
                    f.write(f"### Iteración {i+1}\n\n")
                    
                    # Extraer datos detallados
                    baseline = result.get("baseline_result", {})
                    high_uniform = result.get("high_uniform_result", {})
                    mixed_results = result.get("mixed_results", [])
                    
                    # Métricas clave
                    f.write("#### Métricas Clave\n\n")
                    events_per_second = high_uniform.get("events_per_second", 0)
                    success_rate = high_uniform.get("success_count", 0) / high_uniform.get("total_events", 1)
                    dist_stats = result.get("distribution_stats", {})
                    
                    f.write(f"- **Rendimiento en carga alta**: {events_per_second:.2f} eventos/segundo\n")
                    f.write(f"- **Tasa de éxito en carga alta**: {success_rate:.2%}\n")
                    
                    if "overall" in dist_stats:
                        overall = dist_stats["overall"]
                        f.write(f"- **Coeficiente de variación general**: {overall.get('variation_coefficient', 999):.2f}\n")
                        f.write(f"- **Promedio de eventos procesados por generador**: {overall.get('avg_count', 0):.1f}\n")
                    
                    # Distribución por tipo de carga
                    f.write("\n#### Distribución por Tipo de Carga\n\n")
                    f.write("| Tipo de Carga | Avg. Eventos | Desv. Estándar | Coef. Variación | Tiempo Promedio (ms) |\n")
                    f.write("|--------------|--------------|----------------|-----------------|---------------------|\n")
                    
                    for load_type, stats in dist_stats.items():
                        if load_type != "overall":
                            avg_count = stats.get("avg_count", 0)
                            std_count = stats.get("std_count", 0)
                            variation = stats.get("variation_coefficient", 0)
                            avg_time = stats.get("avg_process_time", 0) * 1000  # ms
                            
                            f.write(f"| {load_type} | {avg_count:.1f} | {std_count:.1f} | {variation:.2f} | {avg_time:.2f} |\n")
                    
                    # Rendimiento en diferentes patrones de carga
                    if mixed_results:
                        f.write("\n#### Rendimiento en Carga Mixta\n\n")
                        f.write("| Ronda | Total Eventos | Exitosos | Tasa de Éxito | Eventos/s |\n")
                        f.write("|-------|--------------|----------|--------------|----------|\n")
                        
                        for j, mix_result in enumerate(mixed_results):
                            total = mix_result.get("total_events", 0)
                            success = mix_result.get("success_count", 0)
                            rate = success / total if total > 0 else 0
                            eps = mix_result.get("events_per_second", 0)
                            
                            f.write(f"| {j+1} | {total} | {success} | {rate:.2%} | {eps:.2f} |\n")
            
            # Conclusiones
            f.write("\n## Conclusiones y Recomendaciones\n\n")
            
            # Valoración general de recuperación
            if peak_results:
                avg_recovery = sum(r.get("recovery_ratio", 0) for r in peak_results) / len(peak_results) if peak_results else 0
                
                if avg_recovery > 0.9:
                    f.write("### Recuperación de Picos\n\n")
                    f.write("**Valoración: EXCELENTE**\n\n")
                    f.write("El sistema muestra una recuperación casi completa después de picos de carga extremos. Esto indica que la expansión dinámica y los mecanismos de aislamiento están funcionando de manera óptima.\n\n")
                elif avg_recovery > 0.7:
                    f.write("### Recuperación de Picos\n\n")
                    f.write("**Valoración: BUENA**\n\n")
                    f.write("El sistema se recupera satisfactoriamente después de picos de carga, aunque hay margen de mejora. Los mecanismos de expansión dinámica funcionan pero podrían optimizarse para lograr una recuperación más completa.\n\n")
                elif avg_recovery > 0.4:
                    f.write("### Recuperación de Picos\n\n")
                    f.write("**Valoración: ACEPTABLE**\n\n")
                    f.write("El sistema se recupera parcialmente después de picos de carga, pero necesita mejoras significativas. Se recomienda revisar los algoritmos de recuperación y expansión dinámica para mejorar la resiliencia del sistema.\n\n")
                else:
                    f.write("### Recuperación de Picos\n\n")
                    f.write("**Valoración: INSUFICIENTE**\n\n")
                    f.write("El sistema muestra una recuperación deficiente después de picos de carga. Es necesario rediseñar los mecanismos de recuperación y expansión dinámica para mejorar la resiliencia.\n\n")
            
            # Valoración general de distribución
            if dist_results:
                avg_variation = sum(r.get("distribution_stats", {}).get("overall", {}).get("variation_coefficient", 999) for r in dist_results) / len(dist_results) if dist_results else 999
                
                if avg_variation < 0.2:
                    f.write("### Distribución de Carga\n\n")
                    f.write("**Valoración: EXCELENTE**\n\n")
                    f.write("El sistema distribuye la carga de manera muy uniforme entre los componentes. Esto indica un excelente balanceo de carga y un uso eficiente de los recursos disponibles.\n\n")
                elif avg_variation < 0.4:
                    f.write("### Distribución de Carga\n\n")
                    f.write("**Valoración: BUENA**\n\n")
                    f.write("El sistema distribuye la carga de manera adecuada, con niveles aceptables de variación. El balanceo de carga funciona correctamente en la mayoría de los escenarios.\n\n")
                elif avg_variation < 0.6:
                    f.write("### Distribución de Carga\n\n")
                    f.write("**Valoración: ACEPTABLE**\n\n")
                    f.write("El sistema muestra cierta capacidad para distribuir la carga, pero con variaciones significativas. Se recomienda mejorar los algoritmos de distribución para lograr un uso más uniforme de los recursos.\n\n")
                else:
                    f.write("### Distribución de Carga\n\n")
                    f.write("**Valoración: INSUFICIENTE**\n\n")
                    f.write("El sistema muestra una distribución de carga muy desigual. Es necesario rediseñar los mecanismos de balanceo de carga para lograr un uso más eficiente de los recursos.\n\n")
            
            # Recomendaciones específicas
            f.write("### Recomendaciones Específicas\n\n")
            
            recommendations = []
            
            # Analizar resultados y generar recomendaciones específicas
            if peak_results:
                recovery_ratios = [r.get("recovery_ratio", 0) for r in peak_results]
                avg_recovery = sum(recovery_ratios) / len(recovery_ratios) if recovery_ratios else 0
                
                if avg_recovery < 0.7:
                    recommendations.append("Mejorar los algoritmos de recuperación tras picos de carga. Considerar técnicas como back-pressure para evitar sobrecarga del sistema.")
                    
                peak_success_rates = [p.get("peak_metrics", {}).get("success_rate", 0) for p in peak_results]
                avg_peak_success = sum(peak_success_rates) / len(peak_success_rates) if peak_success_rates else 0
                
                if avg_peak_success < 0.5:
                    recommendations.append("El sistema maneja inadecuadamente los picos de carga. Implementar mecanismos de degradación graceful para mantener funcionalidad básica durante picos extremos.")
            
            if dist_results:
                variation_coeffs = [r.get("distribution_stats", {}).get("overall", {}).get("variation_coefficient", 999) for r in dist_results]
                avg_variation = sum(variation_coeffs) / len(variation_coeffs) if variation_coeffs else 999
                
                if avg_variation > 0.4:
                    recommendations.append("Mejorar el balanceo de carga entre componentes. Considerar algoritmos más sofisticados como round-robin ponderado o least-connections.")
                
                success_rates = [r.get("high_uniform_result", {}).get("success_count", 0) / 
                               r.get("high_uniform_result", {}).get("total_events", 1) for r in dist_results]
                avg_success = sum(success_rates) / len(success_rates) if success_rates else 0
                
                if avg_success < 0.7:
                    recommendations.append("Optimizar el rendimiento en carga alta. Revisar bloqueos y cuellos de botella que reducen la tasa de éxito.")
            
            # Añadir recomendaciones generales si no hay suficientes específicas
            if len(recommendations) < 3:
                recommendations.append("Implementar monitoreo en tiempo real de la distribución de carga para detectar desequilibrios.")
                recommendations.append("Optimizar el algoritmo de escalado dinámico para responder más rápido a picos repentinos.")
                recommendations.append("Establecer límites por componente para evitar que un solo componente consuma todos los recursos.")
                recommendations.append("Implementar un sistema de prioridades para garantizar que los eventos críticos se procesen incluso durante picos de carga.")
            
            # Escribir recomendaciones
            for i, rec in enumerate(recommendations[:5]):  # Limitar a 5 recomendaciones
                f.write(f"{i+1}. {rec}\n")
        
        logger.info(f"Informe generado en {report_file}")
        return report_file
    
    def _generate_graphs(self, timestamp):
        """
        Generar gráficos basados en los resultados de las pruebas.
        
        Args:
            timestamp: Timestamp para los nombres de archivo
            
        Returns:
            Diccionario con rutas a los gráficos generados
        """
        graph_paths = {}
        
        # Gráfico de recuperación de picos
        peak_results = self.results.get("peak_recovery", [])
        if peak_results:
            try:
                # Crear figura
                plt.figure(figsize=(10, 6))
                
                # Preparar datos
                iterations = range(1, len(peak_results) + 1)
                recovery_ratios = [r.get("recovery_ratio", 0) for r in peak_results]
                baseline_rates = [r.get("baseline_metrics", {}).get("success_rate", 0) for r in peak_results]
                peak_rates = [r.get("peak_metrics", {}).get("success_rate", 0) for r in peak_results]
                recovery_rates = [r.get("recovery_metrics", {}).get("success_rate", 0) for r in peak_results]
                
                # Graficar
                plt.plot(iterations, baseline_rates, 'o-', label='Línea Base', color='green')
                plt.plot(iterations, peak_rates, 'o-', label='Pico de Carga', color='red')
                plt.plot(iterations, recovery_rates, 'o-', label='Recuperación', color='blue')
                
                # Línea de referencia (recuperación perfecta)
                plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
                
                # Configurar gráfico
                plt.title('Tasas de Éxito en Pruebas de Recuperación de Picos')
                plt.xlabel('Iteración')
                plt.ylabel('Tasa de Éxito')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1.1)
                
                # Guardar
                graph_path = os.path.join(self.output_dir, "graphs", f"peak_recovery_graph_{timestamp}.png")
                plt.savefig(graph_path)
                plt.close()
                
                graph_paths["peak_recovery_graph"] = graph_path
                logger.info(f"Gráfico de recuperación generado en {graph_path}")
                
                # Gráfico adicional: Ratio de recuperación
                plt.figure(figsize=(10, 6))
                plt.bar(iterations, recovery_ratios, color='purple')
                plt.axhline(y=1.0, color='green', linestyle='--', label='Recuperación Perfecta')
                plt.axhline(y=0.7, color='orange', linestyle='--', label='Recuperación Buena')
                plt.axhline(y=0.4, color='red', linestyle='--', label='Recuperación Mínima')
                
                plt.title('Ratio de Recuperación Tras Picos de Carga')
                plt.xlabel('Iteración')
                plt.ylabel('Ratio de Recuperación')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Guardar
                graph_path = os.path.join(self.output_dir, "graphs", f"recovery_ratio_graph_{timestamp}.png")
                plt.savefig(graph_path)
                plt.close()
                
                graph_paths["recovery_ratio_graph"] = graph_path
                
            except Exception as e:
                logger.error(f"Error al generar gráfico de recuperación: {e}")
        
        # Gráfico de distribución de carga
        dist_results = self.results.get("distribution", [])
        if dist_results:
            try:
                # Crear figura
                plt.figure(figsize=(10, 6))
                
                # Preparar datos
                iterations = range(1, len(dist_results) + 1)
                variation_coeffs = [r.get("distribution_stats", {}).get("overall", {}).get("variation_coefficient", 0) for r in dist_results]
                success_rates = [r.get("high_uniform_result", {}).get("success_count", 0) / 
                               r.get("high_uniform_result", {}).get("total_events", 1) for r in dist_results]
                
                # Crear ejes para dos métricas diferentes
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                # Coeficiente de variación (eje izquierdo)
                color = 'tab:blue'
                ax1.set_xlabel('Iteración')
                ax1.set_ylabel('Coeficiente de Variación', color=color)
                ax1.plot(iterations, variation_coeffs, 'o-', color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_ylim(0, min(max(variation_coeffs) * 1.2, 1.0))
                
                # Tasa de éxito (eje derecho)
                ax2 = ax1.twinx()
                color = 'tab:orange'
                ax2.set_ylabel('Tasa de Éxito', color=color)
                ax2.plot(iterations, success_rates, 'o-', color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim(0, 1.1)
                
                # Líneas de referencia
                ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
                ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
                
                # Título y configuración
                plt.title('Distribución de Carga y Tasa de Éxito')
                fig.tight_layout()
                
                # Guardar
                graph_path = os.path.join(self.output_dir, "graphs", f"distribution_graph_{timestamp}.png")
                plt.savefig(graph_path)
                plt.close()
                
                graph_paths["distribution_graph"] = graph_path
                logger.info(f"Gráfico de distribución generado en {graph_path}")
                
                # Gráfico adicional: Distribución por tipo de carga
                if dist_results[0].get("distribution_stats"):
                    # Extraer tipos y promedios
                    load_types = []
                    avg_variations = []
                    
                    for load_type, stats in dist_results[0].get("distribution_stats", {}).items():
                        if load_type != "overall":
                            load_types.append(load_type)
                            
                            # Calcular promedio de variación para este tipo a través de todas las iteraciones
                            variations = []
                            for result in dist_results:
                                if load_type in result.get("distribution_stats", {}):
                                    variations.append(result["distribution_stats"][load_type].get("variation_coefficient", 0))
                            
                            avg_variations.append(sum(variations) / len(variations) if variations else 0)
                    
                    if load_types and avg_variations:
                        plt.figure(figsize=(10, 6))
                        bars = plt.bar(load_types, avg_variations, color=['blue', 'green', 'orange', 'red'])
                        
                        # Añadir etiquetas de valor sobre las barras
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                    f'{height:.2f}', ha='center', va='bottom')
                        
                        plt.title('Coeficiente de Variación por Tipo de Carga')
                        plt.ylabel('Coeficiente de Variación')
                        plt.xlabel('Tipo de Carga')
                        plt.grid(axis='y', alpha=0.3)
                        
                        # Guardar
                        graph_path = os.path.join(self.output_dir, "graphs", f"load_type_variation_{timestamp}.png")
                        plt.savefig(graph_path)
                        plt.close()
                        
                        graph_paths["load_type_variation"] = graph_path
                
            except Exception as e:
                logger.error(f"Error al generar gráfico de distribución: {e}")
        
        return graph_paths


async def main():
    """Función principal."""
    try:
        logger.info("Iniciando generación de informe de recuperación de carga")
        
        generator = LoadRecoveryReportGenerator()
        
        # Por defecto, ejecutamos una iteración (puede ser ajustado)
        num_iterations = 1
        if len(sys.argv) > 1:
            try:
                num_iterations = int(sys.argv[1])
            except ValueError:
                logger.warning(f"Valor de iteraciones inválido: {sys.argv[1]}. Usando 1 por defecto.")
        
        await generator.run_tests(num_iterations)
        report_file = generator.generate_report()
        
        if report_file:
            logger.info(f"Informe generado exitosamente en: {report_file}")
        else:
            logger.error("No se pudo generar el informe")
            return 1
        
    except Exception as e:
        logger.error(f"Error durante la generación del informe: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Ejecutar la función principal dentro de un event loop
    loop = asyncio.get_event_loop()
    sys.exit(loop.run_until_complete(main()))