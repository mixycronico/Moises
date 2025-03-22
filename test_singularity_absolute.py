"""
Prueba extrema del Sistema Genesis en Modo Singularidad Absoluta.

Este script ejecuta pruebas extraordinariamente extremas para verificar las capacidades
del Sistema Genesis en Modo Singularidad Absoluta, comprobando su resiliencia
a niveles de intensidad nunca antes alcanzados (hasta 3.0).

Las pruebas evalúan:
1. Resiliencia bajo condiciones extremas (intensidad 3.0)
2. Rendimiento de componentes esenciales y no esenciales
3. Capacidad para mantener funcionamiento coherente bajo estrés extremo
4. Respuesta a anomalías temporales severas
5. Eficacia del colapso dimensional y túnel cuántico

La meta es verificar el funcionamiento perfecto (100% de éxito) en todos los casos.
"""

import asyncio
import logging
import time
import random
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from genesis_singularity_absolute import (
    SingularityCoordinator, 
    TestComponent, 
    SystemMode,
    EventPriority
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("resultados_singularidad_absoluta.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnomalyGenerator:
    """Generador de anomalías para pruebas de estrés extremo."""
    
    def __init__(self):
        """Inicializar generador de anomalías."""
        self.anomaly_types = [
            "temporal_distortion", 
            "quantum_uncertainty",
            "dimensional_collapse",
            "reality_breach",
            "causality_violation",
            "probability_storm",
            "information_paradox",
            "gravitational_singularity",
            "entropy_spike",
            "vacuum_decay"
        ]
        
    def generate_temporal_anomaly(self, intensity: float) -> Dict[str, Any]:
        """
        Generar anomalía temporal con intensidad especificada.
        
        Args:
            intensity: Intensidad de la anomalía (0.0-3.0+)
            
        Returns:
            Datos de la anomalía
        """
        # Seleccionar tipo de anomalía
        anomaly_type = random.choice(self.anomaly_types)
        
        # Calcular potencia según intensidad
        power = intensity * (0.8 + random.random() * 0.4)
        
        # Crear datos de anomalía
        anomaly = {
            "type": anomaly_type,
            "timestamp": time.time(),
            "intensity": intensity,
            "power": power,
            "duration": random.uniform(0.001, 0.1) * intensity,
            "coordinates": {
                "x": random.uniform(-1, 1),
                "y": random.uniform(-1, 1),
                "z": random.uniform(-1, 1),
                "t": random.uniform(-1, 1)
            },
            "quantum_state": {
                "superposition": random.random() > 0.5,
                "entanglement": random.random() > 0.7,
                "coherence": max(0, 1 - intensity/3)
            },
            "potential_impact": min(100, intensity * 33.33),  # 0-100%
            "random_seed": random.randint(1, 1000000)
        }
        
        return anomaly
        
    def generate_batch(self, count: int, intensity: float) -> List[Dict[str, Any]]:
        """
        Generar lote de anomalías.
        
        Args:
            count: Número de anomalías
            intensity: Intensidad base
            
        Returns:
            Lista de anomalías
        """
        anomalies = []
        
        for _ in range(count):
            # Variar ligeramente la intensidad
            individual_intensity = intensity * (0.9 + random.random() * 0.2)
            anomalies.append(self.generate_temporal_anomaly(individual_intensity))
            
        return anomalies


class SingularityTester:
    """Test runner para el Sistema Genesis - Modo Singularidad Absoluta."""
    
    def __init__(self):
        """Inicializar entorno de pruebas."""
        self.anomaly_generator = AnomalyGenerator()
        self.coordinator = None
        self.components = {}
        self.essential_components = []
        self.non_essential_components = []
        self.results = {}
        self.anomalies_generated = 0
        self.events_processed = 0
        self.requests_processed = 0
        
    async def setup(self):
        """Preparar entorno para pruebas."""
        logger.info("Configurando entorno de pruebas para Singularidad Absoluta")
        
        # Crear coordinador
        self.coordinator = SingularityCoordinator(host="localhost", port=8080, max_connections=1000)
        
        # Crear componentes esenciales
        for i in range(3):
            component = TestComponent(f"essential_{i}", is_essential=True)
            self.components[component.id] = component
            self.essential_components.append(component)
            self.coordinator.register_component(component.id, component)
            
        # Crear componentes no esenciales
        for i in range(7):
            component = TestComponent(f"component_{i}", is_essential=False)
            self.components[component.id] = component
            self.non_essential_components.append(component)
            self.coordinator.register_component(component.id, component)
            
        # Iniciar coordinador
        await self.coordinator.start()
        
        # Iniciar listeners para cada componente
        for component_id, component in self.components.items():
            asyncio.create_task(component.listen_local())
            
        logger.info(f"Entorno preparado con {len(self.components)} componentes")
            
    async def test_intensity(self, intensity: float, iterations: int = 3) -> Dict[str, Any]:
        """
        Probar sistema con una intensidad específica.
        
        Args:
            intensity: Intensidad de la prueba (0.0-3.0+)
            iterations: Número de iteraciones
            
        Returns:
            Resultados de las pruebas
        """
        test_start = time.time()
        logger.info(f"Probando intensidad {intensity:.2f} ({iterations} iteraciones)")
        
        results = {
            "intensity": intensity,
            "iterations": iterations,
            "start_time": test_start,
            "cycles": [],
            "success_rates": {
                "overall": 0.0,
                "essential": 0.0,
                "non_essential": 0.0
            },
            "anomalies": 0,
            "events": 0,
            "requests": 0
        }
        
        # Ejecutar ciclos de prueba
        for i in range(iterations):
            logger.info(f"Iteración {i+1}/{iterations} para intensidad {intensity:.2f}")
            cycle_results = await self._run_test_cycle(intensity, i)
            results["cycles"].append(cycle_results)
            
            # Actualizar contadores
            results["anomalies"] += cycle_results["anomalies_generated"]
            results["events"] += cycle_results["events_processed"]
            results["requests"] += cycle_results["requests_processed"]
        
        # Calcular estadísticas
        test_end = time.time()
        results["duration"] = test_end - test_start
        
        # Promediar tasas de éxito
        total_success = 0.0
        essential_success = 0.0
        non_essential_success = 0.0
        
        for cycle in results["cycles"]:
            total_success += cycle["success_rates"]["overall"]
            essential_success += cycle["success_rates"]["essential"]
            non_essential_success += cycle["success_rates"]["non_essential"]
            
        if results["cycles"]:
            results["success_rates"]["overall"] = total_success / len(results["cycles"])
            results["success_rates"]["essential"] = essential_success / len(results["cycles"])
            results["success_rates"]["non_essential"] = non_essential_success / len(results["cycles"])
        
        # Guardar resultados
        self._save_results(results, intensity)
        
        logger.info(f"Pruebas completadas para intensidad {intensity:.2f}")
        logger.info(f"Tasa de éxito global: {results['success_rates']['overall']:.2f}%")
        logger.info(f"Tasa componentes esenciales: {results['success_rates']['essential']:.2f}%")
        logger.info(f"Tasa componentes no esenciales: {results['success_rates']['non_essential']:.2f}%")
        
        return results
    
    async def _run_test_cycle(self, intensity: float, cycle_index: int) -> Dict[str, Any]:
        """
        Ejecutar un ciclo de prueba.
        
        Args:
            intensity: Intensidad de la prueba
            cycle_index: Índice del ciclo
            
        Returns:
            Resultados del ciclo
        """
        cycle_id = f"cycle_{cycle_index}_{int(time.time())}"
        cycle_start = time.time()
        
        # Contadores de éxito para este ciclo
        success_counters = {
            "total_attempts": 0,
            "total_successes": 0,
            "essential_attempts": 0,
            "essential_successes": 0,
            "non_essential_attempts": 0,
            "non_essential_successes": 0
        }
        
        # 1. Generar anomalías temporales
        logger.info(f"Iniciando ciclo de anomalías {cycle_id} con intensidad {intensity:.2f}")
        anomalies_start = time.time()
        
        # Número de anomalías proporcional a intensidad
        num_anomalies = max(5, int(30 * intensity))
        logger.info(f"Generando {num_anomalies} anomalías temporales (intensidad: {intensity:.2f})")
        
        anomalies = self.anomaly_generator.generate_batch(num_anomalies, intensity)
        self.anomalies_generated += len(anomalies)
        
        # Procesar cada anomalía
        anomaly_results = []
        for anomaly in anomalies:
            # Variar ligeramente la intensidad
            individual_intensity = intensity * (0.9 + random.random() * 0.2)
            
            # Crear evento
            event_data = {
                "anomaly": anomaly,
                "cycle_id": cycle_id,
                "intensity": individual_intensity
            }
            
            # Emitir evento de anomalía temporal
            await self.coordinator.emit_local(
                "temporal_anomaly", 
                event_data, 
                "tester",
                priority=EventPriority.CRITICAL,
                intensity=individual_intensity
            )
            
            # Breve pausa para permitir propagación
            await asyncio.sleep(0.001)
            
        anomalies_end = time.time()
        anomalies_duration = anomalies_end - anomalies_start
        
        # 2. Realizar peticiones para probar resiliencia
        logger.info(f"Iniciando ciclo de procesamiento {cycle_id} con intensidad {intensity:.2f}")
        processing_start = time.time()
        
        # Número de eventos proporcional a intensidad
        num_events = max(50, int(100 * intensity))
        num_requests = max(30, int(50 * intensity))
        
        # Emitir eventos regulares
        for i in range(num_events):
            # Tipo de evento aleatorio
            event_type = random.choice([
                "data_update", "status_change", "configuration_update", 
                "metric_report", "health_check", "system_alert"
            ])
            
            # Datos de evento
            event_data = {
                "id": f"evt_{i}_{int(time.time()*1000)}",
                "timestamp": time.time(),
                "cycle_id": cycle_id,
                "value": random.random() * 100,
                "parameters": {
                    "param1": random.choice([True, False]),
                    "param2": random.randint(1, 100),
                    "param3": "test_value_" + str(random.randint(1, 1000))
                }
            }
            
            # Prioridad variable
            priority = random.choice([
                EventPriority.CRITICAL, EventPriority.HIGH, 
                EventPriority.NORMAL, EventPriority.LOW
            ])
            
            # Emitir evento
            await self.coordinator.emit_local(
                event_type, 
                event_data, 
                "tester",
                priority=priority,
                intensity=intensity
            )
            
            self.events_processed += 1
            
            # Breve pausa entre eventos
            if i % 10 == 0:
                await asyncio.sleep(0.001)
        
        # Realizar peticiones a componentes
        request_results = []
        
        for i in range(num_requests):
            # Seleccionar componente aleatorio
            is_essential = random.random() < 0.3  # 30% a componentes esenciales
            
            if is_essential and self.essential_components:
                component = random.choice(self.essential_components)
                success_counters["essential_attempts"] += 1
            elif self.non_essential_components:
                component = random.choice(self.non_essential_components)
                success_counters["non_essential_attempts"] += 1
            else:
                continue
                
            # Tipo de petición
            request_type = random.choice([
                "get_data", "process_data", "validate_input", 
                "compute_metrics", "check_status", "update_config"
            ])
            
            # Datos de petición
            request_data = {
                "id": f"req_{i}_{int(time.time()*1000)}",
                "timestamp": time.time(),
                "cycle_id": cycle_id,
                "parameters": {
                    "param1": random.random() * 100,
                    "param2": random.choice(["option1", "option2", "option3"]),
                    "param3": [random.randint(1, 100) for _ in range(5)]
                }
            }
            
            # Realizar petición
            try:
                result = await self.coordinator.request(
                    component.id,
                    request_type,
                    request_data,
                    "tester",
                    intensity=intensity
                )
                
                # Verificar éxito
                request_success = (
                    result is not None and 
                    isinstance(result, dict) and 
                    result.get("success", False)
                )
                
                # Registrar éxito/fallo
                if request_success:
                    if is_essential:
                        success_counters["essential_successes"] += 1
                    else:
                        success_counters["non_essential_successes"] += 1
                        
                success_counters["total_attempts"] += 1
                if request_success:
                    success_counters["total_successes"] += 1
                    
                request_results.append({
                    "component_id": component.id,
                    "is_essential": component.is_essential,
                    "request_type": request_type,
                    "success": request_success,
                    "data": str(result)[:100] + ("..." if result and len(str(result)) > 100 else "")
                })
                
                self.requests_processed += 1
                
            except Exception as e:
                logger.error(f"Error al realizar petición: {str(e)}")
                request_results.append({
                    "component_id": component.id,
                    "is_essential": component.is_essential,
                    "request_type": request_type,
                    "success": False,
                    "error": str(e)
                })
                
                success_counters["total_attempts"] += 1
            
            # Breve pausa entre peticiones
            if i % 5 == 0:
                await asyncio.sleep(0.001)
                
        processing_end = time.time()
        processing_duration = processing_end - processing_start
        
        # Calcular tasas de éxito
        success_rates = {
            "overall": (success_counters["total_successes"] / max(1, success_counters["total_attempts"])) * 100,
            "essential": (success_counters["essential_successes"] / max(1, success_counters["essential_attempts"])) * 100,
            "non_essential": (success_counters["non_essential_successes"] / max(1, success_counters["non_essential_attempts"])) * 100
        }
        
        # Recopilar estadísticas de componentes
        component_stats = {}
        for component_id, component in self.components.items():
            component_stats[component_id] = component.get_stats()
        
        # Recopilar estadísticas del coordinador
        coordinator_stats = self.coordinator.get_stats()
        
        cycle_end = time.time()
        cycle_duration = cycle_end - cycle_start
        
        # Pausar brevemente entre ciclos
        await asyncio.sleep(0.1)
        
        # Resultados del ciclo
        cycle_results = {
            "cycle_id": cycle_id,
            "intensity": intensity,
            "duration": cycle_duration,
            "anomalies_generated": len(anomalies),
            "anomalies_duration": anomalies_duration,
            "events_processed": num_events,
            "requests_processed": len(request_results),
            "processing_duration": processing_duration,
            "success_counters": success_counters,
            "success_rates": success_rates,
            "request_results_sample": request_results[:5],  # Muestra limitada para no sobrecargar
            "component_stats_sample": {k: component_stats[k] for k in list(component_stats.keys())[:3]},
            "coordinator_stats": {
                "system_mode": coordinator_stats.get("system_mode"),
                "uptime_seconds": coordinator_stats.get("uptime_seconds"),
                "component_count": coordinator_stats.get("component_count"),
                "success_rate": coordinator_stats.get("success_rate"),
                "collapse_factor": coordinator_stats.get("collapse_factor")
            }
        }
        
        return cycle_results
    
    def _save_results(self, results: Dict[str, Any], intensity: float) -> None:
        """
        Guardar resultados de prueba.
        
        Args:
            results: Resultados completos
            intensity: Intensidad de la prueba
        """
        # Guardar en memoria
        self.results[f"intensity_{intensity:.2f}"] = results
        
        # Guardar en archivo
        filename = f"resultados_singularidad_{intensity:.2f}.json"
        with open(filename, "w") as f:
            # Crear versión serializable
            serializable = {
                "intensity": results["intensity"],
                "iterations": results["iterations"],
                "start_time": results["start_time"],
                "duration": results["duration"],
                "success_rates": results["success_rates"],
                "anomalies": results["anomalies"],
                "events": results["events"],
                "requests": results["requests"],
                "cycles_count": len(results["cycles"]),
            }
            json.dump(serializable, f, indent=2)
            
        logger.info(f"Resultados guardados en {filename}")
    
    def generate_report(self, intensities: List[float]) -> None:
        """
        Generar informe gráfico de resultados.
        
        Args:
            intensities: Lista de intensidades probadas
        """
        # Preparar datos
        intensity_values = []
        overall_success = []
        essential_success = []
        non_essential_success = []
        
        for intensity in intensities:
            result_key = f"intensity_{intensity:.2f}"
            if result_key in self.results:
                result = self.results[result_key]
                intensity_values.append(intensity)
                overall_success.append(result["success_rates"]["overall"])
                essential_success.append(result["success_rates"]["essential"])
                non_essential_success.append(result["success_rates"]["non_essential"])
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        plt.plot(intensity_values, overall_success, 'b-', marker='o', label='Total')
        plt.plot(intensity_values, essential_success, 'g-', marker='s', label='Componentes Esenciales')
        plt.plot(intensity_values, non_essential_success, 'r-', marker='x', label='Componentes No Esenciales')
        
        # Línea de 100%
        plt.axhline(y=100, color='k', linestyle='--', alpha=0.3)
        
        # Etiquetas y leyenda
        plt.xlabel('Intensidad')
        plt.ylabel('Tasa de Éxito (%)')
        plt.title('Rendimiento del Sistema Genesis - Modo Singularidad Absoluta')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mostrar valores
        for i, intensity in enumerate(intensity_values):
            plt.annotate(f"{overall_success[i]:.2f}%", 
                        (intensity, overall_success[i]), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center')
        
        # Guardar gráfico
        plt.savefig('rendimiento_singularidad_absoluta.png')
        logger.info("Informe gráfico generado: rendimiento_singularidad_absoluta.png")
        
        # Crear informe de texto
        with open("informe_singularidad_absoluta.md", "w") as f:
            f.write("# Informe de Pruebas: Sistema Genesis - Modo Singularidad Absoluta\n\n")
            f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Resumen de Resultados\n\n")
            f.write("| Intensidad | Global | Esenciales | No Esenciales |\n")
            f.write("|------------|--------|------------|---------------|\n")
            
            for i, intensity in enumerate(intensity_values):
                f.write(f"| {intensity:.2f} | {overall_success[i]:.2f}% | {essential_success[i]:.2f}% | {non_essential_success[i]:.2f}% |\n")
            
            f.write("\n## Análisis del Rendimiento\n\n")
            
            # Determinar si se alcanzó el objetivo
            target_intensity = max(intensity_values)
            target_result = self.results.get(f"intensity_{target_intensity:.2f}")
            
            if target_result and target_result["success_rates"]["overall"] >= 99.9:
                f.write(f"El Sistema Genesis - Modo Singularidad Absoluta ha **SUPERADO** el objetivo, logrando un rendimiento excepcional de {target_result['success_rates']['overall']:.2f}% a intensidad {target_intensity:.2f}.\n\n")
                f.write("Este resultado representa un avance significativo respecto al punto de ruptura anterior de 1.00 en el Modo Luz.\n\n")
            else:
                f.write(f"El Sistema Genesis - Modo Singularidad Absoluta ha alcanzado un rendimiento de {target_result['success_rates']['overall']:.2f}% a intensidad {target_intensity:.2f}.\n\n")
            
            f.write("### Componentes Esenciales\n\n")
            
            essential_target = target_result["success_rates"]["essential"] if target_result else 0
            if essential_target >= 99.9:
                f.write(f"Los componentes esenciales mantuvieron una protección **perfecta** de {essential_target:.2f}%, garantizando la integridad operativa incluso bajo condiciones extremas.\n\n")
            else:
                f.write(f"Los componentes esenciales alcanzaron una protección de {essential_target:.2f}%.\n\n")
                
            f.write("![Gráfico de Rendimiento](rendimiento_singularidad_absoluta.png)\n\n")
            
            f.write("## Conclusiones\n\n")
            if target_result and target_result["success_rates"]["overall"] >= 99.9:
                f.write("El Modo Singularidad Absoluta representa una evolución trascendental del sistema, superando todas las expectativas y estableciendo un nuevo estándar en resiliencia extrema.\n\n")
                f.write("La capacidad de mantener un funcionamiento perfecto bajo intensidades que antes causaban degradación significativa demuestra el potencial revolucionario de los mecanismos implementados.\n\n")
            else:
                f.write("El Modo Singularidad Absoluta muestra mejoras significativas, pero requiere optimizaciones adicionales para alcanzar la resiliencia perfecta en intensidades extremas.\n\n")
                
            f.write("**Clasificación**: Informe Técnico - Análisis de Rendimiento\n")
        
        logger.info("Informe detallado generado: informe_singularidad_absoluta.md")


async def main():
    """Función principal para ejecutar pruebas."""
    tester = SingularityTester()
    
    try:
        # Configurar entorno
        await tester.setup()
        
        logger.info("=== Iniciando pruebas de Singularidad Absoluta ===")
        
        # Probar intensidades crecientes
        intensities = [0.5, 1.0, 2.0, 3.0]
        
        for intensity in intensities:
            await tester.test_intensity(intensity, iterations=3)
            
        # Generar informe
        tester.generate_report(intensities)
        
        logger.info("=== Pruebas completadas ===")
        
    except Exception as e:
        logger.error(f"Error en pruebas: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Breve pausa para que logs se completen
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    # Ejecutar pruebas
    asyncio.run(main())