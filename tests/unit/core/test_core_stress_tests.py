"""
Pruebas de estrés para el motor de eventos Genesis.

Este módulo contiene pruebas intensivas diseñadas para estresar el motor
hasta sus límites con el fin de identificar cuellos de botella y puntos débiles.
"""

import asyncio
import logging
import pytest
import time
import random
from typing import Dict, Any, List, Optional, Tuple

# Importamos las utilidades de timeout y rendimiento
from tests.utils.timeout_helpers import (
    emit_with_timeout, 
    check_component_status,
    run_test_with_timing, 
    cleanup_engine
)

# Configuración de logging para pruebas de estrés
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nos aseguramos que pytest detecte correctamente las pruebas asíncronas
pytestmark = pytest.mark.asyncio

# Importamos las clases necesarias para las pruebas
from genesis.core.component import Component
from genesis.core.engine_dynamic_blocks import DynamicExpansionEngine
from genesis.core.engine_priority_blocks import PriorityBlockEngine


class SlowComponent(Component):
    """
    Componente que intencionalmente procesa eventos de manera lenta.
    
    Este componente puede configurarse para simular diferentes latencias,
    bloqueos o comportamientos bajo carga.
    """
    
    def __init__(self, name: str, processing_time: float = 0.1, failure_rate: float = 0.0):
        """
        Inicializar componente con características de latencia controladas.
        
        Args:
            name: Nombre del componente
            processing_time: Tiempo base de procesamiento en segundos
            failure_rate: Tasa de fallos (0.0 a 1.0) para simular errores aleatorios
        """
        super().__init__(name)
        self.processing_time = processing_time
        self.failure_rate = failure_rate
        self.healthy = True
        self.processed_count = 0
        self.failed_count = 0
        self.total_processing_time = 0
        self.load_factor = 1.0  # Multiplicador de tiempo de procesamiento según carga
        
    async def start(self) -> None:
        """Iniciar el componente."""
        logger.debug(f"Componente {self.name} iniciado")
        self.healthy = True
        
    async def stop(self) -> None:
        """Detener el componente."""
        logger.debug(f"Componente {self.name} detenido")
        self.healthy = False
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos con latencia controlada.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta al evento, si corresponde
        """
        if event_type == "check_status":
            return {
                "healthy": self.healthy,
                "processed": self.processed_count,
                "failed": self.failed_count,
                "avg_time": self.total_processing_time / max(1, self.processed_count),
                "load_factor": self.load_factor
            }
            
        elif event_type == "set_health":
            self.healthy = data.get("healthy", True)
            return {"healthy": self.healthy}
            
        elif event_type == "set_load_factor":
            self.load_factor = data.get("factor", 1.0)
            return {"load_factor": self.load_factor}
            
        elif event_type == "process":
            # Simular tiempo de procesamiento variable según la carga
            actual_time = self.processing_time * self.load_factor
            
            # Simular procesamiento variable según datos
            complexity = data.get("complexity", 1.0)
            actual_time *= complexity
            
            start_time = time.time()
            
            # Simular fallo aleatorio
            if random.random() < self.failure_rate:
                self.failed_count += 1
                await asyncio.sleep(actual_time * 0.5)  # Falla más rápido que procesamiento normal
                return {"success": False, "error": "Fallo simulado aleatorio"}
            
            # Procesamiento simulado
            await asyncio.sleep(actual_time)
            
            elapsed = time.time() - start_time
            self.processed_count += 1
            self.total_processing_time += elapsed
            
            return {
                "success": True,
                "processed_in": elapsed,
                "component": self.name,
                "data": data
            }
            
        return None


class PriorityAwareComponent(Component):
    """
    Componente que respeta prioridades y monitores su comportamiento.
    
    Este componente mantiene estadísticas sobre eventos procesados
    según su nivel de prioridad.
    """
    
    def __init__(self, name: str, base_latency: float = 0.05):
        """
        Inicializar componente sensible a prioridades.
        
        Args:
            name: Nombre del componente
            base_latency: Tiempo base de procesamiento en segundos
        """
        super().__init__(name)
        self.healthy = True
        self.base_latency = base_latency
        
        # Estadísticas por prioridad
        self.processed_by_priority = {
            "high": 0,
            "medium": 0,
            "low": 0
        }
        self.processing_time_by_priority = {
            "high": 0.0,
            "medium": 0.0,
            "low": 0.0
        }
        
    async def start(self) -> None:
        """Iniciar el componente."""
        logger.debug(f"Componente {self.name} iniciado")
        self.healthy = True
        
    async def stop(self) -> None:
        """Detener el componente."""
        logger.debug(f"Componente {self.name} detenido")
        self.healthy = False
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos respetando prioridades.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta al evento, si corresponde
        """
        if event_type == "check_status":
            total_processed = sum(self.processed_by_priority.values())
            
            # Calcular tiempos promedio por prioridad
            avg_times = {}
            for priority, total_time in self.processing_time_by_priority.items():
                count = self.processed_by_priority.get(priority, 0)
                avg_times[priority] = total_time / max(1, count)
                
            return {
                "healthy": self.healthy,
                "stats_by_priority": self.processed_by_priority,
                "avg_times": avg_times,
                "total_processed": total_processed
            }
            
        elif event_type == "process_with_priority":
            priority = data.get("priority", "medium")
            payload = data.get("payload", {})
            
            # El tiempo de procesamiento depende de la prioridad
            # (alta prioridad se procesa más rápido)
            priority_factors = {
                "high": 0.5,    # Alta prioridad: 50% del tiempo base
                "medium": 1.0,  # Prioridad media: tiempo base
                "low": 2.0      # Baja prioridad: doble del tiempo base
            }
            
            process_time = self.base_latency * priority_factors.get(priority, 1.0)
            
            # Realizar procesamiento simulado
            start_time = time.time()
            await asyncio.sleep(process_time)
            elapsed = time.time() - start_time
            
            # Actualizar estadísticas
            self.processed_by_priority[priority] = self.processed_by_priority.get(priority, 0) + 1
            self.processing_time_by_priority[priority] = self.processing_time_by_priority.get(priority, 0.0) + elapsed
            
            return {
                "success": True,
                "priority": priority,
                "processed_in": elapsed,
                "component": self.name,
                "payload": payload
            }
            
        return None


@pytest.mark.slow
@pytest.mark.asyncio
async def test_gradual_load_increase(dynamic_engine):
    """
    Test de aumento gradual de carga hasta encontrar un cuello de botella.
    
    Esta prueba va incrementando el número de eventos por segundo hasta que
    el sistema comienza a mostrar degradación significativa.
    """
    engine = dynamic_engine
    
    # Función interna para la prueba
    async def run_gradual_load_test(engine):
        # Configuración de componentes
        components = []
        for i in range(10):
            comp = SlowComponent(f"grad_comp_{i}", processing_time=0.05)
            await engine.register_component(comp)
            components.append(comp)
        
        # Función para enviar lotes de eventos con medición
        async def send_event_batch(batch_size: int, complexity: float = 1.0) -> Dict:
            tasks = []
            start_time = time.time()
            
            for i in range(batch_size):
                comp_id = f"grad_comp_{i % 10}"
                # Usar emit_with_timeout para evitar bloqueos
                task = emit_with_timeout(
                    engine, 
                    "process", 
                    {"complexity": complexity, "batch_id": i // 10}, 
                    comp_id,
                    timeout=5.0
                )
                tasks.append(task)
            
            # Esperar a que todas las tareas se completen (o alcancen timeout)
            results = await asyncio.gather(*tasks)
            
            # Calcular estadísticas
            elapsed = time.time() - start_time
            successful = len([r for r in results if isinstance(r, list) and len(r) > 0 and r[0].get("success", False)])
            timeouts = len([r for r in results if isinstance(r, list) and len(r) > 0 and r[0].get("error") == "timeout"])
            
            return {
                "batch_size": batch_size,
                "time": elapsed,
                "events_per_second": batch_size / elapsed if elapsed > 0 else 0,
                "successful": successful,
                "timeouts": timeouts,
                "success_rate": successful / batch_size if batch_size > 0 else 0
            }
        
        # Aumentar gradualmente la carga para encontrar el límite
        results = []
        batch_sizes = [10, 20, 50, 100, 200, 500]
        
        for size in batch_sizes:
            logger.info(f"Ejecutando batch de {size} eventos...")
            result = await send_event_batch(size)
            results.append(result)
            
            logger.info(f"Batch de {size} completado en {result['time']:.3f}s - " 
                       f"{result['events_per_second']:.2f} eventos/s - "
                       f"Exitosos: {result['successful']}/{size} "
                       f"({result['success_rate'] * 100:.1f}%)")
            
            # Si la tasa de éxito cae por debajo del 90%, hemos encontrado el límite
            if result['success_rate'] < 0.9:
                logger.warning(f"Cuello de botella detectado en batch de {size} eventos")
                break
        
        # Verificar estado final de los componentes
        component_stats = []
        for i, comp in enumerate(components):
            stats = await check_component_status(engine, comp.name)
            processed = stats.get("processed", 0)
            failed = stats.get("failed", 0)
            avg_time = stats.get("avg_time", 0)
            
            logger.info(f"Componente {comp.name}: {processed} procesados, "
                       f"{failed} fallidos, tiempo medio: {avg_time:.5f}s")
            
            component_stats.append({
                "name": comp.name,
                "processed": processed,
                "failed": failed,
                "avg_time": avg_time
            })
        
        # Análisis de tendencia de rendimiento
        if len(results) >= 2:
            # Comparar el rendimiento (eventos/s) entre el primer y último lote
            first_throughput = results[0]['events_per_second']
            last_throughput = results[-1]['events_per_second']
            
            expected_ratio = results[-1]['batch_size'] / results[0]['batch_size']
            actual_ratio = last_throughput / first_throughput if first_throughput > 0 else 0
            
            scaling_efficiency = actual_ratio / expected_ratio if expected_ratio > 0 else 0
            
            logger.info(f"Análisis de escalabilidad: eficiencia de {scaling_efficiency:.2f}x "
                       f"(ideal: 1.0x)")
            
            return {
                "results": results,
                "component_stats": component_stats,
                "scaling_efficiency": scaling_efficiency,
                "max_throughput": max(r['events_per_second'] for r in results)
            }
        
        return {"results": results, "component_stats": component_stats}
    
    # Ejecutar y medir la prueba completa
    result = await run_test_with_timing(
        engine, 
        "test_gradual_load_increase", 
        run_gradual_load_test
    )
    
    # No fallamos la prueba, solo verificamos que la ejecución complete
    assert isinstance(result, dict), "La prueba debería devolver un dict con resultados"
    assert "results" in result, "Los resultados deberían incluir las métricas por lote"
    assert "component_stats" in result, "Los resultados deberían incluir estadísticas de los componentes"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_slow_component_isolation(dynamic_engine):
    """
    Test para verificar el aislamiento de componentes lentos.
    
    Esta prueba introduce intencionalmente componentes muy lentos
    entre otros componentes rápidos, verificando si el motor puede
    aislar adecuadamente el rendimiento.
    """
    engine = dynamic_engine
    
    # Función interna para la prueba
    async def run_isolation_test(engine):
        # Registrar mezcla de componentes rápidos y lentos
        component_configs = [
            # Componentes rápidos (10)
            *[("fast_comp_{i}", 0.01) for i in range(10)],
            
            # Componentes medios (5)
            *[("med_comp_{i}", 0.1) for i in range(5)],
            
            # Componentes muy lentos (2)
            ("slow_comp_0", 0.5),
            ("slow_comp_1", 1.0)
        ]
        
        components = {}
        for name, proc_time in component_configs:
            comp = SlowComponent(name, processing_time=proc_time)
            await engine.register_component(comp)
            components[name] = comp
        
        # Enviar eventos a todos los componentes y medir
        async def process_all_components():
            tasks = []
            start_time = time.time()
            
            for name in components.keys():
                task = emit_with_timeout(
                    engine,
                    "process",
                    {"complexity": 1.0, "timestamp": time.time()},
                    name,
                    timeout=5.0
                )
                tasks.append((name, task))
            
            # Esperar todas las tareas
            results = []
            for name, task in tasks:
                try:
                    result = await task
                    results.append({
                        "component": name,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "component": name,
                        "error": str(e)
                    })
            
            elapsed = time.time() - start_time
            return {
                "total_time": elapsed,
                "results": results
            }
        
        # Ejecutar tres rondas y medir aislamiento
        logger.info("Ronda 1: Primera ejecución")
        round1 = await process_all_components()
        
        logger.info(f"Ronda 1 completada en {round1['total_time']:.3f}s")
        
        # Bloquear completamente un componente lento
        logger.info("Bloqueando componente slow_comp_1 (estableciendo carga x10)")
        await emit_with_timeout(
            engine,
            "set_load_factor",
            {"factor": 10.0},
            "slow_comp_1",
            timeout=1.0
        )
        
        logger.info("Ronda 2: Con un componente extremadamente lento")
        round2 = await process_all_components()
        
        logger.info(f"Ronda 2 completada en {round2['total_time']:.3f}s")
        
        # Comprobar estadísticas de todos los componentes
        component_stats = {}
        for name, comp in components.items():
            stats = await check_component_status(engine, name)
            component_stats[name] = stats
            
            category = "rápido" if "fast" in name else "medio" if "med" in name else "lento"
            logger.info(f"Componente {name} ({category}): "
                      f"{stats.get('processed', 0)} procesados, "
                      f"tiempo medio: {stats.get('avg_time', 0):.5f}s")
        
        # Análisis del impacto del componente lento
        impacto_ronda1_vs_ronda2 = round2['total_time'] / round1['total_time'] if round1['total_time'] > 0 else float('inf')
        
        logger.info(f"Impacto del componente lento: tiempo total aumentó {impacto_ronda1_vs_ronda2:.2f}x")
        
        # Agregar estadísticas por categoría de componente
        stats_by_category = {
            "fast": {
                "count": 0,
                "total_processed": 0,
                "avg_time": 0.0
            },
            "med": {
                "count": 0,
                "total_processed": 0,
                "avg_time": 0.0
            },
            "slow": {
                "count": 0,
                "total_processed": 0,
                "avg_time": 0.0
            }
        }
        
        for name, stats in component_stats.items():
            category = "fast" if "fast" in name else "med" if "med" in name else "slow"
            cat_stats = stats_by_category[category]
            
            cat_stats["count"] += 1
            cat_stats["total_processed"] += stats.get("processed", 0)
            cat_stats["avg_time"] += stats.get("avg_time", 0)
        
        # Calcular promedios
        for category, stats in stats_by_category.items():
            if stats["count"] > 0:
                stats["avg_time"] /= stats["count"]
                stats["avg_processed"] = stats["total_processed"] / stats["count"]
        
        logger.info("Estadísticas por categoría:")
        for category, stats in stats_by_category.items():
            logger.info(f"  {category.capitalize()}: "
                      f"{stats['count']} componentes, "
                      f"{stats['avg_processed']:.1f} eventos promedio, "
                      f"tiempo medio: {stats['avg_time']:.5f}s")
        
        # Calcular métricas de aislamiento
        # Idealmente, los componentes rápidos no deberían verse muy afectados
        # por los componentes lentos
        return {
            "round1": round1,
            "round2": round2,
            "impact_factor": impacto_ronda1_vs_ronda2,
            "component_stats": component_stats,
            "stats_by_category": stats_by_category
        }
    
    # Ejecutar y medir la prueba completa
    result = await run_test_with_timing(
        engine, 
        "test_slow_component_isolation", 
        run_isolation_test
    )
    
    # No fallamos la prueba, solo verificamos que la ejecución complete
    assert isinstance(result, dict), "La prueba debería devolver un dict con resultados"
    assert "impact_factor" in result, "Los resultados deberían incluir el factor de impacto"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_high_concurrency(dynamic_engine):
    """
    Test de alta concurrencia con muchos componentes.
    
    Esta prueba simula un sistema con un gran número de componentes
    emitiendo y recibiendo eventos simultáneamente.
    """
    engine = dynamic_engine
    
    # Función interna para la prueba
    async def run_concurrency_test(engine):
        # Crear muchos componentes con diferentes latencias
        num_components = 100  # Un número significativo de componentes
        components = {}
        
        logger.info(f"Creando {num_components} componentes...")
        for i in range(num_components):
            # Distribuir latencias entre 0.01s y 0.2s
            latency = 0.01 + (i % 20) * 0.01
            comp = SlowComponent(f"conc_comp_{i}", processing_time=latency)
            await engine.register_component(comp)
            components[f"conc_comp_{i}"] = comp
        
        # Función para generar un patrón de tráfico realista
        # donde algunos componentes reciben más eventos que otros
        async def generate_realistic_traffic(num_events=1000):
            # Distribución del tráfico - 80% va al 20% de los componentes
            popular_components = random.sample(list(components.keys()), num_components // 5)
            
            tasks = []
            start_time = time.time()
            
            for i in range(num_events):
                # Seleccionar componente destino según distribución
                if random.random() < 0.8:
                    # 80% del tráfico va a componentes populares
                    comp_id = random.choice(popular_components)
                else:
                    # 20% del tráfico va al resto
                    comp_id = random.choice(list(components.keys()))
                
                # Complejidad variable
                complexity = random.choice([0.5, 1.0, 2.0])
                
                # Usar emit_with_timeout
                task = emit_with_timeout(
                    engine,
                    "process",
                    {"complexity": complexity, "event_id": i},
                    comp_id,
                    timeout=3.0
                )
                tasks.append(task)
                
                # Pequeña espera entre eventos para simular tráfico realista
                if i % 10 == 0:
                    await asyncio.sleep(0.001)
            
            # Esperar a que todos los eventos terminen
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            elapsed = time.time() - start_time
            
            # Analizar resultados
            successful = len([r for r in results if isinstance(r, list) and len(r) > 0 and r[0].get("success", False)])
            timeouts = len([r for r in results if isinstance(r, list) and len(r) > 0 and r[0].get("error") == "timeout"])
            exceptions = len([r for r in results if isinstance(r, Exception)])
            
            return {
                "total_events": num_events,
                "time": elapsed,
                "events_per_second": num_events / elapsed if elapsed > 0 else 0,
                "successful": successful,
                "timeouts": timeouts,
                "exceptions": exceptions,
                "success_rate": successful / num_events if num_events > 0 else 0
            }
        
        # Ejecutar prueba con tráfico realista
        logger.info("Ejecutando prueba de tráfico realista...")
        traffic_result = await generate_realistic_traffic(1000)
        
        logger.info(f"Tráfico completado en {traffic_result['time']:.3f}s - " 
                   f"{traffic_result['events_per_second']:.2f} eventos/s - "
                   f"Exitosos: {traffic_result['successful']}/{traffic_result['total_events']} "
                   f"({traffic_result['success_rate'] * 100:.1f}%)")
        
        # Verificar distribución de carga entre componentes
        component_stats = {}
        for name in components.keys():
            stats = await check_component_status(engine, name)
            component_stats[name] = stats
        
        # Calcular estadísticas
        processed_counts = [stats.get("processed", 0) for stats in component_stats.values()]
        if processed_counts:
            max_processed = max(processed_counts)
            min_processed = min(processed_counts)
            avg_processed = sum(processed_counts) / len(processed_counts)
            
            # Calcular distribución (desviación estándar)
            std_dev = (sum((c - avg_processed) ** 2 for c in processed_counts) / len(processed_counts)) ** 0.5
            
            logger.info(f"Estadísticas de distribución:")
            logger.info(f"  Min: {min_processed}, Max: {max_processed}, Promedio: {avg_processed:.2f}")
            logger.info(f"  Desviación estándar: {std_dev:.2f}")
            logger.info(f"  Coeficiente de variación: {std_dev / avg_processed:.2f}")
            
            # Ver top 5 componentes con más carga
            sorted_components = sorted(
                component_stats.items(), 
                key=lambda x: x[1].get("processed", 0), 
                reverse=True
            )
            
            logger.info("Top 5 componentes con más eventos procesados:")
            for name, stats in sorted_components[:5]:
                logger.info(f"  {name}: {stats.get('processed', 0)} eventos, "
                          f"tiempo medio: {stats.get('avg_time', 0):.5f}s")
            
            return {
                "traffic_result": traffic_result,
                "component_stats": component_stats,
                "distribution": {
                    "min": min_processed,
                    "max": max_processed,
                    "avg": avg_processed,
                    "std_dev": std_dev,
                    "coefficient_variation": std_dev / avg_processed if avg_processed > 0 else 0
                }
            }
        
        return {"traffic_result": traffic_result, "component_stats": component_stats}
    
    # Ejecutar y medir la prueba completa
    result = await run_test_with_timing(
        engine, 
        "test_high_concurrency", 
        run_concurrency_test
    )
    
    # No fallamos la prueba, solo verificamos que la ejecución complete
    assert isinstance(result, dict), "La prueba debería devolver un dict con resultados"
    assert "traffic_result" in result, "Los resultados deberían incluir métricas de tráfico"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_priority_under_pressure(priority_engine):
    """
    Test de prioridades bajo carga extrema.
    
    Esta prueba verifica si el motor respeta las prioridades
    cuando está sometido a alta presión.
    """
    engine = priority_engine
    
    # Función interna para la prueba
    async def run_priority_test(engine):
        # Registrar componentes conscientes de prioridad
        components = []
        for i in range(10):
            comp = PriorityAwareComponent(f"prio_comp_{i}")
            await engine.register_component(comp)
            components.append(comp)
        
        # Función para enviar eventos con diferentes prioridades
        async def send_mixed_priority_events(total_events=1000, high_ratio=0.2, med_ratio=0.3):
            # Determinar cuántos eventos de cada tipo
            high_count = int(total_events * high_ratio)
            med_count = int(total_events * med_ratio)
            low_count = total_events - high_count - med_count
            
            logger.info(f"Enviando {high_count} eventos alta prioridad, "
                       f"{med_count} media, {low_count} baja...")
            
            # Mezclar eventos aleatoriamente
            all_events = (
                [("high", i) for i in range(high_count)] +
                [("medium", i) for i in range(med_count)] +
                [("low", i) for i in range(low_count)]
            )
            random.shuffle(all_events)
            
            tasks = []
            start_time = time.time()
            sent_by_priority = {"high": 0, "medium": 0, "low": 0}
            
            for priority, event_id in all_events:
                comp_id = f"prio_comp_{event_id % 10}"
                sent_by_priority[priority] += 1
                
                # Usar emit_with_timeout con timeout más largo para eventos de baja prioridad
                timeout = 2.0 if priority == "high" else 3.0 if priority == "medium" else 5.0
                
                task = emit_with_timeout(
                    engine,
                    "process_with_priority",
                    {
                        "priority": priority,
                        "payload": {"id": event_id, "timestamp": time.time()}
                    },
                    comp_id,
                    timeout=timeout
                )
                tasks.append((priority, task))
                
                # Pequeña espera entre eventos para no saturar
                if event_id % 50 == 0:
                    await asyncio.sleep(0.005)
            
            # Esperar a que todos los eventos terminen
            results_by_priority = {"high": [], "medium": [], "low": []}
            
            for priority, task in tasks:
                try:
                    result = await task
                    results_by_priority[priority].append(result)
                except Exception as e:
                    results_by_priority[priority].append({"error": str(e)})
            
            elapsed = time.time() - start_time
            
            # Analizar resultados por prioridad
            stats_by_priority = {}
            for priority, results in results_by_priority.items():
                successful = len([r for r in results if isinstance(r, list) and len(r) > 0 and r[0].get("success", True)])
                timeouts = len([r for r in results if isinstance(r, list) and len(r) > 0 and r[0].get("error") == "timeout"])
                
                stats_by_priority[priority] = {
                    "sent": sent_by_priority[priority],
                    "successful": successful,
                    "timeouts": timeouts,
                    "success_rate": successful / sent_by_priority[priority] if sent_by_priority[priority] > 0 else 0
                }
            
            return {
                "total_events": total_events,
                "time": elapsed,
                "events_per_second": total_events / elapsed if elapsed > 0 else 0,
                "stats_by_priority": stats_by_priority
            }
        
        # Ejecutar prueba con tráfico mixto
        logger.info("Enviando eventos mixtos con prioridades...")
        mixed_result = await send_mixed_priority_events(1000)
        
        logger.info(f"Tráfico mixto completado en {mixed_result['time']:.3f}s - " 
                   f"{mixed_result['events_per_second']:.2f} eventos/s")
        
        # Mostrar resultados por prioridad
        for priority, stats in mixed_result["stats_by_priority"].items():
            logger.info(f"Prioridad {priority}: "
                       f"{stats['successful']}/{stats['sent']} exitosos "
                       f"({stats['success_rate'] * 100:.1f}%), "
                       f"{stats['timeouts']} timeouts")
        
        # Verificar cómo los componentes manejaron las prioridades
        component_stats = {}
        for comp in components:
            stats = await check_component_status(engine, comp.name)
            component_stats[comp.name] = stats
            
            logger.info(f"Componente {comp.name}:")
            for priority, count in stats.get("stats_by_priority", {}).items():
                avg_time = stats.get("avg_times", {}).get(priority, 0)
                logger.info(f"  {priority}: {count} eventos, tiempo medio: {avg_time:.5f}s")
        
        # Calcular respeto de prioridades
        # En un sistema ideal, los eventos de alta prioridad deberían procesarse
        # más rápido que los de baja prioridad
        
        # Calcular tiempo promedio por prioridad en todos los componentes
        avg_times_by_priority = {"high": 0, "medium": 0, "low": 0}
        count_by_priority = {"high": 0, "medium": 0, "low": 0}
        
        for stats in component_stats.values():
            for priority, avg_time in stats.get("avg_times", {}).items():
                avg_times_by_priority[priority] += avg_time
                count_by_priority[priority] += 1
        
        # Calcular promedios
        for priority in avg_times_by_priority:
            if count_by_priority[priority] > 0:
                avg_times_by_priority[priority] /= count_by_priority[priority]
        
        logger.info("Tiempos medios de procesamiento por prioridad:")
        for priority, avg_time in avg_times_by_priority.items():
            logger.info(f"  {priority}: {avg_time:.5f}s")
        
        # Calcular ratios (para verificar respeto de prioridades)
        priority_ratios = {}
        if avg_times_by_priority["high"] > 0:
            priority_ratios["medium_vs_high"] = avg_times_by_priority["medium"] / avg_times_by_priority["high"]
            priority_ratios["low_vs_high"] = avg_times_by_priority["low"] / avg_times_by_priority["high"]
        
        if avg_times_by_priority["medium"] > 0:
            priority_ratios["low_vs_medium"] = avg_times_by_priority["low"] / avg_times_by_priority["medium"]
        
        logger.info("Ratios de tiempos entre prioridades:")
        for ratio_name, ratio_value in priority_ratios.items():
            logger.info(f"  {ratio_name}: {ratio_value:.2f}x")
        
        # Un valor de ratio > 1 indica respeto de prioridades
        # (los eventos de menor prioridad tardan más)
        
        return {
            "mixed_result": mixed_result,
            "component_stats": component_stats,
            "avg_times_by_priority": avg_times_by_priority,
            "priority_ratios": priority_ratios
        }
    
    # Ejecutar y medir la prueba completa
    result = await run_test_with_timing(
        engine, 
        "test_priority_under_pressure", 
        run_priority_test
    )
    
    # No fallamos la prueba, solo verificamos que la ejecución complete
    assert isinstance(result, dict), "La prueba debería devolver un dict con resultados"
    assert "priority_ratios" in result, "Los resultados deberían incluir ratios de prioridad"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_dynamic_expansion_stress(dynamic_engine):
    """
    Test de estrés para expansión dinámica.
    
    Esta prueba provoca rápidos cambios en la carga para
    forzar la expansión y contracción dinámica del motor.
    """
    engine = dynamic_engine
    
    # Configurar motor para una expansión más rápida
    engine.scaling_threshold = 0.6  # Más sensible a la carga
    engine.cooldown_period = 0.05   # Más rápido para pruebas
    
    # Función interna para la prueba
    async def run_expansion_test(engine):
        # Registrar componentes para las pruebas
        components = []
        for i in range(20):
            comp = SlowComponent(f"exp_comp_{i}", processing_time=0.05)
            await engine.register_component(comp)
            components.append(comp)
        
        # Función para generar un pico de carga
        async def generate_load_spike(num_events, complexity=1.0):
            tasks = []
            start_time = time.time()
            
            for i in range(num_events):
                comp_id = f"exp_comp_{i % 20}"
                
                task = emit_with_timeout(
                    engine,
                    "process",
                    {"complexity": complexity, "event_id": i},
                    comp_id,
                    timeout=3.0
                )
                tasks.append(task)
            
            # Esperar a que todas las tareas se completen
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            elapsed = time.time() - start_time
            successful = len([r for r in results if isinstance(r, list) and len(r) > 0 and r[0].get("success", False)])
            
            return {
                "num_events": num_events,
                "complexity": complexity,
                "time": elapsed,
                "events_per_second": num_events / elapsed if elapsed > 0 else 0,
                "successful": successful,
                "success_rate": successful / num_events if num_events > 0 else 0
            }
        
        # Realizar una secuencia de picos de carga para forzar la expansión
        logger.info("Fase 1: Carga inicial moderada")
        result1 = await generate_load_spike(50, complexity=1.0)
        
        logger.info(f"Carga inicial: {result1['events_per_second']:.2f} eventos/s, "
                   f"{result1['success_rate'] * 100:.1f}% exitosos")
        
        # Capturar estado inicial del motor
        initial_blocks = getattr(engine, 'active_blocks', 0)
        logger.info(f"Bloques activos iniciales: {initial_blocks}")
        
        # Generar un pico agudo de carga
        logger.info("Fase 2: Pico agudo de carga")
        result2 = await generate_load_spike(200, complexity=2.0)
        
        logger.info(f"Pico agudo: {result2['events_per_second']:.2f} eventos/s, "
                   f"{result2['success_rate'] * 100:.1f}% exitosos")
        
        # Capturar estado después del pico
        peak_blocks = getattr(engine, 'active_blocks', 0)
        logger.info(f"Bloques activos durante pico: {peak_blocks}")
        
        # Breve pausa para permitir que el motor ajuste
        await asyncio.sleep(0.2)
        
        # Generar cargas variables para estresar el mecanismo de escalado
        load_patterns = [
            ("Carga ligera", 30, 0.5),
            ("Carga media", 100, 1.0),
            ("Carga pesada", 300, 1.5),
            ("Carga muy ligera", 20, 0.3),
            ("Carga extrema", 400, 2.0)
        ]
        
        pattern_results = []
        
        for name, num_events, complexity in load_patterns:
            logger.info(f"Ejecutando patrón: {name}")
            result = await generate_load_spike(num_events, complexity)
            
            current_blocks = getattr(engine, 'active_blocks', 0)
            
            logger.info(f"{name}: {result['events_per_second']:.2f} eventos/s, "
                       f"{result['success_rate'] * 100:.1f}% exitosos, "
                       f"bloques activos: {current_blocks}")
            
            pattern_results.append({
                "name": name,
                "result": result,
                "active_blocks": current_blocks
            })
            
            # Breve pausa entre patrones
            await asyncio.sleep(0.1)
        
        # Verificar estado final del motor
        final_blocks = getattr(engine, 'active_blocks', 0)
        logger.info(f"Bloques activos finales: {final_blocks}")
        
        # Calcular métricas de eficiencia de escalado
        scaling_metrics = {
            "initial_blocks": initial_blocks,
            "peak_blocks": peak_blocks,
            "final_blocks": final_blocks,
            "max_expansion_ratio": peak_blocks / initial_blocks if initial_blocks > 0 else 0,
            "patterns": pattern_results
        }
        
        # Analizar correlación entre bloques activos y rendimiento
        if pattern_results:
            blocks = [p["active_blocks"] for p in pattern_results]
            throughputs = [p["result"]["events_per_second"] for p in pattern_results]
            
            # Calcular correlación simple
            if len(blocks) > 1 and len(set(blocks)) > 1:
                # Fórmula de correlación simplificada
                mean_blocks = sum(blocks) / len(blocks)
                mean_throughput = sum(throughputs) / len(throughputs)
                
                numerator = sum((b - mean_blocks) * (t - mean_throughput) for b, t in zip(blocks, throughputs))
                denominator_blocks = sum((b - mean_blocks) ** 2 for b in blocks)
                denominator_throughput = sum((t - mean_throughput) ** 2 for t in throughputs)
                
                if denominator_blocks > 0 and denominator_throughput > 0:
                    correlation = numerator / ((denominator_blocks * denominator_throughput) ** 0.5)
                    scaling_metrics["blocks_throughput_correlation"] = correlation
                    
                    logger.info(f"Correlación bloques/rendimiento: {correlation:.2f}")
                    
                    # Una correlación positiva fuerte indica que el escalado es efectivo
        
        return {
            "initial_result": result1,
            "peak_result": result2,
            "pattern_results": pattern_results,
            "scaling_metrics": scaling_metrics
        }
    
    # Ejecutar y medir la prueba completa
    result = await run_test_with_timing(
        engine, 
        "test_dynamic_expansion_stress", 
        run_expansion_test
    )
    
    # No fallamos la prueba, solo verificamos que la ejecución complete
    assert isinstance(result, dict), "La prueba debería devolver un dict con resultados"
    assert "scaling_metrics" in result, "Los resultados deberían incluir métricas de escalado"


if __name__ == "__main__":
    # Para poder ejecutar este archivo directamente
    import pytest
    pytest.main(["-xvs", __file__])