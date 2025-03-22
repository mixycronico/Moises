"""
Pruebas de carga y concurrencia para el core del sistema Genesis.

Este módulo contiene pruebas que verifican el comportamiento del sistema
bajo condiciones de alta carga, con múltiples eventos concurrentes
y componentes que procesan eventos simultáneamente.
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set
import random

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class LoadTestComponent(Component):
    """Componente para pruebas de carga que registra eventos recibidos."""
    
    def __init__(self, name: str, processing_time: float = 0, random_delay: bool = False,
                 fail_probability: float = 0):
        """
        Inicializar componente de prueba de carga.
        
        Args:
            name: Nombre del componente
            processing_time: Tiempo fijo (en segundos) que tarda en procesar cada evento
            random_delay: Si es True, añade un retardo aleatorio al tiempo de procesamiento
            fail_probability: Probabilidad (0-1) de que el procesamiento de un evento falle
        """
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.processing_time = processing_time
        self.random_delay = random_delay
        self.fail_probability = fail_probability
        self.event_types_received: Set[str] = set()
        self.total_events_processed = 0
        self.failed_events = 0
        self.processing_times: List[float] = []
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Iniciando componente de carga {self.name}")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Deteniendo componente de carga {self.name}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento con posible retardo y fallo.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
            
        Raises:
            Exception: Si el evento debe fallar según la probabilidad configurada
        """
        start_time = time.time()
        
        # Registrar información básica del evento
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "received_at": start_time
        })
        
        self.event_types_received.add(event_type)
        self.total_events_processed += 1
        
        # Simular tiempo de procesamiento
        delay = self.processing_time
        if self.random_delay:
            # Añadir entre 0 y 2x el tiempo base configurado
            delay += random.random() * self.processing_time * 2
        
        if delay > 0:
            await asyncio.sleep(delay)
        
        # Simular fallo aleatorio
        if random.random() < self.fail_probability:
            self.failed_events += 1
            raise Exception(f"Error simulado en {self.name} al procesar {event_type}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        self.processing_times.append(processing_time)
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas de rendimiento del componente.
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        return {
            "total_events": self.total_events_processed,
            "unique_event_types": len(self.event_types_received),
            "failed_events": self.failed_events,
            "avg_processing_time": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            "max_processing_time": max(self.processing_times) if self.processing_times else 0,
            "min_processing_time": min(self.processing_times) if self.processing_times else 0
        }


@pytest.mark.asyncio
async def test_high_load_single_component():
    """Prueba el sistema con alta carga de eventos enviados a un solo componente."""
    # Crear motor no bloqueante con timeout más largo para pruebas de carga
    engine = EngineNonBlocking(test_mode=True, )
    
    # Crear componente para pruebas de carga
    comp = LoadTestComponent("load_test", processing_time=0.001)
    
    # Registrar componente
    engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    
    # Número de eventos a enviar
    num_events = 1000
    
    # Tiempo de inicio
    start_time = time.time()
    
    # Enviar eventos de forma concurrente
    tasks = []
    for i in range(num_events):
        event_data = {"id": i, "value": f"test_{i}"}
        tasks.append(engine.emit_event(f"test_event_{i % 10}", event_data, "test_source"))
    
    # Esperar a que todos los eventos se envíen
    await asyncio.gather(*tasks)
    
    # Tiempo de finalización
    end_time = time.time()
    total_time = end_time - start_time
    
    # Esperar un poco más para asegurar que todos los eventos se han procesado
    await asyncio.sleep(0.5)
    
    # Verificar resultados
    metrics = comp.get_metrics()
    
    # Verificar que todos los eventos se procesaron
    assert metrics["total_events"] == num_events, f"No se procesaron todos los eventos. Esperados: {num_events}, Procesados: {metrics['total_events']}"
    
    # Verificar que se recibieron todos los tipos de eventos (10 tipos diferentes)
    assert metrics["unique_event_types"] == 10, f"No se recibieron todos los tipos de eventos. Esperados: 10, Recibidos: {metrics['unique_event_types']}"
    
    # Verificar que no hubo fallos
    assert metrics["failed_events"] == 0, f"Hubo eventos fallidos: {metrics['failed_events']}"
    
    # Log de rendimiento
    logger.info(f"Rendimiento para {num_events} eventos en un componente:")
    logger.info(f"Tiempo total: {total_time:.4f} segundos")
    logger.info(f"Eventos por segundo: {num_events / total_time:.2f}")
    logger.info(f"Tiempo promedio de procesamiento: {metrics['avg_processing_time']:.6f} segundos")
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_high_load_multiple_components():
    """Prueba el sistema con alta carga de eventos enviados a múltiples componentes."""
    # Crear motor no bloqueante con timeout más largo para pruebas de carga
    engine = EngineNonBlocking(test_mode=True, )
    
    # Número de componentes
    num_components = 5
    
    # Crear componentes con diferentes características
    components = [
        LoadTestComponent(f"comp_{i}", 
                         processing_time=0.001 * (i + 1),  # Tiempos de procesamiento diferentes
                         random_delay=(i % 2 == 0),  # Algunos con retardo aleatorio
                         fail_probability=0.01 if i == num_components - 1 else 0)  # El último tiene probabilidad de fallo
        for i in range(num_components)
    ]
    
    # Registrar componentes
    for comp in components:
        engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    
    # Número de eventos por componente
    events_per_component = 200
    total_events = events_per_component * num_components
    
    # Tiempo de inicio
    start_time = time.time()
    
    # Enviar eventos concurrentes a todos los componentes
    tasks = []
    for i in range(total_events):
        event_type = f"test_event_{i % 10}"  # 10 tipos diferentes de eventos
        event_data = {"id": i, "value": f"test_{i}", "timestamp": time.time()}
        tasks.append(engine.emit_event(event_type, event_data, "test_source"))
    
    # Esperar a que todos los eventos se envíen
    await asyncio.gather(*tasks)
    
    # Tiempo de finalización
    end_time = time.time()
    total_time = end_time - start_time
    
    # Esperar un poco más para asegurar que todos los eventos se han procesado
    await asyncio.sleep(1)
    
    # Recopilar métricas de todos los componentes
    all_metrics = {comp.name: comp.get_metrics() for comp in components}
    
    # Verificar resultados
    total_processed = sum(metrics["total_events"] for metrics in all_metrics.values())
    expected_total = total_events * num_components  # Cada evento va a todos los componentes
    
    # Debe haber al menos eventos_por_componente * num_componentes * num_componentes eventos procesados en total
    assert total_processed == expected_total, f"No se procesaron todos los eventos. Esperados: {expected_total}, Procesados: {total_processed}"
    
    # Verificar que todos los componentes recibieron eventos
    for comp_name, metrics in all_metrics.items():
        assert metrics["total_events"] > 0, f"El componente {comp_name} no recibió eventos"
        assert metrics["unique_event_types"] == 10, f"El componente {comp_name} no recibió todos los tipos de eventos"
    
    # Verificar que el componente con probabilidad de fallo tuvo al menos algunos fallos
    failing_comp = components[-1]
    assert failing_comp.failed_events > 0, f"El componente {failing_comp.name} debería haber tenido fallos"
    
    # Verificar que el sistema siguió funcionando a pesar de los fallos
    assert all_metrics[failing_comp.name]["total_events"] > failing_comp.failed_events, "El componente con fallos debería haber procesado algunos eventos correctamente"
    
    # Log de rendimiento
    logger.info(f"Rendimiento para {total_events} eventos en {num_components} componentes:")
    logger.info(f"Tiempo total: {total_time:.4f} segundos")
    logger.info(f"Eventos por segundo: {total_events / total_time:.2f}")
    logger.info(f"Total eventos procesados: {total_processed}")
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_burst_load():
    """Prueba el sistema con ráfagas de eventos de alta intensidad."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True, )
    
    # Crear componentes con diferentes velocidades de procesamiento
    fast_comp = LoadTestComponent("fast", processing_time=0.0005)
    medium_comp = LoadTestComponent("medium", processing_time=0.002)
    slow_comp = LoadTestComponent("slow", processing_time=0.01, random_delay=True)
    
    # Registrar componentes
    engine.register_component(fast_comp)
    engine.register_component(medium_comp)
    engine.register_component(slow_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Número de ráfagas
    num_bursts = 5
    # Eventos por ráfaga
    events_per_burst = 200
    # Tiempo entre ráfagas
    time_between_bursts = 0.5
    
    # Tiempo de inicio
    start_time = time.time()
    
    # Enviar ráfagas de eventos
    for burst in range(num_bursts):
        # Enviar eventos en ráfaga
        burst_tasks = []
        for i in range(events_per_burst):
            event_type = f"burst_{burst}_event_{i % 5}"
            event_data = {"burst": burst, "id": i, "timestamp": time.time()}
            burst_tasks.append(engine.emit_event(event_type, event_data, "burst_test"))
        
        # Esperar a que se envíen todos los eventos de la ráfaga
        await asyncio.gather(*burst_tasks)
        
        # Esperar antes de la siguiente ráfaga
        if burst < num_bursts - 1:
            await asyncio.sleep(time_between_bursts)
    
    # Tiempo después de enviar todas las ráfagas
    burst_end_time = time.time()
    burst_total_time = burst_end_time - start_time
    
    # Esperar a que se procesen todos los eventos
    # El tiempo de espera es mayor porque el componente lento puede tardar más
    await asyncio.sleep(2)
    
    # Tiempo final
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calcular el total de eventos enviados
    total_events = num_bursts * events_per_burst
    
    # Verificar que todos los componentes recibieron los eventos
    fast_metrics = fast_comp.get_metrics()
    medium_metrics = medium_comp.get_metrics()
    slow_metrics = slow_comp.get_metrics()
    
    # Cada componente debe haber recibido todos los eventos
    assert fast_metrics["total_events"] == total_events, f"El componente rápido no recibió todos los eventos: {fast_metrics['total_events']} vs {total_events}"
    assert medium_metrics["total_events"] == total_events, f"El componente medio no recibió todos los eventos: {medium_metrics['total_events']} vs {total_events}"
    assert slow_metrics["total_events"] == total_events, f"El componente lento no recibió todos los eventos: {slow_metrics['total_events']} vs {total_events}"
    
    # Verificar los tiempos de procesamiento relativos
    assert fast_metrics["avg_processing_time"] < medium_metrics["avg_processing_time"], "El componente rápido debería ser más rápido que el medio"
    assert medium_metrics["avg_processing_time"] < slow_metrics["avg_processing_time"], "El componente medio debería ser más rápido que el lento"
    
    # Logs de rendimiento
    logger.info(f"Rendimiento para {total_events} eventos en {num_bursts} ráfagas:")
    logger.info(f"Tiempo total de ráfagas: {burst_total_time:.4f} segundos")
    logger.info(f"Tiempo total hasta procesar todo: {total_time:.4f} segundos")
    logger.info(f"Ráfagas por segundo: {num_bursts / burst_total_time:.2f}")
    logger.info(f"Eventos por segundo (durante ráfagas): {total_events / burst_total_time:.2f}")
    logger.info(f"Eventos por segundo (total): {total_events / total_time:.2f}")
    
    # Log de tiempos de procesamiento por componente
    logger.info(f"Componente rápido - tiempo promedio: {fast_metrics['avg_processing_time']:.6f}s")
    logger.info(f"Componente medio - tiempo promedio: {medium_metrics['avg_processing_time']:.6f}s")
    logger.info(f"Componente lento - tiempo promedio: {slow_metrics['avg_processing_time']:.6f}s")
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_long_running_components():
    """Prueba componentes que tardan mucho tiempo en procesar eventos."""
    # Crear motor no bloqueante con timeout más largo
    engine = EngineNonBlocking(test_mode=True, component_timeout=2.0)
    
    # Crear componentes con diferentes tiempos de procesamiento
    fast_comp = LoadTestComponent("fast", processing_time=0.001)
    slow_comp = LoadTestComponent("slow", processing_time=1.5)  # Casi alcanza el timeout
    very_slow_comp = LoadTestComponent("very_slow", processing_time=3.0)  # Excede el timeout
    
    # Registrar componentes
    engine.register_component(fast_comp)
    engine.register_component(slow_comp)
    engine.register_component(very_slow_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Número de eventos a enviar
    num_events = 10
    
    # Enviar eventos
    for i in range(num_events):
        await engine.emit_event(f"test_event_{i}", {"id": i}, "timeout_test")
        # Esperar un poco entre eventos para no saturar el sistema
        await asyncio.sleep(0.1)
    
    # Esperar lo suficiente para que se procesen algunos eventos
    await asyncio.sleep(5)
    
    # Verificar resultados
    fast_metrics = fast_comp.get_metrics()
    slow_metrics = slow_comp.get_metrics()
    very_slow_metrics = very_slow_comp.get_metrics()
    
    # El componente rápido debe procesar todos los eventos
    assert fast_metrics["total_events"] == num_events, f"El componente rápido debería procesar todos los eventos: {fast_metrics['total_events']} vs {num_events}"
    
    # El componente lento debe procesar algunos eventos (depende del timeout exacto)
    assert slow_metrics["total_events"] > 0, f"El componente lento debería procesar al menos algunos eventos"
    
    # El componente muy lento puede no procesar todos los eventos debido al timeout
    logger.info(f"Componente muy lento procesó {very_slow_metrics['total_events']} de {num_events} eventos")
    
    # Detener motor
    await engine.stop()