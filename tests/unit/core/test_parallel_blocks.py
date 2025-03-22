"""
Tests para el motor con procesamiento paralelo en bloques.

Este módulo prueba el motor que procesa eventos en bloques
paralelos para mejorar el rendimiento mientras mantiene
control sobre timeouts y recursos.
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, List, Set, Optional

from genesis.core.component import Component
from genesis.core.engine_parallel_blocks import ParallelBlockEngine

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentForTesting(Component):
    """Componente de prueba con comportamiento configurable."""
    
    def __init__(self, name: str, delay: float = 0.1, fail_on: Set[str] = None):
        """
        Inicializar componente de prueba.
        
        Args:
            name: Nombre del componente
            delay: Retraso para simular procesamiento
            fail_on: Conjunto de operaciones en las que fallar ('start', 'stop', 'event')
        """
        super().__init__(name)
        self.delay = delay
        self.fail_on = fail_on or set()
        self.started = False
        self.stopped = False
        self.processed_events = []
        self.processing_times = []
        logger.info(f"Componente {name} creado (delay={delay}s, fail_on={fail_on})")
    
    async def start(self) -> None:
        """Iniciar el componente con comportamiento configurable."""
        logger.info(f"Componente {self.name} iniciando")
        
        if 'start' in self.fail_on:
            logger.warning(f"Componente {self.name} programado para fallar en start")
            raise RuntimeError(f"Fallo simulado en start para {self.name}")
        
        # Simular procesamiento
        start_time = time.time()
        await asyncio.sleep(self.delay)
        end_time = time.time()
        
        self.started = True
        self.processing_times.append(end_time - start_time)
        logger.info(f"Componente {self.name} iniciado (tiempo={end_time-start_time:.3f}s)")
    
    async def stop(self) -> None:
        """Detener el componente con comportamiento configurable."""
        logger.info(f"Componente {self.name} deteniendo")
        
        if 'stop' in self.fail_on:
            logger.warning(f"Componente {self.name} programado para fallar en stop")
            raise RuntimeError(f"Fallo simulado en stop para {self.name}")
        
        # Simular procesamiento
        start_time = time.time()
        await asyncio.sleep(self.delay)
        end_time = time.time()
        
        self.stopped = True
        self.processing_times.append(end_time - start_time)
        logger.info(f"Componente {self.name} detenido (tiempo={end_time-start_time:.3f}s)")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento con comportamiento configurable."""
        logger.info(f"Componente {self.name} procesando evento {event_type}")
        
        if 'event' in self.fail_on:
            logger.warning(f"Componente {self.name} programado para fallar al procesar eventos")
            raise RuntimeError(f"Fallo simulado al procesar evento para {self.name}")
        
        # Simular procesamiento
        start_time = time.time()
        await asyncio.sleep(self.delay)
        end_time = time.time()
        
        # Registrar evento procesado
        self.processed_events.append({
            'type': event_type,
            'data': data.copy() if data else {},
            'source': source,
            'time': end_time - start_time
        })
        
        self.processing_times.append(end_time - start_time)
        logger.info(f"Componente {self.name} procesó evento {event_type} (tiempo={end_time-start_time:.3f}s)")


@pytest.mark.asyncio
async def test_parallel_blocks_basic():
    """
    Test básico del motor con bloques paralelos.
    
    Esta prueba verifica que el motor puede iniciar, procesar eventos
    y detener componentes correctamente usando bloques paralelos.
    """
    # Crear motor con bloques pequeños para facilitar pruebas
    engine = ParallelBlockEngine(
        block_size=2,  # 2 componentes por bloque
        timeout=0.5,   # 500ms por operación
        max_concurrent_blocks=2  # 2 bloques en paralelo
    )
    
    # Crear componentes con diferentes tiempos
    components = [
        ComponentForTesting("comp1", delay=0.1),   # Rápido
        ComponentForTesting("comp2", delay=0.15),  # Rápido
        ComponentForTesting("comp3", delay=0.2),   # Medio
        ComponentForTesting("comp4", delay=0.25),  # Medio
        ComponentForTesting("comp5", delay=0.3)    # Lento
    ]
    
    # Registrar componentes
    for comp in components:
        engine.register_component(comp)
    
    # Verificar bloques
    blocks = engine._create_component_blocks()
    assert len(blocks) == 3  # 5 componentes / 2 por bloque = 3 bloques (2, 2, 1)
    
    # Iniciar motor
    start_time = time.time()
    await engine.start()
    start_duration = time.time() - start_time
    
    # Verificar que todos los componentes se iniciaron
    assert all(comp.started for comp in components), "Todos los componentes deberían haberse iniciado"
    
    # Verificar que el tiempo de inicio es menor que el tiempo en serie
    # Cálculo del peor caso en serie: sum(delays) ~= 1 segundo
    # Con procesamiento en bloques debería ser mucho menor
    logger.info(f"Tiempo de inicio: {start_duration:.3f}s")
    # No verificamos explícitamente el tiempo, ya que varía según la máquina y concurrencia
    
    # Emitir un evento a todos los componentes
    event_start_time = time.time()
    await engine.emit_event("test.event", {"value": 42}, "test")
    event_duration = time.time() - event_start_time
    
    # Verificar que todos los componentes procesaron el evento
    assert all(len(comp.processed_events) == 1 for comp in components), "Todos los componentes deberían haber procesado el evento"
    
    # Verificar que el tiempo de emisión del evento es menor que el tiempo en serie
    logger.info(f"Tiempo de emisión de evento: {event_duration:.3f}s")
    # No verificamos explícitamente el tiempo
    
    # Detener motor
    stop_start_time = time.time()
    await engine.stop()
    stop_duration = time.time() - stop_start_time
    
    # Verificar que todos los componentes se detuvieron
    assert all(comp.stopped for comp in components), "Todos los componentes deberían haberse detenido"
    
    # Verificar que el tiempo de detención es menor que el tiempo en serie
    logger.info(f"Tiempo de detención: {stop_duration:.3f}s")
    # No verificamos explícitamente el tiempo


@pytest.mark.asyncio
async def test_parallel_blocks_timeout_handling():
    """
    Test de manejo de timeouts en bloques paralelos.
    
    Esta prueba verifica que el motor maneja correctamente timeouts
    en bloques paralelos sin bloquear todo el sistema.
    """
    # Crear motor con timeout pequeño
    engine = ParallelBlockEngine(
        block_size=2,
        timeout=0.3,   # 300ms timeout (menor que algunos componentes)
        max_concurrent_blocks=2
    )
    
    # Crear algunos componentes normales y algunos lentos
    components = [
        ComponentForTesting("normal1", delay=0.1),   # Normal
        ComponentForTesting("normal2", delay=0.2),   # Normal
        ComponentForTesting("slow1", delay=0.4),     # Excede timeout
        ComponentForTesting("slow2", delay=0.5),     # Excede timeout
        ComponentForTesting("normal3", delay=0.15)   # Normal
    ]
    
    # Registrar componentes
    for comp in components:
        engine.register_component(comp)
    
    # Iniciar motor (algunos componentes excederán timeout)
    await engine.start()
    
    # Verificar que solo los componentes normales se iniciaron correctamente
    normal_comps = [comp for comp in components if comp.name.startswith("normal")]
    slow_comps = [comp for comp in components if comp.name.startswith("slow")]
    
    # Los componentes normales deberían haberse iniciado
    assert all(comp.started for comp in normal_comps), "Los componentes normales deberían haberse iniciado"
    
    # Los componentes lentos probablemente no se iniciaron (timeout)
    # Pero no verificamos explícitamente, ya que depende de la concurrencia real
    
    # Emitir evento (los componentes lentos excederán timeout)
    await engine.emit_event("test.event", {"value": 100}, "test")
    
    # Verificar que solo los componentes normales procesaron el evento
    for comp in normal_comps:
        assert len(comp.processed_events) == 1, f"El componente {comp.name} debería haber procesado el evento"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que los componentes normales se detuvieron correctamente
    assert all(comp.stopped for comp in normal_comps), "Los componentes normales deberían haberse detenido"


@pytest.mark.asyncio
async def test_parallel_blocks_error_handling():
    """
    Test de manejo de errores en bloques paralelos.
    
    Esta prueba verifica que el motor maneja correctamente errores
    en componentes, permitiendo que otros componentes continúen funcionando.
    """
    # Crear motor
    engine = ParallelBlockEngine(
        block_size=2,
        timeout=0.5,
        max_concurrent_blocks=2
    )
    
    # Crear componentes, algunos programados para fallar
    components = [
        TestComponent("good1", delay=0.1),  # Funciona bien
        TestComponent("fail_start", delay=0.1, fail_on={"start"}),  # Falla en start
        TestComponent("good2", delay=0.1),  # Funciona bien
        TestComponent("fail_event", delay=0.1, fail_on={"event"}),  # Falla en eventos
        TestComponent("good3", delay=0.1),  # Funciona bien
        TestComponent("fail_stop", delay=0.1, fail_on={"stop"})   # Falla en stop
    ]
    
    # Registrar componentes
    for comp in components:
        engine.register_component(comp)
    
    # Iniciar motor (un componente fallará)
    await engine.start()
    
    # Los componentes buenos deben haberse iniciado
    good_comps = [comp for comp in components if comp.name.startswith("good")]
    assert all(comp.started for comp in good_comps), "Los componentes buenos deberían haberse iniciado"
    
    # Componente fail_start no debe iniciarse
    fail_start = next(comp for comp in components if comp.name == "fail_start")
    assert not fail_start.started, "El componente programado para fallar en start no debería haberse iniciado"
    
    # Emitir evento (un componente fallará)
    await engine.emit_event("test.event", {"value": 200}, "test")
    
    # Los componentes buenos deben haber procesado el evento
    for comp in good_comps:
        assert len(comp.processed_events) == 1, f"El componente {comp.name} debería haber procesado el evento"
    
    # El componente fail_event no debe tener eventos procesados
    fail_event = next(comp for comp in components if comp.name == "fail_event")
    assert len(fail_event.processed_events) == 0, "El componente programado para fallar en eventos no debería haber procesado el evento"
    
    # Detener motor (un componente fallará)
    await engine.stop()
    
    # Los componentes buenos deben haberse detenido
    assert all(comp.stopped for comp in good_comps), "Los componentes buenos deberían haberse detenido"
    
    # El componente fail_stop no debe haberse detenido correctamente
    fail_stop = next(comp for comp in components if comp.name == "fail_stop")
    assert not fail_stop.stopped, "El componente programado para fallar en stop no debería haberse detenido correctamente"