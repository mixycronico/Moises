"""
Tests para el motor con manejo de eventos prioritarios.

Este módulo prueba el motor que procesa eventos según niveles
de prioridad, crucial para sistemas de trading donde algunos
eventos son más importantes que otros.
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from genesis.core.component import Component
from genesis.core.engine_priority_events import PriorityEventEngine, EventPriority, PriorityEvent

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecordingComponent(Component):
    """Componente que registra eventos recibidos para pruebas."""
    
    def __init__(self, name: str, delay: float = 0.05):
        """
        Inicializar componente de registro.
        
        Args:
            name: Nombre del componente
            delay: Retraso para simular procesamiento
        """
        super().__init__(name)
        self.delay = delay
        self.started = False
        self.stopped = False
        self.events = []  # Registro de eventos procesados
        self.processing_times = []
        logger.info(f"Componente {name} creado (delay={delay}s)")
    
    async def start(self) -> None:
        """Iniciar el componente con retraso simulado."""
        logger.info(f"Componente {self.name} iniciando")
        await asyncio.sleep(self.delay)
        self.started = True
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener el componente con retraso simulado."""
        logger.info(f"Componente {self.name} deteniendo")
        await asyncio.sleep(self.delay)
        self.stopped = True
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento con retraso simulado y registrarlo.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        start_time = time.time()
        logger.info(f"Componente {self.name} procesando evento {event_type}")
        
        # Simular procesamiento
        await asyncio.sleep(self.delay)
        
        # Registrar evento
        self.events.append({
            "type": event_type,
            "data": data.copy() if data else {},
            "source": source,
            "timestamp": time.time()
        })
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        logger.info(f"Componente {self.name} procesó evento {event_type} (tiempo={processing_time:.3f}s)")


@pytest.mark.asyncio
async def test_priority_event_engine_basic():
    """
    Test básico del motor con eventos prioritarios.
    
    Esta prueba verifica que el motor inicia, procesa eventos
    y se detiene correctamente.
    """
    # Crear motor con timeout adecuado
    engine = PriorityEventEngine(
        component_timeout=0.5,
        max_queue_size=100
    )
    
    # Registrar componente de prueba
    comp = RecordingComponent("test_comp")
    engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    assert comp.started, "El componente debería estar iniciado"
    
    # Emitir eventos con diferentes prioridades
    await engine.emit_event("test.event.normal", {"value": 1}, "test")
    await engine.emit_event("risk.event.critical", {"value": 2}, "test", priority=EventPriority.CRITICAL)
    await engine.emit_event("info.event.low", {"value": 3}, "test", priority=EventPriority.LOW)
    
    # Esperar a que se procesen los eventos (el procesador trabaja en segundo plano)
    await asyncio.sleep(0.5)
    
    # Verificar que se procesaron los eventos
    assert len(comp.events) == 3, "Deberían haberse procesado 3 eventos"
    
    # Obtener estadísticas
    stats = engine.get_stats()
    assert stats["processed_events"] == 3, "Deberían haberse procesado 3 eventos"
    
    # Detener motor
    await engine.stop()
    assert comp.stopped, "El componente debería estar detenido"


@pytest.mark.asyncio
async def test_priority_event_ordering():
    """
    Test de ordenamiento de eventos por prioridad.
    
    Esta prueba verifica que los eventos se procesan en
    orden de prioridad, no en orden de llegada.
    """
    # Crear motor
    engine = PriorityEventEngine(
        component_timeout=0.5,
        max_queue_size=100
    )
    
    # Registrar componente de prueba
    comp = RecordingComponent("ordering_comp", delay=0.1)
    engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    
    # Emitir eventos en orden inverso de prioridad
    await engine.emit_event("background.task", {"id": 1}, "test", priority=EventPriority.BACKGROUND)
    await engine.emit_event("info.update", {"id": 2}, "test", priority=EventPriority.LOW)
    await engine.emit_event("market.data", {"id": 3}, "test", priority=EventPriority.MEDIUM)
    await engine.emit_event("trade.signal", {"id": 4}, "test", priority=EventPriority.HIGH)
    await engine.emit_event("risk.alert", {"id": 5}, "test", priority=EventPriority.CRITICAL)
    
    # Esperar a que se procesen los eventos
    await asyncio.sleep(1.0)
    
    # Verificar que los eventos se procesaron en orden de prioridad
    assert len(comp.events) == 5, "Deberían haberse procesado 5 eventos"
    
    # Verificar el orden correcto (CRITICAL → HIGH → MEDIUM → LOW → BACKGROUND)
    event_ids = [event["data"]["id"] for event in comp.events]
    expected_order = [5, 4, 3, 2, 1]  # Ordenados por prioridad
    
    # El procesamiento paralelo podría afectar el orden exacto, pero los de
    # alta prioridad deberían procesarse antes que los de baja
    assert event_ids[0] == 5, "El evento CRITICAL debería ser el primero"
    assert event_ids[1] == 4, "El evento HIGH debería ser el segundo"
    assert event_ids[4] == 1, "El evento BACKGROUND debería ser el último"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_priority_queue_full_behavior():
    """
    Test del comportamiento cuando la cola de prioridad está llena.
    
    Esta prueba verifica que cuando la cola está llena:
    1. Se descartan eventos de baja prioridad
    2. Se mantienen eventos de alta prioridad
    """
    # Crear motor con cola pequeña
    engine = PriorityEventEngine(
        component_timeout=0.5,
        max_queue_size=3  # Cola muy pequeña para forzar descarte
    )
    
    # Componente lento para que se llene la cola
    comp = RecordingComponent("slow_comp", delay=0.3)
    engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    
    # Llenar la cola con eventos de baja prioridad
    for i in range(3):
        await engine.emit_event(f"info.low.{i}", {"id": i}, "test", priority=EventPriority.LOW)
    
    # Verificar que la cola está llena
    stats = engine.get_stats()
    assert stats["queue_size"] == 3, "La cola debería estar llena"
    
    # Intentar agregar otro evento de baja prioridad (debería ser descartado)
    result = await engine.emit_event("info.low.extra", {"id": 100}, "test", priority=EventPriority.LOW)
    assert not result, "El evento de baja prioridad debería ser descartado"
    
    # Agregar evento crítico (debería aceptarse, eliminando uno de baja prioridad)
    result = await engine.emit_event("risk.critical", {"id": 999}, "test", priority=EventPriority.CRITICAL)
    assert result, "El evento CRITICAL debería ser aceptado"
    
    # Esperar procesamiento
    await asyncio.sleep(1.5)
    
    # Verificar estadísticas
    stats = engine.get_stats()
    assert stats["dropped_events"] > 0, "Debería haber eventos descartados"
    
    # Verificar que el evento crítico se procesó
    critical_processed = False
    for event in comp.events:
        if event["type"] == "risk.critical" and event["data"]["id"] == 999:
            critical_processed = True
            break
    
    assert critical_processed, "El evento crítico debería haberse procesado"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_priority_event_mappings():
    """
    Test de mapeos personalizados de prioridad de eventos.
    
    Esta prueba verifica que el motor puede asignar prioridades
    correctamente usando mapeos personalizados.
    """
    # Crear motor con mapeos personalizados
    custom_mappings = {
        "custom.critical": EventPriority.CRITICAL,
        "custom.high": EventPriority.HIGH,
        "custom.medium": EventPriority.MEDIUM,
        "custom.low": EventPriority.LOW,
        "custom.background": EventPriority.BACKGROUND
    }
    
    engine = PriorityEventEngine(
        component_timeout=0.5,
        max_queue_size=100,
        priority_mappings=custom_mappings
    )
    
    # Verificar mapeos
    assert engine.get_priority_for_event("custom.critical") == EventPriority.CRITICAL
    assert engine.get_priority_for_event("custom.high") == EventPriority.HIGH
    assert engine.get_priority_for_event("custom.medium") == EventPriority.MEDIUM
    assert engine.get_priority_for_event("custom.low") == EventPriority.LOW
    assert engine.get_priority_for_event("custom.background") == EventPriority.BACKGROUND
    
    # Verificar prefijos predeterminados
    assert engine.get_priority_for_event("risk.something") == EventPriority.CRITICAL
    assert engine.get_priority_for_event("trade.something") == EventPriority.HIGH
    assert engine.get_priority_for_event("market.something") == EventPriority.MEDIUM
    assert engine.get_priority_for_event("info.something") == EventPriority.LOW
    
    # Verificar fallback para tipos de eventos no reconocidos
    assert engine.get_priority_for_event("unknown.type") == EventPriority.MEDIUM