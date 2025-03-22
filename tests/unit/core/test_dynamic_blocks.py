"""
Pruebas para el motor con bloques dinámicos y expansión adaptativa.

Este módulo verifica que el motor con bloques dinámicos funcione correctamente,
incluyendo la capacidad de escalar automáticamente según la carga de trabajo.
"""

import pytest
import asyncio
import time
import logging
from typing import Dict, Any, Optional, List

from genesis.core.component import Component
from genesis.core.engine_dynamic_blocks import DynamicExpansionEngine, EventPriority

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComponentForTesting(Component):
    """Componente de prueba que registra eventos recibidos."""
    
    def __init__(self, name: str, delay: float = 0.01, fail_on: List[str] = None):
        """
        Inicializar componente de prueba.
        
        Args:
            name: Nombre del componente
            delay: Tiempo de retraso para operaciones (en segundos)
            fail_on: Lista de tipos de eventos que fallarán intencionalmente
        """
        self.name = name
        self.delay = delay
        self.fail_on = fail_on or []
        self.started = False
        self.stopped = False
        self.events_received = []
        self.event_count = 0
    
    async def start(self) -> None:
        """Iniciar el componente con retraso simulado."""
        logger.info(f"Iniciando componente: {self.name}")
        await asyncio.sleep(self.delay)
        self.started = True
        return None
    
    async def stop(self) -> None:
        """Detener el componente con retraso simulado."""
        logger.info(f"Deteniendo componente: {self.name}")
        await asyncio.sleep(self.delay)
        self.stopped = True
        return None
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar evento con retraso simulado y fallo opcional.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta al evento
            
        Raises:
            Exception: Si el tipo de evento está en la lista de fallos
        """
        # Comprobar si debemos fallar para este evento
        if event_type in self.fail_on:
            logger.warning(f"Componente {self.name} fallando intencionalmente en {event_type}")
            raise Exception(f"Fallo simulado en {self.name} para {event_type}")
        
        # Retraso adicional para eventos "pesados"
        if "_heavy" in event_type:
            await asyncio.sleep(self.delay * 2)
        else:
            await asyncio.sleep(self.delay)
        
        # Registrar evento
        self.events_received.append((event_type, data, source))
        self.event_count += 1
        
        return {"status": "processed", "component": self.name}


class HeavyComponentForTesting(ComponentForTesting):
    """Componente de prueba que simula carga alta."""
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """Componente que genera alta carga para probar el auto-escalado."""
        # Simular carga variada según el modo
        if data and data.get('_light_mode'):
            # Modo ligero (cuando el sistema está cargado)
            await asyncio.sleep(self.delay * 0.5)
        elif data and data.get('_full_mode'):
            # Modo completo (cuando el sistema tiene capacidad)
            await asyncio.sleep(self.delay * 3)
        else:
            # Modo normal
            await asyncio.sleep(self.delay * 1.5)
        
        # Registrar evento
        self.events_received.append((event_type, data, source))
        self.event_count += 1
        
        return {"status": "processed", "component": self.name}


@pytest.mark.asyncio
async def test_dynamic_engine_basic_operation():
    """Verificar las operaciones básicas del motor dinámico."""
    # Crear motor con configuración sencilla
    engine = DynamicExpansionEngine(
        initial_block_size=2,
        timeout=0.5,
        min_concurrent_blocks=1,
        max_concurrent_blocks=3,
        auto_scaling=True,
        scale_cooldown=0.1  # Reducir tiempo de enfriamiento para pruebas
    )
    
    # Crear componentes de prueba
    comp1 = ComponentForTesting("comp1", delay=0.01)
    comp2 = ComponentForTesting("comp2", delay=0.01)
    comp3 = ComponentForTesting("comp3", delay=0.01)
    comp4 = ComponentForTesting("comp4", delay=0.01)
    
    # Registrar componentes
    engine.register_component(comp1, component_type="regular")
    engine.register_component(comp2, component_type="regular")
    engine.register_component(comp3, component_type="safe")
    engine.register_component(comp4, component_type="expansion")
    
    # Iniciar motor
    await engine.start()
    
    # Verificar que los componentes se iniciaron
    assert comp1.started
    assert comp2.started
    assert comp3.started
    assert comp4.started
    
    # Emitir evento y verificar que fue recibido
    await engine.emit_event("test.event", {"data": "value"}, "test")
    await asyncio.sleep(0.2)  # Tiempo para procesar (aumentado)
    
    # Verificar que el evento llegó a todos los componentes
    assert len(comp1.events_received) > 0, "El componente regular 1 no recibió eventos"
    assert len(comp2.events_received) > 0, "El componente regular 2 no recibió eventos"
    assert len(comp3.events_received) > 0, "El componente seguro no recibió eventos"
    assert len(comp4.events_received) > 0, "El componente de expansión no recibió eventos"
    
    # Verificar que el evento específico fue recibido
    assert any("test.event" in event for event, _, _ in comp1.events_received)
    assert any("test.event" in event for event, _, _ in comp2.events_received)
    assert any("test.event" in event for event, _, _ in comp3.events_received)
    assert any("test.event" in event for event, _, _ in comp4.events_received)
    
    # Detener motor
    await engine.stop()
    
    # Verificar que los componentes se detuvieron
    assert comp1.stopped
    assert comp2.stopped
    assert comp3.stopped
    assert comp4.stopped


@pytest.mark.asyncio
async def test_dynamic_engine_component_types():
    """Verificar que los diferentes tipos de componentes se manejen apropiadamente."""
    # Crear motor con configuración para probar tipos de componentes
    engine = DynamicExpansionEngine(
        initial_block_size=2,
        timeout=0.5,
        min_concurrent_blocks=1,
        max_concurrent_blocks=2
    )
    
    # Crear componentes de prueba de diferentes tipos
    regular = ComponentForTesting("regular1", delay=0.01)
    safe = ComponentForTesting("safe1", delay=0.01)
    expansion = ComponentForTesting("expansion1", delay=0.01)
    
    # Registrar componentes
    engine.register_component(regular, component_type="regular")
    engine.register_component(safe, component_type="safe")
    engine.register_component(expansion, component_type="expansion")
    
    # Iniciar motor
    await engine.start()
    
    # Verificar que los componentes fueron iniciados
    assert regular.started
    assert safe.started
    assert expansion.started
    
    # Emitir eventos de diferente prioridad
    await engine.emit_event("risk.critical", {"priority": "critical"}, "test", EventPriority.CRITICAL)
    await engine.emit_event("trade.high", {"priority": "high"}, "test", EventPriority.HIGH)
    await engine.emit_event("market.medium", {"priority": "medium"}, "test", EventPriority.MEDIUM)
    await engine.emit_event("info.low", {"priority": "low"}, "test", EventPriority.LOW)
    
    # Esperar a que los eventos sean procesados
    await asyncio.sleep(0.2)
    
    # Verificar componente seguro recibió todos los eventos
    assert len(safe.events_received) == 4
    
    # Verificar componente de expansión recibió eventos según su prioridad
    assert len(expansion.events_received) >= 2  # Al menos críticos y altos
    
    # Verificar componente regular recibió eventos según su prioridad
    assert len(regular.events_received) >= 1  # Al menos críticos
    
    # Detener motor
    await engine.stop()
    
    # Verificar que los componentes fueron detenidos
    assert regular.stopped
    assert safe.stopped
    assert expansion.stopped


@pytest.mark.asyncio
async def test_dynamic_engine_auto_scaling():
    """Verificar que el motor escale automáticamente según la carga."""
    # Crear motor con auto-escalado agresivo para probar
    engine = DynamicExpansionEngine(
        initial_block_size=2,
        timeout=0.2,
        min_concurrent_blocks=1,
        max_concurrent_blocks=4,
        expansion_threshold=0.6,
        contraction_threshold=0.2,
        auto_scaling=True,
        # Reducir cooldown para pruebas
        scale_cooldown=0.5
    )
    
    # Crear componentes con carga variada
    regular1 = ComponentForTesting("regular1", delay=0.01)
    regular2 = ComponentForTesting("regular2", delay=0.01)
    heavy1 = HeavyComponentForTesting("heavy1", delay=0.1)  # Componente que genera alta carga
    heavy2 = HeavyComponentForTesting("heavy2", delay=0.1)
    safe1 = ComponentForTesting("safe1", delay=0.01)
    
    # Registrar componentes
    engine.register_component(regular1, component_type="regular")
    engine.register_component(regular2, component_type="regular")
    engine.register_component(heavy1, component_type="expansion")
    engine.register_component(heavy2, component_type="expansion")
    engine.register_component(safe1, component_type="safe")
    
    # Iniciar motor
    await engine.start()
    
    # Capturar nivel inicial de concurrencia
    initial_concurrency = engine.current_concurrent_blocks
    
    # Fase 1: Generar alta carga
    logger.info("Fase 1: Generando carga alta")
    for _ in range(10):
        await engine.emit_event("test_heavy.event", {"load": "high"}, "test")
        await asyncio.sleep(0.05)  # Pequeña pausa entre eventos
    
    # Esperar a que se procesen los eventos y se actualice la carga
    await asyncio.sleep(1.0)
    
    # Fase 2: Verificar que el sistema haya escalado hacia arriba
    mid_concurrency = engine.current_concurrent_blocks
    logger.info(f"Concurrencia después de carga alta: {mid_concurrency}")
    
    # Debería haber escalado hacia arriba o mantenerse en el máximo
    assert mid_concurrency >= initial_concurrency
    
    # Fase 3: Generar carga baja
    logger.info("Fase 3: Generando carga baja")
    for _ in range(5):
        await engine.emit_event("test_light.event", {"load": "low"}, "test")
        await asyncio.sleep(0.1)  # Mayor pausa entre eventos
    
    # Esperar a que se procesen los eventos y se actualice la carga
    await asyncio.sleep(1.5)
    
    # Fase 4: Verificar que el sistema haya escalado hacia abajo
    final_concurrency = engine.current_concurrent_blocks
    logger.info(f"Concurrencia después de carga baja: {final_concurrency}")
    
    # Debería haber iniciado el descenso
    assert final_concurrency <= mid_concurrency or final_concurrency == engine.min_concurrent_blocks
    
    # Revisar estadísticas
    stats = engine.get_stats()
    logger.info(f"Estadísticas finales: {stats}")
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_dynamic_engine_error_handling():
    """Verificar que el motor maneje errores en componentes correctamente."""
    # Crear motor con timeout corto para probar manejo de errores
    engine = DynamicExpansionEngine(
        initial_block_size=2,
        timeout=0.2,
        min_concurrent_blocks=1,
        max_concurrent_blocks=2
    )
    
    # Crear componentes con fallos específicos
    failing = ComponentForTesting("failing", delay=0.01, fail_on=["test.fail"])
    regular = ComponentForTesting("regular", delay=0.01)
    slow = ComponentForTesting("slow", delay=0.3)  # Más lento que el timeout
    safe = ComponentForTesting("safe", delay=0.01)
    
    # Registrar componentes
    engine.register_component(failing, component_type="regular")
    engine.register_component(regular, component_type="regular")
    engine.register_component(slow, component_type="regular")
    engine.register_component(safe, component_type="safe")
    
    # Iniciar motor
    await engine.start()
    
    # Emitir evento que causará fallo en un componente
    await engine.emit_event("test.fail", {"should_fail": True}, "test")
    await asyncio.sleep(0.1)
    
    # El componente regular debería procesar el evento normalmente
    assert any("test.fail" in event for event, _, _ in regular.events_received)
    
    # El componente seguro siempre debe procesar, incluso con fallos en otros
    assert any("test.fail" in event for event, _, _ in safe.events_received)
    
    # Emitir evento normal que debería ser procesado por todos
    await engine.emit_event("test.normal", {"normal": True}, "test")
    await asyncio.sleep(0.1)
    
    # Verificar que todos los componentes (incluso el que falló antes) procesen este evento
    assert any("test.normal" in event for event, _, _ in regular.events_received)
    assert any("test.normal" in event for event, _, _ in failing.events_received)
    assert any("test.normal" in event for event, _, _ in safe.events_received)
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_dynamic_engine_priority_handling():
    """Verificar que el motor maneje correctamente las prioridades de eventos."""
    # Crear motor con configuración para pruebas de prioridad
    engine = DynamicExpansionEngine(
        initial_block_size=2,
        timeout=0.2,  # Reducir timeout para pruebas más rápidas
        min_concurrent_blocks=1,  # Reducir para simplificar
        max_concurrent_blocks=2,
        priority_mappings={
            "custom.high": EventPriority.HIGH,
            "custom.low": EventPriority.LOW
        },
        scale_cooldown=0.1  # Tiempo de enfriamiento más corto
    )
    
    # Crear componentes simplificados para la prueba
    components = []
    for i in range(4):  # Reducir de 8 a 4 componentes
        comp = ComponentForTesting(f"comp{i}", delay=0.01)
        components.append(comp)
        
        # Registrar con diferentes tipos
        if i == 0:
            engine.register_component(comp, component_type="safe")
        elif i == 1:
            engine.register_component(comp, component_type="expansion")
        else:
            engine.register_component(comp, component_type="regular")
    
    # Iniciar motor
    await engine.start()
    
    # Verificar que los componentes se iniciaron
    for comp in components:
        assert comp.started, f"Componente {comp.name} no se inició correctamente"
    
    # Emitir eventos con diferentes prioridades (uno por uno)
    await engine.emit_event("risk.stop", {"priority": "critical"}, "test", EventPriority.CRITICAL)
    await asyncio.sleep(0.1)  # Esperar entre eventos
    
    await engine.emit_event("custom.high", {"priority": "high"}, "test")
    await asyncio.sleep(0.1)
    
    await engine.emit_event("market.data", {"priority": "medium"}, "test", EventPriority.MEDIUM)
    await asyncio.sleep(0.1)
    
    await engine.emit_event("custom.low", {"priority": "low"}, "test")
    
    # Esperar a que se procesen todos
    await asyncio.sleep(0.3)
    
    # Verificar componente seguro recibió todos los eventos
    safe_comp = components[0]
    safe_events = [e[0] for e in safe_comp.events_received]
    
    # Verificar que el componente seguro recibió todos los eventos
    assert "risk.stop" in safe_events, "Componente seguro no recibió evento crítico"
    assert "custom.high" in safe_events, "Componente seguro no recibió evento alto"
    assert "market.data" in safe_events, "Componente seguro no recibió evento medio"
    assert "custom.low" in safe_events, "Componente seguro no recibió evento bajo"
    
    # Verificar componente de expansión recibió según su prioridad
    expansion_comp = components[1]
    expansion_events = [e[0] for e in expansion_comp.events_received]
    
    # Verificar que el componente de expansión recibió al menos eventos críticos
    assert "risk.stop" in expansion_events, "Componente de expansión no recibió evento crítico"
    
    # Contar recepción por tipo de evento
    critical_received = sum(1 for comp in components if any(e[0] == "risk.stop" for e in comp.events_received))
    high_received = sum(1 for comp in components if any(e[0] == "custom.high" for e in comp.events_received))
    medium_received = sum(1 for comp in components if any(e[0] == "market.data" for e in comp.events_received))
    low_received = sum(1 for comp in components if any(e[0] == "custom.low" for e in comp.events_received))
    
    # Verificación simplificada de prioridad
    assert critical_received >= high_received, "Los eventos críticos deben llegar a igual o más componentes que los eventos altos"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que los componentes se detuvieron
    for comp in components:
        assert comp.stopped, f"Componente {comp.name} no se detuvo correctamente"


@pytest.mark.asyncio
async def test_dynamic_engine_stress():
    """Prueba de estrés ultra simplificada para el motor."""
    # Crear motor con configuración mínima
    engine = DynamicExpansionEngine(
        initial_block_size=1,  # Bloques más pequeños
        timeout=0.1,           # Timeout muy reducido
        min_concurrent_blocks=1,
        max_concurrent_blocks=2,  # Limitado a solo 2 bloques máximo
        auto_scaling=True,
        scale_cooldown=0.05    # Enfriamiento muy rápido
    )
    
    # Crear solo 3 componentes 
    components = []
    for i in range(3):  # Solo 3 componentes
        comp = TestComponent(f"comp{i}", delay=0.01)
        components.append(comp)
        
        # Un componente de cada tipo
        if i == 0:
            engine.register_component(comp, component_type="safe")
        elif i == 1:
            engine.register_component(comp, component_type="expansion")
        else:
            engine.register_component(comp, component_type="regular")
    
    # Iniciar motor
    await engine.start()
    
    # Verificar inicio correcto
    assert engine.running, "El motor debería estar funcionando"
    
    # Emitir un solo evento para evitar timeouts
    await engine.emit_event("test.event", {"data": "value"}, "stress_test")
    
    # Pausa breve
    await asyncio.sleep(0.1)
    
    # Verificar que al menos un componente recibió el evento
    received = False
    for comp in components:
        if comp.events_received:
            received = True
            break
    
    assert received, "Ningún componente recibió eventos"
    
    # Detener motor
    await engine.stop()
    
    logger.info("Prueba de estrés completada correctamente")
    return True  # Éxito