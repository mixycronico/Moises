"""
Test especializado para detectar problemas de deadlock y race conditions en el EventBus.

Este módulo prueba específicamente escenarios que pueden causar bloqueos o condiciones de carrera
en el EventBus, enfocándose en comportamientos que podrían provocar fallos en cascada.
"""

import asyncio
import logging
import time
import pytest
from typing import Dict, Any, List, Optional

# Importar las clases necesarias del sistema
from genesis.core.engine_non_blocking import EngineNonBlocking
from genesis.core.component import Component
from genesis.core.event_bus import EventBus

# Importar utilidades para timeout
from tests.utils.timeout_helpers import (
    emit_with_timeout,
    check_component_status,
    run_test_with_timing,
    safe_get_response,
    cleanup_engine
)

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EventBusTestComponent(Component):
    """
    Componente especializado para probar el EventBus bajo diferentes condiciones.
    
    Este componente puede simular diferentes patrones de respuesta:
    - Retardo controlado
    - Bloqueo temporal
    - Generación de errores
    - Registro de eventos recibidos
    """
    
    def __init__(self, name: str, delay: float = 0.0, block_types: Optional[List[str]] = None, 
                 fail_types: Optional[List[str]] = None):
        """
        Inicializar componente de prueba.
        
        Args:
            name: Nombre del componente
            delay: Retardo en segundos para el procesamiento de eventos
            block_types: Lista de tipos de eventos que causarán bloqueo temporal
            fail_types: Lista de tipos de eventos que generarán excepciones
        """
        super().__init__(name)
        self.events_received = []
        self.delay = delay
        self.block_types = block_types or []
        self.fail_types = fail_types or []
        self.error_count = 0
        self.success_count = 0
        
    async def start(self) -> None:
        """Iniciar el componente."""
        logger.debug(f"Componente {self.name} iniciado")
        
    async def stop(self) -> None:
        """Detener el componente."""
        logger.debug(f"Componente {self.name} detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar un evento con diferentes comportamientos según su tipo.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta al evento, si corresponde
            
        Raises:
            Exception: Si el tipo de evento está en la lista de fail_types
        """
        # Registrar el evento recibido
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        })
        
        # Aplicar retardo si está configurado
        if self.delay > 0:
            await asyncio.sleep(self.delay)
            
        # Generar bloqueo temporal para tipos específicos
        if event_type in self.block_types:
            logger.warning(f"Componente {self.name} bloqueado temporalmente por evento {event_type}")
            # Bloqueo más largo que podría causar timeouts
            await asyncio.sleep(1.0)
            
        # Generar error para tipos específicos
        if event_type in self.fail_types:
            self.error_count += 1
            error_msg = f"Error simulado en componente {self.name} para evento {event_type}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        # Para eventos normales, incrementar contador y devolver respuesta
        self.success_count += 1
        return {
            "component": self.name,
            "event_type": event_type,
            "status": "processed",
            "processed_count": self.success_count,
            "error_count": self.error_count
        }

# Fixture para un motor con EventBus limpio
@pytest.fixture
async def engine_fixture():
    """Proporcionar un motor con EventBus limpio para cada test."""
    # Crear un motor en modo prueba para mayor control
    engine = EngineNonBlocking(test_mode=True)
    
    # Verificar que el event_bus está disponible
    assert engine.event_bus is not None, "EventBus no disponible en el motor"
    
    yield engine
    
    # Limpieza completa al finalizar
    await cleanup_engine(engine)
    logger.debug("Motor limpiado tras la prueba")

@pytest.mark.asyncio
async def test_eventbus_basic_operation(engine_fixture):
    """
    Prueba básica del funcionamiento del EventBus.
    
    Verifica que los eventos se publiquen correctamente y que los componentes
    reciban las respuestas esperadas.
    """
    engine = engine_fixture
    
    # Registrar componente simple
    comp = EventBusTestComponent("test_component")
    await engine.register_component(comp)
    
    # Verificar que el componente está registrado
    assert "test_component" in engine.components, "Componente no registrado correctamente"
    
    # Emitir evento simple con timeout
    response = await emit_with_timeout(
        engine, "test_event", {"data": "test"}, "test_source", timeout=1.0
    )
    
    # Verificar respuesta con manejo defensivo
    assert response is not None, "No hay respuesta del EventBus"
    assert isinstance(response, list), "Respuesta no es una lista"
    assert len(response) > 0, "Lista de respuestas vacía"
    
    # Verificar que el componente recibió el evento
    assert len(comp.events_received) == 1, "Componente no recibió el evento"
    assert comp.events_received[0]["type"] == "test_event", "Tipo de evento incorrecto"
    
    # Verificar respuesta del componente
    component_name = safe_get_response(response, "component", default=None)
    assert component_name == "test_component", "Respuesta no proviene del componente correcto"
    
    # Desregistrar componente
    await engine.unregister_component("test_component")
    
    # Verificar que se desregistró correctamente
    assert "test_component" not in engine.components, "Componente no desregistrado correctamente"

@pytest.mark.asyncio
async def test_eventbus_concurrent_events(engine_fixture):
    """
    Prueba el EventBus bajo carga concurrente.
    
    Verifica que el EventBus pueda manejar múltiples eventos simultáneos
    sin bloquearse o perder eventos.
    """
    engine = engine_fixture
    
    # Registrar componentes con diferentes comportamientos
    normal_comp = EventBusTestComponent("normal_component")
    slow_comp = EventBusTestComponent("slow_component", delay=0.05)
    
    await engine.register_component(normal_comp)
    await engine.register_component(slow_comp)
    
    # Generar múltiples eventos concurrentes
    event_count = 50
    start_time = time.time()
    
    # Función para emitir eventos en paralelo
    async def emit_events():
        tasks = []
        for i in range(event_count):
            # Alternar entre componentes como destino
            target = "normal_component" if i % 2 == 0 else "slow_component"
            # Crear tarea para emitir evento con timeout
            task = asyncio.create_task(
                emit_with_timeout(
                    engine, 
                    f"concurrent_event_{i}", 
                    {"index": i}, 
                    target,
                    timeout=1.0
                )
            )
            tasks.append(task)
            
        # Esperar a que todas las tareas se completen
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
        
    # Ejecutar emisión concurrente con timeout global
    try:
        results = await asyncio.wait_for(emit_events(), timeout=5.0)
        elapsed = time.time() - start_time
        logger.info(f"Procesamiento concurrente completado en {elapsed:.3f} segundos")
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        pytest.fail(f"Timeout detectado después de {elapsed:.3f} segundos - posible deadlock en EventBus")
    
    # Verificar resultados
    # Contar eventos procesados por cada componente
    normal_count = len(normal_comp.events_received)
    slow_count = len(slow_comp.events_received)
    total_processed = normal_count + slow_count
    
    logger.info(f"Eventos procesados: {total_processed}/{event_count} (normal: {normal_count}, lento: {slow_count})")
    
    # Verificar que se procesaron todos los eventos
    assert total_processed >= event_count, f"No se procesaron todos los eventos: {total_processed}/{event_count}"

@pytest.mark.asyncio
async def test_eventbus_error_resilience(engine_fixture):
    """
    Prueba la resiliencia del EventBus ante errores en componentes.
    
    Verifica que:
    1. Los errores en un componente no impiden el funcionamiento general del EventBus
    2. Los eventos siguen fluyendo a otros componentes incluso si uno falla
    3. El EventBus puede recuperarse después de errores
    """
    engine = engine_fixture
    
    # Registrar componentes: uno normal y uno que falla
    normal_comp = EventBusTestComponent("normal_component")
    failing_comp = EventBusTestComponent("failing_component", fail_types=["will_fail"])
    
    await engine.register_component(normal_comp)
    await engine.register_component(failing_comp)
    
    # Fase 1: Enviar evento que causará error en un componente
    logger.info("Fase 1: Enviar evento que causará error")
    response = await emit_with_timeout(
        engine, "will_fail", {"should_fail": True}, "test_source", timeout=1.0
    )
    
    # El EventBus debe seguir funcionando a pesar del error
    assert normal_comp.events_received, "Componente normal no recibió eventos"
    assert failing_comp.error_count > 0, "No se registraron errores en el componente problemático"
    
    # Fase 2: Enviar evento normal después del fallo
    logger.info("Fase 2: Enviar evento normal después del fallo")
    
    # Limpiar eventos recibidos para verificar nuevos eventos
    normal_comp.events_received = []
    failing_comp.events_received = []
    
    response = await emit_with_timeout(
        engine, "normal_event", {"data": "post_error"}, "test_source", timeout=1.0
    )
    
    # Verificar que ambos componentes recibieron el evento normal
    assert len(normal_comp.events_received) > 0, "Componente normal no recibió eventos post-error"
    assert len(failing_comp.events_received) > 0, "Componente que falló no recibió eventos post-error"

@pytest.mark.asyncio
async def test_eventbus_long_running_handlers(engine_fixture):
    """
    Prueba el comportamiento del EventBus con manejadores de eventos de larga duración.
    
    Verifica que:
    1. El EventBus puede manejar componentes que tardan mucho en procesar eventos
    2. No se producen deadlocks cuando hay componentes lentos
    3. Las respuestas se recopilan correctamente incluso de componentes lentos
    """
    engine = engine_fixture
    
    # Registrar un componente con retraso significativo
    slow_comp = EventBusTestComponent("slow_component", delay=0.5)
    very_slow_comp = EventBusTestComponent("very_slow_component", delay=1.0, block_types=["blocking_event"])
    
    await engine.register_component(slow_comp)
    await engine.register_component(very_slow_comp)
    
    # Fase 1: Evento normal - debe completarse, pero tomará tiempo
    logger.info("Fase 1: Procesando evento normal con componentes lentos")
    start_time = time.time()
    
    response = await emit_with_timeout(
        engine, "slow_event", {"data": "slow_processing"}, "test_source", timeout=2.0
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Evento normal procesado en {elapsed:.3f} segundos")
    
    # Verificar respuestas
    assert response is not None, "No hay respuesta para evento normal"
    assert len(response) >= 2, "No se recibieron respuestas de todos los componentes"
    
    # Fase 2: Evento bloqueante - debe manejar timeout correctamente
    logger.info("Fase 2: Procesando evento bloqueante")
    start_time = time.time()
    
    # Este evento causará bloqueo en very_slow_comp
    response = await emit_with_timeout(
        engine, "blocking_event", {"data": "blocked"}, "test_source", timeout=1.5, retries=1
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Evento bloqueante procesado (o con timeout) en {elapsed:.3f} segundos")
    
    # Verificar que el eventbus sigue funcionando después de evento bloqueante
    # Enviemos un evento simple para confirmarlo
    response = await emit_with_timeout(
        engine, "post_blocking", {"data": "after_blocking"}, "test_source", timeout=1.0
    )
    
    assert response is not None, "EventBus no responde después de evento bloqueante"
    
@pytest.mark.asyncio
async def test_eventbus_cleanup_after_errors(engine_fixture):
    """
    Prueba que el EventBus limpia correctamente los recursos después de errores.
    
    Verifica que no queden tareas pendientes o suscriptores fantasma
    después de que ocurran errores en los componentes.
    """
    engine = engine_fixture
    
    # Registrar componentes que fallan de diferentes maneras
    error_comp = EventBusTestComponent("error_component", fail_types=["error_event"])
    block_comp = EventBusTestComponent("block_component", block_types=["long_event"])
    
    await engine.register_component(error_comp)
    await engine.register_component(block_comp)
    
    # Fase 1: Provocar errores
    logger.info("Fase 1: Provocando errores")
    await emit_with_timeout(
        engine, "error_event", {"data": "error"}, "test_source", timeout=1.0
    )
    
    # Intentar evento de larga duración con timeout
    logger.info("Fase 2: Evento de larga duración con timeout")
    await emit_with_timeout(
        engine, "long_event", {"data": "long"}, "test_source", timeout=0.5
    )
    
    # Desregistrar componentes
    logger.info("Fase 3: Desregistrando componentes")
    await engine.unregister_component("error_component")
    await engine.unregister_component("block_component")
    
    # Verificar que no quedan componentes registrados
    assert "error_component" not in engine.components, "Componente no desregistrado"
    assert "block_component" not in engine.components, "Componente no desregistrado"
    
    # Obtener tareas pendientes (excepto la tarea actual)
    pending = [t for t in asyncio.all_tasks() 
               if not t.done() and t != asyncio.current_task()]
    
    # Si hay tareas pendientes, imprimirlas para diagnóstico
    if pending:
        logger.warning(f"Hay {len(pending)} tareas pendientes después de la limpieza")
        for i, task in enumerate(pending):
            logger.warning(f"Tarea {i+1}: {task.get_name()}")
    
    # No debería haber tareas pendientes relacionadas con el EventBus
    eventbus_tasks = [t for t in pending if "eventbus" in t.get_name().lower()]
    assert not eventbus_tasks, f"Hay {len(eventbus_tasks)} tareas del EventBus pendientes"

@pytest.mark.asyncio
async def test_eventbus_shutdown_behavior(engine_fixture):
    """
    Prueba el comportamiento del EventBus durante el apagado del motor.
    
    Verifica que:
    1. El EventBus se cierra limpiamente
    2. No quedan suscriptores activos
    3. No hay tareas pendientes
    """
    engine = engine_fixture
    
    # Registrar varios componentes
    for i in range(5):
        comp = EventBusTestComponent(f"comp_{i}")
        await engine.register_component(comp)
    
    # Verificar componentes registrados
    assert len(engine.components) == 5, f"No todos los componentes se registraron: {len(engine.components)}/5"
    
    # Enviar algunos eventos
    for i in range(3):
        await emit_with_timeout(
            engine, f"test_event_{i}", {"index": i}, "test_source", timeout=1.0
        )
    
    # Detener el motor y verificar que se completa sin errores
    logger.info("Deteniendo el motor")
    start_time = time.time()
    
    try:
        await asyncio.wait_for(engine.stop(), timeout=2.0)
        elapsed = time.time() - start_time
        logger.info(f"Motor detenido en {elapsed:.3f} segundos")
    except Exception as e:
        pytest.fail(f"Error al detener el motor: {e}")
    
    # Verificar que el EventBus está marcado como detenido
    assert not engine.running, "El motor sigue marcado como en ejecución"
    
    # Verificar que no hay tareas pendientes relacionadas con el sistema
    # Dar un poco de tiempo para que se limpien las tareas
    await asyncio.sleep(0.1)
    
    # Obtener tareas pendientes (excepto la tarea actual)
    pending = [t for t in asyncio.all_tasks() 
               if not t.done() and t != asyncio.current_task()]
    
    # Filtrar tareas relacionadas con el EventBus o el motor
    system_tasks = [t for t in pending 
                    if any(x in t.get_name().lower() for x in ["event", "bus", "engine", "component"])]
    
    # No debería haber tareas del sistema pendientes
    assert not system_tasks, f"Hay {len(system_tasks)} tareas del sistema pendientes después del apagado"

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])