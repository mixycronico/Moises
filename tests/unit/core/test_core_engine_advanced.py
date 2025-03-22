"""
Tests avanzados para el motor principal (core.engine).

Este módulo prueba escenarios avanzados del motor principal,
incluyendo manejo de concurrencia, recuperación de errores,
y condiciones de carrera.
"""

import pytest
import asyncio
import logging
import time
import random
from unittest.mock import Mock, patch, AsyncMock

from genesis.core.engine import Engine
from genesis.core.component import Component
from genesis.core.event_bus import EventBus


class DelayedComponent(Component):
    """Componente que simula retrasos en el procesamiento."""
    
    def __init__(self, name, delay_range=(0.01, 0.05)):
        """Inicializar con un rango de retraso aleatorio."""
        super().__init__(name)
        self.delay_range = delay_range
        self.processed_events = []
        self.error_rate = 0.0  # Probabilidad de error (0.0 - 1.0)
        
    async def start(self) -> None:
        """Iniciar el componente."""
        self.is_running = True
        
    async def stop(self) -> None:
        """Detener el componente."""
        self.is_running = False
        
    async def handle_event(self, event_type, data, source):
        """Manejar evento con retraso aleatorio y posibilidad de error."""
        # Simular retraso en el procesamiento
        delay = random.uniform(*self.delay_range)
        await asyncio.sleep(delay)
        
        # Registrar el evento procesado
        processed_event = {
            "type": event_type,
            "data": data,
            "source": source,
            "processed_at": time.time(),
            "processing_time": delay
        }
        self.processed_events.append(processed_event)
        
        # Simular errores aleatorios si la tasa de error > 0
        if self.error_rate > 0 and random.random() < self.error_rate:
            raise Exception(f"Random error in {self.name} while processing {event_type}")
        
        # Retornar respuesta
        return {
            "status": "processed",
            "by": self.name,
            "delay": delay,
            "original_data": data
        }


class StateTrackingComponent(Component):
    """Componente que trackea el estado del sistema."""
    
    def __init__(self, name):
        """Inicializar con estado del sistema."""
        super().__init__(name)
        self.system_state = {}
        self.state_updates = []
        
    async def start(self) -> None:
        """Iniciar el componente."""
        self.is_running = True
        
    async def stop(self) -> None:
        """Detener el componente."""
        self.is_running = False
        
    async def handle_event(self, event_type, data, source):
        """Actualizar estado del sistema basado en eventos."""
        if event_type == "state_update":
            # Actualizar estado
            for key, value in data.items():
                self.system_state[key] = value
            
            # Registrar actualización
            self.state_updates.append({
                "source": source,
                "timestamp": time.time(),
                "data": data.copy()
            })
            
            return {"status": "state_updated", "current_state": self.system_state.copy()}
        
        elif event_type == "state_query":
            # Consultar estado actual
            query_key = data.get("key")
            if query_key:
                return {
                    "key": query_key,
                    "value": self.system_state.get(query_key),
                    "found": query_key in self.system_state
                }
            else:
                return {"full_state": self.system_state.copy()}
        
        return None


@pytest.fixture
def event_bus():
    """Proporcionar un bus de eventos para pruebas."""
    return EventBus()


@pytest.fixture
def advanced_engine(event_bus):
    """Proporcionar un motor con componentes avanzados para pruebas."""
    engine = Engine(event_bus)
    
    # Componentes de procesamiento con diferentes latencias
    fast_processor = DelayedComponent("fast_processor", delay_range=(0.001, 0.01))
    medium_processor = DelayedComponent("medium_processor", delay_range=(0.01, 0.05))
    slow_processor = DelayedComponent("slow_processor", delay_range=(0.05, 0.1))
    
    # Componente de tracking de estado
    state_tracker = StateTrackingComponent("state_tracker")
    
    # Registrar componentes
    engine.register_component(fast_processor)
    engine.register_component(medium_processor)
    engine.register_component(slow_processor)
    engine.register_component(state_tracker)
    
    return engine


@pytest.mark.asyncio
async def test_concurrent_event_processing(advanced_engine):
    """Verificar procesamiento concurrente de múltiples eventos."""
    # Iniciar el motor
    await advanced_engine.start()
    
    # Generar múltiples eventos simultáneamente
    num_events = 10
    event_futures = []
    
    for i in range(num_events):
        event_data = {"event_id": i, "payload": f"Concurrent test payload {i}"}
        future = advanced_engine.event_bus.emit("concurrent_test", event_data, "test_source")
        event_futures.append(future)
    
    # Esperar a que todos los eventos se procesen
    await asyncio.gather(*event_futures)
    
    # Verificar que todos los componentes procesaron todos los eventos
    for component_name in ["fast_processor", "medium_processor", "slow_processor"]:
        component = advanced_engine.components[component_name]
        assert len(component.processed_events) == num_events
        
        # Verificar que los event_ids están completos
        event_ids = [event["data"]["event_id"] for event in component.processed_events]
        assert set(event_ids) == set(range(num_events))
    
    # Detener el motor
    await advanced_engine.stop()


@pytest.mark.asyncio
async def test_engine_error_resilience(advanced_engine):
    """Verificar que el motor es resiliente a errores en componentes."""
    # Configurar uno de los componentes para fallar frecuentemente
    error_component = advanced_engine.components["medium_processor"]
    error_component.error_rate = 0.5  # 50% de probabilidad de error
    
    # Iniciar el motor
    await advanced_engine.start()
    
    # Enviar varios eventos que podrían causar errores
    num_events = 20
    event_futures = []
    
    for i in range(num_events):
        event_data = {"event_id": i, "payload": f"Error resilience test {i}"}
        future = advanced_engine.event_bus.emit("error_test", event_data, "test_source")
        event_futures.append(future)
    
    # Esperar a que todos los eventos se procesen, algunos con errores
    await asyncio.gather(*event_futures, return_exceptions=True)
    
    # Verificar que los otros componentes siguen funcionando
    for component_name in ["fast_processor", "slow_processor"]:
        component = advanced_engine.components[component_name]
        assert len(component.processed_events) == num_events
    
    # Verificar que el sistema sigue respondiendo después de los errores
    # Enviar un evento de estado para verificar
    response = await advanced_engine.event_bus.emit_with_response(
        "state_query", {"key": "test_key"}, "test_source"
    )
    assert any(r.get("key") == "test_key" for r in response)
    
    # Detener el motor
    await advanced_engine.stop()


@pytest.mark.asyncio
async def test_system_state_consistency(advanced_engine):
    """Verificar consistencia del estado del sistema bajo carga."""
    # Iniciar el motor
    await advanced_engine.start()
    
    # Componente que trackea el estado
    state_tracker = advanced_engine.components["state_tracker"]
    
    # Realizar múltiples actualizaciones de estado desde diferentes fuentes
    updates_per_source = 5
    sources = ["source1", "source2", "source3"]
    
    # Diccionario para trackear el estado esperado
    expected_state = {}
    
    # Enviar actualizaciones concurrentes
    update_futures = []
    
    for source in sources:
        for i in range(updates_per_source):
            update_data = {
                f"{source}_key_{i}": f"{source}_value_{i}",
                f"shared_key": f"{source}_{i}"  # Clave compartida que será sobrescrita
            }
            
            # Actualizar el estado esperado
            expected_state.update(update_data)
            
            # Enviar actualización
            future = advanced_engine.event_bus.emit("state_update", update_data, source)
            update_futures.append(future)
    
    # Esperar a que todas las actualizaciones se procesen
    await asyncio.gather(*update_futures)
    
    # La última actualización a shared_key determinará su valor final
    # Pero no podemos saber cuál será debido a la concurrencia
    # Verificamos que existe y eliminamos para comparación
    assert "shared_key" in state_tracker.system_state
    del expected_state["shared_key"]
    shared_key_value = state_tracker.system_state.pop("shared_key")
    
    # Verificar que todas las demás claves coinciden
    for key, value in expected_state.items():
        assert state_tracker.system_state[key] == value
    
    # Verificar el número total de actualizaciones
    assert len(state_tracker.state_updates) == len(sources) * updates_per_source
    
    # Detener el motor
    await advanced_engine.stop()


@pytest.mark.asyncio
async def test_event_processing_ordering(advanced_engine):
    """Verificar el orden de procesamiento de eventos en componentes asíncronos."""
    # Iniciar el motor
    await advanced_engine.start()
    
    # Enviar eventos secuenciales con dependencias
    event_sequence = [
        {"id": 1, "requires": None},
        {"id": 2, "requires": 1},
        {"id": 3, "requires": 2},
        {"id": 4, "requires": 3},
        {"id": 5, "requires": 4}
    ]
    
    # Enviar eventos en orden, esperando confirmación de cada uno
    for event in event_sequence:
        response = await advanced_engine.event_bus.emit_with_response(
            "sequential_event", event, "test_source"
        )
        
        # Verificar que todos los componentes procesaron el evento
        assert len(response) == 4  # 3 processors + state_tracker
        
        # Verificar que todos devolvieron status=processed (excepto state_tracker)
        processor_responses = [r for r in response if "status" in r and r["status"] == "processed"]
        assert len(processor_responses) == 3
    
    # Verificar orden de procesamiento en cada componente
    for component_name in ["fast_processor", "medium_processor", "slow_processor"]:
        component = advanced_engine.components[component_name]
        
        # Extraer IDs en orden de procesamiento
        processed_ids = [event["data"]["id"] for event in component.processed_events]
        
        # Verificar que los IDs están en orden (puede haber otros eventos, pero estos deben estar ordenados entre sí)
        for i in range(1, len(event_sequence)):
            # El evento i+1 debe venir después del evento i
            assert processed_ids.index(i) < processed_ids.index(i+1)
    
    # Detener el motor
    await advanced_engine.stop()


@pytest.mark.asyncio
async def test_load_balancing_between_components(advanced_engine):
    """Verificar distribución de carga entre componentes similares."""
    # Crear componentes adicionales del mismo tipo
    num_additional = 3
    worker_components = []
    
    for i in range(num_additional):
        worker = DelayedComponent(f"worker_{i}", delay_range=(0.01, 0.03))
        advanced_engine.register_component(worker)
        worker_components.append(worker)
    
    # Iniciar el motor
    await advanced_engine.start()
    
    # Crear un bus de eventos con un despachador personalizado que implemente round-robin
    event_count = 100
    worker_selector = 0
    
    # Mockear emit para implementar round-robin entre workers
    original_emit = advanced_engine.event_bus.emit
    
    async def round_robin_emit(event_type, data, source):
        nonlocal worker_selector
        
        if event_type == "work_task":
            # Seleccionar worker en round-robin
            worker_name = f"worker_{worker_selector}"
            worker_selector = (worker_selector + 1) % num_additional
            
            # Enviar evento solo al worker seleccionado
            worker = advanced_engine.components.get(worker_name)
            if worker:
                return await worker.handle_event(event_type, data, source)
            
        # Para otros eventos, usar comportamiento normal
        return await original_emit(event_type, data, source)
    
    # Reemplazar temporalmente el método emit
    advanced_engine.event_bus.emit = round_robin_emit
    
    # Enviar tareas a los workers
    for i in range(event_count):
        task_data = {"task_id": i, "workload": f"Task {i} data"}
        await advanced_engine.event_bus.emit("work_task", task_data, "dispatcher")
    
    # Restaurar el método emit original
    advanced_engine.event_bus.emit = original_emit
    
    # Verificar la distribución de tareas
    worker_loads = [len(worker.processed_events) for worker in worker_components]
    
    # La distribución debe ser aproximadamente uniforme
    expected_per_worker = event_count / num_additional
    
    # Verificar que ningún worker recibió más de 10% extra o menos de carga
    for load in worker_loads:
        assert abs(load - expected_per_worker) < expected_per_worker * 0.1
    
    # Detener el motor
    await advanced_engine.stop()


@pytest.mark.asyncio
async def test_component_priority_based_execution():
    """Verificar que los componentes se ejecutan según su prioridad."""
    # Crear un nuevo motor con prioridades específicas y modo de prueba para evitar tareas en background
    new_engine = Engine(EventBus(test_mode=True))
    
    # Crear componentes con diferentes prioridades
    # Reducir delay_range a valores más pequeños para acelerar el test
    high_priority = DelayedComponent("high_priority", delay_range=(0.001, 0.002))
    medium_priority = DelayedComponent("medium_priority", delay_range=(0.001, 0.002))
    low_priority = DelayedComponent("low_priority", delay_range=(0.001, 0.002))
    
    # Añadir componentes en orden inverso para asegurar que la prioridad funciona
    new_engine.register_component(low_priority, priority=10)
    new_engine.register_component(medium_priority, priority=50)
    new_engine.register_component(high_priority, priority=100)
    
    # Capturar el orden de inicio
    startup_order = []
    
    # Reemplazar el método start de cada componente para registrar el orden
    # Usar una función de fábrica para asegurar que cada reemplazo tenga su propio alcance
    def create_start_replacement(component, original_start):
        async def replacement():
            startup_order.append(component.name)
            return await original_start()
        return replacement
    
    # Reemplazar los métodos start
    for name, component in new_engine.components.items():
        original_start = component.start
        component.start = create_start_replacement(component, original_start)
    
    # Iniciar el motor
    await new_engine.start()
    
    # Verificar el orden de inicio (mayor prioridad primero)
    assert len(startup_order) == 3, f"Expected 3 components to start, got {len(startup_order)}: {startup_order}"
    assert startup_order[0] == "high_priority", f"Expected high_priority first, got {startup_order}"
    assert startup_order[1] == "medium_priority", f"Expected medium_priority second, got {startup_order}" 
    assert startup_order[2] == "low_priority", f"Expected low_priority third, got {startup_order}"
    
    # Detener el motor sin intentar enviar eventos adicionales
    await new_engine.stop()