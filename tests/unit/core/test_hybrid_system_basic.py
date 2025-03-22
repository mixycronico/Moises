"""
Pruebas básicas para el sistema híbrido API + WebSocket.

Este módulo contiene pruebas que verifican que el sistema híbrido
funciona correctamente y evita deadlocks que ocurrían en versiones
anteriores del motor de Genesis.
"""

import pytest
import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional, Set

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from genesis.core.component import Component
from genesis.core.genesis_hybrid import ComponentAPI, GenesisHybridCoordinator

# Configurar logging solo para pruebas
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("genesis.test")

class TestComponent(ComponentAPI):
    """Componente de prueba para el sistema híbrido."""
    
    def __init__(self, id: str):
        super().__init__(id)
        self.requests_handled = []
        self.results = {}
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitud de prueba."""
        self.requests_handled.append((request_type, data, source))
        self.metrics["requests_processed"] += 1
        
        if request_type == "echo":
            return {"echo": data.get("message", ""), "source": source}
        
        elif request_type == "sleep":
            duration = data.get("duration", 0.1)
            await asyncio.sleep(duration)
            return {"slept": duration}
        
        elif request_type == "get_results":
            return self.results
        
        return None
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento de prueba."""
        await super().on_event(event_type, data, source)
        
        if event_type == "store_result":
            key = data.get("key", "default")
            value = data.get("value")
            self.results[key] = value

class DeadlockComponent(ComponentAPI):
    """
    Componente que simula comportamientos que causaban deadlocks
    en el sistema anterior.
    """
    
    def __init__(self, id: str, coordinator: GenesisHybridCoordinator):
        super().__init__(id)
        self.coordinator = coordinator
        self.recursive_count = 0
        self.circular_count = 0
        self.blocking_count = 0
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitud que podría causar deadlock."""
        self.metrics["requests_processed"] += 1
        
        if request_type == "recursive_call":
            """Simular llamadas recursivas (A llama a A)."""
            depth = data.get("depth", 1)
            self.recursive_count += 1
            
            if depth > 1:
                # Hacer una llamada recursiva
                result = await self.coordinator.request(
                    self.id,  # Llamarse a sí mismo (causa de deadlock en sistema anterior)
                    "recursive_call",
                    {"depth": depth - 1},
                    self.id
                )
                return {
                    "current_depth": depth,
                    "next_result": result,
                    "recursive_count": self.recursive_count
                }
            
            return {"depth": depth, "recursive_count": self.recursive_count}
        
        elif request_type == "circular_call":
            """Simular llamadas circulares (A llama a B, B llama a A)."""
            target_id = data.get("target_id")
            depth = data.get("depth", 2)
            self.circular_count += 1
            
            if depth > 0 and target_id:
                # Hacer una llamada al objetivo, que podría llamarnos de vuelta
                result = await self.coordinator.request(
                    target_id,
                    "circular_call",
                    {
                        "target_id": self.id,  # Establecer el objetivo como nosotros mismos
                        "depth": depth - 1
                    },
                    self.id
                )
                return {
                    "current_depth": depth,
                    "target": target_id,
                    "next_result": result,
                    "circular_count": self.circular_count
                }
            
            return {"depth": depth, "circular_count": self.circular_count}
        
        elif request_type == "blocking_call":
            """Simular llamadas que bloquean durante un tiempo."""
            duration = data.get("duration", 0.5)
            self.blocking_count += 1
            
            # Simular procesamiento bloqueante
            await asyncio.sleep(duration)
            
            return {
                "duration": duration,
                "blocking_count": self.blocking_count
            }
        
        elif request_type == "mixed_call":
            """Combinar varias formas de deadlock potencial."""
            target_id = data.get("target_id")
            block_first = data.get("block_first", True)
            
            if block_first:
                # Primero bloquear, luego hacer llamada circular
                await asyncio.sleep(0.2)
                
                if target_id:
                    result = await self.coordinator.request(
                        target_id,
                        "mixed_call",
                        {"target_id": self.id, "block_first": False},
                        self.id
                    )
                    return {"strategy": "block_then_call", "next_result": result}
            else:
                # Primero hacer llamada circular, luego bloquear
                if target_id:
                    result = await self.coordinator.request(
                        target_id,
                        "blocking_call",
                        {"duration": 0.3},
                        self.id
                    )
                    
                    await asyncio.sleep(0.1)
                    return {"strategy": "call_then_block", "next_result": result}
            
            return {"strategy": "no_action"}
        
        return None

@pytest.fixture
async def coordinator():
    """Fixture que proporciona un coordinador híbrido para pruebas."""
    coord = GenesisHybridCoordinator(host="localhost", port=0)  # Puerto 0 para evitar conflictos
    yield coord

@pytest.mark.asyncio
async def test_basic_request_response(coordinator):
    """Prueba que las solicitudes básicas funcionan correctamente."""
    # Crear y registrar componentes
    comp1 = TestComponent("comp1")
    comp2 = TestComponent("comp2")
    
    coordinator.register_component("comp1", comp1)
    coordinator.register_component("comp2", comp2)
    
    await coordinator.start()
    
    try:
        # Hacer una solicitud simple
        result = await coordinator.request(
            "comp2",
            "echo",
            {"message": "Hello from comp1"},
            "comp1"
        )
        
        assert result is not None
        assert result["echo"] == "Hello from comp1"
        assert result["source"] == "comp1"
        
        # Verificar que la solicitud se registró en el componente
        assert len(comp2.requests_handled) == 1
        assert comp2.requests_handled[0][0] == "echo"
        assert comp2.requests_handled[0][1]["message"] == "Hello from comp1"
        assert comp2.requests_handled[0][2] == "comp1"
        
    finally:
        await coordinator.stop()

@pytest.mark.asyncio
async def test_broadcast_event(coordinator):
    """Prueba que los eventos se difunden correctamente a todos los componentes."""
    # Crear y registrar componentes
    comp1 = TestComponent("comp1")
    comp2 = TestComponent("comp2")
    comp3 = TestComponent("comp3")
    
    coordinator.register_component("comp1", comp1)
    coordinator.register_component("comp2", comp2)
    coordinator.register_component("comp3", comp3)
    
    await coordinator.start()
    
    try:
        # Emitir un evento
        await coordinator.broadcast_event(
            "store_result",
            {"key": "test_key", "value": "test_value"},
            "comp1"  # Origen
        )
        
        # Dar tiempo para que se procese el evento
        await asyncio.sleep(0.1)
        
        # Verificar que comp2 y comp3 recibieron el evento (comp1 no, es el origen)
        assert len(comp1.events_received) == 0  # El origen no recibe su propio evento
        assert len(comp2.events_received) == 1
        assert len(comp3.events_received) == 1
        
        # Verificar que los datos se almacenaron
        assert "test_key" in comp2.results
        assert comp2.results["test_key"] == "test_value"
        
        assert "test_key" in comp3.results
        assert comp3.results["test_key"] == "test_value"
        
    finally:
        await coordinator.stop()

@pytest.mark.asyncio
async def test_recursive_calls_no_deadlock(coordinator):
    """
    Prueba que las llamadas recursivas no causan deadlock en el sistema híbrido.
    
    Este tipo de llamadas (un componente llamándose a sí mismo) causaban
    deadlocks en el sistema anterior.
    """
    # Crear y registrar componente
    deadlock_comp = DeadlockComponent("deadlock1", coordinator)
    coordinator.register_component("deadlock1", deadlock_comp)
    
    await coordinator.start()
    
    try:
        # Hacer una llamada recursiva profunda
        result = await coordinator.request(
            "deadlock1",
            "recursive_call",
            {"depth": 5},  # Profundidad 5, causaría deadlock en el sistema anterior
            "test"
        )
        
        # Verificar que se completó correctamente
        assert result is not None
        assert result["current_depth"] == 5
        assert "next_result" in result
        
        # Verificar la profundidad de la recursión
        current = result
        depth = 5
        while "next_result" in current and current["next_result"] is not None:
            current = current["next_result"]
            depth -= 1
            assert current["current_depth"] == depth if depth > 1 else None
        
        # Verificar contador de llamadas recursivas
        assert deadlock_comp.recursive_count == 5
        
    finally:
        await coordinator.stop()

@pytest.mark.asyncio
async def test_circular_calls_no_deadlock(coordinator):
    """
    Prueba que las llamadas circulares no causan deadlock en el sistema híbrido.
    
    Este tipo de llamadas (A llama a B, B llama a A) causaban
    deadlocks en el sistema anterior.
    """
    # Crear y registrar componentes
    deadlock_comp1 = DeadlockComponent("deadlock1", coordinator)
    deadlock_comp2 = DeadlockComponent("deadlock2", coordinator)
    
    coordinator.register_component("deadlock1", deadlock_comp1)
    coordinator.register_component("deadlock2", deadlock_comp2)
    
    await coordinator.start()
    
    try:
        # Iniciar cadena de llamadas circulares
        result = await coordinator.request(
            "deadlock1",
            "circular_call",
            {
                "target_id": "deadlock2",
                "depth": 4  # Profundidad 4, alternará entre ambos componentes
            },
            "test"
        )
        
        # Verificar que se completó correctamente
        assert result is not None
        assert result["current_depth"] == 4
        assert "next_result" in result
        
        # Verificar que las llamadas alternaron entre componentes
        assert deadlock_comp1.circular_count + deadlock_comp2.circular_count == 4
        
    finally:
        await coordinator.stop()

@pytest.mark.asyncio
async def test_blocking_calls_parallel(coordinator):
    """
    Prueba que las llamadas bloqueantes se procesan en paralelo
    sin afectar a otras operaciones.
    """
    # Crear y registrar componentes
    deadlock_comp = DeadlockComponent("deadlock1", coordinator)
    test_comp = TestComponent("test1")
    
    coordinator.register_component("deadlock1", deadlock_comp)
    coordinator.register_component("test1", test_comp)
    
    await coordinator.start()
    
    try:
        # Iniciar llamada bloqueante
        blocking_task = asyncio.create_task(
            coordinator.request(
                "deadlock1",
                "blocking_call",
                {"duration": 1.0},  # Bloqueo de 1 segundo
                "test"
            )
        )
        
        # Sin esperar que termine, hacer varias llamadas rápidas a otro componente
        results = []
        for i in range(5):
            result = await coordinator.request(
                "test1",
                "echo",
                {"message": f"Fast call {i}"},
                "test"
            )
            results.append(result)
            await asyncio.sleep(0.1)
        
        # Todas las llamadas rápidas deberían completarse antes que la bloqueante
        assert len(results) == 5
        assert all(r is not None for r in results)
        
        # Esperar que la llamada bloqueante termine
        blocking_result = await blocking_task
        assert blocking_result is not None
        assert blocking_result["duration"] == 1.0
        
    finally:
        await coordinator.stop()

@pytest.mark.asyncio
async def test_mixed_potential_deadlock_patterns(coordinator):
    """
    Prueba que combinaciones de patrones potenciales de deadlock
    no causan problemas en el sistema híbrido.
    """
    # Crear y registrar componentes
    deadlock_comp1 = DeadlockComponent("deadlock1", coordinator)
    deadlock_comp2 = DeadlockComponent("deadlock2", coordinator)
    
    coordinator.register_component("deadlock1", deadlock_comp1)
    coordinator.register_component("deadlock2", deadlock_comp2)
    
    await coordinator.start()
    
    try:
        # Iniciar patrón mixto que combina bloqueo y llamada circular
        result = await coordinator.request(
            "deadlock1",
            "mixed_call",
            {
                "target_id": "deadlock2",
                "block_first": True
            },
            "test"
        )
        
        # Verificar que se completó correctamente
        assert result is not None
        assert result["strategy"] == "block_then_call"
        assert "next_result" in result
        
        # Hacer otro tipo de patrón mixto
        result2 = await coordinator.request(
            "deadlock2",
            "mixed_call",
            {
                "target_id": "deadlock1",
                "block_first": False
            },
            "test"
        )
        
        # Verificar que se completó correctamente
        assert result2 is not None
        assert result2["strategy"] == "call_then_block"
        assert "next_result" in result2
        
    finally:
        await coordinator.stop()

@pytest.mark.asyncio
async def test_many_concurrent_requests(coordinator):
    """
    Prueba que el sistema puede manejar muchas solicitudes concurrentes
    sin deadlocks ni problemas de rendimiento.
    """
    # Crear y registrar componentes
    deadlock_comp = DeadlockComponent("deadlock1", coordinator)
    test_comp = TestComponent("test1")
    
    coordinator.register_component("deadlock1", deadlock_comp)
    coordinator.register_component("test1", test_comp)
    
    await coordinator.start()
    
    try:
        # Crear muchas solicitudes concurrentes de diferentes tipos
        tasks = []
        
        # Solicitudes de eco (rápidas)
        for i in range(20):
            tasks.append(
                coordinator.request(
                    "test1",
                    "echo",
                    {"message": f"Concurrent {i}"},
                    "test"
                )
            )
        
        # Solicitudes de bloqueo (lentas)
        for i in range(5):
            tasks.append(
                coordinator.request(
                    "deadlock1",
                    "blocking_call",
                    {"duration": random.uniform(0.1, 0.5)},
                    "test"
                )
            )
        
        # Solicitudes recursivas
        for i in range(3):
            tasks.append(
                coordinator.request(
                    "deadlock1",
                    "recursive_call",
                    {"depth": random.randint(2, 4)},
                    "test"
                )
            )
        
        # Esperar a que todas las solicitudes se completen
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verificar que no hubo excepciones
        for result in results:
            assert not isinstance(result, Exception)
        
        # Verificar que se completaron todas las solicitudes
        assert len(results) == 28
        assert all(r is not None for r in results)
        
    finally:
        await coordinator.stop()

@pytest.mark.asyncio
async def test_request_timeout_does_not_block(coordinator):
    """
    Prueba que un timeout en una solicitud no bloquea el sistema
    ni impide otras operaciones.
    """
    # Crear y registrar componentes
    deadlock_comp = DeadlockComponent("deadlock1", coordinator)
    test_comp = TestComponent("test1")
    
    coordinator.register_component("deadlock1", deadlock_comp)
    coordinator.register_component("test1", test_comp)
    
    await coordinator.start()
    
    try:
        # Hacer una solicitud con un timeout muy corto que definitivamente fallará
        timeout_task = asyncio.create_task(
            coordinator.request(
                "deadlock1",
                "blocking_call",
                {"duration": 1.0},  # Bloquea por 1 segundo
                "test",
                timeout=0.1  # Timeout de 0.1 segundos
            )
        )
        
        # Mientras tanto, hacer otras solicitudes que deberían funcionar
        await asyncio.sleep(0.05)  # Pequeña pausa para asegurar que la primera solicitud comenzó
        
        echo_result = await coordinator.request(
            "test1",
            "echo",
            {"message": "This should work despite timeout"},
            "test"
        )
        
        # El timeout debería ocurrir antes de que termine la solicitud bloqueante
        timeout_result = await timeout_task
        
        # Verificar que el timeout resultó en None
        assert timeout_result is None
        
        # La solicitud de eco debería haberse completado normalmente
        assert echo_result is not None
        assert echo_result["echo"] == "This should work despite timeout"
        
    finally:
        await coordinator.stop()

@pytest.mark.asyncio
async def test_simultaneous_api_and_websocket(coordinator):
    """
    Prueba que el sistema puede manejar simultáneamente API y WebSocket
    sin interferencias.
    """
    # Crear y registrar componentes
    test_comp1 = TestComponent("test1")
    test_comp2 = TestComponent("test2")
    
    coordinator.register_component("test1", test_comp1)
    coordinator.register_component("test2", test_comp2)
    
    await coordinator.start()
    
    try:
        # Hacer solicitudes API mientras se emiten eventos WebSocket
        api_tasks = []
        for i in range(10):
            api_tasks.append(
                coordinator.request(
                    "test1",
                    "echo",
                    {"message": f"API call {i}"},
                    "test"
                )
            )
            
            if i % 2 == 0:
                await coordinator.broadcast_event(
                    "store_result",
                    {"key": f"event_{i}", "value": f"value_{i}"},
                    "test"
                )
        
        # Esperar a que todas las solicitudes API se completen
        api_results = await asyncio.gather(*api_tasks)
        
        # Verificar que todas las solicitudes API se completaron correctamente
        assert len(api_results) == 10
        assert all(r is not None for r in api_results)
        
        # Dar tiempo para que se procesen todos los eventos
        await asyncio.sleep(0.1)
        
        # Verificar que los eventos se recibieron correctamente
        keys_in_results = [k for k in test_comp2.results.keys() if k.startswith("event_")]
        assert len(keys_in_results) == 5  # Eventos enviados en iteraciones pares (0,2,4,6,8)
        
    finally:
        await coordinator.stop()