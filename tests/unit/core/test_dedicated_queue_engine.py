"""
Test para el motor con colas dedicadas.

Este módulo prueba el funcionamiento del DedicatedQueueEngine y su bus de eventos
para verificar que resuelve los problemas de deadlocks y timeouts.
"""

import asyncio
import logging
import pytest
import time
from typing import Dict, Any, List, Optional

from genesis.core.base import Component
from genesis.core.event_bus_dedicated_queues import DedicatedQueueEventBus
from genesis.core.engine_dedicated_queues import DedicatedQueueEngine

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestComponent(Component):
    """Componente de prueba para verificar el motor con colas dedicadas."""
    
    def __init__(self, name: str, slow_response: bool = False):
        """
        Inicializar componente de prueba.
        
        Args:
            name: Nombre del componente
            slow_response: Si es True, este componente simulará respuestas lentas
        """
        super().__init__(name)
        self.events_received = []
        self.slow_response = slow_response
        self.health = True
        
    async def start(self) -> None:
        """Iniciar componente."""
        self.running = True
        logger.debug(f"Componente {self.name} iniciado")
        
    async def stop(self) -> None:
        """Detener componente."""
        self.running = False
        logger.debug(f"Componente {self.name} detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos recibidos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            
        Returns:
            Respuesta opcional al evento
        """
        # Registrar evento recibido
        self.events_received.append((event_type, data, source))
        logger.debug(f"Componente {self.name} recibió evento: {event_type} de {source}")
        
        # Simular respuesta lenta
        if self.slow_response:
            logger.debug(f"Componente {self.name} simulando respuesta lenta")
            await asyncio.sleep(0.2)
            
        # Responder según el tipo de evento
        if event_type == "check_status":
            return {
                "component": self.name,
                "healthy": self.health
            }
            
        elif event_type == "set_health":
            self.health = data.get("healthy", True)
            return {
                "component": self.name,
                "health_updated": True,
                "new_health": self.health
            }
            
        elif event_type == "echo":
            return {
                "component": self.name,
                "echo": data.get("message", "")
            }
            
        # Para eventos desconocidos
        return {
            "component": self.name,
            "event_processed": True
        }

@pytest.fixture
async def dedicated_engine():
    """Fixture que proporciona un motor con colas dedicadas para pruebas."""
    # Crear motor en modo test
    engine = DedicatedQueueEngine(test_mode=True)
    
    yield engine
    
    # Limpieza
    if engine.running:
        await engine.stop()

@pytest.mark.asyncio
async def test_register_component(dedicated_engine):
    """Probar registro de componentes en el motor de colas dedicadas."""
    # Crear componente
    comp = TestComponent("test_component")
    
    # Registrar componente
    await dedicated_engine.register_component(comp)
    
    # Verificar registro
    assert "test_component" in dedicated_engine.components
    assert dedicated_engine.components["test_component"] == comp

@pytest.mark.asyncio
async def test_start_stop_engine(dedicated_engine):
    """Probar inicio y detención del motor de colas dedicadas."""
    # Crear componentes
    comp_a = TestComponent("comp_a")
    comp_b = TestComponent("comp_b")
    
    # Registrar componentes
    await dedicated_engine.register_component(comp_a)
    await dedicated_engine.register_component(comp_b)
    
    # Iniciar motor
    await dedicated_engine.start()
    
    # Verificar estado
    assert dedicated_engine.running
    assert comp_a.running
    assert comp_b.running
    
    # Detener motor
    await dedicated_engine.stop()
    
    # Verificar estado
    assert not dedicated_engine.running
    assert not comp_a.running
    assert not comp_b.running

@pytest.mark.asyncio
async def test_event_delivery(dedicated_engine):
    """Probar entrega de eventos entre componentes."""
    # Crear componentes
    comp_a = TestComponent("comp_a")
    comp_b = TestComponent("comp_b")
    
    # Registrar componentes
    await dedicated_engine.register_component(comp_a)
    await dedicated_engine.register_component(comp_b)
    
    # Iniciar motor
    await dedicated_engine.start()
    
    # Emitir evento desde comp_a
    await dedicated_engine.event_bus.emit(
        "test_event", 
        {"message": "Hello from A"}, 
        "comp_a"
    )
    
    # Esperar brevemente para que se procese el evento
    await asyncio.sleep(0.1)
    
    # Verificar que comp_b recibió el evento
    assert len(comp_b.events_received) > 0
    assert comp_b.events_received[0][0] == "test_event"
    assert comp_b.events_received[0][1]["message"] == "Hello from A"
    assert comp_b.events_received[0][2] == "comp_a"
    
    # Verificar que comp_a no recibió su propio evento
    assert len(comp_a.events_received) == 0

@pytest.mark.asyncio
async def test_emit_with_response(dedicated_engine):
    """Probar emisión de eventos con respuesta."""
    # Crear componentes
    comp_a = TestComponent("comp_a")
    comp_b = TestComponent("comp_b")
    
    # Registrar componentes
    await dedicated_engine.register_component(comp_a)
    await dedicated_engine.register_component(comp_b)
    
    # Iniciar motor
    await dedicated_engine.start()
    
    # Emitir evento con espera de respuesta
    responses = await dedicated_engine.event_bus.emit_with_response(
        "echo",
        {"message": "Testing responses"},
        "comp_a"
    )
    
    # Verificar respuestas recibidas
    assert len(responses) > 0
    
    # Encontrar respuesta de comp_b
    comp_b_response = None
    for response in responses:
        if response.get("component") == "comp_b":
            comp_b_response = response
            break
            
    assert comp_b_response is not None
    assert comp_b_response.get("echo") == "Testing responses"

@pytest.mark.asyncio
async def test_slow_component_handling(dedicated_engine):
    """Probar manejo de componentes lentos sin bloquear otros."""
    # Crear componentes (uno lento y uno normal)
    slow_comp = TestComponent("slow_comp", slow_response=True)
    fast_comp = TestComponent("fast_comp")
    
    # Registrar componentes
    await dedicated_engine.register_component(slow_comp)
    await dedicated_engine.register_component(fast_comp)
    
    # Iniciar motor
    await dedicated_engine.start()
    
    # Emitir múltiples eventos
    for i in range(5):
        await dedicated_engine.event_bus.emit(
            f"test_event_{i}",
            {"index": i},
            "source"
        )
    
    # Esperar a que se procesen los eventos
    await asyncio.sleep(0.5)
    
    # Verificar que ambos componentes recibieron eventos
    assert len(slow_comp.events_received) > 0
    assert len(fast_comp.events_received) > 0
    
    # Verificar que el componente rápido no fue bloqueado por el lento
    # (debería haber procesado todos los eventos mientras el lento aún está procesando)
    assert len(fast_comp.events_received) >= len(slow_comp.events_received)

@pytest.mark.asyncio
async def test_component_removal(dedicated_engine):
    """Probar eliminación de componentes durante la ejecución."""
    # Crear componentes
    comp_a = TestComponent("comp_a")
    comp_b = TestComponent("comp_b")
    
    # Registrar componentes
    await dedicated_engine.register_component(comp_a)
    await dedicated_engine.register_component(comp_b)
    
    # Iniciar motor
    await dedicated_engine.start()
    
    # Verificar componentes iniciales
    assert "comp_a" in dedicated_engine.components
    assert "comp_b" in dedicated_engine.components
    
    # Eliminar un componente
    result = await dedicated_engine.remove_component("comp_a")
    
    # Verificar resultado
    assert result is True
    assert "comp_a" not in dedicated_engine.components
    assert "comp_b" in dedicated_engine.components
    assert not comp_a.running
    
    # Intentar emitir evento
    await dedicated_engine.event_bus.emit(
        "test_event",
        {"message": "After removal"},
        "source"
    )
    
    # Esperar procesamiento
    await asyncio.sleep(0.1)
    
    # Verificar que solo comp_b recibió el evento
    assert len(comp_b.events_received) > 0
    assert comp_b.events_received[-1][0] == "test_event"

@pytest.mark.asyncio
async def test_component_status_retrieval(dedicated_engine):
    """Probar obtención del estado de componentes."""
    # Crear componentes
    comp_a = TestComponent("comp_a")
    comp_b = TestComponent("comp_b")
    
    # Registrar componentes
    await dedicated_engine.register_component(comp_a)
    await dedicated_engine.register_component(comp_b)
    
    # Iniciar motor
    await dedicated_engine.start()
    
    # Obtener estado
    status = await dedicated_engine.get_component_status()
    
    # Verificar estado
    assert status["total_components"] == 2
    assert status["running_components"] == 2
    assert "comp_a" in status["components"]
    assert "comp_b" in status["components"]
    assert status["components"]["comp_a"]["running"] is True
    assert status["components"]["comp_b"]["running"] is True