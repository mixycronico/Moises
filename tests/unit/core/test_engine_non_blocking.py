"""
Pruebas para la implementación no bloqueante del motor.

Este archivo contiene pruebas específicas para verificar que la versión
no bloqueante del motor resuelve los problemas de timeouts.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, List

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Silenciar logs durante las pruebas
logging.basicConfig(level=logging.CRITICAL)

# Componente de prueba normal que guarda eventos
class SimpleTestComponent(Component):
    """Componente simple para pruebas que guarda eventos."""
    
    def __init__(self, name):
        """Inicializar el componente."""
        super().__init__(name)
        self.events_received = []
        self.started = False
        self.stopped = False
    
    async def start(self):
        """Iniciar el componente."""
        self.started = True
    
    async def stop(self):
        """Detener el componente."""
        self.stopped = True
    
    async def handle_event(self, event_type, data, source):
        """Manejar un evento y guardarlo."""
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        return None

# Componente de prueba que falla al manejar ciertos eventos
class FailingTestComponent(Component):
    """Componente para pruebas que falla al manejar ciertos eventos."""
    
    def __init__(self, name):
        """Inicializar el componente."""
        super().__init__(name)
        self.events_received = []
        self.started = False
        self.stopped = False
        self.fail_on_events = ["test.error"]
    
    async def start(self):
        """Iniciar el componente."""
        self.started = True
    
    async def stop(self):
        """Detener el componente."""
        self.stopped = True
    
    async def handle_event(self, event_type, data, source):
        """Manejar un evento, fallando si es del tipo especificado."""
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        
        if event_type in self.fail_on_events:
            raise Exception(f"Error simulado en {self.name} al manejar {event_type}")
        
        return None

# Componente de prueba que bloquea durante mucho tiempo
class BlockingTestComponent(Component):
    """Componente para pruebas que bloquea por mucho tiempo."""
    
    def __init__(self, name, block_time=1.0):
        """
        Inicializar el componente.
        
        Args:
            name: Nombre del componente
            block_time: Tiempo de bloqueo en segundos
        """
        super().__init__(name)
        self.events_received = []
        self.started = False
        self.stopped = False
        self.block_time = block_time
    
    async def start(self):
        """Iniciar el componente, bloqueando intencionalmente."""
        await asyncio.sleep(self.block_time)
        self.started = True
    
    async def stop(self):
        """Detener el componente, bloqueando intencionalmente."""
        await asyncio.sleep(self.block_time)
        self.stopped = True
    
    async def handle_event(self, event_type, data, source):
        """Manejar un evento, bloqueando intencionalmente."""
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        
        # Bloquear por el tiempo especificado
        await asyncio.sleep(self.block_time)
        
        return None

@pytest.mark.asyncio
async def test_engine_non_blocking_start_stop():
    """Verificar que el motor no bloqueante puede iniciar y detener sin timeouts."""
    # Crear el motor en modo prueba
    engine = EngineNonBlocking(test_mode=True)
    
    # Registrar un componente normal
    component = SimpleTestComponent("simple")
    engine.register_component(component)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que el componente se inició
    assert component.started
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que el componente se detuvo
    assert component.stopped

@pytest.mark.asyncio
async def test_engine_non_blocking_with_blocking_components():
    """Verificar que el motor no se bloquea con componentes que tardan mucho."""
    # Crear el motor en modo prueba
    engine = EngineNonBlocking(test_mode=True)
    
    # Registrar un componente que bloquea por mucho tiempo
    component = BlockingTestComponent("blocking", block_time=2.0)
    engine.register_component(component)
    
    # Iniciar el motor (debería usar timeouts para evitar bloqueos)
    await engine.start()
    
    # No verificamos si component.started es True porque el timeout
    # podría haber impedido que el componente termine de iniciarse.
    # Lo importante es que engine.start() no se bloquee indefinidamente.
    
    # Verificar que el motor está marcado como iniciado aunque el componente
    # puede que no haya terminado su inicialización
    assert engine.running
    
    # Detener el motor (también debería usar timeouts)
    await engine.stop()
    
    # Verificar que el motor está marcado como detenido
    assert not engine.running

@pytest.mark.asyncio
async def test_engine_non_blocking_error_handling():
    """Verificar que el motor maneja correctamente los errores en componentes."""
    # Crear el motor en modo prueba
    engine = EngineNonBlocking(test_mode=True)
    
    # Registrar un componente normal y uno que falla
    normal_component = SimpleTestComponent("normal")
    failing_component = FailingTestComponent("failing")
    
    engine.register_component(normal_component)
    engine.register_component(failing_component)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que ambos componentes se iniciaron
    assert normal_component.started
    assert failing_component.started
    
    # Limpiar eventos del inicio
    normal_component.events_received.clear()
    failing_component.events_received.clear()
    
    # Emitir evento que no causa error
    await engine.emit_event("test.normal", {"message": "normal"}, "test")
    
    # Dar tiempo para que se procesen los eventos
    await asyncio.sleep(0.1)
    
    # Verificar que ambos componentes recibieron el evento
    assert len(normal_component.events_received) == 1
    assert normal_component.events_received[0]["type"] == "test.normal"
    
    assert len(failing_component.events_received) == 1
    assert failing_component.events_received[0]["type"] == "test.normal"
    
    # Emitir evento que causa error en el componente que falla
    await engine.emit_event("test.error", {"message": "error"}, "test")
    
    # Dar tiempo para que se procesen los eventos
    await asyncio.sleep(0.1)
    
    # El componente normal debería haber recibido el evento error
    assert len(normal_component.events_received) == 2
    assert normal_component.events_received[1]["type"] == "test.error"
    
    # El componente que falla debería haber recibido el evento
    # (se registra antes de fallar)
    assert len(failing_component.events_received) == 2
    assert failing_component.events_received[1]["type"] == "test.error"
    
    # Emitir otro evento normal para verificar que el sistema sigue funcionando
    await engine.emit_event("test.after_error", {"message": "after error"}, "test")
    
    # Dar tiempo para que se procesen los eventos
    await asyncio.sleep(0.1)
    
    # Verificar que ambos componentes recibieron el evento post-error
    assert len(normal_component.events_received) == 3
    assert normal_component.events_received[2]["type"] == "test.after_error"
    
    assert len(failing_component.events_received) == 3
    assert failing_component.events_received[2]["type"] == "test.after_error"
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que ambos componentes se detuvieron
    assert normal_component.stopped
    assert failing_component.stopped

@pytest.mark.asyncio
async def test_engine_non_blocking_emit_with_response():
    """Verificar que el motor puede manejar respuestas de eventos."""
    # Crear componente que retorna valores
    class RespondingComponent(Component):
        def __init__(self, name):
            super().__init__(name)
            self.started = False
            self.stopped = False
        
        async def start(self):
            self.started = True
        
        async def stop(self):
            self.stopped = True
            
        async def handle_event(self, event_type, data, source):
            if event_type == "test.request":
                return {"response": f"Hello from {self.name}", "value": data.get("value", 0) * 2}
            return None
    
    # Crear el motor
    engine = EngineNonBlocking(test_mode=True)
    
    # Registrar varios componentes que responden
    engine.register_component(RespondingComponent("comp1"))
    engine.register_component(RespondingComponent("comp2"))
    
    # Iniciar el motor
    await engine.start()
    
    # Emitir evento y recopilar respuestas
    responses = await engine.emit_event_with_response(
        "test.request", {"value": 5}, "test"
    )
    
    # Verificar las respuestas
    assert len(responses) == 2
    
    # Las respuestas pueden venir en cualquier orden, así que las ordenamos por nombre
    responses.sort(key=lambda r: r.get("response", ""))
    
    assert "Hello from comp1" in responses[0]["response"]
    assert responses[0]["value"] == 10
    
    assert "Hello from comp2" in responses[1]["response"]
    assert responses[1]["value"] == 10
    
    # Detener el motor
    await engine.stop()