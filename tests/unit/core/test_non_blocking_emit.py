"""
Prueba específica del método emit_event en el motor no bloqueante.

Este módulo contiene pruebas que se centran en el comportamiento del
método emit_event del motor no bloqueante, para verificar que
maneja correctamente los errores y la comunicación entre componentes.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, List

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging para suprimir mensajes durante las pruebas
logging.basicConfig(level=logging.CRITICAL)

# Componente básico para pruebas
class SimpleComponent(Component):
    """Componente simple para pruebas básicas."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.started = False
        self.stopped = False
    
    async def start(self) -> None:
        """Iniciar el componente."""
        self.started = True
    
    async def stop(self) -> None:
        """Detener el componente."""
        self.stopped = True
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar un evento."""
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        return None
    
    def clear_events(self) -> None:
        """Limpiar eventos registrados."""
        self.events.clear()

# Componente que falla con ciertos eventos
class FailComponent(SimpleComponent):
    """Componente que falla al procesar ciertos eventos."""
    
    def __init__(self, name: str, fail_on: List[str]):
        super().__init__(name)
        self.fail_on = fail_on
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento, fallando si es necesario."""
        # Primero registrar el evento
        await super().handle_event(event_type, data, source)
        
        # Luego verificar si debe fallar
        if event_type in self.fail_on:
            raise RuntimeError(f"Error simulado en {self.name}")
        
        return None

@pytest.mark.asyncio
async def test_engine_emit_event_basic():
    """Prueba básica del método emit_event del motor no bloqueante."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componente simple
    comp = SimpleComponent("test")
    
    # Registrar componente
    engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    
    # Limpiar eventos iniciales
    comp.clear_events()
    
    # Emitir evento
    await engine.emit_event("test_event", {"data": "test"}, "test_source")
    
    # Esperar un poco para que se procese el evento
    await asyncio.sleep(0.1)
    
    # Verificar que el componente recibió el evento
    assert len(comp.events) == 1, "El componente no recibió el evento"
    assert comp.events[0]["type"] == "test_event", "El tipo de evento es incorrecto"
    assert comp.events[0]["data"] == {"data": "test"}, "Los datos del evento son incorrectos"
    assert comp.events[0]["source"] == "test_source", "La fuente del evento es incorrecta"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que el componente se detuvo
    assert comp.stopped, "El componente no se detuvo"

@pytest.mark.asyncio
async def test_engine_emit_event_with_error():
    """Prueba del método emit_event cuando un componente genera un error."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    normal = SimpleComponent("normal")
    failing = FailComponent("failing", ["error_event"])
    
    # Registrar componentes
    engine.register_component(normal)
    engine.register_component(failing)
    
    # Iniciar motor
    await engine.start()
    
    # Limpiar eventos iniciales
    normal.clear_events()
    failing.clear_events()
    
    # Emitir evento normal
    await engine.emit_event("normal_event", {"data": "normal"}, "test")
    
    # Esperar un poco para que se procese el evento
    await asyncio.sleep(0.1)
    
    # Verificar que ambos componentes recibieron el evento normal
    assert len(normal.events) == 1, "El componente normal no recibió el evento normal"
    assert normal.events[0]["type"] == "normal_event"
    
    assert len(failing.events) == 1, "El componente failing no recibió el evento normal"
    assert failing.events[0]["type"] == "normal_event"
    
    # Emitir evento que causa error
    await engine.emit_event("error_event", {"data": "error"}, "test")
    
    # Esperar un poco para que se procese el evento
    await asyncio.sleep(0.1)
    
    # Verificar que ambos componentes recibieron el evento de error
    assert len(normal.events) == 2, "El componente normal no recibió el evento de error"
    assert normal.events[1]["type"] == "error_event"
    
    assert len(failing.events) == 2, "El componente failing no recibió el evento de error"
    assert failing.events[1]["type"] == "error_event"
    
    # Emitir otro evento normal para verificar que el sistema sigue funcionando
    await engine.emit_event("final_event", {"data": "final"}, "test")
    
    # Esperar un poco para que se procese el evento
    await asyncio.sleep(0.1)
    
    # Verificar que ambos componentes recibieron el evento final
    assert len(normal.events) == 3, "El componente normal no recibió el evento final"
    assert normal.events[2]["type"] == "final_event"
    
    assert len(failing.events) == 3, "El componente failing no recibió el evento final"
    assert failing.events[2]["type"] == "final_event"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que los componentes se detuvieron
    assert normal.stopped, "El componente normal no se detuvo"
    assert failing.stopped, "El componente failing no se detuvo"

@pytest.mark.asyncio
async def test_engine_emit_multiple_components():
    """Prueba del método emit_event con múltiples componentes."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear varios componentes
    components = [
        SimpleComponent(f"comp_{i}") 
        for i in range(5)
    ]
    failing = FailComponent("failing", ["error_event"])
    
    # Registrar todos los componentes
    for comp in components + [failing]:
        engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    
    # Limpiar eventos iniciales
    for comp in components + [failing]:
        comp.clear_events()
    
    # Emitir evento
    await engine.emit_event("test_event", {"data": "test"}, "test")
    
    # Esperar un poco para que se procese el evento
    await asyncio.sleep(0.1)
    
    # Verificar que todos los componentes recibieron el evento
    for comp in components + [failing]:
        assert len(comp.events) == 1, f"El componente {comp.name} no recibió el evento"
        assert comp.events[0]["type"] == "test_event"
    
    # Emitir evento que causa error
    await engine.emit_event("error_event", {"data": "error"}, "test")
    
    # Esperar un poco para que se procese el evento
    await asyncio.sleep(0.1)
    
    # Verificar que todos los componentes recibieron el evento de error
    for comp in components + [failing]:
        assert len(comp.events) == 2, f"El componente {comp.name} no recibió el evento de error"
        assert comp.events[1]["type"] == "error_event"
    
    # Emitir evento final
    await engine.emit_event("final_event", {"data": "final"}, "test")
    
    # Esperar un poco para que se procese el evento
    await asyncio.sleep(0.1)
    
    # Verificar que todos los componentes recibieron el evento final
    for comp in components + [failing]:
        assert len(comp.events) == 3, f"El componente {comp.name} no recibió el evento final"
        assert comp.events[2]["type"] == "final_event"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que todos los componentes se detuvieron
    for comp in components + [failing]:
        assert comp.stopped, f"El componente {comp.name} no se detuvo"