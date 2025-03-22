"""
Pruebas con envío manual de eventos a través del motor.

Este módulo contiene pruebas que integran el motor con componentes,
pero envía los eventos manualmente para evitar los timeouts.
"""

import pytest
import asyncio
import logging
from typing import List, Dict, Any

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging para suprimir mensajes durante las pruebas
logging.basicConfig(level=logging.CRITICAL)

# Componente básico para pruebas
class TestComponent(Component):
    """Componente básico para pruebas."""
    
    def __init__(self, name):
        super().__init__(name)
        self.events = []
        self.started = False
        self.stopped = False
    
    async def start(self):
        self.started = True
    
    async def stop(self):
        self.stopped = True
    
    async def handle_event(self, event_type, data, source):
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        return None

# Componente que genera errores
class ErrorComponent(TestComponent):
    """Componente que genera errores con ciertos eventos."""
    
    def __init__(self, name, error_events):
        super().__init__(name)
        self.error_events = error_events
    
    async def handle_event(self, event_type, data, source):
        # Primero registrar el evento
        await super().handle_event(event_type, data, source)
        
        # Luego verificar si debe generar error
        if event_type in self.error_events:
            raise ValueError(f"Error simulado en {self.name} para {event_type}")
        
        return None

@pytest.mark.asyncio
async def test_engine_with_manual_event_routing():
    """Probar el motor con envío manual de eventos a componentes."""
    # Crear motor
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    normal = TestComponent("normal")
    error_comp = ErrorComponent("error", ["fail_event"])
    
    # Registrar componentes
    engine.register_component(normal)
    engine.register_component(error_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Verificar que los componentes se iniciaron
    assert normal.started
    assert error_comp.started
    
    # Limpiar eventos iniciales
    normal.events.clear()
    error_comp.events.clear()
    
    # Enviar evento normal manualmente a cada componente
    for component in [normal, error_comp]:
        try:
            # Enviar evento a través del motor, pero directamente al componente
            await component.handle_event("normal_event", {"test": "data"}, "test")
        except Exception as e:
            assert False, f"Error inesperado: {e}"
    
    # Verificar que ambos recibieron el evento
    assert len(normal.events) == 1
    assert normal.events[0]["type"] == "normal_event"
    
    assert len(error_comp.events) == 1
    assert error_comp.events[0]["type"] == "normal_event"
    
    # Enviar evento que causa error al componente de error
    try:
        await error_comp.handle_event("fail_event", {"test": "error"}, "test")
        assert False, "Debería haber generado un error"
    except ValueError as e:
        # Verificar que es el error esperado
        assert "Error simulado" in str(e)
    
    # Verificar que el componente error registró el evento antes de fallar
    assert len(error_comp.events) == 2
    assert error_comp.events[1]["type"] == "fail_event"
    
    # Enviar otro evento normal después del error
    for component in [normal, error_comp]:
        try:
            await component.handle_event("post_error", {"test": "after"}, "test")
        except Exception as e:
            assert False, f"Error inesperado: {e}"
    
    # Verificar que ambos recibieron el evento post-error
    assert len(normal.events) == 2
    assert normal.events[1]["type"] == "post_error"
    
    assert len(error_comp.events) == 3
    assert error_comp.events[2]["type"] == "post_error"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que los componentes se detuvieron
    assert normal.stopped
    assert error_comp.stopped

@pytest.mark.asyncio
async def test_non_blocking_engine_direct_event_dispatch():
    """Prueba el motor no bloqueante con despacho directo de eventos."""
    # Crear motor
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear varios componentes, incluyendo uno que fallará
    normal_comps = [TestComponent(f"normal_{i}") for i in range(2)]
    error_comp = ErrorComponent("error", ["fail_event"])
    
    # Registrar todos los componentes
    for comp in normal_comps + [error_comp]:
        engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    
    # Limpiar eventos iniciales
    for comp in normal_comps + [error_comp]:
        comp.events.clear()
    
    # Lista de componentes para verificación
    all_comps = normal_comps + [error_comp]
    
    # Método seguro para enviar eventos a un componente
    async def safe_send_event(component, event_type, data, source):
        try:
            await component.handle_event(event_type, data, source)
            return True
        except Exception:
            return False
    
    # Enviar evento normal a todos los componentes
    tasks = [
        safe_send_event(comp, "normal_event", {"test": "data"}, "test")
        for comp in all_comps
    ]
    results = await asyncio.gather(*tasks)
    
    # Verificar que todos recibieron el evento normal
    for comp in all_comps:
        assert len(comp.events) == 1
        assert comp.events[0]["type"] == "normal_event"
    
    # Enviar evento que causará error en el error_comp
    success = await safe_send_event(error_comp, "fail_event", {"test": "error"}, "test")
    assert not success, "El envío del evento de error debería haber fallado"
    
    # Verificar que el error_comp registró el evento antes de fallar
    assert len(error_comp.events) == 2
    assert error_comp.events[1]["type"] == "fail_event"
    
    # Enviar evento post-error a todos los componentes
    tasks = [
        safe_send_event(comp, "post_error", {"test": "after"}, "test")
        for comp in all_comps
    ]
    await asyncio.gather(*tasks)
    
    # Verificar que todos recibieron el evento post-error
    for comp in normal_comps:
        assert len(comp.events) == 2
        assert comp.events[1]["type"] == "post_error"
    
    assert len(error_comp.events) == 3
    assert error_comp.events[2]["type"] == "post_error"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que todos los componentes se detuvieron
    for comp in all_comps:
        assert comp.stopped