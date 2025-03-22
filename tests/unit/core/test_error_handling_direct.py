"""
Prueba directa de manejo de errores sin depender del motor.

Este módulo contiene pruebas directas de los componentes
sin pasar por el motor, para verificar que funcionan correctamente
antes de integrarlos.
"""

import pytest
import asyncio
import logging
from typing import List, Dict, Any

from genesis.core.component import Component

# Configurar logging para suprimir mensajes durante las pruebas
logging.basicConfig(level=logging.CRITICAL)

# Componente básico para pruebas
class BasicComponent(Component):
    """Implementación básica de un componente para pruebas."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.events = []  # Registro de eventos recibidos
        self.started = False
        self.stopped = False
    
    async def start(self):
        """Iniciar el componente."""
        self.started = True
    
    async def stop(self):
        """Detener el componente."""
        self.stopped = True
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str):
        """Manejar un evento."""
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        return None
    
    def clear_events(self):
        """Limpiar los eventos registrados."""
        self.events.clear()

# Componente con manejo de errores
class ErrorComponent(BasicComponent):
    """Componente que genera un error al recibir ciertos eventos."""
    
    def __init__(self, name: str, failing_events: List[str]):
        super().__init__(name)
        self.failing_events = failing_events
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str):
        """Manejar un evento, fallando si es necesario."""
        # Primero, registrar el evento
        await super().handle_event(event_type, data, source)
        
        # Generar error si es un evento problemático
        if event_type in self.failing_events:
            raise RuntimeError(f"Error simulado en {self.name} al manejar {event_type}")
        
        return None

@pytest.mark.asyncio
async def test_error_component_basic_functionality():
    """Verificar la funcionalidad básica del componente con errores."""
    # Crear componente
    comp = ErrorComponent("test", ["fail_event"])
    
    # Iniciar componente
    await comp.start()
    assert comp.started
    
    # Probar con evento normal
    await comp.handle_event("normal_event", {"test": "data"}, "test")
    assert len(comp.events) == 1
    assert comp.events[0]["type"] == "normal_event"
    
    # Probar con evento que causa error
    try:
        await comp.handle_event("fail_event", {"test": "error"}, "test")
        # Si llegamos aquí, la prueba falla
        assert False, "Debería haber generado un error"
    except RuntimeError as e:
        # Verificar que es el error esperado
        assert "Error simulado" in str(e)
    
    # Verificar que el evento se registró antes de generar el error
    assert len(comp.events) == 2
    assert comp.events[1]["type"] == "fail_event"
    
    # Probar que el componente sigue funcionando después del error
    await comp.handle_event("another_event", {"test": "after"}, "test")
    assert len(comp.events) == 3
    assert comp.events[2]["type"] == "another_event"
    
    # Detener componente
    await comp.stop()
    assert comp.stopped

@pytest.mark.asyncio
async def test_multiple_error_components():
    """Verificar que múltiples componentes manejan errores correctamente."""
    # Crear varios componentes
    comps = [
        ErrorComponent(f"comp_{i}", ["fail_event"]) 
        for i in range(3)
    ]
    
    # Iniciar componentes
    for comp in comps:
        await comp.start()
        assert comp.started
    
    # Probar con evento normal en todos
    for comp in comps:
        await comp.handle_event("normal_event", {"test": "data"}, "test")
        assert len(comp.events) == 1
        assert comp.events[0]["type"] == "normal_event"
    
    # Probar con evento que causa error en todos
    for comp in comps:
        try:
            await comp.handle_event("fail_event", {"test": "error"}, "test")
            # Si llegamos aquí, la prueba falla
            assert False, "Debería haber generado un error"
        except RuntimeError as e:
            # Verificar que es el error esperado
            assert "Error simulado" in str(e)
    
    # Verificar que cada componente registró el evento antes del error
    for comp in comps:
        assert len(comp.events) == 2
        assert comp.events[1]["type"] == "fail_event"
    
    # Probar que todos siguen funcionando después del error
    for comp in comps:
        await comp.handle_event("after_event", {"test": "after"}, "test")
        assert len(comp.events) == 3
        assert comp.events[2]["type"] == "after_event"
    
    # Detener componentes
    for comp in comps:
        await comp.stop()
        assert comp.stopped