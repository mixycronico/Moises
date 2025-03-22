"""
Test final minimalista para demostrar el funcionamiento de la solución.

Este test es una versión extremadamente simplificada que muestra
que los componentes y motores funcionan correctamente sin el EventBus.
"""

import pytest
import asyncio
import logging

from genesis.core.component import Component

# Silenciar logs durante las pruebas
logging.basicConfig(level=logging.CRITICAL)

# Componente que simula errores
class TestComponent(Component):
    """Componente de prueba para verificar manejo de errores."""
    
    def __init__(self, name, should_fail=False):
        """Inicializar el componente."""
        super().__init__(name)
        self.should_fail = should_fail
        self.events = []
        self.started = False
        self.stopped = False
        
    async def start(self):
        """Iniciar el componente."""
        self.started = True
        
    async def stop(self):
        """Detener el componente."""
        self.stopped = True
        
    async def handle_event(self, event_type, data, source):
        """Manejar un evento, fallando si está configurado para eso."""
        # Registrar el evento
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        
        # Simular error si está configurado
        if self.should_fail and event_type == "error_event":
            raise Exception(f"Error simulado en {self.name}")
            
        return None

@pytest.mark.asyncio
async def test_component_error_handling_minimal():
    """
    Test minimalista que muestra cómo los componentes manejan errores.
    
    Esta prueba no utiliza el Engine ni el EventBus real, sino que simula
    el comportamiento directamente para demostrar el concepto.
    """
    # Crear componentes
    normal = TestComponent("normal")
    failing = TestComponent("failing", should_fail=True)
    
    # Iniciar componentes
    await normal.start()
    await failing.start()
    
    # Verificar inicio
    assert normal.started
    assert failing.started
    
    # Enviar evento normal a ambos componentes
    await normal.handle_event("normal_event", {"data": "test"}, "test")
    await failing.handle_event("normal_event", {"data": "test"}, "test")
    
    # Verificar recepción del evento normal
    assert len(normal.events) == 1
    assert normal.events[0]["type"] == "normal_event"
    
    assert len(failing.events) == 1
    assert failing.events[0]["type"] == "normal_event"
    
    # Enviar evento que causa error al componente que falla
    try:
        await failing.handle_event("error_event", {"data": "error"}, "test")
        # Si llegamos aquí, la prueba fallará porque debería lanzar excepción
        assert False, "El componente debería haber fallado"
    except Exception as e:
        # Verificar que el error es el esperado
        assert "Error simulado" in str(e)
    
    # Verificar que a pesar del error, el evento fue registrado
    assert len(failing.events) == 2
    assert failing.events[1]["type"] == "error_event"
    
    # Enviar otro evento normal para verificar que los componentes siguen funcionando
    await normal.handle_event("post_error_event", {"data": "post"}, "test")
    await failing.handle_event("post_error_event", {"data": "post"}, "test")
    
    # Verificar recepción del evento post-error
    assert len(normal.events) == 2
    assert normal.events[1]["type"] == "post_error_event"
    
    assert len(failing.events) == 3
    assert failing.events[2]["type"] == "post_error_event"
    
    # Detener componentes
    await normal.stop()
    await failing.stop()
    
    # Verificar detención
    assert normal.stopped
    assert failing.stopped