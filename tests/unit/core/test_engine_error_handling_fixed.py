"""
Test de manejo de errores utilizando la implementación no bloqueante del motor.

Este test demuestra que la implementación no bloqueante resuelve los problemas
de timeout que se encontraban en la versión original del motor.
"""

import pytest
import asyncio
import logging

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Silenciar logs durante las pruebas
logging.basicConfig(level=logging.CRITICAL)

# Implementación de un componente que falla intencionalmente
class FailingComponent(Component):
    """Componente que falla al manejar ciertos eventos para pruebas."""
    
    def __init__(self, name, should_fail=True):
        """
        Inicializar componente.
        
        Args:
            name: Nombre del componente
            should_fail: Si True, el componente fallará al manejar eventos test_event
        """
        super().__init__(name)
        self.events_received = []
        self.started = False
        self.stopped = False
        self.should_fail = should_fail
    
    async def start(self):
        """Iniciar el componente."""
        self.started = True
    
    async def stop(self):
        """Detener el componente."""
        self.stopped = True
    
    async def handle_event(self, event_type, data, source):
        """
        Manejar un evento, fallando si está configurado para hacerlo.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        
        # Simular un error si está configurado para fallar
        if self.should_fail and event_type == "test_event":
            raise Exception(f"Error simulado en el manejador de {self.name}")
        
        return None

@pytest.mark.asyncio
async def test_engine_error_handling_with_non_blocking():
    """
    Test que demuestra que la versión no bloqueante del motor maneja correctamente
    los errores en componentes sin causar bloqueos.
    """
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes de prueba
    normal_component = FailingComponent("normal", should_fail=False)
    failing_component = FailingComponent("failing", should_fail=True)
    
    # Registrar componentes
    engine.register_component(normal_component)
    engine.register_component(failing_component)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que ambos componentes se iniciaron
    assert normal_component.started
    assert failing_component.started
    
    # Limpiar eventos recibidos hasta ahora (eventos de inicio del sistema)
    normal_component.events_received.clear()
    failing_component.events_received.clear()
    
    # Emitir un evento que causará un error en el componente con fallo
    await engine.emit_event("test_event", {"message": "Test"}, "test_source")
    
    # Dar tiempo para que los eventos se procesen
    await asyncio.sleep(0.1)
    
    # Verificar que el componente normal recibió el evento
    assert len(normal_component.events_received) > 0
    test_events = [e for e in normal_component.events_received if e["type"] == "test_event"]
    assert len(test_events) > 0
    
    # Verificar que el componente con fallo recibió el evento antes de fallar
    assert len(failing_component.events_received) > 0
    test_events = [e for e in failing_component.events_received if e["type"] == "test_event"]
    assert len(test_events) > 0
    
    # Emitir otro evento para verificar que el sistema sigue funcionando
    # a pesar del error anterior
    await engine.emit_event("another_event", {"message": "After error"}, "test_source")
    
    # Dar tiempo para que los eventos se procesen
    await asyncio.sleep(0.1)
    
    # Verificar que el componente normal recibió el nuevo evento
    after_events = [e for e in normal_component.events_received if e["type"] == "another_event"]
    assert len(after_events) > 0
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que los componentes se detuvieron
    assert normal_component.stopped
    assert failing_component.stopped
    
    # Si llegamos aquí sin timeouts, la prueba es exitosa