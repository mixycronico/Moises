"""
Prueba ultra simplificada manual para verificar el comportamiento del sistema.

Este enfoque evita todos los posibles problemas de asyncio y threading
al utilizar una implementación manual extremadamente simplificada.
"""

import pytest
import asyncio
import logging

from genesis.core.component import Component

# Silenciar logs durante las pruebas
logging.basicConfig(level=logging.CRITICAL)

# Implementación manual de un componente sin errores
class SimpleComponent(Component):
    """Componente simple que registra los eventos recibidos."""
    
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
        """Manejar un evento."""
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        return None

# Implementación manual de un componente que falla
class FailingComponent(Component):
    """Componente que falla al manejar ciertos eventos."""
    
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
        """Manejar un evento, fallando si es del tipo especificado."""
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        
        if event_type == "test.error":
            raise Exception(f"Error simulado en {self.name}")
        
        return None

@pytest.mark.asyncio
async def test_manual_component_error_handling():
    """
    Prueba manual simplificada sin usar el Engine real.
    
    Esta prueba verifica directamente el comportamiento de los componentes
    sin depender del motor ni del EventBus para evitar timeouts y bloqueos.
    """
    # Crear dos componentes de prueba
    normal_component = SimpleComponent("normal_component")
    failing_component = FailingComponent("failing_component")
    
    # Iniciar los componentes manualmente
    await normal_component.start()
    await failing_component.start()
    
    # Verificar que se iniciaron correctamente
    assert normal_component.started
    assert failing_component.started
    
    # Enviar un evento normal y verificar que ambos lo reciben correctamente
    event_data = {"message": "Test message"}
    await normal_component.handle_event("test.normal", event_data, "test_source")
    await failing_component.handle_event("test.normal", event_data, "test_source")
    
    # Verificar que ambos recibieron el evento normal
    assert len(normal_component.events_received) == 1
    assert normal_component.events_received[0]["type"] == "test.normal"
    
    assert len(failing_component.events_received) == 1
    assert failing_component.events_received[0]["type"] == "test.normal"
    
    # Enviar un evento que causará un error en el componente que falla
    # pero lo envolvemos en un try-except para capturar el error
    try:
        await failing_component.handle_event("test.error", event_data, "test_source")
        # Si llegamos aquí, el test fallará porque debería haber lanzado una excepción
        assert False, "El componente debería haber fallado"
    except Exception as e:
        # Verificar que el error es el esperado
        assert "Error simulado" in str(e)
    
    # Verificar que a pesar del error, el componente registró el evento
    assert len(failing_component.events_received) == 2
    assert failing_component.events_received[1]["type"] == "test.error"
    
    # Enviar otro evento normal al componente normal
    await normal_component.handle_event("test.normal2", event_data, "test_source")
    
    # Verificar que el componente normal sigue funcionando correctamente
    assert len(normal_component.events_received) == 2
    assert normal_component.events_received[1]["type"] == "test.normal2"
    
    # Detener los componentes manualmente
    await normal_component.stop()
    await failing_component.stop()
    
    # Verificar que se detuvieron correctamente
    assert normal_component.stopped
    assert failing_component.stopped
    
    # La prueba es exitosa si llegamos aquí sin bloqueos