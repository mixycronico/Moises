"""
Prueba ultra simplificada para verificar el manejo de errores en EventBus.

Esta prueba se enfoca únicamente en verificar que el EventBus maneje
correctamente las excepciones durante el procesamiento de eventos.
"""

import pytest
import asyncio
import logging
from genesis.core.event_bus import EventBus

# Silenciar logs durante las pruebas
logging.basicConfig(level=logging.CRITICAL)

@pytest.mark.asyncio
async def test_event_bus_error_handling_basic():
    """
    Probar que el EventBus maneja correctamente las excepciones 
    durante el procesamiento de eventos.
    """
    # Crear un event_bus en modo test
    event_bus = EventBus(test_mode=True)
    
    # Rastrear llamadas a handlers
    handlers_called = []
    
    # Handler normal que siempre funciona
    async def normal_handler(event_type, data, source):
        handlers_called.append(("normal", event_type, data))
        
    # Handler que siempre falla
    async def error_handler(event_type, data, source):
        handlers_called.append(("error", event_type, data))
        raise Exception("Error simulado para test")
        
    # Handler que se ejecuta después del error
    async def after_error_handler(event_type, data, source):
        handlers_called.append(("after", event_type, data))
    
    # Registrar handlers
    event_bus.subscribe("test_event", normal_handler, priority=100)  # Alta prioridad, se ejecuta primero
    event_bus.subscribe("test_event", error_handler, priority=50)    # Media prioridad, se ejecuta segundo
    event_bus.subscribe("test_event", after_error_handler, priority=1)  # Baja prioridad, se ejecuta último
    
    # Emitir un evento
    await event_bus.emit("test_event", {"value": 42}, "test_source")
    
    # Verificar que todos los handlers fueron llamados
    assert len(handlers_called) == 3, f"Se esperaban 3 llamadas a handlers, pero se encontraron {len(handlers_called)}"
    
    # Verificar el orden y que todos fueron llamados
    assert handlers_called[0][0] == "normal", "El primer handler llamado debería ser el normal"
    assert handlers_called[1][0] == "error", "El segundo handler llamado debería ser el error"
    assert handlers_called[2][0] == "after", "El tercer handler llamado debería ser el after_error"
    
    # Verificar que todos recibieron los datos correctos
    for handler_name, event_type, data in handlers_called:
        assert event_type == "test_event", f"Handler {handler_name} recibió un tipo de evento incorrecto"
        assert data["value"] == 42, f"Handler {handler_name} recibió datos incorrectos"

if __name__ == "__main__":
    asyncio.run(test_event_bus_error_handling_basic())