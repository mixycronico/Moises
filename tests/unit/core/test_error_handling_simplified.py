"""
Prueba simplificada de manejo de errores para el Engine no bloqueante.

Este módulo contiene pruebas más simples y controladas para verificar
el manejo de errores en componentes sin causar timeouts.
"""

import pytest
import asyncio
import logging

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging para suprimir mensajes durante las pruebas
logging.basicConfig(level=logging.CRITICAL)

# Componente básico para pruebas
class TestComponent(Component):
    """Implementación básica de componente para pruebas."""
    
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
        # Registrar todos los eventos
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        
        return None

# Componente que falla al recibir un evento específico
class ErrorComponent(TestComponent):
    """Componente que genera un error al recibir un evento específico."""
    
    async def handle_event(self, event_type, data, source):
        # Registrar todos los eventos (llamada al método padre)
        await super().handle_event(event_type, data, source)
        
        # Generar error para eventos específicos
        if event_type == "fail_event":
            raise RuntimeError(f"Error simulado en {self.name}")
        
        return None

@pytest.mark.asyncio
async def test_engine_with_error_component():
    """Prueba el manejo de errores en componentes con el Engine no bloqueante."""
    # Crear motor en modo de prueba
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes - uno normal y uno que genere errores
    normal = TestComponent("normal")
    error_comp = ErrorComponent("error")
    
    # Registrar componentes
    engine.register_component(normal)
    engine.register_component(error_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Verificar que el componente con error se inició
    assert error_comp.started
    
    # Limpiar eventos iniciales
    error_comp.events.clear()
    
    # Enviar evento que causará el error
    await engine.emit_event("fail_event", {"test": "data"}, "test")
    
    # Pequeña pausa para asegurarnos que se procese
    await asyncio.sleep(0.1)
    
    # Verificar que el evento se registró antes de que ocurriera el error
    assert len(error_comp.events) == 1
    assert error_comp.events[0]["type"] == "fail_event"
    
    # Enviar otro evento para verificar que el sistema sigue funcionando
    await engine.emit_event("normal_event", {"test": "after_error"}, "test")
    
    # Pequeña pausa para asegurarnos que se procese
    await asyncio.sleep(0.1)
    
    # Verificar que el segundo evento también se recibió
    assert len(error_comp.events) == 2
    assert error_comp.events[1]["type"] == "normal_event"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que el componente se detuvo
    assert error_comp.stopped

@pytest.mark.asyncio
async def test_multiple_error_components():
    """Prueba varios componentes que generan errores simultáneamente."""
    # Crear motor en modo de prueba
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear varios componentes de error
    error_comps = [
        ErrorComponent(f"error_{i}") 
        for i in range(3)
    ]
    
    # Registrar todos los componentes
    for comp in error_comps:
        engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    
    # Limpiar eventos iniciales
    for comp in error_comps:
        comp.events.clear()
    
    # Enviar evento que causará errores en todos los componentes
    await engine.emit_event("fail_event", {"test": "data"}, "test")
    
    # Pequeña pausa para asegurarnos que se procese
    await asyncio.sleep(0.1)
    
    # Verificar que todos los componentes recibieron el evento antes del error
    for comp in error_comps:
        assert len(comp.events) == 1
        assert comp.events[0]["type"] == "fail_event"
    
    # Enviar otro evento normal
    await engine.emit_event("normal_event", {"test": "after_error"}, "test")
    
    # Pequeña pausa para asegurarnos que se procese
    await asyncio.sleep(0.1)
    
    # Verificar que todos recibieron el segundo evento
    for comp in error_comps:
        assert len(comp.events) == 2
        assert comp.events[1]["type"] == "normal_event"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que todos los componentes se detuvieron
    for comp in error_comps:
        assert comp.stopped