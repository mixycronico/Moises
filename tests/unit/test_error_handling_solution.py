"""
Solución final para pruebas de manejo de errores en el sistema Genesis.

Este módulo implementa una prueba exhaustiva que verifica el correcto
funcionamiento del manejo de errores en componentes utilizando el motor
no bloqueante que hemos desarrollado.
"""

import pytest
import asyncio
import logging
from typing import List, Dict, Any, Optional

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging básico para las pruebas
logging.basicConfig(level=logging.CRITICAL)

# Componente básico para pruebas
class BasicComponent(Component):
    """Componente básico para pruebas que registra todos los eventos recibidos."""
    
    def __init__(self, name: str):
        """
        Inicializar componente.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.events_received: List[Dict[str, Any]] = []
        self.started = False
        self.stopped = False
        
    async def start(self) -> None:
        """Iniciar el componente."""
        self.started = True
        
    async def stop(self) -> None:
        """Detener el componente."""
        self.stopped = True
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento y registrarlo.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Ninguno
        """
        # Registrar el evento recibido
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        return None

# Componente que falla intencionalmente para pruebas
class FailingComponent(BasicComponent):
    """Componente que falla intencionalmente al recibir ciertos tipos de eventos."""
    
    def __init__(self, name: str, failing_event_types: List[str]):
        """
        Inicializar componente que falla con ciertos eventos.
        
        Args:
            name: Nombre del componente
            failing_event_types: Lista de tipos de eventos que causan fallo
        """
        super().__init__(name)
        self.failing_event_types = failing_event_types
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento, fallando si es uno de los tipos especificados.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Ninguno
            
        Raises:
            Exception: Si el tipo de evento está en la lista de eventos que fallan
        """
        # Primero registrar el evento (esto siempre debe ocurrir)
        await super().handle_event(event_type, data, source)
        
        # Verificar si este tipo de evento debe causar un fallo
        if event_type in self.failing_event_types:
            raise Exception(f"Error simulado en {self.name} al manejar {event_type}")
            
        return None

# Componente que bloquea temporalmente para pruebas
class BlockingComponent(BasicComponent):
    """Componente que bloquea temporalmente al recibir ciertos tipos de eventos."""
    
    def __init__(self, name: str, blocking_event_types: List[str], block_time: float):
        """
        Inicializar componente que bloquea con ciertos eventos.
        
        Args:
            name: Nombre del componente
            blocking_event_types: Lista de tipos de eventos que causan bloqueo
            block_time: Tiempo en segundos que el componente se bloquea
        """
        super().__init__(name)
        self.blocking_event_types = blocking_event_types
        self.block_time = block_time
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento, bloqueando si es uno de los tipos especificados.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Ninguno
        """
        # Primero registrar el evento
        await super().handle_event(event_type, data, source)
        
        # Bloquear si es necesario
        if event_type in self.blocking_event_types:
            await asyncio.sleep(self.block_time)
            
        return None

@pytest.mark.asyncio
async def test_complete_error_handling_solution():
    """
    Prueba completa que verifica que el sistema maneja correctamente
    errores y bloqueos en componentes utilizando el motor no bloqueante.
    """
    # Crear motor no bloqueante en modo de prueba
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes para la prueba
    components = [
        BasicComponent("basic1"),
        BasicComponent("basic2"),
        FailingComponent("failing", ["error_event", "multi_error"]),
        BlockingComponent("blocking", ["block_event", "multi_error"], 0.2)
    ]
    
    # Registrar todos los componentes en el motor
    for component in components:
        engine.register_component(component)
        
    # Iniciar el motor
    await engine.start()
    
    # Verificar que todos los componentes se iniciaron
    for component in components:
        assert component.started, f"El componente {component.name} no se inició"
        
    # Limpiar eventos iniciales (system.started) para simplificar las verificaciones
    for component in components:
        component.events_received.clear()
        
    # Emitir evento normal para establecer línea base
    await engine.emit_event("normal_event", {"data": "test"}, "test_source")
    
    # Dar tiempo para que los eventos se procesen
    await asyncio.sleep(0.1)
    
    # Verificar que todos los componentes recibieron el evento normal
    for component in components:
        normal_events = [e for e in component.events_received if e["type"] == "normal_event"]
        assert len(normal_events) == 1, f"El componente {component.name} no recibió el evento normal"
        
    # Emitir evento que causa error en un componente
    await engine.emit_event("error_event", {"data": "error"}, "test_source")
    
    # Dar tiempo para que los eventos se procesen
    await asyncio.sleep(0.1)
    
    # Verificar que todos los componentes recibieron el evento de error
    for component in components:
        error_events = [e for e in component.events_received if e["type"] == "error_event"]
        assert len(error_events) == 1, f"El componente {component.name} no recibió el evento de error"
        
    # Emitir evento que causa bloqueo en un componente
    await engine.emit_event("block_event", {"data": "block"}, "test_source")
    
    # Dar tiempo para que los eventos se procesen (incluyendo el tiempo de bloqueo)
    await asyncio.sleep(0.3)
    
    # Verificar que todos los componentes recibieron el evento de bloqueo
    for component in components:
        block_events = [e for e in component.events_received if e["type"] == "block_event"]
        assert len(block_events) == 1, f"El componente {component.name} no recibió el evento de bloqueo"
        
    # Emitir evento que causa tanto error como bloqueo en diferentes componentes
    await engine.emit_event("multi_error", {"data": "multiple"}, "test_source")
    
    # Dar tiempo para que los eventos se procesen (incluyendo el tiempo de bloqueo)
    await asyncio.sleep(0.3)
    
    # Verificar que todos los componentes recibieron el evento múltiple
    for component in components:
        multi_events = [e for e in component.events_received if e["type"] == "multi_error"]
        assert len(multi_events) == 1, f"El componente {component.name} no recibió el evento múltiple"
        
    # Verificar que el sistema aún funciona después de errores y bloqueos
    await engine.emit_event("final_event", {"data": "final"}, "test_source")
    
    # Dar tiempo para que los eventos se procesen
    await asyncio.sleep(0.1)
    
    # Verificar que todos los componentes recibieron el evento final
    for component in components:
        final_events = [e for e in component.events_received if e["type"] == "final_event"]
        assert len(final_events) == 1, f"El componente {component.name} no recibió el evento final"
        
    # Finalmente, verificar que cada componente recibió exactamente 5 eventos en total
    # (normal, error, block, multi, final)
    for component in components:
        assert len(component.events_received) == 5, f"El componente {component.name} no recibió los 5 eventos esperados"
        
    # Detener el motor
    await engine.stop()
    
    # Verificar que todos los componentes se detuvieron
    for component in components:
        assert component.stopped, f"El componente {component.name} no se detuvo"
        
    # Si llegamos aquí sin timeouts ni excepciones, la prueba es exitosa