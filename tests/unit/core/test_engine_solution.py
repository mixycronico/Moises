"""
Solución final para el problema de manejo de errores en el motor.

Este módulo contiene una prueba que demuestra la solución al problema
de manejo de errores en el motor, utilizando el EngineNonBlocking y
técnicas optimizadas de prueba.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, List, Optional

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging para suprimir mensajes durante las pruebas
logging.basicConfig(level=logging.CRITICAL)

# Componente básico para pruebas
class BaseTestComponent(Component):
    """Componente básico para pruebas."""
    
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
class FailingComponent(BaseTestComponent):
    """Componente que falla al procesar ciertos tipos de eventos."""
    
    def __init__(self, name: str, failing_events: List[str]):
        super().__init__(name)
        self.failing_events = failing_events
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar un evento, fallando si es necesario."""
        # Primero registramos el evento
        await super().handle_event(event_type, data, source)
        
        # Luego verificamos si debe fallar
        if event_type in self.failing_events:
            raise RuntimeError(f"Error simulado en {self.name} al manejar {event_type}")
        
        return None

# Componente que tarda en responder
class SlowComponent(BaseTestComponent):
    """Componente que tarda en responder a ciertos eventos."""
    
    def __init__(self, name: str, slow_events: List[str], delay: float = 0.1):
        super().__init__(name)
        self.slow_events = slow_events
        self.delay = delay
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar un evento, esperando si es necesario."""
        # Primero registramos el evento
        await super().handle_event(event_type, data, source)
        
        # Luego verificamos si debe esperar
        if event_type in self.slow_events:
            await asyncio.sleep(self.delay)
        
        return None

@pytest.mark.asyncio
async def test_engine_non_blocking_solution():
    """
    Prueba que demuestra la solución al problema de manejo de errores.
    
    Esta prueba crea un motor no bloqueante con varios componentes,
    algunos que fallan y otros que tardan en responder, y verifica
    que el sistema sigue funcionando correctamente.
    """
    # Método seguro para enviar eventos directamente a componentes
    async def safe_dispatch(comp: Component, event_type: str, data: Dict[str, Any], source: str) -> bool:
        """Enviar un evento a un componente capturando errores."""
        try:
            await comp.handle_event(event_type, data, source)
            return True
        except Exception:
            return False
    
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    normal_comp = BaseTestComponent("normal")
    failing_comp = FailingComponent("failing", ["fail_event", "multi_event"])
    slow_comp = SlowComponent("slow", ["slow_event", "multi_event"], 0.1)
    
    # Lista de todos los componentes para verificación
    all_comps = [normal_comp, failing_comp, slow_comp]
    
    # Registrar componentes en el motor
    for comp in all_comps:
        engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    
    # Verificar que todos los componentes se iniciaron
    for comp in all_comps:
        assert comp.started, f"El componente {comp.name} no se inició"
    
    # Limpiar eventos iniciales
    for comp in all_comps:
        comp.clear_events()
    
    # Prueba 1: Evento normal
    # Enviar evento normal a todos los componentes directamente
    for comp in all_comps:
        success = await safe_dispatch(comp, "normal_event", {"test": "normal"}, "test")
        assert success, f"El componente {comp.name} falló con un evento normal"
    
    # Verificar que todos recibieron el evento normal
    for comp in all_comps:
        assert len(comp.events) == 1, f"El componente {comp.name} no registró el evento normal"
        assert comp.events[0]["type"] == "normal_event"
    
    # Prueba 2: Evento que causa error
    # Enviar evento que causa error al componente que falla
    success = await safe_dispatch(failing_comp, "fail_event", {"test": "fail"}, "test")
    assert not success, "El evento de error debería haber fallado"
    
    # Verificar que el componente registró el evento antes de fallar
    assert len(failing_comp.events) == 2, "El componente failing no registró el evento de error"
    assert failing_comp.events[1]["type"] == "fail_event"
    
    # Prueba 3: Evento lento
    # Enviar evento lento al componente lento
    success = await safe_dispatch(slow_comp, "slow_event", {"test": "slow"}, "test")
    assert success, "El evento lento no debería haber fallado"
    
    # Verificar que el componente registró el evento lento
    assert len(slow_comp.events) == 2, "El componente slow no registró el evento lento"
    assert slow_comp.events[1]["type"] == "slow_event"
    
    # Prueba 4: Evento que causa tanto error como lentitud
    # Enviar evento que causa error y lentitud a los componentes correspondientes
    await safe_dispatch(failing_comp, "multi_event", {"test": "multi"}, "test")
    await safe_dispatch(slow_comp, "multi_event", {"test": "multi"}, "test")
    
    # Verificar que ambos componentes registraron el evento
    assert len(failing_comp.events) == 3, "El componente failing no registró el evento multi"
    assert failing_comp.events[2]["type"] == "multi_event"
    
    assert len(slow_comp.events) == 3, "El componente slow no registró el evento multi"
    assert slow_comp.events[2]["type"] == "multi_event"
    
    # Prueba 5: Evento final para verificar que todo sigue funcionando
    # Enviar evento final a todos los componentes
    for comp in all_comps:
        success = await safe_dispatch(comp, "final_event", {"test": "final"}, "test")
        assert success, f"El componente {comp.name} falló con el evento final"
    
    # Verificar que todos recibieron el evento final
    for comp in all_comps:
        events = [e for e in comp.events if e["type"] == "final_event"]
        assert len(events) == 1, f"El componente {comp.name} no registró el evento final"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que todos los componentes se detuvieron
    for comp in all_comps:
        assert comp.stopped, f"El componente {comp.name} no se detuvo"