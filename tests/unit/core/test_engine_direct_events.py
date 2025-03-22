"""
Prueba simplificada del motor con eventos directos.

Esta prueba utiliza un enfoque más directo para verificar que
el motor puede enviar eventos a los componentes correctamente.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, Optional, List

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventCollectorComponent(Component):
    """Componente que colecciona eventos para pruebas."""
    
    def __init__(self, name: str):
        """
        Inicializar componente.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.collected_events: List[Dict[str, Any]] = []
        self.started = False
        self.stopped = False
        logger.info(f"Componente {name} creado")
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Componente {self.name} iniciando")
        self.started = True
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Componente {self.name} deteniendo")
        self.stopped = True
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Coleccionar evento recibido.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            
        Returns:
            Ninguno
        """
        event_info = {
            "type": event_type,
            "data": data.copy() if data else {},
            "source": source
        }
        
        logger.info(f"Componente {self.name} recibió evento: {event_info}")
        self.collected_events.append(event_info)
        
        return None
    
    def get_events_of_type(self, event_type: str) -> List[Dict[str, Any]]:
        """
        Obtener eventos de un tipo específico.
        
        Args:
            event_type: Tipo de evento a buscar
            
        Returns:
            Lista de eventos del tipo especificado
        """
        return [e for e in self.collected_events if e["type"] == event_type]
    
    def has_event_of_type(self, event_type: str) -> bool:
        """
        Verificar si se ha recibido un evento de un tipo específico.
        
        Args:
            event_type: Tipo de evento a buscar
            
        Returns:
            True si se ha recibido al menos un evento del tipo, False en caso contrario
        """
        return any(e["type"] == event_type for e in self.collected_events)


@pytest.mark.asyncio
async def test_engine_direct_events():
    """
    Prueba que el motor entregue eventos correctamente.
    
    Esta prueba verifica que el motor EngineNonBlocking puede entregar
    eventos a los componentes registrados, y que los componentes reciben
    los eventos y sus datos correctamente.
    """
    # Crear un motor no bloqueante para pruebas
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear dos componentes para verificar la entrega de eventos
    comp1 = EventCollectorComponent("collector1")
    comp2 = EventCollectorComponent("collector2")
    
    # Registrar los componentes en el motor
    engine.register_component(comp1)
    engine.register_component(comp2)
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que el motor está iniciado
    assert engine.running, "El motor debería estar iniciado"
    
    # Verificar que los componentes están iniciados
    assert comp1.started, "El componente 1 debería estar iniciado"
    assert comp2.started, "El componente 2 debería estar iniciado"
    
    # Esperar un poco para permitir que se procesen eventos de inicio
    await asyncio.sleep(0.2)
    
    # Verificar que los componentes recibieron eventos de sistema
    system_events1 = comp1.get_events_of_type("system.started")
    system_events2 = comp2.get_events_of_type("system.started")
    
    logger.info(f"Eventos de sistema recibidos por comp1: {system_events1}")
    logger.info(f"Eventos de sistema recibidos por comp2: {system_events2}")
    
    assert len(system_events1) > 0, "El componente 1 debería haber recibido al menos un evento system.started"
    assert len(system_events2) > 0, "El componente 2 debería haber recibido al menos un evento system.started"
    
    # Enviar evento personalizado
    custom_data = {"value": 42, "name": "test"}
    await engine.emit_event("custom.test", custom_data, "test")
    
    # Permitir que se procese el evento
    await asyncio.sleep(0.2)
    
    # Verificar que los componentes recibieron el evento personalizado
    custom_events1 = comp1.get_events_of_type("custom.test")
    custom_events2 = comp2.get_events_of_type("custom.test")
    
    logger.info(f"Eventos personalizados recibidos por comp1: {custom_events1}")
    logger.info(f"Eventos personalizados recibidos por comp2: {custom_events2}")
    
    assert len(custom_events1) > 0, "El componente 1 debería haber recibido el evento personalizado"
    assert len(custom_events2) > 0, "El componente 2 debería haber recibido el evento personalizado"
    
    # Verificar que los datos del evento son correctos
    assert custom_events1[0]["data"]["value"] == 42, "Los datos del evento deberían ser correctos (comp1)"
    assert custom_events2[0]["data"]["name"] == "test", "Los datos del evento deberían ser correctos (comp2)"
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que el motor está detenido
    assert not engine.running, "El motor debería estar detenido"
    
    # Verificar que los componentes están detenidos
    assert comp1.stopped, "El componente 1 debería estar detenido"
    assert comp2.stopped, "El componente 2 debería estar detenido"