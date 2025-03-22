"""
Test ultra minimalista para componentes.

Este módulo contiene pruebas directas de componentes
sin utilizar el motor, para identificar problemas básicos.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any

from genesis.core.component import Component

# Configurar logging mínimo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalComponent(Component):
    """Componente ultra minimalista para pruebas elementales."""
    
    def __init__(self, name: str):
        """Inicializar componente con nombre."""
        super().__init__(name)
        self.event_count = 0
        self.events = []
    
    async def start(self) -> None:
        """Iniciar componente (implementación vacía)."""
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener componente (implementación vacía)."""
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento de la forma más simple posible.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
        """
        self.event_count += 1
        self.events.append({
            "type": event_type,
            "data": data.copy() if data else {},
            "source": source
        })
        logger.info(f"Componente {self.name} procesó evento {event_type} (total: {self.event_count})")


@pytest.mark.asyncio
async def test_minimal_component_direct():
    """
    Test directo de un componente sin usar el motor.
    
    Esta prueba verifica el funcionamiento básico de un componente
    de forma aislada, sin intervención del motor.
    """
    # 1. Crear componente minimalista
    comp = MinimalComponent("test_minimal")
    
    # 2. Iniciar componente directamente
    await comp.start()
    
    # 3. Enviar eventos directamente
    await comp.handle_event("test.event", {"id": 1}, "test")
    await comp.handle_event("another.event", {"id": 2}, "test")
    
    # 4. Verificar procesamiento
    assert comp.event_count == 2, "Debería haber procesado 2 eventos"
    assert len(comp.events) == 2, "Debería tener 2 eventos registrados"
    
    # 5. Verificar datos de eventos
    assert comp.events[0]["type"] == "test.event", "El primer evento debería ser 'test.event'"
    assert comp.events[1]["type"] == "another.event", "El segundo evento debería ser 'another.event'"
    
    # 6. Detener componente
    await comp.stop()


@pytest.mark.asyncio
async def test_multiple_components_direct():
    """
    Test directo de múltiples componentes sin usar el motor.
    
    Esta prueba verifica la interacción básica entre componentes
    sin utilizar el motor, para aislar problemas potenciales.
    """
    # 1. Crear componentes
    comp1 = MinimalComponent("comp1")
    comp2 = MinimalComponent("comp2")
    
    # 2. Iniciar componentes
    await comp1.start()
    await comp2.start()
    
    # 3. Simular emisión de eventos a través de bucle manual
    for target in [comp1, comp2]:
        await target.handle_event("system.started", {}, "system")
    
    # 4. Simular eventos de aplicación
    event_types = ["app.init", "app.update", "app.ready"]
    event_data = {"timestamp": 123456789}
    
    for event_type in event_types:
        for target in [comp1, comp2]:
            await target.handle_event(event_type, event_data, "app")
    
    # 5. Verificar procesamiento
    expected_events = 1 + len(event_types)  # system.started + eventos de app
    
    assert comp1.event_count == expected_events, f"comp1 debería haber procesado {expected_events} eventos"
    assert comp2.event_count == expected_events, f"comp2 debería haber procesado {expected_events} eventos"
    
    # 6. Verificar tipos de eventos
    comp1_types = [e["type"] for e in comp1.events]
    comp2_types = [e["type"] for e in comp2.events]
    
    for event_type in ["system.started"] + event_types:
        assert event_type in comp1_types, f"comp1 debería haber recibido evento {event_type}"
        assert event_type in comp2_types, f"comp2 debería haber recibido evento {event_type}"
    
    # 7. Detener componentes
    await comp1.stop()
    await comp2.stop()