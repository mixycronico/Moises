"""
Test ultra minimalista para motor de eventos.

Este módulo contiene pruebas ultra simplificadas
que se centran en la funcionalidad esencial del motor.
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

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
        self.started = False
        self.stopped = False
    
    async def start(self) -> None:
        """Iniciar componente."""
        self.started = True
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener componente."""
        self.stopped = True
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


class UltraMinimalEngine:
    """
    Motor ultra minimalista para pruebas elementales.
    
    Esta implementación contiene sólo lo esencial para hacer
    pruebas básicas sin ninguna complejidad adicional.
    """
    
    def __init__(self):
        """Inicializar motor ultra minimalista."""
        self._components = {}  # name -> component
        self.running = False
        logger.info("Motor ultra minimalista creado")
    
    def register_component(self, component: Component) -> None:
        """
        Registrar componente en el motor.
        
        Args:
            component: Componente a registrar
        """
        self._components[component.name] = component
        logger.info(f"Componente {component.name} registrado")
    
    async def start(self) -> None:
        """Iniciar el motor y todos los componentes registrados."""
        logger.info("Iniciando motor ultra minimalista")
        
        # Iniciar componentes secuencialmente
        for name, component in self._components.items():
            await component.start()
            logger.info(f"Componente {name} iniciado")
        
        self.running = True
        logger.info("Motor ultra minimalista iniciado")
    
    async def stop(self) -> None:
        """Detener el motor y todos los componentes registrados."""
        logger.info("Deteniendo motor ultra minimalista")
        
        # Detener componentes secuencialmente
        for name, component in self._components.items():
            await component.stop()
            logger.info(f"Componente {name} detenido")
        
        self.running = False
        logger.info("Motor ultra minimalista detenido")
    
    async def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, source: str = "system") -> None:
        """
        Emitir evento a todos los componentes registrados.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento (opcional)
            source: Origen del evento (por defecto: "system")
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} con motor detenido")
            return
        
        logger.info(f"Emitiendo evento {event_type} desde {source}")
        event_data = data or {}
        
        # Enviar evento a cada componente secuencialmente
        for name, component in self._components.items():
            await component.handle_event(event_type, event_data, source)
            logger.info(f"Componente {name} procesó evento {event_type}")


@pytest.mark.asyncio
async def test_ultra_minimal_engine():
    """
    Test básico del motor ultra minimalista.
    
    Esta prueba verifica el funcionamiento esencial reducido
    a su mínima expresión.
    """
    # 1. Crear motor ultra minimalista
    engine = UltraMinimalEngine()
    
    # 2. Crear componentes simples
    comp1 = MinimalComponent("comp1")
    comp2 = MinimalComponent("comp2")
    
    # 3. Registrar componentes
    engine.register_component(comp1)
    engine.register_component(comp2)
    
    # 4. Iniciar motor
    await engine.start()
    
    # 5. Verificar inicio
    assert engine.running, "El motor debería estar iniciado"
    assert comp1.started, "El componente 1 debería estar iniciado"
    assert comp2.started, "El componente 2 debería estar iniciado"
    
    # 6. Emitir evento simple
    await engine.emit_event("test.event", {"id": 1}, "test")
    
    # 7. Verificar procesamiento inmediato (sin esperas)
    assert comp1.event_count == 1, "El componente 1 debería haber procesado el evento"
    assert comp2.event_count == 1, "El componente 2 debería haber procesado el evento"
    
    # 8. Verificar contenido del evento
    assert comp1.events[0]["type"] == "test.event", "El tipo de evento debería ser 'test.event'"
    assert comp1.events[0]["data"]["id"] == 1, "El ID del evento debería ser 1"
    assert comp1.events[0]["source"] == "test", "La fuente del evento debería ser 'test'"
    
    # 9. Detener motor
    await engine.stop()
    
    # 10. Verificar parada
    assert not engine.running, "El motor debería estar detenido"
    assert comp1.stopped, "El componente 1 debería estar detenido"
    assert comp2.stopped, "El componente 2 debería estar detenido"


@pytest.mark.asyncio
async def test_ultra_minimal_engine_multiple_events():
    """
    Test con múltiples eventos en el motor ultra minimalista.
    
    Esta prueba verifica el manejo básico de múltiples eventos.
    """
    # 1. Crear motor ultra minimalista
    engine = UltraMinimalEngine()
    
    # 2. Crear componentes
    components = [
        MinimalComponent(f"comp_{i}") for i in range(3)
    ]
    
    # 3. Registrar componentes
    for comp in components:
        engine.register_component(comp)
    
    # 4. Iniciar motor
    await engine.start()
    
    # 5. Verificar inicio
    assert engine.running, "El motor debería estar iniciado"
    for comp in components:
        assert comp.started, f"El componente {comp.name} debería estar iniciado"
    
    # 6. Emitir múltiples eventos
    event_types = ["app.init", "app.update", "app.ready"]
    for i, event_type in enumerate(event_types):
        await engine.emit_event(event_type, {"index": i}, "app")
    
    # 7. Verificar procesamiento
    for comp in components:
        assert comp.event_count == len(event_types), f"{comp.name} debería haber procesado {len(event_types)} eventos"
        
        # Verificar tipos de eventos
        received_types = [e["type"] for e in comp.events]
        for event_type in event_types:
            assert event_type in received_types, f"{comp.name} debería haber recibido evento {event_type}"
    
    # 8. Detener motor
    await engine.stop()
    
    # 9. Verificar parada
    assert not engine.running, "El motor debería estar detenido"
    for comp in components:
        assert comp.stopped, f"El componente {comp.name} debería estar detenido"