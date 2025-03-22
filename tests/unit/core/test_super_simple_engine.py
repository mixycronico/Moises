"""
Test para el motor ultra simplificado.

Este módulo contiene pruebas para el motor ultra simplificado
que ayudan a identificar y solucionar problemas de timeout.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, List, Optional

from genesis.core.component import Component
from genesis.core.engine_super_simple import SuperSimpleEngine

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestComponent(Component):
    """Componente de prueba para testing."""
    
    def __init__(self, name: str, delay: float = 0.0):
        """
        Inicializar componente con nombre.
        
        Args:
            name: Nombre del componente
            delay: Retraso en segundos para simular procesamiento
        """
        super().__init__(name)
        self.delay = delay
        self.started = False
        self.stopped = False
        self.events = []
        self.event_count = 0
        logger.info(f"Componente {name} creado (delay={delay}s)")
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Componente {self.name} iniciando")
        # Simular trabajo con delay controlado
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.started = True
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Componente {self.name} deteniendo")
        # Simular trabajo con delay controlado
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.stopped = True
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
        """
        logger.info(f"Componente {self.name} recibiendo evento {event_type}")
        # Registrar evento
        self.events.append({
            "type": event_type,
            "data": data.copy() if data else {},
            "source": source
        })
        self.event_count += 1
        
        # Simular trabajo con delay controlado
        if self.delay > 0:
            await asyncio.sleep(self.delay)
            
        logger.info(f"Componente {self.name} procesó evento {event_type} (total: {self.event_count})")


@pytest.mark.asyncio
async def test_super_simple_engine_basic():
    """
    Test básico del motor ultra simplificado.
    
    Esta prueba verifica el funcionamiento esencial del motor,
    sin complejidades adicionales.
    """
    # 1. Crear motor
    engine = SuperSimpleEngine()
    
    # 2. Crear componentes simples
    fast_comp = TestComponent("fast", delay=0.01)
    slow_comp = TestComponent("slow", delay=0.2)
    
    # 3. Registrar componentes
    engine.register_component(fast_comp)
    engine.register_component(slow_comp)
    
    # 4. Iniciar motor
    await engine.start()
    
    # 5. Verificar inicio
    assert engine.running, "El motor debería estar ejecutándose"
    assert fast_comp.started, "El componente rápido debería estar iniciado"
    assert slow_comp.started, "El componente lento debería estar iniciado"
    
    # 6. Emitir un evento simple
    await engine.emit_event("test.event", {"id": 1}, "test")
    
    # 7. Breve espera para asegurar procesamiento
    await asyncio.sleep(0.5)
    
    # 8. Verificar procesamiento
    assert fast_comp.event_count > 0, "El componente rápido debería haber recibido al menos un evento"
    assert slow_comp.event_count > 0, "El componente lento debería haber recibido al menos un evento"
    
    # 9. Emitir otro evento para verificar consistencia
    await engine.emit_event("test.another", {"id": 2}, "test")
    
    # 10. Espera corta
    await asyncio.sleep(0.5)
    
    # 11. Verificar procesamiento adicional
    assert fast_comp.event_count > 1, "El componente rápido debería haber recibido el segundo evento"
    assert slow_comp.event_count > 1, "El componente lento debería haber recibido el segundo evento"
    
    # 12. Detener el motor
    await engine.stop()
    
    # 13. Verificar parada
    assert not engine.running, "El motor debería estar detenido"
    assert fast_comp.stopped, "El componente rápido debería estar detenido"
    assert slow_comp.stopped, "El componente lento debería estar detenido"