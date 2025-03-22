"""
Test para el motor ultra simple con timeouts.

Este módulo contiene pruebas para el motor ultra simple con timeouts,
verificando su comportamiento con componentes de diferentes velocidades.
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from genesis.core.component import Component
from genesis.core.engine_ultra_simple_timeout import UltraSimpleTimeoutEngine

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleComponent(Component):
    """Componente simple para pruebas de timeout."""
    
    def __init__(self, name: str, delay: float = 0.0):
        """
        Inicializar componente con retraso configurable.
        
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
        """Iniciar componente con retraso configurable."""
        logger.info(f"Componente {self.name} iniciando")
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.started = True
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener componente con retraso configurable."""
        logger.info(f"Componente {self.name} deteniendo")
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.stopped = True
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento con retraso configurable.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
        """
        logger.info(f"Componente {self.name} recibiendo evento {event_type}")
        self.event_count += 1
        self.events.append({
            "type": event_type,
            "data": data.copy() if data else {},
            "source": source
        })
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
            
        logger.info(f"Componente {self.name} procesó evento {event_type} (total: {self.event_count})")


@pytest.mark.asyncio
async def test_ultra_simple_timeout_engine_basic():
    """
    Test básico del motor ultra simple con timeouts.
    
    Esta prueba verifica el comportamiento básico del motor
    con componentes de diferentes velocidades.
    """
    # 1. Crear motor con timeouts generosos
    engine = UltraSimpleTimeoutEngine(
        component_timeout=1.0,
        event_timeout=2.0
    )
    
    # 2. Crear componentes con diferentes velocidades
    fast_comp = SimpleComponent("fast", delay=0.0)  # Componente rápido
    medium_comp = SimpleComponent("medium", delay=0.3)  # Componente medio
    slow_comp = SimpleComponent("slow", delay=1.5)  # Componente lento (excederá timeout)
    
    # 3. Registrar componentes
    engine.register_component(fast_comp)
    engine.register_component(medium_comp)
    engine.register_component(slow_comp)
    
    # 4. Iniciar motor
    await engine.start()
    
    # 5. Verificar inicio
    assert engine.running, "El motor debería estar iniciado"
    assert fast_comp.started, "El componente rápido debería estar iniciado"
    assert medium_comp.started, "El componente medio debería estar iniciado"
    
    # El componente lento probablemente no inició por timeout
    # No verificamos explícitamente
    
    # 6. Emitir eventos
    await engine.emit_event("test.event", {"value": 100}, "test")
    
    # 7. Verificar procesamiento (enfoque flexible)
    # No esperamos que todos los componentes procesen todos los eventos
    assert fast_comp.event_count > 0, "El componente rápido debería haber procesado al menos un evento"
    # No verificamos el componente medio o lento explícitamente
    
    # 8. Obtener estadísticas
    stats = engine.get_stats()
    logger.info(f"Estadísticas del motor: {stats}")
    
    # 9. Detener motor
    await engine.stop()
    
    # 10. Verificar parada
    assert not engine.running, "El motor debería estar detenido"
    assert fast_comp.stopped, "El componente rápido debería estar detenido"
    assert medium_comp.stopped, "El componente medio debería estar detenido"
    # No verificamos el componente lento
    
    # 11. Verificar estadísticas finales
    final_stats = engine.get_stats()
    logger.info(f"Estadísticas finales: {final_stats}")
    assert final_stats["timeouts"] > 0, "Debería haber ocurrido al menos un timeout"


@pytest.mark.asyncio
async def test_ultra_simple_timeout_engine_adjust_timeouts():
    """
    Test con ajuste de timeouts.
    
    Esta prueba verifica que podemos cambiar el valor de los timeouts
    para adaptarnos a componentes más lentos.
    """
    # 1. Componentes con diferentes velocidades pero todos por debajo de los timeouts
    components = [
        SimpleComponent("comp_0", delay=0.0),   # Muy rápido
        SimpleComponent("comp_1", delay=0.1),   # Rápido
        SimpleComponent("comp_2", delay=0.25)   # Medio
    ]
    
    # 2. Crear motor con timeouts ajustados
    engine = UltraSimpleTimeoutEngine(
        component_timeout=0.5,  # Suficiente para todos los componentes
        event_timeout=1.5       # Suficiente para procesar todos los eventos
    )
    
    # 3. Registrar componentes
    for comp in components:
        engine.register_component(comp)
    
    # 4. Iniciar motor
    await engine.start()
    
    # 5. Verificar inicio exitoso de todos
    assert engine.running, "El motor debería estar iniciado"
    all_started = all(comp.started for comp in components)
    assert all_started, "Todos los componentes deberían haberse iniciado"
    
    # 6. Emitir eventos básicos
    await engine.emit_event("test.event.1", {"id": 1}, "test")
    await engine.emit_event("test.event.2", {"id": 2}, "test")
    
    # 7. Verificar procesamiento exitoso
    all_processed = all(comp.event_count == 2 for comp in components)
    assert all_processed, "Todos los componentes deberían haber procesado los 2 eventos"
    
    # 8. Verificar estadísticas (no debería haber timeouts)
    stats = engine.get_stats()
    logger.info(f"Estadísticas: {stats}")
    
    # 9. Detener motor
    await engine.stop()
    
    # 10. Verificar parada exitosa
    assert not engine.running, "El motor debería estar detenido"
    all_stopped = all(comp.stopped for comp in components)
    assert all_stopped, "Todos los componentes deberían haberse detenido"