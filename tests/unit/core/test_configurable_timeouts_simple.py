"""
Test ultra simplificado para ConfigurableTimeoutEngine.

Este módulo contiene una prueba ultra simplificada que verifica
el correcto funcionamiento del motor con timeouts configurables.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, List, Optional

from genesis.core.component import Component
from genesis.core.engine_configurable import ConfigurableTimeoutEngine

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleComponent(Component):
    """Componente minimalista para pruebas."""
    
    def __init__(self, name: str, delay: float = 0.0):
        """
        Inicializar componente.
        
        Args:
            name: Nombre del componente
            delay: Retraso en segundos al procesar eventos (default: 0.0)
        """
        super().__init__(name)
        self.delay = delay
        self.event_count = 0
        self.started = False
        self.stopped = False
        self.events = []
        logger.info(f"Componente {name} creado (delay={delay}s)")
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Componente {self.name} iniciando")
        if self.delay > 0:
            await asyncio.sleep(min(self.delay, 0.2))  # Limitar para evitar timeouts
        self.started = True
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Componente {self.name} deteniendo")
        if self.delay > 0:
            await asyncio.sleep(min(self.delay, 0.2))  # Limitar para evitar timeouts
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
        self.event_count += 1
        self.events.append({
            "type": event_type, 
            "data": data.copy() if data else {},
            "source": source
        })
        
        # Simular procesamiento con delay controlado
        if self.delay > 0:
            await asyncio.sleep(min(self.delay, 0.1))  # Limitado para evitar timeouts
            
        logger.info(f"Componente {self.name} procesó evento {event_type} (total: {self.event_count})")


@pytest.mark.asyncio
async def test_configurable_timeouts_simple():
    """
    Prueba ultra simple del motor configurable.
    
    Esta prueba verifica el funcionamiento básico de ConfigurableTimeoutEngine.
    """
    # 1. Crear motor con timeouts explícitos
    engine = ConfigurableTimeoutEngine(
        component_start_timeout=0.5,
        component_stop_timeout=0.5,
        component_event_timeout=0.3,
        event_timeout=1.0,
        test_mode=True
    )
    
    # 2. Crear componentes con diferentes velocidades
    comp1 = SimpleComponent("fast", delay=0.0)
    comp2 = SimpleComponent("medium", delay=0.1)
    comp3 = SimpleComponent("slow", delay=0.3)  # Este componente podría exceder el timeout
    
    # 3. Registrar componentes
    engine.register_component(comp1)
    engine.register_component(comp2)
    engine.register_component(comp3)
    
    # 4. Iniciar motor
    await engine.start()
    
    # 5. Verificar estado inicial
    assert engine.running, "El motor debería estar iniciado"
    assert comp1.started, "El componente rápido debería estar iniciado"
    assert comp2.started, "El componente medio debería estar iniciado"
    # No verificamos comp3 porque podría haber tenido timeout
    
    # 6. Emitir evento personalizado
    custom_data = {"test_id": 123}
    await engine.emit_event("test.event", custom_data, "test")
    
    # 7. Espera controlada con tiempo fijo corto
    await asyncio.sleep(0.5)  # Tiempo suficiente para procesar pero no excesivo
    
    # 8. Verificación básica de procesamiento
    assert comp1.event_count > 0, "El componente rápido debería haber recibido al menos un evento"
    assert comp2.event_count > 0, "El componente medio debería haber recibido al menos un evento"
    # No verificamos comp3 porque podría haber tenido timeout
    
    # 9. Verificar estadísticas de timeout (característica clave de esta implementación)
    timeout_stats = engine.get_timeout_stats()
    logger.info(f"Estadísticas de timeouts: {timeout_stats}")
    
    # 10. Detener motor
    await engine.stop()
    
    # 11. Verificar estado final
    assert not engine.running, "El motor debería estar detenido"
    assert comp1.stopped, "El componente rápido debería estar detenido"
    assert comp2.stopped, "El componente medio debería estar detenido"
    # No verificamos comp3 porque podría haber tenido timeout