"""
Prueba básica para el motor de timeouts configurables.

Este módulo contiene una prueba simplificada para verificar
que el motor configurable funciona correctamente con timeouts personalizados.
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List

from genesis.core.component import Component
from genesis.core.engine_configurable import ConfigurableTimeoutEngine

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlowComponent(Component):
    """Componente que se demora un tiempo configurable en iniciar y procesar eventos."""
    
    def __init__(self, name: str, start_delay: float = 0.0, event_delay: float = 0.0):
        """
        Inicializar componente lento.
        
        Args:
            name: Nombre del componente
            start_delay: Tiempo de retraso en inicio (segundos)
            event_delay: Tiempo de retraso en procesamiento de eventos (segundos)
        """
        super().__init__(name)
        self.start_delay = start_delay
        self.event_delay = event_delay
        self.events = []
        self.started = False
        self.stopped = False
        logger.info(f"Componente {name} creado con start_delay={start_delay}s, event_delay={event_delay}s")
    
    async def start(self) -> None:
        """Iniciar componente con retraso configurable."""
        logger.info(f"Componente {self.name} iniciando (espera {self.start_delay}s)")
        
        if self.start_delay > 0:
            await asyncio.sleep(self.start_delay)
        
        self.started = True
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Componente {self.name} deteniendo")
        self.stopped = True
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Procesar evento con retraso configurable.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            
        Returns:
            Ninguno
        """
        start_time = time.time()
        logger.info(f"Componente {self.name} procesando evento {event_type} (espera {self.event_delay}s)")
        
        # Registrar evento antes de retrasarse
        event_info = {
            "type": event_type,
            "data": data.copy() if data else {},
            "source": source,
            "received_at": start_time
        }
        self.events.append(event_info)
        
        # Aplicar retraso configurable
        if self.event_delay > 0:
            await asyncio.sleep(self.event_delay)
        
        end_time = time.time()
        logger.info(f"Componente {self.name} terminó de procesar evento {event_type} (duración: {end_time - start_time:.2f}s)")
        
        return None


@pytest.mark.asyncio
async def test_configurable_timeout_basic():
    """
    Prueba básica del motor con timeouts configurables.
    
    Esta prueba verifica que el motor con timeouts configurables:
    1. Permite que componentes rápidos funcionen correctamente
    2. Maneja componentes lentos según los timeouts configurados
    """
    # Crear motor con timeouts configurables
    engine = ConfigurableTimeoutEngine(
        test_mode=True,
        component_start_timeout=1.0,  # 1 segundo para inicio de componentes
        component_event_timeout=0.5   # 0.5 segundos para procesamiento de eventos
    )
    
    # Crear un componente rápido y otro lento
    fast_comp = SlowComponent("fast", start_delay=0.1, event_delay=0.1)
    slow_comp = SlowComponent("slow", start_delay=0.3, event_delay=0.8)  # El evento excede el timeout configurado
    
    # Registrar componentes
    engine.register_component(fast_comp)
    engine.register_component(slow_comp)
    
    # Medir tiempo de inicio
    start_time = time.time()
    
    # Iniciar motor
    await engine.start()
    
    # Calcular tiempo total de inicio
    init_time = time.time() - start_time
    logger.info(f"Tiempo total de inicio del motor: {init_time:.2f}s")
    
    # Verificar que el motor está iniciado
    assert engine.running, "El motor debería estar iniciado"
    
    # Verificar que los componentes se iniciaron según sus tiempos
    assert fast_comp.started, "El componente rápido debería estar iniciado"
    assert slow_comp.started, "El componente lento debería estar iniciado (bajo el timeout)"
    
    # Enviar evento de prueba
    test_event_data = {"test_id": 123, "value": "test_value"}
    event_send_time = time.time()
    logger.info(f"Enviando evento de prueba a las {event_send_time:.2f}")
    
    await engine.emit_event("test.event", test_event_data, "test")
    
    # Esperar un poco para permitir procesamiento
    await asyncio.sleep(0.6)
    
    # Obtener estadísticas de timeouts
    timeout_stats = engine.get_timeout_stats()
    logger.info(f"Estadísticas de timeouts: {timeout_stats}")
    
    # Verificar que el componente rápido recibió y procesó el evento
    assert any(e["type"] == "test.event" for e in fast_comp.events), \
        "El componente rápido debería haber recibido el evento de prueba"
    
    # El componente lento debería haber recibido el evento, pero su procesamiento
    # podría haberse interrumpido por el timeout
    assert any(e["type"] == "test.event" for e in slow_comp.events), \
        "El componente lento debería haber recibido el evento de prueba"
    
    # Verificar que hubo al menos un timeout de evento (del componente lento)
    assert timeout_stats["timeouts"]["component_event"] > 0, \
        "Debería haberse registrado al menos un timeout de evento"
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que el motor está detenido
    assert not engine.running, "El motor debería estar detenido"
    
    # Verificar que los componentes están detenidos
    assert fast_comp.stopped, "El componente rápido debería estar detenido"
    assert slow_comp.stopped, "El componente lento debería estar detenido"