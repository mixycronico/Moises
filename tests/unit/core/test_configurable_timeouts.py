"""
Pruebas específicas para el motor con timeouts configurables.

Este módulo se enfoca exclusivamente en probar las características
de timeout configurable del motor mejorado de Genesis.
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, Optional

from genesis.core.component import Component
from genesis.core.engine_configurable import ConfigurableTimeoutEngine

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DelayedComponent(Component):
    """Componente simple que se retrasa un tiempo específico."""
    
    def __init__(self, name: str, start_delay: float = 0.0, stop_delay: float = 0.0, event_delay: float = 0.0):
        """
        Inicializar componente con retrasos configurables.
        
        Args:
            name: Nombre del componente
            start_delay: Tiempo de retraso al iniciar (segundos)
            stop_delay: Tiempo de retraso al detener (segundos)
            event_delay: Tiempo de retraso al procesar eventos (segundos)
        """
        super().__init__(name)
        self.start_delay = start_delay
        self.stop_delay = stop_delay
        self.event_delay = event_delay
        self.started = False
        self.stopped = False
        self.events_processed = 0
        self.start_called = 0
        self.stop_called = 0
    
    async def start(self) -> None:
        """Iniciar componente con retraso configurado."""
        self.start_called += 1
        logger.info(f"Componente {self.name} iniciando (retrasando {self.start_delay}s)")
        
        if self.start_delay > 0:
            await asyncio.sleep(self.start_delay)
        
        self.started = True
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener componente con retraso configurado."""
        self.stop_called += 1
        logger.info(f"Componente {self.name} deteniendo (retrasando {self.stop_delay}s)")
        
        if self.stop_delay > 0:
            await asyncio.sleep(self.stop_delay)
        
        self.stopped = True
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Procesar evento con retraso configurado.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            
        Returns:
            Resultado opcional
        """
        self.events_processed += 1
        logger.info(f"Componente {self.name} procesando evento {event_type} (retrasando {self.event_delay}s)")
        
        if self.event_delay > 0:
            await asyncio.sleep(self.event_delay)
        
        logger.info(f"Componente {self.name} terminó de procesar evento {event_type}")
        return {"processed_by": self.name, "count": self.events_processed}


@pytest.mark.asyncio
async def test_component_with_timeout_smaller_than_delay():
    """
    Prueba un componente cuyo tiempo de inicio excede el timeout.
    
    Demuestra que el motor maneja correctamente timeouts configurables.
    """
    # Crear motor con timeout muy pequeño
    engine = ConfigurableTimeoutEngine(
        test_mode=True,
        component_start_timeout=0.1  # 100ms timeout (muy pequeño)
    )
    
    # Crear componente que tarda más que el timeout
    slow_component = DelayedComponent("slow_start", start_delay=0.3)  # 300ms (mayor que el timeout)
    
    # Registrar componente
    engine.register_component(slow_component)
    
    # Iniciar motor (debería continuar a pesar del timeout)
    start_time = time.time()
    await engine.start()
    elapsed = time.time() - start_time
    
    # Verificar que el inicio del motor no tomó demasiado tiempo (el timeout funcionó)
    assert elapsed < 0.5, f"El inicio del motor tomó demasiado tiempo: {elapsed:.2f}s"
    
    # Verificar que el motor está marcado como iniciado
    assert engine.running, "El motor debería estar marcado como iniciado"
    
    # Verificar que el componente no completó su inicio (debido al timeout)
    assert not slow_component.started, "El componente no debería haberse iniciado completamente debido al timeout"
    
    # Verificar que se intentó iniciar el componente
    assert slow_component.start_called > 0, "El método start() del componente debería haberse llamado"
    
    # Verificar estadísticas de timeout
    stats = engine.get_timeout_stats()
    logger.info(f"Estadísticas de timeout: {stats}")
    assert stats["timeouts"]["component_start"] > 0, "Debería haberse registrado al menos un timeout de inicio"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que el motor está marcado como detenido
    assert not engine.running, "El motor debería estar marcado como detenido"


@pytest.mark.asyncio
async def test_component_with_timeout_larger_than_delay():
    """
    Prueba un componente cuyo tiempo de inicio es menor que el timeout.
    
    Demuestra que los timeouts configurables permiten operaciones
    que de otra manera fallarían.
    """
    # Crear motor con timeout suficientemente grande
    engine = ConfigurableTimeoutEngine(
        test_mode=True,
        component_start_timeout=0.5  # 500ms timeout (suficiente)
    )
    
    # Crear componente que tarda menos que el timeout
    slow_component = DelayedComponent("reasonable_start", start_delay=0.2)  # 200ms (menor que el timeout)
    
    # Registrar componente
    engine.register_component(slow_component)
    
    # Iniciar motor (debería completar sin timeout)
    start_time = time.time()
    await engine.start()
    elapsed = time.time() - start_time
    
    # Verificar que el inicio tomó el tiempo esperado
    assert 0.2 <= elapsed <= 0.6, f"El inicio tomó un tiempo inesperado: {elapsed:.2f}s"
    
    # Verificar que el motor está iniciado
    assert engine.running, "El motor debería estar iniciado"
    
    # Verificar que el componente completó su inicio
    assert slow_component.started, "El componente debería haberse iniciado completamente"
    
    # Verificar que se llamó al método start() exactamente una vez
    assert slow_component.start_called == 1, "El método start() debería haberse llamado exactamente una vez"
    
    # Verificar estadísticas de timeout
    stats = engine.get_timeout_stats()
    logger.info(f"Estadísticas de timeout: {stats}")
    assert stats["timeouts"]["component_start"] == 0, "No debería haberse registrado ningún timeout de inicio"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que el motor está marcado como detenido
    assert not engine.running, "El motor debería estar marcado como detenido"


@pytest.mark.asyncio
async def test_event_timeout_configurable():
    """
    Prueba que los timeouts de eventos son configurables.
    
    Demuestra que el motor puede manejar componentes lentos con
    timeouts adecuados.
    """
    # Crear motor con timeout de evento configurable
    engine = ConfigurableTimeoutEngine(
        test_mode=True,
        component_event_timeout=0.3,  # 300ms timeout para eventos
        event_timeout=0.5  # 500ms timeout para emisión de eventos
    )
    
    # Crear componentes con diferentes tiempos de procesamiento
    fast_component = DelayedComponent("fast", event_delay=0.01)  # 10ms (muy rápido)
    medium_component = DelayedComponent("medium", event_delay=0.1)  # 100ms (medio)
    slow_component = DelayedComponent("slow", event_delay=0.4)  # 400ms (excede el timeout de componente)
    
    # Registrar componentes
    engine.register_component(fast_component)
    engine.register_component(medium_component)
    engine.register_component(slow_component)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar algunos eventos
    num_events = 3
    for i in range(num_events):
        await engine.emit_event(f"test_event_{i}", {"id": i}, "test")
        # Esperar un poco entre eventos
        await asyncio.sleep(0.1)
    
    # Esperar un poco más para permitir procesamiento
    await asyncio.sleep(0.5)
    
    # Verificar que los componentes procesaron eventos según su velocidad
    assert fast_component.events_processed == num_events, \
        "El componente rápido debería haber procesado todos los eventos"
    
    assert medium_component.events_processed == num_events, \
        "El componente medio debería haber procesado todos los eventos"
    
    # El componente lento puede no haber procesado todos los eventos debido al timeout
    # pero debería haber procesado al menos algunos
    assert slow_component.events_processed > 0, \
        "El componente lento debería haber procesado al menos algunos eventos"
    
    # Verificar estadísticas de timeout
    stats = engine.get_timeout_stats()
    logger.info(f"Estadísticas de timeout: {stats}")
    
    # Detener motor
    await engine.stop()
    
    # Verificar que el motor está detenido
    assert not engine.running, "El motor debería estar detenido"