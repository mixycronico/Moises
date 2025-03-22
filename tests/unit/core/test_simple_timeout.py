"""
Prueba simple para verificar el comportamiento de timeout con EngineNonBlocking.
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, Optional

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDelayComponent(Component):
    """Componente que simplemente se retrasa un tiempo específico."""
    
    def __init__(self, name: str, delay: float = 0.0):
        """
        Inicializar componente con retraso.
        
        Args:
            name: Nombre del componente
            delay: Tiempo de retraso (segundos)
        """
        super().__init__(name)
        self.delay = delay
        self.started = False
        self.events = []
    
    async def start(self) -> None:
        """Iniciar componente con retraso."""
        logger.info(f"Componente {self.name} iniciando (retrasando {self.delay}s)")
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        self.started = True
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """Manejar evento."""
        self.events.append({"type": event_type, "data": data, "source": source})
        logger.info(f"Componente {self.name} recibió evento {event_type}")
        
        # Aplicar retraso también a eventos
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        return None


@pytest.mark.asyncio
async def test_simple_timeout():
    """
    Prueba simple para verificar el funcionamiento del timeout en modo prueba.
    
    Esta prueba crea un componente con retraso mayor que el timeout del motor
    y verifica que el sistema no se bloquee.
    """
    # Crear motor en modo prueba (timeout = 0.5s por defecto)
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componente con retraso mayor que el timeout
    comp = SimpleDelayComponent("delayed", delay=2.0)  # 2 segundos (mayor que el timeout)
    
    # Registrar componente
    engine.register_component(comp)
    
    # Medir tiempo de inicio
    start_time = time.time()
    
    # Iniciar motor
    await engine.start()
    
    # Medir tiempo transcurrido
    elapsed = time.time() - start_time
    
    # El inicio debería haber tomado poco tiempo debido al timeout (no debería esperar 2s)
    assert elapsed < 1.0, f"El inicio tomó demasiado tiempo: {elapsed:.2f}s"
    
    # El motor debería estar marcado como iniciado
    assert engine.running, "El motor debería estar marcado como iniciado"
    
    # Enviar un evento simple
    await engine.emit_event("test_event", {"value": "test"}, "test")
    
    # Esperar un poco para dar tiempo al evento
    await asyncio.sleep(0.1)
    
    # Detener motor
    await engine.stop()
    
    # Verificar que el motor está detenido
    assert not engine.running, "El motor debería estar detenido"
    
    logger.info("Prueba simple de timeout completada correctamente")