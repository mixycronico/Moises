"""
Test simple para verificar el funcionamiento del método register_component.
"""

import asyncio
import logging
from genesis.core.engine_non_blocking import EngineNonBlocking
from genesis.core.component import Component

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleComponent(Component):
    """Componente simple para pruebas."""
    
    def __init__(self, name):
        super().__init__(name)
    
    async def start(self):
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self):
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type, data, source):
        logger.info(f"Componente {self.name} recibió evento {event_type}")
        return {"processed": True, "component": self.name}

async def test_register_component():
    """Test simple para register_component"""
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componente
    component = SimpleComponent("test_component")
    
    # Registrar componente
    await engine.register_component(component)
    
    # Verificar que se registró correctamente
    assert "test_component" in engine.components
    
    # Cleanup
    await engine.stop()
    
if __name__ == "__main__":
    asyncio.run(test_register_component())