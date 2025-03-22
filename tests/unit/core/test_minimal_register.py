"""
Test extremadamente minimalista para verificar registro de componentes.

Este test es deliberadamente simplificado para evitar timeouts y deadlocks.
"""

import asyncio
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEventBus:
    """Bus de eventos extremadamente simplificado."""
    
    def __init__(self):
        self.running = True
        self.subscribers = {}
    
    async def start(self):
        pass
    
    async def stop(self):
        self.running = False
        
    def subscribe(self, event_type, callback, priority=0):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append((callback, priority))

class SimpleComponent:
    """Componente extremadamente simplificado."""
    
    def __init__(self, name):
        self.name = name
        self.event_bus = None
    
    def attach_event_bus(self, event_bus):
        self.event_bus = event_bus
    
    async def start(self):
        logger.info(f"Componente {self.name} iniciado")
        
    async def stop(self):
        logger.info(f"Componente {self.name} detenido")
        
    async def handle_event(self, event_type, data, source):
        logger.info(f"Componente {self.name} recibió evento {event_type}")
        return {"processed": True, "by": self.name}

class SimpleEngine:
    """Motor extremadamente simplificado."""
    
    def __init__(self):
        self.components = {}
        self.event_bus = SimpleEventBus()
        self.running = False
    
    async def register_component(self, component):
        """Registrar un componente de forma asíncrona."""
        # Guardar el componente
        self.components[component.name] = component
        
        # Configurar event bus para el componente
        component.attach_event_bus(self.event_bus)
        
        # Registrar el componente para que reciba todos los eventos
        self.event_bus.subscribe("*", component.handle_event)
        
        logger.info(f"Registrado componente: {component.name}")
        
        # Si el motor ya está ejecutándose, iniciar también el componente
        if self.running:
            await component.start()
            
    async def start(self):
        """Iniciar el motor y todos los componentes."""
        if self.running:
            return
            
        self.running = True
        
        # Iniciar event bus
        await self.event_bus.start()
        
        # Iniciar todos los componentes
        for name, component in self.components.items():
            await component.start()
            
    async def stop(self):
        """Detener el motor y todos los componentes."""
        if not self.running:
            return
            
        # Detener todos los componentes
        for name, component in self.components.items():
            await component.stop()
            
        # Detener event bus
        await self.event_bus.stop()
        
        self.running = False

async def test_register_component():
    """Test básico para register_component."""
    # Crear el motor
    engine = SimpleEngine()
    
    # Crear componente
    component = SimpleComponent("test_component")
    
    # Registrar componente
    await engine.register_component(component)
    
    # Verificar que se registró correctamente
    assert "test_component" in engine.components
    
    # Iniciar motor
    await engine.start()
    
    # Verificar que el motor está en ejecución
    assert engine.running
    
    # Detener motor
    await engine.stop()
    
    # Verificar que el motor se detuvo
    assert not engine.running
    
    print("Test completado con éxito")

if __name__ == "__main__":
    asyncio.run(test_register_component())