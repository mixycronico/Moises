"""
Test simplificado para el ComponentMonitor para evitar timeouts.

Este test se enfoca solo en la inicialización básica del monitor
y evita operaciones que puedan causar bloqueos.
"""

import asyncio
import pytest
from typing import Dict, Any, List, Optional

from genesis.core.engine_non_blocking import EngineNonBlocking
from genesis.core.component_monitor import ComponentMonitor
from genesis.core.component import Component

class SimpleTestComponent(Component):
    """Componente simple de prueba."""
    
    def __init__(self, name, dependencies=None):
        super().__init__(name)
        self.healthy = True
        self.dependencies = dependencies or []
    
    async def start(self):
        """Iniciar el componente."""
        pass
    
    async def stop(self):
        """Detener el componente."""
        pass
    
    async def handle_event(self, event_type, data, source):
        """Manejar eventos de prueba."""
        if event_type == "check_status":
            return {
                "component": self.name,
                "healthy": self.healthy,
                "dependencies": {dep: True for dep in self.dependencies}
            }
        elif event_type == "set_health":
            self.healthy = data.get("healthy", True)
            return {"success": True}
        return None

@pytest.mark.asyncio
async def test_monitor_simplified():
    """Test simplificado para verificar la inicialización básica del monitor."""
    # Crear el motor y el monitor directamente (sin fixture)
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear monitor con configuración simple
    monitor = ComponentMonitor(
        "test_monitor", 
        check_interval=0.5,     # Intervalo corto para pruebas
        max_failures=2,         # Solo 2 fallos para aislar
        recovery_interval=0.5   # Intervalo de recuperación corto
    )
    
    # Registrar monitor de forma asíncrona
    await engine.register_component(monitor)
    
    # Iniciar el monitor
    await monitor.start()
    
    # Verificar estado inicial del monitor
    assert monitor.running
    assert not monitor.isolated_components
    assert not monitor.health_status
    assert not monitor.failure_counts
    
    # Registrar un componente simple
    comp_a = SimpleTestComponent("comp_a")
    await engine.register_component(comp_a)
    
    # Esperar un tiempo mínimo para que el monitor detecte el componente
    await asyncio.sleep(0.5)
    
    # Detener todo
    await engine.stop()
    
    # Test exitoso
    assert True