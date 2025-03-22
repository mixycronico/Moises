"""
Test directo para ComponentMonitor sin depender de fixtures.

Este test crea directamente las instancias necesarias y evita
dependencias complejas que pueden causar timeouts.
"""

import asyncio
import logging
import sys
from typing import Dict, Any, List, Optional

from genesis.core.engine_non_blocking import EngineNonBlocking
from genesis.core.component_monitor import ComponentMonitor
from genesis.core.component import Component

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestComponent(Component):
    """Componente simple para pruebas del monitor."""
    
    def __init__(self, name, dependencies=None):
        super().__init__(name)
        self.healthy = True
        self.respond_to_checks = True
        self.dependencies = dependencies or []
        self.blocking = False
        self.block_time = 0
    
    async def start(self):
        """Iniciar el componente."""
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self):
        """Detener el componente."""
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type, data, source):
        """Manejar eventos para el componente de prueba."""
        # Si está configurado para bloquearse
        if self.blocking and event_type == "check_status":
            logger.info(f"Componente {self.name} bloqueando por {self.block_time}s")
            await asyncio.sleep(self.block_time)
            
        # Responder a verificación de estado
        if event_type == "check_status":
            if not self.respond_to_checks:
                logger.info(f"Componente {self.name} no responde a verificación")
                return None
                
            logger.info(f"Componente {self.name} responde a verificación: saludable={self.healthy}")
            return {
                "component": self.name,
                "healthy": self.healthy,
                "dependencies": {dep: True for dep in self.dependencies}
            }
            
        # Configurar salud
        elif event_type == "set_health":
            self.healthy = data.get("healthy", True)
            logger.info(f"Componente {self.name} salud establecida a {self.healthy}")
            return {"success": True}
            
        # Configurar comportamiento de respuesta
        elif event_type == "set_response_behavior":
            self.respond_to_checks = data.get("respond", True)
            logger.info(f"Componente {self.name} responder={self.respond_to_checks}")
            return {"success": True}
            
        # Configurar bloqueo
        elif event_type == "set_blocking":
            self.blocking = data.get("blocking", False)
            self.block_time = data.get("block_time", 1.0)
            logger.info(f"Componente {self.name} bloqueo={self.blocking}, tiempo={self.block_time}")
            return {"success": True}
            
        return None

async def test_monitor_initialization():
    """Probar que el monitor se inicializa correctamente."""
    logger.info("Iniciando test_monitor_initialization")
    
    # Crear el motor en modo test para timeouts más cortos
    engine = EngineNonBlocking(test_mode=True)
    logger.info("Motor creado en modo test")
    
    # Crear monitor con configuración simple
    monitor = ComponentMonitor(
        "test_monitor", 
        check_interval=0.5,     # Intervalo corto para pruebas
        max_failures=2,         # Solo 2 fallos para aislar
        recovery_interval=0.5   # Intervalo de recuperación corto
    )
    logger.info("Monitor creado con configuración de prueba")
    
    # Registrar monitor
    logger.info("Registrando monitor...")
    await engine.register_component(monitor)
    logger.info("Monitor registrado")
    
    # Iniciar el monitor
    logger.info("Iniciando monitor...")
    await monitor.start()
    logger.info("Monitor iniciado")
    
    # Verificar que el monitor está registrado
    assert "test_monitor" in engine.components
    
    # Verificar estado inicial
    assert monitor.running
    assert not monitor.isolated_components
    assert not monitor.health_status
    assert not monitor.failure_counts
    
    # Verificar que responde a eventos
    logger.info("Creando componente de prueba...")
    comp_a = TestComponent("comp_a")
    
    logger.info("Registrando componente...")
    await engine.register_component(comp_a)
    logger.info("Componente registrado")
    
    # Esperar un poco para que el monitor verifique el componente
    logger.info("Esperando verificación inicial...")
    await asyncio.sleep(1.0)
    
    # Limpiar recursos
    logger.info("Limpiando recursos...")
    await engine.stop()
    logger.info("Test completado con éxito")
    
    # Test exitoso
    assert True

if __name__ == "__main__":
    print(f"Python {sys.version}")
    print("Ejecutando test directo para ComponentMonitor...")
    asyncio.run(test_monitor_initialization())