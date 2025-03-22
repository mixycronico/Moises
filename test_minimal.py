"""
Prueba mínima del sistema híbrido Genesis API + WebSocket.

Este script hace una prueba mínima del sistema híbrido.
"""

import asyncio
import logging
import sys
from typing import Dict, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_minimal")

# Clase ComponentAPI simulada
class ComponentAPI:
    def __init__(self, id: str):
        self.id = id
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        logger.info(f"Componente {self.id} procesando solicitud {request_type} de {source}")
        if request_type == "echo":
            return {"echo": data.get("message", ""), "source": source}
        return None
    
    async def start(self) -> None:
        logger.info(f"Componente {self.id} iniciado")
    
    async def stop(self) -> None:
        logger.info(f"Componente {self.id} detenido")

# Coordinator simulado
class Coordinator:
    def __init__(self):
        self.components = {}
        
    def register_component(self, id: str, component: ComponentAPI) -> None:
        self.components[id] = component
        logger.info(f"Componente {id} registrado")
    
    async def request(self, target_id: str, request_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        if target_id not in self.components:
            logger.error(f"Componente {target_id} no encontrado")
            return None
        
        try:
            result = await self.components[target_id].process_request(request_type, data, source)
            return result
        except Exception as e:
            logger.error(f"Error en solicitud: {e}")
            return None
    
    async def start(self) -> None:
        for comp_id, comp in self.components.items():
            await comp.start()
        logger.info("Coordinador iniciado")
    
    async def stop(self) -> None:
        for comp_id, comp in self.components.items():
            await comp.stop()
        logger.info("Coordinador detenido")

async def run_test():
    """Ejecutar prueba mínima."""
    # Crear coordinador
    coordinator = Coordinator()
    
    # Crear componentes
    comp1 = ComponentAPI("comp1")
    comp2 = ComponentAPI("comp2")
    
    # Registrar componentes
    coordinator.register_component("comp1", comp1)
    coordinator.register_component("comp2", comp2)
    
    # Iniciar coordinador
    await coordinator.start()
    
    try:
        # Hacer solicitud
        result = await coordinator.request(
            "comp2",
            "echo",
            {"message": "Hola desde comp1"},
            "comp1"
        )
        
        # Verificar resultado
        if result and result.get("echo") == "Hola desde comp1":
            logger.info(f"¡Prueba exitosa! Resultado: {result}")
        else:
            logger.error(f"Prueba fallida. Resultado: {result}")
            
    finally:
        # Detener coordinador
        await coordinator.stop()

# Ejecutar prueba
if __name__ == "__main__":
    asyncio.run(run_test())