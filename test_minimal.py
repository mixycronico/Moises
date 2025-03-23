"""
Prueba mínima para verificar la funcionalidad asíncrona básica.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMinimal")

class TestMechanism:
    """Mecanismo de prueba para verificar async/await."""
    
    def __init__(self):
        """Inicializar estado del mecanismo."""
        self.stats = {"invocations": 0}
        
    async def _register_invocation(self, success: bool = True) -> None:
        """Registrar una invocación con cambio de contexto asíncrono."""
        self.stats["invocations"] += 1
        
        # Permitir cambio de contexto para evitar bloqueos
        await asyncio.sleep(0)
        
        # Pequeña pausa para simular trabajo
        await asyncio.sleep(0.01)
        
        logger.info(f"Invocación registrada. Total: {self.stats['invocations']}")
    
    async def test_operation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Operación de prueba que usa el método asíncrono."""
        logger.info(f"Iniciando operación con: {data}")
        
        # Registrar invocación (asíncrono)
        await self._register_invocation()
        
        # Simular procesamiento (asíncrono)
        await asyncio.sleep(0.1)
        
        # Retornar resultado
        result = {
            "processed": True,
            "timestamp": time.time(),
            "input": data
        }
        
        logger.info(f"Operación completada: {result}")
        return result

async def main():
    """Función principal."""
    logger.info("Iniciando prueba mínima...")
    
    # Crear instancia de prueba
    mechanism = TestMechanism()
    
    # Ejecutar operación simple
    data = {"test": "data", "value": 123}
    result = await mechanism.test_operation(data)
    
    logger.info(f"Resultado final: {result}")
    logger.info("Prueba finalizada.")

if __name__ == "__main__":
    # Ejecutar con timeout para asegurar que termine
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=2.0))
        logger.info("Prueba completada dentro del tiempo establecido.")
    except asyncio.TimeoutError:
        logger.error("La prueba excedió el tiempo máximo permitido.")