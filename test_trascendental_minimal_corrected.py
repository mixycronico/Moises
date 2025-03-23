"""
Prueba mínima para una implementación simplificada de TranscendentalMechanism
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMinimal")

class SimplifiedTranscendentalMechanism:
    """Versión simplificada de TranscendentalMechanism para pruebas."""
    
    def __init__(self, name: str):
        """Inicializar mecanismo trascendental."""
        self.name = name
        self.stats = {
            "invocations": 0,
            "success_rate": 100.0,
            "last_invocation": None
        }
        
    async def _register_invocation(self, success: bool = True) -> None:
        """
        Registrar invocación del mecanismo de forma asíncrona.
        
        Args:
            success: Si la invocación fue exitosa
        """
        self.stats["invocations"] += 1
        self.stats["last_invocation"] = time.time()
        
        # Actualizar tasa de éxito (peso histórico 0.95)
        if self.stats["invocations"] > 1:
            self.stats["success_rate"] = (
                0.95 * self.stats["success_rate"] + 
                (0.05 * (100.0 if success else 0.0))
            )
        
        # Permitir cambio de contexto para evitar bloqueos
        await asyncio.sleep(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del mecanismo."""
        return self.stats.copy()

async def main():
    """Función principal de prueba"""
    logger.info("Iniciando prueba mínima de mecanismo trascendental simplificado...")
    
    # Crear instancia de la clase simplificada
    mechanism = SimplifiedTranscendentalMechanism("TestMechanism")
    
    # Registrar algunas invocaciones
    for i in range(3):
        logger.info(f"Registrando invocación {i+1}...")
        await mechanism._register_invocation(success=True)
        
        # Verificar que el contador se incrementa
        logger.info(f"Estadísticas después de invocación {i+1}: {json.dumps(mechanism.get_stats())}")
    
    logger.info("Prueba finalizada.")

if __name__ == "__main__":
    # Ejecutar con timeout para asegurar que termine
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=2.0))
        logger.info("Prueba completada dentro del tiempo establecido.")
    except asyncio.TimeoutError:
        logger.error("La prueba excedió el tiempo máximo permitido.")