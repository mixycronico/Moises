"""
Prueba mínima para la clase base TranscendentalMechanism
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMinimal")

# Importar directamente la clase base que queremos probar
from genesis.core.transcendental_external_websocket import TranscendentalMechanism

async def main():
    """Función principal de prueba"""
    logger.info("Iniciando prueba mínima de TranscendentalMechanism...")
    
    # Crear instancia de la clase base
    mechanism = TranscendentalMechanism("TestMechanism")
    
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