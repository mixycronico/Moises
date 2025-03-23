"""
Prueba simplificada de los mecanismos core del Sistema Genesis Singularidad Trascendental V4.
"""

import asyncio
import logging
import time
import json
import sys
import os
from typing import Dict, Any

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Genesis.CoreTest")

# Importar componentes del sistema
from genesis_singularity_transcendental_v4 import (
    DimensionalCollapseV4,
    EventHorizonV4,
    QuantumTimeV4
)

async def test_core_mechanisms():
    """Prueba simplificada de los mecanismos trascendentales."""
    logger.info("=== INICIANDO PRUEBA SIMPLIFICADA DE MECANISMOS CORE ===")
    
    # Inicializar algunos mecanismos clave
    collapse = DimensionalCollapseV4()
    horizon = EventHorizonV4()
    time_mechanism = QuantumTimeV4()
    
    # Estadísticas de prueba
    stats = {
        "start_time": time.time(),
        "operations": 0,
        "successes": 0,
        "failures": 0
    }
    
    # Realizar algunas operaciones de prueba
    try:
        # Prueba de colapso dimensional
        logger.info("Probando colapso dimensional...")
        data = {"test": 1, "complex": {"nested": [1, 2, 3]}}
        result = await collapse.collapse_data(data)
        assert result is not None
        stats["operations"] += 1
        stats["successes"] += 1
        logger.info("Colapso dimensional exitoso")
        
        # Prueba de horizonte de eventos
        logger.info("Probando horizonte de eventos...")
        try:
            # Provocar error intencionalmente
            raise ValueError("Error simulado para prueba")
        except Exception as e:
            result = await horizon.transmute_error(e, {"operation": "test"})
            assert result["transmuted"]
            stats["operations"] += 1
            stats["successes"] += 1
            logger.info("Horizonte de eventos exitoso (transmutación de error)")
        
        # Prueba de tiempo cuántico
        logger.info("Probando tiempo cuántico...")
        async with time_mechanism.nullify_time():
            # Operación que sería costosa en tiempo normal
            await asyncio.sleep(0.001)
            stats["operations"] += 1
            stats["successes"] += 1
            logger.info("Tiempo cuántico exitoso")
            
    except Exception as e:
        logger.error(f"Error en prueba: {str(e)}")
        stats["failures"] += 1
    
    # Completar estadísticas
    elapsed = time.time() - stats["start_time"]
    stats.update({
        "total_time": elapsed,
        "success_rate": (stats["successes"] / stats["operations"] * 100) 
                    if stats["operations"] > 0 else 0
    })
    
    logger.info(f"=== RESULTADOS DE PRUEBA ===")
    logger.info(f"Operaciones: {stats['operations']}")
    logger.info(f"Éxitos: {stats['successes']}")
    logger.info(f"Tasa de éxito: {stats['success_rate']:.2f}%")
    logger.info(f"Tiempo total: {stats['total_time']:.6f}s")
    
    return stats

async def main():
    """Función principal."""
    start_time = time.time()
    
    # Ejecutar prueba
    results = await test_core_mechanisms()
    
    # Guardar resultados
    with open("resultados_prueba_simple.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"Prueba completada en {total_time:.2f}s")
    logger.info(f"Resultados guardados en resultados_prueba_simple.json")

if __name__ == "__main__":
    asyncio.run(main())