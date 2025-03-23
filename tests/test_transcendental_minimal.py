"""
Prueba mínima de un solo mecanismo trascendental.
"""

import asyncio
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger("Test.Minimal")

# Importar solo un mecanismo
from genesis.core.transcendental_external_websocket import DimensionalCollapseV4

async def test_single_mechanism():
    """Probar un solo mecanismo."""
    logger.info("=== PRUEBA MÍNIMA ===")
    
    mechanism = DimensionalCollapseV4()
    
    # Datos de prueba simples
    test_data = {"test": "data", "value": 42}
    
    # Colapsar datos
    result = await mechanism.collapse_data(test_data)
    
    # Verificar resultado
    logger.info(f"Datos originales: {test_data}")
    logger.info(f"Datos colapsados: {result}")
    
    # Obtener estadísticas
    stats = mechanism.get_stats()
    logger.info(f"Estadísticas: {stats}")
    
    return True

async def main():
    """Función principal."""
    logger.info("INICIANDO PRUEBA MÍNIMA")
    
    try:
        result = await test_single_mechanism()
        if result:
            logger.info("✅ PRUEBA EXITOSA")
        else:
            logger.error("❌ PRUEBA FALLIDA")
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
    
    logger.info("PRUEBA COMPLETADA")

if __name__ == "__main__":
    asyncio.run(main())