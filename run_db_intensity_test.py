"""
Script para ejecutar las pruebas de intensidad gradual en la base de datos.
"""
import sys
import asyncio
import logging
from test_db_intensity_gradual import main, TestIntensity, run_tests_with_intensity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("run_db_test")

async def run_specific_intensity(intensity_name):
    """
    Ejecutar una prueba con intensidad específica.
    
    Args:
        intensity_name: Nombre de la intensidad (BASIC, MEDIUM, HIGH, EXTREME)
    """
    try:
        intensity = TestIntensity[intensity_name.upper()]
        logger.info(f"Ejecutando prueba con intensidad {intensity.name}")
        await run_tests_with_intensity(intensity)
    except KeyError:
        logger.error(f"Intensidad no válida: {intensity_name}")
        logger.info("Intensidades válidas: BASIC, MEDIUM, HIGH, EXTREME")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Si se proporciona un argumento, ejecutar solo esa intensidad
        intensity_name = sys.argv[1].upper()
        asyncio.run(run_specific_intensity(intensity_name))
    else:
        # Si no hay argumentos, ejecutar todas las intensidades graduales
        asyncio.run(main())