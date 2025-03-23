"""
Prueba ultra-simplificada de los mecanismos trascendentales.
"""

import logging
import time
import random

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger("Test.UltraSimple")

class SimpleMechanism:
    """Versión simplificada de un mecanismo trascendental."""
    
    def __init__(self, name):
        self.name = name
        self.invocations = 0
        logger.info(f"Mecanismo {name} inicializado")
    
    def process(self, data):
        """Procesar datos con el mecanismo."""
        self.invocations += 1
        
        # Simular procesamiento
        processed = data.copy()
        processed["processed_by"] = self.name
        processed["timestamp"] = time.time()
        
        logger.info(f"Datos procesados por {self.name}: {processed}")
        return processed
    
    def get_stats(self):
        """Obtener estadísticas del mecanismo."""
        stats = {
            "name": self.name,
            "invocations": self.invocations
        }
        logger.info(f"Estadísticas de {self.name}: {stats}")
        return stats

def main():
    """Función principal."""
    logger.info("INICIANDO PRUEBA ULTRA-SIMPLE")
    
    # Crear mecanismo
    mechanism = SimpleMechanism("TestMechanism")
    
    # Datos de prueba
    test_data = {"id": f"test_{int(time.time())}", "value": random.random() * 100}
    
    # Procesar datos
    processed = mechanism.process(test_data)
    logger.info(f"Resultado: {processed}")
    
    # Obtener estadísticas
    stats = mechanism.get_stats()
    logger.info(f"Estadísticas finales: {stats}")
    
    logger.info("PRUEBA COMPLETADA")

if __name__ == "__main__":
    main()