"""
Prueba focalizada del Sistema Genesis en Modo Singularidad Absoluta a intensidad 3.0.

Este script ejecuta una prueba específica para verificar las capacidades
del Sistema Genesis en Modo Singularidad Absoluta a intensidad 3.0.
"""

import asyncio
import logging
import time
import random
import json
from typing import Dict, Any
from genesis_singularity_absolute import (
    SingularityCoordinator, 
    TestComponent, 
    EventPriority
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("intensidad_3.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntensidadTresTester:
    """Tester focalizado para intensidad 3.0."""
    
    def __init__(self):
        """Inicializar tester."""
        self.coordinator = None
        self.components = {}
        self.essential_components = []
        self.non_essential_components = []
        self.success_counters = {
            "total_attempts": 0,
            "total_successes": 0,
            "essential_attempts": 0,
            "essential_successes": 0,
            "non_essential_attempts": 0,
            "non_essential_successes": 0
        }
        
    async def setup(self):
        """Configurar entorno de prueba."""
        logger.info("Configurando entorno para prueba de intensidad 3.0")
        
        # Crear coordinador
        self.coordinator = SingularityCoordinator(host="localhost", port=8080)
        
        # Crear componentes esenciales (3)
        for i in range(3):
            component = TestComponent(f"essential_{i}", is_essential=True)
            self.components[component.id] = component
            self.essential_components.append(component)
            self.coordinator.register_component(component.id, component)
            
        # Crear componentes no esenciales (5)
        for i in range(5):
            component = TestComponent(f"component_{i}", is_essential=False)
            self.components[component.id] = component
            self.non_essential_components.append(component)
            self.coordinator.register_component(component.id, component)
            
        # Iniciar sistema
        await self.coordinator.start()
        
        # Iniciar listeners
        for component_id, component in self.components.items():
            asyncio.create_task(component.listen_local())
            
        logger.info(f"Entorno preparado con {len(self.components)} componentes")
        
    async def test_intensidad_3(self):
        """Ejecutar prueba de intensidad 3.0."""
        intensity = 3.0
        logger.info(f"Iniciando prueba con intensidad {intensity}")
        
        start_time = time.time()
        
        # 1. Generar anomalías extremas
        logger.info("Generando anomalías extremas")
        await self._generate_anomalies(intensity)
        
        # 2. Realizar peticiones de alta carga
        logger.info("Realizando peticiones de prueba")
        await self._make_requests(intensity)
        
        # 3. Calcular tasas de éxito
        success_rates = self._calculate_success_rates()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 4. Guardar resultados
        results = {
            "intensity": intensity,
            "duration": duration,
            "success_rates": success_rates,
            "counters": self.success_counters,
            "timestamp": time.time()
        }
        
        with open("resultados_singularidad_3.00.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Prueba completada en {duration:.2f} segundos")
        logger.info(f"Tasa de éxito global: {success_rates['overall']:.2f}%")
        logger.info(f"Tasa componentes esenciales: {success_rates['essential']:.2f}%")
        logger.info(f"Tasa componentes no esenciales: {success_rates['non_essential']:.2f}%")
        
        return results
        
    async def _generate_anomalies(self, intensity: float):
        """Generar anomalías temporales extremas."""
        # Tipos de anomalías
        anomaly_types = [
            "temporal_distortion", 
            "quantum_uncertainty",
            "dimensional_collapse",
            "reality_breach",
            "causality_violation",
            "probability_storm"
        ]
        
        # Generar 50 anomalías extremas
        num_anomalies = 50
        logger.info(f"Generando {num_anomalies} anomalías (intensidad: {intensity})")
        
        for i in range(num_anomalies):
            # Crear anomalía
            anomaly_type = random.choice(anomaly_types)
            
            # Datos de la anomalía
            anomaly_data = {
                "type": anomaly_type,
                "power": intensity * (0.9 + random.random() * 0.2),
                "timestamp": time.time(),
                "id": f"anomaly_{i}_{int(time.time()*1000)}",
                "coordinates": {
                    "x": random.uniform(-1, 1),
                    "y": random.uniform(-1, 1),
                    "z": random.uniform(-1, 1),
                    "t": random.uniform(-1, 1)
                }
            }
            
            # Emitir evento de anomalía
            await self.coordinator.emit_local(
                f"extreme_anomaly_{anomaly_type}", 
                anomaly_data, 
                "tester",
                priority=EventPriority.CRITICAL,
                intensity=intensity
            )
            
            # Breve pausa cada 10 anomalías
            if i % 10 == 0:
                await asyncio.sleep(0.01)
                
    async def _make_requests(self, intensity: float):
        """Realizar peticiones para probar resiliencia."""
        # Tipos de peticiones
        request_types = [
            "get_data", 
            "process_data", 
            "validate_input", 
            "compute_metrics", 
            "check_status"
        ]
        
        # Realizar 100 peticiones
        num_requests = 100
        logger.info(f"Realizando {num_requests} peticiones (intensidad: {intensity})")
        
        for i in range(num_requests):
            # Seleccionar componente
            is_essential = random.random() < 0.4  # 40% a componentes esenciales
            
            if is_essential and self.essential_components:
                component = random.choice(self.essential_components)
                self.success_counters["essential_attempts"] += 1
            elif self.non_essential_components:
                component = random.choice(self.non_essential_components)
                self.success_counters["non_essential_attempts"] += 1
            else:
                continue
                
            # Tipo de petición
            request_type = random.choice(request_types)
            
            # Datos de petición
            request_data = {
                "id": f"req_{i}_{int(time.time()*1000)}",
                "timestamp": time.time(),
                "parameters": {
                    "param1": random.random() * 100,
                    "param2": random.choice(["option1", "option2", "option3"]),
                    "param3": [random.randint(1, 100) for _ in range(5)]
                }
            }
            
            # Realizar petición
            try:
                result = await self.coordinator.request(
                    component.id,
                    request_type,
                    request_data,
                    "tester",
                    intensity=intensity
                )
                
                # Verificar éxito
                request_success = (
                    result is not None and 
                    isinstance(result, dict) and 
                    result.get("success", False)
                )
                
                # Registrar resultado
                self.success_counters["total_attempts"] += 1
                
                if request_success:
                    self.success_counters["total_successes"] += 1
                    
                    if is_essential:
                        self.success_counters["essential_successes"] += 1
                    else:
                        self.success_counters["non_essential_successes"] += 1
                
            except Exception as e:
                logger.error(f"Error en petición {i}: {str(e)}")
                self.success_counters["total_attempts"] += 1
                
            # Breve pausa cada 10 peticiones
            if i % 10 == 0:
                await asyncio.sleep(0.01)
                
    def _calculate_success_rates(self) -> Dict[str, float]:
        """Calcular tasas de éxito."""
        return {
            "overall": (self.success_counters["total_successes"] / max(1, self.success_counters["total_attempts"])) * 100,
            "essential": (self.success_counters["essential_successes"] / max(1, self.success_counters["essential_attempts"])) * 100,
            "non_essential": (self.success_counters["non_essential_successes"] / max(1, self.success_counters["non_essential_attempts"])) * 100
        }


async def main():
    """Función principal."""
    tester = IntensidadTresTester()
    
    try:
        await tester.setup()
        await tester.test_intensidad_3()
    except Exception as e:
        logger.error(f"Error en prueba: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ejecutar prueba
    asyncio.run(main())