"""
Prueba de integración de la corrección de anomalías temporales con el Sistema Genesis - Modo Luz.

Este script verifica que la corrección implementada funcione correctamente cuando se integra
con el sistema completo, y que permita realizar pruebas más intensas sin producir errores
de tipo 'object NoneType can't be used in 'await' expression'.
"""

import asyncio
import logging
import time
import random
import sys
from typing import Dict, Any, List, Optional, Tuple
from correccion_anomalias_temporales import TemporalContinuumInterface

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_correccion_temporal.log')
    ]
)
logger = logging.getLogger("test_temporal")

# Simulación simplificada de componentes del sistema Genesis
class SimulatedLightComponent:
    """Simula un componente en Modo Luz para pruebas de integración."""
    
    def __init__(self, id: str, essential: bool = False):
        self.id = id
        self.essential = essential
        self.operational = True
        self.events_processed = 0
        self.failures = 0
        self.last_state = "initialized"
        self.temporal_interface = None
        logger.info(f"Componente {id} inicializado (esencial: {essential})")
        
    async def initialize(self, temporal_interface: TemporalContinuumInterface):
        """Conectar con la interfaz temporal."""
        self.temporal_interface = temporal_interface
        self.last_state = "connected"
        await temporal_interface.record_event("component_initialized", {
            "component_id": self.id,
            "essential": self.essential,
            "timestamp": time.time()
        })
        logger.info(f"Componente {self.id} conectado a interfaz temporal")
        
    async def process_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Procesar un evento usando la interfaz temporal corregida."""
        if not self.operational:
            logger.warning(f"Componente {self.id} no operacional, ignorando evento {event_type}")
            return False
            
        try:
            logger.info(f"Componente {self.id} procesando evento {event_type}")
            
            # Registrar en continuo temporal primero
            if self.temporal_interface:
                await self.temporal_interface.record_event(f"processing_{event_type}", {
                    "component_id": self.id,
                    "data": data,
                    "timestamp": time.time()
                })
            
            # Simular procesamiento
            await asyncio.sleep(0.01)  # Tiempo de procesamiento
            
            # Determinar si hay error aleatorio
            if random.random() < 0.05:  # 5% de fallos
                logger.warning(f"Error aleatorio en componente {self.id} procesando {event_type}")
                self.failures += 1
                
                # Registrar fallo
                if self.temporal_interface:
                    await self.temporal_interface.record_event("component_failure", {
                        "component_id": self.id,
                        "event_type": event_type,
                        "reason": "random_failure",
                        "timestamp": time.time()
                    })
                
                return False
            
            # Éxito
            self.events_processed += 1
            self.last_state = f"processed_{event_type}"
            
            # Registrar éxito
            if self.temporal_interface:
                await self.temporal_interface.record_event("component_success", {
                    "component_id": self.id,
                    "event_type": event_type,
                    "timestamp": time.time()
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Error en componente {self.id} procesando {event_type}: {e}")
            self.failures += 1
            return False

# Simulador de pruebas de estrés para el continuo temporal
class TemporalStressTester:
    """
    Simulador de pruebas de estrés para el continuo temporal.
    
    Genera y aplica diferentes tipos de anomalías a intensidades crecientes
    para verificar la resiliencia del sistema.
    """
    
    def __init__(self, temporal_interface: TemporalContinuumInterface, components: List[SimulatedLightComponent]):
        self.temporal_interface = temporal_interface
        self.components = components
        self.test_cycles = 0
        self.anomalies_tested = 0
        self.anomalies_rejected = 0
        self.anomalies_accepted = 0
        self.radiation_events = 0
        logger.info(f"Inicializado tester de estrés temporal con {len(components)} componentes")
        
    async def run_test_cycle(self, base_intensity: float = 0.05) -> Dict[str, Any]:
        """
        Ejecutar un ciclo de prueba completo a una intensidad base.
        
        Args:
            base_intensity: Intensidad base para las pruebas (0-1)
            
        Returns:
            Resultados del ciclo de prueba
        """
        self.test_cycles += 1
        cycle_id = f"cycle_{self.test_cycles}"
        anomaly_types = ["temporal_desync", "paradox", "temporal_loop"]
        total_anomalies = len(anomaly_types)
        
        logger.info(f"Iniciando ciclo de prueba {cycle_id} con intensidad base {base_intensity}")
        
        results = {
            "cycle_id": cycle_id,
            "base_intensity": base_intensity,
            "anomalies_tested": 0,
            "anomalies_accepted": 0,
            "anomalies_rejected": 0,
            "radiation_events": 0,
            "component_failures": 0,
            "component_successes": 0,
            "continuity_intact": True,
            "details": []
        }
        
        # Verificar continuidad inicial
        continuity_intact, verify_results = await self.temporal_interface.verify_continuity()
        results["initial_continuity"] = continuity_intact
        
        # Probar cada tipo de anomalía
        for anomaly_type in anomaly_types:
            # Incrementar intensidad para este tipo
            for intensity_factor in [1.0, 1.5, 2.0]:
                intensity = base_intensity * intensity_factor
                
                # Limitar a máximo 1.0
                if intensity > 1.0:
                    intensity = 1.0
                
                self.anomalies_tested += 1
                results["anomalies_tested"] += 1
                
                # Datos específicos para el tipo
                anomaly_data = {}
                if anomaly_type == "temporal_desync":
                    anomaly_data = {"desync_factor": intensity * 0.2}
                elif anomaly_type == "paradox":
                    anomaly_data = {"value": random.random()}
                elif anomaly_type == "temporal_loop":
                    anomaly_data = {"value": random.random()}
                
                # Inducir anomalía
                logger.info(f"Probando anomalía {anomaly_type} con intensidad {intensity:.2f}")
                success, result = await self.temporal_interface.induce_anomaly(
                    anomaly_type, 
                    intensity, 
                    anomaly_data
                )
                
                # Registrar resultado
                anomaly_result = {
                    "anomaly_type": anomaly_type,
                    "intensity": intensity,
                    "success": success,
                    "result": result
                }
                
                if success:
                    self.anomalies_accepted += 1
                    results["anomalies_accepted"] += 1
                else:
                    self.anomalies_rejected += 1
                    results["anomalies_rejected"] += 1
                
                # Detectar radiación
                if result and result.get("radiation_emitted", False):
                    self.radiation_events += 1
                    results["radiation_events"] += 1
                
                results["details"].append(anomaly_result)
                
                # Probar componentes después de cada anomalía
                component_results = await self._test_components_after_anomaly(anomaly_type, intensity)
                
                results["component_failures"] += component_results["failures"]
                results["component_successes"] += component_results["successes"]
                
                # Pequeña pausa para estabilización
                await asyncio.sleep(0.02)
        
        # Verificar continuidad final
        continuity_intact, verify_results = await self.temporal_interface.verify_continuity()
        results["final_continuity"] = continuity_intact
        
        # Si hay problemas, intentar reparar
        if not continuity_intact:
            logger.warning(f"Continuidad temporal comprometida al final del ciclo {cycle_id}")
            repair_results = await self.temporal_interface.repair_continuity()
            results["repair_results"] = repair_results
            
            # Verificar después de reparación
            continuity_intact, verify_results = await self.temporal_interface.verify_continuity()
            results["post_repair_continuity"] = continuity_intact
        
        return results
    
    async def _test_components_after_anomaly(self, anomaly_type: str, intensity: float) -> Dict[str, int]:
        """
        Probar componentes después de inducir una anomalía.
        
        Args:
            anomaly_type: Tipo de anomalía inducida
            intensity: Intensidad de la anomalía
            
        Returns:
            Resultados de las pruebas de componentes
        """
        successes = 0
        failures = 0
        
        # Generar evento de prueba
        test_event = f"post_{anomaly_type}"
        test_data = {
            "anomaly_type": anomaly_type,
            "intensity": intensity,
            "timestamp": time.time()
        }
        
        # Probar cada componente
        for component in self.components:
            success = await component.process_event(test_event, test_data)
            if success:
                successes += 1
            else:
                failures += 1
        
        return {
            "successes": successes,
            "failures": failures
        }

# Funciones principales de prueba
async def setup_test_environment(num_components: int = 5) -> Tuple[TemporalContinuumInterface, List[SimulatedLightComponent], TemporalStressTester]:
    """
    Configurar entorno de prueba completo.
    
    Args:
        num_components: Número de componentes a crear
        
    Returns:
        Tupla (interfaz_temporal, componentes, tester)
    """
    logger.info(f"Configurando entorno de prueba con {num_components} componentes")
    
    # Crear interfaz temporal
    temporal_interface = TemporalContinuumInterface()
    
    # Crear componentes
    components = []
    for i in range(num_components):
        essential = i < 2  # Los primeros 2 son esenciales
        component = SimulatedLightComponent(f"component_{i}", essential)
        await component.initialize(temporal_interface)
        components.append(component)
    
    # Crear tester
    tester = TemporalStressTester(temporal_interface, components)
    
    return temporal_interface, components, tester

async def run_graduated_stress_test(initial_intensity: float = 0.05, max_intensity: float = 0.25, steps: int = 5) -> Dict[str, Any]:
    """
    Ejecutar prueba de estrés con incremento gradual de intensidad.
    
    Args:
        initial_intensity: Intensidad inicial
        max_intensity: Intensidad máxima
        steps: Número de pasos de incremento
        
    Returns:
        Resultados completos de la prueba
    """
    logger.info(f"Iniciando prueba de estrés graduada: {initial_intensity} -> {max_intensity} en {steps} pasos")
    
    # Configurar entorno
    temporal, components, tester = await setup_test_environment(10)  # 10 componentes
    
    # Incremento por paso
    intensity_step = (max_intensity - initial_intensity) / steps
    
    # Resultados globales
    results = {
        "initial_intensity": initial_intensity,
        "max_intensity": max_intensity,
        "steps": steps,
        "cycles": [],
        "component_success_rate": 0.0,
        "anomaly_acceptance_rate": 0.0,
        "radiation_events": 0,
        "continuity_maintained": True
    }
    
    # Ejecutar ciclos con intensidad creciente
    current_intensity = initial_intensity
    for step in range(steps + 1):  # +1 para incluir el paso final
        logger.info(f"Paso {step+1}/{steps+1}: intensidad {current_intensity:.3f}")
        
        # Ejecutar ciclo
        cycle_results = await tester.run_test_cycle(current_intensity)
        results["cycles"].append(cycle_results)
        
        # Sumar radiaciones
        results["radiation_events"] += cycle_results["radiation_events"]
        
        # Verificar continuidad
        if not cycle_results.get("final_continuity", False):
            results["continuity_maintained"] = False
        
        # Incrementar para próximo ciclo
        current_intensity += intensity_step
        if current_intensity > max_intensity:
            current_intensity = max_intensity
    
    # Calcular estadísticas finales
    total_anomalies = 0
    accepted_anomalies = 0
    total_component_tests = 0
    successful_component_tests = 0
    
    for cycle in results["cycles"]:
        total_anomalies += cycle["anomalies_tested"]
        accepted_anomalies += cycle["anomalies_accepted"]
        total_component_tests += cycle["component_successes"] + cycle["component_failures"]
        successful_component_tests += cycle["component_successes"]
    
    # Calcular tasas
    if total_anomalies > 0:
        results["anomaly_acceptance_rate"] = accepted_anomalies / total_anomalies * 100
    
    if total_component_tests > 0:
        results["component_success_rate"] = successful_component_tests / total_component_tests * 100
    
    # Estadísticas de la interfaz temporal
    results["temporal_stats"] = temporal.get_stats()
    
    return results

# Ejecutar prueba
async def main():
    logger.info("Iniciando prueba de corrección de anomalías temporales")
    
    try:
        # Prueba graduada
        results = await run_graduated_stress_test(
            initial_intensity=0.05,
            max_intensity=0.25,
            steps=5
        )
        
        # Resumir resultados
        logger.info("=== RESUMEN DE RESULTADOS ===")
        logger.info(f"Tasa de éxito en componentes: {results['component_success_rate']:.2f}%")
        logger.info(f"Tasa de aceptación de anomalías: {results['anomaly_acceptance_rate']:.2f}%")
        logger.info(f"Eventos de radiación primordial: {results['radiation_events']}")
        logger.info(f"Continuidad temporal mantenida: {results['continuity_maintained']}")
        logger.info("=============================")
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Prueba completada")

if __name__ == "__main__":
    asyncio.run(main())