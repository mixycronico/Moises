"""
Prueba extrema progresiva para el Sistema Genesis - Modo Luz.

Este script incrementa gradualmente la intensidad de las pruebas
hasta que la tasa de éxito del sistema caiga por debajo del 98%.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, List, Tuple, Optional
import sys

from correccion_anomalias_temporales import TemporalContinuumInterface

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("resultados_extremo_progresivo.log")
    ]
)

logger = logging.getLogger("test_luz_extremo")

class SimulatedComponent:
    """Simula un componente para pruebas extremas."""
    
    def __init__(self, id: str, is_essential: bool = False):
        """
        Inicializar componente simulado.
        
        Args:
            id: Identificador del componente
            is_essential: Si es un componente esencial
        """
        self.id = id
        self.is_essential = is_essential
        self.temporal_interface = None
        self.events_processed = 0
        self.events_success = 0
        self.anomalies_processed = 0
        self.anomalies_rejected = 0
        self.emissions = 0
        
    async def connect_temporal_interface(self, interface: TemporalContinuumInterface) -> None:
        """
        Conectar a interfaz temporal.
        
        Args:
            interface: Interfaz temporal
        """
        self.temporal_interface = interface
        logger.info(f"Componente {self.id} conectado a interfaz temporal")
        
    async def process_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """
        Procesar evento simulado.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            
        Returns:
            True si procesado con éxito, False en caso contrario
        """
        if not self.temporal_interface:
            return False
            
        self.events_processed += 1
        
        # Registrar evento en continuo temporal
        try:
            result = await self.temporal_interface.record_event(event_type, data)
            if result:
                self.events_success += 1
                return True
            return False
        except Exception as e:
            logger.error(f"Error al procesar evento en {self.id}: {e}")
            return False
            
    async def induce_anomaly(self, anomaly_type: str, intensity: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Inducir anomalía temporal.
        
        Args:
            anomaly_type: Tipo de anomalía
            intensity: Intensidad (0-1)
            
        Returns:
            Tupla (éxito, resultado)
        """
        if not self.temporal_interface:
            return False, None
            
        self.anomalies_processed += 1
        
        try:
            result = await self.temporal_interface.induce_anomaly(anomaly_type, intensity)
            if not result[0]:
                self.anomalies_rejected += 1
            return result
        except Exception as e:
            logger.error(f"Error al inducir anomalía en {self.id}: {e}")
            return False, None
            
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del componente."""
        return {
            "id": self.id,
            "is_essential": self.is_essential,
            "events_processed": self.events_processed,
            "events_success": self.events_success,
            "anomalies_processed": self.anomalies_processed,
            "anomalies_rejected": self.anomalies_rejected,
            "success_rate": self.events_success / max(1, self.events_processed) * 100
        }

class ExtremeProgressiveTester:
    """Tester extremo progresivo para Sistema Genesis - Modo Luz."""
    
    def __init__(self, 
                 component_count: int = 10, 
                 essential_ratio: float = 0.3, 
                 target_success_rate: float = 98.0):
        """
        Inicializar tester.
        
        Args:
            component_count: Número de componentes a simular
            essential_ratio: Ratio de componentes esenciales (0-1)
            target_success_rate: Tasa de éxito objetivo para detener las pruebas (%)
        """
        self.component_count = component_count
        self.essential_ratio = essential_ratio
        self.target_success_rate = target_success_rate
        self.components: List[SimulatedComponent] = []
        self.essential_components: List[SimulatedComponent] = []
        self.non_essential_components: List[SimulatedComponent] = []
        self.temporal_interface = TemporalContinuumInterface()
        self.stopped = False
        self.current_intensity = 0.0
        self.current_success_rate = 100.0
        self.iterations_per_intensity = 3
        self.last_test_results = {}
        
    async def setup(self) -> None:
        """Configurar entorno de prueba."""
        logger.info(f"Configurando entorno de prueba extremo progresivo con {self.component_count} componentes")
        
        # Crear componentes
        essential_count = int(self.component_count * self.essential_ratio)
        
        for i in range(self.component_count):
            is_essential = i < essential_count
            component = SimulatedComponent(f"component_{i}", is_essential)
            await component.connect_temporal_interface(self.temporal_interface)
            self.components.append(component)
            
            if is_essential:
                self.essential_components.append(component)
            else:
                self.non_essential_components.append(component)
                
        logger.info(f"Tester extremo inicializado con {len(self.components)} componentes " +
                  f"({len(self.essential_components)} esenciales)")
        
    async def run_intensity_test(self, intensity: float, iterations: int = 3) -> Dict[str, Any]:
        """
        Ejecutar prueba con intensidad específica.
        
        Args:
            intensity: Intensidad de la prueba (0-1)
            iterations: Número de iteraciones
            
        Returns:
            Resultados de la prueba
        """
        logger.info(f"Probando intensidad {intensity:.2f} ({iterations} iteraciones)")
        
        anomaly_results = []
        processing_results = []
        combined_results = []
        
        for i in range(iterations):
            logger.info(f"Iteración {i+1}/{iterations} para intensidad {intensity:.2f}")
            
            # Ejecutar ciclo de prueba
            anomaly_cycle_result = await self._run_anomaly_cycle(intensity)
            anomaly_results.append(anomaly_cycle_result)
            
            processing_cycle_result = await self._run_processing_cycle(intensity)
            processing_results.append(processing_cycle_result)
            
            # Combinar resultados
            cycle_success_rate = (anomaly_cycle_result["success_rate"] + processing_cycle_result["success_rate"]) / 2
            combined_results.append(cycle_success_rate)
            
            logger.info(f"Tasa de éxito ciclo {i+1}: {cycle_success_rate:.2f}%")
            
            # Si estamos muy por debajo del objetivo, detener
            if cycle_success_rate < self.target_success_rate - 5:
                logger.info(f"Tasa de éxito por debajo del objetivo por más de 5% ({cycle_success_rate:.2f}% < {self.target_success_rate}% - 5%), deteniendo...")
                self.stopped = True
                break
                
            # Pausa entre ciclos
            await asyncio.sleep(0.5)
            
        # Calcular resultados agregados
        avg_anomaly_success = sum(r["success_rate"] for r in anomaly_results) / len(anomaly_results)
        avg_processing_success = sum(r["success_rate"] for r in processing_results) / len(processing_results)
        avg_combined_success = sum(combined_results) / len(combined_results)
        
        # Calcular tasas específicas para componentes esenciales
        essential_success_rates = [c.get_stats()["success_rate"] for c in self.essential_components]
        non_essential_success_rates = [c.get_stats()["success_rate"] for c in self.non_essential_components]
        
        essential_avg = sum(essential_success_rates) / len(essential_success_rates) if essential_success_rates else 100.0
        non_essential_avg = sum(non_essential_success_rates) / len(non_essential_success_rates) if non_essential_success_rates else 100.0
        
        results = {
            "intensity": intensity,
            "iterations": len(combined_results),
            "anomaly_success_rate": avg_anomaly_success,
            "processing_success_rate": avg_processing_success,
            "combined_success_rate": avg_combined_success,
            "essential_success_rate": essential_avg,
            "non_essential_success_rate": non_essential_avg,
            "success_rates_per_iteration": combined_results,
            "below_target": avg_combined_success < self.target_success_rate
        }
        
        logger.info(f"Tasa de éxito para intensidad {intensity:.2f}: {avg_combined_success:.2f}%")
        logger.info(f"   - Componentes esenciales: {essential_avg:.2f}%")
        logger.info(f"   - Componentes no esenciales: {non_essential_avg:.2f}%")
        
        self.current_success_rate = avg_combined_success
        return results
        
    async def _run_anomaly_cycle(self, intensity: float) -> Dict[str, Any]:
        """
        Ejecutar ciclo de anomalías temporales.
        
        Args:
            intensity: Intensidad del ciclo (0-1)
            
        Returns:
            Resultados del ciclo
        """
        cycle_id = f"anomaly_cycle_{int(time.time() * 1000)}"
        logger.info(f"Iniciando ciclo de anomalías {cycle_id} con intensidad {intensity:.2f}")
        
        # Verificar continuidad temporal al inicio
        await self.temporal_interface.verify_continuity()
        
        # Calcular número de anomalías basado en intensidad
        anomaly_count = max(3, int(50 * intensity))
        logger.info(f"Generando {anomaly_count} anomalías temporales (intensidad: {intensity:.2f})")
        
        # Tipos de anomalías posibles
        anomaly_types = [
            "temporal_desync", "timeline_fork", "quantum_fluctuation", 
            "causality_inversion", "entropy_spike", "timeline_collapse"
        ]
        
        # Inducir anomalías
        success_count = 0
        total_count = 0
        
        for _ in range(anomaly_count):
            # Seleccionar componente y tipo de anomalía
            component = random.choice(self.components)
            anomaly_type = random.choice(anomaly_types)
            
            # La intensidad real varía ligeramente alrededor de la intensidad base
            actual_intensity = min(1.0, max(0.01, intensity * random.uniform(0.9, 1.1)))
            
            # Intentar inducir anomalía
            result = await component.induce_anomaly(anomaly_type, actual_intensity)
            total_count += 1
            
            if result[0]:  # Si no fue rechazada
                success_count += 1
                
            # Pequeña pausa entre anomalías
            await asyncio.sleep(0.01)
            
        # Verificar continuidad temporal al final
        await self.temporal_interface.verify_continuity()
        
        # Calcular tasa de éxito (en anomalías, el rechazo es éxito)
        success_rate = (total_count - success_count) / total_count * 100 if total_count > 0 else 100.0
        
        return {
            "cycle_id": cycle_id,
            "anomaly_count": anomaly_count,
            "rejected_count": total_count - success_count,
            "total_count": total_count,
            "success_rate": success_rate
        }
        
    async def _run_processing_cycle(self, intensity: float) -> Dict[str, Any]:
        """
        Ejecutar ciclo de procesamiento de eventos.
        
        Args:
            intensity: Intensidad del ciclo (0-1)
            
        Returns:
            Resultados del ciclo
        """
        cycle_id = f"processing_cycle_{int(time.time() * 1000)}"
        logger.info(f"Iniciando ciclo de procesamiento {cycle_id} con intensidad {intensity:.2f}")
        
        # Calcular número de eventos basado en intensidad
        event_count = max(5, int(100 * intensity))
        logger.info(f"Generando {event_count} eventos de procesamiento (intensidad: {intensity:.2f})")
        
        # Tipos de eventos posibles
        event_types = [
            "data_update", "status_check", "configuration_change", 
            "security_scan", "resource_allocation", "background_task"
        ]
        
        # Procesar eventos
        success_count = 0
        total_count = 0
        
        for _ in range(event_count):
            # Seleccionar componente y tipo de evento
            component = random.choice(self.components)
            event_type = random.choice(event_types)
            
            # Crear datos del evento
            data = {
                "timestamp": time.time(),
                "intensity": intensity,
                "component": component.id,
                "cycle_id": cycle_id
            }
            
            # Intentar procesar evento
            result = await component.process_event(event_type, data)
            total_count += 1
            
            if result:
                success_count += 1
                
            # Pequeña pausa entre eventos
            await asyncio.sleep(0.005)
            
        # Calcular tasa de éxito
        success_rate = success_count / total_count * 100 if total_count > 0 else 100.0
        
        return {
            "cycle_id": cycle_id,
            "event_count": event_count,
            "success_count": success_count,
            "total_count": total_count,
            "success_rate": success_rate
        }
        
    async def run_progressive_test(self, 
                                  start_intensity: float = 0.1, 
                                  max_intensity: float = 1.0,
                                  intensity_step: float = 0.1) -> Dict[str, Any]:
        """
        Ejecutar prueba progresiva aumentando intensidad.
        
        Args:
            start_intensity: Intensidad inicial
            max_intensity: Intensidad máxima
            intensity_step: Incremento de intensidad por paso
            
        Returns:
            Resultados completos de la prueba
        """
        logger.info(f"Iniciando prueba progresiva desde {start_intensity:.2f} " +
                  f"hasta {max_intensity:.2f} (incremento: {intensity_step:.2f})")
                  
        results = {}
        current_intensity = start_intensity
        
        while current_intensity <= max_intensity and not self.stopped:
            # Ejecutar prueba con intensidad actual
            intensity_results = await self.run_intensity_test(
                current_intensity, 
                self.iterations_per_intensity
            )
            
            results[current_intensity] = intensity_results
            self.last_test_results = intensity_results
            
            # Si estamos por debajo del objetivo, detener
            if intensity_results["below_target"]:
                logger.info(f"Intensidad {current_intensity:.2f} produjo tasa por debajo del objetivo " +
                          f"({intensity_results['combined_success_rate']:.2f}% < {self.target_success_rate}%), deteniendo...")
                break
                
            # Siguiente intensidad
            current_intensity += intensity_step
            self.current_intensity = current_intensity
        
        # Calcular punto de quiebre
        breaking_point = None
        for intensity, result in sorted(results.items()):
            if result["below_target"]:
                breaking_point = intensity
                break
                
        logger.info(f"Prueba progresiva completa. Punto de quiebre: {breaking_point or 'No encontrado'}")
        logger.info(f"Última tasa de éxito medida: {self.current_success_rate:.2f}%")
        
        return {
            "breaking_point": breaking_point,
            "final_success_rate": self.current_success_rate,
            "target_success_rate": self.target_success_rate,
            "results_by_intensity": results
        }
        
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generar reporte de resultados.
        
        Args:
            results: Resultados de la prueba
            
        Returns:
            Reporte en formato string
        """
        report = []
        report.append("=" * 80)
        report.append("Reporte de Prueba Extrema Progresiva - Sistema Genesis Modo Luz")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"Objetivo de tasa de éxito: {self.target_success_rate:.2f}%")
        report.append(f"Punto de quiebre: {results['breaking_point'] or 'No encontrado'}")
        report.append(f"Tasa de éxito final: {results['final_success_rate']:.2f}%")
        report.append("")
        
        report.append("Resultados por nivel de intensidad:")
        report.append("-" * 60)
        for intensity, result in sorted(results['results_by_intensity'].items()):
            report.append(f"Intensidad {intensity:.2f}:")
            report.append(f"  Tasa de éxito combinada: {result['combined_success_rate']:.2f}%")
            report.append(f"  Tasa de componentes esenciales: {result['essential_success_rate']:.2f}%")
            report.append(f"  Tasa de componentes no esenciales: {result['non_essential_success_rate']:.2f}%")
            report.append("")
            
        report.append("=" * 80)
        report.append("Estadísticas por componente:")
        report.append("-" * 60)
        for component in self.components:
            stats = component.get_stats()
            report.append(f"Componente {stats['id']} {'(esencial)' if stats['is_essential'] else ''}:")
            report.append(f"  Eventos procesados: {stats['events_processed']}")
            report.append(f"  Eventos exitosos: {stats['events_success']}")
            report.append(f"  Tasa de éxito: {stats['success_rate']:.2f}%")
            report.append(f"  Anomalías procesadas: {stats['anomalies_processed']}")
            report.append(f"  Anomalías rechazadas: {stats['anomalies_rejected']}")
            report.append("")
            
        return "\n".join(report)

async def main():
    """Función principal."""
    logger.info("Iniciando prueba extrema progresiva para Sistema Genesis - Modo Luz")
    
    # Configurar tester
    target_rate = 98.0
    if len(sys.argv) > 1:
        try:
            target_rate = float(sys.argv[1])
        except ValueError:
            pass
            
    tester = ExtremeProgressiveTester(
        component_count=15,
        essential_ratio=0.3,
        target_success_rate=target_rate
    )
    
    await tester.setup()
    
    # Ejecutar prueba progresiva
    results = await tester.run_progressive_test(
        start_intensity=0.1,
        max_intensity=2.0,  # Intensidad máxima de 2.0 (200%)
        intensity_step=0.1
    )
    
    # Generar reporte
    report = tester.generate_report(results)
    
    # Guardar reporte
    report_file = "reporte_extremo_progresivo.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    logger.info(f"Reporte guardado en {report_file}")
    logger.info(f"Punto de quiebre: {results['breaking_point'] or 'No encontrado'}")
    logger.info(f"Tasa de éxito final: {results['final_success_rate']:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())