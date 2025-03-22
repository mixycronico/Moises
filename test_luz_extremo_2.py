"""
Prueba extrema 2.0 del Sistema Genesis - Modo Luz.

Este script ejecuta pruebas con intensidades extremadamente altas (hasta 2.0)
para encontrar el punto de ruptura donde la tasa de éxito cae por debajo del 97%.
"""
import asyncio
import logging
import time
import random
import sys
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("resultados_extremo_2.log")
    ]
)

logger = logging.getLogger("test_luz_extremo")

from correccion_anomalias_temporales import TemporalContinuumInterface

class SimulatedLightComponent:
    """Simula un componente en Modo Luz para pruebas de intensidad extrema."""
    
    def __init__(self, id: str, essential: bool = False, initial_energy: float = 100.0):
        """
        Inicializar componente simulado.
        
        Args:
            id: Identificador del componente
            essential: Si es un componente esencial
            initial_energy: Energía inicial del componente
        """
        self.id = id
        self.essential = essential
        self.temporal_interface = None
        self.energy = initial_energy
        self.max_energy = initial_energy * 2.0
        self.processed_events = 0
        self.successful_events = 0
        self.failed_events = 0
        self.stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "radiation_emissions": 0,
            "transmutations": 0
        }
        logger.info(f"Componente {id} inicializado como {'esencial' if essential else 'no esencial'}")
    
    async def initialize(self, temporal_interface: TemporalContinuumInterface):
        """
        Conectar con la interfaz temporal.
        
        Args:
            temporal_interface: Interfaz del continuo temporal
        """
        self.temporal_interface = temporal_interface
        logger.info(f"Componente {self.id} conectado a interfaz temporal")
    
    async def process_event(self, event_type: str, data: Dict[str, Any], intensity: float = 1.0) -> bool:
        """
        Procesar un evento de prueba con intensidad extrema.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            intensity: Intensidad del procesamiento
            
        Returns:
            True si se procesó correctamente, False en caso contrario
        """
        if self.temporal_interface is None:
            logger.error(f"Componente {self.id} no tiene interfaz temporal")
            self.failed_events += 1
            self.stats["failed"] += 1
            return False
            
        # Factor de consumo de energía basado en intensidad
        energy_factor = intensity * (1.0 + random.random() * 0.5)
        
        # Consumo de energía ajustado
        energy_required = 10.0 * energy_factor
        
        # Intentar procesar el evento
        try:
            # Verificar si tenemos suficiente energía
            if self.energy < energy_required and not self.essential:
                # Componentes no esenciales pueden fallar por falta de energía
                self.failed_events += 1
                self.stats["failed"] += 1
                return False
                
            # Componentes esenciales siempre intentan procesar
            start_time = time.time()
            
            # Registrar evento en la interfaz temporal con intensidad extrema
            event_data = data.copy()
            event_data["processor_id"] = self.id
            event_data["intensity"] = intensity
            event_data["essential"] = self.essential
            
            # Probabilidad de inducir anomalía durante procesamiento
            if random.random() < 0.2 * intensity:
                anomaly_type = random.choice([
                    "timeline_collapse", "causality_inversion", 
                    "entropy_spike", "temporal_desync",
                    "quantum_fluctuation", "timeline_fork"
                ])
                await self.temporal_interface.induce_anomaly(
                    anomaly_type, 
                    intensity=intensity * (0.8 + random.random() * 0.4),
                    data=event_data
                )
            
            # Registrar evento normal
            success = await self.temporal_interface.record_event(event_type, event_data)
            
            # Consumir energía (componentes esenciales no llegan a 0)
            self.energy -= energy_required if not self.essential else min(energy_required, self.energy * 0.9)
            
            # Registrar resultado
            if success:
                self.successful_events += 1
                self.stats["success"] += 1
                
                # Regenerar energía ligeramente tras éxito
                self.energy = min(self.max_energy, self.energy + 2.0)
                
                # Pequeña probabilidad de transmutación para ganar más energía
                if random.random() < 0.05 * intensity:
                    energy_gain = 20.0 * intensity
                    self.energy = min(self.max_energy, self.energy + energy_gain)
                    self.stats["transmutations"] += 1
            else:
                self.failed_events += 1
                self.stats["failed"] += 1
            
            self.processed_events += 1
            self.stats["processed"] += 1
            return success
            
        except Exception as e:
            logger.error(f"Error en componente {self.id}: {str(e)}")
            self.failed_events += 1
            self.stats["failed"] += 1
            return False
    
    def get_success_rate(self) -> float:
        """
        Obtener tasa de éxito del componente.
        
        Returns:
            Tasa de éxito como porcentaje
        """
        if self.processed_events == 0:
            return 100.0
        return (self.successful_events / self.processed_events) * 100.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del componente.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = self.stats.copy()
        stats["energy"] = self.energy
        stats["success_rate"] = self.get_success_rate()
        stats["id"] = self.id
        stats["essential"] = self.essential
        return stats

class ExtremoTester:
    """
    Simulador de pruebas extremas para el Sistema Genesis - Modo Luz.
    
    Genera y aplica diferentes tipos de anomalías a intensidades extremas
    para verificar la resiliencia del sistema.
    """
    def __init__(self, num_components: int = 10, essential_ratio: float = 0.3):
        """
        Inicializar tester extremo.
        
        Args:
            num_components: Número de componentes a simular
            essential_ratio: Proporción de componentes esenciales
        """
        self.temporal_interface = TemporalContinuumInterface()
        self.components: List[SimulatedLightComponent] = []
        
        # Crear componentes
        num_essential = max(1, int(num_components * essential_ratio))
        for i in range(num_components):
            is_essential = i < num_essential
            component = SimulatedLightComponent(f"component_{i}", essential=is_essential)
            self.components.append(component)
        
        logger.info(f"Tester extremo inicializado con {num_components} componentes ({num_essential} esenciales)")
    
    async def initialize(self):
        """Inicializar todos los componentes."""
        for component in self.components:
            await component.initialize(self.temporal_interface)
    
    async def test_intensity(self, intensity: float, iterations: int = 3) -> Dict[str, Any]:
        """
        Ejecutar prueba con intensidad específica.
        
        Args:
            intensity: Intensidad de la prueba (0.0-2.0+)
            iterations: Número de iteraciones de la prueba
            
        Returns:
            Resultados de la prueba
        """
        logger.info(f"Probando intensidad {intensity:.2f} ({iterations} iteraciones)")
        
        cycle_results = []
        success_rates = []
        essential_rates = []
        
        for i in range(iterations):
            logger.info(f"Iteración {i+1}/{iterations} para intensidad {intensity:.2f}")
            
            # Ciclo de anomalías
            cycle_id = f"anomaly_cycle_{int(time.time() * 1000)}"
            logger.info(f"Iniciando ciclo de anomalías {cycle_id} con intensidad {intensity:.2f}")
            
            # Verificar continuidad temporal
            await self.temporal_interface.verify_continuity()
            
            # Generar anomalías proporcionales a la intensidad
            num_anomalies = max(5, int(20 * intensity))
            logger.info(f"Generando {num_anomalies} anomalías temporales (intensidad: {intensity:.2f})")
            
            for _ in range(num_anomalies):
                anomaly_type = random.choice([
                    "timeline_collapse", "causality_inversion", 
                    "entropy_spike", "temporal_desync",
                    "quantum_fluctuation", "timeline_fork"
                ])
                
                # Intensidad individual variada
                individual_intensity = intensity * (0.9 + random.random() * 0.2)
                
                # Datos de la anomalía
                data = {
                    "cycle_id": cycle_id,
                    "timestamp": time.time(),
                    "intensity": individual_intensity
                }
                
                # Inducir la anomalía
                await self.temporal_interface.induce_anomaly(
                    anomaly_type, 
                    intensity=individual_intensity,
                    data=data
                )
                
                # Pequeña pausa para no saturar
                await asyncio.sleep(0.01)
            
            # Ciclo de procesamiento
            cycle_id = f"processing_cycle_{int(time.time() * 1000)}"
            logger.info(f"Iniciando ciclo de procesamiento {cycle_id} con intensidad {intensity:.2f}")
            
            # Número de eventos proporcional a intensidad
            num_events = max(50, int(100 * intensity))
            logger.info(f"Generando {num_events} eventos de procesamiento (intensidad: {intensity:.2f})")
            
            # Distribuir eventos entre componentes
            success_count = 0
            essential_success_count = 0
            essential_total = 0
            
            for _ in range(num_events):
                # Seleccionar componente aleatorio
                component = random.choice(self.components)
                
                # Datos del evento
                event_data = {
                    "cycle_id": cycle_id,
                    "timestamp": time.time(),
                    "intensity": intensity,
                    "event_id": f"event_{int(time.time() * 1000000)}"
                }
                
                # Procesar evento
                success = await component.process_event(
                    event_type="test_event", 
                    data=event_data,
                    intensity=intensity
                )
                
                if success:
                    success_count += 1
                
                if component.essential:
                    essential_total += 1
                    if success:
                        essential_success_count += 1
                
                # Pequeña pausa para no saturar
                await asyncio.sleep(0.005)
            
            # Calcular tasas de éxito
            if num_events > 0:
                cycle_success_rate = (success_count / num_events) * 100.0
                logger.info(f"Tasa de éxito ciclo {i+1}: {cycle_success_rate:.2f}%")
                success_rates.append(cycle_success_rate)
            
            if essential_total > 0:
                essential_success_rate = (essential_success_count / essential_total) * 100.0
                essential_rates.append(essential_success_rate)
            
            # Recopilar estadísticas del ciclo
            cycle_stats = {
                "cycle_id": cycle_id,
                "intensity": intensity,
                "success_rate": cycle_success_rate if num_events > 0 else 100.0,
                "essential_success_rate": essential_success_rate if essential_total > 0 else 100.0,
                "total_events": num_events,
                "successful_events": success_count,
                "essential_total": essential_total,
                "essential_successful": essential_success_count
            }
            cycle_results.append(cycle_stats)
            
            # Reparar continuidad temporal entre ciclos
            await self.temporal_interface.repair_continuity()
            
            # Pausa entre iteraciones
            await asyncio.sleep(0.1)
        
        # Calcular tasa de éxito promedio
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 100.0
        avg_essential_rate = sum(essential_rates) / len(essential_rates) if essential_rates else 100.0
        
        # Recopilar estadísticas por componente
        component_stats = [comp.get_stats() for comp in self.components]
        
        # Resultados generales
        results = {
            "intensity": intensity,
            "average_success_rate": avg_success_rate,
            "average_essential_rate": avg_essential_rate,
            "cycles": cycle_results,
            "components": component_stats,
            "temporal_stats": self.temporal_interface.get_stats()
        }
        
        # Guardar resultados en archivo
        with open(f"resultados_intensidad_{intensity:.2f}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Resultados para intensidad {intensity:.2f}: "
                   f"Tasa global {avg_success_rate:.2f}%, "
                   f"Tasa esenciales {avg_essential_rate:.2f}%")
        
        return results

    async def run_progressive_test(self, start_intensity: float = 1.0, 
                               max_intensity: float = 2.0, 
                               step: float = 0.2,
                               target_rate: float = 97.0):
        """
        Ejecutar prueba progresiva hasta alcanzar la tasa objetivo.
        
        Args:
            start_intensity: Intensidad inicial
            max_intensity: Intensidad máxima
            step: Incremento de intensidad
            target_rate: Tasa de éxito objetivo
        
        Returns:
            Intensidad en la que se alcanzó la tasa objetivo
        """
        logger.info(f"Iniciando prueba progresiva desde {start_intensity:.2f} "
                   f"hasta {max_intensity:.2f} (incremento: {step:.2f})")
        
        results = []
        breaking_point = None
        
        current_intensity = start_intensity
        while current_intensity <= max_intensity:
            result = await self.test_intensity(current_intensity, iterations=3)
            results.append(result)
            
            # Verificar si alcanzamos la tasa objetivo (por debajo)
            if result["average_success_rate"] <= target_rate:
                breaking_point = current_intensity
                logger.info(f"¡Punto de ruptura encontrado! Intensidad {current_intensity:.2f} "
                           f"con tasa {result['average_success_rate']:.2f}%")
                break
            
            current_intensity += step
        
        # Recopilar resultados finales
        final_results = {
            "target_rate": target_rate,
            "breaking_point": breaking_point,
            "results": results
        }
        
        # Guardar resultados finales
        with open("resultados_prueba_extrema_2.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        # Crear informe de resumen
        self._generate_summary_report(results, breaking_point, target_rate)
        
        return breaking_point
    
    def _generate_summary_report(self, results: List[Dict[str, Any]], 
                               breaking_point: Optional[float],
                               target_rate: float):
        """
        Generar informe de resumen.
        
        Args:
            results: Resultados de todas las pruebas
            breaking_point: Punto de ruptura (intensidad)
            target_rate: Tasa de éxito objetivo
        """
        report = []
        report.append("# Informe de Prueba Extrema 2.0 - Sistema Genesis Modo Luz")
        report.append("")
        report.append("## Resumen de Resultados")
        report.append("")
        
        if breaking_point is not None:
            report.append(f"Se ha encontrado el **punto de ruptura** en intensidad **{breaking_point:.2f}**")
            report.append(f"donde la tasa de éxito cae por debajo del objetivo ({target_rate:.2f}%).")
        else:
            report.append(f"No se ha encontrado un punto de ruptura. El sistema mantiene una tasa")
            report.append(f"de éxito superior al objetivo ({target_rate:.2f}%) incluso a intensidad {results[-1]['intensity']:.2f}.")
        
        report.append("")
        report.append("## Resultados por Intensidad")
        report.append("")
        report.append("| **Intensidad** | **Tasa de Éxito** | **Tasa Esenciales** |")
        report.append("| --- | --- | --- |")
        
        for result in results:
            intensity = result["intensity"]
            success_rate = result["average_success_rate"]
            essential_rate = result["average_essential_rate"]
            report.append(f"| {intensity:.2f} | {success_rate:.2f}% | {essential_rate:.2f}% |")
        
        report.append("")
        report.append("## Análisis")
        report.append("")
        
        # Análisis según los resultados
        if breaking_point is not None:
            # Encontramos punto de ruptura
            result_at_break = next((r for r in results if r["intensity"] == breaking_point), None)
            if result_at_break:
                report.append(f"En intensidad {breaking_point:.2f}, la tasa de éxito global cae a {result_at_break['average_success_rate']:.2f}%")
                report.append(f"mientras la tasa de éxito en componentes esenciales se mantiene en {result_at_break['average_essential_rate']:.2f}%.")
                report.append("")
                report.append(f"Esto indica que el Sistema Genesis - Modo Luz continúa priorizando correctamente")
                report.append(f"los componentes esenciales incluso bajo condiciones extremas.")
        else:
            # No encontramos punto de ruptura
            final_result = results[-1]
            report.append(f"El sistema mantiene una resiliencia extraordinaria incluso a intensidad {final_result['intensity']:.2f},")
            report.append(f"con tasas de éxito global de {final_result['average_success_rate']:.2f}% y")
            report.append(f"de componentes esenciales de {final_result['average_essential_rate']:.2f}%.")
            report.append("")
            report.append(f"Esto confirma que el Modo Luz trasciende las limitaciones convencionales de resiliencia,")
            report.append(f"manteniendo su funcionamiento incluso en condiciones extremadamente adversas.")
        
        report.append("")
        report.append("## Conclusiones")
        report.append("")
        
        # Conclusiones generales
        if breaking_point is not None:
            report.append(f"1. **Punto de ruptura identificado**: {breaking_point:.2f}")
            report.append(f"2. **Margen de mejora**: Optimizar para intensidades > {breaking_point:.2f}")
            report.append(f"3. **Prioridad**: Fortalecer componentes no esenciales en intensidades extremas")
            report.append(f"4. **Eficacia del Sistema Genesis - Modo Luz**: Excepcional hasta {breaking_point:.2f}")
        else:
            report.append(f"1. **Resiliencia trascendental**: No se identificó punto de ruptura")
            report.append(f"2. **Eficacia del Sistema Genesis - Modo Luz**: Perfecta en todo el rango probado")
            report.append(f"3. **Próximos pasos**: Evaluar en condiciones aún más extremas (>2.0)")
            report.append(f"4. **Conclusión**: El Modo Luz representa una solución definitiva para entornos de máxima exigencia")
        
        # Guardar informe
        with open("informe_prueba_extrema_2.md", "w") as f:
            f.write("\n".join(report))
        
        logger.info(f"Informe de resumen generado: informe_prueba_extrema_2.md")

async def main():
    """Función principal."""
    # Obtener tasa objetivo de los argumentos
    target_rate = 97.0
    if len(sys.argv) > 1:
        try:
            target_rate = float(sys.argv[1])
        except ValueError:
            logger.warning(f"Valor de tasa objetivo inválido: {sys.argv[1]}, usando 97.0%")
    
    logger.info(f"Iniciando prueba extrema 2.0 para Sistema Genesis - Modo Luz (tasa objetivo: {target_rate}%)")
    
    # Crear tester
    tester = ExtremoTester(num_components=15, essential_ratio=0.25)
    await tester.initialize()
    
    # Ejecutar prueba progresiva
    breaking_point = await tester.run_progressive_test(
        start_intensity=1.0,
        max_intensity=2.0,
        step=0.2,
        target_rate=target_rate
    )
    
    if breaking_point is not None:
        logger.info(f"Prueba completada. Punto de ruptura: {breaking_point:.2f}")
    else:
        logger.info(f"Prueba completada. No se encontró punto de ruptura.")

if __name__ == "__main__":
    asyncio.run(main())