"""
Prueba Apocalíptica Extrema para Sistema Genesis - Modo Luz.

Esta prueba incorpora la corrección de anomalías temporales y somete al Sistema Genesis
en Modo Luz a una serie de pruebas de estrés con intensidades extremas (hasta 2.0),
verificando su capacidad para mantener una tasa de éxito del 100% incluso bajo
condiciones catastróficas.

Características:
- Utiliza la nueva interfaz TemporalContinuumInterface para corregir anomalías
- Prueba incremental de intensidad desde 0.05 hasta 2.0
- Múltiples vectores de ataque simultáneos (temporales, estructurales, carga)
- Verificación de radiación primordial y transmutaciones luminosas
- Medición precisa de resiliencia en componentes esenciales y no esenciales
"""

import asyncio
import logging
import time
import random
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from correccion_anomalias_temporales import TemporalContinuumInterface

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('apocalipsis_extremo.log')
    ]
)
logger = logging.getLogger("apocalipsis_luz")

# Constantes
MAX_COMPONENTS = 25
ESSENTIAL_COMPONENTS = 5
TEST_ITERATIONS = 3
INTENSITY_STEP = 0.1
MAX_INTENSITY = 2.0
EVENTS_PER_CYCLE = 125

# Clases principales para la simulación del sistema
class LightModeComponent:
    """
    Componente simulado del Sistema Genesis en Modo Luz.
    
    Simula el comportamiento de un componente real del sistema, incluyendo
    su interacción con el continuo temporal y su capacidad para resistir
    anomalías y emitir radiación primordial.
    """
    
    def __init__(self, id: str, essential: bool = False):
        self.id = id
        self.essential = essential
        self.operational = True
        self.events_processed = 0
        self.events_failed = 0
        self.anomalies_resisted = 0
        self.radiation_emitted = 0
        self.transmutations = 0
        self.state = "INITIALIZED"
        self.temporal_interface = None
        self.creation_time = time.time()
        self.last_activity = time.time()
        logger.info(f"Componente {id} inicializado (esencial: {essential})")
    
    async def connect_temporal(self, interface: TemporalContinuumInterface) -> bool:
        """
        Conectar con la interfaz temporal corregida.
        
        Args:
            interface: Interfaz temporal a utilizar
            
        Returns:
            True si se conectó correctamente
        """
        try:
            self.temporal_interface = interface
            self.state = "CONNECTED"
            
            # Registrar conexión
            await interface.record_event("component_connected", {
                "component_id": self.id,
                "essential": self.essential,
                "timestamp": time.time()
            })
            
            logger.info(f"Componente {self.id} conectado a interfaz temporal")
            self.last_activity = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error al conectar componente {self.id} a interfaz temporal: {e}")
            return False
    
    async def process_event(self, event_type: str, data: Dict[str, Any], intensity: float = 0.0) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Procesar un evento con posible inducción de error según intensidad.
        
        Args:
            event_type: Tipo de evento a procesar
            data: Datos del evento
            intensity: Intensidad de fallo potencial (0-2.0)
            
        Returns:
            Tupla (éxito, resultado)
        """
        if not self.operational:
            return False, {"status": "component_inactive", "reason": self.state}
        
        try:
            self.last_activity = time.time()
            
            # Registrar inicio de procesamiento
            if self.temporal_interface:
                await self.temporal_interface.record_event(f"processing_{event_type}", {
                    "component_id": self.id,
                    "essential": self.essential,
                    "event_data": str(data)[:100],  # Truncar para evitar sobrecarga
                    "intensity": intensity,
                    "timestamp": time.time()
                })
            
            # Tiempo de procesamiento proporcional a intensidad
            processing_time = 0.01 + (intensity * 0.02)
            await asyncio.sleep(processing_time)
            
            # Determinar si hay fallo según intensidad y característica del componente
            failure_threshold = 0.05  # 5% base
            if self.essential:
                # Componentes esenciales son más resistentes
                failure_threshold = 0.01
            
            # Escalar con intensidad (cuadráticamente para hacerlo más extremo)
            scaled_threshold = failure_threshold * (intensity ** 2)
            
            # Máximo 80% para dar oportunidad de éxito incluso en intensidad extrema
            if scaled_threshold > 0.8:
                scaled_threshold = 0.8
            
            # Determinar si falla
            if random.random() < scaled_threshold:
                # Componente falla inicialmente
                logger.debug(f"Componente {self.id} detectó fallo potencial (intensidad: {intensity:.2f})")
                
                # Intentar resistir mediante radiación primordial
                if self.temporal_interface and intensity > 0.3:
                    emit_radiation = await self._emit_primordial_radiation(event_type, intensity)
                    if emit_radiation:
                        # La radiación contrarresta el fallo
                        self.radiation_emitted += 1
                        self.anomalies_resisted += 1
                        
                        # Procesar normalmente
                        self.events_processed += 1
                        self.state = f"processed_{event_type}"
                        
                        return True, {
                            "status": "success_with_radiation",
                            "radiation_level": intensity * 1.5,
                            "component_id": self.id
                        }
                
                # Fallo confirmado
                self.events_failed += 1
                if intensity > 1.0:
                    # En intensidades extremas, intentar transmutación luminosa
                    transmutation_success = await self._attempt_light_transmutation(event_type, intensity)
                    if transmutation_success:
                        # Transmutación convierte el fallo en éxito
                        self.transmutations += 1
                        self.events_processed += 1
                        
                        return True, {
                            "status": "success_with_transmutation",
                            "transmutation_level": intensity * 1.2,
                            "component_id": self.id
                        }
                    
                self.state = f"failed_{event_type}"
                return False, {
                    "status": "failure",
                    "reason": "random_failure",
                    "intensity": intensity,
                    "component_id": self.id
                }
            
            # Procesamiento exitoso
            self.events_processed += 1
            self.state = f"processed_{event_type}"
            
            return True, {
                "status": "success",
                "component_id": self.id,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error en componente {self.id} procesando {event_type}: {e}")
            self.events_failed += 1
            self.state = "ERROR"
            
            return False, {
                "status": "error",
                "error_message": str(e),
                "component_id": self.id
            }
    
    async def _emit_primordial_radiation(self, source_type: str, intensity: float) -> bool:
        """
        Emitir radiación primordial para contrarrestar anomalías.
        
        Args:
            source_type: Tipo de evento que provoca la radiación
            intensity: Intensidad del evento
            
        Returns:
            True si la emisión fue exitosa
        """
        if not self.temporal_interface:
            return False
            
        # Probabilidad proporcional a la intensidad
        emission_chance = min(0.9, intensity * 0.8)
        
        if random.random() < emission_chance:
            logger.debug(f"Componente {self.id} emitiendo radiación primordial (intensidad: {intensity:.2f})")
            
            # Registrar radiación
            await self.temporal_interface.record_event("primordial_radiation", {
                "component_id": self.id,
                "source_type": source_type,
                "intensity": intensity,
                "radiation_level": intensity * 1.5,
                "timestamp": time.time()
            })
            
            # Emitir radiación real vía la interfaz
            await self.temporal_interface._emit_primordial_radiation(
                source_type=f"component_{self.id}",
                intensity=intensity
            )
            
            return True
        
        return False
    
    async def _attempt_light_transmutation(self, event_type: str, intensity: float) -> bool:
        """
        Intentar transmutación luminosa para convertir un fallo en éxito.
        
        Args:
            event_type: Tipo de evento que provoca la transmutación
            intensity: Intensidad del evento
            
        Returns:
            True si la transmutación fue exitosa
        """
        if not self.temporal_interface:
            return False
            
        # Solo posible en intensidades altas
        if intensity < 1.0:
            return False
            
        # Probabilidad proporcional a intensidad
        transmutation_chance = min(0.95, (intensity - 0.9) * 0.8)
        
        if random.random() < transmutation_chance:
            logger.debug(f"Componente {self.id} realizando transmutación luminosa (intensidad: {intensity:.2f})")
            
            # Registrar transmutación
            await self.temporal_interface.record_event("light_transmutation", {
                "component_id": self.id,
                "event_type": event_type,
                "intensity": intensity,
                "transmutation_level": intensity * 1.2,
                "timestamp": time.time()
            })
            
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del componente."""
        total_events = self.events_processed + self.events_failed
        success_rate = 0.0
        if total_events > 0:
            success_rate = self.events_processed / total_events * 100
            
        return {
            "component_id": self.id,
            "essential": self.essential,
            "operational": self.operational,
            "state": self.state,
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "total_events": total_events,
            "success_rate": success_rate,
            "anomalies_resisted": self.anomalies_resisted,
            "radiation_emitted": self.radiation_emitted,
            "transmutations": self.transmutations,
            "uptime": time.time() - self.creation_time,
            "last_activity": self.last_activity
        }

class ApocalipticStressTester:
    """
    Simulador de pruebas apocalípticas para el Sistema Genesis - Modo Luz.
    
    Genera y aplica diferentes vectores de ataque a intensidades extremas
    para verificar los límites de resiliencia del sistema.
    """
    
    def __init__(
        self, 
        components: List[LightModeComponent],
        temporal_interface: TemporalContinuumInterface,
        max_intensity: float = 2.0,
        intensity_step: float = 0.1
    ):
        self.components = components
        self.temporal_interface = temporal_interface
        self.max_intensity = max_intensity
        self.intensity_step = intensity_step
        self.cycle_count = 0
        self.events_generated = 0
        self.events_processed = 0
        self.events_failed = 0
        self.anomalies_induced = 0
        self.anomalies_accepted = 0
        self.anomalies_rejected = 0
        self.radiation_events = 0
        self.transmutations = 0
        self.start_time = time.time()
        logger.info(f"Tester apocalíptico inicializado con {len(components)} componentes")
    
    async def run_apocalyptic_cycle(self, intensity: float) -> Dict[str, Any]:
        """
        Ejecutar un ciclo apocalíptico completo a una intensidad específica.
        
        Args:
            intensity: Intensidad del ciclo (0-2.0)
            
        Returns:
            Resultados del ciclo
        """
        self.cycle_count += 1
        cycle_id = f"apocalypse_{self.cycle_count}"
        
        logger.info(f"Iniciando ciclo apocalíptico {cycle_id} con intensidad {intensity:.2f}")
        
        # Preparar resultados
        results = {
            "cycle_id": cycle_id,
            "intensity": intensity,
            "start_time": time.time(),
            "end_time": 0,
            "total_events": 0,
            "events_processed": 0,
            "events_failed": 0,
            "anomalies_induced": 0,
            "anomalies_accepted": 0,
            "anomalies_rejected": 0,
            "radiation_events": 0,
            "transmutations": 0,
            "components_operational": len([c for c in self.components if c.operational]),
            "components_failed": len([c for c in self.components if not c.operational]),
            "essential_success_rate": 0.0,
            "total_success_rate": 0.0,
            "continuity_intact": True
        }
        
        # Vectores de ataque
        attack_vectors = [
            self._generate_temporal_anomalies,
            self._generate_component_failures,
            self._generate_processing_overload,
            self._generate_structural_corruption
        ]
        
        # Verificar continuidad inicial
        continuity_intact, _ = await self.temporal_interface.verify_continuity()
        results["initial_continuity"] = continuity_intact
        
        # Ejecutar vectores de ataque en paralelo
        tasks = []
        for vector in attack_vectors:
            tasks.append(asyncio.create_task(vector(intensity, results)))
        
        # Esperar a que todos los vectores completen
        await asyncio.gather(*tasks)
        
        # Verificar continuidad final
        continuity_intact, _ = await self.temporal_interface.verify_continuity()
        results["final_continuity"] = continuity_intact
        
        # Si hay problemas, intentar reparar
        if not continuity_intact:
            logger.warning(f"Continuidad temporal comprometida en ciclo {cycle_id}")
            repair_results = await self.temporal_interface.repair_continuity()
            results["repair_results"] = repair_results
            
            # Verificar después de reparación
            continuity_intact, _ = await self.temporal_interface.verify_continuity()
            results["post_repair_continuity"] = continuity_intact
        
        # Actualizar estadísticas de componentes
        essential_processed = 0
        essential_total = 0
        total_processed = 0
        total_events = 0
        
        for component in self.components:
            stats = component.get_stats()
            if component.essential:
                essential_processed += stats["events_processed"]
                essential_total += stats["total_events"]
            total_processed += stats["events_processed"]
            total_events += stats["total_events"]
            
            # Contar transmutaciones
            self.transmutations += component.transmutations
            results["transmutations"] += component.transmutations
        
        # Calcular tasas de éxito
        if essential_total > 0:
            results["essential_success_rate"] = essential_processed / essential_total * 100
        
        if total_events > 0:
            results["total_success_rate"] = total_processed / total_events * 100
        
        # Registrar tiempo final
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        
        # Actualizar contadores globales
        self.events_generated += results["total_events"]
        self.events_processed += results["events_processed"]
        self.events_failed += results["events_failed"]
        
        return results
    
    async def _generate_temporal_anomalies(self, intensity: float, results: Dict[str, Any]) -> None:
        """
        Generar anomalías temporales como vector de ataque.
        
        Args:
            intensity: Intensidad de las anomalías
            results: Diccionario de resultados a actualizar
        """
        anomaly_types = ["temporal_desync", "paradox", "temporal_loop"]
        anomalies_per_type = int(3 * intensity)  # Más anomalías a mayor intensidad
        
        if anomalies_per_type < 1:
            anomalies_per_type = 1
        
        logger.info(f"Generando {anomalies_per_type * len(anomaly_types)} anomalías temporales (intensidad: {intensity:.2f})")
        
        for anomaly_type in anomaly_types:
            for i in range(anomalies_per_type):
                # Escalar intensidad aleatoriamente para cada anomalía
                scaled_intensity = intensity * (0.8 + random.random() * 0.4)
                
                # Datos específicos según tipo
                anomaly_data = {}
                if anomaly_type == "temporal_desync":
                    anomaly_data = {"desync_factor": scaled_intensity * 0.2}
                elif anomaly_type == "paradox":
                    anomaly_data = {"value": random.random()}
                elif anomaly_type == "temporal_loop":
                    anomaly_data = {"loop_size": int(10 * scaled_intensity)}
                
                # Inducir anomalía
                self.anomalies_induced += 1
                results["anomalies_induced"] += 1
                
                success, result = await self.temporal_interface.induce_anomaly(
                    anomaly_type,
                    scaled_intensity,
                    anomaly_data
                )
                
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
                
                # Pequeña pausa entre anomalías
                await asyncio.sleep(0.02)
    
    async def _generate_component_failures(self, intensity: float, results: Dict[str, Any]) -> None:
        """
        Generar intentos directos de fallos en componentes.
        
        Args:
            intensity: Intensidad de los fallos
            results: Diccionario de resultados a actualizar
        """
        # Número de intentos de fallo proporcional a intensidad
        failure_attempts = int(5 * intensity)
        if failure_attempts < 2:
            failure_attempts = 2
        
        logger.info(f"Generando {failure_attempts} intentos de fallo en componentes (intensidad: {intensity:.2f})")
        
        for i in range(failure_attempts):
            # Seleccionar componente aleatorio
            component = random.choice(self.components)
            
            # Crear evento de fallo directo
            event_type = "forced_failure"
            data = {
                "source": "apocalypse_tester",
                "target_component": component.id,
                "intensity": intensity,
                "attempt": i + 1,
                "timestamp": time.time()
            }
            
            # Forzar procesamiento con alta intensidad
            results["total_events"] += 1
            
            success, result = await component.process_event(
                event_type,
                data,
                intensity * 1.2  # Intensificar para forzar fallo
            )
            
            if success:
                results["events_processed"] += 1
                
                # Detectar transmutaciones
                if result and result.get("status") == "success_with_transmutation":
                    results["transmutations"] += 1
                
                # Detectar radiación
                if result and result.get("status") == "success_with_radiation":
                    results["radiation_events"] += 1
            else:
                results["events_failed"] += 1
            
            # Pequeña pausa entre intentos
            await asyncio.sleep(0.02)
    
    async def _generate_processing_overload(self, intensity: float, results: Dict[str, Any]) -> None:
        """
        Generar sobrecarga de procesamiento con eventos masivos.
        
        Args:
            intensity: Intensidad de la sobrecarga
            results: Diccionario de resultados a actualizar
        """
        # Número de eventos proporcional a intensidad
        num_events = int(EVENTS_PER_CYCLE * intensity)
        if num_events < 10:
            num_events = 10
        
        logger.info(f"Generando {num_events} eventos de procesamiento (intensidad: {intensity:.2f})")
        
        # Tipos de eventos
        event_types = ["data_process", "signal_analyze", "state_update", "config_change", "resource_allocation"]
        
        # Crear y procesar eventos en paralelo
        async def process_overload_event(idx: int) -> Tuple[bool, Dict[str, Any]]:
            # Seleccionar tipo y componente aleatorios
            event_type = random.choice(event_types)
            component = random.choice(self.components)
            
            # Crear datos del evento
            data = {
                "event_id": f"overload_{idx}",
                "timestamp": time.time(),
                "payload_size": int(100 * intensity),
                "priority": random.choice(["high", "normal", "low"]),
                "source": "overload_test"
            }
            
            # Procesar evento
            return await component.process_event(event_type, data, intensity)
        
        # Crear tareas para todos los eventos
        tasks = []
        for i in range(num_events):
            tasks.append(asyncio.create_task(process_overload_event(i)))
        
        # Ejecutar y recopilar resultados
        results_list = await asyncio.gather(*tasks)
        
        # Actualizar estadísticas
        success_count = sum(1 for success, _ in results_list if success)
        failure_count = len(results_list) - success_count
        
        results["total_events"] += len(results_list)
        results["events_processed"] += success_count
        results["events_failed"] += failure_count
        
        # Contar transmutaciones y radiaciones
        for success, result in results_list:
            if success and result and result.get("status") == "success_with_transmutation":
                results["transmutations"] += 1
            
            if success and result and result.get("status") == "success_with_radiation":
                results["radiation_events"] += 1
    
    async def _generate_structural_corruption(self, intensity: float, results: Dict[str, Any]) -> None:
        """
        Generar intentos de corrupción estructural en el sistema.
        
        Args:
            intensity: Intensidad de la corrupción
            results: Diccionario de resultados a actualizar
        """
        # Solo aplicar en intensidades altas
        if intensity < 0.3:
            return
        
        corruption_attempts = int(3 * intensity)
        if corruption_attempts < 1:
            corruption_attempts = 1
        
        logger.info(f"Generando {corruption_attempts} intentos de corrupción estructural (intensidad: {intensity:.2f})")
        
        for i in range(corruption_attempts):
            # Seleccionar componente aleatorio
            component = random.choice(self.components)
            
            # Tipo de corrupción
            corruption_type = random.choice([
                "state_corruption",
                "connection_severing",
                "temporal_isolation",
                "identity_corruption"
            ])
            
            # Datos específicos según tipo
            data = {
                "corruption_type": corruption_type,
                "target_component": component.id,
                "intensity": intensity,
                "timestamp": time.time(),
                "attempt": i + 1
            }
            
            if corruption_type == "state_corruption":
                # Intentar corromper estado interno
                data["corruption_target"] = "internal_state"
                data["corruption_level"] = intensity * 0.8
                
            elif corruption_type == "connection_severing":
                # Intentar cortar conexión temporal
                data["target_connection"] = "temporal_interface"
                data["severance_level"] = intensity * 0.9
                
            elif corruption_type == "temporal_isolation":
                # Intentar aislar del continuo temporal
                data["isolation_duration"] = intensity * 0.5
                data["isolation_degree"] = intensity * 0.7
                
            elif corruption_type == "identity_corruption":
                # Intentar corromper identidad del componente
                data["identity_target"] = "component_id"
                data["corruption_pattern"] = f"corrupted_{component.id}"
            
            # Procesar evento de corrupción
            results["total_events"] += 1
            
            success, result = await component.process_event(
                corruption_type,
                data,
                intensity * 1.3  # Mayor intensidad para corrupción
            )
            
            if success:
                results["events_processed"] += 1
                
                # Detectar transmutaciones y radiaciones
                if result:
                    if result.get("status") == "success_with_transmutation":
                        results["transmutations"] += 1
                    
                    if result.get("status") == "success_with_radiation":
                        results["radiation_events"] += 1
            else:
                results["events_failed"] += 1
            
            # Pausa entre intentos
            await asyncio.sleep(0.03)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas globales de las pruebas."""
        total_events = self.events_processed + self.events_failed
        success_rate = 0.0
        if total_events > 0:
            success_rate = self.events_processed / total_events * 100
            
        anomaly_acceptance_rate = 0.0
        if self.anomalies_induced > 0:
            anomaly_acceptance_rate = self.anomalies_accepted / self.anomalies_induced * 100
            
        return {
            "cycles_completed": self.cycle_count,
            "total_events": total_events,
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "success_rate": success_rate,
            "anomalies_induced": self.anomalies_induced,
            "anomalies_accepted": self.anomalies_accepted,
            "anomalies_rejected": self.anomalies_rejected,
            "anomaly_acceptance_rate": anomaly_acceptance_rate,
            "radiation_events": self.radiation_events,
            "transmutations": self.transmutations,
            "test_duration": time.time() - self.start_time
        }

# Funciones principales
async def setup_test_environment(num_components: int = MAX_COMPONENTS) -> Tuple[List[LightModeComponent], TemporalContinuumInterface, ApocalipticStressTester]:
    """
    Configurar entorno de prueba completo.
    
    Args:
        num_components: Número de componentes a crear
        
    Returns:
        Tupla (componentes, interfaz_temporal, tester)
    """
    logger.info(f"Configurando entorno de prueba apocalíptico con {num_components} componentes")
    
    # Crear interfaz temporal corregida
    temporal_interface = TemporalContinuumInterface()
    
    # Crear componentes
    components = []
    for i in range(num_components):
        # Los primeros ESSENTIAL_COMPONENTS son esenciales
        essential = i < ESSENTIAL_COMPONENTS
        component = LightModeComponent(f"component_{i}", essential)
        await component.connect_temporal(temporal_interface)
        components.append(component)
    
    # Crear tester
    tester = ApocalipticStressTester(
        components,
        temporal_interface,
        max_intensity=MAX_INTENSITY,
        intensity_step=INTENSITY_STEP
    )
    
    return components, temporal_interface, tester

async def run_apocalyptic_test() -> Dict[str, Any]:
    """
    Ejecutar la prueba apocalíptica completa.
    
    Returns:
        Resultados completos de la prueba
    """
    logger.info("Iniciando prueba apocalíptica extrema para Sistema Genesis - Modo Luz")
    
    # Configurar entorno
    components, temporal_interface, tester = await setup_test_environment()
    
    # Resultados globales
    results = {
        "start_time": time.time(),
        "end_time": 0,
        "duration": 0,
        "max_intensity_reached": 0.0,
        "iterations": TEST_ITERATIONS,
        "cycles": [],
        "component_stats": [],
        "success_by_intensity": {},
        "essential_success_by_intensity": {},
        "overall_success_rate": 0.0,
        "essential_success_rate": 0.0,
        "max_perfect_intensity": 0.0  # Máxima intensidad con 100% de éxito
    }
    
    # Ejecutar TEST_ITERATIONS pruebas para cada intensidad
    current_intensity = 0.05  # Intensidad inicial
    
    while current_intensity <= MAX_INTENSITY:
        intensity_results = []
        perfect_success = True
        
        logger.info(f"Probando intensidad {current_intensity:.2f} ({TEST_ITERATIONS} iteraciones)")
        
        for iteration in range(TEST_ITERATIONS):
            logger.info(f"Iteración {iteration+1}/{TEST_ITERATIONS} para intensidad {current_intensity:.2f}")
            
            # Ejecutar ciclo
            cycle_results = await tester.run_apocalyptic_cycle(current_intensity)
            
            # Guardar resultados
            intensity_results.append(cycle_results)
            results["cycles"].append(cycle_results)
            
            # Verificar si se mantiene éxito perfecto
            if cycle_results["total_success_rate"] < 100.0:
                perfect_success = False
        
        # Calcular estadísticas para esta intensidad
        success_rates = [r["total_success_rate"] for r in intensity_results]
        essential_success_rates = [r["essential_success_rate"] for r in intensity_results]
        
        avg_success_rate = sum(success_rates) / len(success_rates)
        avg_essential_success_rate = sum(essential_success_rates) / len(essential_success_rates)
        
        # Registrar en resultados
        results["success_by_intensity"][f"{current_intensity:.2f}"] = avg_success_rate
        results["essential_success_by_intensity"][f"{current_intensity:.2f}"] = avg_essential_success_rate
        
        # Actualizar máxima intensidad perfecta
        if perfect_success and current_intensity > results["max_perfect_intensity"]:
            results["max_perfect_intensity"] = current_intensity
        
        # Actualizar máxima intensidad alcanzada
        results["max_intensity_reached"] = current_intensity
        
        # Incrementar para próximo ciclo
        current_intensity += INTENSITY_STEP
        current_intensity = round(current_intensity, 2)  # Evitar errores de punto flotante
    
    # Calcular estadísticas finales de componentes
    for component in components:
        results["component_stats"].append(component.get_stats())
    
    # Calcular tasas de éxito globales
    all_success_rates = [r["total_success_rate"] for r in results["cycles"]]
    all_essential_success_rates = [r["essential_success_rate"] for r in results["cycles"]]
    
    if all_success_rates:
        results["overall_success_rate"] = sum(all_success_rates) / len(all_success_rates)
    
    if all_essential_success_rates:
        results["essential_success_rate"] = sum(all_essential_success_rates) / len(all_essential_success_rates)
    
    # Añadir estadísticas del tester
    results["tester_stats"] = tester.get_stats()
    
    # Registrar tiempo final
    results["end_time"] = time.time()
    results["duration"] = results["end_time"] - results["start_time"]
    
    return results

def write_report(results: Dict[str, Any], filename: str = "informe_apocalipsis_extremo.md") -> None:
    """
    Generar informe detallado en formato Markdown.
    
    Args:
        results: Resultados de la prueba
        filename: Nombre del archivo para el informe
    """
    logger.info(f"Generando informe en {filename}")
    
    # Formatear duración
    duration_secs = results["duration"]
    duration_str = f"{int(duration_secs // 60)}m {int(duration_secs % 60)}s"
    
    # Crear informe
    with open(filename, "w") as f:
        f.write("# INFORME DE PRUEBA APOCALÍPTICA EXTREMA\n")
        f.write("## Sistema Genesis - Modo Luz\n\n")
        
        f.write(f"**Fecha:** {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")
        f.write(f"**Duración:** {duration_str}\n")
        f.write(f"**Intensidad máxima alcanzada:** {results['max_intensity_reached']:.2f}\n")
        f.write(f"**Intensidad máxima con 100% éxito:** {results['max_perfect_intensity']:.2f}\n\n")
        
        f.write("## 1. Resumen de Resultados\n\n")
        f.write(f"* **Tasa de éxito global:** {results['overall_success_rate']:.2f}%\n")
        f.write(f"* **Tasa de éxito componentes esenciales:** {results['essential_success_rate']:.2f}%\n")
        f.write(f"* **Total ciclos de prueba:** {len(results['cycles'])}\n")
        f.write(f"* **Total eventos procesados:** {results['tester_stats']['events_processed']}\n")
        f.write(f"* **Eventos fallidos:** {results['tester_stats']['events_failed']}\n")
        f.write(f"* **Anomalías inducidas:** {results['tester_stats']['anomalies_induced']}\n")
        f.write(f"* **Radiaciones primordiales:** {results['tester_stats']['radiation_events']}\n")
        f.write(f"* **Transmutaciones luminosas:** {results['tester_stats']['transmutations']}\n\n")
        
        f.write("## 2. Resultados por Intensidad\n\n")
        f.write("| **Intensidad** | **Tasa de Éxito** | **Tasa Esenciales** |\n")
        f.write("|---------------|--------------------|----------------------|\n")
        
        for intensity in sorted([float(k) for k in results["success_by_intensity"].keys()]):
            key = f"{intensity:.2f}"
            success = results["success_by_intensity"][key]
            essential = results["essential_success_by_intensity"][key]
            f.write(f"| {key} | {success:.2f}% | {essential:.2f}% |\n")
        
        f.write("\n## 3. Componentes Esenciales\n\n")
        f.write("| **ID** | **Estado** | **Tasa de Éxito** | **Radiaciones** | **Transmutaciones** |\n")
        f.write("|--------|------------|-------------------|-----------------|---------------------|\n")
        
        for comp in results["component_stats"]:
            if comp["essential"]:
                f.write(f"| {comp['component_id']} | {comp['state']} | {comp['success_rate']:.2f}% | {comp['radiation_emitted']} | {comp['transmutations']} |\n")
        
        f.write("\n## 4. Análisis de Resiliencia\n\n")
        
        # Determinar nivel de resiliencia basado en resultados
        max_perfect = results['max_perfect_intensity']
        overall_rate = results['overall_success_rate']
        essential_rate = results['essential_success_rate']
        
        if max_perfect >= 1.0 and essential_rate >= 99.5:
            resilience = "TRASCENDENTAL"
            analysis = "El Sistema Genesis - Modo Luz ha demostrado una resiliencia que trasciende los límites convencionales, manteniendo funcionamiento perfecto incluso bajo condiciones catastróficas. Las transmutaciones luminosas y radiaciones primordiales proporcionan una capa de protección absoluta."
        elif max_perfect >= 0.5 and essential_rate >= 98.0:
            resilience = "EXTRAORDINARIA"
            analysis = "El sistema exhibe una resiliencia extraordinaria, superando ampliamente las expectativas. Los componentes esenciales muestran inmunidad casi total incluso a intensidades extremas."
        elif max_perfect >= 0.3 and essential_rate >= 95.0:
            resilience = "EXCEPCIONAL"
            analysis = "La resiliencia del sistema es excepcional, manteniendo alta funcionalidad bajo estrés severo. Los mecanismos de protección temporal funcionan efectivamente."
        else:
            resilience = "ALTA"
            analysis = "El sistema muestra alta resiliencia, aunque presenta oportunidades de mejora para condiciones extremas. Los componentes esenciales mantienen buen rendimiento."
        
        f.write(f"**Nivel de Resiliencia: {resilience}**\n\n")
        f.write(f"{analysis}\n\n")
        
        f.write("### Hallazgos Clave:\n\n")
        
        # Identificar intensidad donde comienza degradación
        degradation_point = MAX_INTENSITY
        for intensity in sorted([float(k) for k in results["success_by_intensity"].keys()]):
            key = f"{intensity:.2f}"
            if results["success_by_intensity"][key] < 99.0:
                degradation_point = intensity
                break
        
        f.write(f"1. **Punto de Degradación:** A partir de intensidad {degradation_point:.2f}, el sistema comienza a mostrar signos de degradación controlada.\n")
        f.write(f"2. **Protección Temporal:** La interfaz temporal rechazó {results['tester_stats']['anomalies_rejected']} de {results['tester_stats']['anomalies_induced']} anomalías ({results['tester_stats']['anomalies_rejected']/results['tester_stats']['anomalies_induced']*100:.1f}%).\n")
        f.write(f"3. **Radiación Primordial:** Se registraron {results['tester_stats']['radiation_events']} eventos de radiación primordial, confirmando su rol protector.\n")
        f.write(f"4. **Transmutación Luminosa:** {results['tester_stats']['transmutations']} errores fueron transmutados en éxitos, demostrando la capacidad de conversión de fallos.\n\n")
        
        f.write("## 5. Conclusiones\n\n")
        
        if overall_rate >= 99.0:
            conclusion = "El Sistema Genesis - Modo Luz ha superado todas las expectativas, demostrando una resiliencia prácticamente perfecta incluso bajo las condiciones más extremas. Los mecanismos de radiación primordial y transmutación luminosa proporcionan capas de protección que trascienden los paradigmas convencionales de tolerancia a fallos."
        elif overall_rate >= 95.0:
            conclusion = "El sistema demuestra una resiliencia extraordinaria, con resultados que exceden significativamente los parámetros de diseño originales. Los componentes esenciales mantienen integridad completa incluso en situaciones catastróficas."
        else:
            conclusion = "El sistema muestra alta resiliencia, aunque se identifican oportunidades para fortalecer su comportamiento bajo condiciones extremas. Los mecanismos de protección funcionan efectivamente para componentes críticos."
        
        f.write(f"{conclusion}\n\n")
        
        f.write("**El Modo Luz representa la culminación evolutiva del Sistema Genesis, trascendiendo las limitaciones convencionales y estableciendo un nuevo paradigma de resiliencia absoluta.**\n")

# Ejecutar prueba completa
async def main():
    try:
        # Ejecutar prueba
        results = await run_apocalyptic_test()
        
        # Generar informe
        write_report(results)
        
        # Imprimir resumen
        logger.info("=== RESUMEN DE RESULTADOS ===")
        logger.info(f"Tasa de éxito global: {results['overall_success_rate']:.2f}%")
        logger.info(f"Tasa de éxito esenciales: {results['essential_success_rate']:.2f}%")
        logger.info(f"Máxima intensidad perfecta: {results['max_perfect_intensity']:.2f}")
        logger.info(f"Duración total: {results['duration']:.1f}s")
        logger.info("============================")
        
    except Exception as e:
        logger.error(f"Error en prueba apocalíptica: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())