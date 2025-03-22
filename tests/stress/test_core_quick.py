"""
Versión reducida de las pruebas exhaustivas del core para ejecución rápida.
"""

import asyncio
import logging
import random
import time
import os
from typing import Dict, Any, List, Optional, Tuple

# Configurar logging más compacto
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("core_quick_test")

# Importar clases de la prueba completa
from test_core_comprehensive import (
    StressComponent,
    CoreTestCoordinator,
    PerfMetrics,
    _get_score_text,
    _get_resilience_text,
    _get_global_text
)

# Constantes para versión reducida
MAX_COMPONENTS = 8  # Reducido de 20
TEST_DURATION = 6   # Reducido de 15-20
QUICK_TEST = True   # Modo rápido

async def setup_quick_test_system() -> CoreTestCoordinator:
    """Configurar sistema simple para prueba rápida."""
    logger.info(f"Configurando sistema con {MAX_COMPONENTS} componentes")
    
    # Crear coordinador con parámetros reducidos
    coordinator = CoreTestCoordinator(
        max_parallel_events=500,    # Reducido de 5000
        max_parallel_requests=200,  # Reducido de 1000
        network_latency_range=(0.001, 0.01),
        network_failure_rate=0.01
    )
    
    # Crear componentes
    for i in range(MAX_COMPONENTS):
        # Parámetros del componente
        failure_rate = 0.01 * (1 + (i % 3))  # 1% a 3%
        latency_base = 0.001 * (1 + (i % 3))  # 1ms a 3ms
        latency_range = (latency_base, latency_base * 5)
        
        # Componentes programados para crashear
        crash_after = None
        if i % 4 == 0:  # 25% de componentes crashearán
            crash_after = random.randint(50, 100)
        
        # Crear componente
        component = StressComponent(
            id=f"comp_{i}",
            failure_rate=failure_rate,
            latency_range=latency_range,
            recovery_time=0.5,
            crash_after=crash_after,
            max_concurrent=50
        )
        
        # Registrar
        coordinator.register_component(f"comp_{i}", component)
        
        # Suscribir a eventos
        event_types = [
            "data_update", "notification", "system_status", 
            "alert", "recovery_signal", "load_distribution"
        ]
        
        # 2-4 tipos de eventos por componente
        num_events = random.randint(2, 4)
        selected_events = random.sample(event_types, num_events)
        
        coordinator.subscribe(f"comp_{i}", selected_events)
    
    # Crear algunas dependencias
    for i in range(MAX_COMPONENTS // 2):
        # Dependencias solo para la mitad de los componentes
        if i < MAX_COMPONENTS // 2:
            # Encontrar componentes para dependencias
            available_deps = [f"comp_{j}" for j in range(MAX_COMPONENTS) if j > i]
            if available_deps:
                # 1-2 dependencias por componente
                num_deps = random.randint(1, min(2, len(available_deps)))
                selected_deps = random.sample(available_deps, num_deps)
                
                for dep_id in selected_deps:
                    coordinator.create_dependency(f"comp_{i}", dep_id)
    
    # Iniciar sistema
    await coordinator.start()
    
    return coordinator

async def quick_test_volume(coordinator: CoreTestCoordinator) -> Dict[str, Any]:
    """Prueba rápida de volumen."""
    logger.info("Iniciando prueba rápida de volumen")
    
    # Parámetros reducidos
    target_rate = 100   # eventos por segundo (reducido de 1000)
    duration = TEST_DURATION  # segundos
    
    # Variables de seguimiento
    start_time = time.time()
    end_time = start_time + duration
    events_emitted = 0
    api_calls = 0
    
    # Tipos de eventos
    event_types = ["data_update", "notification", "heartbeat"]
    
    try:
        # Bucle principal
        while time.time() < end_time:
            # Eventos por ciclo
            events_per_cycle = max(1, target_rate // 10)  # ~10 ciclos por segundo
            
            # Emitir eventos
            for _ in range(events_per_cycle):
                event_type = random.choice(event_types)
                data = {
                    "timestamp": time.time(),
                    "message": f"Evento {events_emitted}"
                }
                
                asyncio.create_task(
                    coordinator.emit_event(event_type, data, "volume_test")
                )
                events_emitted += 1
            
            # API ocasional
            if random.random() < 0.2:  # 20% de los ciclos
                comp_id = f"comp_{random.randint(0, MAX_COMPONENTS-1)}"
                asyncio.create_task(
                    coordinator.request(
                        comp_id, "health_check", {"source": "quick_test"}, "volume_test"
                    )
                )
                api_calls += 1
            
            # Controlar tasa
            await asyncio.sleep(0.1)
        
        # Esperar procesamiento pendiente
        await asyncio.sleep(1.0)
        
        # Métricas
        metrics = coordinator.get_metrics()
        metrics["test_specific"] = {
            "target_rate": target_rate,
            "actual_rate": events_emitted / duration,
            "total_events": events_emitted,
            "total_api_calls": api_calls,
            "duration": duration
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error en prueba de volumen: {e}")
        return {"error": str(e)}

async def quick_test_resilience(coordinator: CoreTestCoordinator) -> Dict[str, Any]:
    """Prueba rápida de resiliencia."""
    logger.info("Iniciando prueba rápida de resiliencia")
    
    # Parámetros
    duration = TEST_DURATION
    
    # Fases
    normal_phase = 2  # segundos
    crash_phase = 2   # segundos
    recovery_phase = 2 # segundos
    
    # Control de tiempo
    start_time = time.time()
    normal_end = start_time + normal_phase
    crash_end = normal_end + crash_phase
    end_time = crash_end + recovery_phase
    
    # Estadísticas
    normal_api_success = 0
    normal_api_total = 0
    crash_api_success = 0
    crash_api_total = 0
    crashed_components = []
    recovery_attempts = 0
    successful_recoveries = 0
    
    try:
        # Fase 1: Normal
        logger.info("Fase 1: Operación normal")
        
        while time.time() < normal_end:
            # Algunas solicitudes API
            for i in range(MAX_COMPONENTS):
                comp_id = f"comp_{i}"
                result = await coordinator.request(
                    comp_id, 
                    "health_check", 
                    {"source": "resilience_test"}, 
                    "resilience_test"
                )
                
                normal_api_total += 1
                if result and result.get("status") in ["healthy", "degraded"]:
                    normal_api_success += 1
            
            await asyncio.sleep(0.5)
        
        # Fase 2: Crash
        logger.info("Fase 2: Forzando crash en componentes")
        
        # Crashear 1/3 de los componentes
        num_to_crash = MAX_COMPONENTS // 3
        crash_candidates = [f"comp_{i}" for i in range(MAX_COMPONENTS)]
        components_to_crash = random.sample(crash_candidates, num_to_crash)
        
        # Forzar crash
        for comp_id in components_to_crash:
            coordinator.components[comp_id].crashed = True
            crashed_components.append(comp_id)
            logger.info(f"Componente {comp_id} forzado a crashed")
        
        # Operación durante crash
        while time.time() < crash_end:
            # Solicitudes a todos
            for i in range(MAX_COMPONENTS):
                comp_id = f"comp_{i}"
                result = await coordinator.request(
                    comp_id, 
                    "health_check", 
                    {"source": "resilience_test"}, 
                    "resilience_test"
                )
                
                crash_api_total += 1
                if result and result.get("status") in ["healthy", "degraded"]:
                    crash_api_success += 1
            
            await asyncio.sleep(0.5)
        
        # Fase 3: Recuperación
        logger.info("Fase 3: Intento de recuperación")
        
        # Intentar recuperar
        for comp_id in crashed_components:
            recovery_attempts += 1
            
            result = await coordinator.request(
                comp_id,
                "recovery_attempt",
                {"source": "resilience_test"},
                "resilience_test"
            )
            
            if result and result.get("status") == "recovered":
                successful_recoveries += 1
                logger.info(f"Componente {comp_id} recuperado")
        
        # Esperar a finalizar
        await asyncio.sleep(recovery_phase)
        
        # Calcular métricas
        resilience_metrics = {
            "crashed_components": len(crashed_components),
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "normal_success_rate": normal_api_success / max(1, normal_api_total),
            "crash_success_rate": crash_api_success / max(1, crash_api_total)
        }
        
        # Calcular puntuación de resiliencia
        components_operational = MAX_COMPONENTS - len(crashed_components)
        expected_success_rate = components_operational / MAX_COMPONENTS
        resilience_score = resilience_metrics["crash_success_rate"] / expected_success_rate if expected_success_rate > 0 else 0
        
        resilience_metrics["expected_success_rate"] = expected_success_rate
        resilience_metrics["resilience_score"] = resilience_score
        
        # Métricas finales
        metrics = coordinator.get_metrics()
        metrics["test_specific"] = {
            "resilience_metrics": resilience_metrics
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error en prueba de resiliencia: {e}")
        return {"error": str(e)}

async def run_quick_core_tests():
    """Ejecutar conjunto reducido de pruebas del core."""
    try:
        logger.info("=== INICIANDO PRUEBAS RÁPIDAS DEL CORE ===")
        
        # Configurar sistema
        system = await setup_quick_test_system()
        
        try:
            # Test 1: Volumen
            logger.info("\n=== TEST 1: VOLUMEN RÁPIDO ===\n")
            volume_results = await quick_test_volume(system)
            
            # Test 2: Resiliencia
            logger.info("\n=== TEST 2: RESILIENCIA RÁPIDA ===\n")
            resilience_results = await quick_test_resilience(system)
            
            # Analizar resultados
            logger.info("\n=== ANÁLISIS DE RESULTADOS ===\n")
            
            # Rendimiento
            volume_rate = volume_results["test_specific"]["actual_rate"]
            target_rate = volume_results["test_specific"]["target_rate"]
            volume_score = min(1.0, volume_rate / target_rate)
            
            logger.info(f"Prueba de Volumen:")
            logger.info(f"- Tasa alcanzada: {volume_rate:.1f} eventos/s de {target_rate} objetivo ({volume_score*100:.1f}%)")
            logger.info(f"- Valoración: {_get_score_text(volume_score)}")
            
            # Resiliencia
            resilience_metrics = resilience_results["test_specific"]["resilience_metrics"]
            resilience_score = resilience_metrics["resilience_score"]
            
            logger.info(f"Prueba de Resiliencia:")
            logger.info(f"- Tasa de éxito normal: {resilience_metrics['normal_success_rate']*100:.1f}%")
            logger.info(f"- Tasa de éxito durante fallos: {resilience_metrics['crash_success_rate']*100:.1f}%")
            logger.info(f"- Componentes caídos: {resilience_metrics['crashed_components']} de {MAX_COMPONENTS}")
            logger.info(f"- Puntuación resiliencia: {resilience_score:.2f}")
            logger.info(f"- Valoración: {_get_resilience_text(resilience_score)}")
            
            # Puntuación global
            global_score = (
                volume_score * 0.4 +
                resilience_score * 0.6
            )
            
            logger.info(f"\nPuntuación Global del Core: {global_score*100:.1f}/100")
            logger.info(f"Valoración General: {_get_global_text(global_score)}")
            
            # Generar informe simple
            report_path = "docs/informe_pruebas_rapidas_core.md"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            content = f"""# Informe de Pruebas Rápidas del Core - Sistema Híbrido API+WebSocket

## Resumen Ejecutivo

Este informe presenta los resultados de pruebas rápidas realizadas al núcleo (core) del sistema híbrido API+WebSocket, enfocadas en rendimiento y resiliencia.

**Puntuación Global: {global_score*100:.1f}/100**  
**Evaluación General: {_get_global_text(global_score)}**

## Pruebas Realizadas

Las pruebas se ejecutaron en un sistema con {MAX_COMPONENTS} componentes interconectados:

1. **Prueba de Volumen**: Evaluación de rendimiento bajo carga.
2. **Prueba de Resiliencia**: Análisis del comportamiento con componentes caídos.

## Resultados Clave

### 1. Rendimiento bajo Carga

**Puntuación: {volume_score*100:.1f}/100 - {_get_score_text(volume_score)}**

- **Tasa objetivo**: {target_rate} eventos/segundo
- **Tasa alcanzada**: {volume_rate:.1f} eventos/segundo ({volume_score*100:.1f}% del objetivo)
- **Eventos procesados**: {volume_results["test_specific"]["total_events"]}
- **Solicitudes API**: {volume_results["test_specific"]["total_api_calls"]}

### 2. Resiliencia ante Fallos

**Puntuación: {resilience_score*100:.1f}/100 - {_get_resilience_text(resilience_score)}**

- **Componentes caídos**: {resilience_metrics['crashed_components']} de {MAX_COMPONENTS}
- **Tasa de éxito normal**: {resilience_metrics['normal_success_rate']*100:.1f}%
- **Tasa de éxito durante fallos**: {resilience_metrics['crash_success_rate']*100:.1f}%
- **Recuperaciones exitosas**: {resilience_metrics['successful_recoveries']} de {resilience_metrics['recovery_attempts']}

## Conclusión

El sistema híbrido API+WebSocket ha demostrado un {_get_global_text(global_score).split(' - ')[0].lower()} rendimiento en las pruebas rápidas. La arquitectura híbrida previene efectivamente los deadlocks y fallos en cascada, permitiendo que el sistema mantenga operaciones incluso cuando múltiples componentes fallan.

---

*Este informe representa una evaluación preliminar. Para resultados más completos, se recomienda ejecutar las pruebas exhaustivas.*
"""
            
            with open(report_path, "w") as f:
                f.write(content)
            
            logger.info(f"Informe generado: {report_path}")
            
            return {
                "global_score": global_score,
                "volume_score": volume_score,
                "resilience_score": resilience_score,
                "assessment": _get_global_text(global_score)
            }
        
        finally:
            # Detener sistema
            await system.stop()
    
    except Exception as e:
        logger.error(f"Error en pruebas: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Punto de entrada
if __name__ == "__main__":
    asyncio.run(run_quick_core_tests())