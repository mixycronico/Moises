"""
Versión reducida de pruebas de estrés extremo para el sistema híbrido API+WebSocket.
"""

import asyncio
import logging
import random
import time
import json
import os
import sys
from typing import Dict, Any, List, Optional, Set, Tuple, Union

# Importar clases de test_extreme_load.py
from test_extreme_load import (
    HighStressComponent,
    ExtremeCaseCoordinator,
    setup_extreme_test_system,
    analyze_and_report_results,
    generate_markdown_report
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quick_extreme_test")

# Versiones modificadas de las funciones de prueba con parámetros reducidos
async def run_quick_volume_test(coordinator: ExtremeCaseCoordinator) -> Dict[str, Any]:
    """Versión reducida de la prueba de volumen."""
    logger.info("Iniciando prueba rápida de volumen")
    
    # Tipos de eventos para la prueba
    event_types = [
        "data_update", "notification", "status_change", 
        "heartbeat", "metrics_report"
    ]
    
    # Parámetros reducidos
    max_rate = 200  # eventos por segundo
    duration = 10   # segundos
    ramp_up = 2     # segundos
    
    # Control de tiempo
    start_time = time.time()
    end_time = start_time + duration
    events_emitted = 0
    api_requests_sent = 0
    
    try:
        # Bucle principal de emisión
        while time.time() < end_time:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Calcular tasa actual según fase
            if elapsed < ramp_up:
                current_rate = max(1, int((elapsed / ramp_up) * max_rate))
            else:
                current_rate = max_rate
            
            # Determinar eventos a emitir (10% de la tasa)
            effective_count = max(1, int(current_rate / 10))
            
            # Generar y emitir eventos
            emission_tasks = []
            for _ in range(effective_count):
                event_type = random.choice(event_types)
                data = {
                    "timestamp": time.time(),
                    "message": f"Evento {event_type} #{events_emitted}",
                    "priority": random.randint(1, 5)
                }
                
                emission_tasks.append(
                    coordinator.emit_event(event_type, data, "quick_test")
                )
                events_emitted += 1
            
            # Iniciar eventos asíncronamente
            for task in emission_tasks:
                asyncio.create_task(task)
            
            # Incluir algunas solicitudes API
            if random.random() < 0.2:  # ~20% de probabilidad
                target_comp = f"comp_{random.randint(0, len(coordinator.components)-1)}"
                request_type = random.choice(["query", "health"])
                req_data = {"test": True}
                
                asyncio.create_task(coordinator.request(
                    target_comp,
                    request_type,
                    req_data,
                    "quick_test"
                ))
                api_requests_sent += 1
            
            # Controlar tasa
            await asyncio.sleep(0.1)  # ~10 ciclos por segundo
        
        # Esperar procesamiento pendiente
        logger.info(f"Prueba completada. Esperando eventos pendientes...")
        await asyncio.sleep(1.0)
        
        # Métricas finales
        metrics = coordinator.get_performance_metrics()
        metrics["test_specific"] = {
            "target_max_rate": max_rate,
            "actual_rate": events_emitted / duration,
            "total_events_emitted": events_emitted,
            "api_requests_sent": api_requests_sent,
            "test_duration": duration,
            "ramp_up_seconds": ramp_up
        }
        
        logger.info(f"Eventos emitidos: {events_emitted}, API: {api_requests_sent}")
        logger.info(f"Tasa efectiva: {events_emitted / duration:.1f} eventos/s")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error durante prueba de volumen: {e}")
        return {
            "error": str(e),
            "events_emitted": events_emitted,
            "api_requests_sent": api_requests_sent
        }

async def run_quick_failure_test(coordinator: ExtremeCaseCoordinator) -> Dict[str, Any]:
    """Versión reducida de la prueba de fallos."""
    logger.info("Iniciando prueba rápida de fallos simultáneos")
    
    # Parámetros reducidos
    event_rate = 50     # eventos por segundo
    duration = 10       # segundos
    failures_to_inject = 3  # número de fallos a inyectar
    
    start_time = time.time()
    end_time = start_time + duration
    events_emitted = 0
    failures_injected = 0
    
    # Tipos de eventos
    event_types = ["data_update", "notification", "heartbeat"]
    
    # Inyectar algunos fallos al inicio
    components_to_fail = random.sample(list(coordinator.components.keys()), 
                                     min(failures_to_inject, len(coordinator.components)))
    
    logger.info(f"Inyectando fallos en {len(components_to_fail)} componentes: {components_to_fail}")
    
    for i, comp_id in enumerate(components_to_fail):
        component = coordinator.components[comp_id]
        
        # Diferentes tipos de fallos para cada componente
        if i % 3 == 0:
            # Crash
            component.crashed = True
            logger.info(f"Componente {comp_id} forzado a estado crashed")
        elif i % 3 == 1:
            # Aumentar latencia
            old_latency = component.latency_range
            component.latency_range = (old_latency[0] * 2, old_latency[1] * 3)
            logger.info(f"Latencia de {comp_id} aumentada: {old_latency} → {component.latency_range}")
        else:
            # Aumentar tasa de fallos
            old_rate = component.failure_probability
            component.failure_probability = min(0.5, old_rate * 3)
            logger.info(f"Tasa de fallos de {comp_id} aumentada: {old_rate:.2f} → {component.failure_probability:.2f}")
        
        failures_injected += 1
    
    try:
        # Generar carga constante
        while time.time() < end_time:
            # Emitir eventos a tasa constante
            events_this_cycle = max(1, event_rate // 10)
            emission_tasks = []
            
            for _ in range(events_this_cycle):
                event_type = random.choice(event_types)
                data = {
                    "timestamp": time.time(),
                    "message": f"Evento {events_emitted}"
                }
                
                emission_tasks.append(
                    coordinator.emit_event(event_type, data, "failure_test")
                )
                events_emitted += 1
            
            # Iniciar eventos
            for task in emission_tasks:
                asyncio.create_task(task)
            
            # Esperar
            await asyncio.sleep(0.1)
        
        # Esperar procesamiento pendiente
        logger.info("Prueba completada. Esperando eventos pendientes...")
        await asyncio.sleep(1.0)
        
        # Métricas finales
        metrics = coordinator.get_performance_metrics()
        metrics["test_specific"] = {
            "event_rate": event_rate,
            "total_events_emitted": events_emitted,
            "failures_injected": failures_injected,
            "test_duration": duration
        }
        
        logger.info(f"Fallos inyectados: {failures_injected}, Eventos: {events_emitted}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error durante prueba de fallos: {e}")
        return {
            "error": str(e),
            "events_emitted": events_emitted,
            "failures_injected": failures_injected
        }

async def run_quick_extreme_tests():
    """Ejecutar versión rápida de pruebas extremas."""
    try:
        logger.info("=== INICIANDO PRUEBAS RÁPIDAS DE ESTRÉS EXTREMO ===")
        
        # 1. Configurar sistema más pequeño
        system = await setup_extreme_test_system(
            num_components=8,               # Menos componentes
            failure_rates=(0.01, 0.05),     # 1-5% de fallos
            latency_range=(0.01, 0.05),     # 10-50ms de latencia
            crash_component_indices=[2, 5], # Dos componentes crashearán
            network_latency=(0.005, 0.02),  # 5-20ms de latencia de red
            network_failure_rate=0.01,      # 1% de fallos de red
            max_parallel_events=500         # Capacidad reducida
        )
        
        try:
            # 2. Prueba de volumen
            logger.info("\n=== PRUEBA RÁPIDA DE VOLUMEN ===\n")
            volume_results = await run_quick_volume_test(system)
            
            # 3. Prueba de fallos
            logger.info("\n=== PRUEBA RÁPIDA DE FALLOS SIMULTÁNEOS ===\n")
            failure_results = await run_quick_failure_test(system)
            
            # 4. Analizar resultados
            report = await analyze_and_report_results(volume_results, failure_results)
            
            # 5. Generar informe
            await generate_markdown_report(report, "docs/informe_pruebas_extremas_rapido.md")
            
            return report
            
        finally:
            # Detener sistema
            await system.stop()
            
    except Exception as e:
        logger.error(f"Error durante pruebas extremas rápidas: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Punto de entrada
if __name__ == "__main__":
    asyncio.run(run_quick_extreme_tests())