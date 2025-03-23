"""
Prueba rápida para el sistema Genesis con optimizaciones avanzadas.

Versión simplificada de la prueba extrema para obtener resultados rápidos.
Evalúa las mejoras clave:
- Timeout global
- Circuit Breaker predictivo
- Checkpointing diferencial
- Procesamiento por lotes
- Modo PRE-SAFE
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List

# Importar sistema optimizado extremo
from genesis_hybrid_resilience_extreme import (
    HybridCoordinator, TestComponent, EventPriority, logger
)

async def test_resiliencia_rapida_v2():
    """Ejecutar prueba rápida del sistema optimizado."""
    logger.info("=== INICIANDO PRUEBA RÁPIDA DE RESILIENCIA AVANZADA V2 ===")
    start_time = time.time()
    
    # Crear coordinador
    coordinator = HybridCoordinator()
    
    # Registrar menos componentes para prueba rápida
    for i in range(10):  # Solo 10 componentes
        fail_rate = random.uniform(0.0, 0.2)
        essential = i < 3  # 3 esenciales
        component = TestComponent(f"component_{i}", essential=essential, fail_rate=fail_rate)
        coordinator.register_component(f"component_{i}", component)
    
    # Iniciar sistema
    await coordinator.start()
    
    try:
        # 1. PRUEBA DE ALTA CARGA (reducida)
        logger.info("=== Prueba de Alta Carga ===")
        start_test = time.time()
        
        # Generar 1000 eventos locales (en lugar de 3000)
        local_tasks = [
            coordinator.emit_local(
                f"event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                priority=EventPriority(random.randint(0, 4))
            )
            for i in range(1000)
        ]
        
        # Generar 50 eventos externos (en lugar de 200)
        external_tasks = [
            coordinator.emit_external(
                f"ext_event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                priority=EventPriority(random.randint(0, 3))
            )
            for i in range(50)
        ]
        
        # Ejecutar en paralelo
        await asyncio.gather(*(local_tasks + external_tasks))
        
        # Tiempo para procesar
        await asyncio.sleep(0.2)
        
        # Resultados procesados
        total_processed = sum(comp.stats["processed_events"] 
                            for comp in coordinator.components.values())
        
        logger.info(f"Eventos emitidos: 1050, Procesados: {total_processed}")
        
        # 2. PRUEBA DE FALLOS SELECTIVOS
        logger.info("=== Prueba de Fallos Selectivos ===")
        
        # Fallar solo 3 componentes
        components_to_fail = ["component_3", "component_4", "component_5"]
        
        fail_tasks = [
            coordinator.request(cid, "fail", {}, "test_system")
            for cid in components_to_fail
        ]
        
        await asyncio.gather(*fail_tasks, return_exceptions=True)
        await asyncio.sleep(0.2)
        
        # 3. PRUEBA DE LATENCIAS
        logger.info("=== Prueba de Latencias ===")
        
        # Menos pruebas de latencia
        latency_results = []
        for latency in [0.1, 0.5, 1.0]:
            component_id = f"component_{random.randint(6, 9)}"
            start_op = time.time()
            operation_result = await coordinator.request(
                component_id, 
                "test_latency", 
                {"delay": latency},
                "test_system"
            )
            end_op = time.time()
            
            latency_results.append({
                "requested_latency": latency,
                "actual_latency": end_op - start_op,
                "success": operation_result is not None
            })
        
        latency_success = sum(1 for r in latency_results if r["success"])
        latency_total = len(latency_results)
        
        # 4. VERIFICACIÓN DE RECUPERACIÓN
        logger.info("=== Verificación de Recuperación ===")
        await asyncio.sleep(0.3)
        
        recovered_count = coordinator.stats["recoveries"]
        active_components = sum(1 for comp in coordinator.components.values() if comp.active)
        
        # 5. RESULTADOS FINALES
        total_duration = time.time() - start_time
        system_stats = coordinator.get_stats()
        
        # Cálculo de métricas
        total_events_sent = 1050  # 1000 locales + 50 externos
        event_process_rate = (total_processed / (total_events_sent * len(coordinator.components))) * 100
        recovery_rate = (recovered_count / len(components_to_fail)) * 100 if components_to_fail else 100
        latency_success_rate = (latency_success / latency_total) * 100 if latency_total else 100
        health_rate = coordinator.system_health
        
        # Tasa global
        global_success_rate = (
            0.4 * event_process_rate + 
            0.3 * recovery_rate + 
            0.2 * latency_success_rate + 
            0.1 * health_rate
        )
        
        # Resultados
        logger.info("\n=== RESUMEN DE PRUEBA RÁPIDA V2 ===")
        logger.info(f"Duración total: {total_duration:.2f}s")
        logger.info(f"Tasa de procesamiento: {event_process_rate:.2f}%")
        logger.info(f"Tasa de recuperación: {recovery_rate:.2f}%")
        logger.info(f"Tasa de éxito con latencia: {latency_success_rate:.2f}%")
        logger.info(f"Salud del sistema: {health_rate:.2f}%")
        logger.info(f"Tasa de éxito global: {global_success_rate:.2f}%")
        logger.info(f"Componentes activos: {active_components}/10")
        logger.info(f"Modo final: {system_stats['mode']}")
        
        return {
            "global_success_rate": global_success_rate,
            "stats": system_stats
        }
    
    finally:
        await coordinator.stop()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(test_resiliencia_rapida_v2())