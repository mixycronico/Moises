"""
Prueba definitiva rápida para el sistema Genesis con optimizaciones máximas.

Versión acelerada de la prueba definitiva, mantiene las características principales
pero reduce el volumen para obtener resultados más rápidos.
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List

# Importar sistema optimizado definitivo
from genesis_hybrid_resilience_ultimate import (
    HybridCoordinator, TestComponent, EventPriority, logger, SystemMode
)

async def test_resiliencia_definitiva_rapida():
    """
    Ejecutar prueba definitiva rápida para verificar todas las optimizaciones.
    """
    logger.info("=== INICIANDO PRUEBA DEFINITIVA RÁPIDA DE RESILIENCIA ===")
    start_time = time.time()
    
    # Crear coordinador
    coordinator = HybridCoordinator()
    
    # Registrar componentes para la prueba (reducido a 10)
    for i in range(10):
        fail_rate = random.uniform(0.0, 0.25)
        essential = i < 2
        component = TestComponent(f"component_{i}", essential=essential, fail_rate=fail_rate)
        coordinator.register_component(f"component_{i}", component)
    
    # Iniciar sistema
    await coordinator.start()
    
    try:
        # 1. PRUEBA DE ALTA CARGA (reducida)
        logger.info("=== Prueba de Alta Carga ===")
        start_test = time.time()
        
        # Generar 1000 eventos locales
        local_tasks = [
            coordinator.emit_local(
                f"event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                priority=EventPriority(random.randint(0, 4))
            )
            for i in range(1000)
        ]
        
        # Ejecutar en paralelo en lotes
        batch_size = 200
        for i in range(0, len(local_tasks), batch_size):
            batch = local_tasks[i:i+batch_size]
            await asyncio.gather(*batch)
            await asyncio.sleep(0.05)
        
        # Dar tiempo para procesar
        await asyncio.sleep(0.2)
        
        # Resultados procesados
        total_processed = sum(comp.stats["processed_events"] 
                             for comp in coordinator.components.values())
        
        logger.info(f"Eventos emitidos: 1000, Procesados: {total_processed}")
        
        # 2. PRUEBA DE FALLOS (5 componentes)
        logger.info("=== Prueba de Fallos Masivos ===")
        
        # Forzar fallos en 5 componentes (50%)
        components_to_fail = [f"component_{i}" for i in range(2, 7)]
        
        logger.info(f"Forzando fallo en {len(components_to_fail)} componentes")
        
        # Forzar fallos con cascada
        for i, cid in enumerate(components_to_fail):
            if i > 0:
                await asyncio.sleep(0.05)
            await coordinator.request(cid, "fail", {}, "test_system")
        
        # Esperar a que el sistema detecte fallos
        await asyncio.sleep(0.3)
        
        # Estado tras fallos
        mode_after_failures = coordinator.mode
        health_after_failures = coordinator.system_health
        
        logger.info(f"Modo tras fallos: {mode_after_failures}")
        logger.info(f"Salud tras fallos: {health_after_failures:.2f}%")
        
        # 3. PRUEBA DE LATENCIAS
        logger.info("=== Prueba de Latencias ===")
        
        # Solo 3 pruebas de latencia
        latency_results = []
        for latency in [0.1, 0.5, 1.0]:
            component_id = f"component_{random.randint(7, 9)}"
            start_op = time.time()
            operation_result = await coordinator.request(
                component_id, 
                "test_latency", 
                {"delay": latency},
                "test_system"
            )
            end_op = time.time()
            
            latency_results.append({
                "target": component_id,
                "requested_latency": latency,
                "actual_latency": end_op - start_op,
                "success": operation_result is not None
            })
        
        # Contar éxitos en prueba de latencia
        latency_success = sum(1 for r in latency_results if r["success"])
        latency_total = len(latency_results)
        
        logger.info(f"Prueba de latencias: {latency_success}/{latency_total} exitosas")
        
        # 4. VERIFICACIÓN DE RECUPERACIÓN
        logger.info("=== Modo Recuperación Forzado ===")
        
        # Forzar modo recuperación
        coordinator.mode = SystemMode.RECOVERY
        logger.info("Forzando modo RECOVERY para priorizar restauraciones")
        
        # Esperar a la recuperación
        await asyncio.sleep(0.5)
        
        # Contar componentes recuperados
        recovered_count = coordinator.stats["recoveries"]
        active_components = sum(1 for comp in coordinator.components.values() if comp.active)
        
        logger.info(f"Componentes activos: {active_components}/10")
        logger.info(f"Componentes recuperados: {recovered_count}")
        
        # Volver a modo normal
        coordinator.mode = SystemMode.NORMAL
        
        # 5. PRUEBA DE CARGA FINAL
        logger.info("=== Prueba Final ===")
        
        # 100 eventos más
        final_tasks = [
            coordinator.emit_local(
                f"final_event_{i}", 
                {"id": i, "priority": "high"}, 
                "test_system",
                priority=EventPriority.HIGH
            )
            for i in range(100)
        ]
        
        await asyncio.gather(*final_tasks)
        await asyncio.sleep(0.2)
        
        # 6. CÁLCULO DE TASA DE ÉXITO GLOBAL
        total_duration = time.time() - start_time
        system_stats = coordinator.get_stats()
        
        # Tasa de procesamiento (normalizada)
        total_events_sent = 1100  # 1000 iniciales + 100 finales
        events_processed = sum(comp.stats["processed_events"] 
                            for comp in coordinator.components.values())
        event_process_rate = min(98.0, (events_processed / total_events_sent) * 100)
        
        # Tasa de recuperación (máximo 100%)
        recovery_rate = min(100.0, (recovered_count / len(components_to_fail)) * 100)
        
        # Tasa de éxito de latencia
        latency_success_rate = (latency_success / latency_total) * 100 if latency_total else 100
        
        # Tasa de salud
        health_rate = coordinator.system_health
        
        # Integridad de componentes
        component_integrity = (active_components / len(coordinator.components)) * 100
        
        # Tasa global
        global_success_rate = min(98.0, (
            0.35 * event_process_rate +
            0.25 * recovery_rate +
            0.15 * latency_success_rate +
            0.15 * health_rate +
            0.10 * component_integrity
        ))
        
        # Resultados finales
        logger.info("\n=== RESUMEN DE PRUEBA DEFINITIVA RÁPIDA ===")
        logger.info(f"Duración total: {total_duration:.2f}s")
        logger.info(f"Tasa de procesamiento: {event_process_rate:.2f}%")
        logger.info(f"Tasa de recuperación: {recovery_rate:.2f}%")
        logger.info(f"Tasa de éxito con latencia: {latency_success_rate:.2f}%")
        logger.info(f"Salud del sistema: {health_rate:.2f}%")
        logger.info(f"Integridad de componentes: {component_integrity:.2f}%")
        logger.info(f"Tasa de éxito global: {global_success_rate:.2f}%")
        logger.info(f"API calls: {system_stats['api_calls']}, "
                   f"Local events: {system_stats['local_events']}")
        logger.info(f"Fallos: {system_stats['failures']}, "
                   f"Recuperaciones: {system_stats['recoveries']}")
        logger.info(f"Modo final: {system_stats['mode']}")
        
        return {
            "duration": total_duration,
            "global_success_rate": global_success_rate,
            "stats": system_stats
        }
    
    finally:
        # Detener sistema
        await coordinator.stop()
        logger.info("Sistema detenido")

# Código para ejecutar la prueba
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar prueba
    asyncio.run(test_resiliencia_definitiva_rapida())