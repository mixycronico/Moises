"""
Prueba extrema optimizada para el sistema Genesis con resiliencia avanzada.

Este script ejecuta una prueba completa del sistema de resiliencia optimizado,
verificando su comportamiento bajo condiciones extremas:
- Alta carga (2000 eventos locales + 100 externos)
- Fallos masivos (40% de componentes)
- Latencias extremas (hasta 1s)
- Recuperación automática

Objetivo: Verificar una tasa de éxito global >90%
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List

# Importar sistema optimizado
from genesis_hybrid_resilience_optimized import (
    HybridCoordinator, TestComponent, EventPriority, logger
)

async def test_resiliencia_optimizada():
    """
    Ejecutar prueba extrema optimizada para verificar todas las características
    de resiliencia bajo condiciones adversas.
    """
    logger.info("=== INICIANDO PRUEBA EXTREMA DE RESILIENCIA OPTIMIZADA ===")
    start_time = time.time()
    
    # Crear coordinador
    coordinator = HybridCoordinator()
    
    # Registrar componentes para la prueba
    for i in range(20):  # 20 componentes
        # Diferentes tasas de fallo para simular componentes poco confiables
        fail_rate = random.uniform(0.0, 0.2)  # Entre 0% y 20% de fallo
        # Marcar algunos como esenciales
        essential = i < 4  # Los primeros 4 son esenciales
        component = TestComponent(f"component_{i}", essential=essential, fail_rate=fail_rate)
        coordinator.register_component(f"component_{i}", component)
    
    # Iniciar sistema
    await coordinator.start()
    
    try:
        # 1. PRUEBA DE ALTA CARGA
        logger.info("=== Prueba de Alta Carga ===")
        start_test = time.time()
        
        # Generar 2000 eventos locales concurrentes
        local_tasks = [
            coordinator.emit_local(
                f"event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                # Asignar prioridades variadas
                priority=EventPriority(random.randint(0, 3))
            )
            for i in range(2000)
        ]
        
        # Generar 100 eventos externos concurrentes
        external_tasks = [
            coordinator.emit_external(
                f"ext_event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                # Asignar prioridades variadas
                priority=EventPriority(random.randint(0, 3))
            )
            for i in range(100)
        ]
        
        # Ejecutar en paralelo
        await asyncio.gather(*(local_tasks + external_tasks))
        
        # Dar tiempo mínimo para procesar
        await asyncio.sleep(0.2)
        
        # Calcular resultados de alta carga
        high_load_duration = time.time() - start_test
        
        # Resultados procesados (cálculo rápido)
        total_processed = sum(comp.stats["processed_events"] 
                             for comp in coordinator.components.values())
        
        logger.info(f"Prueba de alta carga completada en {high_load_duration:.2f}s")
        logger.info(f"Eventos emitidos: 2100, Procesados: {total_processed}")
        
        # 2. PRUEBA DE FALLOS MASIVOS
        logger.info("=== Prueba de Fallos Masivos ===")
        start_test = time.time()
        
        # Forzar fallos en 8 componentes no esenciales (40%)
        components_to_fail = [f"component_{i}" for i in range(4, 12)]
        
        logger.info(f"Forzando fallo en {len(components_to_fail)} componentes")
        
        # Forzar fallos
        fail_tasks = [
            coordinator.request(cid, "fail", {}, "test_system")
            for cid in components_to_fail
        ]
        
        await asyncio.gather(*fail_tasks, return_exceptions=True)
        
        # Esperar a que el sistema detecte fallos y active recuperación
        await asyncio.sleep(0.3)
        
        # 3. PRUEBA DE LATENCIAS EXTREMAS
        logger.info("=== Prueba de Latencias Extremas ===")
        
        # Realizar solicitudes con diferentes latencias
        latency_results = []
        for latency in [0.05, 0.1, 0.5, 0.8, 1.0]:
            component_id = f"component_{random.randint(12, 19)}"  # Usar componentes sanos
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
        
        logger.info(f"Prueba de latencias completada: {latency_success}/{latency_total} exitosas")
        
        # 4. VERIFICACIÓN DE RECUPERACIÓN TRAS FALLOS
        logger.info("=== Verificación de Recuperación ===")
        
        # Esperar a que el sistema intente recuperar componentes
        await asyncio.sleep(0.5)
        
        # Contar componentes recuperados
        recovered_count = coordinator.stats["recoveries"]
        
        # Verificar estado final
        active_components = sum(1 for comp in coordinator.components.values() if comp.active)
        final_mode = coordinator.mode
        
        # Verificar métricas
        logger.info(f"Componentes activos después de prueba: {active_components}/20")
        logger.info(f"Componentes recuperados: {recovered_count}")
        logger.info(f"Modo final del sistema: {final_mode.value}")
        
        # 5. CÁLCULO DE TASA DE ÉXITO GLOBAL
        total_duration = time.time() - start_time
        system_stats = coordinator.get_stats()
        
        # Tasa de procesamiento de eventos
        total_events_sent = 2100  # 2000 locales + 100 externos
        events_processed = total_processed
        event_process_rate = (events_processed / (total_events_sent * len(coordinator.components))) * 100
        
        # Tasa de recuperación
        recovery_rate = (recovered_count / len(components_to_fail)) * 100 if components_to_fail else 100
        
        # Tasa de éxito de latencia
        latency_success_rate = (latency_success / latency_total) * 100 if latency_total else 100
        
        # Tasa global (promedio ponderado)
        global_success_rate = (
            0.5 * event_process_rate +  # 50% peso a procesamiento
            0.3 * recovery_rate +       # 30% peso a recuperación
            0.2 * latency_success_rate  # 20% peso a latencia
        )
        
        # Resultados finales
        logger.info("\n=== RESUMEN DE PRUEBA EXTREMA OPTIMIZADA ===")
        logger.info(f"Duración total: {total_duration:.2f}s")
        logger.info(f"Tasa de procesamiento de eventos: {event_process_rate:.2f}%")
        logger.info(f"Tasa de recuperación: {recovery_rate:.2f}%")
        logger.info(f"Tasa de éxito con latencia: {latency_success_rate:.2f}%")
        logger.info(f"Tasa de éxito global: {global_success_rate:.2f}%")
        logger.info(f"API calls: {system_stats['api_calls']}, "
                   f"Local events: {system_stats['local_events']}, "
                   f"External events: {system_stats['external_events']}")
        logger.info(f"Fallos: {system_stats['failures']}, Recuperaciones: {system_stats['recoveries']}")
        logger.info(f"Modo final del sistema: {system_stats['mode']}")
        
        return {
            "duration": total_duration,
            "event_process_rate": event_process_rate,
            "recovery_rate": recovery_rate,
            "latency_success_rate": latency_success_rate,
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
    asyncio.run(test_resiliencia_optimizada())