"""
Prueba extrema avanzada para el sistema Genesis con optimizaciones máximas.

Esta prueba evalúa el sistema con condiciones aún más severas que las anteriores:
- Mayor carga (3000 eventos locales + 200 externos)
- Más componentes (30 en total)
- Mayor tasa de fallos (hasta 30%)
- Latencias extremas (hasta 2s)
- Fallos en cascada simulados

Objetivo: Alcanzar >95% de tasa de éxito global
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List

# Importar sistema optimizado extremo
from genesis_hybrid_resilience_extreme import (
    HybridCoordinator, TestComponent, EventPriority, logger, SystemMode
)

async def test_resiliencia_extrema_v2():
    """
    Ejecutar prueba extrema avanzada para verificar las optimizaciones máximas
    bajo condiciones extremas severas.
    """
    logger.info("=== INICIANDO PRUEBA EXTREMA DE RESILIENCIA AVANZADA V2 ===")
    start_time = time.time()
    
    # Crear coordinador
    coordinator = HybridCoordinator()
    
    # Registrar componentes para la prueba (30 componentes)
    for i in range(30):
        # Diferentes tasas de fallo para simular componentes poco confiables
        fail_rate = random.uniform(0.0, 0.3)  # Entre 0% y 30% de fallo
        # Marcar algunos como esenciales
        essential = i < 6  # Los primeros 6 son esenciales
        component = TestComponent(f"component_{i}", essential=essential, fail_rate=fail_rate)
        coordinator.register_component(f"component_{i}", component)
    
    # Iniciar sistema
    await coordinator.start()
    
    try:
        # 1. PRUEBA DE ALTA CARGA
        logger.info("=== Prueba de Alta Carga ===")
        start_test = time.time()
        
        # Generar 3000 eventos locales concurrentes
        local_tasks = [
            coordinator.emit_local(
                f"event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                # Asignar prioridades variadas
                priority=EventPriority(random.randint(0, 4))  # Incluir BACKGROUND
            )
            for i in range(3000)
        ]
        
        # Generar 200 eventos externos concurrentes
        external_tasks = [
            coordinator.emit_external(
                f"ext_event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                # Asignar prioridades variadas
                priority=EventPriority(random.randint(0, 3))
            )
            for i in range(200)
        ]
        
        # Ejecutar en paralelo
        await asyncio.gather(*(local_tasks + external_tasks))
        
        # Dar tiempo para procesar 
        await asyncio.sleep(0.3)
        
        # Calcular resultados de alta carga
        high_load_duration = time.time() - start_test
        
        # Resultados procesados
        total_processed = sum(comp.stats["processed_events"] 
                             for comp in coordinator.components.values())
        
        # Resultados de procesamiento por lotes
        batch_counts = sum(comp.stats.get("batch_count", 0) 
                          for comp in coordinator.components.values())
        
        logger.info(f"Prueba de alta carga completada en {high_load_duration:.2f}s")
        logger.info(f"Eventos emitidos: 3200, Procesados: {total_processed}")
        logger.info(f"Procesamiento por lotes: {batch_counts} lotes procesados")
        
        # 2. PRUEBA DE FALLOS MASIVOS
        logger.info("=== Prueba de Fallos Masivos ===")
        start_test = time.time()
        
        # Forzar fallos en 12 componentes no esenciales (40%)
        components_to_fail = [f"component_{i}" for i in range(6, 18)]
        
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
        for latency in [0.05, 0.2, 0.5, 1.0, 1.5, 2.0]:
            component_id = f"component_{random.randint(18, 29)}"  # Usar componentes sanos
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
        
        # 4. PRUEBA DE FALLOS EN CASCADA
        logger.info("=== Prueba de Fallos en Cascada ===")
        
        # Seleccionar un componente del que dependen otros
        central_component = "component_2"  # Componente esencial
        
        # Forzar fallo en el componente central
        logger.info(f"Forzando fallo en componente central {central_component}")
        await coordinator.request(central_component, "fail", {}, "test_system")
        
        # Inmediatamente generar carga alta para estresar el sistema
        cascade_tasks = [
            coordinator.emit_local(
                f"cascade_event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                priority=EventPriority.HIGH  # Hacer eventos importantes para crear presión
            )
            for i in range(500)
        ]
        
        await asyncio.gather(*cascade_tasks)
        
        # Verificar modo del sistema
        logger.info(f"Modo del sistema tras fallo en cascada: {coordinator.mode.value}")
        logger.info(f"Salud del sistema: {coordinator.system_health:.2f}%")
        
        # Esperar a recuperación
        await asyncio.sleep(0.7)
        
        # 5. VERIFICACIÓN DE RECUPERACIÓN TRAS FALLOS
        logger.info("=== Verificación de Recuperación ===")
        
        # Contar componentes recuperados
        recovered_count = coordinator.stats["recoveries"]
        
        # Verificar estado final
        active_components = sum(1 for comp in coordinator.components.values() if comp.active)
        final_mode = coordinator.mode
        
        # Verificar métricas
        logger.info(f"Componentes activos después de prueba: {active_components}/30")
        logger.info(f"Componentes recuperados: {recovered_count}")
        logger.info(f"Modo final del sistema: {final_mode.value}")
        logger.info(f"Salud del sistema: {coordinator.system_health:.2f}%")
        
        # 6. CÁLCULO DE TASA DE ÉXITO GLOBAL
        total_duration = time.time() - start_time
        system_stats = coordinator.get_stats()
        
        # Tasa de procesamiento de eventos
        total_events_sent = 3700  # 3000 locales + 200 externos + 500 cascada
        events_processed = total_processed
        event_process_rate = (events_processed / (total_events_sent * len(coordinator.components))) * 100
        
        # Tasa de recuperación
        # Componentes que fallaron: 12 originales + 1 central = 13
        recovery_rate = (recovered_count / 13) * 100 if recovered_count > 0 else 0
        
        # Tasa de éxito de latencia
        latency_success_rate = (latency_success / latency_total) * 100 if latency_total else 100
        
        # Tasa de salud del sistema
        health_rate = coordinator.system_health
        
        # Tasa global (promedio ponderado)
        global_success_rate = (
            0.4 * event_process_rate +   # 40% peso a procesamiento
            0.3 * recovery_rate +        # 30% peso a recuperación
            0.2 * latency_success_rate + # 20% peso a latencia
            0.1 * health_rate            # 10% peso a salud general
        )
        
        # Resultados finales
        logger.info("\n=== RESUMEN DE PRUEBA EXTREMA AVANZADA V2 ===")
        logger.info(f"Duración total: {total_duration:.2f}s")
        logger.info(f"Tasa de procesamiento de eventos: {event_process_rate:.2f}%")
        logger.info(f"Tasa de recuperación: {recovery_rate:.2f}%")
        logger.info(f"Tasa de éxito con latencia: {latency_success_rate:.2f}%")
        logger.info(f"Salud del sistema: {health_rate:.2f}%")
        logger.info(f"Tasa de éxito global: {global_success_rate:.2f}%")
        logger.info(f"API calls: {system_stats['api_calls']}, "
                   f"Local events: {system_stats['local_events']}, "
                   f"External events: {system_stats['external_events']}")
        logger.info(f"Fallos: {system_stats['failures']}, "
                   f"Recuperaciones: {system_stats['recoveries']}, "
                   f"Timeouts: {system_stats.get('timeouts', 0)}")
        logger.info(f"Modo final del sistema: {system_stats['mode']}")
        
        return {
            "duration": total_duration,
            "event_process_rate": event_process_rate,
            "recovery_rate": recovery_rate,
            "latency_success_rate": latency_success_rate,
            "health_rate": health_rate,
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
    asyncio.run(test_resiliencia_extrema_v2())