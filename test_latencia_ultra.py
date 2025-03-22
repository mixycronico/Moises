"""
Prueba específica del manejo de latencias extremas para el sistema Genesis Ultra.

Esta prueba se enfoca exclusivamente en evaluar y mejorar el manejo de latencias
extremas, que es el punto débil identificado en la prueba anterior.
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List

# Importar sistema ultra-optimizado
from genesis_hybrid_resilience_ultra import (
    HybridCoordinator, TestComponent, EventPriority, logger, SystemMode,
    CircuitBreaker, CircuitState
)

async def test_latencia_ultra():
    """
    Ejecutar prueba específica de latencias extremas.
    
    Esta prueba:
    1. Crea un sistema pequeño (5 componentes)
    2. Realiza pruebas de latencia progresivas
    3. Analiza el comportamiento y falla
    4. Ajusta parámetros durante la ejecución
    5. Repite las pruebas para verificar mejora
    """
    logger.info("=== INICIANDO PRUEBA ESPECÍFICA DE LATENCIAS EXTREMAS ===")
    start_time = time.time()
    
    # Crear coordinador
    coordinator = HybridCoordinator()
    
    # Registrar solo 5 componentes para prueba específica
    for i in range(5):
        # Sin fallos aleatorios para esta prueba
        fail_rate = 0.0
        # 2 componentes esenciales, 3 no esenciales
        essential = i < 2
        component = TestComponent(f"component_{i}", essential=essential, fail_rate=fail_rate)
        coordinator.register_component(f"component_{i}", component)
    
    # Iniciar sistema
    await coordinator.start()
    
    try:
        # 1. PRUEBA DE LATENCIAS PROGRESIVAS
        logger.info("=== Fase 1: Prueba de Latencias Progresivas ===")
        
        # Probar latencias desde pequeñas hasta extremas
        latency_results_1 = []
        for latency in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            # Alternar entre componentes esenciales y no esenciales
            is_essential = latency < 2.0
            component_id = f"component_{0 if is_essential else 2}"
            
            start_op = time.time()
            operation_result = await coordinator.request(
                component_id, 
                "test_latency", 
                {"delay": latency},
                "test_system"
            )
            end_op = time.time()
            
            actual_latency = end_op - start_op
            success = operation_result is not None
            
            latency_results_1.append({
                "component": component_id,
                "essential": is_essential,
                "requested_latency": latency,
                "actual_latency": actual_latency,
                "success": success,
                "timeout_ratio": actual_latency / latency if latency > 0 else 1.0
            })
            
            # Mostrar resultado individual
            logger.info(f"Latencia {latency:.1f}s en {component_id}: " +
                       f"{'✓' if success else '✗'} ({actual_latency:.2f}s)")
            
            # Pausa para evitar interferencia
            await asyncio.sleep(0.1)
        
        # Analizar resultados
        successes_1 = sum(1 for r in latency_results_1 if r["success"])
        success_rate_1 = (successes_1 / len(latency_results_1)) * 100
        
        logger.info(f"Resultados fase 1: {successes_1}/{len(latency_results_1)} exitosas ({success_rate_1:.1f}%)")
        
        # Analizar por umbral de latencia
        for threshold in [1.0, 2.0, 3.0]:
            tests = [r for r in latency_results_1 if r["requested_latency"] <= threshold]
            if tests:
                success_count = sum(1 for r in tests if r["success"])
                logger.info(f"  Latencias <= {threshold:.1f}s: {success_count}/{len(tests)} " +
                           f"({(success_count/len(tests))*100:.1f}%)")
        
        # 2. AJUSTE DE PARÁMETROS
        logger.info("=== Fase 2: Ajuste de Parámetros ===")
        
        # Ajustar timeouts en componentes
        for component_id, component in coordinator.components.items():
            # Duplicar el timeout para circuit breaker
            component.circuit_breaker.recovery_timeout *= 2.0
            
            # Resetear estado del circuit breaker
            if component.circuit_breaker.state != CircuitState.CLOSED:
                component.circuit_breaker.state = CircuitState.CLOSED
                component.circuit_breaker.failure_count = 0
                component.circuit_breaker.degradation_score = 0
                logger.info(f"Reset de circuit breaker para {component_id}")
        
        # Esperar a que los ajustes se apliquen
        await asyncio.sleep(0.5)
        
        # 3. REPETIR PRUEBAS CON PARÁMETROS AJUSTADOS
        logger.info("=== Fase 3: Repetición con Parámetros Ajustados ===")
        
        # Repetir las mismas pruebas
        latency_results_2 = []
        for latency in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            # Alternar entre componentes esenciales y no esenciales
            is_essential = latency < 2.0
            component_id = f"component_{1 if is_essential else 3}"  # Usar componentes diferentes
            
            start_op = time.time()
            # Usar timeout más largo (2.5x la latencia esperada)
            timeout_override = latency * 2.5
            operation_result = await coordinator.request(
                component_id, 
                "test_latency", 
                {"delay": latency},
                "test_system",
                timeout=timeout_override
            )
            end_op = time.time()
            
            actual_latency = end_op - start_op
            success = operation_result is not None
            
            latency_results_2.append({
                "component": component_id,
                "essential": is_essential,
                "requested_latency": latency,
                "actual_latency": actual_latency,
                "success": success,
                "timeout_ratio": actual_latency / latency if latency > 0 else 1.0,
                "timeout_override": timeout_override
            })
            
            # Mostrar resultado individual
            logger.info(f"Latencia {latency:.1f}s en {component_id} (timeout={timeout_override:.1f}s): " +
                       f"{'✓' if success else '✗'} ({actual_latency:.2f}s)")
            
            # Pausa para evitar interferencia
            await asyncio.sleep(0.1)
        
        # Analizar resultados ajustados
        successes_2 = sum(1 for r in latency_results_2 if r["success"])
        success_rate_2 = (successes_2 / len(latency_results_2)) * 100
        
        logger.info(f"Resultados fase 3: {successes_2}/{len(latency_results_2)} exitosas ({success_rate_2:.1f}%)")
        
        # Analizar por umbral de latencia
        for threshold in [1.0, 2.0, 3.0]:
            tests = [r for r in latency_results_2 if r["requested_latency"] <= threshold]
            if tests:
                success_count = sum(1 for r in tests if r["success"])
                logger.info(f"  Latencias <= {threshold:.1f}s: {success_count}/{len(tests)} " +
                           f"({(success_count/len(tests))*100:.1f}%)")
        
        # 4. PRUEBA DE SOLUCIÓN DE REINTENTOS PARALELOS
        logger.info("=== Fase 4: Solución de Reintentos Paralelos ===")
        
        # En esta fase, se modifica dinámicamente la configuración para usar reintentos paralelos
        # específicamente para operaciones con latencia alta
        
        latency_results_3 = []
        for latency in [1.0, 2.0, 3.0]:
            # Usar componente no esencial con latencia alta
            component_id = "component_4"
            
            # Forzar modo ULTRA para activar procesamiento paralelo
            prev_mode = coordinator.mode
            coordinator.mode = SystemMode.ULTRA
            
            start_op = time.time()
            # Usar timeout largo pero con parallel_attempts=2
            operation_result = await coordinator.request(
                component_id, 
                "test_latency", 
                {"delay": latency},
                "test_system",
                timeout=latency * 2,
                priority=True  # Forzar procesamiento prioritario
            )
            end_op = time.time()
            
            # Restaurar modo
            coordinator.mode = prev_mode
            
            actual_latency = end_op - start_op
            success = operation_result is not None
            
            latency_results_3.append({
                "component": component_id,
                "requested_latency": latency,
                "actual_latency": actual_latency,
                "success": success,
                "mode": "ULTRA+parallel"
            })
            
            # Mostrar resultado individual
            logger.info(f"Latencia {latency:.1f}s con paralelo: " +
                       f"{'✓' if success else '✗'} ({actual_latency:.2f}s)")
            
            # Pausa para evitar interferencia
            await asyncio.sleep(0.2)
        
        # Analizar resultados paralelos
        successes_3 = sum(1 for r in latency_results_3 if r["success"])
        success_rate_3 = (successes_3 / len(latency_results_3)) * 100
        
        logger.info(f"Resultados fase 4: {successes_3}/{len(latency_results_3)} exitosas ({success_rate_3:.1f}%)")
        
        # 5. RESUMEN GLOBAL
        total_tests = len(latency_results_1) + len(latency_results_2) + len(latency_results_3)
        total_success = successes_1 + successes_2 + successes_3
        overall_success_rate = (total_success / total_tests) * 100
        
        improvement = success_rate_3 - success_rate_1
        
        logger.info("\n=== RESUMEN DE PRUEBA DE LATENCIAS EXTREMAS ===")
        logger.info(f"Fase 1 (original): {success_rate_1:.1f}%")
        logger.info(f"Fase 3 (parámetros ajustados): {success_rate_2:.1f}%")
        logger.info(f"Fase 4 (reintentos paralelos): {success_rate_3:.1f}%")
        logger.info(f"Tasa de éxito global: {overall_success_rate:.1f}%")
        logger.info(f"Mejora absoluta: {improvement:.1f}%")
        logger.info(f"Mejora relativa: {(improvement/success_rate_1)*100:.1f}%")
        
        # 6. RECOMENDACIONES
        logger.info("\n=== RECOMENDACIONES ===")
        
        if success_rate_3 > 90:
            logger.info("1. Implementar reintentos paralelos para todas las operaciones con latencia >1s")
            logger.info("2. Aumentar timeouts dinámicamente basados en latencia esperada (2.5x)")
            logger.info("3. Activar modo ULTRA automáticamente cuando se detecten latencias altas")
            
            # En caso de éxito 100%
            if success_rate_3 == 100:
                logger.info("4. La solución es óptima y puede ser implementada como está")
        else:
            logger.info("1. Revisar y optimizar la lógica de timeouts dinámicos")
            logger.info("2. Aumentar el número de reintentos paralelos a 3 para componentes no esenciales")
            logger.info("3. Considerar fallbacks más agresivos para operaciones con latencia >2s")
        
        return {
            "phase1_rate": success_rate_1,
            "phase2_rate": success_rate_2,
            "phase3_rate": success_rate_3,
            "overall_rate": overall_success_rate,
            "improvement": improvement
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
    asyncio.run(test_latencia_ultra())