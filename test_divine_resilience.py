"""
Prueba extrema final para Genesis Divino.
"""

import asyncio
import logging
from random import random
from genesis_divine_resilience import TestComponent, GenesisHybridCoordinator, logger

async def simulate_component_failure(coordinator, component_id):
    for _ in range(15):
        await coordinator.request(component_id, "ping", {"fail": True}, "test")
        await asyncio.sleep(0.002)

async def simulate_high_load(coordinator):
    tasks = [
        coordinator.emit_local(f"event_{i}", {"value": i}, "test", priority="CRITICAL" if i % 10 == 0 else "NORMAL")
        for i in range(5000)
    ]
    await asyncio.gather(*tasks)

async def extreme_test(coordinator):
    comps = [TestComponent(f"comp{i}", is_essential=(i < 3)) for i in range(10)]
    for i, comp in enumerate(comps):
        coordinator.register_component(f"comp{i}", comp)

    coordinator.running = True
    logger.info("Iniciando prueba extrema del modo divino...")

    # Alta carga
    logger.info("Simulando alta carga (5000 eventos)...")
    await simulate_high_load(coordinator)

    # Fallos masivos
    logger.info("Simulando fallos masivos (80% de componentes)...")
    failure_tasks = [simulate_component_failure(coordinator, f"comp{i}") for i in range(8)]  # 80% fallos
    await asyncio.gather(*failure_tasks)

    # Latencias extremas
    logger.info("Simulando latencias extremas...")
    for i in range(8, 10):
        await coordinator.request(f"comp{i}", "ping", {"fail": True}, "test")
    await asyncio.sleep(0.05)

    # Resultados
    success_rate = sum(1 for comp in comps if not comp.failed) / len(comps)
    processed_events = sum(len(comp.local_events) for comp in comps)
    latency_success = coordinator.stats["recoveries"] / max(1, coordinator.stats["failures"])
    
    logger.info("=== RESULTADOS PRUEBA DIVINA ===")
    logger.info(f"Tasa de éxito global: {success_rate * 100:.2f}%")
    logger.info(f"Eventos procesados: {processed_events}/5000 ({processed_events/5000*100:.2f}%)")
    logger.info(f"Tasa de recuperación: {latency_success * 100:.2f}%")
    logger.info(f"Estadísticas: {coordinator.stats}")
    
    # Verificar umbral divino (>99%)
    divine_threshold = 0.99
    combined_score = (success_rate + (processed_events/5000) + latency_success) / 3
    logger.info(f"Puntuación combinada: {combined_score * 100:.2f}%")
    logger.info(f"¿Modo Divino superado? {'SÍ' if combined_score > divine_threshold else 'NO'}")

    coordinator.running = False
    await asyncio.sleep(0.1)  # Dar tiempo para finalizar tareas

async def run_test():
    coordinator = GenesisHybridCoordinator()
    await extreme_test(coordinator)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test())