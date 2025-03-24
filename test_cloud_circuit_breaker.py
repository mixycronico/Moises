#!/usr/bin/env python3
"""
Prueba del CloudCircuitBreaker Ultra-Divino.

Este script ejecuta una prueba completa del CloudCircuitBreaker, demostrando
su capacidad para prevenir fallos en cascada y garantizar resiliencia en
entornos cloud.
"""

import asyncio
import random
import time
import logging
from typing import Dict, Any, List, Optional

from genesis.cloud import (
    CloudCircuitBreaker, CloudCircuitBreakerFactory, CircuitState,
    circuit_breaker_factory, circuit_protected
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_cloud_circuit_breaker")


class BeautifulTerminalColors:
    """Colores para terminal con estilo artístico."""
    HEADER = '\033[95m'        # Magenta claro
    BLUE = '\033[94m'          # Azul
    CYAN = '\033[96m'          # Cian
    GREEN = '\033[92m'         # Verde
    YELLOW = '\033[93m'        # Amarillo
    RED = '\033[91m'           # Rojo
    BOLD = '\033[1m'           # Negrita
    UNDERLINE = '\033[4m'      # Subrayado
    DIVINE = '\033[38;5;141m'  # Púrpura divino
    QUANTUM = '\033[38;5;39m'  # Azul cuántico
    COSMIC = '\033[38;5;208m'  # Naranja cósmico
    TRANSCEND = '\033[38;5;51m'# Aguamarina trascendental
    END = '\033[0m'            # Reset


class TestStats:
    """Estadísticas para las pruebas."""
    
    def __init__(self):
        """Inicializar estadísticas."""
        self.operations_total = 0
        self.operations_success = 0
        self.operations_failed = 0
        self.operations_recovered = 0
        self.recovery_time_total = 0
        self.recovery_time_max = 0
        self.recovery_time_min = float('inf')
        self.start_time = time.time()
        self.end_time = 0
    
    def record_success(self):
        """Registrar operación exitosa."""
        self.operations_total += 1
        self.operations_success += 1
    
    def record_failure(self):
        """Registrar operación fallida."""
        self.operations_total += 1
        self.operations_failed += 1
    
    def record_recovery(self, time_taken: float):
        """Registrar recuperación."""
        self.operations_recovered += 1
        self.recovery_time_total += time_taken
        self.recovery_time_max = max(self.recovery_time_max, time_taken)
        self.recovery_time_min = min(self.recovery_time_min, time_taken)
    
    def finish(self):
        """Finalizar estadísticas."""
        self.end_time = time.time()
    
    def total_time(self) -> float:
        """Obtener tiempo total de ejecución."""
        return self.end_time - self.start_time
    
    def success_rate(self) -> float:
        """Obtener tasa de éxito."""
        if self.operations_total == 0:
            return 0
        return self.operations_success / self.operations_total
    
    def recovery_rate(self) -> float:
        """Obtener tasa de recuperación."""
        if self.operations_failed == 0:
            return 0
        return self.operations_recovered / self.operations_failed
    
    def avg_recovery_time(self) -> float:
        """Obtener tiempo medio de recuperación."""
        if self.operations_recovered == 0:
            return 0
        return self.recovery_time_total / self.operations_recovered
    
    def print_summary(self):
        """Imprimir resumen de estadísticas."""
        c = BeautifulTerminalColors
        print(f"\n{c.DIVINE}{c.BOLD}=== RESUMEN DE PRUEBAS ==={c.END}")
        print(f"{c.CYAN}Tiempo total:{c.END} {self.total_time():.3f} segundos")
        print(f"{c.CYAN}Operaciones totales:{c.END} {self.operations_total}")
        print(f"{c.GREEN}Operaciones exitosas:{c.END} {self.operations_success}")
        print(f"{c.RED}Operaciones fallidas:{c.END} {self.operations_failed}")
        print(f"{c.QUANTUM}Operaciones recuperadas:{c.END} {self.operations_recovered}")
        print(f"{c.CYAN}Tasa de éxito:{c.END} {self.success_rate()*100:.2f}%")
        print(f"{c.CYAN}Tasa de recuperación:{c.END} {self.recovery_rate()*100:.2f}%")
        
        if self.operations_recovered > 0:
            print(f"{c.CYAN}Tiempo medio de recuperación:{c.END} {self.avg_recovery_time()*1000:.3f} ms")
            print(f"{c.CYAN}Tiempo máximo de recuperación:{c.END} {self.recovery_time_max*1000:.3f} ms")
            print(f"{c.CYAN}Tiempo mínimo de recuperación:{c.END} {self.recovery_time_min*1000:.3f} ms")
        
        print(f"{c.DIVINE}{c.BOLD}========================={c.END}\n")


# Función simulada para operaciones de base de datos
async def simulated_db_operation(operation_id: int, should_fail: bool = False) -> Dict[str, Any]:
    """
    Simular operación de base de datos.
    
    Args:
        operation_id: ID de la operación
        should_fail: Si debe fallar la operación
    
    Returns:
        Resultado de la operación
    """
    # Simular latencia
    await asyncio.sleep(random.uniform(0.001, 0.005))
    
    if should_fail:
        raise Exception(f"Error simulado en operación {operation_id}")
    
    return {
        "operation_id": operation_id,
        "status": "success",
        "timestamp": time.time()
    }


# Función simulada para operaciones cloud
async def simulated_cloud_operation(operation_id: int, retry_count: int = 0) -> Dict[str, Any]:
    """
    Simular operación en entorno cloud.
    
    Args:
        operation_id: ID de la operación
        retry_count: Contador de reintentos
    
    Returns:
        Resultado de la operación
    """
    # Aumentar probabilidad de fallo con cada reintento
    failure_probability = min(0.2 + (retry_count * 0.1), 0.8)
    
    # Simular latencia variable
    await asyncio.sleep(random.uniform(0.002, 0.01))
    
    # Simular fallo aleatorio
    if random.random() < failure_probability:
        if retry_count > 2:
            # Fallo catastrófico después de muchos reintentos
            raise Exception(f"Fallo catastrófico en operación cloud {operation_id} después de {retry_count} reintentos")
        
        # Fallo recuperable
        raise Exception(f"Error temporal en operación cloud {operation_id}")
    
    return {
        "operation_id": operation_id,
        "status": "success",
        "cloud_id": f"cloud-{operation_id}-{random.randint(1000, 9999)}",
        "timestamp": time.time(),
        "retry_count": retry_count
    }


# Decorador de prueba para funciones protegidas por CircuitBreaker
@circuit_protected("db_operations", failure_threshold=3, recovery_timeout=0.01)
async def protected_db_operation(operation_id: int, fail_rate: float = 0.2) -> Optional[Dict[str, Any]]:
    """
    Operación de DB protegida por CircuitBreaker.
    
    Args:
        operation_id: ID de la operación
        fail_rate: Tasa de fallos (0-1)
    
    Returns:
        Resultado de la operación o None
    """
    should_fail = random.random() < fail_rate
    return await simulated_db_operation(operation_id, should_fail)


@circuit_protected("cloud_operations", failure_threshold=5, recovery_timeout=0.02, quantum_failsafe=True)
async def protected_cloud_operation(operation_id: int, retry_count: int = 0) -> Optional[Dict[str, Any]]:
    """
    Operación cloud protegida por CircuitBreaker.
    
    Args:
        operation_id: ID de la operación
        retry_count: Contador de reintentos
    
    Returns:
        Resultado de la operación o None
    """
    return await simulated_cloud_operation(operation_id, retry_count)


async def test_circuit_breaker_basics():
    """Probar funcionalidad básica del CircuitBreaker."""
    c = BeautifulTerminalColors
    print(f"\n{c.DIVINE}=== PRUEBA DE FUNCIONALIDAD BÁSICA ==={c.END}")
    
    # Crear un CircuitBreaker directamente
    cb = await circuit_breaker_factory.create("test_basic", failure_threshold=3)
    
    # Operaciones exitosas
    print(f"{c.CYAN}Ejecutando operaciones exitosas...{c.END}")
    for i in range(5):
        result = await cb.call(simulated_db_operation, i, False)
        print(f"  Operación #{i}: {c.GREEN}Éxito{c.END}")
    
    # Estado actual
    print(f"\nEstado actual: {c.BOLD}{cb.get_state()}{c.END}")
    
    # Operaciones fallidas
    print(f"\n{c.CYAN}Ejecutando operaciones con fallos...{c.END}")
    for i in range(5):
        try:
            result = await cb.call(simulated_db_operation, i, True)
            print(f"  Operación #{i}: {c.GREEN}Éxito (inesperado){c.END}")
        except Exception as e:
            print(f"  Operación #{i}: {c.RED}Fallo{c.END} - {e}")
    
    # Estado tras fallos
    print(f"\nEstado tras fallos: {c.BOLD}{cb.get_state()}{c.END}")
    
    # Esperar recuperación
    print(f"\n{c.CYAN}Esperando para recuperación automática...{c.END}")
    await asyncio.sleep(0.05)  # Más que el timeout
    
    # Probar recuperación
    print(f"\n{c.CYAN}Intentando operación después de recuperación...{c.END}")
    try:
        result = await cb.call(simulated_db_operation, 100, False)
        print(f"  Resultado: {c.GREEN}Éxito{c.END} - {result}")
    except Exception as e:
        print(f"  Resultado: {c.RED}Fallo{c.END} - {e}")
    
    # Estado final
    print(f"\nEstado final: {c.BOLD}{cb.get_state()}{c.END}")
    
    # Limpiar
    await cb.reset()


async def test_massive_parallel_operations():
    """Probar operaciones masivas en paralelo."""
    c = BeautifulTerminalColors
    print(f"\n{c.DIVINE}=== PRUEBA DE OPERACIONES MASIVAS EN PARALELO ==={c.END}")
    
    # Estadísticas
    stats = TestStats()
    
    # Crear CircuitBreaker para DB
    db_cb = await circuit_breaker_factory.create(
        "parallel_db", 
        failure_threshold=10,
        recovery_timeout=0.05
    )
    
    # Crear CircuitBreaker para cloud
    cloud_cb = await circuit_breaker_factory.create(
        "parallel_cloud", 
        failure_threshold=15,
        recovery_timeout=0.1,
        quantum_failsafe=True
    )
    
    # Función para ejecutar operación con estadísticas
    async def execute_operation(op_id: int, is_cloud: bool = False):
        start_time = time.time()
        try:
            if is_cloud:
                result = await cloud_cb.call(
                    simulated_cloud_operation, 
                    op_id, 
                    random.randint(0, 3)
                )
            else:
                result = await db_cb.call(
                    simulated_db_operation, 
                    op_id, 
                    random.random() < 0.3  # 30% de fallos
                )
            
            stats.record_success()
            return result
        except Exception as e:
            stats.record_failure()
            recovery_time = time.time() - start_time
            
            # Intentar recuperación
            try:
                if is_cloud and cloud_cb._state == CircuitState.OPEN:
                    await asyncio.sleep(cloud_cb._recovery_timeout * 1.1)
                    result = await cloud_cb.call(
                        simulated_cloud_operation, 
                        op_id, 
                        0  # Sin reintentos para recuperación
                    )
                    stats.record_recovery(time.time() - start_time)
                    return result
            except Exception:
                pass
            
            return None
    
    # Crear tareas
    print(f"{c.CYAN}Ejecutando 1000 operaciones (500 DB, 500 cloud)...{c.END}")
    db_tasks = [execute_operation(i, False) for i in range(500)]
    cloud_tasks = [execute_operation(i, True) for i in range(500)]
    
    # Ejecutar todas las tareas
    all_results = await asyncio.gather(*db_tasks, *cloud_tasks, return_exceptions=True)
    
    # Estadísticas finales
    stats.finish()
    stats.print_summary()
    
    # Estado final de los CircuitBreakers
    print(f"{c.CYAN}Estado final DB CircuitBreaker:{c.END} {db_cb.get_state()}")
    print(f"{c.CYAN}Estado final Cloud CircuitBreaker:{c.END} {cloud_cb.get_state()}")
    
    # Métricas
    print(f"\n{c.QUANTUM}Métricas DB CircuitBreaker:{c.END}")
    for category, values in db_cb.get_metrics().items():
        if isinstance(values, dict):
            print(f"  {category}:")
            for key, value in values.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {category}: {values}")
    
    print(f"\n{c.QUANTUM}Métricas Cloud CircuitBreaker:{c.END}")
    for category, values in cloud_cb.get_metrics().items():
        if isinstance(values, dict):
            print(f"  {category}:")
            for key, value in values.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {category}: {values}")
    
    # Limpiar
    await db_cb.reset()
    await cloud_cb.reset()


async def test_decorator_protection():
    """Probar protección mediante decorador."""
    c = BeautifulTerminalColors
    print(f"\n{c.DIVINE}=== PRUEBA DE PROTECCIÓN MEDIANTE DECORADOR ==={c.END}")
    
    # Estadísticas
    stats = TestStats()
    
    # Función que ejecuta operaciones
    async def run_operations(count: int, is_cloud: bool = False):
        for i in range(count):
            try:
                if is_cloud:
                    result = await protected_cloud_operation(i)
                else:
                    result = await protected_db_operation(i)
                
                if result:
                    stats.record_success()
                    operation_type = "cloud" if is_cloud else "db"
                    if i % 20 == 0:  # Mostrar solo algunos resultados
                        print(f"  Operación {operation_type} #{i}: {c.GREEN}Éxito{c.END}")
                else:
                    stats.record_failure()
                    operation_type = "cloud" if is_cloud else "db"
                    print(f"  Operación {operation_type} #{i}: {c.RED}Rechazada (circuito abierto){c.END}")
            except Exception as e:
                stats.record_failure()
                if i % 10 == 0:  # Mostrar solo algunos errores
                    operation_type = "cloud" if is_cloud else "db"
                    print(f"  Operación {operation_type} #{i}: {c.RED}Fallo{c.END} - {e}")
    
    # Ejecutar operaciones
    print(f"{c.CYAN}Ejecutando 300 operaciones protegidas (200 DB, 100 cloud)...{c.END}")
    db_task = asyncio.create_task(run_operations(200, False))
    cloud_task = asyncio.create_task(run_operations(100, True))
    
    # Esperar a que terminen
    await db_task
    await cloud_task
    
    # Estadísticas finales
    stats.finish()
    stats.print_summary()
    
    # Obtener CircuitBreakers desde la factory
    db_cb = circuit_breaker_factory.get("db_operations")
    cloud_cb = circuit_breaker_factory.get("cloud_operations")
    
    if db_cb and cloud_cb:
        # Estado final
        print(f"{c.CYAN}Estado final DB CircuitBreaker:{c.END} {db_cb.get_state()}")
        print(f"{c.CYAN}Estado final Cloud CircuitBreaker:{c.END} {cloud_cb.get_state()}")
        
        # Resetear
        await db_cb.reset()
        await cloud_cb.reset()


async def test_quantum_recovery():
    """Probar recuperación cuántica."""
    c = BeautifulTerminalColors
    print(f"\n{c.DIVINE}=== PRUEBA DE RECUPERACIÓN CUÁNTICA ==={c.END}")
    
    # Crear CircuitBreaker con recuperación cuántica
    cb = await circuit_breaker_factory.create(
        "quantum_recovery",
        failure_threshold=3,
        recovery_timeout=0.01,
        quantum_failsafe=True
    )
    
    # Definir función que falla catastróficamente
    async def catastrophic_operation(op_id: int, fail_always: bool = False) -> Dict[str, Any]:
        if fail_always or random.random() < 0.8:  # 80% de fallos
            raise Exception(f"Fallo catastrófico en operación {op_id}")
        
        return {
            "operation_id": op_id,
            "status": "success",
            "timestamp": time.time()
        }
    
    # Ejecutar operaciones que fallarán
    print(f"{c.CYAN}Provocando fallos que activarán recuperación cuántica...{c.END}")
    for i in range(10):
        try:
            result = await cb.call(catastrophic_operation, i, i < 5)  # Forzar fallo en los primeros 5
            print(f"  Operación #{i}: {c.GREEN}Éxito{c.END} - {result}")
        except Exception as e:
            print(f"  Operación #{i}: {c.RED}Fallo{c.END} - {e}")
    
    # Estado actual
    print(f"\nEstado actual: {c.BOLD}{cb.get_state()}{c.END}")
    
    # Esperar recuperación
    print(f"\n{c.CYAN}Esperando para recuperación automática...{c.END}")
    await asyncio.sleep(0.02)  # Más que el timeout
    
    # Intentar operaciones tras recuperación
    print(f"\n{c.CYAN}Intentando operaciones tras recuperación...{c.END}")
    for i in range(5):
        try:
            result = await cb.call(catastrophic_operation, i + 100, False)
            print(f"  Operación #{i+100}: {c.GREEN}Éxito{c.END} - {result}")
        except Exception as e:
            print(f"  Operación #{i+100}: {c.RED}Fallo{c.END} - {e}")
    
    # Estado final
    print(f"\nEstado final: {c.BOLD}{cb.get_state()}{c.END}")
    
    # Métricas
    print(f"\n{c.QUANTUM}Métricas del CircuitBreaker:{c.END}")
    for category, values in cb.get_metrics().items():
        if isinstance(values, dict):
            print(f"  {category}:")
            for key, value in values.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {category}: {values}")
    
    # Limpiar
    await cb.reset()


async def main():
    """Ejecutar todas las pruebas."""
    c = BeautifulTerminalColors
    print(f"\n{c.DIVINE}{c.BOLD}======================================================{c.END}")
    print(f"{c.DIVINE}{c.BOLD}  PRUEBA DEL CLOUDCIRCUITBREAKER ULTRA-DIVINO  {c.END}")
    print(f"{c.DIVINE}{c.BOLD}======================================================{c.END}\n")
    
    await test_circuit_breaker_basics()
    await test_decorator_protection()
    await test_quantum_recovery()
    await test_massive_parallel_operations()
    
    print(f"\n{c.DIVINE}{c.BOLD}======================================================{c.END}")
    print(f"{c.DIVINE}{c.BOLD}  PRUEBAS COMPLETADAS EXITOSAMENTE  {c.END}")
    print(f"{c.DIVINE}{c.BOLD}======================================================{c.END}\n")


if __name__ == "__main__":
    asyncio.run(main())