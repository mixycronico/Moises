"""
Prueba específica para el patrón Circuit Breaker.

Este script prueba el comportamiento del Circuit Breaker
implementado en genesis/core/circuit_breaker.py, verificando:
1. Transiciones correctas entre estados: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
2. Manejo adecuado de fallos consecutivos
3. Comportamiento de recovery automático
4. Comportamiento en estado HALF_OPEN
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum, auto

from genesis.core.circuit_breaker import (
    CircuitBreaker, CircuitBreakerRegistry, CircuitState, with_circuit_breaker
)

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("circuit_breaker_test")

# Servicio simulado con comportamiento controlado
class ServiceState(Enum):
    """Estados posibles del servicio simulado."""
    HEALTHY = auto()  # Funciona correctamente
    DEGRADED = auto()  # Funciona pero con latencia alta
    FAILING = auto()   # Falla constantemente

class SimulatedService:
    """Servicio simulado para probar el Circuit Breaker."""
    
    def __init__(self, name: str):
        """
        Inicializar servicio.
        
        Args:
            name: Nombre del servicio
        """
        self.name = name
        self.state = ServiceState.HEALTHY
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.response_times: List[float] = []
    
    async def set_state(self, state: ServiceState) -> None:
        """
        Establecer estado del servicio.
        
        Args:
            state: Nuevo estado
        """
        logger.info(f"Servicio {self.name} cambiando a estado: {state.name}")
        self.state = state
    
    async def reset_stats(self) -> None:
        """Resetear estadísticas."""
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.response_times = []
    
    async def call(self, operation: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Llamar al servicio.
        
        Args:
            operation: Operación a realizar
            params: Parámetros de la operación
            
        Returns:
            Resultado de la operación
            
        Raises:
            Exception: Si el servicio está en estado FAILING
            TimeoutError: Si el servicio está en estado DEGRADED y supera el umbral
        """
        params = params or {}
        self.call_count += 1
        start_time = time.time()
        
        try:
            # Simular comportamiento según estado
            if self.state == ServiceState.HEALTHY:
                # Respuesta normal
                await asyncio.sleep(0.1)
                result = {"status": "success", "operation": operation, "params": params}
                self.success_count += 1
                
            elif self.state == ServiceState.DEGRADED:
                # Alta latencia
                delay = random.uniform(0.5, 1.5)
                await asyncio.sleep(delay)
                
                # 30% de probabilidad de timeout
                if random.random() < 0.3:
                    self.failure_count += 1
                    raise asyncio.TimeoutError(f"Timeout en {self.name}: {operation}")
                
                result = {"status": "success", "operation": operation, "params": params}
                self.success_count += 1
                
            else:  # ServiceState.FAILING
                # Siempre falla
                await asyncio.sleep(0.1)
                self.failure_count += 1
                raise Exception(f"Error en {self.name}: {operation}")
                
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            raise e

# Cliente que utiliza el servicio con Circuit Breaker
class ServiceClient:
    """Cliente para acceder al servicio utilizando Circuit Breaker."""
    
    def __init__(self, service: SimulatedService, 
                circuit_name: str = None,
                failure_threshold: int = 3,
                recovery_timeout: float = 5.0,
                half_open_max_calls: int = 1,
                success_threshold: int = 2):
        """
        Inicializar cliente.
        
        Args:
            service: Instancia del servicio
            circuit_name: Nombre para el Circuit Breaker
            failure_threshold: Fallos consecutivos para abrir el circuito
            recovery_timeout: Tiempo hasta half-open en segundos
            half_open_max_calls: Máximo de llamadas en estado half-open
            success_threshold: Éxitos consecutivos para cerrar el circuito
        """
        self.service = service
        self.circuit_name = circuit_name or f"circuit_{service.name}"
        
        # Crear Circuit Breaker
        self.breaker = CircuitBreaker(
            name=self.circuit_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            success_threshold=success_threshold,
            timeout=1.0
        )
        
        # Estadísticas
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.rejections = 0
    
    async def call(self, operation: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Llamar al servicio a través del Circuit Breaker.
        
        Args:
            operation: Operación a realizar
            params: Parámetros de la operación
            
        Returns:
            Resultado de la operación o None si es rechazada
        """
        self.calls += 1
        
        try:
            result = await self.breaker.execute(
                self.service.call, operation, params
            )
            self.successes += 1
            return result
            
        except Exception as e:
            if isinstance(e, asyncio.CancelledError):
                # Rechazado por el circuito
                logger.warning(f"Llamada rechazada por Circuit Breaker: {operation}")
                self.rejections += 1
            else:
                # Fallo real
                logger.error(f"Error en llamada: {type(e).__name__}: {e}")
                self.failures += 1
            
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas combinadas.
        
        Returns:
            Estadísticas del cliente y Circuit Breaker
        """
        breaker_metrics = self.breaker.get_metrics()
        
        return {
            "client": {
                "calls": self.calls,
                "successes": self.successes,
                "failures": self.failures,
                "rejections": self.rejections
            },
            "circuit_breaker": breaker_metrics
        }
    
    def get_state(self) -> CircuitState:
        """
        Obtener estado actual del Circuit Breaker.
        
        Returns:
            Estado actual
        """
        return self.breaker.state

# Pruebas para verificar comportamiento del Circuit Breaker
async def test_circuit_breaker() -> Dict[str, Any]:
    """
    Probar el comportamiento del Circuit Breaker.
    
    Returns:
        Resultados de las pruebas
    """
    results = {
        "tests": 0,
        "passed": 0,
        "state_transitions": {
            "closed_to_open": False,
            "open_to_half_open": False,
            "half_open_to_closed": False,
            "half_open_to_open": False
        },
        "failure_threshold_respected": False,
        "recovery_timeout_respected": False,
        "half_open_limiting": False,
        "success_threshold_respected": False
    }
    
    # Configuración de prueba
    service = SimulatedService("test_service")
    client = ServiceClient(
        service=service,
        failure_threshold=3,
        recovery_timeout=2.0,  # 2 segundos para pruebas rápidas
        half_open_max_calls=1,
        success_threshold=2
    )
    
    # TEST 1: Transición de CLOSED a OPEN
    logger.info("=== Prueba 1: Transición de CLOSED a OPEN ===")
    await service.set_state(ServiceState.FAILING)
    await service.reset_stats()
    
    # Verificar estado inicial
    assert client.get_state() == CircuitState.CLOSED
    
    # Realizar llamadas hasta que se abra el circuito
    results["tests"] += 1
    current_state = client.get_state()
    failure_count = 0
    
    for i in range(10):  # Máximo 10 intentos
        await client.call("test_operation", {"test_id": i})
        
        new_state = client.get_state()
        if new_state == CircuitState.OPEN and current_state == CircuitState.CLOSED:
            logger.info(f"✓ Circuito cambió de CLOSED a OPEN después de {i+1} llamadas")
            results["state_transitions"]["closed_to_open"] = True
            
            # Verificar si respeta el umbral de fallos
            failure_count = service.failure_count
            if failure_count == 3:  # failure_threshold
                logger.info("✓ Respeta el umbral de fallos (3)")
                results["failure_threshold_respected"] = True
            else:
                logger.warning(f"✗ No respeta el umbral de fallos. Esperado: 3, Actual: {failure_count}")
            
            break
            
        current_state = new_state
    
    # Verificar resultado
    if results["state_transitions"]["closed_to_open"]:
        if results["failure_threshold_respected"]:
            results["passed"] += 1
    else:
        logger.error("✗ No se detectó transición de CLOSED a OPEN")
    
    # TEST 2: Verificar que las llamadas son rechazadas en estado OPEN
    logger.info("=== Prueba 2: Rechazo de llamadas en estado OPEN ===")
    
    # Verificar que estamos en OPEN
    if client.get_state() == CircuitState.OPEN:
        results["tests"] += 1
        rejection_detected = False
        
        # Intentar varias llamadas
        for i in range(5):
            prev_rejections = client.rejections
            await client.call("test_open_rejection", {"test_id": i})
            
            if client.rejections > prev_rejections:
                rejection_detected = True
                break
        
        if rejection_detected:
            logger.info("✓ Las llamadas son rechazadas en estado OPEN")
            results["passed"] += 1
        else:
            logger.error("✗ Las llamadas no son rechazadas en estado OPEN")
    
    # TEST 3: Transición de OPEN a HALF_OPEN después del timeout
    logger.info("=== Prueba 3: Transición de OPEN a HALF_OPEN ===")
    
    # Verificar que estamos en OPEN
    if client.get_state() == CircuitState.OPEN:
        results["tests"] += 1
        start_time = time.time()
        
        # Esperar a que ocurra la transición
        timeout_detected = False
        half_open_detected = False
        
        # Esperar un poco más que el recovery_timeout configurado
        await asyncio.sleep(2.5)  # recovery_timeout=2.0
        
        # Verificar si cambió a HALF_OPEN
        if client.get_state() == CircuitState.HALF_OPEN:
            elapsed = time.time() - start_time
            logger.info(f"✓ Circuito cambió de OPEN a HALF_OPEN después de {elapsed:.2f}s")
            
            # Verificar si respeta el recovery timeout
            if 2.0 <= elapsed <= 3.0:  # Un poco de margen
                logger.info("✓ Respeta el timeout de recuperación (~2s)")
                results["recovery_timeout_respected"] = True
            else:
                logger.warning(f"✗ No respeta el timeout. Esperado: ~2s, Actual: {elapsed:.2f}s")
            
            results["state_transitions"]["open_to_half_open"] = True
            
            if results["recovery_timeout_respected"]:
                results["passed"] += 1
        else:
            logger.error("✗ No se detectó transición de OPEN a HALF_OPEN")
    
    # TEST 4: Limitación de llamadas en estado HALF_OPEN
    logger.info("=== Prueba 4: Limitación de llamadas en estado HALF_OPEN ===")
    
    # Verificar que estamos en HALF_OPEN
    if client.get_state() == CircuitState.HALF_OPEN:
        results["tests"] += 1
        
        # Restablecer el servicio para que funcione
        await service.set_state(ServiceState.HEALTHY)
        
        # Hacer varias llamadas simultáneas
        tasks = [client.call("test_half_open", {"test_id": i}) for i in range(3)]
        await asyncio.gather(*tasks)
        
        # Verificar si hubo rechazos (debería haber 2 rechazos, 1 aceptada)
        stats = await client.get_stats()
        
        if stats["client"]["rejections"] > 0:
            logger.info(f"✓ Se limitan las llamadas en HALF_OPEN: {stats['client']['rejections']} rechazos")
            results["half_open_limiting"] = True
            results["passed"] += 1
        else:
            logger.error("✗ No se detectó limitación de llamadas en HALF_OPEN")
    
    # TEST 5: Transición de HALF_OPEN a CLOSED después de éxitos
    logger.info("=== Prueba 5: Transición de HALF_OPEN a CLOSED ===")
    
    # Verificar si estamos en HALF_OPEN
    current_state = client.get_state()
    if current_state == CircuitState.HALF_OPEN:
        results["tests"] += 1
        
        # Hacer varias llamadas secuenciales
        success_counter = 0
        transition_detected = False
        
        for i in range(5):  # Intentar hasta 5 veces
            await client.call("test_success", {"test_id": i})
            
            # Verificar si cambió a CLOSED
            new_state = client.get_state()
            if new_state == CircuitState.CLOSED and current_state == CircuitState.HALF_OPEN:
                logger.info(f"✓ Circuito cambió de HALF_OPEN a CLOSED después de {i+1} llamadas exitosas")
                transition_detected = True
                success_counter = i + 1
                results["state_transitions"]["half_open_to_closed"] = True
                break
                
            current_state = new_state
        
        # Verificar si respeta el umbral de éxitos
        if transition_detected:
            if success_counter == 2:  # success_threshold
                logger.info("✓ Respeta el umbral de éxitos (2)")
                results["success_threshold_respected"] = True
                results["passed"] += 1
            else:
                logger.warning(f"✗ No respeta el umbral de éxitos. Esperado: 2, Actual: {success_counter}")
        else:
            logger.error("✗ No se detectó transición de HALF_OPEN a CLOSED")
    
    # TEST 6: Transición de HALF_OPEN a OPEN si hay fallos
    logger.info("=== Prueba 6: Transición de HALF_OPEN a OPEN en caso de fallos ===")
    
    # Forzar estado HALF_OPEN abriendo el circuito y esperando timeout
    if client.get_state() == CircuitState.CLOSED:
        # Establecer servicio como fallido
        await service.set_state(ServiceState.FAILING)
        
        # Hacer llamadas hasta que se abra
        for i in range(5):
            await client.call("test_operation", {"test_id": i})
            if client.get_state() == CircuitState.OPEN:
                break
        
        # Esperar a que pase a HALF_OPEN
        await asyncio.sleep(2.5)  # recovery_timeout=2.0
    
    # Verificar si estamos en HALF_OPEN
    current_state = client.get_state()
    if current_state == CircuitState.HALF_OPEN:
        results["tests"] += 1
        
        # Configurar servicio como fallido
        await service.set_state(ServiceState.FAILING)
        
        # Hacer una llamada que debería fallar
        await client.call("test_half_open_failure", {})
        
        # Verificar si cambió a OPEN
        if client.get_state() == CircuitState.OPEN:
            logger.info("✓ Circuito cambió de HALF_OPEN a OPEN tras fallo")
            results["state_transitions"]["half_open_to_open"] = True
            results["passed"] += 1
        else:
            logger.error("✗ No se detectó transición de HALF_OPEN a OPEN tras fallo")
    
    # Resumen final
    all_state_transitions = all(results["state_transitions"].values())
    if all_state_transitions:
        logger.info("✓ Todas las transiciones de estado funcionan correctamente")
    
    return results

async def main():
    """Función principal para ejecutar las pruebas."""
    start_time = time.time()
    
    try:
        results = await test_circuit_breaker()
        
        # Imprimir resultados
        logger.info("\n=== Resultados de las pruebas ===")
        
        # Mostrar resultado global
        success_rate = results["passed"] / results["tests"] * 100 if results["tests"] > 0 else 0
        logger.info(f"Total: {results['passed']}/{results['tests']} pruebas exitosas ({success_rate:.1f}%)")
        
        # Mostrar transiciones de estado
        logger.info("\nTransiciones de estado:")
        for transition, success in results["state_transitions"].items():
            logger.info(f"  {transition}: {'✓' if success else '✗'}")
        
        # Mostrar comportamientos específicos
        logger.info("\nComportamientos verificados:")
        logger.info(f"  Umbral de fallos respetado: {'✓' if results['failure_threshold_respected'] else '✗'}")
        logger.info(f"  Timeout de recuperación respetado: {'✓' if results['recovery_timeout_respected'] else '✗'}")
        logger.info(f"  Limitación de llamadas en HALF_OPEN: {'✓' if results['half_open_limiting'] else '✗'}")
        logger.info(f"  Umbral de éxitos respetado: {'✓' if results['success_threshold_respected'] else '✗'}")
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
    
    finally:
        # Tiempo total
        elapsed = time.time() - start_time
        logger.info(f"Tiempo total de ejecución: {elapsed:.2f} segundos")

if __name__ == "__main__":
    asyncio.run(main())