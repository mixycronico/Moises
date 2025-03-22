"""
Prueba específica para el sistema de reintentos adaptativos con backoff exponencial y jitter.

Este script prueba el comportamiento del sistema de reintentos adaptativos
implementado en genesis/core/retry_adaptive.py, verificando que:
1. El backoff exponencial funcione según la fórmula: nuevo_timeout = base * 2^intento ± random(0, jitter)
2. El sistema respete el número máximo de reintentos
3. El sistema maneje correctamente los timeouts
"""

import asyncio
import logging
import time
import random
import statistics
from typing import List, Dict, Any, Tuple

from genesis.core.retry_adaptive import with_retry, RetryConfig, AdaptiveRetry, retry_operation

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backoff_test")

# Servicio simulado con fallas controladas
class UnstableService:
    """Servicio que falla de manera controlada para pruebas."""
    
    def __init__(self, name: str):
        """
        Inicializar servicio inestable.
        
        Args:
            name: Nombre del servicio
        """
        self.name = name
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.delays: List[float] = []
        self.fail_after = -1  # Fallar después de N llamadas
        self.always_fail = False
        self.timeout_seconds = 0.0  # Simular timeout
    
    async def configure(self, fail_after: int = -1, always_fail: bool = False, timeout_seconds: float = 0.0) -> None:
        """
        Configurar comportamiento del servicio.
        
        Args:
            fail_after: Fallar después de N llamadas exitosas, -1 para no fallar
            always_fail: Si True, siempre falla
            timeout_seconds: Tiempo a esperar antes de responder (para simular timeouts)
        """
        self.fail_after = fail_after
        self.always_fail = always_fail
        self.timeout_seconds = timeout_seconds
    
    async def reset_stats(self) -> None:
        """Resetear estadísticas."""
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.delays.clear()
    
    async def call(self, operation: str, request_id: str) -> Dict[str, Any]:
        """
        Llamada al servicio.
        
        Args:
            operation: Operación solicitada
            request_id: ID de la solicitud
            
        Returns:
            Respuesta del servicio
            
        Raises:
            Exception: Si se configura para fallar
            asyncio.TimeoutError: Si se configura un timeout
        """
        start_time = time.time()
        self.call_count += 1
        
        # Registrar tiempo de inicio para calcular retrasos entre llamadas
        if hasattr(self, "last_call_time") and self.last_call_time > 0:
            delay = start_time - self.last_call_time
            self.delays.append(delay)
            
        self.last_call_time = start_time
        
        # Simular timeout
        if self.timeout_seconds > 0:
            await asyncio.sleep(self.timeout_seconds)
            
        # Simular fallo
        should_fail = (
            self.always_fail or 
            (self.fail_after > 0 and self.success_count >= self.fail_after)
        )
        
        if should_fail:
            self.failure_count += 1
            raise Exception(f"Error simulado en {self.name}: {operation}")
        
        # Simular operación exitosa
        await asyncio.sleep(0.05)  # Tiempo base de procesamiento
        self.success_count += 1
        
        return {
            "service": self.name,
            "operation": operation,
            "request_id": request_id,
            "timestamp": time.time(),
            "success": True
        }

# Clase de cliente para acceder al servicio con reintentos
class ServiceClient:
    """Cliente para acceder al servicio inestable con reintentos."""
    
    def __init__(self, service: UnstableService):
        """
        Inicializar cliente.
        
        Args:
            service: Instancia del servicio inestable
        """
        self.service = service
        self.retry_stats: Dict[str, Any] = {
            "retries": 0,
            "failures": 0,
            "successes": 0
        }
    
    @with_retry(base_delay=0.1, max_retries=5, jitter_factor=0.2)
    async def call_with_retry(self, operation: str, request_id: str) -> Dict[str, Any]:
        """
        Llamar al servicio con retry vía decorador.
        
        Args:
            operation: Operación solicitada
            request_id: ID de la solicitud
            
        Returns:
            Respuesta del servicio
        """
        return await self.service.call(operation, request_id)
    
    async def call_with_manual_retry(self, operation: str, request_id: str) -> Dict[str, Any]:
        """
        Llamar al servicio con retry vía función.
        
        Args:
            operation: Operación solicitada
            request_id: ID de la solicitud
            
        Returns:
            Respuesta del servicio
        """
        # Configurar reintentos personalizados
        retry_config = RetryConfig(
            base_delay=0.1,
            max_delay=2.0,
            max_retries=5,
            jitter_factor=0.2
        )
        
        try:
            # Ejecutar con reintentos
            result = await retry_operation(
                self.service.call,
                operation,
                request_id,
                retry_config=retry_config
            )
            self.retry_stats["successes"] += 1
            return result
        except Exception as e:
            self.retry_stats["failures"] += 1
            raise e

# Pruebas para verificar backoff exponencial
async def test_backoff_exponential() -> Dict[str, Any]:
    """
    Probar el backoff exponencial.
    
    Returns:
        Resultados de las pruebas
    """
    results = {
        "tests": 0,
        "passed": 0,
        "backoff_verification": False,
        "max_retries_respected": False,
        "timeout_handling": False,
        "delays": [],
        "theoretical_delays": []
    }
    
    # Crear servicio y cliente
    service = UnstableService("test_service")
    client = ServiceClient(service)
    
    # TEST 1: Verificar que el backoff exponencial funciona según la fórmula
    logger.info("=== Prueba 1: Verificación de backoff exponencial ===")
    
    # Configurar servicio para fallar siempre
    await service.configure(always_fail=True)
    await service.reset_stats()
    
    # Intentar llamada que siempre fallará, para medir los retrasos entre reintentos
    try:
        results["tests"] += 1
        await client.call_with_retry("test_operation", "request_1")
    except Exception:
        # Esperado, verificar delays
        if len(service.delays) > 3:
            # Guardar delays medidos
            results["delays"] = service.delays.copy()
            
            # Calcular delays teóricos según la fórmula
            base_delay = 0.1
            theoretical_delays = []
            for i in range(1, len(service.delays)):
                # Sin jitter para comparación: base * 2^(intento-1)
                delay = base_delay * (2 ** (i-1))
                theoretical_delays.append(delay)
            
            results["theoretical_delays"] = theoretical_delays
            
            # Verificar que los delays siguen aproximadamente la curva exponencial
            # Permitimos cierta variación debido al jitter
            valid_pattern = True
            for i in range(1, min(len(service.delays), len(theoretical_delays))):
                measured = service.delays[i]
                theoretical = theoretical_delays[i-1]
                
                # El delay debe estar en el rango de ±50% del teórico debido al jitter
                lower_bound = theoretical * 0.5
                upper_bound = theoretical * 1.5
                
                if not (lower_bound <= measured <= upper_bound):
                    logger.warning(f"Retry {i}: Delay {measured:.3f}s fuera del rango esperado [{lower_bound:.3f}s, {upper_bound:.3f}s]")
                    valid_pattern = False
            
            if valid_pattern:
                logger.info("✓ Los delays siguen el patrón de backoff exponencial con jitter")
                results["backoff_verification"] = True
                results["passed"] += 1
            else:
                logger.error("✗ Los delays no siguen el patrón esperado")
    
    # TEST 2: Verificar respeto al número máximo de reintentos
    logger.info("=== Prueba 2: Respeto al número máximo de reintentos ===")
    
    # Resetear servicio
    await service.reset_stats()
    
    try:
        results["tests"] += 1
        await client.call_with_retry("test_operation", "request_2")
    except Exception:
        # Verificar que el número de llamadas sea igual a max_retries + 1 (llamada original)
        expected_calls = 5 + 1  # max_retries + 1
        if service.call_count == expected_calls:
            logger.info(f"✓ Se respeta el máximo de reintentos: {expected_calls-1}")
            results["max_retries_respected"] = True
            results["passed"] += 1
        else:
            logger.error(f"✗ Número de reintentos incorrecto: {service.call_count-1}, esperado: {expected_calls-1}")
    
    # TEST 3: Verificar manejo de timeouts
    logger.info("=== Prueba 3: Manejo de timeouts ===")
    
    # Configurar servicio para timeout
    await service.configure(always_fail=False, timeout_seconds=0.5)
    await service.reset_stats()
    
    # Configurar retry con timeout menor
    original_call_with_retry = client.call_with_retry
    
    # Reemplazar con una versión con timeout específico
    client.call_with_retry = with_retry(base_delay=0.1, max_retries=3, jitter_factor=0.2)(
        lambda operation, request_id: asyncio.wait_for(
            service.call(operation, request_id), 
            timeout=0.2  # Timeout menor que el servicio (0.5s)
        )
    )
    
    try:
        results["tests"] += 1
        await client.call_with_retry("test_operation", "request_3")
    except asyncio.TimeoutError:
        # Esperado, verificar reintentos
        if service.call_count > 1:
            logger.info(f"✓ Se realizaron reintentos tras timeouts: {service.call_count}")
            results["timeout_handling"] = True
            results["passed"] += 1
        else:
            logger.error("✗ No se realizaron reintentos tras timeout")
    except Exception as e:
        logger.error(f"✗ Error inesperado en prueba de timeout: {type(e).__name__}: {e}")
    finally:
        # Restaurar método original
        client.call_with_retry = original_call_with_retry
    
    # Resultado global
    logger.info(f"=== Resultado: {results['passed']}/{results['tests']} pruebas exitosas ===")
    
    return results

async def main():
    """Función principal para ejecutar las pruebas."""
    start_time = time.time()
    
    try:
        results = await test_backoff_exponential()
        
        # Imprimir resultados detallados
        logger.info("\n=== Resultados detallados ===")
        
        # Mostrar resultado global
        success_rate = results["passed"] / results["tests"] * 100 if results["tests"] > 0 else 0
        logger.info(f"Total: {results['passed']}/{results['tests']} pruebas exitosas ({success_rate:.1f}%)")
        
        # Mostrar resultados específicos
        logger.info(f"Verificación de backoff exponencial: {'✓' if results['backoff_verification'] else '✗'}")
        logger.info(f"Respeto a máximo de reintentos: {'✓' if results['max_retries_respected'] else '✗'}")
        logger.info(f"Manejo de timeouts: {'✓' if results['timeout_handling'] else '✗'}")
        
        # Mostrar estadísticas de delays si hay datos
        if results["delays"] and len(results["delays"]) > 1:
            delays = results["delays"][1:]  # Ignorar el primer delay
            theoretical = results["theoretical_delays"]
            
            logger.info("\nEstadísticas de delays:")
            logger.info(f"  Min: {min(delays):.3f}s")
            logger.info(f"  Max: {max(delays):.3f}s")
            logger.info(f"  Avg: {statistics.mean(delays):.3f}s")
            
            logger.info("\nComparación con teóricos (sin jitter):")
            for i in range(min(len(delays), len(theoretical))):
                logger.info(f"  Retry {i+1}: Medido={delays[i]:.3f}s, Teórico={theoretical[i]:.3f}s")
    
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
    
    finally:
        # Tiempo total
        elapsed = time.time() - start_time
        logger.info(f"Tiempo total de ejecución: {elapsed:.2f} segundos")

if __name__ == "__main__":
    asyncio.run(main())