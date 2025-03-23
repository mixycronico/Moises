"""
Prueba mínima de las características de resiliencia del sistema Genesis.

Esta es una versión ultra simplificada que demuestra los conceptos clave:
1. Sistema de Reintentos Adaptativos 
2. Circuit Breaker
3. Checkpointing
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_resiliencia_minimo")

# =================== SISTEMA DE REINTENTOS ADAPTATIVOS ===================

async def with_retry(func, max_retries=3, base_delay=0.1):
    """
    Ejecutar una función con reintentos adaptativos.
    
    Args:
        func: Función a ejecutar
        max_retries: Número máximo de reintentos
        base_delay: Tiempo base de espera
    
    Returns:
        Resultado de la función
    
    Raises:
        Exception: Si se agotan los reintentos
    """
    attempt = 0
    while True:
        try:
            start_time = time.time()
            result = await func()
            elapsed = time.time() - start_time
            
            # Éxito
            if attempt > 0:
                logger.info(f"Intento {attempt+1}: Éxito en {elapsed:.3f}s después de {attempt} reintentos")
            return result
            
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(f"Todos los reintentos agotados ({max_retries}). Último error: {e}")
                raise
            
            # Backoff exponencial con jitter
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
            logger.info(f"Intento {attempt}: Fallo - {e}. Reintentando en {delay:.3f}s")
            await asyncio.sleep(delay)

# =================== CIRCUIT BREAKER ===================

class CircuitBreaker:
    """Implementación simple del Circuit Breaker."""
    
    def __init__(self, name, failure_threshold=3, recovery_timeout=5.0):
        self.name = name
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0
        
        logger.info(f"Circuit Breaker '{name}' creado en estado {self.state}")
    
    async def execute(self, func):
        """Ejecutar función con protección del Circuit Breaker."""
        # Si el circuito está abierto, verificar si es tiempo de recuperación
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info(f"Circuit Breaker '{self.name}' cambiando de OPEN a HALF_OPEN")
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                logger.warning(f"Circuit Breaker '{self.name}' abierto. Llamada rechazada.")
                return None
        
        try:
            # Ejecutar la función
            result = await func()
            
            # Registrar éxito
            self.success_count += 1
            self.failure_count = 0
            
            # Si estamos en HALF_OPEN y tenemos éxito, cerrar el circuito
            if self.state == "HALF_OPEN" and self.success_count >= 2:
                logger.info(f"Circuit Breaker '{self.name}' cambiando de HALF_OPEN a CLOSED")
                self.state = "CLOSED"
            
            return result
            
        except Exception as e:
            # Registrar fallo
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = time.time()
            
            # Si excedemos el umbral de fallos, abrir el circuito
            if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit Breaker '{self.name}' cambiando a OPEN tras {self.failure_count} fallos")
                self.state = "OPEN"
            
            # Si estamos en HALF_OPEN y fallamos, volver a abrir
            elif self.state == "HALF_OPEN":
                logger.warning(f"Circuit Breaker '{self.name}' volviendo a OPEN tras fallo en HALF_OPEN")
                self.state = "OPEN"
                
            # Propagar el error
            raise e

# =================== COMPONENTE CON CHECKPOINTING ===================

class SimpleComponent:
    """Componente simple con checkpointing."""
    
    def __init__(self, name, fail_rate=0.0):
        self.name = name
        self.fail_rate = fail_rate
        self.data = {}
        self.checkpoint = {}
        self.call_count = 0
        
        # Circuit breaker incorporado
        self.circuit_breaker = CircuitBreaker(name=f"cb_{name}")
        
        logger.info(f"Componente '{name}' creado")
    
    async def set_fail_rate(self, rate):
        """Establecer tasa de fallos."""
        self.fail_rate = rate
        return {"name": self.name, "fail_rate": self.fail_rate}
    
    async def call(self, operation, param=None):
        """Llamar operación con posibilidad de fallo."""
        self.call_count += 1
        
        # Simular latencia
        await asyncio.sleep(0.05)
        
        # Simular fallo
        if random.random() < self.fail_rate:
            raise Exception(f"Error simulado en {self.name}.{operation}")
        
        if operation == "store":
            if param:
                self.data[param["key"]] = param["value"]
                return {"status": "stored", "key": param["key"]}
        
        elif operation == "retrieve":
            if param and param["key"] in self.data:
                return {"status": "found", "key": param["key"], "value": self.data[param["key"]]}
            return {"status": "not_found", "key": param.get("key")}
        
        elif operation == "crash":
            self.data = {}  # Perder todos los datos
            raise Exception(f"Fallo catastrófico en {self.name}")
        
        return {"status": "ok", "operation": operation}
    
    def save_checkpoint(self):
        """Guardar punto de control."""
        self.checkpoint = {
            "data": self.data.copy(),
            "call_count": self.call_count
        }
        logger.debug(f"Checkpoint guardado para {self.name}")
    
    def restore_from_checkpoint(self):
        """Restaurar desde checkpoint."""
        if self.checkpoint:
            self.data = self.checkpoint.get("data", {}).copy()
            self.call_count = self.checkpoint.get("call_count", 0)
            logger.info(f"Componente {self.name} restaurado desde checkpoint")
            return True
        return False

# =================== PRUEBAS =================== 

async def test_retry_system():
    """Probar sistema de reintentos."""
    logger.info("\n=== Test 1: Sistema de Reintentos ===")
    
    component = SimpleComponent("test_retry")
    await component.set_fail_rate(0.6)  # 60% de fallos
    
    # Usar reintentos
    try:
        result = await with_retry(lambda: component.call("echo", {"message": "Hello"}))
        logger.info(f"Resultado con retry: {result}")
        return True
    except Exception as e:
        logger.error(f"Fallo final: {e}")
        return False

async def test_circuit_breaker():
    """Probar Circuit Breaker."""
    logger.info("\n=== Test 2: Circuit Breaker ===")
    
    component = SimpleComponent("test_circuit")
    # Fallar siempre para garantizar que el circuit breaker se abre
    await component.set_fail_rate(1.0)  # 100% de fallos
    
    results = []
    
    # Varias llamadas que fallarán
    for i in range(5):
        try:
            result = await component.circuit_breaker.execute(
                lambda: component.call("getData", {"id": i})
            )
            results.append({"attempt": i, "success": True, "result": result})
        except Exception as e:
            results.append({"attempt": i, "success": False, "error": str(e)})
        
        logger.info(f"Estado del circuito: {component.circuit_breaker.state}")
    
    # Si llegó a OPEN, el circuito está funcionando
    return component.circuit_breaker.state == "OPEN"

async def test_checkpointing():
    """Probar checkpointing y recuperación."""
    logger.info("\n=== Test 3: Checkpointing ===")
    
    component = SimpleComponent("test_checkpoint")
    
    # Almacenar algunos datos
    await component.call("store", {"key": "name", "value": "Genesis"})
    await component.call("store", {"key": "version", "value": "1.0"})
    
    # Crear checkpoint
    component.save_checkpoint()
    logger.info("Checkpoint creado con 2 valores")
    
    # Verificar datos
    result1 = await component.call("retrieve", {"key": "name"})
    logger.info(f"Valor antes de crash: {result1}")
    
    # Simular crash
    try:
        await component.call("crash")
    except Exception:
        logger.info("Componente sufrió crash simulado")
    
    # Verificar pérdida de datos
    result2 = await component.call("retrieve", {"key": "name"})
    logger.info(f"Valor después de crash: {result2}")
    
    # Restaurar
    component.restore_from_checkpoint()
    
    # Verificar recuperación
    result3 = await component.call("retrieve", {"key": "name"})
    logger.info(f"Valor después de restauración: {result3}")
    
    return result3.get("status") == "found"

async def main():
    """Ejecutar todas las pruebas."""
    logger.info("=== Prueba Mínima de Resiliencia Genesis ===")
    
    results = {}
    
    # Test 1: Sistema de Reintentos
    results["retry"] = await test_retry_system()
    
    # Test 2: Circuit Breaker
    results["circuit_breaker"] = await test_circuit_breaker()
    
    # Test 3: Checkpointing
    results["checkpointing"] = await test_checkpointing()
    
    # Resumen
    logger.info("\n=== Resumen de Pruebas ===")
    for name, result in results.items():
        status = "✓ ÉXITO" if result else "✗ ERROR"
        logger.info(f"Test {name}: {status}")
    
    success_rate = sum(1 for r in results.values() if r) / len(results) * 100
    logger.info(f"\nTasa de éxito: {success_rate:.1f}%")
    
    # Conclusiones
    logger.info("\n=== Conclusiones ===")
    logger.info("La prueba mínima demuestra las tres características principales de resiliencia:")
    logger.info("1. Sistema de Reintentos: Permite recuperación ante fallos transitorios")
    logger.info("2. Circuit Breaker: Aísla componentes fallidos y protege el sistema")
    logger.info("3. Checkpointing: Facilita la recuperación rápida tras fallos")

if __name__ == "__main__":
    asyncio.run(main())