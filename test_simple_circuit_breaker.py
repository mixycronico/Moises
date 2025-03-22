"""
Prueba simple del patrón Circuit Breaker.

Esta es una versión simplificada para demostrar el funcionamiento básico
del patrón Circuit Breaker implementado en el sistema Genesis.
"""

import asyncio
import logging
import time
import random
from enum import Enum, auto

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simple_circuit_breaker_test")

# Definir estados del Circuit Breaker
class CircuitState(Enum):
    """Estados del Circuit Breaker."""
    CLOSED = auto()    # Funcionamiento normal
    OPEN = auto()      # Circuito abierto, rechazan llamadas
    HALF_OPEN = auto() # Semi-abierto, permite algunas llamadas

class SimpleCircuitBreaker:
    """Implementación simplificada de Circuit Breaker."""
    
    def __init__(
        self, 
        name: str, 
        failure_threshold: int = 3,
        recovery_timeout: float = 5.0,
        success_threshold: int = 2
    ):
        """
        Inicializar Circuit Breaker.
        
        Args:
            name: Nombre del circuit breaker
            failure_threshold: Fallos consecutivos para abrir el circuito
            recovery_timeout: Tiempo hasta probar recuperación
            success_threshold: Éxitos consecutivos para cerrar el circuito
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        # Estado
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_state_change = time.time()
        
        logger.info(f"Circuit Breaker '{name}' creado en estado {self.state.name}")
    
    async def execute(self, func, *args, **kwargs):
        """
        Ejecutar función con protección del Circuit Breaker.
        
        Args:
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función o None si el circuito está abierto
            
        Raises:
            Exception: Si ocurre un error y el circuito no está abierto
        """
        # Verificar transición automática de OPEN a HALF_OPEN
        if self.state == CircuitState.OPEN:
            if (time.time() - self.last_state_change) > self.recovery_timeout:
                logger.info(f"Circuit Breaker '{self.name}' cambiando de OPEN a HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.last_state_change = time.time()
        
        # Si el circuito está abierto, rechazar la llamada
        if self.state == CircuitState.OPEN:
            logger.warning(f"Circuit Breaker '{self.name}' abierto. Llamada rechazada.")
            return None
        
        # Si estamos en HALF_OPEN, permitir solo llamada de prueba
        # (En una implementación real, limitaríamos la cantidad de llamadas)
        
        try:
            # Ejecutar la función
            result = await func(*args, **kwargs)
            
            # Registrar éxito
            self.success_count += 1
            self.failure_count = 0
            
            # Si estamos en HALF_OPEN y alcanzamos el umbral de éxitos, cerrar el circuito
            if (self.state == CircuitState.HALF_OPEN and 
                self.success_count >= self.success_threshold):
                logger.info(f"Circuit Breaker '{self.name}' cambiando de HALF_OPEN a CLOSED")
                self.state = CircuitState.CLOSED
                self.last_state_change = time.time()
            
            return result
            
        except Exception as e:
            # Registrar fallo
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = time.time()
            
            # Si excedemos el umbral de fallos, abrir el circuito
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.failure_threshold):
                logger.warning(
                    f"Circuit Breaker '{self.name}' cambiando a OPEN tras "
                    f"{self.failure_count} fallos consecutivos"
                )
                self.state = CircuitState.OPEN
                self.last_state_change = time.time()
            
            # Si estamos en HALF_OPEN y fallamos, volver a abrir el circuito
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit Breaker '{self.name}' volviendo a OPEN tras fallo en HALF_OPEN")
                self.state = CircuitState.OPEN
                self.last_state_change = time.time()
                
            # Propagar el error
            raise e

# Servicio externo simulado
class ExternalService:
    """Servicio externo simulado con comportamiento controlado."""
    
    def __init__(self, name: str):
        """
        Inicializar servicio.
        
        Args:
            name: Nombre del servicio
        """
        self.name = name
        self.healthy = True
        self.call_count = 0
        
    def set_healthy(self, healthy: bool):
        """Establecer si el servicio está saludable."""
        self.healthy = healthy
        logger.info(f"Servicio {self.name} ahora está {'saludable' if healthy else 'degradado'}")
    
    async def call(self, operation: str, param: str = ""):
        """
        Llamar al servicio externo.
        
        Args:
            operation: Operación a realizar
            param: Parámetro opcional
            
        Returns:
            Resultado simulado
            
        Raises:
            Exception: Si el servicio no está saludable
        """
        self.call_count += 1
        
        # Simular latencia
        await asyncio.sleep(0.1)
        
        # Si no está saludable, fallar
        if not self.healthy:
            raise Exception(f"Error en servicio {self.name}: {operation}")
        
        return {
            "service": self.name,
            "operation": operation,
            "param": param,
            "timestamp": time.time()
        }

async def main():
    """Función principal."""
    logger.info("=== Prueba de Circuit Breaker ===")
    
    # Crear servicio y circuit breaker
    service = ExternalService("payment_api")
    breaker = SimpleCircuitBreaker(
        name="payment_circuit",
        failure_threshold=3,
        recovery_timeout=3.0,  # 3 segundos para prueba rápida
        success_threshold=2
    )
    
    # Función para llamar servicio con circuit breaker
    async def call_service(op, param=""):
        try:
            result = await breaker.execute(service.call, op, param)
            if result:
                logger.info(f"Llamada exitosa: {op}, resultado: {result}")
                return True
            else:
                logger.warning(f"Llamada rechazada: {op}")
                return False
        except Exception as e:
            logger.error(f"Error en llamada: {op}, error: {e}")
            return False
    
    # Paso 1: Servicio saludable, llamadas exitosas
    logger.info("\n1. Servicio saludable, llamadas exitosas:")
    for i in range(3):
        await call_service("get_data", f"param_{i}")
    
    # Paso 2: Servicio degradado, circuito se abre
    logger.info("\n2. Servicio degradado, circuito debería abrirse:")
    service.set_healthy(False)
    
    for i in range(5):
        await call_service("process_payment", f"order_{i}")
        logger.info(f"Estado del circuito: {breaker.state.name}, fallos: {breaker.failure_count}")
    
    # Paso 3: Esperar y ver transición a HALF_OPEN
    logger.info("\n3. Esperando timeout de recuperación...")
    await asyncio.sleep(4.0)  # Esperar más que recovery_timeout
    
    logger.info(f"Estado del circuito después de espera: {breaker.state.name}")
    
    # Paso 4: Servicio recuperado, próxima llamada exitosa
    logger.info("\n4. Servicio recuperado, cerrando circuito:")
    service.set_healthy(True)
    
    for i in range(3):
        success = await call_service("verify_account", f"user_{i}")
        logger.info(f"Estado del circuito: {breaker.state.name}, éxitos: {breaker.success_count}")
    
    # Resumen
    logger.info("\n=== Resumen ===")
    logger.info(f"Estado final del circuito: {breaker.state.name}")
    logger.info(f"Total de llamadas al servicio: {service.call_count}")

if __name__ == "__main__":
    asyncio.run(main())