"""
Implementación del patrón Circuit Breaker para el sistema Genesis.

Este módulo proporciona un mecanismo de Circuit Breaker que previene
llamadas a servicios fallidos, permitiendo recuperación automática
y degradación controlada del sistema.
"""

import asyncio
import logging
import time
import random
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, TypeVar, Union, List, Tuple
from functools import wraps
from dataclasses import dataclass, field

# Configuración de logging
logger = logging.getLogger("genesis.circuit_breaker")

# Tipo genérico para la función
T = TypeVar('T')

class CircuitState(Enum):
    """Estados posibles del circuit breaker."""
    CLOSED = auto()  # Funcionamiento normal (permite llamadas)
    OPEN = auto()    # Circuito abierto (rechaza llamadas)
    HALF_OPEN = auto()  # Permitiendo pruebas limitadas

@dataclass
class CircuitStats:
    """Estadísticas del circuit breaker."""
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    rejection_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_calls: int = 0
    response_times: List[float] = field(default_factory=list)
    
    def record_success(self, response_time: float) -> None:
        """Registrar una llamada exitosa."""
        self.success_count += 1
        self.total_calls += 1
        self.last_success_time = time.time()
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.response_times.append(response_time)
        
        # Mantener solo las últimas 100 mediciones
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def record_failure(self, is_timeout: bool = False) -> None:
        """Registrar una llamada fallida."""
        self.failure_count += 1
        self.total_calls += 1
        self.last_failure_time = time.time()
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        
        if is_timeout:
            self.timeout_count += 1
    
    def record_rejection(self) -> None:
        """Registrar una llamada rechazada por el circuito abierto."""
        self.rejection_count += 1
        self.total_calls += 1
    
    def get_failure_rate(self) -> float:
        """Calcular la tasa de fallos actual."""
        if self.success_count + self.failure_count == 0:
            return 0.0
        return self.failure_count / (self.success_count + self.failure_count)
    
    def get_avg_response_time(self) -> float:
        """Calcular el tiempo medio de respuesta."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_percentile_response_time(self, percentile: float = 0.95) -> float:
        """Calcular un percentil del tiempo de respuesta."""
        if not self.response_times:
            return 0.0
        
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * percentile)
        return sorted_times[idx]

class CircuitBreaker:
    """
    Implementación del patrón Circuit Breaker.
    
    Esta clase proporciona protección contra fallos en cascada al detectar
    fallos recurrentes y evitar llamadas a servicios degradados.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        success_threshold: int = 2,
        timeout: float = 10.0,
        exclude_exceptions: Optional[List[type]] = None,
        fallback_value: Any = None
    ):
        """
        Inicializar circuit breaker.
        
        Args:
            name: Nombre único para este circuit breaker
            failure_threshold: Número de fallos consecutivos antes de abrir el circuito
            recovery_timeout: Tiempo (segundos) a esperar antes de cambiar a half-open
            half_open_max_calls: Máximo de llamadas en estado half-open
            success_threshold: Número de éxitos consecutivos para cerrar el circuito
            timeout: Tiempo máximo (segundos) para considerar una llamada como timeout
            exclude_exceptions: Excepciones que no cuentan como fallos del circuito
            fallback_value: Valor por defecto a devolver cuando el circuito está abierto
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.exclude_exceptions = exclude_exceptions or []
        self.fallback_value = fallback_value
        
        # Estado inicial
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self.last_state_change_time = time.time()
        
        # Control de concurrencia para estado half-open
        self.half_open_calls = 0
        self._half_open_lock = asyncio.Lock()
        
        logger.info(f"Circuit Breaker '{name}' inicializado en estado {self.state.name}")
    
    async def execute(
        self, 
        func: Callable[..., Any], 
        *args: Any, 
        **kwargs: Any
    ) -> Any:
        """
        Ejecutar una función con protección de circuit breaker.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales para la función
            **kwargs: Argumentos nombrados para la función
            
        Returns:
            El resultado de la función o fallback_value si el circuito está abierto
            
        Raises:
            Exception: Excepciones no excluidas en estado cerrado o half-open
        """
        # Actualizar estado si es necesario
        await self._check_state_transition()
        
        # Si el circuito está abierto, rechazar la llamada
        if self.state == CircuitState.OPEN:
            logger.warning(
                f"Circuit Breaker '{self.name}' abierto. "
                f"Llamada rechazada. Próxima recuperación en "
                f"{self._get_remaining_recovery_time():.1f}s"
            )
            self.stats.record_rejection()
            return self.fallback_value
        
        # Si el circuito está medio abierto, limitar las llamadas
        if self.state == CircuitState.HALF_OPEN:
            async with self._half_open_lock:
                if self.half_open_calls >= self.half_open_max_calls:
                    logger.info(
                        f"Circuit Breaker '{self.name}' en half-open con máximo "
                        f"de {self.half_open_max_calls} llamadas ya en curso. "
                        f"Llamada rechazada."
                    )
                    self.stats.record_rejection()
                    return self.fallback_value
                self.half_open_calls += 1
        
        # Ejecutar la función con medición y manejo de errores
        start_time = time.time()
        try:
            # Si la función es asíncrona, esperarla con timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            else:
                # Función síncrona, ejecutar directamente
                result = func(*args, **kwargs)
            
            # Registrar éxito
            response_time = time.time() - start_time
            self.stats.record_success(response_time)
            
            # Si estamos en half-open y alcanzamos el umbral de éxitos, cerrar el circuito
            if self.state == CircuitState.HALF_OPEN and self.stats.consecutive_successes >= self.success_threshold:
                await self._transition_to_closed()
            
            return result
            
        except asyncio.TimeoutError:
            # Timeout específico
            self._handle_failure(True)
            raise
            
        except Exception as e:
            # Verificar si es una excepción excluida
            if any(isinstance(e, exc_type) for exc_type in self.exclude_exceptions):
                # Las excepciones excluidas no cuentan como fallos del circuito
                logger.debug(f"Circuit Breaker '{self.name}' excepción excluida: {type(e).__name__}")
                raise
            
            # Manejar como fallo del circuito
            self._handle_failure()
            raise
            
        finally:
            # Liberar contador en half-open
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls = max(0, self.half_open_calls - 1)
    
    def _handle_failure(self, is_timeout: bool = False) -> None:
        """Manejar un fallo y actualizar estado si es necesario."""
        # Registrar fallo
        self.stats.record_failure(is_timeout)
        
        # Si excedemos el umbral en estado cerrado, abrir el circuito
        if (self.state == CircuitState.CLOSED and 
            self.stats.consecutive_failures >= self.failure_threshold):
            asyncio.create_task(self._transition_to_open())
        
        # Si fallamos en estado half-open, volver a abrir
        elif self.state == CircuitState.HALF_OPEN:
            asyncio.create_task(self._transition_to_open())
    
    async def _check_state_transition(self) -> None:
        """Verificar si es necesario cambiar el estado del circuito."""
        # Si el circuito está abierto pero ha pasado el tiempo de recuperación
        if (self.state == CircuitState.OPEN and 
            time.time() - self.last_state_change_time >= self.recovery_timeout):
            await self._transition_to_half_open()
    
    async def _transition_to_open(self) -> None:
        """Transición a estado abierto."""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.last_state_change_time = time.time()
            logger.warning(
                f"Circuit Breaker '{self.name}' transición a OPEN "
                f"tras {self.stats.consecutive_failures} fallos consecutivos"
            )
            # Emitir evento asíncrono de cambio de estado
            asyncio.create_task(self._notify_state_change())
    
    async def _transition_to_half_open(self) -> None:
        """Transición a estado medio abierto."""
        if self.state != CircuitState.HALF_OPEN:
            self.state = CircuitState.HALF_OPEN
            self.half_open_calls = 0
            self.last_state_change_time = time.time()
            logger.info(
                f"Circuit Breaker '{self.name}' transición a HALF_OPEN "
                f"tras {self.recovery_timeout}s en estado OPEN"
            )
            # Emitir evento asíncrono de cambio de estado
            asyncio.create_task(self._notify_state_change())
    
    async def _transition_to_closed(self) -> None:
        """Transición a estado cerrado."""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.last_state_change_time = time.time()
            logger.info(
                f"Circuit Breaker '{self.name}' transición a CLOSED "
                f"tras {self.stats.consecutive_successes} éxitos consecutivos"
            )
            # Emitir evento asíncrono de cambio de estado
            asyncio.create_task(self._notify_state_change())
    
    def _get_remaining_recovery_time(self) -> float:
        """Calcular tiempo restante para recuperación en estado abierto."""
        if self.state != CircuitState.OPEN:
            return 0.0
        
        elapsed = time.time() - self.last_state_change_time
        remaining = max(0.0, self.recovery_timeout - elapsed)
        return remaining
    
    async def _notify_state_change(self) -> None:
        """Notificar cambio de estado (para ser extendido si es necesario)."""
        # Por defecto, solo registra. Puede ser sobrescrito para notificaciones.
        pass
    
    def reset(self) -> None:
        """Resetear el circuit breaker a estado inicial cerrado."""
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self.last_state_change_time = time.time()
        self.half_open_calls = 0
        logger.info(f"Circuit Breaker '{self.name}' reseteado a estado CLOSED")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del circuit breaker."""
        metrics = {
            "name": self.name,
            "state": self.state.name,
            "state_duration": time.time() - self.last_state_change_time,
            "failure_rate": self.stats.get_failure_rate(),
            "avg_response_time": self.stats.get_avg_response_time(),
            "p95_response_time": self.stats.get_percentile_response_time(0.95),
            "consecutive_failures": self.stats.consecutive_failures,
            "consecutive_successes": self.stats.consecutive_successes,
            "success_count": self.stats.success_count,
            "failure_count": self.stats.failure_count,
            "timeout_count": self.stats.timeout_count,
            "rejection_count": self.stats.rejection_count,
            "total_calls": self.stats.total_calls
        }
        
        # Agregar tiempo restante si está en estado abierto
        if self.state == CircuitState.OPEN:
            metrics["recovery_remaining"] = self._get_remaining_recovery_time()
        
        return metrics

class CircuitBreakerRegistry:
    """
    Registro global de circuit breakers.
    
    Permite acceder a circuit breakers por nombre y obtener métricas globales.
    """
    
    def __init__(self):
        """Inicializar registro vacío."""
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        **kwargs: Any
    ) -> CircuitBreaker:
        """
        Obtener un circuit breaker existente o crear uno nuevo.
        
        Args:
            name: Nombre único del circuit breaker
            failure_threshold: Fallos consecutivos para abrir
            recovery_timeout: Tiempo hasta half-open
            **kwargs: Parámetros adicionales para CircuitBreaker
            
        Returns:
            Circuit breaker existente o nuevo
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                **kwargs
            )
        
        return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """
        Obtener un circuit breaker por nombre.
        
        Args:
            name: Nombre del circuit breaker
            
        Returns:
            Circuit breaker o None si no existe
        """
        return self._breakers.get(name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener métricas de todos los circuit breakers.
        
        Returns:
            Diccionario de métricas por nombre de circuit breaker
        """
        return {name: breaker.get_metrics() for name, breaker in self._breakers.items()}
    
    def reset_all(self) -> None:
        """Resetear todos los circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

# Decorador para aplicar circuit breaker
def with_circuit_breaker(
    name: str = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    **kwargs: Any
) -> Callable:
    """
    Decorador para aplicar circuit breaker a una función.
    
    Args:
        name: Nombre para el circuit breaker (por defecto: nombre de la función)
        failure_threshold: Número de fallos consecutivos para abrir
        recovery_timeout: Tiempo de recuperación en segundos
        **kwargs: Parámetros adicionales para CircuitBreaker
        
    Returns:
        Función decorada con protección de circuit breaker
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Usar nombre de función si no se proporciona
        breaker_name = name or f"cb_{func.__module__}.{func.__name__}"
        
        # Obtener breaker del registro
        breaker = registry.get_or_create(
            breaker_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            **kwargs
        )
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await breaker.execute(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Para funciones síncronas, ejecutar en el bucle de eventos actual
            return asyncio.get_event_loop().run_until_complete(
                breaker.execute(func, *args, **kwargs)
            )
        
        # Usar el wrapper apropiado según si la función es asíncrona
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        # Exponer el circuit breaker en la función
        wrapper.circuit_breaker = breaker  # type: ignore
        
        return wrapper
    
    return decorator

# Registro global
registry = CircuitBreakerRegistry()

# Ejemplos de uso:
"""
# Como decorador en función asíncrona
@with_circuit_breaker(failure_threshold=3, recovery_timeout=60.0)
async def fetch_external_api(url: str) -> Dict[str, Any]:
    # Implementación...
    pass

# Uso directo
async def process_user_request(user_id: str) -> None:
    breaker = registry.get_or_create("user_service", failure_threshold=5)
    try:
        result = await breaker.execute(user_service.get_user, user_id)
        # Procesar resultado...
    except Exception as e:
        # Manejar error...
        pass
"""