"""
Sistema de Reintentos Adaptativo para Genesis.

Este módulo implementa un avanzado mecanismo de reintentos con backoff exponencial
y jitter para proporcionar un comportamiento adaptativo frente a fallas transitorias.
"""

import asyncio
import logging
import random
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union, List, Tuple, Coroutine
from functools import wraps

# Configuración de logging
logger = logging.getLogger("genesis.retry")

# Tipo genérico para la función
T = TypeVar('T')

class RetryConfig:
    """Configuración para el sistema de reintentos adaptativos."""
    
    def __init__(
        self,
        base_delay: float = 0.1,
        max_delay: float = 10.0,
        max_retries: int = 5,
        jitter_factor: float = 0.1,
        timeout_multiplier: float = 1.5,
        retry_exceptions: Optional[List[type]] = None,
        retry_status_codes: Optional[List[int]] = None
    ):
        """
        Inicializar configuración de reintentos.
        
        Args:
            base_delay: Retraso base inicial en segundos
            max_delay: Retraso máximo en segundos
            max_retries: Número máximo de reintentos
            jitter_factor: Factor de aleatoriedad (0.0-1.0)
            timeout_multiplier: Multiplicador para cada timeout sucesivo
            retry_exceptions: Lista de excepciones que deberían provocar reintento
            retry_status_codes: Códigos de estado que deberían provocar reintento
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.jitter_factor = jitter_factor
        self.timeout_multiplier = timeout_multiplier
        self.retry_exceptions = retry_exceptions or [
            asyncio.TimeoutError, 
            ConnectionError, 
            ConnectionRefusedError,
            ConnectionResetError,
            TimeoutError
        ]
        self.retry_status_codes = retry_status_codes or [408, 429, 500, 502, 503, 504]
        
        # Historial para adaptación
        self.success_history: List[bool] = []
        self.latency_history: List[float] = []
        self.exception_counts: Dict[str, int] = {}
        
    def calculate_delay(self, attempt: int, recent_failure_rate: float = 0.0) -> float:
        """
        Calcular el retraso para el siguiente intento usando backoff exponencial con jitter.
        
        Args:
            attempt: Número de intento (0-indexed)
            recent_failure_rate: Tasa de fallos reciente (0.0-1.0)
        
        Returns:
            Tiempo de espera en segundos
        """
        # Aumentar la base si hay muchos fallos recientes
        adjusted_base = self.base_delay * (1 + recent_failure_rate)
        
        # Cálculo de backoff exponencial
        delay = min(adjusted_base * (2 ** attempt), self.max_delay)
        
        # Agregar jitter para prevenir thundering herd
        jitter_range = delay * self.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        
        return max(0.001, delay + jitter)
    
    def should_retry(self, exception: Exception) -> bool:
        """
        Determinar si se debe reintentar basado en la excepción.
        
        Args:
            exception: La excepción capturada
            
        Returns:
            True si se debe reintentar
        """
        # Verificar si el tipo de excepción está en la lista
        should_retry = any(isinstance(exception, exc_type) for exc_type in self.retry_exceptions)
        
        # Verificar si es una respuesta HTTP con código de estado para reintentar
        if hasattr(exception, 'status') and getattr(exception, 'status') in self.retry_status_codes:
            should_retry = True
            
        # Registrar la excepción para adaptación
        exception_name = type(exception).__name__
        if exception_name not in self.exception_counts:
            self.exception_counts[exception_name] = 0
        self.exception_counts[exception_name] += 1
        
        return should_retry
    
    def record_result(self, success: bool, latency: float = 0.0) -> None:
        """
        Registrar resultado para adaptación.
        
        Args:
            success: Si la operación fue exitosa
            latency: Tiempo de respuesta en segundos
        """
        self.success_history.append(success)
        if latency > 0:
            self.latency_history.append(latency)
            
        # Mantener historial limitado (últimos 100 intentos)
        if len(self.success_history) > 100:
            self.success_history.pop(0)
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)
    
    def get_recent_failure_rate(self) -> float:
        """
        Calcular tasa de fallos reciente.
        
        Returns:
            Tasa de fallos (0.0-1.0)
        """
        if not self.success_history:
            return 0.0
        
        return 1.0 - (sum(self.success_history) / len(self.success_history))
    
    def get_timeout(self, attempt: int) -> float:
        """
        Calcular timeout para el intento actual.
        
        Args:
            attempt: Número de intento (0-indexed)
            
        Returns:
            Timeout en segundos
        """
        # Calcular basado en latencias históricas
        if self.latency_history:
            # P95 de latencias como base
            sorted_latencies = sorted(self.latency_history)
            p95_index = int(0.95 * len(sorted_latencies))
            base_timeout = sorted_latencies[p95_index] * 2  # 2x del P95
        else:
            # Sin historial, usar valor por defecto
            base_timeout = 1.0
        
        # Incrementar timeout con cada intento
        multiplier = self.timeout_multiplier ** attempt
        return base_timeout * multiplier

class AdaptiveRetry:
    """
    Implementación del sistema de reintentos adaptativos.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Inicializar sistema de reintentos.
        
        Args:
            config: Configuración de reintentos o None para usar valores por defecto
        """
        self.config = config or RetryConfig()
        self.stats = {
            "attempts": 0,
            "retries": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0
        }
    
    async def execute(
        self, 
        func: Callable[..., Coroutine[Any, Any, T]], 
        *args: Any, 
        **kwargs: Any
    ) -> T:
        """
        Ejecutar una función con reintentos adaptativos.
        
        Args:
            func: Función asíncrona a ejecutar
            *args: Argumentos posicionales para la función
            **kwargs: Argumentos nombrados para la función
            
        Returns:
            El resultado de la función
            
        Raises:
            Exception: La última excepción si se agotan los reintentos
        """
        attempt = 0
        start_time = time.time()
        last_exception = None
        
        while attempt <= self.config.max_retries:
            self.stats["attempts"] += 1
            
            # Calcular timeout para este intento
            timeout = self.config.get_timeout(attempt)
            
            # Ejecutar con timeout
            func_start_time = time.time()
            try:
                # Usar timeout basado en intentos
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                
                # Registrar éxito
                latency = time.time() - func_start_time
                self.config.record_result(True, latency)
                self.stats["successes"] += 1
                self.stats["total_time"] += time.time() - start_time
                
                return result
                
            except Exception as exc:
                # Registrar fallo
                latency = time.time() - func_start_time
                self.config.record_result(False, latency)
                last_exception = exc
                
                # Verificar si debemos reintentar esta excepción
                if not self.config.should_retry(exc) or attempt >= self.config.max_retries:
                    self.stats["failures"] += 1
                    self.stats["total_time"] += time.time() - start_time
                    raise exc
                
                # Calcular delay con backoff
                failure_rate = self.config.get_recent_failure_rate()
                delay = self.config.calculate_delay(attempt, failure_rate)
                
                logger.warning(
                    f"Reintento {attempt+1}/{self.config.max_retries} "
                    f"en {delay:.2f}s tras error: {type(exc).__name__}: {str(exc)}"
                )
                
                # Esperar antes del siguiente intento
                await asyncio.sleep(delay)
                attempt += 1
                self.stats["retries"] += 1
        
        # Si llegamos aquí, se agotaron los reintentos
        self.stats["failures"] += 1
        self.stats["total_time"] += time.time() - start_time
        
        if last_exception:
            raise last_exception
        
        raise RuntimeError("Se agotaron los reintentos sin una excepción específica")

def with_retry(
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    max_retries: int = 3,
    jitter_factor: float = 0.1
) -> Callable:
    """
    Decorador para aplicar reintentos adaptativos a una función asíncrona.
    
    Args:
        base_delay: Retraso base inicial en segundos
        max_delay: Retraso máximo en segundos
        max_retries: Número máximo de reintentos
        jitter_factor: Factor de aleatoriedad (0.0-1.0)
        
    Returns:
        Decorador configurado
    """
    config = RetryConfig(
        base_delay=base_delay,
        max_delay=max_delay,
        max_retries=max_retries,
        jitter_factor=jitter_factor
    )
    retry = AdaptiveRetry(config)
    
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry.execute(func, *args, **kwargs)
        
        # Exponer configuración y estadísticas
        wrapper.retry_config = config  # type: ignore
        wrapper.retry_stats = retry.stats  # type: ignore
        
        return wrapper
    
    return decorator

# Función de utilidad para retry manual
async def retry_operation(
    operation: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    retry_config: Optional[RetryConfig] = None,
    **kwargs: Any
) -> T:
    """
    Ejecutar una operación con reintentos adaptativos.
    
    Args:
        operation: Función asíncrona a ejecutar
        *args: Argumentos posicionales
        retry_config: Configuración personalizada o None para valores por defecto
        **kwargs: Argumentos nombrados
    
    Returns:
        El resultado de la operación
    """
    retry = AdaptiveRetry(retry_config)
    return await retry.execute(operation, *args, **kwargs)

# Singleton global para uso en todo el sistema
default_retry = AdaptiveRetry()

# Ejemplos de uso:
"""
# Como decorador
@with_retry(max_retries=5, base_delay=0.2)
async def fetch_data(url: str) -> Dict[str, Any]:
    # Implementación...
    pass

# Como función explícita
async def process_item(item_id: str) -> None:
    try:
        result = await retry_operation(api_client.get_item, item_id, 
                                      retry_config=RetryConfig(max_retries=3))
        # Procesar resultado...
    except Exception as e:
        # Manejar error final...
        pass
"""