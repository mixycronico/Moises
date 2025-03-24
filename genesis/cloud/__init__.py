"""
Módulo cloud del Sistema Genesis Ultra-Divino.

Este paquete contiene componentes adaptados para entornos cloud,
permitiendo una arquitectura híbrida que puede funcionar tanto
localmente como en servicios serverless.
"""

from .circuit_breaker import (
    CloudCircuitBreaker, CloudCircuitBreakerFactory, CircuitState,
    circuit_breaker_factory, circuit_protected
)

__all__ = [
    'CloudCircuitBreaker', 'CloudCircuitBreakerFactory', 'CircuitState',
    'circuit_breaker_factory', 'circuit_protected'
]