"""
Módulos cloud para el Sistema Genesis Ultra-Divino.

Este paquete contiene los componentes cloud del Sistema Genesis:
- CircuitBreaker: Para prevenir fallos en cascada
- DistributedCheckpoint: Para respaldo y recuperación de estado
- LoadBalancer: Para distribución inteligente de carga
- REST API: Para integración con sistemas externos
"""

from .circuit_breaker import (
    CloudCircuitBreaker, 
    CloudCircuitBreakerFactory, 
    CircuitState,
    circuit_breaker_factory, 
    circuit_protected
)
from .distributed_checkpoint import (
    DistributedCheckpointManager, 
    CheckpointStorageType, 
    CheckpointConsistencyLevel, 
    CheckpointState, 
    CheckpointMetadata,
    checkpoint_manager
)
from .load_balancer import (
    CloudLoadBalancer, 
    CloudLoadBalancerManager, 
    CloudNode,
    BalancerAlgorithm, 
    ScalingPolicy, 
    BalancerState,
    SessionAffinityMode, 
    NodeHealthStatus,
    load_balancer_manager
)
from .rest_api import (
    CloudAPI,
    create_cloud_api,
    UserRole,
    APIUser
)

__all__ = [
    'CloudCircuitBreaker', 
    'CloudCircuitBreakerFactory', 
    'CircuitState',
    'circuit_breaker_factory', 
    'circuit_protected',
    'DistributedCheckpointManager', 
    'CheckpointStorageType', 
    'CheckpointConsistencyLevel', 
    'CheckpointState', 
    'CheckpointMetadata',
    'checkpoint_manager',
    'CloudLoadBalancer', 
    'CloudLoadBalancerManager', 
    'CloudNode',
    'BalancerAlgorithm', 
    'ScalingPolicy', 
    'BalancerState',
    'SessionAffinityMode', 
    'NodeHealthStatus',
    'load_balancer_manager',
    'CloudAPI',
    'create_cloud_api',
    'UserRole',
    'APIUser'
]