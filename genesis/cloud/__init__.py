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

from .distributed_checkpoint import (
    DistributedCheckpointManager, CheckpointStorageType, 
    CheckpointConsistencyLevel, CheckpointState, CheckpointMetadata,
    CheckpointStorageProvider, LocalFileStorageProvider, 
    PostgreSQLStorageProvider, MemoryStorageProvider, HybridStorageProvider,
    checkpoint_manager, checkpoint_state
)

from .load_balancer import (
    CloudLoadBalancer, CloudLoadBalancerManager, CloudNode,
    BalancerAlgorithm, ScalingPolicy, BalancerState,
    SessionAffinityMode, NodeHealthStatus,
    load_balancer_manager, distributed, distributed_with_circuit_breaker,
    distributed_with_checkpoint, ultra_resilient
)

__all__ = [
    # Circuit Breaker
    'CloudCircuitBreaker', 'CloudCircuitBreakerFactory', 'CircuitState',
    'circuit_breaker_factory', 'circuit_protected',
    
    # Distributed Checkpoint Manager
    'DistributedCheckpointManager', 'CheckpointStorageType', 
    'CheckpointConsistencyLevel', 'CheckpointState', 'CheckpointMetadata',
    'CheckpointStorageProvider', 'LocalFileStorageProvider', 
    'PostgreSQLStorageProvider', 'MemoryStorageProvider', 'HybridStorageProvider',
    'checkpoint_manager', 'checkpoint_state',
    
    # Cloud Load Balancer
    'CloudLoadBalancer', 'CloudLoadBalancerManager', 'CloudNode',
    'BalancerAlgorithm', 'ScalingPolicy', 'BalancerState',
    'SessionAffinityMode', 'NodeHealthStatus',
    'load_balancer_manager', 'distributed', 'distributed_with_circuit_breaker',
    'distributed_with_checkpoint', 'ultra_resilient'
]