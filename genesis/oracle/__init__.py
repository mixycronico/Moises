"""
Módulo oracle del Sistema Genesis Ultra-Divino.

Este paquete contiene el Oráculo Cuántico y el Adaptador ARMAGEDÓN,
componentes fundamentales para las capacidades predictivas y de resiliencia 
del Sistema Genesis en su configuración Ultra-Divina.
"""

from .quantum_oracle import QuantumOracle, OracleState, ConfidenceCategory
from .armageddon_adapter import ArmageddonAdapter, ArmageddonPattern, ArmageddonMode

__all__ = [
    'QuantumOracle', 'OracleState', 'ConfidenceCategory',
    'ArmageddonAdapter', 'ArmageddonPattern', 'ArmageddonMode'
]