"""
Módulo de compatibilidad para la clase base de estrategias.

Este módulo reexporta la clase Strategy desde base.py como BaseStrategy
para mantener compatibilidad con el código existente.
"""

from genesis.strategies.base import Strategy

# Reexportar Strategy como BaseStrategy para compatibilidad
BaseStrategy = Strategy