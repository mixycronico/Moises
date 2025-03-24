"""
Gestor de escalabilidad de capital para el Sistema Genesis.

Este módulo proporciona mecanismos avanzados para ajustar la distribución
de capital y parámetros operativos según el nivel de fondos, manteniendo
la eficiencia incluso cuando el capital crece significativamente.
"""

# Importar la clase CapitalScalingManager desde balance_manager
from genesis.accounting.balance_manager import CapitalScalingManager, MetricasInstrumento, Metrics, scaling_metrics

__all__ = [
    'CapitalScalingManager',
    'MetricasInstrumento',
    'Metrics',
    'scaling_metrics',
]