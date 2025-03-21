"""
Módulo de análisis de rendimiento y visualización para el sistema Genesis.

Este módulo proporciona herramientas para analizar el rendimiento de 
estrategias de trading, visualizar datos del mercado, y generar informes.
"""

from genesis.analytics.performance import PerformanceTracker
from genesis.analytics.performance_analyzer import PerformanceAnalyzer
from genesis.analytics.reporting import ReportGenerator
from genesis.analytics.visualization import Visualizer

__all__ = [
    "PerformanceTracker", 
    "PerformanceAnalyzer", 
    "ReportGenerator", 
    "Visualizer"
]