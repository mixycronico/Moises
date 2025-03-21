"""
Módulo de reportes para el sistema Genesis.

Este paquete proporciona funcionalidad para generar informes y reportes
sobre el rendimiento del sistema, transacciones realizadas, y métricas
de estrategias de trading.
"""

from genesis.reporting.report_generator import ReportGenerator
from genesis.reporting.log_integration import LogReportIntegration

__all__ = ["ReportGenerator", "LogReportIntegration"]