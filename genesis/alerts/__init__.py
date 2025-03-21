"""
Módulo de alertas y notificaciones para el sistema Genesis.

Este módulo proporciona componentes para monitorear métricas clave del sistema
y enviar alertas cuando se alcanzan condiciones específicas.
"""

from .alert_manager import AlertManager

__all__ = ["AlertManager"]