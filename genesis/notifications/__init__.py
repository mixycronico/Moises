"""
Módulo de notificaciones para el sistema Genesis.

Este módulo proporciona funcionalidades para enviar notificaciones
a través de diferentes canales como email, SMS, etc.
"""

from genesis.notifications.base import NotificationChannel
from genesis.notifications.alert_manager import AlertCondition, AlertManager
from genesis.notifications.email_notifier import EmailNotifier
from genesis.notifications.initializer import (
    initialize_notifications, 
    send_test_notification,
    email_notifier,
    alert_manager
)

__all__ = [
    "NotificationChannel", 
    "AlertCondition", 
    "AlertManager", 
    "EmailNotifier",
    "initialize_notifications",
    "send_test_notification",
    "email_notifier",
    "alert_manager"
]