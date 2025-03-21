"""
Pruebas unitarias para el monitoreo y alertas del sistema.

Este módulo prueba las funcionalidades relacionadas con el monitoreo
del sistema y la generación de alertas basadas en métricas.
"""

import pytest
from unittest.mock import Mock, patch

from genesis.notifications.alert_manager import AlertManager


# Fixture para AlertManager
@pytest.fixture
def alert_manager():
    """Fixture que proporciona una instancia de AlertManager con notificadores simulados."""
    manager = AlertManager()
    manager.email_notifier = Mock()
    manager.sms_notifier = Mock()
    return manager


# Pruebas para monitoreo de drawdown
def test_alert_manager_drawdown_exceeds_limit(alert_manager):
    """Prueba que se envíe una alerta cuando el drawdown excede el límite."""
    drawdown = 20  # Mayor al límite típico (asumimos 15%)
    alert_manager.check_drawdown(drawdown)
    
    # Verificar que se envió la alerta
    assert alert_manager.email_notifier.send_notification.called or alert_manager.sms_notifier.send_notification.called


def test_alert_manager_drawdown_within_limit(alert_manager):
    """Prueba que no se envíe alerta cuando el drawdown está dentro del límite."""
    drawdown = 10  # Menor al límite típico (asumimos 15%)
    alert_manager.check_drawdown(drawdown)
    
    # Verificar que no se envió ninguna alerta
    assert not alert_manager.email_notifier.send_notification.called
    assert not alert_manager.sms_notifier.send_notification.called


def test_alert_manager_drawdown_at_limit(alert_manager):
    """Prueba el comportamiento cuando el drawdown está exactamente en el límite."""
    # Asumimos que el límite es 15% y que no se dispara si es igual
    drawdown = 15
    alert_manager.check_drawdown(drawdown)
    
    # Verificar que no se envió ninguna alerta (ajustar si la lógica es diferente)
    assert not alert_manager.email_notifier.send_notification.called
    assert not alert_manager.sms_notifier.send_notification.called


def test_alert_manager_drawdown_negative(alert_manager):
    """Prueba el manejo de un drawdown negativo."""
    drawdown = -5  # Valor inválido
    
    with pytest.raises(ValueError, match="Drawdown cannot be negative"):
        alert_manager.check_drawdown(drawdown)
    
    # Verificar que no se envió ninguna alerta
    assert not alert_manager.email_notifier.send_notification.called
    assert not alert_manager.sms_notifier.send_notification.called


def test_alert_manager_drawdown_excessive(alert_manager):
    """Prueba un drawdown extremadamente alto."""
    drawdown = 100  # 100%, pérdida total
    alert_manager.check_drawdown(drawdown)
    
    # Verificar que se envió la alerta
    assert alert_manager.email_notifier.send_notification.called or alert_manager.sms_notifier.send_notification.called


def test_alert_manager_custom_limit(alert_manager):
    """Prueba el comportamiento con un límite de drawdown personalizado."""
    # Establecer un límite personalizado
    alert_manager.drawdown_limit = 25  # 25%
    
    # Drawdown por debajo del límite
    alert_manager.check_drawdown(20)
    assert not alert_manager.email_notifier.send_notification.called
    
    # Drawdown por encima del límite
    alert_manager.email_notifier.send_notification.reset_mock()
    alert_manager.check_drawdown(30)
    assert alert_manager.email_notifier.send_notification.called


def test_alert_manager_send_alert_failure(alert_manager):
    """Prueba el manejo de un fallo en el envío de la alerta."""
    # Configurar el mock para que lance una excepción
    alert_manager.email_notifier.send_notification.side_effect = Exception("Failed to send alert")
    alert_manager.sms_notifier.send_notification.side_effect = Exception("Failed to send alert")
    
    # Esto no debería propagar la excepción
    drawdown = 20
    alert_manager.check_drawdown(drawdown)
    
    # Verificar que se intentó enviar la alerta
    assert alert_manager.email_notifier.send_notification.called or alert_manager.sms_notifier.send_notification.called


def test_alert_manager_invalid_limit_configuration(alert_manager):
    """Prueba la configuración de un límite inválido."""
    with pytest.raises(ValueError, match="Drawdown limit must be between 0 and 100"):
        alert_manager.drawdown_limit = -10
    
    with pytest.raises(ValueError, match="Drawdown limit must be between 0 and 100"):
        alert_manager.drawdown_limit = 150


def test_alert_manager_multiple_checks(alert_manager):
    """Prueba múltiples verificaciones consecutivas de drawdown."""
    # Primer chequeo: drawdown excede el límite
    alert_manager.check_drawdown(20)
    assert alert_manager.email_notifier.send_notification.called
    
    # Resetear mocks
    alert_manager.email_notifier.send_notification.reset_mock()
    
    # Segundo chequeo: drawdown sigue excediendo
    alert_manager.check_drawdown(25)
    assert alert_manager.email_notifier.send_notification.called
    
    # Resetear mocks
    alert_manager.email_notifier.send_notification.reset_mock()
    
    # Tercer chequeo: drawdown dentro del límite
    alert_manager.check_drawdown(10)
    assert not alert_manager.email_notifier.send_notification.called


# Pruebas para monitoreo de volatilidad
def test_alert_manager_volatility_exceeds_limit(alert_manager):
    """Prueba que se envíe una alerta cuando la volatilidad excede el límite."""
    volatility = 8.0  # 8% de volatilidad (asumimos un límite de 5%)
    alert_manager.check_volatility(volatility, symbol="BTC/USDT")
    
    # Verificar que se envió la alerta
    assert alert_manager.email_notifier.send_notification.called or alert_manager.sms_notifier.send_notification.called


def test_alert_manager_volatility_within_limit(alert_manager):
    """Prueba que no se envíe alerta cuando la volatilidad está dentro del límite."""
    volatility = 3.0  # 3% de volatilidad (asumimos un límite de 5%)
    alert_manager.check_volatility(volatility, symbol="BTC/USDT")
    
    # Verificar que no se envió ninguna alerta
    assert not alert_manager.email_notifier.send_notification.called
    assert not alert_manager.sms_notifier.send_notification.called


# Pruebas para monitoreo de cambios de precio
def test_alert_manager_price_change_exceeds_limit(alert_manager):
    """Prueba que se envíe una alerta cuando el cambio de precio excede el límite."""
    # Cambio significativo en el precio (asumimos un límite de 5%)
    alert_manager.check_price_change(
        symbol="BTC/USDT",
        old_price=40000,
        new_price=44000  # 10% de aumento
    )
    
    # Verificar que se envió la alerta
    assert alert_manager.email_notifier.send_notification.called or alert_manager.sms_notifier.send_notification.called


def test_alert_manager_price_change_within_limit(alert_manager):
    """Prueba que no se envíe alerta cuando el cambio de precio está dentro del límite."""
    # Cambio pequeño en el precio (asumimos un límite de 5%)
    alert_manager.check_price_change(
        symbol="BTC/USDT",
        old_price=40000,
        new_price=41000  # 2.5% de aumento
    )
    
    # Verificar que no se envió ninguna alerta
    assert not alert_manager.email_notifier.send_notification.called
    assert not alert_manager.sms_notifier.send_notification.called


# Pruebas para monitoreo de balance
def test_alert_manager_balance_below_minimum(alert_manager):
    """Prueba que se envíe una alerta cuando el balance cae por debajo del mínimo."""
    # Balance por debajo del mínimo (asumimos un mínimo de 1000 USDT)
    alert_manager.check_balance(
        asset="USDT",
        balance=500,
        min_balance=1000
    )
    
    # Verificar que se envió la alerta
    assert alert_manager.email_notifier.send_notification.called or alert_manager.sms_notifier.send_notification.called


def test_alert_manager_balance_above_minimum(alert_manager):
    """Prueba que no se envíe alerta cuando el balance está por encima del mínimo."""
    # Balance por encima del mínimo
    alert_manager.check_balance(
        asset="USDT",
        balance=1500,
        min_balance=1000
    )
    
    # Verificar que no se envió ninguna alerta
    assert not alert_manager.email_notifier.send_notification.called
    assert not alert_manager.sms_notifier.send_notification.called