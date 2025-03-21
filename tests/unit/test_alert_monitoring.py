"""
Pruebas unitarias para el monitoreo y alertas del sistema.

Este módulo prueba las funcionalidades relacionadas con el monitoreo
del sistema y la generación de alertas basadas en métricas.
"""

import pytest
from unittest.mock import Mock
import logging
import sys
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_alerts")

# Asegurarse que podemos importar los módulos de Genesis
sys.path.insert(0, os.getcwd())

# Importar componentes del sistema
from genesis.alerts import AlertManager

# Fixture para AlertManager
@pytest.fixture
def alert_manager():
    """Fixture que proporciona una instancia de AlertManager con notificadores simulados."""
    # Crear mocks para los notificadores
    email_notifier = Mock()
    email_notifier.send_notification = Mock()
    
    sms_notifier = Mock()
    sms_notifier.send_notification = Mock()
    
    # Crear AlertManager con los notificadores simulados
    manager = AlertManager()
    manager.add_notifier("email", email_notifier)
    manager.add_notifier("sms", sms_notifier)
    
    # Configurar límites para las pruebas
    manager.drawdown_limit = 15  # 15% de drawdown es el límite
    manager.volatility_limit = 25  # 25% es el límite de volatilidad
    manager.price_change_limit = 10  # 10% es el límite de cambio de precio
    manager.min_balance = 1000  # $1000 es el balance mínimo
    
    return manager

# Pruebas para check_drawdown
def test_alert_manager_drawdown_exceeds_limit(alert_manager):
    """Prueba que se envíe una alerta cuando el drawdown excede el límite."""
    drawdown = 20  # Mayor al límite de 15%
    alert_manager.check_drawdown(drawdown)
    
    # Verificar que se envió la alerta por email y SMS
    alert_manager._notifiers["email"].send_notification.assert_called_once()
    alert_manager._notifiers["sms"].send_notification.assert_called_once()

def test_alert_manager_drawdown_within_limit(alert_manager):
    """Prueba que no se envíe alerta cuando el drawdown está dentro del límite."""
    drawdown = 10  # Menor al límite de 15%
    alert_manager.check_drawdown(drawdown)
    
    # Verificar que no se enviaron alertas
    alert_manager._notifiers["email"].send_notification.assert_not_called()
    alert_manager._notifiers["sms"].send_notification.assert_not_called()

def test_alert_manager_drawdown_at_limit(alert_manager):
    """Prueba el comportamiento cuando el drawdown está exactamente en el límite."""
    drawdown = 15  # Exactamente en el límite
    alert_manager.check_drawdown(drawdown)
    
    # Por defecto, no se envía alerta cuando está exactamente en el límite
    # Esto puede variar según la implementación real
    alert_manager._notifiers["email"].send_notification.assert_not_called()
    alert_manager._notifiers["sms"].send_notification.assert_not_called()

def test_alert_manager_drawdown_negative(alert_manager):
    """Prueba el manejo de un drawdown negativo."""
    drawdown = -5  # Valor inválido
    
    # Debería lanzar una excepción
    with pytest.raises(ValueError):
        alert_manager.check_drawdown(drawdown)

def test_alert_manager_drawdown_excessive(alert_manager):
    """Prueba un drawdown extremadamente alto."""
    drawdown = 100  # 100%, pérdida total
    alert_manager.check_drawdown(drawdown)
    
    # Debería enviar alerta de máxima prioridad
    alert_manager._notifiers["email"].send_notification.assert_called_once()
    alert_manager._notifiers["sms"].send_notification.assert_called_once()

def test_alert_manager_custom_limit(alert_manager):
    """Prueba el comportamiento con un límite de drawdown personalizado."""
    # Cambiar el límite
    old_limit = alert_manager.drawdown_limit
    alert_manager.drawdown_limit = 25  # Nuevo límite del 25%
    
    # Drawdown por debajo del nuevo límite
    alert_manager.check_drawdown(20)
    alert_manager._notifiers["email"].send_notification.assert_not_called()
    
    # Drawdown por encima del nuevo límite
    alert_manager._notifiers["email"].send_notification.reset_mock()
    alert_manager._notifiers["sms"].send_notification.reset_mock()
    alert_manager.check_drawdown(30)
    alert_manager._notifiers["email"].send_notification.assert_called_once()
    
    # Restaurar el límite original
    alert_manager.drawdown_limit = old_limit

def test_alert_manager_send_alert_failure(alert_manager):
    """Prueba el manejo de un fallo en el envío de la alerta."""
    # Simular fallo en el envío
    alert_manager._notifiers["email"].send_notification.side_effect = Exception("Failed to send email")
    
    # El sistema debe manejar la excepción sin fallar
    try:
        alert_manager.check_drawdown(20)  # Drawdown superior al límite
        # Si llegamos aquí, se manejó correctamente la excepción
        assert True
    except:
        pytest.fail("La excepción no fue manejada correctamente")

def test_alert_manager_invalid_limit_configuration(alert_manager):
    """Prueba la configuración de un límite inválido."""
    # Intentar establecer un límite negativo
    with pytest.raises(ValueError):
        alert_manager.drawdown_limit = -10
    
    # Intentar establecer un límite superior a 100%
    with pytest.raises(ValueError):
        alert_manager.drawdown_limit = 150

def test_alert_manager_multiple_checks(alert_manager):
    """Prueba múltiples verificaciones consecutivas de drawdown."""
    # Primera verificación: drawdown dentro del límite
    alert_manager.check_drawdown(10)
    alert_manager._notifiers["email"].send_notification.assert_not_called()
    
    # Segunda verificación: drawdown excede el límite
    alert_manager._notifiers["email"].send_notification.reset_mock()
    alert_manager._notifiers["sms"].send_notification.reset_mock()
    alert_manager.check_drawdown(20)
    alert_manager._notifiers["email"].send_notification.assert_called_once()
    
    # Tercera verificación: drawdown sigue excediendo el límite
    alert_manager._notifiers["email"].send_notification.reset_mock()
    alert_manager._notifiers["sms"].send_notification.reset_mock()
    alert_manager.check_drawdown(25)
    # Según la implementación, podría no enviar alerta de nuevo en un período corto de tiempo
    # para evitar spam, o podría enviar una alerta de seguimiento
    
    # Cuarta verificación: drawdown vuelve a estar dentro del límite
    alert_manager._notifiers["email"].send_notification.reset_mock()
    alert_manager._notifiers["sms"].send_notification.reset_mock()
    alert_manager.check_drawdown(5)
    # Podría enviar una notificación de "recuperación" o no enviar nada
    # Depende de la implementación

# Pruebas adicionales para otros tipos de alertas
def test_alert_manager_volatility_exceeds_limit(alert_manager):
    """Prueba que se envíe una alerta cuando la volatilidad excede el límite."""
    volatility = 30  # Mayor al límite de 25%
    alert_manager.check_volatility(volatility)
    
    alert_manager._notifiers["email"].send_notification.assert_called_once()
    alert_manager._notifiers["sms"].send_notification.assert_called_once()

def test_alert_manager_volatility_within_limit(alert_manager):
    """Prueba que no se envíe alerta cuando la volatilidad está dentro del límite."""
    volatility = 20  # Menor al límite de 25%
    alert_manager.check_volatility(volatility)
    
    alert_manager._notifiers["email"].send_notification.assert_not_called()
    alert_manager._notifiers["sms"].send_notification.assert_not_called()

def test_alert_manager_price_change_exceeds_limit(alert_manager):
    """Prueba que se envíe una alerta cuando el cambio de precio excede el límite."""
    price_change = 15  # Mayor al límite de 10%
    alert_manager.check_price_change(price_change)
    
    alert_manager._notifiers["email"].send_notification.assert_called_once()
    alert_manager._notifiers["sms"].send_notification.assert_called_once()

def test_alert_manager_price_change_within_limit(alert_manager):
    """Prueba que no se envíe alerta cuando el cambio de precio está dentro del límite."""
    price_change = 5  # Menor al límite de 10%
    alert_manager.check_price_change(price_change)
    
    alert_manager._notifiers["email"].send_notification.assert_not_called()
    alert_manager._notifiers["sms"].send_notification.assert_not_called()

def test_alert_manager_balance_below_minimum(alert_manager):
    """Prueba que se envíe una alerta cuando el balance cae por debajo del mínimo."""
    balance = 900  # Menor al mínimo de $1000
    alert_manager.check_balance(balance)
    
    alert_manager._notifiers["email"].send_notification.assert_called_once()
    alert_manager._notifiers["sms"].send_notification.assert_called_once()

def test_alert_manager_balance_above_minimum(alert_manager):
    """Prueba que no se envíe alerta cuando el balance está por encima del mínimo."""
    balance = 1200  # Mayor al mínimo de $1000
    alert_manager.check_balance(balance)
    
    alert_manager._notifiers["email"].send_notification.assert_not_called()
    alert_manager._notifiers["sms"].send_notification.assert_not_called()