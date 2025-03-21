"""
Pruebas unitarias para el sistema de notificaciones.

Este módulo prueba los componentes del sistema de notificaciones,
incluyendo el gestor de alertas y los diferentes métodos de notificación.
"""

import pytest
import datetime
from unittest.mock import Mock, patch, MagicMock
import asyncio

from genesis.notifications.alert_manager import AlertManager
from genesis.core.event_bus import EventBus


class TestEmailNotifier:
    """Simulación del componente de notificaciones por email para pruebas."""
    
    def __init__(self):
        self.sent_emails = []
    
    async def send_notification(self, to, subject, body):
        """Simula el envío de un email."""
        self.sent_emails.append({
            "to": to,
            "subject": subject,
            "body": body
        })
        return True


class TestSMSNotifier:
    """Simulación del componente de notificaciones por SMS para pruebas."""
    
    def __init__(self):
        self.sent_sms = []
    
    async def send_notification(self, to, message):
        """Simula el envío de un SMS."""
        self.sent_sms.append({
            "to": to,
            "message": message
        })
        return True


@pytest.fixture
def alert_manager():
    """Fixture para crear un gestor de alertas para pruebas."""
    event_bus = EventBus()
    manager = AlertManager(event_bus=event_bus)
    
    # Configurar notificadores simulados
    manager.email_notifier = TestEmailNotifier()
    manager.sms_notifier = TestSMSNotifier()
    
    return manager


@pytest.mark.asyncio
async def test_alert_manager_initialization(alert_manager):
    """Probar la inicialización del gestor de alertas."""
    assert alert_manager is not None
    assert alert_manager.email_notifier is not None
    assert alert_manager.sms_notifier is not None
    
    # Verificar que se inicia correctamente
    await alert_manager.start()
    assert alert_manager.running is True
    
    # Verificar que se detiene correctamente
    await alert_manager.stop()
    assert alert_manager.running is False


@pytest.mark.asyncio
async def test_alert_manager_price_alert(alert_manager):
    """Probar la creación y activación de alertas de precio."""
    # Configurar gestor de alertas
    await alert_manager.start()
    
    # Añadir alerta de precio
    alert_id = alert_manager.add_price_alert(
        symbol="BTC/USDT",
        price=40000,
        condition="above",
        message="Bitcoin superó los $40,000",
        user_id="user123",
        notification_methods=["email", "sms"]
    )
    
    # Verificar que la alerta se ha creado
    assert len(alert_manager.alerts) == 1
    assert alert_manager.alerts[alert_id].symbol == "BTC/USDT"
    assert alert_manager.alerts[alert_id].price == 40000
    
    # Simular evento de precio que activa la alerta
    await alert_manager.handle_event(
        "market.price_update",
        {
            "symbol": "BTC/USDT",
            "price": 41000,
            "timestamp": datetime.datetime.utcnow().isoformat()
        },
        "market_data"
    )
    
    # Esperar un poco para que se procesen las notificaciones asíncronas
    await asyncio.sleep(0.1)
    
    # Verificar que se enviaron las notificaciones
    assert len(alert_manager.email_notifier.sent_emails) == 1
    assert "Bitcoin superó" in alert_manager.email_notifier.sent_emails[0]["subject"]
    
    assert len(alert_manager.sms_notifier.sent_sms) == 1
    assert "Bitcoin superó" in alert_manager.sms_notifier.sent_sms[0]["message"]
    
    # Detener gestor de alertas
    await alert_manager.stop()


@pytest.mark.asyncio
async def test_alert_manager_volume_alert(alert_manager):
    """Probar la creación y activación de alertas de volumen."""
    # Configurar gestor de alertas
    await alert_manager.start()
    
    # Añadir alerta de volumen
    alert_id = alert_manager.add_volume_alert(
        symbol="ETH/USDT",
        volume=1000000,
        timeframe="1h",
        message="Volumen alto en Ethereum",
        user_id="user123",
        notification_methods=["email"]
    )
    
    # Verificar que la alerta se ha creado
    assert len(alert_manager.alerts) == 1
    assert alert_manager.alerts[alert_id].symbol == "ETH/USDT"
    assert alert_manager.alerts[alert_id].volume == 1000000
    
    # Simular evento de volumen que activa la alerta
    await alert_manager.handle_event(
        "market.volume_update",
        {
            "symbol": "ETH/USDT",
            "volume": 1500000,
            "timeframe": "1h",
            "timestamp": datetime.datetime.utcnow().isoformat()
        },
        "market_data"
    )
    
    # Esperar un poco para que se procesen las notificaciones asíncronas
    await asyncio.sleep(0.1)
    
    # Verificar que se enviaron las notificaciones
    assert len(alert_manager.email_notifier.sent_emails) == 1
    assert "Volumen alto" in alert_manager.email_notifier.sent_emails[0]["subject"]
    
    # Verificar que no se enviaron SMS (no se especificó como método)
    assert len(alert_manager.sms_notifier.sent_sms) == 0
    
    # Detener gestor de alertas
    await alert_manager.stop()


@pytest.mark.asyncio
async def test_alert_manager_volatility_alert(alert_manager):
    """Probar la creación y activación de alertas de volatilidad."""
    # Configurar gestor de alertas
    await alert_manager.start()
    
    # Añadir alerta de volatilidad
    alert_id = alert_manager.add_volatility_alert(
        symbol="BTC/USDT",
        percent=5.0,
        timeframe="15m",
        message="Alta volatilidad en Bitcoin",
        user_id="user123",
        notification_methods=["sms"]
    )
    
    # Verificar que la alerta se ha creado
    assert len(alert_manager.alerts) == 1
    assert alert_manager.alerts[alert_id].symbol == "BTC/USDT"
    assert alert_manager.alerts[alert_id].percent == 5.0
    
    # Simular evento de volatilidad que activa la alerta
    await alert_manager.handle_event(
        "market.volatility_update",
        {
            "symbol": "BTC/USDT",
            "volatility": 6.2,
            "timeframe": "15m",
            "timestamp": datetime.datetime.utcnow().isoformat()
        },
        "market_data"
    )
    
    # Esperar un poco para que se procesen las notificaciones asíncronas
    await asyncio.sleep(0.1)
    
    # Verificar que se enviaron las notificaciones
    assert len(alert_manager.sms_notifier.sent_sms) == 1
    assert "Alta volatilidad" in alert_manager.sms_notifier.sent_sms[0]["message"]
    
    # Verificar que no se enviaron emails (no se especificó como método)
    assert len(alert_manager.email_notifier.sent_emails) == 0
    
    # Detener gestor de alertas
    await alert_manager.stop()


@pytest.mark.asyncio
async def test_alert_manager_multiple_alerts(alert_manager):
    """Probar la gestión de múltiples alertas simultáneas."""
    # Configurar gestor de alertas
    await alert_manager.start()
    
    # Añadir múltiples alertas
    alert_id1 = alert_manager.add_price_alert(
        symbol="BTC/USDT",
        price=40000,
        condition="above",
        message="Bitcoin superó los $40,000",
        user_id="user123",
        notification_methods=["email"]
    )
    
    alert_id2 = alert_manager.add_price_alert(
        symbol="BTC/USDT",
        price=38000,
        condition="below",
        message="Bitcoin por debajo de $38,000",
        user_id="user123",
        notification_methods=["sms"]
    )
    
    alert_id3 = alert_manager.add_price_alert(
        symbol="ETH/USDT",
        price=3000,
        condition="above",
        message="Ethereum superó los $3,000",
        user_id="user456",
        notification_methods=["email", "sms"]
    )
    
    # Verificar que las alertas se han creado
    assert len(alert_manager.alerts) == 3
    
    # Simular evento de precio que activa la primera y tercera alerta
    await alert_manager.handle_event(
        "market.price_update",
        {
            "symbol": "BTC/USDT",
            "price": 41000,
            "timestamp": datetime.datetime.utcnow().isoformat()
        },
        "market_data"
    )
    
    await alert_manager.handle_event(
        "market.price_update",
        {
            "symbol": "ETH/USDT",
            "price": 3200,
            "timestamp": datetime.datetime.utcnow().isoformat()
        },
        "market_data"
    )
    
    # Esperar un poco para que se procesen las notificaciones asíncronas
    await asyncio.sleep(0.1)
    
    # Verificar que se enviaron las notificaciones correctas
    assert len(alert_manager.email_notifier.sent_emails) == 2
    assert len(alert_manager.sms_notifier.sent_sms) == 1
    
    # Simular evento que activa la segunda alerta
    await alert_manager.handle_event(
        "market.price_update",
        {
            "symbol": "BTC/USDT",
            "price": 37000,
            "timestamp": datetime.datetime.utcnow().isoformat()
        },
        "market_data"
    )
    
    # Esperar un poco para que se procesen las notificaciones asíncronas
    await asyncio.sleep(0.1)
    
    # Verificar que se enviaron las notificaciones adicionales
    assert len(alert_manager.email_notifier.sent_emails) == 2  # No cambió
    assert len(alert_manager.sms_notifier.sent_sms) == 2  # Aumentó en 1
    
    # Detener gestor de alertas
    await alert_manager.stop()


@pytest.mark.asyncio
async def test_alert_manager_remove_alerts(alert_manager):
    """Probar la eliminación de alertas."""
    # Configurar gestor de alertas
    await alert_manager.start()
    
    # Añadir alertas
    alert_id1 = alert_manager.add_price_alert(
        symbol="BTC/USDT",
        price=40000,
        condition="above",
        message="Bitcoin superó los $40,000",
        user_id="user123",
        notification_methods=["email"]
    )
    
    alert_id2 = alert_manager.add_price_alert(
        symbol="ETH/USDT",
        price=3000,
        condition="above",
        message="Ethereum superó los $3,000",
        user_id="user123",
        notification_methods=["email"]
    )
    
    # Verificar que hay dos alertas
    assert len(alert_manager.alerts) == 2
    
    # Eliminar una alerta
    success = alert_manager.remove_alert(alert_id1)
    assert success is True
    
    # Verificar que queda una alerta
    assert len(alert_manager.alerts) == 1
    assert alert_id2 in alert_manager.alerts
    
    # Intentar eliminar una alerta que no existe
    success = alert_manager.remove_alert("nonexistent_id")
    assert success is False
    
    # Verificar que sigue habiendo una alerta
    assert len(alert_manager.alerts) == 1
    
    # Detener gestor de alertas
    await alert_manager.stop()


@pytest.mark.asyncio
async def test_alert_manager_list_user_alerts(alert_manager):
    """Probar la obtención de alertas por usuario."""
    # Configurar gestor de alertas
    await alert_manager.start()
    
    # Añadir alertas para diferentes usuarios
    alert_manager.add_price_alert(
        symbol="BTC/USDT",
        price=40000,
        condition="above",
        message="Bitcoin superó los $40,000",
        user_id="user123",
        notification_methods=["email"]
    )
    
    alert_manager.add_price_alert(
        symbol="ETH/USDT",
        price=3000,
        condition="above",
        message="Ethereum superó los $3,000",
        user_id="user123",
        notification_methods=["email"]
    )
    
    alert_manager.add_price_alert(
        symbol="XRP/USDT",
        price=1.0,
        condition="above",
        message="XRP superó $1.0",
        user_id="user456",
        notification_methods=["email"]
    )
    
    # Obtener alertas por usuario
    user123_alerts = alert_manager.get_user_alerts("user123")
    user456_alerts = alert_manager.get_user_alerts("user456")
    user789_alerts = alert_manager.get_user_alerts("user789")
    
    # Verificar resultados
    assert len(user123_alerts) == 2
    assert len(user456_alerts) == 1
    assert len(user789_alerts) == 0
    
    # Verificar símbolos en las alertas
    user123_symbols = [alert.symbol for alert in user123_alerts.values()]
    assert "BTC/USDT" in user123_symbols
    assert "ETH/USDT" in user123_symbols
    
    # Detener gestor de alertas
    await alert_manager.stop()