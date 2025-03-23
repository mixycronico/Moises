"""
Inicializador del sistema de notificaciones de Genesis.

Este módulo configura y proporciona acceso a los diferentes canales
de notificación del sistema, como email, alertas internas, etc.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List

from genesis.notifications.email_notifier import EmailNotifier
from genesis.notifications.alert_manager import AlertManager, AlertCondition, AlertType, AlertStatus

logger = logging.getLogger(__name__)

# Instancias globales
email_notifier: Optional[EmailNotifier] = None
alert_manager: Optional[AlertManager] = None

async def initialize_notifications(
    enable_email: bool = True,
    enable_alerts: bool = True,
    email_config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Inicializar el sistema de notificaciones.
    
    Args:
        enable_email: Activar notificaciones por email
        enable_alerts: Activar sistema de alertas
        email_config: Configuración adicional para email
        
    Returns:
        True si se inicializó correctamente
    """
    global email_notifier, alert_manager
    
    success = True
    
    # Inicializar notificador de email
    if enable_email:
        email_success = await _initialize_email_notifier(email_config)
        if not email_success:
            logger.warning("No se pudo inicializar el notificador de email")
            success = False
    
    # Inicializar administrador de alertas
    if enable_alerts:
        alert_success = await _initialize_alert_manager()
        if not alert_success:
            logger.warning("No se pudo inicializar el administrador de alertas")
            success = False
    
    if success:
        logger.info("Sistema de notificaciones inicializado correctamente")
    else:
        logger.warning("Sistema de notificaciones inicializado con errores")
        
    return success

async def _initialize_email_notifier(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Inicializar el notificador de email.
    
    Args:
        config: Configuración adicional
        
    Returns:
        True si se inicializó correctamente
    """
    global email_notifier
    
    try:
        # Obtener configuración de variables de entorno o usar valores predeterminados
        email = os.environ.get("GMAIL_EMAIL")
        password = os.environ.get("GMAIL_APP_PASSWORD")
        
        if not email or not password:
            logger.error("No se encontraron credenciales de Gmail en variables de entorno")
            return False
        
        # Crear instancia del notificador
        email_notifier = EmailNotifier(
            sender_email=email,
            sender_password=password,
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            use_tls=True
        )
        
        # Aplicar configuración adicional si la hay
        if config:
            for key, value in config.items():
                if hasattr(email_notifier, key):
                    setattr(email_notifier, key, value)
        
        logger.info(f"Notificador de email inicializado con cuenta {email}")
        
        return True
    except Exception as e:
        logger.error(f"Error inicializando notificador de email: {e}")
        return False

async def _initialize_alert_manager() -> bool:
    """
    Inicializar el administrador de alertas.
    
    Returns:
        True si se inicializó correctamente
    """
    global alert_manager
    
    try:
        # Crear instancia del administrador de alertas
        alert_manager = AlertManager()
        
        # Configurar condiciones de alerta predeterminadas
        await _configure_default_alerts()
        
        logger.info("Administrador de alertas inicializado correctamente")
        
        return True
    except Exception as e:
        logger.error(f"Error inicializando administrador de alertas: {e}")
        return False

async def _configure_default_alerts() -> None:
    """Configurar condiciones de alerta predeterminadas."""
    global alert_manager
    
    if not alert_manager:
        return
    
    # Alertas para movimientos de precio
    await alert_manager.add_condition(
        AlertCondition(
            name="price_drop_10",
            description="Alerta cuando el precio cae más del 10%",
            alert_type=AlertType.PRICE,
            threshold=-10.0,
            comparison="<",
            timeframe="1h",
            symbols=["BTC", "ETH", "ADA", "DOT", "BNB"]
        )
    )
    
    await alert_manager.add_condition(
        AlertCondition(
            name="price_rise_15",
            description="Alerta cuando el precio sube más del 15%",
            alert_type=AlertType.PRICE,
            threshold=15.0,
            comparison=">",
            timeframe="1h",
            symbols=["BTC", "ETH", "ADA", "DOT", "BNB"]
        )
    )
    
    # Alertas para cambios de tendencia
    await alert_manager.add_condition(
        AlertCondition(
            name="trend_change",
            description="Alerta cuando ocurre un cambio de tendencia significativo",
            alert_type=AlertType.TREND,
            threshold=0.5,
            comparison="change",
            timeframe="4h",
            symbols=["BTC", "ETH"]
        )
    )
    
    # Alerta de oportunidades de arbitraje
    await alert_manager.add_condition(
        AlertCondition(
            name="arbitrage_opportunity",
            description="Alerta cuando hay una oportunidad de arbitraje > 3%",
            alert_type=AlertType.ARBITRAGE,
            threshold=3.0,
            comparison=">",
            timeframe="5m"
        )
    )
    
    # Alerta de estrategia que alcanza punto óptimo de entrada
    await alert_manager.add_condition(
        AlertCondition(
            name="strategy_entry",
            description="Punto óptimo de entrada según estrategia",
            alert_type=AlertType.STRATEGY,
            threshold=0.8,
            comparison=">",
            timeframe="1d",
            strategy_id="reinforcement_ensemble"
        )
    )
    
    logger.info(f"Configuradas {await alert_manager.get_condition_count()} condiciones de alerta predeterminadas")

async def send_test_notification(
    recipient: str,
    subject: str = "Prueba del Sistema Genesis",
    message: str = "Esta es una prueba del Sistema de Notificaciones de Genesis.",
    html_message: Optional[str] = None
) -> bool:
    """
    Enviar notificación de prueba por email.
    
    Args:
        recipient: Destinatario
        subject: Asunto del correo
        message: Mensaje en texto plano
        html_message: Mensaje en formato HTML (opcional)
        
    Returns:
        True si se envió correctamente
    """
    global email_notifier
    
    if not email_notifier:
        success = await initialize_notifications(enable_email=True, enable_alerts=False)
        if not success:
            logger.error("No se pudo inicializar el notificador de email para prueba")
            return False
    
    try:
        if not html_message:
            # Crear versión HTML básica del mensaje si no se proporciona
            html_message = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background: linear-gradient(135deg, #0066cc, #00cc99); color: white; padding: 15px; border-radius: 5px 5px 0 0; }}
                    .content {{ background-color: #f8f9fa; padding: 20px; border-radius: 0 0 5px 5px; border: 1px solid #e9ecef; border-top: none; }}
                    .footer {{ font-size: 0.8em; color: #6c757d; margin-top: 20px; text-align: center; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h2 style="margin: 0;">Sistema Genesis</h2>
                    </div>
                    <div class="content">
                        <h3>Notificación de Prueba</h3>
                        <p>{message}</p>
                        <p>Si recibiste este correo, significa que el sistema de notificaciones está configurado correctamente.</p>
                    </div>
                    <div class="footer">
                        <p>Este mensaje ha sido generado automáticamente. Por favor, no responda a este correo.</p>
                    </div>
                </div>
            </body>
            </html>
            """
        
        # Enviar correo
        success = await email_notifier.send(
            recipient=recipient,
            subject=subject,
            message=message,
            html_message=html_message
        )
        
        if success:
            logger.info(f"Correo de prueba enviado a {recipient}")
        else:
            logger.error(f"Error al enviar correo de prueba a {recipient}")
            
        return success
    except Exception as e:
        logger.error(f"Error enviando correo de prueba: {e}")
        return False