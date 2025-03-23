"""
Inicializador del sistema de notificaciones para Genesis.

Este módulo inicializa los componentes relacionados con notificaciones,
incluyendo el notificador de correo electrónico con las credenciales
configuradas en las variables de entorno.
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any

from genesis.notifications.email_notifier import EmailNotifier, EmailConfig
from genesis.notifications.alert_manager import AlertManager

logger = logging.getLogger(__name__)

# Instancia global del notificador de email
email_notifier: Optional[EmailNotifier] = None
alert_manager: Optional[AlertManager] = None

async def initialize_notifications() -> Dict[str, Any]:
    """
    Inicializar componentes de notificaciones.
    
    Returns:
        Diccionario con los componentes inicializados
    """
    global email_notifier, alert_manager
    
    logger.info("Inicializando sistema de notificaciones...")
    
    # Configurar notificador de email
    try:
        email_config = EmailConfig(
            host="smtp.gmail.com",
            port=587,
            username=os.environ.get("GMAIL_EMAIL"),
            password=os.environ.get("GMAIL_APP_PASSWORD"),
            use_tls=True,
            default_sender=os.environ.get("GMAIL_EMAIL"),
            rate_limit=100
        )
        
        email_notifier = EmailNotifier(config=email_config)
        await email_notifier.start()
        logger.info(f"Notificador de email inicializado con cuenta: {email_config.username}")
        
        # Inicializar gestor de alertas
        alert_manager = AlertManager()
        await alert_manager.start()
        logger.info("Gestor de alertas inicializado")
        
        return {
            "email_notifier": email_notifier,
            "alert_manager": alert_manager
        }
        
    except Exception as e:
        logger.error(f"Error inicializando sistema de notificaciones: {e}")
        return {}

async def send_test_notification(recipient: str) -> bool:
    """
    Enviar notificación de prueba.
    
    Args:
        recipient: Dirección de correo del destinatario
        
    Returns:
        True si se envió correctamente, False en caso contrario
    """
    global email_notifier
    
    if not email_notifier:
        logger.error("Notificador de email no inicializado")
        return False
        
    try:
        logger.info(f"Enviando correo de prueba a {recipient}...")
        
        # Crear contenido HTML para la prueba
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
                    <h3>Prueba de Notificación</h3>
                    <p>Este es un mensaje de prueba del Sistema Genesis para verificar que el sistema de notificaciones por correo electrónico está funcionando correctamente.</p>
                    <p>El sistema está configurado para enviar alertas y reportes a su dirección de correo electrónico.</p>
                    <p>Características del sistema de notificaciones:</p>
                    <ul>
                        <li>Alertas de precio para criptomonedas</li>
                        <li>Notificaciones de operaciones de trading</li>
                        <li>Reportes diarios y semanales de rendimiento</li>
                        <li>Alertas de anomalías y patrones de mercado</li>
                    </ul>
                </div>
                <div class="footer">
                    <p>Este mensaje ha sido generado automáticamente. Por favor, no responda a este correo.</p>
                    <p>&copy; {2025} Sistema Genesis - Modo Singularidad Trascendental</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Enviar correo
        success = await email_notifier.send(
            recipient=recipient,
            subject="Prueba del Sistema Genesis - Notificaciones",
            message="Este es un mensaje de prueba del Sistema Genesis.",
            html_message=html_message
        )
        
        if success:
            logger.info(f"Correo de prueba enviado correctamente a {recipient}")
        else:
            logger.error(f"Error al enviar correo de prueba a {recipient}")
            
        return success
        
    except Exception as e:
        logger.error(f"Error al enviar correo de prueba: {e}")
        return False