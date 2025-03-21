"""
Implementación de notificaciones por email.

Este módulo proporciona una implementación del canal de notificación
para enviar notificaciones por correo electrónico.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio
import os
from typing import Dict, Any, List, Optional, Tuple

from genesis.notifications.base import NotificationChannel
from genesis.config.settings import settings
from genesis.utils.logger import setup_logging


class EmailNotifier(NotificationChannel):
    """
    Canal de notificación por correo electrónico.
    
    Envía notificaciones utilizando un servidor SMTP.
    """
    
    def __init__(
        self,
        name: str = "email",
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        sender: Optional[str] = None,
        use_tls: bool = True
    ):
        """
        Inicializar el notificador de email.
        
        Args:
            name: Nombre del canal
            smtp_server: Servidor SMTP
            smtp_port: Puerto SMTP
            username: Nombre de usuario para autenticación
            password: Contraseña para autenticación
            sender: Dirección de correo del remitente
            use_tls: Si se debe usar TLS
        """
        super().__init__(name)
        
        # Cargar configuración desde settings o environment variables
        self.smtp_server = smtp_server or settings.get('notifications.email.smtp_server') or os.environ.get('EMAIL_SMTP_SERVER')
        self.smtp_port = smtp_port or settings.get('notifications.email.smtp_port') or int(os.environ.get('EMAIL_SMTP_PORT', 587))
        self.username = username or settings.get('notifications.email.username') or os.environ.get('EMAIL_USERNAME')
        self.password = password or settings.get('notifications.email.password') or os.environ.get('EMAIL_PASSWORD')
        self.sender = sender or settings.get('notifications.email.sender') or os.environ.get('EMAIL_SENDER')
        self.use_tls = use_tls
        
        # Validar configuración
        if not all([self.smtp_server, self.smtp_port, self.username, self.password, self.sender]):
            self.logger.warning("Configuración incompleta para el notificador de email")
    
    async def send(self, 
                  recipient: str, 
                  subject: str, 
                  message: str, 
                  html_message: Optional[str] = None,
                  cc: Optional[List[str]] = None,
                  bcc: Optional[List[str]] = None,
                  attachments: Optional[List[Tuple[str, bytes]]] = None,
                  **kwargs) -> bool:
        """
        Enviar una notificación por correo electrónico.
        
        Args:
            recipient: Dirección de correo del destinatario
            subject: Asunto del correo
            message: Cuerpo del mensaje en texto plano
            html_message: Cuerpo del mensaje en HTML (opcional)
            cc: Lista de direcciones en copia
            bcc: Lista de direcciones en copia oculta
            attachments: Lista de archivos adjuntos (nombre, datos)
            **kwargs: Parámetros adicionales
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        # Validar configuración
        if not all([self.smtp_server, self.smtp_port, self.username, self.password, self.sender]):
            self.logger.error("Configuración incompleta para el notificador de email")
            return False
        
        # Ejecutar en un thread para evitar bloquear el loop de asyncio
        try:
            return await asyncio.to_thread(
                self._send_email,
                recipient,
                subject,
                message,
                html_message,
                cc,
                bcc,
                attachments
            )
        except Exception as e:
            self.logger.error(f"Error al enviar email: {e}")
            return False
    
    def _send_email(
        self,
        recipient: str,
        subject: str,
        message: str,
        html_message: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[Tuple[str, bytes]]] = None
    ) -> bool:
        """
        Enviar un correo electrónico (método sincrónico).
        
        Args:
            recipient: Dirección de correo del destinatario
            subject: Asunto del correo
            message: Cuerpo del mensaje en texto plano
            html_message: Cuerpo del mensaje en HTML (opcional)
            cc: Lista de direcciones en copia
            bcc: Lista de direcciones en copia oculta
            attachments: Lista de archivos adjuntos (nombre, datos)
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        # Crear el mensaje
        email_message = MIMEMultipart("alternative")
        email_message["Subject"] = subject
        email_message["From"] = self.sender
        email_message["To"] = recipient
        
        # Añadir CC si existe
        if cc:
            email_message["Cc"] = ", ".join(cc)
        
        # Añadir texto plano
        part1 = MIMEText(message, "plain")
        email_message.attach(part1)
        
        # Añadir HTML si existe
        if html_message:
            part2 = MIMEText(html_message, "html")
            email_message.attach(part2)
        
        # Combinar destinatarios
        recipients = [recipient]
        if cc:
            recipients.extend(cc)
        if bcc:
            recipients.extend(bcc)
        
        try:
            # Crear conexión segura
            context = ssl.create_default_context()
            
            # Conectar al servidor y enviar
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                
                # Autenticar
                server.login(self.username, self.password)
                
                # Enviar
                server.sendmail(self.sender, recipients, email_message.as_string())
            
            self.logger.info(f"Email enviado a {recipient}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al enviar email: {e}")
            return False


# Exportación para uso fácil
email_notifier = EmailNotifier()