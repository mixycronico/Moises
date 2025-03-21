"""
Notificador de correo electrónico para el sistema Genesis.

Este módulo proporciona funcionalidades para enviar notificaciones
por correo electrónico de forma asíncrona y con formato HTML.
"""

import asyncio
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict, Any, Union

from genesis.core.base import Component

class EmailNotifier(Component):
    """
    Cliente avanzado para envío asíncrono de notificaciones por correo electrónico.
    
    Este componente gestiona el envío de correos electrónicos para notificaciones
    del sistema, con soporte para formatos HTML y reintentos automáticos.
    """
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        max_retries: int = 3,
        name: str = "email_notifier"
    ):
        """
        Inicializar el cliente de notificación por correo.
        
        Args:
            smtp_server: Dirección del servidor SMTP (e.g., smtp.gmail.com)
            smtp_port: Puerto del servidor SMTP (e.g., 465 para SSL)
            username: Correo electrónico del remitente
            password: Contraseña o token de aplicación del correo
            max_retries: Máximo número de reintentos en caso de fallo
            name: Nombre del componente
        """
        super().__init__(name)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """Iniciar el notificador de correo."""
        await super().start()
        self._validate_config()
        self.logger.info("Notificador de correo iniciado")

    async def stop(self) -> None:
        """Detener el notificador de correo."""
        await super().stop()
        self.logger.info("Notificador de correo detenido")

    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        if event_type == "notification.email.send":
            subject = data.get("subject", "Notificación de Genesis")
            message = data.get("message", "")
            recipients = data.get("recipients")
            html = data.get("html", False)
            
            try:
                await self.send_email(subject, message, recipients, html)
                await self.emit_event("notification.email.sent", {
                    "success": True,
                    "subject": subject,
                    "recipients": recipients,
                    "request_id": data.get("request_id")
                })
            except Exception as e:
                self.logger.error(f"Error al enviar correo: {e}")
                await self.emit_event("notification.email.error", {
                    "success": False,
                    "error": str(e),
                    "subject": subject,
                    "recipients": recipients,
                    "request_id": data.get("request_id")
                })

    def _validate_config(self):
        """Validar los parámetros de configuración."""
        if not all([self.smtp_server, self.smtp_port, self.username, self.password]):
            raise ValueError("Todos los parámetros de configuración (server, port, username, password) son obligatorios.")
        if not isinstance(self.smtp_port, int) or self.smtp_port <= 0:
            raise ValueError("El puerto SMTP debe ser un entero positivo.")

    def _create_message(self, subject: str, body: str, recipients: List[str], html: bool = False) -> MIMEMultipart:
        """
        Crear el mensaje de correo con soporte para HTML.
        
        Args:
            subject: Asunto del correo
            body: Cuerpo del mensaje
            recipients: Lista de destinatarios
            html: Si el cuerpo es HTML
            
        Returns:
            Mensaje MIME creado
        """
        msg = MIMEMultipart("alternative")
        msg['From'] = self.username
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, "html" if html else "plain"))
        return msg

    async def _send_email_with_retry(self, msg: MIMEMultipart, recipients: List[str]) -> None:
        """
        Enviar email con reintentos en caso de error.
        
        Args:
            msg: Mensaje MIME a enviar
            recipients: Lista de destinatarios
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self._send_email_sync(msg, recipients)
                )
                # Si llegamos aquí, el envío fue exitoso
                self.logger.info(f"Correo enviado a: {', '.join(recipients)}")
                return
            except (smtplib.SMTPException, ConnectionError) as e:
                if attempt == self.max_retries:
                    self.logger.error(f"Fallo crítico al enviar correo tras {self.max_retries} intentos: {e}")
                    raise
                    
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.warning(f"Intento {attempt}/{self.max_retries} fallido: {e}. Reintentando en {wait_time}s...")
                await asyncio.sleep(wait_time)

    def _send_email_sync(self, msg: MIMEMultipart, recipients: List[str]) -> None:
        """
        Enviar email de forma síncrona.
        
        Args:
            msg: Mensaje MIME a enviar
            recipients: Lista de destinatarios
        """
        with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
            server.login(self.username, self.password)
            server.sendmail(self.username, recipients, msg.as_string())

    async def send_email(
        self, 
        subject: str, 
        message: str, 
        recipients: Optional[List[str]] = None, 
        html: bool = False
    ) -> None:
        """
        Enviar un correo electrónico de forma asíncrona.
        
        Args:
            subject: Asunto del correo
            message: Cuerpo del mensaje (texto plano o HTML)
            recipients: Lista de destinatarios; si None, usa el username
            html: Indica si el mensaje es en formato HTML
        """
        recipients = recipients or [self.username]
        if not isinstance(recipients, list) or not all(isinstance(r, str) for r in recipients):
            raise ValueError("Los destinatarios deben ser una lista de strings.")

        msg = self._create_message(subject, message, recipients, html)
        await self._send_email_with_retry(msg, recipients)

    @classmethod
    def create_html_message(
        cls, 
        title: str, 
        details: Dict[str, Union[str, int, float]], 
        footer_text: Optional[str] = None
    ) -> str:
        """
        Crear un mensaje HTML con formato profesional.
        
        Args:
            title: Título del mensaje
            details: Diccionario de detalles a mostrar
            footer_text: Texto opcional para el pie de página
            
        Returns:
            Mensaje HTML formateado
        """
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; color: #333; }
                h2 { color: #2c3e50; }
                .detail { margin: 5px 0; }
                .footer { font-size: 12px; color: #777; margin-top: 20px; }
            </style>
        </head>
        <body>
            <h2>{title}</h2>
            {details}
            <div class="footer">
                {footer}
            </div>
        </body>
        </html>
        """
        # Convertir detalles a HTML
        details_html = "".join(f"<p class='detail'><strong>{k}:</strong> {v}</p>" for k, v in details.items())
        
        # Texto del pie de página
        footer = footer_text or "Enviado por Genesis"
        
        return html.format(
            title=title,
            details=details_html,
            footer=footer
        )