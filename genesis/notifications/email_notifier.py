"""
Sistema de notificaciones por email para el sistema Genesis.

Este módulo implementa funcionalidades para enviar notificaciones
por correo electrónico a usuarios sobre eventos del sistema,
alertas de trading, y reportes periódicos.
"""

import logging
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.utils import formatdate
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta

from genesis.core.base import Component
from genesis.notifications.base import NotificationChannel

class EmailConfig:
    """Configuración para el servidor SMTP."""
    
    def __init__(
        self,
        host: str = "smtp.gmail.com",
        port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        default_sender: Optional[str] = None,
        rate_limit: int = 100  # mensajes por hora
    ):
        """
        Inicializar configuración de email.
        
        Args:
            host: Host del servidor SMTP
            port: Puerto del servidor SMTP
            username: Nombre de usuario
            password: Contraseña
            use_tls: Usar TLS para la conexión
            default_sender: Dirección de correo predeterminada para envío
            rate_limit: Límite de mensajes por hora
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.default_sender = default_sender
        self.rate_limit = rate_limit

class EmailNotifier(Component, NotificationChannel):
    """
    Componente para enviar notificaciones por correo electrónico.
    
    Este componente gestiona el envío de correos electrónicos
    para alertas, reportes y otras notificaciones del sistema.
    """
    
    def __init__(
        self,
        name: str = "email_notifier",
        config: Optional[EmailConfig] = None,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        use_tls: bool = True
    ):
        """
        Inicializar el notificador de email.
        
        Args:
            name: Nombre del componente
            config: Configuración del servidor SMTP
            sender_email: Correo electrónico del remitente
            sender_password: Contraseña del remitente
            smtp_server: Servidor SMTP
            smtp_port: Puerto SMTP
            use_tls: Usar TLS para la conexión
        """
        Component.__init__(self, name)
        NotificationChannel.__init__(self, name)
        
        self.logger = logging.getLogger(__name__)
        
        # Si se proporcionan parámetros directos, crear un EmailConfig
        if sender_email or sender_password or smtp_server or smtp_port:
            self.config = EmailConfig(
                host=smtp_server,
                port=smtp_port,
                username=sender_email,
                password=sender_password,
                use_tls=use_tls,
                default_sender=sender_email
            )
        else:
            self.config = config or EmailConfig()
        
        # Variables para control de tasa de envío
        self.sent_count = 0
        self.sending_period_start = datetime.now()
        self.pending_emails = asyncio.Queue()
        self.recipient_history: Dict[str, List[datetime]] = {}
        
        # Conjunto para evitar duplicados en corto tiempo
        self.recent_messages: Set[str] = set()
        
        # Estado de conexión
        self.connection = None
        self.consumer_task = None
        
    async def start(self) -> None:
        """Iniciar el notificador de email."""
        await super().start()
        self.consumer_task = asyncio.create_task(self._consume_queue())
        self.logger.info("Notificador de email iniciado")
        
    async def stop(self) -> None:
        """Detener el notificador de email."""
        if self.consumer_task:
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass
            self.consumer_task = None
            
        # Cerrar conexión SMTP si está abierta
        if self.connection:
            try:
                self.connection.quit()
            except Exception:
                pass
            finally:
                self.connection = None
                
        await super().stop()
        self.logger.info("Notificador de email detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente origen
        """
        if event_type == "alert.triggered":
            # Procesar alerta
            alert_type = data.get("alert_type")
            if not alert_type:
                return
                
            recipients = data.get("recipients", [])
            if not recipients:
                return
                
            subject = data.get("subject", f"Alerta: {alert_type}")
            message = data.get("message", "Se ha activado una alerta")
            priority = data.get("priority", "normal")
            
            # Enviar notificación a cada destinatario
            for recipient in recipients:
                await self.send_alert(
                    recipient=recipient,
                    subject=subject,
                    message=message,
                    alert_type=alert_type,
                    priority=priority,
                    data=data
                )
                
        elif event_type == "report.generated":
            # Enviar informe generado
            report_type = data.get("report_type")
            recipients = data.get("recipients", [])
            
            if not report_type or not recipients:
                return
                
            report_path = data.get("report_path")
            subject = data.get("subject", f"Informe: {report_type}")
            message = data.get("message", f"Se ha generado un informe de tipo {report_type}")
            
            # Enviar a destinatarios
            for recipient in recipients:
                await self.send_report(
                    recipient=recipient,
                    subject=subject,
                    message=message,
                    report_type=report_type,
                    attachment_path=report_path,
                    data=data
                )
                
    async def send(
        self,
        recipient: str,
        subject: str,
        message: str,
        **kwargs
    ) -> bool:
        """
        Enviar un mensaje de correo electrónico.
        
        Args:
            recipient: Dirección de correo del destinatario
            subject: Asunto del mensaje
            message: Contenido del mensaje
            **kwargs: Argumentos adicionales
            
        Returns:
            True si se encola correctamente, False en caso contrario
        """
        # Verificar configuración
        if not self.config.username or not self.config.password:
            self.logger.error("Falta configuración de SMTP")
            return False
            
        # Verificar límite de tasa para destinatario
        if not self._check_recipient_rate_limit(recipient):
            self.logger.warning(f"Límite de tasa excedido para {recipient}")
            return False
            
        # Verificar duplicados recientes
        message_hash = f"{recipient}:{subject}:{message[:100]}"
        if message_hash in self.recent_messages:
            self.logger.info(f"Mensaje duplicado para {recipient}, no enviado")
            return False
            
        # Crear mensaje
        email = self._create_email_message(
            recipient=recipient,
            subject=subject,
            message=message,
            **kwargs
        )
        
        # Añadir a la cola
        await self.pending_emails.put(email)
        
        # Actualizar conjunto de mensajes recientes
        self.recent_messages.add(message_hash)
        
        # Limpiar mensajes antiguos (más de 5 minutos)
        self._clean_recent_messages()
        
        return True
        
    async def send_alert(
        self,
        recipient: str,
        subject: str,
        message: str,
        alert_type: str,
        priority: str = "normal",
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Enviar una alerta por correo electrónico.
        
        Args:
            recipient: Dirección de correo del destinatario
            subject: Asunto del mensaje
            message: Contenido del mensaje
            alert_type: Tipo de alerta
            priority: Prioridad (low, normal, high, critical)
            data: Datos adicionales
            
        Returns:
            True si se envía correctamente, False en caso contrario
        """
        # Crear contenido HTML para la alerta
        html_message = self._create_html_alert(
            message=message,
            alert_type=alert_type,
            priority=priority,
            data=data or {}
        )
        
        # Configurar prioridad en cabeceras
        headers = None
        if priority == "high" or priority == "critical":
            headers = {
                "X-Priority": "1",
                "X-MSMail-Priority": "High",
                "Importance": "High"
            }
            
        return await self.send(
            recipient=recipient,
            subject=subject,
            message=message,
            html_message=html_message,
            headers=headers
        )
        
    async def send_report(
        self,
        recipient: str,
        subject: str,
        message: str,
        report_type: str,
        attachment_path: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Enviar un informe por correo electrónico.
        
        Args:
            recipient: Dirección de correo del destinatario
            subject: Asunto del mensaje
            message: Contenido del mensaje
            report_type: Tipo de informe
            attachment_path: Ruta al archivo adjunto
            data: Datos adicionales
            
        Returns:
            True si se envía correctamente, False en caso contrario
        """
        # Crear contenido HTML para el informe
        html_message = self._create_html_report(
            message=message,
            report_type=report_type,
            data=data or {}
        )
        
        # Configurar adjuntos
        attachments = []
        if attachment_path:
            attachments.append(attachment_path)
            
        return await self.send(
            recipient=recipient,
            subject=subject,
            message=message,
            html_message=html_message,
            attachments=attachments
        )
        
    def _create_email_message(
        self,
        recipient: str,
        subject: str,
        message: str,
        html_message: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> MIMEMultipart:
        """
        Crear un mensaje de correo electrónico.
        
        Args:
            recipient: Dirección de correo del destinatario
            subject: Asunto del mensaje
            message: Contenido del mensaje en texto plano
            html_message: Contenido HTML opcional
            attachments: Lista de rutas a archivos adjuntos
            headers: Cabeceras adicionales
            
        Returns:
            Mensaje MIME
        """
        # Crear mensaje
        email = MIMEMultipart("alternative" if html_message else "mixed")
        email["Subject"] = subject
        email["From"] = self.config.default_sender or self.config.username
        email["To"] = recipient
        email["Date"] = formatdate(localtime=True)
        
        # Añadir cabeceras adicionales
        if headers:
            for name, value in headers.items():
                email[name] = value
                
        # Añadir parte de texto plano
        email.attach(MIMEText(message, "plain"))
        
        # Añadir parte HTML si está disponible
        if html_message:
            email.attach(MIMEText(html_message, "html"))
            
        # Añadir adjuntos
        if attachments:
            for path in attachments:
                try:
                    with open(path, "rb") as file:
                        part = MIMEApplication(file.read(), Name=path.split("/")[-1])
                    part["Content-Disposition"] = f'attachment; filename="{path.split("/")[-1]}"'
                    email.attach(part)
                except Exception as e:
                    self.logger.error(f"Error adjuntando archivo {path}: {e}")
                    
        return email
        
    def _create_html_alert(
        self,
        message: str,
        alert_type: str,
        priority: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Crear contenido HTML para una alerta.
        
        Args:
            message: Mensaje de la alerta
            alert_type: Tipo de alerta
            priority: Prioridad
            data: Datos adicionales
            
        Returns:
            Contenido HTML
        """
        # Configurar color según prioridad
        color = {
            "low": "#007bff",      # Azul
            "normal": "#28a745",   # Verde
            "high": "#ffc107",     # Amarillo
            "critical": "#dc3545"  # Rojo
        }.get(priority, "#28a745")
        
        # Extraer detalles relevantes
        details = {}
        if alert_type == "price_alert":
            details = {
                "Símbolo": data.get("symbol", ""),
                "Precio": data.get("price", ""),
                "Condición": data.get("condition", ""),
                "Precio de referencia": data.get("reference_price", "")
            }
        elif alert_type == "volatility_alert":
            details = {
                "Símbolo": data.get("symbol", ""),
                "Volatilidad": f"{data.get('volatility', '')}",
                "Umbral": f"{data.get('threshold', '')}",
                "Periodo": data.get("period", "")
            }
        elif alert_type == "volume_alert":
            details = {
                "Símbolo": data.get("symbol", ""),
                "Volumen": data.get("volume", ""),
                "Volumen normal": data.get("avg_volume", ""),
                "Incremento": f"{data.get('increase', '')}%"
            }
        elif alert_type == "pattern_alert":
            details = {
                "Símbolo": data.get("symbol", ""),
                "Patrón": data.get("pattern", ""),
                "Timeframe": data.get("timeframe", ""),
                "Confianza": f"{data.get('confidence', '')}%"
            }
            
        # Crear tabla HTML para detalles
        details_html = ""
        if details:
            details_html = "<table style='width: 100%; border-collapse: collapse;'>"
            for key, value in details.items():
                details_html += f"<tr><td style='padding: 8px; border: 1px solid #ddd; text-align: left; font-weight: bold;'>{key}</td>"
                details_html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: left;'>{value}</td></tr>"
            details_html += "</table>"
            
        # Crear HTML
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .alert-container {{ border: 1px solid {color}; border-radius: 5px; margin: 20px auto; padding: 20px; max-width: 600px; }}
                .alert-header {{ background-color: {color}; color: white; padding: 10px; margin: -20px -20px 15px -20px; border-radius: 5px 5px 0 0; }}
                .alert-priority {{ float: right; font-weight: normal; }}
                .alert-time {{ color: #666; font-size: 0.9em; margin-top: 15px; }}
                .alert-footer {{ margin-top: 20px; font-size: 0.8em; color: #666; }}
            </style>
        </head>
        <body>
            <div class="alert-container">
                <div class="alert-header">
                    <h2 style="margin: 0;">{alert_type.replace('_', ' ').title()} <span class="alert-priority">Prioridad: {priority.upper()}</span></h2>
                </div>
                
                <p>{message}</p>
                
                {details_html}
                
                <p class="alert-time">Generado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                
                <div class="alert-footer">
                    <p>Este es un mensaje automático del sistema Genesis. Por favor no responda a este correo.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
        
    def _create_html_report(
        self,
        message: str,
        report_type: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Crear contenido HTML para un informe.
        
        Args:
            message: Mensaje del informe
            report_type: Tipo de informe
            data: Datos adicionales
            
        Returns:
            Contenido HTML
        """
        # Configurar detalles según tipo de informe
        title = report_type.replace('_', ' ').title()
        
        # Extraer detalles relevantes
        details_html = ""
        if report_type == "daily_performance":
            # Crear tabla de rendimiento
            details = {
                "Fecha": data.get("date", datetime.now().strftime("%d/%m/%Y")),
                "Rentabilidad": f"{data.get('profit_pct', '0.0')}%",
                "Operaciones": f"{data.get('trades', '0')}",
                "Ganadoras": f"{data.get('winning_trades', '0')}",
                "Perdedoras": f"{data.get('losing_trades', '0')}"
            }
            
            details_html = "<h3>Resumen de rendimiento</h3>"
            details_html += "<table style='width: 100%; border-collapse: collapse;'>"
            for key, value in details.items():
                details_html += f"<tr><td style='padding: 8px; border: 1px solid #ddd; text-align: left; font-weight: bold;'>{key}</td>"
                details_html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: left;'>{value}</td></tr>"
            details_html += "</table>"
            
            # Añadir top operaciones si existen
            if "top_trades" in data:
                details_html += "<h3>Mejores operaciones</h3>"
                details_html += "<table style='width: 100%; border-collapse: collapse;'>"
                details_html += "<tr><th style='padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;'>Símbolo</th>"
                details_html += "<th style='padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;'>Tipo</th>"
                details_html += "<th style='padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;'>Rentabilidad</th></tr>"
                
                for trade in data.get("top_trades", [])[:5]:
                    details_html += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'>{trade.get('symbol', '')}</td>"
                    details_html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{trade.get('side', '')}</td>"
                    details_html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{trade.get('profit_pct', '')}%</td></tr>"
                    
                details_html += "</table>"
                
        elif report_type == "weekly_summary":
            # Crear resumen semanal
            details = {
                "Periodo": data.get("period", ""),
                "Rentabilidad": f"{data.get('profit_pct', '0.0')}%",
                "Operaciones totales": f"{data.get('total_trades', '0')}",
                "Ratio de éxito": f"{data.get('win_rate', '0.0')}%",
                "Mejor día": data.get("best_day", ""),
                "Peor día": data.get("worst_day", "")
            }
            
            details_html = "<h3>Resumen semanal</h3>"
            details_html += "<table style='width: 100%; border-collapse: collapse;'>"
            for key, value in details.items():
                details_html += f"<tr><td style='padding: 8px; border: 1px solid #ddd; text-align: left; font-weight: bold;'>{key}</td>"
                details_html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: left;'>{value}</td></tr>"
            details_html += "</table>"
            
        elif report_type == "monthly_analysis":
            # Crear análisis mensual
            details = {
                "Mes": data.get("month", ""),
                "Rentabilidad": f"{data.get('profit_pct', '0.0')}%",
                "Operaciones": f"{data.get('total_trades', '0')}",
                "Ratio de éxito": f"{data.get('win_rate', '0.0')}%",
                "Drawdown máximo": f"{data.get('max_drawdown', '0.0')}%",
                "Ratio de Sharpe": f"{data.get('sharpe_ratio', '0.0')}",
                "Mejor estrategia": data.get("best_strategy", "")
            }
            
            details_html = "<h3>Análisis mensual</h3>"
            details_html += "<table style='width: 100%; border-collapse: collapse;'>"
            for key, value in details.items():
                details_html += f"<tr><td style='padding: 8px; border: 1px solid #ddd; text-align: left; font-weight: bold;'>{key}</td>"
                details_html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: left;'>{value}</td></tr>"
            details_html += "</table>"
            
        # Crear HTML
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .report-container {{ border: 1px solid #ddd; border-radius: 5px; margin: 20px auto; padding: 20px; max-width: 800px; }}
                .report-header {{ background-color: #f8f9fa; padding: 15px; margin: -20px -20px 15px -20px; border-bottom: 1px solid #ddd; border-radius: 5px 5px 0 0; }}
                .report-title {{ color: #007bff; margin: 0; }}
                .report-time {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
                .report-content {{ margin-bottom: 20px; }}
                .report-footer {{ margin-top: 30px; font-size: 0.8em; color: #666; border-top: 1px solid #ddd; padding-top: 15px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="report-container">
                <div class="report-header">
                    <h1 class="report-title">{title}</h1>
                    <p class="report-time">Generado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                </div>
                
                <div class="report-content">
                    <p>{message}</p>
                    
                    {details_html}
                </div>
                
                <div class="report-footer">
                    <p>Este informe ha sido generado automáticamente por el sistema Genesis.</p>
                    <p>Si tiene alguna pregunta, no responda a este correo. Contacte al administrador del sistema.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
        
    async def _consume_queue(self) -> None:
        """Consumidor de cola de emails pendientes."""
        while True:
            try:
                # Verificar y resetear contador de período
                now = datetime.now()
                if (now - self.sending_period_start).total_seconds() > 3600:  # 1 hora
                    self.sending_period_start = now
                    self.sent_count = 0
                    
                # Verificar límite de tasa
                if self.sent_count >= self.config.rate_limit:
                    # Esperar hasta el siguiente período
                    wait_time = 3600 - (now - self.sending_period_start).total_seconds()
                    if wait_time > 0:
                        self.logger.warning(f"Límite de tasa alcanzado. Esperando {wait_time:.1f} segundos")
                        await asyncio.sleep(wait_time)
                        continue
                        
                # Obtener siguiente email de la cola
                email = await self.pending_emails.get()
                
                # Enviar email
                success = await self._send_email(email)
                
                if success:
                    self.sent_count += 1
                    
                # Espacio entre envíos para evitar sobrecarga
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error en consumidor de emails: {e}")
                await asyncio.sleep(5)  # Esperar antes de reintentar
                
    async def _send_email(self, email: MIMEMultipart) -> bool:
        """
        Enviar un email usando SMTP.
        
        Args:
            email: Mensaje MIME
            
        Returns:
            True si se envía correctamente, False en caso contrario
        """
        # Extraer destinatario para registro
        recipient = email["To"]
        
        try:
            # Establecer conexión si no existe
            if not self.connection:
                self.connection = smtplib.SMTP(self.config.host, self.config.port)
                
                if self.config.use_tls:
                    self.connection.starttls()
                    
                if self.config.username and self.config.password:
                    self.connection.login(self.config.username, self.config.password)
                    
            # Enviar email
            self.connection.send_message(email)
            
            # Actualizar historial del destinatario
            if recipient not in self.recipient_history:
                self.recipient_history[recipient] = []
                
            self.recipient_history[recipient].append(datetime.now())
            
            self.logger.info(f"Email enviado a {recipient}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error enviando email a {recipient}: {e}")
            
            # Reiniciar conexión en caso de error
            if self.connection:
                try:
                    self.connection.quit()
                except Exception:
                    pass
                finally:
                    self.connection = None
                    
            return False
            
    def _check_recipient_rate_limit(self, recipient: str) -> bool:
        """
        Verificar límite de tasa para un destinatario.
        
        Args:
            recipient: Dirección de correo
            
        Returns:
            True si está por debajo del límite, False en caso contrario
        """
        now = datetime.now()
        
        # Inicializar historial si no existe
        if recipient not in self.recipient_history:
            self.recipient_history[recipient] = []
            return True
            
        # Filtrar envíos en la última hora
        recent_sends = [
            t for t in self.recipient_history[recipient]
            if (now - t).total_seconds() < 3600
        ]
        
        # Actualizar historial
        self.recipient_history[recipient] = recent_sends
        
        # Verificar límite (máximo 10 emails por hora a un mismo destinatario)
        return len(recent_sends) < 10
        
    def _clean_recent_messages(self) -> None:
        """Limpiar mensajes recientes duplicados (más de 5 minutos)."""
        # Implementar si es necesario mantener un historial más largo
        # Por ahora simplemente limitamos el tamaño del conjunto
        if len(self.recent_messages) > 500:
            self.recent_messages = set()
            
    def configure(self, config: EmailConfig) -> None:
        """
        Actualizar la configuración.
        
        Args:
            config: Nueva configuración
        """
        self.config = config
        
        # Reiniciar conexión para aplicar nueva configuración
        if self.connection:
            try:
                self.connection.quit()
            except Exception:
                pass
            finally:
                self.connection = None
                
        self.logger.info("Configuración de email actualizada")