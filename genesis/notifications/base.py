"""
Clases base para las notificaciones.

Este módulo define interfaces para todos los tipos de notificaciones
soportados por el sistema Genesis.
"""

import abc
from typing import Dict, Any, List, Optional
from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class NotificationChannel(abc.ABC):
    """Clase base abstracta para canales de notificación."""
    
    def __init__(self, name: str):
        """
        Inicializar el canal de notificación.
        
        Args:
            name: Nombre del canal
        """
        self.name = name
        self.logger = setup_logging(f"notification.{name}")
    
    @abc.abstractmethod
    async def send(self, 
                  recipient: str, 
                  subject: str, 
                  message: str, 
                  **kwargs) -> bool:
        """
        Enviar una notificación.
        
        Args:
            recipient: Destinatario de la notificación
            subject: Asunto de la notificación
            message: Cuerpo del mensaje
            **kwargs: Parámetros adicionales específicos del canal
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        pass


class NotificationManager(Component):
    """
    Gestor de notificaciones para el sistema Genesis.
    
    Este componente coordina el envío de notificaciones a través
    de diferentes canales.
    """
    
    def __init__(self, name: str = "notification_manager"):
        """
        Inicializar el gestor de notificaciones.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        self.channels: Dict[str, NotificationChannel] = {}
        self.default_channel: Optional[str] = None
    
    def register_channel(self, channel: NotificationChannel, is_default: bool = False) -> None:
        """
        Registrar un canal de notificación.
        
        Args:
            channel: Canal de notificación
            is_default: Si es el canal predeterminado
        """
        self.channels[channel.name] = channel
        if is_default or self.default_channel is None:
            self.default_channel = channel.name
        self.logger.info(f"Canal de notificación registrado: {channel.name}")
    
    async def start(self) -> None:
        """Iniciar el gestor de notificaciones."""
        await super().start()
        self.logger.info("Gestor de notificaciones iniciado")
    
    async def stop(self) -> None:
        """Detener el gestor de notificaciones."""
        await super().stop()
        self.logger.info("Gestor de notificaciones detenido")
    
    async def send_notification(self, 
                              recipient: str, 
                              subject: str, 
                              message: str,
                              channel: Optional[str] = None,
                              **kwargs) -> bool:
        """
        Enviar una notificación.
        
        Args:
            recipient: Destinatario de la notificación
            subject: Asunto de la notificación
            message: Cuerpo del mensaje
            channel: Canal a utilizar (opcional, usa el predeterminado si no se especifica)
            **kwargs: Parámetros adicionales específicos del canal
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        # Determinar el canal a utilizar
        channel_name = channel or self.default_channel
        if not channel_name or channel_name not in self.channels:
            self.logger.error(f"Canal de notificación no encontrado: {channel_name}")
            return False
        
        # Obtener el canal
        notification_channel = self.channels[channel_name]
        
        # Enviar la notificación
        try:
            result = await notification_channel.send(recipient, subject, message, **kwargs)
            if result:
                self.logger.info(f"Notificación enviada a {recipient} a través de {channel_name}")
                # Emitir evento de notificación enviada
                await self.emit_event("notification.sent", {
                    "recipient": recipient,
                    "subject": subject,
                    "channel": channel_name
                })
            else:
                self.logger.error(f"Error al enviar notificación a {recipient} a través de {channel_name}")
            return result
        except Exception as e:
            self.logger.error(f"Excepción al enviar notificación: {e}")
            return False
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Este método procesa eventos que pueden generar notificaciones,
        como alertas, trades, etc.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Procesar eventos que puedan generar notificaciones
        if event_type == "alert.triggered":
            # Generar notificación de alerta
            await self._handle_alert(data)
        elif event_type == "trade.opened" or event_type == "trade.closed":
            # Generar notificación de trade
            await self._handle_trade_event(event_type, data)
    
    async def _handle_alert(self, data: Dict[str, Any]) -> None:
        """
        Manejar una alerta y enviar notificación.
        
        Args:
            data: Datos de la alerta
        """
        # Extraer datos de la alerta
        alert_type = data.get("type", "unknown")
        symbol = data.get("symbol", "unknown")
        message = data.get("message", "Alerta sin mensaje")
        severity = data.get("severity", "info")
        recipients = data.get("recipients", [])
        
        # Si no hay destinatarios, usar los predeterminados
        if not recipients:
            # Aquí se podría consultar una lista de destinatarios predeterminados
            # Por ahora, simplemente logueamos el error
            self.logger.warning(f"Alerta sin destinatarios: {alert_type} - {symbol}")
            return
        
        # Crear asunto y mensaje
        subject = f"Alerta de Trading [{severity.upper()}]: {alert_type} - {symbol}"
        body = f"""
        Se ha detectado una alerta en el sistema Genesis:
        
        Símbolo: {symbol}
        Tipo: {alert_type}
        Severidad: {severity}
        
        Detalles:
        {message}
        
        Este mensaje ha sido generado automáticamente, no responda a este correo.
        """
        
        # Enviar a cada destinatario
        for recipient in recipients:
            await self.send_notification(recipient, subject, body)
    
    async def _handle_trade_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Manejar un evento de trade y enviar notificación.
        
        Args:
            event_type: Tipo de evento (trade.opened o trade.closed)
            data: Datos del evento
        """
        # Extraer datos del trade
        trade_id = data.get("trade_id", "unknown")
        symbol = data.get("symbol", "unknown")
        side = data.get("side", "unknown")
        price = data.get("price", 0)
        recipients = data.get("recipients", [])
        
        # Si no hay destinatarios, usar los predeterminados
        if not recipients:
            # Aquí se podría consultar una lista de destinatarios predeterminados
            # Por ahora, simplemente logueamos el error
            self.logger.warning(f"Evento de trade sin destinatarios: {event_type} - {trade_id}")
            return
        
        # Determinar si es apertura o cierre
        is_open = event_type == "trade.opened"
        event_name = "abierto" if is_open else "cerrado"
        
        # Crear asunto y mensaje
        subject = f"Trade {event_name}: {symbol} - {side.upper()}"
        
        # Construir mensaje según sea apertura o cierre
        if is_open:
            body = f"""
            Se ha {event_name} un nuevo trade:
            
            ID: {trade_id}
            Símbolo: {symbol}
            Dirección: {side.upper()}
            Precio de entrada: {price}
            
            Este mensaje ha sido generado automáticamente, no responda a este correo.
            """
        else:
            # Para cierre, incluir información de ganancias/pérdidas
            entry_price = data.get("entry_price", 0)
            profit = data.get("profit", 0)
            profit_percent = data.get("profit_percent", 0)
            
            body = f"""
            Se ha {event_name} un trade:
            
            ID: {trade_id}
            Símbolo: {symbol}
            Dirección: {side.upper()}
            Precio de entrada: {entry_price}
            Precio de salida: {price}
            Resultado: {'Ganancia' if profit >= 0 else 'Pérdida'} de {profit} ({profit_percent:.2f}%)
            
            Este mensaje ha sido generado automáticamente, no responda a este correo.
            """
        
        # Enviar a cada destinatario
        for recipient in recipients:
            await self.send_notification(recipient, subject, body)