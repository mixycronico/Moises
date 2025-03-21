"""
Gestor de alertas para el sistema Genesis.

Este módulo proporciona funcionalidades para crear, gestionar y enviar
alertas basadas en diferentes condiciones del mercado y del sistema.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from genesis.core.base import Component
from genesis.notifications.email_notifier import EmailNotifier

class AlertCondition:
    """
    Clase para representar una condición de alerta.
    
    Esta clase define las condiciones para disparar una alerta,
    incluyendo el tipo de condición y los parámetros específicos.
    """
    
    def __init__(
        self,
        name: str,
        symbol: str,
        condition_type: str,
        params: Dict[str, Any],
        message_template: str
    ):
        """
        Inicializar una condición de alerta.
        
        Args:
            name: Nombre de la alerta
            symbol: Símbolo de trading
            condition_type: Tipo de condición (precio, indicador, etc.)
            params: Parámetros específicos de la condición
            message_template: Plantilla para el mensaje de alerta
        """
        self.name = name
        self.symbol = symbol
        self.condition_type = condition_type
        self.params = params
        self.message_template = message_template
        self.last_triggered = None
        self.enabled = True
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario para almacenamiento o transmisión.
        
        Returns:
            Representación como diccionario
        """
        return {
            "name": self.name,
            "symbol": self.symbol,
            "condition_type": self.condition_type,
            "params": self.params,
            "message_template": self.message_template,
            "last_triggered": self.last_triggered,
            "enabled": self.enabled
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertCondition':
        """
        Crear una condición de alerta desde un diccionario.
        
        Args:
            data: Datos de la condición
            
        Returns:
            Instancia de AlertCondition
        """
        condition = cls(
            name=data["name"],
            symbol=data["symbol"],
            condition_type=data["condition_type"],
            params=data["params"],
            message_template=data["message_template"]
        )
        condition.last_triggered = data.get("last_triggered")
        condition.enabled = data.get("enabled", True)
        return condition
        
class AlertManager(Component):
    """
    Gestor de alertas del sistema.
    
    Este componente gestiona la creación, evaluación y envío de alertas
    basadas en diferentes condiciones y canales.
    """
    
    def __init__(
        self,
        email_notifier: EmailNotifier,
        name: str = "alert_manager"
    ):
        """
        Inicializar el gestor de alertas.
        
        Args:
            email_notifier: Instancia de EmailNotifier para enviar correos
            name: Nombre del componente
        """
        super().__init__(name)
        self.email_notifier = email_notifier
        self.conditions: Dict[str, AlertCondition] = {}
        self.throttle_times: Dict[str, int] = {}  # Para evitar envío excesivo
        self.logger = logging.getLogger(__name__)
        
    async def start(self) -> None:
        """Iniciar el gestor de alertas."""
        await super().start()
        self.logger.info("Gestor de alertas iniciado")
        
    async def stop(self) -> None:
        """Detener el gestor de alertas."""
        await super().stop()
        self.logger.info("Gestor de alertas detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        if event_type == "alert.add_condition":
            condition = AlertCondition(
                name=data.get("name", ""),
                symbol=data.get("symbol", ""),
                condition_type=data.get("condition_type", ""),
                params=data.get("params", {}),
                message_template=data.get("message_template", "")
            )
            self.add_condition(condition)
            
        elif event_type == "alert.remove_condition":
            name = data.get("name")
            if name:
                self.remove_condition(name)
                
        elif event_type == "alert.check_price":
            symbol = data.get("symbol")
            price = data.get("price")
            if symbol and price is not None:
                await self.check_price_alerts(symbol, price)
                
        elif event_type == "alert.send":
            alert_type = data.get("type")
            alert_data = data.get("data", {})
            recipients = data.get("recipients")
            await self.send_alert(alert_type, alert_data, recipients)
            
    def add_condition(self, condition: AlertCondition) -> bool:
        """
        Añadir una condición de alerta.
        
        Args:
            condition: Condición a añadir
            
        Returns:
            True si se añadió correctamente, False en caso contrario
        """
        if not condition.name:
            self.logger.error("No se puede añadir una condición sin nombre.")
            return False
            
        if condition.name in self.conditions:
            self.logger.warning(f"Reemplazando condición existente: {condition.name}")
            
        self.conditions[condition.name] = condition
        self.logger.info(f"Condición añadida: {condition.name}, tipo: {condition.condition_type}")
        return True
        
    def remove_condition(self, name: str) -> bool:
        """
        Eliminar una condición por su nombre.
        
        Args:
            name: Nombre de la condición
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        if name in self.conditions:
            del self.conditions[name]
            self.logger.info(f"Condición eliminada: {name}")
            return True
        else:
            self.logger.warning(f"Condición no encontrada: {name}")
            return False
            
    def enable_condition(self, name: str) -> bool:
        """
        Habilitar una condición.
        
        Args:
            name: Nombre de la condición
            
        Returns:
            True si se habilitó correctamente, False en caso contrario
        """
        if name in self.conditions:
            self.conditions[name].enabled = True
            self.logger.info(f"Condición habilitada: {name}")
            return True
        return False
        
    def disable_condition(self, name: str) -> bool:
        """
        Deshabilitar una condición.
        
        Args:
            name: Nombre de la condición
            
        Returns:
            True si se deshabilitó correctamente, False en caso contrario
        """
        if name in self.conditions:
            self.conditions[name].enabled = False
            self.logger.info(f"Condición deshabilitada: {name}")
            return True
        return False
        
    def get_conditions(self, filter_type: Optional[str] = None) -> List[AlertCondition]:
        """
        Obtener todas las condiciones o filtradas por tipo.
        
        Args:
            filter_type: Tipo de condición para filtrar
            
        Returns:
            Lista de condiciones
        """
        if filter_type:
            return [c for c in self.conditions.values() if c.condition_type == filter_type]
        else:
            return list(self.conditions.values())
            
    async def check_price_alerts(self, symbol: str, current_price: float) -> None:
        """
        Verificar alertas de precio para un símbolo.
        
        Args:
            symbol: Símbolo de trading
            current_price: Precio actual
        """
        triggered = []
        for condition in self.get_conditions("price"):
            if not condition.enabled or condition.symbol != symbol:
                continue
                
            # Evaluar la condición de precio
            if self._evaluate_price_condition(condition, current_price):
                # Verificar si está dentro del tiempo de limitación
                if self._can_trigger(condition.name):
                    triggered.append(condition)
                    condition.last_triggered = datetime.utcnow().isoformat()
                    
        # Enviar alertas para condiciones disparadas
        for condition in triggered:
            # Formato del mensaje
            message = self._format_message(condition, {
                "symbol": symbol,
                "price": current_price,
                "time": datetime.utcnow().isoformat()
            })
            
            # Emitir evento
            await self.emit_event("alert.triggered", {
                "condition_name": condition.name,
                "symbol": symbol,
                "price": current_price
            })
            
            # Enviar notificación por correo
            subject = f"Alerta de precio - {symbol}"
            details = {
                "Símbolo": symbol,
                "Precio": f"{current_price:.2f}",
                "Condición": condition.name,
                "Tipo": condition.condition_type
            }
            html_message = EmailNotifier.create_html_message("Alerta de Precio", details)
            
            # Usar email_notifier para enviar
            await self.email_notifier.send_email(
                subject=subject,
                message=html_message,
                recipients=condition.params.get("recipients"),
                html=True
            )
            
    def _evaluate_price_condition(self, condition: AlertCondition, price: float) -> bool:
        """
        Evaluar si una condición de precio se cumple.
        
        Args:
            condition: Condición a evaluar
            price: Precio actual
            
        Returns:
            True si la condición se cumple, False en caso contrario
        """
        params = condition.params
        condition_type = params.get("condition", "above")
        target_price = params.get("target_price", 0.0)
        
        if condition_type == "above" and price > target_price:
            return True
        elif condition_type == "below" and price < target_price:
            return True
        elif condition_type == "change":
            # Condición de cambio porcentual
            pct_change = params.get("pct_change", 5.0)
            reference_price = params.get("reference_price", price)
            change = abs(price - reference_price) / reference_price * 100
            return change >= pct_change
            
        return False
        
    def _can_trigger(self, condition_name: str) -> bool:
        """
        Verificar si una condición puede dispararse según límites de tiempo.
        
        Args:
            condition_name: Nombre de la condición
            
        Returns:
            True si puede dispararse, False en caso contrario
        """
        now = datetime.utcnow().timestamp()
        last_time = self.throttle_times.get(condition_name, 0)
        
        # Límite predeterminado de 5 minutos (300 segundos) entre alertas
        min_interval = 300
        
        if now - last_time >= min_interval:
            self.throttle_times[condition_name] = now
            return True
            
        return False
        
    def _format_message(self, condition: AlertCondition, data: Dict[str, Any]) -> str:
        """
        Formatear un mensaje de alerta.
        
        Args:
            condition: Condición de la alerta
            data: Datos para el formato
            
        Returns:
            Mensaje formateado
        """
        try:
            return condition.message_template.format(**data)
        except KeyError as e:
            self.logger.error(f"Error al formatear mensaje: {e}")
            return f"Alerta: {condition.name} disparada."
            
    async def send_alert(
        self, 
        alert_type: str, 
        alert_data: Dict[str, Any], 
        recipients: Optional[List[str]] = None
    ) -> None:
        """
        Enviar una alerta específica.
        
        Args:
            alert_type: Tipo de alerta (anomalía, estrategia, etc.)
            alert_data: Datos de la alerta
            recipients: Lista de destinatarios
        """
        if alert_type == "anomalia":
            await self.alerta_anomalia(
                symbol=alert_data.get("symbol", ""),
                z_score=alert_data.get("z_score", 0.0),
                price=alert_data.get("price", 0.0),
                mean=alert_data.get("mean", 0.0),
                std=alert_data.get("std", 0.0),
                destinatarios=recipients
            )
        elif alert_type == "estrategia":
            await self.alerta_estrategia(
                strategy_name=alert_data.get("strategy_name", ""),
                performance=alert_data.get("performance", 0.0),
                capital=alert_data.get("capital", 0.0),
                destinatarios=recipients
            )
        elif alert_type == "falla_sistema":
            await self.alerta_falla_sistema(
                error_message=alert_data.get("error_message", ""),
                destinatarios=recipients
            )
        elif alert_type == "kill_switch":
            await self.alerta_kill_switch(
                market_drop=alert_data.get("market_drop", 0.0),
                capital=alert_data.get("capital", 0.0),
                destinatarios=recipients
            )
        else:
            self.logger.warning(f"Tipo de alerta desconocido: {alert_type}")
            
    async def alerta_anomalia(
        self, 
        symbol: str, 
        z_score: float, 
        price: float, 
        mean: float, 
        std: float,
        destinatarios: Optional[List[str]] = None
    ) -> None:
        """
        Enviar una alerta de anomalía en el mercado con formato HTML.
        
        Args:
            symbol: Símbolo del par de trading
            z_score: Valor de la anomalía (z-score)
            price: Precio actual
            mean: Media del precio
            std: Desviación estándar
            destinatarios: Lista de destinatarios
        """
        try:
            asunto = f"🚨 Anomalía Detectada en {symbol}"
            details = {
                "Símbolo": symbol,
                "Z-Score": f"{z_score:.2f}",
                "Precio Actual": f"${price:.2f}",
                "Promedio": f"${mean:.2f}",
                "Desviación Estándar": f"${std:.2f}"
            }
            mensaje = EmailNotifier.create_html_message("Anomalía Detectada", details)
            await self.email_notifier.send_email(asunto, mensaje, destinatarios, html=True)
            self.logger.info(f"Alerta de anomalía enviada para {symbol}, z_score={z_score:.2f}")
        except Exception as e:
            self.logger.error(f"Error al enviar alerta de anomalía: {e}")
            
    async def alerta_estrategia(
        self, 
        strategy_name: str,
        performance: float, 
        capital: float,
        destinatarios: Optional[List[str]] = None
    ) -> None:
        """
        Enviar una alerta sobre el desempeño de una estrategia con formato HTML.
        
        Args:
            strategy_name: Nombre de la estrategia
            performance: Rendimiento de la estrategia
            capital: Capital actual
            destinatarios: Lista de destinatarios
        """
        try:
            asunto = f"📈 Actualización de Estrategia: {strategy_name}"
            details = {
                "Estrategia": strategy_name,
                "Rendimiento": f"{performance:.2f}%",
                "Capital Actual": f"${capital:.2f}"
            }
            mensaje = EmailNotifier.create_html_message("Actualización de Estrategia", details)
            await self.email_notifier.send_email(asunto, mensaje, destinatarios, html=True)
            self.logger.info(f"Alerta de estrategia enviada para {strategy_name}, rendimiento={performance:.2f}%")
        except Exception as e:
            self.logger.error(f"Error al enviar alerta de estrategia: {e}")
            
    async def alerta_falla_sistema(
        self, 
        error_message: str, 
        destinatarios: Optional[List[str]] = None
    ) -> None:
        """
        Enviar una alerta de fallo crítico del sistema con formato HTML.
        
        Args:
            error_message: Mensaje de error
            destinatarios: Lista de destinatarios
        """
        try:
            asunto = "⚠️ Falla Crítica en Genesis"
            details = {
                "Mensaje de Error": error_message,
                "Sistema": "Genesis Trading Platform",
                "Timestamp": datetime.utcnow().isoformat()
            }
            mensaje = EmailNotifier.create_html_message("Falla Crítica del Sistema", details)
            await self.email_notifier.send_email(asunto, mensaje, destinatarios, html=True)
            self.logger.critical(f"Alerta de falla crítica enviada: {error_message}")
        except Exception as e:
            self.logger.error(f"Error al enviar alerta de falla: {e}")
            
    async def alerta_kill_switch(
        self, 
        market_drop: float, 
        capital: float,
        destinatarios: Optional[List[str]] = None
    ) -> None:
        """
        Enviar una alerta cuando se activa el kill switch con formato HTML.
        
        Args:
            market_drop: Caída del mercado en porcentaje
            capital: Capital actual
            destinatarios: Lista de destinatarios
        """
        try:
            asunto = "🛑 Kill Switch Activado"
            details = {
                "Causa": f"Caída del mercado del {market_drop:.2%}",
                "Capital Actual": f"${capital:.2f}",
                "Acción": "Todo convertido a USDT",
                "Timestamp": datetime.utcnow().isoformat()
            }
            mensaje = EmailNotifier.create_html_message("Kill Switch Activado", details)
            await self.email_notifier.send_email(asunto, mensaje, destinatarios, html=True)
            self.logger.critical(f"Alerta de kill switch enviada: caída del {market_drop:.2%}")
        except Exception as e:
            self.logger.error(f"Error al enviar alerta de kill switch: {e}")
            
    async def enviar_alerta_genérica(
        self, 
        asunto: str, 
        mensaje: str, 
        destinatarios: Optional[List[str]] = None
    ) -> None:
        """
        Enviar una alerta genérica por correo electrónico.
        
        Args:
            asunto: Asunto del correo
            mensaje: Cuerpo del mensaje (texto plano)
            destinatarios: Lista de correos electrónicos
        """
        try:
            await self.email_notifier.send_email(asunto, mensaje, destinatarios, html=False)
            self.logger.info(f"Alerta genérica enviada: {asunto}")
        except Exception as e:
            self.logger.error(f"Error al enviar alerta genérica: {e}")
            
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener historial de alertas disparadas.
        
        Args:
            limit: Número máximo de alertas a devolver
            
        Returns:
            Lista de alertas disparadas
        """
        # En una implementación real, esto consultaría una base de datos
        # o un archivo de logs para obtener el historial.
        # Aquí simplemente devolvemos una lista de condiciones
        # que han sido disparadas recientemente.
        triggered = []
        for condition in self.conditions.values():
            if condition.last_triggered:
                triggered.append(condition.to_dict())
                
        # Ordenar por fecha de disparo (más reciente primero)
        triggered.sort(key=lambda x: x.get("last_triggered", ""), reverse=True)
        
        # Limitar resultados
        return triggered[:limit]