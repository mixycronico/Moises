"""
Gestor de alertas para el sistema Genesis.

Este módulo proporciona funcionalidades para crear, gestionar y enviar
alertas basadas en diferentes condiciones del mercado y del sistema.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime, timedelta
from enum import Enum, auto

from genesis.core.base import Component
from genesis.utils.helpers import generate_id, format_timestamp

class AlertType(Enum):
    """Tipos de alertas disponibles en el sistema."""
    PRICE = auto()          # Alertas basadas en precio
    VOLUME = auto()         # Alertas basadas en volumen
    TREND = auto()          # Alertas de cambio de tendencia
    INDICATOR = auto()      # Alertas basadas en indicadores técnicos
    VOLATILITY = auto()     # Alertas de volatilidad
    PATTERN = auto()        # Alertas de patrones de velas
    ARBITRAGE = auto()      # Alertas de oportunidades de arbitraje
    STRATEGY = auto()       # Alertas relacionadas con estrategias
    SYSTEM = auto()         # Alertas del sistema
    CUSTOM = auto()         # Alertas personalizadas

class AlertStatus(Enum):
    """Estados posibles para las alertas."""
    ACTIVE = auto()         # Alerta activa
    TRIGGERED = auto()      # Alerta disparada
    DISABLED = auto()       # Alerta desactivada
    EXPIRED = auto()        # Alerta expirada
    PENDING = auto()        # Alerta pendiente de activación

class AlertCondition:
    """
    Clase para representar una condición de alerta.
    
    Esta clase define las condiciones para disparar una alerta,
    incluyendo el tipo de condición y los parámetros específicos.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        alert_type: AlertType = AlertType.PRICE,
        threshold: float = 0.0,
        comparison: str = ">",
        timeframe: str = "1h",
        symbols: List[str] = None,
        symbol: str = "",
        condition_type: str = "",
        params: Dict[str, Any] = None,
        message_template: str = "",
        strategy_id: str = ""
    ):
        """
        Inicializar una condición de alerta.
        
        Args:
            name: Nombre de la alerta
            description: Descripción de la alerta
            alert_type: Tipo de alerta (enum AlertType)
            threshold: Umbral para la condición
            comparison: Tipo de comparación (">", "<", "==", etc.)
            timeframe: Marco temporal (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            symbols: Lista de símbolos para la alerta (para alertas multi-símbolo)
            symbol: Símbolo de trading (para compatibilidad)
            condition_type: Tipo de condición (para compatibilidad)
            params: Parámetros específicos para la condición (para compatibilidad)
            message_template: Plantilla para el mensaje de alerta
            strategy_id: ID de la estrategia (para alertas de tipo STRATEGY)
        """
        self.id = generate_id()
        self.name = name
        self.description = description
        self.alert_type = alert_type
        
        # Compatibilidad con versión anterior
        if not symbol and symbols and len(symbols) > 0:
            symbol = symbols[0]
        self.symbol = symbol
        
        # Determinar condition_type basado en alert_type si no se proporciona
        if not condition_type:
            if alert_type == AlertType.PRICE:
                condition_type = "price"
            elif alert_type == AlertType.VOLUME:
                condition_type = "volume"
            elif alert_type == AlertType.TREND:
                condition_type = "trend"
            elif alert_type == AlertType.INDICATOR:
                condition_type = "indicator"
            else:
                condition_type = alert_type.name.lower()
        self.condition_type = condition_type
        
        # Construir parámetros
        if params is None:
            params = {
                "threshold": threshold,
                "comparison": comparison,
                "timeframe": timeframe
            }
            if symbols:
                params["symbols"] = symbols
            if strategy_id:
                params["strategy_id"] = strategy_id
                
        self.params = params
        
        # Crear un mensaje por defecto si no se proporciona
        if not message_template:
            if alert_type == AlertType.PRICE:
                symbol_str = symbols[0] if symbols else symbol if symbol else "crypto"
                message_template = f"Alerta de precio para {symbol_str}: precio {{price}} {comparison} {threshold}"
            elif alert_type == AlertType.TREND:
                symbol_str = symbols[0] if symbols else symbol if symbol else "mercado"
                message_template = f"Cambio de tendencia detectado para {symbol_str}"
            elif alert_type == AlertType.STRATEGY:
                message_template = f"Señal de estrategia {strategy_id}: {{message}}"
            else:
                message_template = f"Alerta: {{message}}"
                
        self.message_template = message_template
        self.enabled = True
        self.created_at = datetime.now()
        self.last_triggered = None
        self.trigger_count = 0
        self.status = AlertStatus.ACTIVE
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con los datos de la condición
        """
        return {
            "id": self.id,
            "name": self.name,
            "symbol": self.symbol,
            "condition_type": self.condition_type,
            "params": self.params,
            "message_template": self.message_template,
            "enabled": self.enabled,
            "created_at": format_timestamp(self.created_at),
            "last_triggered": format_timestamp(self.last_triggered) if self.last_triggered else None,
            "trigger_count": self.trigger_count
        }

class AlertManager(Component):
    """
    Gestor de alertas para el sistema de trading.
    
    Este componente gestiona las condiciones de alerta, evalúa los datos
    del mercado contra estas condiciones, y envía notificaciones cuando
    se activan las alertas.
    """
    
    def __init__(self, name: str = "alert_manager"):
        """
        Inicializar el gestor de alertas.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        self.conditions: Dict[str, AlertCondition] = {}
        self.notification_channels: Dict[str, Any] = {}
        self.user_subscriptions: Dict[str, List[str]] = {}  # user_id -> [condition_ids]
        
        # Cache de datos recientes para evaluación de condiciones
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        self.indicator_cache: Dict[str, Dict[str, Any]] = {}
        
        # Conjunto para evitar alertas duplicadas en corto tiempo
        self.recent_alerts: Set[str] = set()
        
        # Contadores para estadísticas
        self.alert_stats = {
            "alerts_added": 0,
            "alerts_triggered": 0,
            "alerts_updated": 0,
            "alerts_deleted": 0
        }
        
    async def start(self) -> None:
        """Iniciar el gestor de alertas."""
        await super().start()
        self.logger.info("Gestor de alertas iniciado")
        
    async def stop(self) -> None:
        """Detener el gestor de alertas."""
        await super().stop()
        self.logger.info("Gestor de alertas detenido")
        
    async def add_condition(self, condition: AlertCondition) -> Dict[str, Any]:
        """
        Añadir una nueva condición de alerta.
        
        Args:
            condition: Condición de alerta a añadir
            
        Returns:
            Resultado de la operación con ID de la condición
        """
        if not condition.id:
            condition.id = generate_id()
            
        # Guardar condición
        self.conditions[condition.id] = condition
        
        # Actualizar estadísticas
        self.alert_stats["alerts_added"] += 1
        
        self.logger.info(f"Condición de alerta añadida: {condition.name} (ID: {condition.id})")
        
        return {
            "success": True,
            "condition_id": condition.id,
            "condition": condition.to_dict()
        }
    
    async def get_condition_count(self) -> int:
        """
        Obtener número total de condiciones registradas.
        
        Returns:
            Número de condiciones
        """
        return len(self.conditions)
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente origen
        """
        if event_type == "market_data.update":
            symbol = data.get("symbol")
            if not symbol:
                return
                
            # Actualizar cache
            self.market_data_cache[symbol] = data
            
            # Evaluar condiciones para este símbolo
            await self._evaluate_conditions(symbol)
            
        elif event_type == "indicator.update":
            symbol = data.get("symbol")
            if not symbol:
                return
                
            # Actualizar cache de indicadores
            if symbol not in self.indicator_cache:
                self.indicator_cache[symbol] = {}
                
            indicator = data.get("indicator")
            if indicator:
                self.indicator_cache[symbol][indicator] = data
                
            # Evaluar condiciones para este símbolo
            await self._evaluate_conditions(symbol)
            
    async def create_alert(
        self,
        name: str,
        user_id: str,
        symbol: str,
        condition_type: str,
        params: Dict[str, Any],
        message_template: Optional[str] = None,
        notification_channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Crear una nueva alerta.
        
        Args:
            name: Nombre de la alerta
            user_id: ID del usuario
            symbol: Símbolo de trading
            condition_type: Tipo de condición (precio, indicador, etc.)
            params: Parámetros específicos para la condición
            message_template: Plantilla para el mensaje (opcional)
            notification_channels: Canales para notificación (opcional)
            
        Returns:
            Diccionario con el resultado de la operación
        """
        # Crear mensaje por defecto si no se proporciona
        if not message_template:
            if condition_type == "price":
                message_template = f"Alerta de precio para {symbol}: {{price}} {params.get('operator', '')} {params.get('value', '')}"
            elif condition_type == "indicator":
                message_template = f"Alerta de indicador {params.get('indicator', '')} para {symbol}: {{value}} {params.get('operator', '')} {params.get('threshold', '')}"
            elif condition_type == "volume":
                message_template = f"Alerta de volumen para {symbol}: Volumen actual {{volume}}"
            elif condition_type == "volatility":
                message_template = f"Alerta de volatilidad para {symbol}: Volatilidad {{volatility}}"
            else:
                message_template = f"Alerta para {symbol}: {{message}}"
                
        # Crear condición
        condition = AlertCondition(
            name=name,
            symbol=symbol,
            condition_type=condition_type,
            params=params,
            message_template=message_template
        )
        
        # Guardar condición
        self.conditions[condition.id] = condition
        
        # Asociar al usuario
        if user_id not in self.user_subscriptions:
            self.user_subscriptions[user_id] = []
            
        self.user_subscriptions[user_id].append(condition.id)
        
        # Registrar asociación con canales de notificación
        if notification_channels:
            # TODO: Implementar asociación con canales específicos
            pass
            
        self.logger.info(f"Alerta creada: {name} para {symbol} (ID: {condition.id})")
        
        return {
            "success": True,
            "condition_id": condition.id,
            "condition": condition.to_dict()
        }
        
    async def update_alert(
        self,
        condition_id: str,
        name: Optional[str] = None,
        enabled: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        message_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Actualizar una alerta existente.
        
        Args:
            condition_id: ID de la condición
            name: Nuevo nombre (opcional)
            enabled: Estado habilitado/deshabilitado (opcional)
            params: Nuevos parámetros (opcional)
            message_template: Nueva plantilla de mensaje (opcional)
            
        Returns:
            Diccionario con el resultado de la operación
        """
        if condition_id not in self.conditions:
            return {
                "success": False,
                "error": f"Condición {condition_id} no encontrada"
            }
            
        condition = self.conditions[condition_id]
        
        # Actualizar campos
        if name is not None:
            condition.name = name
            
        if enabled is not None:
            condition.enabled = enabled
            
        if params is not None:
            # Actualizar solo los parámetros proporcionados
            for key, value in params.items():
                condition.params[key] = value
                
        if message_template is not None:
            condition.message_template = message_template
            
        self.logger.info(f"Alerta actualizada: {condition_id}")
        
        return {
            "success": True,
            "condition": condition.to_dict()
        }
        
    async def delete_alert(self, condition_id: str) -> Dict[str, Any]:
        """
        Eliminar una alerta.
        
        Args:
            condition_id: ID de la condición
            
        Returns:
            Diccionario con el resultado de la operación
        """
        if condition_id not in self.conditions:
            return {
                "success": False,
                "error": f"Condición {condition_id} no encontrada"
            }
            
        # Eliminar de las condiciones
        condition = self.conditions.pop(condition_id)
        
        # Eliminar de las suscripciones de usuarios
        for user_id, condition_ids in self.user_subscriptions.items():
            if condition_id in condition_ids:
                self.user_subscriptions[user_id].remove(condition_id)
                
        self.logger.info(f"Alerta eliminada: {condition.name} (ID: {condition_id})")
        
        return {
            "success": True,
            "condition_id": condition_id
        }
        
    def get_user_alerts(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Obtener alertas de un usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Lista de condiciones de alerta
        """
        if user_id not in self.user_subscriptions:
            return []
            
        condition_ids = self.user_subscriptions[user_id]
        results = []
        
        for condition_id in condition_ids:
            if condition_id in self.conditions:
                results.append(self.conditions[condition_id].to_dict())
                
        return results
        
    def get_alert(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener detalles de una alerta.
        
        Args:
            condition_id: ID de la condición
            
        Returns:
            Detalles de la condición o None si no existe
        """
        if condition_id not in self.conditions:
            return None
            
        return self.conditions[condition_id].to_dict()
        
    async def _evaluate_conditions(self, symbol: str) -> None:
        """
        Evaluar condiciones para un símbolo.
        
        Args:
            symbol: Símbolo de trading
        """
        # Filtrar condiciones para este símbolo
        relevant_conditions = [
            condition for condition in self.conditions.values()
            if condition.symbol == symbol and condition.enabled
        ]
        
        if not relevant_conditions:
            return
            
        # Obtener datos para evaluación
        market_data = self.market_data_cache.get(symbol)
        if not market_data:
            return
            
        # Evaluar cada condición
        for condition in relevant_conditions:
            try:
                # Evaluar condición
                is_triggered, details = await self._evaluate_condition(condition, market_data)
                
                if is_triggered:
                    # Verificar duplicados recientes (últimos 5 minutos)
                    alert_key = f"{condition.id}:{datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    if alert_key in self.recent_alerts:
                        continue
                        
                    # Actualizar estado
                    condition.last_triggered = datetime.now()
                    condition.trigger_count += 1
                    
                    # Registrar como alerta reciente
                    self.recent_alerts.add(alert_key)
                    
                    # Limitar tamaño de conjunto
                    if len(self.recent_alerts) > 100:
                        self.recent_alerts = set(list(self.recent_alerts)[-50:])
                        
                    # Preparar mensaje
                    message = condition.message_template
                    for key, value in details.items():
                        message = message.replace(f"{{{key}}}", str(value))
                        
                    # Emitir evento de alerta
                    await self._trigger_alert(condition, message, details)
                    
            except Exception as e:
                self.logger.error(f"Error evaluando condición {condition.id}: {e}")
                
    async def _evaluate_condition(
        self, 
        condition: AlertCondition, 
        market_data: Dict[str, Any]
    ) -> tuple:
        """
        Evaluar una condición de alerta.
        
        Args:
            condition: Condición a evaluar
            market_data: Datos de mercado
            
        Returns:
            Tupla (is_triggered, details)
        """
        result = False
        details = {}
        
        # Extraer tipo de condición y parámetros
        condition_type = condition.condition_type
        params = condition.params
        
        if condition_type == "price":
            # Condición basada en precio
            price_key = params.get("price_key", "close")
            operator = params.get("operator", ">=")
            value = float(params.get("value", 0))
            
            # Obtener precio actual
            current_price = None
            if "ticker" in market_data:
                current_price = market_data["ticker"].get(price_key)
            elif "price_data" in market_data and market_data["price_data"]:
                current_price = market_data["price_data"][-1].get(price_key)
                
            if current_price is not None:
                # Evaluar condición
                if operator == ">=" and current_price >= value:
                    result = True
                elif operator == "<=" and current_price <= value:
                    result = True
                elif operator == ">" and current_price > value:
                    result = True
                elif operator == "<" and current_price < value:
                    result = True
                    
                details = {
                    "price": current_price,
                    "value": value,
                    "operator": operator
                }
                
        elif condition_type == "indicator":
            # Condición basada en indicador
            indicator_name = params.get("indicator")
            field = params.get("field", "value")
            operator = params.get("operator", ">=")
            threshold = float(params.get("threshold", 0))
            
            # Obtener valor del indicador
            indicator_value = None
            if indicator_name and condition.symbol in self.indicator_cache:
                indicator_data = self.indicator_cache[condition.symbol].get(indicator_name)
                if indicator_data and "values" in indicator_data:
                    values = indicator_data["values"]
                    if values:
                        indicator_value = values[-1].get(field)
                        
            if indicator_value is not None:
                # Evaluar condición
                if operator == ">=" and indicator_value >= threshold:
                    result = True
                elif operator == "<=" and indicator_value <= threshold:
                    result = True
                elif operator == ">" and indicator_value > threshold:
                    result = True
                elif operator == "<" and indicator_value < threshold:
                    result = True
                    
                details = {
                    "indicator": indicator_name,
                    "value": indicator_value,
                    "threshold": threshold,
                    "operator": operator
                }
                
        elif condition_type == "volume":
            # Condición basada en volumen
            baseline = params.get("baseline", "avg")  # avg, prev, fixed
            operator = params.get("operator", ">")
            multiplier = float(params.get("multiplier", 2.0))
            fixed_value = float(params.get("value", 0.0))
            
            # Obtener volumen actual
            current_volume = None
            prev_volume = None
            avg_volume = None
            
            if "ticker" in market_data:
                current_volume = market_data["ticker"].get("volume")
            elif "price_data" in market_data and len(market_data["price_data"]) > 1:
                price_data = market_data["price_data"]
                current_volume = price_data[-1].get("volume")
                if len(price_data) > 1:
                    prev_volume = price_data[-2].get("volume")
                    
                # Calcular volumen promedio (últimos 5 periodos)
                volumes = [p.get("volume", 0) for p in price_data[-5:] if "volume" in p]
                if volumes:
                    avg_volume = sum(volumes) / len(volumes)
                    
            if current_volume is not None:
                # Determinar valor de comparación
                compare_value = 0
                if baseline == "avg" and avg_volume is not None:
                    compare_value = avg_volume * multiplier
                elif baseline == "prev" and prev_volume is not None:
                    compare_value = prev_volume * multiplier
                elif baseline == "fixed":
                    compare_value = fixed_value
                    
                # Evaluar condición
                if operator == ">" and current_volume > compare_value:
                    result = True
                elif operator == "<" and current_volume < compare_value:
                    result = True
                    
                details = {
                    "volume": current_volume,
                    "compare_value": compare_value,
                    "avg_volume": avg_volume,
                    "operator": operator
                }
                
        elif condition_type == "volatility":
            # Condición basada en volatilidad
            period = int(params.get("period", 14))
            operator = params.get("operator", ">")
            threshold = float(params.get("threshold", 0.02))  # 2%
            
            if "price_data" in market_data and len(market_data["price_data"]) >= period:
                # Calcular volatilidad
                price_data = market_data["price_data"][-period:]
                prices = [p.get("close", 0) for p in price_data]
                
                if prices and len(prices) >= period:
                    returns = []
                    for i in range(1, len(prices)):
                        if prices[i-1] > 0:
                            returns.append((prices[i] - prices[i-1]) / prices[i-1])
                            
                    if returns:
                        volatility = sum(r*r for r in returns) / len(returns)
                        volatility = volatility**0.5  # Raíz cuadrada
                        
                        # Evaluar condición
                        if operator == ">" and volatility > threshold:
                            result = True
                        elif operator == "<" and volatility < threshold:
                            result = True
                            
                        details = {
                            "volatility": volatility,
                            "threshold": threshold,
                            "period": period,
                            "operator": operator
                        }
                        
        elif condition_type == "pattern":
            # Condición basada en patrón de velas
            pattern = params.get("pattern")
            confidence = float(params.get("min_confidence", 70.0))
            
            if "patterns" in market_data:
                patterns = market_data["patterns"]
                if pattern in patterns:
                    pattern_confidence = patterns[pattern].get("confidence", 0)
                    if pattern_confidence >= confidence:
                        result = True
                        details = {
                            "pattern": pattern,
                            "confidence": pattern_confidence,
                            "min_confidence": confidence
                        }
                        
        return (result, details)
        
    async def _trigger_alert(
        self, 
        condition: AlertCondition, 
        message: str, 
        details: Dict[str, Any]
    ) -> None:
        """
        Disparar una alerta.
        
        Args:
            condition: Condición activada
            message: Mensaje de la alerta
            details: Detalles adicionales
        """
        # Encontrar usuarios suscritos a esta alerta
        recipients = []
        for user_id, condition_ids in self.user_subscriptions.items():
            if condition.id in condition_ids:
                recipients.append(user_id)
                
        if not recipients:
            return
            
        # Preparar evento de alerta
        alert_data = {
            "condition_id": condition.id,
            "condition_name": condition.name,
            "condition_type": condition.condition_type,
            "symbol": condition.symbol,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "details": details,
            "recipients": recipients
        }
        
        # Emitir evento
        self.logger.info(f"Alerta disparada: {condition.name} para {condition.symbol}")
        await self.emit_event("alert.triggered", alert_data)
        
    async def send_alert(
        self, 
        user_id: str,
        alert_type: str,
        message: str,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Union[str, int, float]]] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Enviar una alerta manual.
        
        Args:
            user_id: ID del usuario destinatario
            alert_type: Tipo de alerta
            message: Mensaje
            symbol: Símbolo relacionado (opcional)
            details: Detalles adicionales (opcional)
            priority: Prioridad (low, normal, high, critical)
            
        Returns:
            Resultado de la operación
        """
        # Preparar datos de la alerta
        alert_data = {
            "alert_type": alert_type,
            "message": message,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "priority": priority,
            "recipients": [user_id],
            "details": details or {}
        }
        
        # Emitir evento
        self.logger.info(f"Alerta manual enviada a {user_id}: {alert_type}")
        await self.emit_event("alert.triggered", alert_data)
        
        return {
            "success": True,
            "alert_id": generate_id(),
            "timestamp": datetime.now().isoformat()
        }
        
    def register_notification_channel(self, channel_name: str, channel) -> None:
        """
        Registrar un canal de notificación.
        
        Args:
            channel_name: Nombre del canal
            channel: Instancia del canal
        """
        self.notification_channels[channel_name] = channel
        self.logger.info(f"Canal de notificación registrado: {channel_name}")
        
    def unregister_notification_channel(self, channel_name: str) -> bool:
        """
        Eliminar un canal de notificación.
        
        Args:
            channel_name: Nombre del canal
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        if channel_name in self.notification_channels:
            del self.notification_channels[channel_name]
            self.logger.info(f"Canal de notificación eliminado: {channel_name}")
            return True
        return False