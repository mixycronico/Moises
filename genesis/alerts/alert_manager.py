"""
Gestor de alertas para el sistema Genesis.

Este módulo proporciona un gestor centralizado para monitorear métricas críticas del sistema
y generar alertas cuando se exceden los límites configurados.
"""

import logging
from typing import Dict, Any, Optional, List

class AlertManager:
    """
    Gestiona la generación y envío de alertas basadas en métricas del sistema.
    
    Esta clase monitorea métricas clave como drawdown, volatilidad, cambios de precio
    y balance, generando alertas cuando los valores exceden los límites configurados.
    """
    
    def __init__(self):
        """Inicializar el gestor de alertas."""
        self._logger = logging.getLogger(__name__)
        self._notifiers = {}
        
        # Límites por defecto (usando propiedades privadas)
        self._drawdown_limit = 15  # 15% de drawdown máximo
        self._volatility_limit = 25  # 25% de volatilidad máxima
        self._price_change_limit = 10  # 10% de cambio de precio máximo
        self._min_balance = 1000  # $1000 de balance mínimo
        
        # Historial de alertas para evitar spam
        self._alert_history = {}
        
    @property
    def drawdown_limit(self) -> float:
        """Obtener el límite de drawdown."""
        return self._drawdown_limit
        
    @drawdown_limit.setter
    def drawdown_limit(self, value: float) -> None:
        """
        Establecer el límite de drawdown.
        
        Args:
            value: Nuevo límite de drawdown (porcentaje)
            
        Raises:
            ValueError: Si el valor es negativo o mayor a 100
        """
        if value < 0:
            raise ValueError("El límite de drawdown no puede ser negativo")
        if value > 100:
            raise ValueError("El límite de drawdown no puede ser mayor al 100%")
        self._drawdown_limit = value
        
    @property
    def volatility_limit(self) -> float:
        """Obtener el límite de volatilidad."""
        return self._volatility_limit
        
    @volatility_limit.setter
    def volatility_limit(self, value: float) -> None:
        """
        Establecer el límite de volatilidad.
        
        Args:
            value: Nuevo límite de volatilidad (porcentaje)
            
        Raises:
            ValueError: Si el valor es negativo o mayor a 100
        """
        if value < 0:
            raise ValueError("El límite de volatilidad no puede ser negativo")
        if value > 100:
            raise ValueError("El límite de volatilidad no puede ser mayor al 100%")
        self._volatility_limit = value
        
    @property
    def price_change_limit(self) -> float:
        """Obtener el límite de cambio de precio."""
        return self._price_change_limit
        
    @price_change_limit.setter
    def price_change_limit(self, value: float) -> None:
        """
        Establecer el límite de cambio de precio.
        
        Args:
            value: Nuevo límite de cambio de precio (porcentaje)
            
        Raises:
            ValueError: Si el valor es negativo o mayor a 100
        """
        if value < 0:
            raise ValueError("El límite de cambio de precio no puede ser negativo")
        if value > 100:
            raise ValueError("El límite de cambio de precio no puede ser mayor al 100%")
        self._price_change_limit = value
        
    @property
    def min_balance(self) -> float:
        """Obtener el balance mínimo."""
        return self._min_balance
        
    @min_balance.setter
    def min_balance(self, value: float) -> None:
        """
        Establecer el balance mínimo.
        
        Args:
            value: Nuevo balance mínimo
            
        Raises:
            ValueError: Si el valor es negativo
        """
        if value < 0:
            raise ValueError("El balance mínimo no puede ser negativo")
        self._min_balance = value
    
    def add_notifier(self, name: str, notifier) -> None:
        """
        Añadir un mecanismo de notificación.
        
        Args:
            name: Identificador del notificador
            notifier: Objeto notificador con método send_notification
        """
        self._notifiers[name] = notifier
        self._logger.info(f"Notificador '{name}' registrado")
    
    def check_drawdown(self, drawdown: float) -> bool:
        """
        Comprobar si el drawdown excede el límite configurado.
        
        Args:
            drawdown: Porcentaje de drawdown actual
            
        Returns:
            True si se generó una alerta, False en caso contrario
            
        Raises:
            ValueError: Si el drawdown es negativo
        """
        if drawdown < 0:
            raise ValueError("Drawdown cannot be negative")
            
        if drawdown > self.drawdown_limit:
            message = f"Drawdown excedido: {drawdown}%"
            self._send_alert(message)
            return True
            
        return False
    
    def check_volatility(self, volatility: float) -> bool:
        """
        Comprobar si la volatilidad excede el límite configurado.
        
        Args:
            volatility: Porcentaje de volatilidad actual
            
        Returns:
            True si se generó una alerta, False en caso contrario
            
        Raises:
            ValueError: Si la volatilidad es negativa
        """
        if volatility < 0:
            raise ValueError("Volatility cannot be negative")
            
        if volatility > self.volatility_limit:
            message = f"Volatilidad excedida: {volatility}%"
            self._send_alert(message)
            return True
            
        return False
    
    def check_price_change(self, price_change: float) -> bool:
        """
        Comprobar si el cambio de precio excede el límite configurado.
        
        Args:
            price_change: Porcentaje de cambio de precio
            
        Returns:
            True si se generó una alerta, False en caso contrario
        """
        if abs(price_change) > self.price_change_limit:
            direction = "subida" if price_change > 0 else "bajada"
            message = f"Cambio brusco de precio: {abs(price_change)}% de {direction}"
            self._send_alert(message)
            return True
            
        return False
    
    def check_balance(self, balance: float) -> bool:
        """
        Comprobar si el balance está por debajo del mínimo configurado.
        
        Args:
            balance: Balance actual en USD
            
        Returns:
            True si se generó una alerta, False en caso contrario
        """
        if balance < self.min_balance:
            message = f"Balance bajo: ${balance} (mínimo: ${self.min_balance})"
            self._send_alert(message)
            return True
            
        return False
    
    def _send_alert(self, message: str, priority: str = "normal") -> None:
        """
        Enviar una alerta a través de todos los notificadores registrados.
        
        Args:
            message: Mensaje de la alerta
            priority: Prioridad de la alerta (normal, alta)
        """
        self._logger.warning(f"Alerta: {message} (prioridad: {priority})")
        
        for name, notifier in self._notifiers.items():
            try:
                if name == "email":
                    notifier.send_notification(
                        to="admin@genesis-trading.com",
                        subject=f"Genesis Alert ({priority})",
                        body=message
                    )
                elif name == "sms":
                    notifier.send_notification(
                        to="+1234567890",
                        message=message
                    )
                else:
                    # Interfaz genérica para otros notificadores
                    notifier.send_notification(message=message, priority=priority)
                    
            except Exception as e:
                self._logger.error(f"Error al enviar alerta a través de {name}: {e}")