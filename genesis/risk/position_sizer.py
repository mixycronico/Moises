import logging
from typing import Dict, Any, Tuple, Optional

class PositionSizer:
    """Calculador de tamaño de posición para operaciones de trading."""

    def __init__(self, default_risk_percentage: float = 1.0):
        """
        Inicializa el calculador de tamaño de posición.
        
        Args:
            default_risk_percentage: Porcentaje de riesgo por defecto (1% = 1.0)
        """
        self._risk_percentage = default_risk_percentage
        self._account_balance = 0.0
        self.logger = logging.getLogger("PositionSizer")

    def set_risk_percentage(self, percentage: float) -> None:
        """
        Establece el porcentaje de riesgo por operación.
        
        Args:
            percentage: Porcentaje de riesgo (1% = 1.0)
        """
        if percentage <= 0 or percentage > 100:
            self.logger.warning(f"Porcentaje de riesgo inválido: {percentage}")
            return
        self._risk_percentage = percentage
        self.logger.info(f"Porcentaje de riesgo establecido a {percentage}%")

    def set_account_balance(self, balance: float) -> None:
        """
        Establece el balance de la cuenta.
        
        Args:
            balance: Balance de la cuenta en la moneda base
        """
        if balance <= 0:
            self.logger.warning(f"Balance inválido: {balance}")
            return
        self._account_balance = balance

    def calculate_position_size(self, entry_price: float, symbol: str, 
                              stop_loss_percentage: Optional[float] = None,
                              stop_loss_price: Optional[float] = None) -> float:
        """
        Calcula el tamaño de posición en valor de la moneda base.
        
        Args:
            entry_price: Precio de entrada
            symbol: Símbolo de trading
            stop_loss_percentage: Porcentaje de stop loss desde el precio de entrada
            stop_loss_price: Precio exacto de stop loss
            
        Returns:
            Tamaño de posición en valor de la moneda base
        """
        if self._account_balance <= 0:
            self.logger.error("Balance de cuenta no establecido")
            return 0
            
        # Cantidad de capital a arriesgar
        risk_amount = self._account_balance * (self._risk_percentage / 100)
        
        # Calcular la distancia al stop loss
        if stop_loss_price:
            # Si tenemos un precio de stop loss exacto
            risk_per_unit = abs(entry_price - stop_loss_price)
        elif stop_loss_percentage:
            # Si tenemos un porcentaje de stop loss
            risk_per_unit = entry_price * (stop_loss_percentage / 100)
        else:
            self.logger.error("Debe proporcionar stop_loss_percentage o stop_loss_price")
            return 0
            
        # Evitar división por cero
        if risk_per_unit <= 0:
            self.logger.error(f"Riesgo por unidad inválido: {risk_per_unit}")
            return 0
            
        # Calcular el tamaño de la posición en valor
        position_size = risk_amount / (risk_per_unit / entry_price)
        
        self.logger.info(f"Tamaño de posición calculado: {position_size} para {symbol}")
        return position_size
        
    def calculate_units(self, position_size: float, price: float) -> float:
        """
        Calcula el número de unidades a comprar/vender.
        
        Args:
            position_size: Tamaño de posición en valor
            price: Precio actual
            
        Returns:
            Número de unidades
        """
        if price <= 0:
            self.logger.error(f"Precio inválido: {price}")
            return 0
            
        units = position_size / price
        self.logger.info(f"Unidades calculadas: {units}")
        return units