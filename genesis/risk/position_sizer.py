"""
Calculador de tamaño de posición.

Este módulo proporciona cálculos para determinar el tamaño óptimo de una posición
basado en el balance de la cuenta, la tolerancia al riesgo, y las características
del mercado.
"""

import logging
from typing import Dict, Any, Optional

class PositionSizer:
    """
    Calcula el tamaño óptimo de posición basado en parámetros de riesgo.
    
    Esta clase implementa varios métodos para calcular el tamaño de posición,
    incluyendo un enfoque basado en el riesgo porcentual por operación.
    """
    
    def __init__(self):
        """Inicializar el calculador de tamaño de posición."""
        self._logger = logging.getLogger(__name__)
        self._risk_percentage = 1.0  # 1% de riesgo por defecto
        self._account_balance = 0.0
        self._max_position_size = 0.0  # Límite máximo para el tamaño de posición
    
    def set_risk_percentage(self, percentage: float) -> None:
        """
        Establecer el porcentaje de riesgo por operación.
        
        Args:
            percentage: Porcentaje de capital a arriesgar por operación (0-100)
            
        Raises:
            ValueError: Si el porcentaje está fuera del rango válido
        """
        if percentage < 0 or percentage > 100:
            raise ValueError("Risk percentage must be between 0 and 100")
            
        self._risk_percentage = percentage
        self._logger.info(f"Risk percentage set to {percentage}%")
    
    def set_account_balance(self, balance: float) -> None:
        """
        Establecer el balance de la cuenta.
        
        Args:
            balance: Balance actual de la cuenta en USD
            
        Raises:
            ValueError: Si el balance es negativo
        """
        if balance < 0:
            raise ValueError("Account balance cannot be negative")
            
        self._account_balance = balance
        self._logger.info(f"Account balance set to ${balance}")
    
    def set_max_position_size(self, max_size: float) -> None:
        """
        Establecer el tamaño máximo de posición.
        
        Args:
            max_size: Tamaño máximo de posición en USD
            
        Raises:
            ValueError: Si el tamaño máximo es negativo
        """
        if max_size < 0:
            raise ValueError("Max position size cannot be negative")
            
        self._max_position_size = max_size
        self._logger.info(f"Max position size set to ${max_size}")
    
    def calculate_position_size(
        self, 
        entry_price: float, 
        symbol: str, 
        stop_loss_percentage: Optional[float] = None
    ) -> float:
        """
        Calcular el tamaño óptimo de posición en USD.
        
        Args:
            entry_price: Precio de entrada
            symbol: Símbolo de trading
            stop_loss_percentage: Distancia al stop-loss en porcentaje
            
        Returns:
            Tamaño de la posición en USD
        """
        if stop_loss_percentage is None:
            # Si no se proporciona un porcentaje de stop-loss, usamos un valor por defecto
            stop_loss_percentage = 5  # 5% de distancia al stop-loss por defecto
        
        # Calcular el riesgo en USD
        risk_amount = (self._account_balance * self._risk_percentage) / 100
        
        # Calcular el tamaño de posición basado en el stop-loss
        position_size = risk_amount / (stop_loss_percentage / 100)
        
        # Limitar el tamaño de posición si es necesario
        if self._max_position_size > 0 and position_size > self._max_position_size:
            position_size = self._max_position_size
            self._logger.warning(f"Position size capped at ${position_size} due to max limit")
            
        self._logger.info(f"Calculated position size for {symbol}: ${position_size}")
        return position_size
    
    def calculate_units(self, position_size: float, price: float) -> float:
        """
        Convertir el tamaño de posición de USD a unidades del activo.
        
        Args:
            position_size: Tamaño de posición en USD
            price: Precio actual del activo
            
        Returns:
            Cantidad de unidades del activo
        """
        if price <= 0:
            raise ValueError("Price must be positive")
            
        units = position_size / price
        return units