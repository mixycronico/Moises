"""
Calculador de stop-loss para gestión de riesgos.

Este módulo proporciona cálculos para determinar niveles adecuados de stop-loss
basados en la volatilidad del mercado, características de la posición, y parámetros de riesgo.
"""

import logging
from typing import Dict, Any, Optional, Union

class StopLossCalculator:
    """
    Calculador de stop-loss para gestión de riesgos.
    
    Calcula niveles adecuados de stop-loss basados en volatilidad del mercado,
    posición, y parámetros de riesgo.
    """
    
    def __init__(self):
        """Inicializar el calculador de stop-loss."""
        self._logger = logging.getLogger(__name__)
        self._default_multiplier = 2.0  # Multiplicador por defecto para ATR
        self._trailing_percentage = 1.0  # 1% de trailing stop por defecto
    
    def set_default_multiplier(self, multiplier: float) -> None:
        """
        Establecer el multiplicador por defecto para el cálculo basado en ATR.
        
        Args:
            multiplier: Multiplicador de ATR
        """
        if multiplier <= 0:
            raise ValueError("Multiplier must be positive")
            
        self._default_multiplier = multiplier
        self._logger.info(f"Default ATR multiplier set to {multiplier}")
    
    def set_trailing_percentage(self, percentage: float) -> None:
        """
        Establecer el porcentaje para trailing stop.
        
        Args:
            percentage: Porcentaje para trailing stop
        """
        if percentage <= 0 or percentage > 100:
            raise ValueError("Trailing percentage must be between 0 and 100")
            
        self._trailing_percentage = percentage
        self._logger.info(f"Trailing stop percentage set to {percentage}%")
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        atr: Optional[float] = None, 
        side: str = "buy",
        multiplier: Optional[float] = None
    ) -> float:
        """
        Calcular el precio de stop-loss basado en ATR o un porcentaje fijo.
        
        Args:
            entry_price: Precio de entrada
            atr: Average True Range (opcional)
            side: Lado de la operación ('buy' o 'sell')
            multiplier: Multiplicador de ATR (opcional, usa el por defecto si no se proporciona)
            
        Returns:
            Precio de stop-loss
        """
        is_long = side.lower() == "buy"
        
        if multiplier is None:
            multiplier = self._default_multiplier
            
        if atr is not None:
            # Calcular stop-loss basado en ATR
            stop_distance = atr * multiplier
        else:
            # Usar un porcentaje fijo del precio
            stop_distance = entry_price * 0.05  # 5% por defecto
            
        # Calcular precio de stop-loss según el lado de la operación
        if is_long:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
            
        return stop_loss
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        side: str = "buy",
        percentage: Optional[float] = None
    ) -> float:
        """
        Calcular el precio de trailing stop.
        
        Args:
            entry_price: Precio de entrada
            current_price: Precio actual
            side: Lado de la operación ('buy' o 'sell')
            percentage: Porcentaje para trailing stop (opcional, usa el por defecto si no se proporciona)
            
        Returns:
            Precio de trailing stop
        """
        is_long = side.lower() == "buy"
        
        if percentage is None:
            percentage = self._trailing_percentage
            
        # Convertir porcentaje a decimal
        trail_factor = percentage / 100
        
        # Calcular precio de trailing stop según el lado de la operación
        if is_long:
            trailing_stop = current_price * (1 - trail_factor)
        else:
            trailing_stop = current_price * (1 + trail_factor)
            
        return trailing_stop
    
    async def calculate(
        self,
        symbol: str,
        signal_type: str,
        position_size: float,
        price: Optional[float] = None,
        atr_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calcular nivel de stop-loss para un trade.
        
        Args:
            symbol: Símbolo del par de trading
            signal_type: Tipo de señal ('buy' o 'sell')
            position_size: Tamaño de posición en moneda cotizada
            price: Precio actual (opcional)
            atr_value: Valor de Average True Range (opcional)
            
        Returns:
            Detalles de stop-loss incluyendo precio y porcentaje
        """
        is_long = signal_type.lower() == 'buy'
        
        if price is None:
            return {
                "type": "fixed",
                "percentage": 0.05,  # 5% por defecto
                "price": None,
                "is_long": is_long
            }
        
        # Calcular stop-loss
        stop_loss = self.calculate_stop_loss(
            entry_price=price,
            atr=atr_value,
            side=signal_type
        )
        
        # Calcular el porcentaje
        stop_percentage = abs(stop_loss - price) / price
        
        return {
            "type": "fixed",
            "percentage": stop_percentage,
            "price": stop_loss,
            "is_long": is_long
        }
