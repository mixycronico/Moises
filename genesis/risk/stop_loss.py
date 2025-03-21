import logging
from typing import Dict, Any, Optional

class StopLossCalculator:
    """Calculador de stop-loss estático y dinámico."""

    def __init__(self, default_multiplier: float = 2.0):
        """
        Inicializa el calculador de stop-loss.
        
        Args:
            default_multiplier: Multiplicador de ATR por defecto
        """
        self._atr_multiplier = default_multiplier
        self._trailing_percentage = 1.0  # 1% por defecto
        self.logger = logging.getLogger("StopLossCalculator")

    def set_default_multiplier(self, multiplier: float) -> None:
        """
        Establece el multiplicador de ATR por defecto.
        
        Args:
            multiplier: Multiplicador de ATR
        """
        if multiplier <= 0:
            self.logger.warning(f"Multiplicador inválido: {multiplier}")
            return
        self._atr_multiplier = multiplier

    def set_trailing_percentage(self, percentage: float) -> None:
        """
        Establece el porcentaje para el trailing stop.
        
        Args:
            percentage: Porcentaje para trailing stop (1% = 1.0)
        """
        if percentage <= 0 or percentage > 100:
            self.logger.warning(f"Porcentaje de trailing stop inválido: {percentage}")
            return
        self._trailing_percentage = percentage

    def calculate_stop_loss(self, entry_price: float, atr: float, side: str) -> float:
        """
        Calcula el stop-loss basado en ATR.
        
        Args:
            entry_price: Precio de entrada
            atr: Average True Range
            side: Dirección de la operación ('buy' o 'sell')
            
        Returns:
            Precio de stop-loss
        """
        stop_distance = atr * self._atr_multiplier
        
        if side.lower() == "buy":
            # Para posiciones largas, el stop está por debajo
            stop_price = entry_price - stop_distance
        else:
            # Para posiciones cortas, el stop está por encima
            stop_price = entry_price + stop_distance
            
        self.logger.info(f"Stop-loss calculado: {stop_price} (entry: {entry_price}, ATR: {atr})")
        return stop_price
        
    def calculate_trailing_stop(self, entry_price: float, current_price: float, 
                               side: str, highest_price: Optional[float] = None) -> float:
        """
        Calcula el trailing stop basado en el precio actual.
        
        Args:
            entry_price: Precio de entrada
            current_price: Precio actual
            side: Dirección de la operación ('buy' o 'sell')
            highest_price: Precio más alto/bajo alcanzado (opcional)
            
        Returns:
            Precio de trailing stop
        """
        # Usar el precio actual como referencia si no se proporciona precio más alto/bajo
        reference_price = highest_price if highest_price is not None else current_price
        
        if side.lower() == "buy":
            # Para posiciones largas, trailing stop por debajo
            stop_price = reference_price * (1 - self._trailing_percentage / 100)
        else:
            # Para posiciones cortas, trailing stop por encima
            stop_price = reference_price * (1 + self._trailing_percentage / 100)
            
        self.logger.info(f"Trailing stop calculado: {stop_price} (ref: {reference_price})")
        return stop_price