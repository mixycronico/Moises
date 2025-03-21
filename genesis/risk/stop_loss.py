"""
Stop-loss calculator module.

This module handles the calculation of appropriate stop-loss levels
for risk management.
"""

from typing import Dict, Any, Optional
import numpy as np

from genesis.config.settings import settings


class StopLossCalculator:
    """
    Stop-loss calculator for risk management.
    
    Calculates appropriate stop-loss levels based on market volatility,
    position, and risk parameters.
    """
    
    def __init__(self):
        """Initialize the stop-loss calculator."""
        self.default_stop_loss_pct = settings.get('risk.stop_loss_pct', 0.05)  # 5%
    
    async def calculate(
        self,
        symbol: str,
        signal_type: str,
        position_size: float,
        price: Optional[float] = None,
        atr_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate stop-loss level for a trade.
        
        Args:
            symbol: Trading pair symbol
            signal_type: Signal type ('buy' or 'sell')
            position_size: Position size in quote currency
            price: Current price (optional)
            atr_value: Average True Range value (optional)
            
        Returns:
            Stop-loss details including price and percentage
        """
        # In a real implementation, we would fetch market data here
        # For now, use a simple percentage-based stop loss
        is_long = signal_type.lower() == 'buy'
        
        # Use ATR-based stop loss if provided, otherwise use default percentage
        if atr_value is not None and price is not None:
            # Typical ATR-based stop: 2x ATR from entry price
            multiplier = 2.0
            stop_pct = (atr_value * multiplier) / price
            stop_price = price * (1 - stop_pct) if is_long else price * (1 + stop_pct)
        else:
            stop_pct = self.default_stop_loss_pct
            stop_price = price * (1 - stop_pct) if is_long and price else None
        
        return {
            "type": "fixed",
            "percentage": stop_pct,
            "price": stop_price,
            "is_long": is_long
        }
    
    def calculate_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        is_long: bool,
        atr_value: Optional[float] = None,
        activation_pct: float = 0.01  # 1% profit to activate
    ) -> Dict[str, Any]:
        """
        Calculate a trailing stop-loss level.
        
        Args:
            current_price: Current market price
            entry_price: Entry price of the position
            is_long: Whether the position is long (True) or short (False)
            atr_value: Average True Range value (optional)
            activation_pct: Percentage profit before trailing stop activates
            
        Returns:
            Trailing stop details
        """
        # Calculate profit/loss percentage
        pnl_pct = (current_price / entry_price - 1) if is_long else (entry_price / current_price - 1)
        
        # Determine if trailing stop should be active
        is_active = pnl_pct >= activation_pct
        
        # Calculate trailing distance
        if atr_value is not None:
            # Use ATR-based trailing distance
            trail_distance = atr_value * 2
        else:
            # Use percentage-based trailing distance
            trail_distance = current_price * self.default_stop_loss_pct
        
        # Calculate stop price
        if is_long:
            stop_price = current_price - trail_distance if is_active else entry_price * (1 - self.default_stop_loss_pct)
        else:
            stop_price = current_price + trail_distance if is_active else entry_price * (1 + self.default_stop_loss_pct)
        
        return {
            "type": "trailing",
            "stop_price": stop_price,
            "trail_distance": trail_distance,
            "is_active": is_active,
            "activation_threshold": entry_price * (1 + activation_pct) if is_long else entry_price * (1 - activation_pct)
        }
