"""
Position sizing module.

This module calculates appropriate position sizes based on risk parameters,
account balance, and market conditions.
"""

from typing import Dict, Any, Optional
from genesis.config.settings import settings


class PositionSizer:
    """
    Position sizer to calculate appropriate trade sizes.
    
    Determines position sizes based on account balance, risk tolerance,
    and signal strength.
    """
    
    def __init__(self):
        """Initialize the position sizer."""
        self.min_trade_size = 10.0  # Minimum trade size in quote currency
        self.default_risk = settings.get('risk.max_risk_per_trade', 0.02)  # 2%
    
    def calculate(
        self, 
        portfolio_value: float,
        risk_percent: Optional[float] = None,
        signal_strength: float = 1.0,
        max_size: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            portfolio_value: Total portfolio value in quote currency
            risk_percent: Percentage of portfolio to risk (0.01 = 1%)
            signal_strength: Strength of the trading signal (0.0 to 1.0)
            max_size: Maximum position size in quote currency
            
        Returns:
            Position size in quote currency
        """
        if risk_percent is None:
            risk_percent = self.default_risk
        
        # Base position size as a percentage of portfolio
        position_size = portfolio_value * risk_percent * signal_strength
        
        # Apply maximum size limit if specified
        if max_size is not None and position_size > max_size:
            position_size = max_size
        
        # Apply minimum size limit
        if position_size < self.min_trade_size:
            position_size = self.min_trade_size
        
        # Ensure it doesn't exceed portfolio value
        if position_size > portfolio_value:
            position_size = portfolio_value
        
        return position_size
    
    def calculate_units(
        self,
        position_size: float,
        current_price: float,
        min_quantity: Optional[float] = None
    ) -> float:
        """
        Calculate quantity units based on position size and price.
        
        Args:
            position_size: Position size in quote currency
            current_price: Current asset price
            min_quantity: Minimum quantity allowed by the exchange
            
        Returns:
            Quantity in base currency units
        """
        quantity = position_size / current_price
        
        # Apply minimum quantity if provided
        if min_quantity is not None and quantity < min_quantity:
            quantity = min_quantity
        
        return quantity
