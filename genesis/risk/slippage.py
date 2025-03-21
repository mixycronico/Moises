import logging
from typing import Dict, Any, Tuple

class SlippageController:
    """Maneja la validación del slippage en cada orden."""

    def __init__(self, base_max_slippage: float, liquidity_factor: float):
        self.base_max_slippage = base_max_slippage
        self.liquidity_factor = liquidity_factor
        self.logger = logging.getLogger("SlippageController")

    def validate_slippage(self, expected_price: float, execution_price: float, liquidity: float) -> bool:
        """Valida el slippage dinámicamente según liquidez."""
        dynamic_max_slippage = self.base_max_slippage * (1 + (self.liquidity_factor / liquidity))
        slippage = abs(execution_price - expected_price) / expected_price
        if slippage > dynamic_max_slippage:
            self.logger.warning(f"Slippage excesivo: {slippage:.6f} (límite: {dynamic_max_slippage:.6f})")
            return False
        return True