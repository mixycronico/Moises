import logging
from typing import Dict, Any

class LiquidityChecker:
    """Evalúa la liquidez considerando la profundidad del order book."""

    def __init__(self, min_liquidity: float, depth_levels: int):
        self.min_liquidity = min_liquidity
        self.depth_levels = depth_levels
        self.logger = logging.getLogger("LiquidityChecker")

    def check_liquidity(self, order_book: Dict[str, Any]) -> bool:
        """Verifica la liquidez en múltiples niveles del order book."""
        bids = order_book.get("bids", [])[:self.depth_levels]
        asks = order_book.get("asks", [])[:self.depth_levels]
        bid_volume = sum([bid[1] for bid in bids])  # [precio, volumen]
        ask_volume = sum([ask[1] for ask in asks])
        liquidity = min(bid_volume, ask_volume)
        if liquidity >= self.min_liquidity:
            return True
        self.logger.warning(f"Liquidez insuficiente: {liquidity} < {self.min_liquidity}")
        return False