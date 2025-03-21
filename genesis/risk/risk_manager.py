import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import asyncio
import time

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class RiskManager:
    """Manejo de riesgo avanzado para operaciones de trading."""

    def __init__(self, config: Dict[str, Any]):
        self.max_risk_per_trade = config.get("max_risk_per_trade", 0.02)
        self.stop_loss_atr_multiplier = config.get("stop_loss_atr_multiplier", 2.0)
        self.trailing_stop_atr_multiplier = config.get("trailing_stop_atr_multiplier", 1.5)
        self.max_slippage = config.get("max_slippage", 0.001)
        self.logger = logging.getLogger("RiskManager")

    def calculate_position_size(self, capital: float, entry_price: float, atr: float) -> float:
        """Calcula el tamaño de la posición basado en el capital y el ATR."""
        risk_amount = capital * self.max_risk_per_trade
        stop_loss_distance = atr * self.stop_loss_atr_multiplier
        position_size = risk_amount / (stop_loss_distance * entry_price)
        self.logger.info(f"Posición calculada: {position_size} (capital: {capital}, ATR: {atr})")
        return round(position_size, 6)

    def apply_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """Calcula el stop-loss dinámico basado en ATR."""
        stop_loss_distance = atr * self.stop_loss_atr_multiplier
        stop_loss = entry_price - stop_loss_distance if side == "buy" else entry_price + stop_loss_distance
        return round(stop_loss, 6)

    def apply_trailing_stop(self, current_price: float, highest_price: float, side: str, atr: float) -> float:
        """Calcula el trailing stop adaptativo basado en ATR."""
        trailing_stop_distance = atr * self.trailing_stop_atr_multiplier
        if side == "buy":
            trailing_stop = max(highest_price - trailing_stop_distance, self.apply_stop_loss(current_price, side, atr))
        else:
            trailing_stop = min(highest_price + trailing_stop_distance, self.apply_stop_loss(current_price, side, atr))
        return round(trailing_stop, 6)