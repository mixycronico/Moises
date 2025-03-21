"""
Risk manager implementation.

The risk manager coordinates risk-related functions like position sizing,
stop-loss management, and exposure control.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from genesis.config.settings import settings
from genesis.core.base import Component
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator
from genesis.utils.logger import setup_logging


class RiskManager(Component):
    """
    Risk manager for controlling trading risk.
    
    The risk manager coordinates position sizing, stop-loss levels,
    and overall portfolio risk.
    """
    
    def __init__(self, name: str = "risk_manager"):
        """
        Initialize the risk manager.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Create sub-components
        self.position_sizer = PositionSizer()
        self.stop_loss_calculator = StopLossCalculator()
        
        # Risk config
        self.max_risk_per_trade = settings.get('risk.max_risk_per_trade', 0.02)  # 2%
        self.max_risk_total = settings.get('risk.max_risk_total', 0.2)  # 20%
        
        # State tracking
        self.portfolio_value = 0.0
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        self.total_risk = 0.0
    
    async def start(self) -> None:
        """Start the risk manager."""
        await super().start()
        self.logger.info("Risk manager started")
    
    async def stop(self) -> None:
        """Stop the risk manager."""
        await super().stop()
        self.logger.info("Risk manager stopped")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Handle events from the event bus.
        
        Args:
            event_type: Event type
            data: Event data
            source: Source component
        """
        if event_type == "strategy.signal":
            await self._handle_signal(data)
        
        elif event_type == "trade.opened":
            await self._handle_trade_opened(data)
        
        elif event_type == "trade.closed":
            await self._handle_trade_closed(data)
        
        elif event_type == "account.balance":
            self.portfolio_value = data.get("total_value", self.portfolio_value)
    
    async def _handle_signal(self, data: Dict[str, Any]) -> None:
        """
        Handle a strategy signal.
        
        This method processes a trading signal, applies risk management,
        and emits a trade decision event.
        
        Args:
            data: Signal data
        """
        symbol = data.get("symbol")
        strategy = data.get("strategy")
        signal = data.get("signal", {})
        
        if not symbol or not strategy or not signal:
            return
        
        signal_type = signal.get("type")
        
        if signal_type in ("buy", "sell"):
            # Check if we're already at max risk
            if self.total_risk >= self.max_risk_total:
                self.logger.warning(f"Maximum portfolio risk reached ({self.total_risk:.2%}), ignoring signal")
                return
            
            # Check if we already have an active trade for this symbol
            if symbol in self.active_trades:
                self.logger.info(f"Already have an active trade for {symbol}, ignoring new signal")
                return
            
            # Calculate position size
            position_size = self.position_sizer.calculate(
                self.portfolio_value,
                self.max_risk_per_trade,
                signal.get("strength", 1.0)
            )
            
            # Calculate stop loss
            stop_loss = await self.stop_loss_calculator.calculate(
                symbol, signal_type, position_size
            )
            
            # Calculate risk for this trade
            risk_amount = position_size * stop_loss["percentage"]
            risk_percent = risk_amount / self.portfolio_value
            
            # Check if this trade would exceed max risk per trade
            if risk_percent > self.max_risk_per_trade:
                self.logger.warning(
                    f"Trade risk ({risk_percent:.2%}) exceeds maximum per trade ({self.max_risk_per_trade:.2%})"
                )
                # Adjust position size to meet max risk
                position_size = (self.max_risk_per_trade * self.portfolio_value) / stop_loss["percentage"]
                risk_percent = self.max_risk_per_trade
            
            # Emit trade decision
            await self.emit_event("risk.trade_decision", {
                "symbol": symbol,
                "strategy": strategy,
                "signal_type": signal_type,
                "position_size": position_size,
                "stop_loss": stop_loss,
                "risk_percent": risk_percent,
                "approved": True,
                "reason": "Signal approved with risk management"
            })
        
        elif signal_type in ("exit", "close"):
            # Check if we have an active trade for this symbol
            if symbol in self.active_trades:
                await self.emit_event("risk.trade_decision", {
                    "symbol": symbol,
                    "strategy": strategy,
                    "signal_type": signal_type,
                    "trade_id": self.active_trades[symbol].get("trade_id"),
                    "approved": True,
                    "reason": "Exit signal approved"
                })
            else:
                self.logger.warning(f"Received exit signal for {symbol} but no active trade found")
    
    async def _handle_trade_opened(self, data: Dict[str, Any]) -> None:
        """
        Handle a trade opened event.
        
        Updates the risk tracking with the new trade.
        
        Args:
            data: Trade data
        """
        symbol = data.get("symbol")
        trade_id = data.get("trade_id")
        position_size = data.get("position_size", 0.0)
        stop_loss_pct = data.get("stop_loss_percent", 0.0)
        
        if not symbol or not trade_id:
            return
        
        # Calculate risk for this trade
        risk_amount = position_size * stop_loss_pct
        risk_percent = risk_amount / self.portfolio_value
        
        # Update active trades and total risk
        self.active_trades[symbol] = {
            "trade_id": trade_id,
            "position_size": position_size,
            "stop_loss_percent": stop_loss_pct,
            "risk_percent": risk_percent
        }
        
        self.total_risk += risk_percent
        
        self.logger.info(
            f"Added trade {trade_id} for {symbol} with risk {risk_percent:.2%}, total risk now {self.total_risk:.2%}"
        )
    
    async def _handle_trade_closed(self, data: Dict[str, Any]) -> None:
        """
        Handle a trade closed event.
        
        Updates the risk tracking when a trade is closed.
        
        Args:
            data: Trade data
        """
        symbol = data.get("symbol")
        trade_id = data.get("trade_id")
        
        if not symbol or not trade_id:
            return
        
        # Check if we have this trade
        if symbol in self.active_trades:
            risk_percent = self.active_trades[symbol].get("risk_percent", 0.0)
            self.total_risk -= risk_percent
            del self.active_trades[symbol]
            
            self.logger.info(
                f"Removed trade {trade_id} for {symbol}, released risk {risk_percent:.2%}, "
                f"total risk now {self.total_risk:.2%}"
            )
