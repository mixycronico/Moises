"""
Performance tracking and analysis module.

This module provides components for tracking and analyzing the performance
of trading strategies and the overall system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class PerformanceTracker(Component):
    """
    Performance tracker for trading strategies.
    
    This component tracks and analyzes the performance of trading strategies
    and the overall system, calculating metrics like returns, drawdowns,
    Sharpe ratio, etc.
    """
    
    def __init__(self, name: str = "performance_tracker"):
        """
        Initialize the performance tracker.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # State tracking
        self.trades: List[Dict[str, Any]] = []
        self.equity_history: Dict[datetime, float] = {}
        self.starting_equity = 0.0
        self.current_equity = 0.0
    
    async def start(self) -> None:
        """Start the performance tracker."""
        await super().start()
        self.logger.info("Performance tracker started")
    
    async def stop(self) -> None:
        """Stop the performance tracker."""
        await super().stop()
        self.logger.info("Performance tracker stopped")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Handle events from the event bus.
        
        Args:
            event_type: Event type
            data: Event data
            source: Source component
        """
        if event_type == "account.balance":
            # Update equity tracking
            self.current_equity = data.get("total_value", self.current_equity)
            
            if not self.equity_history:
                # First entry
                self.starting_equity = self.current_equity
            
            # Record current equity value
            self.equity_history[datetime.now()] = self.current_equity
            
            # Calculate and emit performance metrics
            metrics = self.calculate_performance_metrics()
            await self.emit_event("performance.metrics", metrics)
        
        elif event_type == "trade.closed":
            # Record completed trade
            trade = data.copy()
            trade["timestamp"] = datetime.now().isoformat()
            self.trades.append(trade)
            
            # Calculate and emit trade metrics
            trade_metrics = self.calculate_trade_metrics(trade)
            await self.emit_event("performance.trade", trade_metrics)
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics for the overall system.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "current_equity": self.current_equity,
            "starting_equity": self.starting_equity
        }
        
        # Check if we have enough equity history
        if len(self.equity_history) > 1:
            equity_series = pd.Series(
                list(self.equity_history.values()),
                index=list(self.equity_history.keys())
            )
            
            # Total return
            metrics["total_return"] = (self.current_equity / self.starting_equity) - 1
            
            # Calculate daily returns
            daily_returns = equity_series.resample('D').last().pct_change().dropna()
            
            if len(daily_returns) > 0:
                # Annualized return
                metrics["annualized_return"] = ((1 + metrics["total_return"]) ** (365 / len(daily_returns))) - 1
                
                # Volatility
                metrics["volatility"] = daily_returns.std() * np.sqrt(365)
                
                # Sharpe Ratio
                risk_free_rate = 0.02  # Assume 2% annual risk-free rate
                daily_rf = (1 + risk_free_rate) ** (1/365) - 1
                metrics["sharpe_ratio"] = (daily_returns.mean() - daily_rf) / daily_returns.std() * np.sqrt(365)
                
                # Sortino Ratio
                downside_returns = daily_returns[daily_returns < 0]
                if len(downside_returns) > 0:
                    downside_deviation = downside_returns.std() * np.sqrt(365)
                    metrics["sortino_ratio"] = (metrics["annualized_return"] - risk_free_rate) / downside_deviation
                
                # Maximum Drawdown
                cumulative_returns = (1 + daily_returns).cumprod()
                max_drawdown = 0
                peak = cumulative_returns.iloc[0]
                
                for value in cumulative_returns:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                metrics["max_drawdown"] = max_drawdown
        
        # Trade metrics
        if self.trades:
            win_count = sum(1 for trade in self.trades if trade.get("profit", 0) > 0)
            loss_count = sum(1 for trade in self.trades if trade.get("profit", 0) <= 0)
            
            metrics["total_trades"] = len(self.trades)
            metrics["win_count"] = win_count
            metrics["loss_count"] = loss_count
            metrics["win_rate"] = win_count / len(self.trades) if len(self.trades) > 0 else 0
            
            profits = [trade.get("profit", 0) for trade in self.trades if trade.get("profit", 0) > 0]
            losses = [trade.get("profit", 0) for trade in self.trades if trade.get("profit", 0) <= 0]
            
            metrics["avg_profit"] = sum(profits) / len(profits) if profits else 0
            metrics["avg_loss"] = sum(losses) / len(losses) if losses else 0
            metrics["profit_factor"] = abs(sum(profits) / sum(losses)) if sum(losses) != 0 else float('inf')
        
        return metrics
    
    def calculate_trade_metrics(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics for a single trade.
        
        Args:
            trade: Trade data
            
        Returns:
            Dictionary of trade metrics
        """
        metrics = {
            "trade_id": trade.get("trade_id"),
            "symbol": trade.get("symbol"),
            "strategy": trade.get("strategy"),
            "side": trade.get("side"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "profit": trade.get("profit", 0),
            "profit_pct": trade.get("profit_pct", 0),
            "duration": trade.get("duration", 0)
        }
        
        # Calculate trade-specific metrics
        if trade.get("side") == "long":
            metrics["r_multiple"] = (trade.get("profit", 0) / trade.get("position_size", 1)) / \
                                   trade.get("stop_loss_percent", 0.01) if trade.get("stop_loss_percent", 0) > 0 else 0
        else:
            metrics["r_multiple"] = (trade.get("profit", 0) / trade.get("position_size", 1)) / \
                                   trade.get("stop_loss_percent", 0.01) if trade.get("stop_loss_percent", 0) > 0 else 0
        
        # Get overall trade history metrics
        if len(self.trades) > 1:
            # Calculate win streak
            win_streak = 0
            for t in reversed(self.trades):
                if t.get("profit", 0) > 0:
                    win_streak += 1
                else:
                    break
            
            metrics["win_streak"] = win_streak
            
            # Calculate losing streak
            lose_streak = 0
            for t in reversed(self.trades):
                if t.get("profit", 0) <= 0:
                    lose_streak += 1
                else:
                    break
            
            metrics["lose_streak"] = lose_streak
        
        return metrics
