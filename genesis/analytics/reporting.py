"""
Reporting module for generating trading reports.

This module provides functionality for generating periodic reports
and summaries of trading activity and performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import os
from datetime import datetime, timedelta
import json
import logging
import asyncio

from genesis.core.base import Component
from genesis.utils.logger import setup_logging
from genesis.analytics.visualization import Visualizer


class ReportGenerator(Component):
    """
    Report generator for creating trading reports.
    
    This component generates periodic reports and summaries of
    trading activity and performance.
    """
    
    def __init__(
        self, 
        name: str = "report_generator",
        report_dir: str = "reports"
    ):
        """
        Initialize the report generator.
        
        Args:
            name: Component name
            report_dir: Directory for storing reports
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Configuration
        self.report_dir = report_dir
        
        # Create report directory if it doesn't exist
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Visualizer for charts
        self.visualizer = Visualizer()
        
        # State tracking
        self.trades_data: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.equity_history: Dict[str, float] = {}
        self.last_daily_report = None
        self.last_weekly_report = None
        self.last_monthly_report = None
    
    async def start(self) -> None:
        """Start the report generator."""
        await super().start()
        
        # Start visualizer
        await self.visualizer.start()
        
        # Start scheduler for periodic reports
        asyncio.create_task(self._schedule_reports())
        
        self.logger.info("Report generator started")
    
    async def stop(self) -> None:
        """Stop the report generator."""
        await self.visualizer.stop()
        await super().stop()
        self.logger.info("Report generator stopped")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Handle events from the event bus.
        
        Args:
            event_type: Event type
            data: Event data
            source: Source component
        """
        if event_type == "trade.closed":
            # Save trade data
            trade = data.copy()
            trade["timestamp"] = datetime.now().isoformat()
            self.trades_data.append(trade)
        
        elif event_type == "performance.metrics":
            # Save performance metrics
            self.performance_metrics = data.copy()
            
            # Update equity history
            if "current_equity" in data:
                self.equity_history[datetime.now().isoformat()] = data["current_equity"]
    
    async def _schedule_reports(self) -> None:
        """Background task to schedule periodic reports."""
        while self.running:
            try:
                now = datetime.now()
                
                # Daily report (end of day)
                if now.hour == 23 and now.minute >= 55 and (
                    self.last_daily_report is None or 
                    (now - self.last_daily_report).days >= 1
                ):
                    await self.generate_daily_report()
                    self.last_daily_report = now
                
                # Weekly report (end of week)
                if now.weekday() == 6 and now.hour >= 23 and (
                    self.last_weekly_report is None or 
                    (now - self.last_weekly_report).days >= 7
                ):
                    await self.generate_weekly_report()
                    self.last_weekly_report = now
                
                # Monthly report (end of month)
                is_last_day = (now + timedelta(days=1)).day == 1
                if is_last_day and now.hour >= 23 and (
                    self.last_monthly_report is None or 
                    (now - self.last_monthly_report).days >= 28
                ):
                    await self.generate_monthly_report()
                    self.last_monthly_report = now
                
                # Check every minute
                await asyncio.sleep(60)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in report scheduler: {e}")
                await asyncio.sleep(300)  # Sleep longer on error
    
    async def generate_daily_report(self) -> Optional[str]:
        """
        Generate a daily trading report.
        
        Returns:
            Path to the generated report file
        """
        try:
            now = datetime.now()
            report_date = now.strftime("%Y-%m-%d")
            report_name = f"daily_report_{report_date}.json"
            report_path = os.path.join(self.report_dir, report_name)
            
            # Filter trades for today
            today_start = datetime.combine(now.date(), datetime.min.time())
            today_trades = [
                trade for trade in self.trades_data 
                if datetime.fromisoformat(trade["timestamp"]) >= today_start
            ]
            
            # Create report data
            report_data = {
                "report_type": "daily",
                "date": report_date,
                "timestamp": now.isoformat(),
                "performance_metrics": self.performance_metrics,
                "trades": today_trades,
                "trade_count": len(today_trades),
                "summary": self._generate_trade_summary(today_trades),
            }
            
            # Write report to file
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"Daily report generated: {report_path}")
            
            # Emit report event
            await self.emit_event("report.generated", {
                "report_type": "daily",
                "report_path": report_path,
                "report_date": report_date
            })
            
            return report_path
        
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
            return None
    
    async def generate_weekly_report(self) -> Optional[str]:
        """
        Generate a weekly trading report.
        
        Returns:
            Path to the generated report file
        """
        try:
            now = datetime.now()
            week_start = (now - timedelta(days=now.weekday() + 1)).strftime("%Y-%m-%d")
            week_end = now.strftime("%Y-%m-%d")
            report_name = f"weekly_report_{week_start}_to_{week_end}.json"
            report_path = os.path.join(self.report_dir, report_name)
            
            # Filter trades for the week
            week_start_date = now - timedelta(days=now.weekday() + 1)
            week_trades = [
                trade for trade in self.trades_data 
                if datetime.fromisoformat(trade["timestamp"]) >= week_start_date
            ]
            
            # Create report data
            report_data = {
                "report_type": "weekly",
                "week_start": week_start,
                "week_end": week_end,
                "timestamp": now.isoformat(),
                "performance_metrics": self.performance_metrics,
                "trades": week_trades,
                "trade_count": len(week_trades),
                "summary": self._generate_trade_summary(week_trades),
            }
            
            # Write report to file
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"Weekly report generated: {report_path}")
            
            # Emit report event
            await self.emit_event("report.generated", {
                "report_type": "weekly",
                "report_path": report_path,
                "week_start": week_start,
                "week_end": week_end
            })
            
            return report_path
        
        except Exception as e:
            self.logger.error(f"Error generating weekly report: {e}")
            return None
    
    async def generate_monthly_report(self) -> Optional[str]:
        """
        Generate a monthly trading report.
        
        Returns:
            Path to the generated report file
        """
        try:
            now = datetime.now()
            month_start = now.replace(day=1).strftime("%Y-%m-%d")
            month_end = now.strftime("%Y-%m-%d")
            report_name = f"monthly_report_{now.strftime('%Y-%m')}.json"
            report_path = os.path.join(self.report_dir, report_name)
            
            # Filter trades for the month
            month_start_date = now.replace(day=1)
            month_trades = [
                trade for trade in self.trades_data 
                if datetime.fromisoformat(trade["timestamp"]) >= month_start_date
            ]
            
            # Generate monthly equity chart if we have equity history
            equity_chart = None
            if self.equity_history:
                equity_df = pd.DataFrame(
                    [(datetime.fromisoformat(ts), value) for ts, value in self.equity_history.items()],
                    columns=["timestamp", "equity"]
                )
                equity_df = equity_df[equity_df["timestamp"] >= month_start_date]
                
                if not equity_df.empty:
                    equity_df.set_index("timestamp", inplace=True)
                    self.performance_metrics["equity_curve"] = equity_df
                    chart_data = await self.visualizer.generate_equity_chart(self.performance_metrics)
                    if chart_data:
                        equity_chart = chart_data
            
            # Create report data
            report_data = {
                "report_type": "monthly",
                "month": now.strftime("%Y-%m"),
                "month_start": month_start,
                "month_end": month_end,
                "timestamp": now.isoformat(),
                "performance_metrics": self.performance_metrics,
                "trades": month_trades,
                "trade_count": len(month_trades),
                "summary": self._generate_trade_summary(month_trades),
                "equity_chart": equity_chart
            }
            
            # Write report to file
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"Monthly report generated: {report_path}")
            
            # Emit report event
            await self.emit_event("report.generated", {
                "report_type": "monthly",
                "report_path": report_path,
                "month": now.strftime("%Y-%m")
            })
            
            return report_path
        
        except Exception as e:
            self.logger.error(f"Error generating monthly report: {e}")
            return None
    
    def _generate_trade_summary(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of trades.
        
        Args:
            trades: List of trades
            
        Returns:
            Summary statistics
        """
        if not trades:
            return {
                "trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate": 0,
                "total_profit": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "best_trade": None,
                "worst_trade": None
            }
        
        # Calculate basic statistics
        win_trades = [trade for trade in trades if trade.get("profit", 0) > 0]
        loss_trades = [trade for trade in trades if trade.get("profit", 0) <= 0]
        
        win_count = len(win_trades)
        loss_count = len(loss_trades)
        win_rate = win_count / len(trades) if trades else 0
        
        total_profit = sum(trade.get("profit", 0) for trade in trades)
        avg_profit = sum(trade.get("profit", 0) for trade in win_trades) / win_count if win_count else 0
        avg_loss = sum(trade.get("profit", 0) for trade in loss_trades) / loss_count if loss_count else 0
        
        profit_factor = abs(avg_profit * win_count) / abs(avg_loss * loss_count) if avg_loss * loss_count else float('inf')
        
        # Find best and worst trades
        best_trade = max(trades, key=lambda x: x.get("profit", 0)) if trades else None
        worst_trade = min(trades, key=lambda x: x.get("profit", 0)) if trades else None
        
        # Group by symbol
        trades_by_symbol = {}
        for trade in trades:
            symbol = trade.get("symbol", "unknown")
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        symbol_performance = {}
        for symbol, symbol_trades in trades_by_symbol.items():
            symbol_win_trades = [trade for trade in symbol_trades if trade.get("profit", 0) > 0]
            symbol_win_rate = len(symbol_win_trades) / len(symbol_trades) if symbol_trades else 0
            symbol_profit = sum(trade.get("profit", 0) for trade in symbol_trades)
            
            symbol_performance[symbol] = {
                "trade_count": len(symbol_trades),
                "win_rate": symbol_win_rate,
                "total_profit": symbol_profit
            }
        
        return {
            "trade_count": len(trades),
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "best_trade": {
                "symbol": best_trade.get("symbol"),
                "profit": best_trade.get("profit"),
                "timestamp": best_trade.get("timestamp")
            } if best_trade else None,
            "worst_trade": {
                "symbol": worst_trade.get("symbol"),
                "profit": worst_trade.get("profit"),
                "timestamp": worst_trade.get("timestamp")
            } if worst_trade else None,
            "symbol_performance": symbol_performance
        }

