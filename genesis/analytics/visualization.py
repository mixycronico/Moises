"""
Visualization module for trading performance and analytics.

This module provides functions and classes for visualizing trading performance,
market data, and other analytics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from typing import Dict, List, Any, Optional, Tuple, Union
import io
import base64
from datetime import datetime, timedelta

from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class Visualizer(Component):
    """
    Visualization component for creating charts and graphs.
    
    This component generates visualizations for trading performance,
    market data, and other analytics.
    """
    
    def __init__(self, name: str = "visualizer"):
        """
        Initialize the visualizer.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Set default style
        plt.style.use('dark_background')
    
    async def start(self) -> None:
        """Start the visualizer."""
        await super().start()
        self.logger.info("Visualizer started")
    
    async def stop(self) -> None:
        """Stop the visualizer."""
        await super().stop()
        self.logger.info("Visualizer stopped")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Handle events from the event bus.
        
        Args:
            event_type: Event type
            data: Event data
            source: Source component
        """
        if event_type == "performance.metrics":
            # Generate performance charts
            equity_chart = await self.generate_equity_chart(data)
            if equity_chart:
                await self.emit_event("visualization.chart", {
                    "chart_type": "equity",
                    "chart_data": equity_chart,
                    "timestamp": datetime.now().isoformat()
                })
    
    async def generate_equity_chart(self, metrics: Dict[str, Any]) -> Optional[str]:
        """
        Generate an equity curve chart.
        
        Args:
            metrics: Performance metrics containing equity history
            
        Returns:
            Base64-encoded PNG image data
        """
        if "equity_curve" not in metrics:
            return None
        
        equity_data = metrics["equity_curve"]
        if not equity_data:
            return None
        
        try:
            # Convert to DataFrame if it's not already
            if not isinstance(equity_data, pd.DataFrame):
                df = pd.DataFrame(equity_data, columns=["timestamp", "equity"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            else:
                df = equity_data
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot equity curve
            ax.plot(df.index, df["equity"], color='#00b3b3', linewidth=2)
            
            # Add labels and title
            ax.set_title("Equity Curve", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Equity (USD)", fontsize=12)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Add grid
            ax.grid(alpha=0.3)
            
            # Add current equity and profit/loss annotation
            if len(df) > 0:
                latest_equity = df["equity"].iloc[-1]
                initial_equity = df["equity"].iloc[0]
                profit_loss = latest_equity - initial_equity
                pct_change = (profit_loss / initial_equity) * 100
                
                annotation_text = f"Current: ${latest_equity:.2f}\nP/L: ${profit_loss:.2f} ({pct_change:.2f}%)"
                ax.annotate(annotation_text, 
                           xy=(df.index[-1], latest_equity),
                           xytext=(0, 10),
                           textcoords="offset points",
                           fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.8))
            
            # Tight layout
            plt.tight_layout()
            
            # Convert to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
        
        except Exception as e:
            self.logger.error(f"Error generating equity chart: {e}")
            return None
    
    def generate_drawdown_chart(self, equity_series: pd.Series) -> Optional[str]:
        """
        Generate a drawdown chart.
        
        Args:
            equity_series: Series of equity values
            
        Returns:
            Base64-encoded PNG image data
        """
        try:
            # Calculate drawdown
            rolling_max = equity_series.cummax()
            drawdown = (equity_series / rolling_max - 1) * 100
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot drawdown
            ax.fill_between(drawdown.index, drawdown.values, 0, color='#ff3333', alpha=0.5)
            ax.plot(drawdown.index, drawdown.values, color='#ff3333', linewidth=1)
            
            # Add labels and title
            ax.set_title("Drawdown Chart", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Drawdown (%)", fontsize=12)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Add grid
            ax.grid(alpha=0.3)
            
            # Add max drawdown annotation
            if len(drawdown) > 0:
                max_dd = drawdown.min()
                max_dd_date = drawdown.idxmin()
                
                ax.annotate(f"Max DD: {max_dd:.2f}%", 
                           xy=(max_dd_date, max_dd),
                           xytext=(0, -20),
                           textcoords="offset points",
                           fontsize=10,
                           arrowprops=dict(arrowstyle="->", color="white"),
                           bbox=dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.8))
            
            # Tight layout
            plt.tight_layout()
            
            # Convert to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
        
        except Exception as e:
            self.logger.error(f"Error generating drawdown chart: {e}")
            return None
    
    def generate_trade_distribution_chart(self, trades: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate a trade distribution chart.
        
        Args:
            trades: List of trade data
            
        Returns:
            Base64-encoded PNG image data
        """
        if not trades:
            return None
        
        try:
            # Extract profit/loss percentages
            profit_pcts = [trade.get("profit_pct", 0) for trade in trades]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create histogram
            bins = np.linspace(min(profit_pcts), max(profit_pcts), 20)
            ax.hist(profit_pcts, bins=bins, alpha=0.7, color='#4d88ff')
            
            # Add labels and title
            ax.set_title("Trade Profit/Loss Distribution", fontsize=16)
            ax.set_xlabel("Profit/Loss (%)", fontsize=12)
            ax.set_ylabel("Number of Trades", fontsize=12)
            
            # Add grid
            ax.grid(alpha=0.3)
            
            # Add statistics
            win_count = sum(1 for pct in profit_pcts if pct > 0)
            loss_count = sum(1 for pct in profit_pcts if pct <= 0)
            win_rate = win_count / len(profit_pcts) if profit_pcts else 0
            avg_win = np.mean([pct for pct in profit_pcts if pct > 0]) if any(pct > 0 for pct in profit_pcts) else 0
            avg_loss = np.mean([pct for pct in profit_pcts if pct <= 0]) if any(pct <= 0 for pct in profit_pcts) else 0
            
            stats_text = (f"Win Rate: {win_rate:.2%}\n"
                         f"Avg Win: {avg_win:.2%}\n"
                         f"Avg Loss: {avg_loss:.2%}\n"
                         f"Profit Factor: {abs(avg_win * win_count) / abs(avg_loss * loss_count) if avg_loss * loss_count else float('inf'):.2f}")
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.8))
            
            # Tight layout
            plt.tight_layout()
            
            # Convert to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
        
        except Exception as e:
            self.logger.error(f"Error generating trade distribution chart: {e}")
            return None
    
    def generate_price_chart(
        self, 
        df: pd.DataFrame, 
        indicators: Optional[List[str]] = None,
        signals: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """
        Generate a price chart with indicators and signals.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicator columns to include
            signals: List of trade signals to display
            
        Returns:
            Base64-encoded PNG image data
        """
        try:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price data
            ax1.plot(df.index, df['close'], label='Price', color='#ffffff', linewidth=1)
            
            # Add moving averages if available
            if indicators:
                for indicator in indicators:
                    if indicator in df.columns:
                        if indicator.startswith('sma_'):
                            period = indicator.split('_')[1]
                            ax1.plot(df.index, df[indicator], label=f'SMA {period}', linewidth=1)
                        elif indicator.startswith('ema_'):
                            period = indicator.split('_')[1]
                            ax1.plot(df.index, df[indicator], label=f'EMA {period}', linewidth=1)
            
            # Add Bollinger Bands if available
            if 'bb_upper' in df.columns and 'bb_middle' in df.columns and 'bb_lower' in df.columns:
                ax1.plot(df.index, df['bb_upper'], 'b--', alpha=0.3, linewidth=1)
                ax1.plot(df.index, df['bb_middle'], 'b-', alpha=0.3, linewidth=1)
                ax1.plot(df.index, df['bb_lower'], 'b--', alpha=0.3, linewidth=1)
                ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1, color='blue')
            
            # Add signals if available
            if signals:
                buy_signals = [signal for signal in signals if signal.get('type') == 'buy']
                sell_signals = [signal for signal in signals if signal.get('type') == 'sell']
                
                for signal in buy_signals:
                    if 'timestamp' in signal and 'price' in signal:
                        timestamp = pd.to_datetime(signal['timestamp'])
                        if timestamp in df.index:
                            ax1.scatter(timestamp, signal['price'], color='green', marker='^', s=100)
                
                for signal in sell_signals:
                    if 'timestamp' in signal and 'price' in signal:
                        timestamp = pd.to_datetime(signal['timestamp'])
                        if timestamp in df.index:
                            ax1.scatter(timestamp, signal['price'], color='red', marker='v', s=100)
            
            # Set title and labels for price chart
            ax1.set_title(f"Price Chart", fontsize=16)
            ax1.set_ylabel("Price", fontsize=12)
            ax1.grid(alpha=0.3)
            ax1.legend(loc='upper left')
            
            # Plot volume on the second subplot
            ax2.bar(df.index, df['volume'], color='#5c5c8a', alpha=0.7)
            ax2.set_ylabel("Volume", fontsize=12)
            ax2.grid(alpha=0.3)
            
            # Format x-axis dates
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Tight layout
            plt.tight_layout()
            
            # Convert to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
        
        except Exception as e:
            self.logger.error(f"Error generating price chart: {e}")
            return None

