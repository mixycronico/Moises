"""
Main entry point for the Genesis trading system.

Este módulo inicializa y arranca el sistema, configurando todos los componentes
y proporcionando el punto de entrada principal para la operación.
También expone la aplicación Flask para Gunicorn.
"""

import asyncio
import os
import signal
import argparse
import logging
from typing import List, Dict, Any, Optional

# Importar la aplicación Flask para Gunicorn
from app import app

from genesis.config.settings import settings
from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus
from genesis.exchanges.ccxt_wrapper import CCXTExchange
from genesis.data.market_data import MarketDataManager
from genesis.data.analyzer import MarketAnalyzer
from genesis.strategies.trend_following import MovingAverageCrossover, MACDStrategy
from genesis.strategies.mean_reversion import RSIStrategy, BollingerBandsStrategy
from genesis.strategies.sentiment_based import SentimentStrategy
from genesis.risk.manager import RiskManager
from genesis.analytics.performance import PerformanceTracker
from genesis.analytics.visualization import Visualizer
from genesis.analytics.reporting import ReportGenerator
from genesis.db.repository import Repository
from genesis.db.models import Base
from genesis.db.paper_trading_models import PaperTradingAccount
from genesis.trading.paper_trading import PaperTradingManager
from genesis.api.server import APIServer
from genesis.workers.scheduler import Scheduler, Task
from genesis.workers.processor import TaskProcessor
from genesis.utils.logger import setup_logging


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Genesis Trading System")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--exchange", 
        type=str, 
        default=settings.get("exchanges.default_exchange", "binance"),
        help="Exchange to connect to"
    )
    parser.add_argument(
        "--symbols", 
        type=str, 
        default=",".join(settings.get("trading.default_symbols", ["BTC/USDT"])),
        help="Comma-separated list of trading symbols"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest mode"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Run only the API server"
    )
    parser.add_argument(
        "--paper-trading",
        action="store_true",
        help="Run in paper trading mode"
    )
    return parser.parse_args()


async def setup_system(args) -> Engine:
    """
    Set up the trading system components.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Initialized system engine
    """
    logger = setup_logging("genesis")
    logger.info("Setting up Genesis trading system")
    
    # Create the main engine
    engine = Engine()
    
    # Initialize database and repository
    repo = Repository()
    await repo.create_tables(Base)
    
    # API server (always enabled)
    api_server = APIServer()
    engine.register_component(api_server)
    
    # If API-only mode, return early
    if args.api_only:
        logger.info("Running in API-only mode")
        return engine
    
    # Parse symbols
    symbols = args.symbols.split(",")
    
    # Setup paper trading or live exchange
    if args.paper_trading:
        logger.info("Running in paper trading mode with Binance Testnet")
        # Use testnet for paper trading
        exchange = CCXTExchange(args.exchange, config={"testnet": True})
        engine.register_component(exchange)
        
        # Add paper trading manager
        paper_trading_manager = PaperTradingManager()
        engine.register_component(paper_trading_manager)
        
        # Create default account if none exists
        await paper_trading_manager.ensure_default_account()
    else:
        # Use regular exchange
        logger.info(f"Using live exchange: {args.exchange}")
        exchange = CCXTExchange(args.exchange)
        engine.register_component(exchange)
    
    # Market data components
    market_data = MarketDataManager()
    engine.register_component(market_data)
    
    market_analyzer = MarketAnalyzer()
    engine.register_component(market_analyzer)
    
    # Strategies
    ma_crossover = MovingAverageCrossover(fast_period=20, slow_period=50)
    engine.register_component(ma_crossover)
    
    macd_strategy = MACDStrategy()
    engine.register_component(macd_strategy)
    
    rsi_strategy = RSIStrategy()
    engine.register_component(rsi_strategy)
    
    bb_strategy = BollingerBandsStrategy()
    engine.register_component(bb_strategy)
    
    # Risk management
    risk_manager = RiskManager()
    engine.register_component(risk_manager)
    
    # Analytics
    performance_tracker = PerformanceTracker()
    engine.register_component(performance_tracker)
    
    visualizer = Visualizer()
    engine.register_component(visualizer)
    
    report_generator = ReportGenerator()
    engine.register_component(report_generator)
    
    # Workers
    scheduler = Scheduler()
    engine.register_component(scheduler)
    
    task_processor = TaskProcessor()
    engine.register_component(task_processor)
    
    # Add scheduled tasks
    # Daily performance report
    async def generate_daily_report():
        await report_generator.generate_daily_report()
    
    scheduler.add_task(Task(
        name="daily_report",
        coro_func=generate_daily_report,
        cron="0 0 * * *"  # Midnight every day
    ))
    
    # Update market data every minute
    async def update_market_data():
        for symbol in symbols:
            await market_data.fetch_latest_candles(symbol, "1m")
    
    scheduler.add_task(Task(
        name="update_market_data",
        coro_func=update_market_data,
        interval=60  # Every minute
    ))
    
    logger.info("System setup complete")
    return engine


async def main():
    """Main entry point for the system."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up the system
    engine = await setup_system(args)
    
    # Start the engine
    await engine.start()
    
    # Run until shutdown
    await engine.run_forever()


if __name__ == "__main__":
    try:
        # Run the main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("System shutdown requested by user")
    except Exception as e:
        print(f"Unhandled exception: {e}")
        raise

