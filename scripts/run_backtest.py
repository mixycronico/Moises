#!/usr/bin/env python3
"""
Script para ejecutar un backtest completo con datos de Binance Testnet.

Este script utiliza el motor de backtesting de Genesis junto con los
módulos de gestión de riesgos para evaluar el rendimiento de una estrategia
de trading en datos históricos.
"""
import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Asegurar que el directorio raíz esté en el path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from genesis.risk.risk_manager import RiskManager
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLoss
from genesis.backtesting.engine import BacktestEngine
from genesis.strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from genesis.strategies.base import Strategy, SignalType
from genesis.backtesting.portfolio import BacktestPortfolio
from genesis.utils.logger import setup_logging

# Configurar logging
logger = setup_logging("backtest", "INFO")

class SimpleSMAStrategy(Strategy):
    """
    Estrategia simple basada en cruce de medias móviles (SMA).
    
    Esta estrategia genera señales de compra cuando la SMA rápida cruza
    por encima de la SMA lenta, y señales de venta cuando cruza por debajo.
    """
    
    def __init__(
        self, 
        name="simple_sma_strategy", 
        fast_period=20, 
        slow_period=50,
        timeframe="1d"
    ):
        """
        Inicializar la estrategia.
        
        Args:
            name: Nombre de la estrategia
            fast_period: Período para la SMA rápida
            slow_period: Período para la SMA lenta
            timeframe: Marco temporal
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.timeframe = timeframe
        self.logger = logging.getLogger(__name__)
    
    async def generate_signal(self, symbol, data):
        """
        Generar señales de trading basadas en cruces de SMA.
        
        Args:
            symbol: Símbolo de trading
            data: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con la señal generada
        """
        if len(data) < self.slow_period:
            return {"signal_type": SignalType.HOLD, "symbol": symbol}
        
        # Calcular medias móviles
        data['sma_fast'] = data['close'].rolling(window=self.fast_period).mean()
        data['sma_slow'] = data['close'].rolling(window=self.slow_period).mean()
        
        # Identificar cruces (1 para cruce alcista, -1 para cruce bajista, 0 sin cruce)
        data['sma_cross'] = 0
        data.loc[(data['sma_fast'] > data['sma_slow']) & 
                 (data['sma_fast'].shift(1) <= data['sma_slow'].shift(1)), 'sma_cross'] = 1
        data.loc[(data['sma_fast'] < data['sma_slow']) & 
                 (data['sma_fast'].shift(1) >= data['sma_slow'].shift(1)), 'sma_cross'] = -1
        
        # Obtener la última señal
        current_price = data['close'].iloc[-1]
        current_cross = data['sma_cross'].iloc[-1]
        
        if current_cross == 1:
            return {
                "signal_type": SignalType.BUY,
                "symbol": symbol,
                "price": current_price,
                "timestamp": data.index[-1]
            }
        elif current_cross == -1:
            return {
                "signal_type": SignalType.SELL,
                "symbol": symbol,
                "price": current_price,
                "timestamp": data.index[-1]
            }
        else:
            return {
                "signal_type": SignalType.HOLD,
                "symbol": symbol,
                "price": current_price,
                "timestamp": data.index[-1]
            }

def load_testnet_data(symbol, timeframe):
    """
    Cargar datos de trading de Binance Testnet desde CSV.
    
    Args:
        symbol: Símbolo de trading (e.g., "BTC/USDT")
        timeframe: Marco temporal (e.g., "1h", "1d")
        
    Returns:
        DataFrame con datos OHLCV
    """
    clean_symbol = symbol.replace("/", "_")
    file_path = os.path.join(root_dir, "data", f"{clean_symbol}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"No se encontraron datos para {symbol} en timeframe {timeframe}")
        logger.info(f"Buscando en: {file_path}")
        # Listar los archivos disponibles
        data_dir = os.path.join(root_dir, "data")
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            logger.info(f"Archivos disponibles: {files}")
        return None
    
    # Cargar datos
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    
    # Asegurarse de que las columnas coinciden con lo esperado por el motor de backtesting
    df.columns = [col.lower() for col in df.columns]
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Columna requerida {col} no encontrada en los datos")
            return None
    
    return df

async def run_sma_backtest(symbol="BTC/USDT", timeframe="1d", fast_period=20, slow_period=50):
    """
    Ejecutar un backtest con la estrategia de cruce de medias móviles.
    
    Args:
        symbol: Símbolo de trading
        timeframe: Marco temporal
        fast_period: Período para la SMA rápida
        slow_period: Período para la SMA lenta
    """
    logger.info(f"Iniciando backtest para {symbol} en timeframe {timeframe}")
    
    # Cargar datos
    data = load_testnet_data(symbol, timeframe)
    if data is None:
        logger.error("No se pudieron cargar los datos. Abortando backtest.")
        return
    
    logger.info(f"Datos cargados: {len(data)} registros de {data.index[0]} a {data.index[-1]}")
    
    # Configurar stop loss y position sizer
    stop_loss_calculator = StopLoss(risk_percentage=0.05)
    position_sizer = PositionSizer(risk_per_trade=0.02)
    
    # Configurar el motor de backtest
    backtest_engine = BacktestEngine(
        initial_capital=10000,
        risk_per_trade=0.02,
        stop_loss_calculator=stop_loss_calculator,
        position_sizer=position_sizer,
        fee_rate=0.001  # 0.1% de comisión
    )
    
    # Configurar la estrategia
    strategy = SimpleSMAStrategy(
        fast_period=fast_period,
        slow_period=slow_period,
        timeframe=timeframe
    )
    
    # Ejecutar el backtest
    results, stats = await backtest_engine.run_backtest(
        strategy=strategy,
        data={symbol: data},
        symbol=symbol,
        timeframe=timeframe
    )
    
    # Mostrar resultados
    logger.info("======= RESULTADOS DEL BACKTEST =======")
    logger.info(f"Capital inicial: ${backtest_engine.initial_capital:.2f}")
    logger.info(f"Capital final: ${stats['final_capital']:.2f}")
    logger.info(f"Rendimiento total: {stats['total_return']:.2f}%")
    logger.info(f"Drawdown máximo: {stats['max_drawdown']:.2f}%")
    logger.info(f"Ratio de Sharpe: {stats['sharpe_ratio']:.2f}")
    logger.info(f"Número de operaciones: {stats['total_trades']}")
    logger.info(f"Ratio de aciertos: {stats['win_rate']:.2f}%")
    logger.info(f"Profit factor: {stats['profit_factor']:.2f}")
    
    # Crear y guardar gráfico
    plot_backtest_results(data, results, stats, symbol, fast_period, slow_period)
    
    return results, stats

def plot_backtest_results(data, results, stats, symbol, fast_period, slow_period):
    """
    Visualizar los resultados del backtest.
    
    Args:
        data: DataFrame con datos OHLCV
        results: Resultados del backtest
        stats: Estadísticas del backtest
        symbol: Símbolo de trading
        fast_period: Período de la SMA rápida
        slow_period: Período de la SMA lenta
    """
    # Calcular SMA para la visualización
    data['sma_fast'] = data['close'].rolling(window=fast_period).mean()
    data['sma_slow'] = data['close'].rolling(window=slow_period).mean()
    
    # Extraer datos de los resultados
    trades = results.get('trades', [])
    equity_curve = results.get('equity_curve', [])
    
    # Crear figura
    plt.figure(figsize=(15, 12))
    
    # Gráfico 1: Precios y SMA
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'], label='Precio', color='blue', alpha=0.5)
    plt.plot(data.index, data['sma_fast'], label=f'SMA {fast_period}', color='orange')
    plt.plot(data.index, data['sma_slow'], label=f'SMA {slow_period}', color='green')
    
    # Marcar operaciones en el gráfico
    buy_dates = [trade['entry_time'] for trade in trades if trade.get('side') == 'buy' and 'entry_time' in trade]
    buy_prices = [trade['entry_price'] for trade in trades if trade.get('side') == 'buy' and 'entry_price' in trade]
    sell_dates = [trade['exit_time'] for trade in trades if 'exit_time' in trade and 'exit_price' in trade]
    sell_prices = [trade['exit_price'] for trade in trades if 'exit_time' in trade and 'exit_price' in trade]
    
    plt.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Compra')
    plt.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Venta')
    
    plt.title(f'Estrategia de Cruce de SMA {fast_period}/{slow_period} - {symbol}')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Curva de capital
    plt.subplot(3, 1, 2)
    equity_index = data.index[:len(equity_curve)]
    plt.plot(equity_index, equity_curve, label='Capital', color='purple')
    plt.title('Curva de Capital')
    plt.xlabel('Fecha')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 3: Drawdown
    plt.subplot(3, 1, 3)
    equity_series = pd.Series(equity_curve, index=equity_index)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    plt.plot(equity_index, drawdown, label='Drawdown', color='red')
    plt.title('Drawdown (%)')
    plt.xlabel('Fecha')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Añadir texto con estadísticas
    text = (
        f"Capital inicial: ${stats['initial_capital']:.2f}\n"
        f"Capital final: ${stats['final_capital']:.2f}\n"
        f"Rendimiento: {stats['total_return']:.2f}%\n"
        f"Drawdown máximo: {stats['max_drawdown']:.2f}%\n"
        f"Ratio de Sharpe: {stats['sharpe_ratio']:.2f}\n"
        f"Operaciones: {stats['total_trades']}\n"
        f"Win rate: {stats['win_rate']:.2f}%\n"
        f"Profit factor: {stats['profit_factor']:.2f}"
    )
    plt.figtext(0.02, 0.02, text, fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_dir = os.path.join(root_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"backtest_{symbol.replace('/', '_')}_{fast_period}_{slow_period}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    logger.info(f"Gráfico guardado en: {filepath}")
    
    # Mostrar gráfico
    plt.show()

async def main():
    """Función principal."""
    # Ejecutar backtest con diferentes parámetros para SMA
    fast_periods = [10, 20, 50]
    slow_periods = [50, 100, 200]
    
    symbol = "BTC/USDT"
    timeframe = "1d"
    
    best_return = -float('inf')
    best_params = None
    best_stats = None
    
    logger.info(f"Optimizando parámetros para {symbol} en timeframe {timeframe}")
    
    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue  # Saltar combinaciones inválidas
                
            logger.info(f"Probando SMA: {fast}/{slow}")
            _, stats = await run_sma_backtest(
                symbol=symbol, 
                timeframe=timeframe,
                fast_period=fast,
                slow_period=slow
            )
            
            if stats['total_return'] > best_return:
                best_return = stats['total_return']
                best_params = (fast, slow)
                best_stats = stats
    
    logger.info("======= MEJORES PARÁMETROS =======")
    logger.info(f"SMA rápida: {best_params[0]}")
    logger.info(f"SMA lenta: {best_params[1]}")
    logger.info(f"Rendimiento: {best_return:.2f}%")
    logger.info(f"Drawdown máximo: {best_stats['max_drawdown']:.2f}%")
    logger.info(f"Ratio de Sharpe: {best_stats['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())