"""
Pruebas unitarias para el módulo de backtesting.

Este módulo prueba las funcionalidades de backtesting del sistema,
incluyendo simulación de estrategias, gestión de datos históricos, y
evaluación de rendimiento en datos pasados.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import datetime

# Importar los componentes del sistema relacionados con backtesting
from genesis.backtesting.engine import BacktestEngine
from genesis.strategies.base import Strategy, SignalType
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator


class TestStrategy(Strategy):
    """Estrategia de prueba para backtesting."""
    
    def __init__(self, name="test_strategy"):
        super().__init__(name)
        self.params = {"param1": 10, "param2": 20}
    
    async def generate_signal(self, symbol, data):
        """Generar señales de prueba para backtesting."""
        # Señal simple: comprar cuando el precio sube, vender cuando baja
        if len(data) < 2:
            return {"symbol": symbol, "signal_type": SignalType.HOLD, "price": data["close"].iloc[-1]}
        
        last_price = data["close"].iloc[-1]
        prev_price = data["close"].iloc[-2]
        
        if last_price > prev_price:
            signal_type = SignalType.BUY
        elif last_price < prev_price:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        return {
            "symbol": symbol,
            "signal_type": signal_type,
            "price": last_price,
            "timestamp": data.index[-1],
            "metadata": {
                "strategy": self.name,
                "params": self.params
            }
        }


@pytest.fixture
def sample_ohlcv_data():
    """Generar datos OHLCV de ejemplo para pruebas de backtesting."""
    # Crear un DataFrame con 100 filas de datos OHLCV
    np.random.seed(42)  # Para reproducibilidad
    
    dates = pd.date_range("2025-01-01", periods=100, freq="1h")
    
    # Crear un camino aleatorio para los precios
    close_prices = np.random.normal(0, 1, 100).cumsum() + 40000
    
    # Calcular otros campos OHLCV basados en los precios de cierre
    df = pd.DataFrame({
        "open": close_prices[:-1].tolist() + [close_prices[-1]],
        "high": close_prices + np.random.uniform(50, 200, 100),
        "low": close_prices - np.random.uniform(50, 200, 100),
        "close": close_prices,
        "volume": np.random.lognormal(10, 1, 100),
    }, index=dates)
    
    return df


@pytest.fixture
def backtest_engine():
    """Crear una instancia de motor de backtesting para pruebas."""
    # Configurar componentes simulados
    strategy = TestStrategy()
    position_sizer = PositionSizer()
    stop_loss_calculator = StopLossCalculator()
    
    # Crear engine de backtesting
    engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.0005
    )
    
    return engine


@pytest.mark.asyncio
async def test_backtest_engine_initialization(backtest_engine):
    """Probar la inicialización del motor de backtesting."""
    assert backtest_engine.initial_capital == 10000.0
    assert backtest_engine.current_capital == 10000.0  # Asumiendo que existe esta propiedad
    assert backtest_engine.commission == 0.001
    assert backtest_engine.slippage == 0.0005
    assert backtest_engine.positions == {}
    assert backtest_engine.trade_history == []
    assert backtest_engine.equity_curve == []


@pytest.mark.asyncio
async def test_backtest_run_simple(backtest_engine, sample_ohlcv_data):
    """Probar la ejecución básica de un backtest."""
    # Configurar la estrategia
    strategy = TestStrategy()
    
    # Preparar datos para una simulación simple
    symbol = "BTC/USDT"
    start_date = "2025-01-01"
    end_date = "2025-01-05"
    
    # Ejecutar el backtest
    results, stats = await backtest_engine.run_backtest(
        strategy=strategy,
        data={symbol: sample_ohlcv_data},
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Verificar los resultados básicos
    assert isinstance(results, dict)
    assert "trades" in results
    assert "equity_curve" in results
    assert "signals" in results
    
    # Verificar estadísticas
    assert isinstance(stats, dict)
    assert "total_trades" in stats
    assert "profit_loss" in stats
    assert "win_rate" in stats
    assert "profit_factor" in stats
    assert "max_drawdown" in stats


@pytest.mark.asyncio
async def test_backtest_position_management(backtest_engine, sample_ohlcv_data):
    """Probar la gestión de posiciones durante el backtesting."""
    # Mockear la generación de señales para forzar una secuencia específica
    strategy = TestStrategy()
    
    # Modificar la función generate_signal para dar señales predefinidas
    signals = [
        {"symbol": "BTC/USDT", "signal_type": SignalType.BUY, "price": 40000},  # Abrir long
        {"symbol": "BTC/USDT", "signal_type": SignalType.HOLD, "price": 40100},  # Mantener
        {"symbol": "BTC/USDT", "signal_type": SignalType.SELL, "price": 41000},  # Cerrar long
        {"symbol": "BTC/USDT", "signal_type": SignalType.SELL, "price": 41000},  # Abrir short
        {"symbol": "BTC/USDT", "signal_type": SignalType.HOLD, "price": 40900},  # Mantener
        {"symbol": "BTC/USDT", "signal_type": SignalType.BUY, "price": 40500}    # Cerrar short
    ]
    
    async def mock_generate_signal(symbol, data):
        # Devolver la señal correspondiente al índice actual
        index = min(len(backtest_engine.equity_curve), len(signals) - 1)
        signal = signals[index].copy()
        signal["timestamp"] = data.index[-1]
        print(f"⏩ Generando señal {signal['signal_type']} en {signal['timestamp']}")
        return signal
    
    strategy.generate_signal = mock_generate_signal
    
    # Ejecutar el backtest
    symbol = "BTC/USDT"
    results, stats = await backtest_engine.run_backtest(
        strategy=strategy,
        data={symbol: sample_ohlcv_data},
        symbol=symbol,
        timeframe="1h"
    )
    
    # Verificar resultados
    assert len(results["trades"]) == 4  # 2 operaciones completas (open + close)
    
    # Verificar que las operaciones long y short se manejaron correctamente
    trades = results["trades"]
    
    # Primera operación: Long
    assert trades[0]["side"] == "buy"
    assert trades[0]["price"] == 40000
    assert trades[1]["side"] == "sell"
    assert trades[1]["price"] == 41000
    assert trades[1]["profit_loss"] > 0  # Operación rentable
    
    # Segunda operación: Short
    assert trades[2]["side"] == "sell"
    assert trades[2]["price"] == 41000
    assert trades[3]["side"] == "buy"
    assert trades[3]["price"] == 40500
    assert trades[3]["profit_loss"] > 0  # Operación rentable


@pytest.mark.asyncio
async def test_backtest_risk_management(backtest_engine, sample_ohlcv_data):
    """Probar la gestión de riesgos durante el backtesting."""
    # Configurar la estrategia
    strategy = TestStrategy()
    
    # Configurar parámetros de gestión de riesgos
    backtest_engine.risk_per_trade = 0.02  # 2% de riesgo por operación
    backtest_engine.use_stop_loss = True
    backtest_engine.use_trailing_stop = True
    
    # Configurar un stop loss calculator simulado
    mock_stop_loss = Mock()
    mock_stop_loss.calculate.return_value = {"price": 39500, "percentage": 0.0125}  # 1.25% de stop loss
    mock_stop_loss.calculate_trailing_stop.return_value = {"price": 39800, "activated": True}
    
    backtest_engine.stop_loss_calculator = mock_stop_loss
    
    # Ejecutar el backtest
    symbol = "BTC/USDT"
    results, stats = await backtest_engine.run_backtest(
        strategy=strategy,
        data={symbol: sample_ohlcv_data},
        symbol=symbol,
        timeframe="1h"
    )
    
    # Verificar que se llamó al calculador de stop loss
    assert mock_stop_loss.calculate.called
    
    # Verificar que las operaciones incluyen información de stop loss
    for trade in results["trades"]:
        if trade["side"] == "buy":  # Solo para operaciones de apertura
            assert "stop_loss" in trade
            assert "risk_amount" in trade


@pytest.mark.asyncio
async def test_backtest_optimization(backtest_engine, sample_ohlcv_data):
    """Probar la optimización de parámetros en backtesting."""
    # Crear una estrategia con parámetros optimizables
    strategy = TestStrategy()
    
    # Definir espacio de parámetros
    param_space = {
        "param1": [5, 10, 15],
        "param2": [10, 20, 30]
    }
    
    # Ejecutar optimización
    optimization_results = await backtest_engine.optimize_strategy(
        strategy=strategy,
        data={"BTC/USDT": sample_ohlcv_data},
        symbol="BTC/USDT",
        param_space=param_space,
        metric="profit_loss"  # Optimizar para mayor beneficio
    )
    
    # Verificar resultados de optimización
    assert isinstance(optimization_results, list)
    assert len(optimization_results) == len(param_space["param1"]) * len(param_space["param2"])
    
    # Ordenar resultados por rendimiento
    sorted_results = sorted(optimization_results, key=lambda x: x["metrics"]["profit_loss"], reverse=True)
    
    # Verificar que los resultados están ordenados correctamente
    assert sorted_results[0]["metrics"]["profit_loss"] >= sorted_results[-1]["metrics"]["profit_loss"]
    
    # Verificar que los parámetros óptimos están disponibles
    assert "params" in sorted_results[0]
    assert "param1" in sorted_results[0]["params"]
    assert "param2" in sorted_results[0]["params"]


@pytest.mark.asyncio
async def test_backtest_with_different_timeframes(backtest_engine):
    """Probar el backtesting con diferentes marcos temporales."""
    # Generar datos para diferentes timeframes
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    data = {}
    
    for tf in timeframes:
        # Generar datos OHLCV para cada timeframe
        periods = 100 if tf != "1d" else 30  # Menos días para el timeframe diario
        dates = pd.date_range("2025-01-01", periods=periods, freq=tf)
        
        np.random.seed(42)  # Para reproducibilidad
        close_prices = np.random.normal(0, 1, periods).cumsum() + 40000
        
        df = pd.DataFrame({
            "open": close_prices[:-1].tolist() + [close_prices[-1]],
            "high": close_prices + np.random.uniform(50, 200, periods),
            "low": close_prices - np.random.uniform(50, 200, periods),
            "close": close_prices,
            "volume": np.random.lognormal(10, 1, periods),
        }, index=dates)
        
        data[f"BTC/USDT_{tf}"] = df
    
    # Probar backtesting en cada timeframe
    for tf in timeframes:
        symbol = f"BTC/USDT_{tf}"
        
        # Ejecutar el backtest
        results, stats = await backtest_engine.run_backtest(
            strategy=TestStrategy(),
            data={symbol: data[symbol]},
            symbol=symbol,
            timeframe=tf
        )
        
        # Verificar resultados básicos para cada timeframe
        assert "trades" in results
        assert "equity_curve" in results
        assert "total_trades" in stats
        
        # El número de operaciones debería variar según el timeframe
        print(f"Timeframe {tf}: {stats['total_trades']} trades")


@pytest.mark.asyncio
async def test_backtest_with_multiple_assets(backtest_engine):
    """Probar el backtesting con múltiples activos."""
    # Generar datos para diferentes activos
    assets = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    data = {}
    
    for asset in assets:
        # Generar datos OHLCV para cada activo
        np.random.seed(42 + assets.index(asset))  # Diferente semilla para cada activo
        periods = 100
        dates = pd.date_range("2025-01-01", periods=periods, freq="1h")
        
        base_price = 40000 if asset == "BTC/USDT" else 3000 if asset == "ETH/USDT" else 1.0
        close_prices = np.random.normal(0, 1, periods).cumsum() + base_price
        
        df = pd.DataFrame({
            "open": close_prices[:-1].tolist() + [close_prices[-1]],
            "high": close_prices + np.random.uniform(base_price * 0.005, base_price * 0.01, periods),
            "low": close_prices - np.random.uniform(base_price * 0.005, base_price * 0.01, periods),
            "close": close_prices,
            "volume": np.random.lognormal(10, 1, periods),
        }, index=dates)
        
        data[asset] = df
    
    # Ejecutar backtesting multi-activo
    results_combined, stats_combined = await backtest_engine.run_multi_asset_backtest(
        strategy=TestStrategy(),
        data=data,
        timeframe="1h"
    )
    
    # Verificar resultados
    assert isinstance(results_combined, dict)
    assert all(asset in results_combined for asset in assets)
    
    # Verificar estadísticas combinadas
    assert "total_trades" in stats_combined
    assert "profit_loss" in stats_combined
    assert "win_rate" in stats_combined
    
    # Las estadísticas combinadas deberían reflejar la suma de todas las operaciones
    total_trades = sum(len(results_combined[asset]["trades"]) for asset in assets if "trades" in results_combined[asset])
    assert stats_combined["total_trades"] == total_trades