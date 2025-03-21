"""
Pruebas unitarias para el módulo de análisis de rendimiento.

Este módulo prueba las funcionalidades relacionadas con el análisis de
rendimiento de estrategias, métricas financieras, y visualización de datos.
"""

import pytest
import datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from genesis.analytics.performance_analyzer import PerformanceAnalyzer
from genesis.core.event_bus import EventBus


@pytest.fixture
def sample_trades_data():
    """Generar datos de prueba para análisis de rendimiento."""
    return [
        {
            "trade_id": "T1",
            "symbol": "BTC/USDT",
            "side": "buy",
            "entry_price": 40000,
            "exit_price": 42000,
            "amount": 0.1,
            "entry_time": datetime.datetime(2025, 1, 1, 10, 0, 0),
            "exit_time": datetime.datetime(2025, 1, 1, 14, 0, 0),
            "profit_loss": 200.0,
            "profit_loss_pct": 5.0,
            "strategy": "macd_cross"
        },
        {
            "trade_id": "T2",
            "symbol": "BTC/USDT",
            "side": "sell",
            "entry_price": 42000,
            "exit_price": 41000,
            "amount": 0.1,
            "entry_time": datetime.datetime(2025, 1, 2, 10, 0, 0),
            "exit_time": datetime.datetime(2025, 1, 2, 16, 0, 0),
            "profit_loss": 100.0,
            "profit_loss_pct": 2.38,
            "strategy": "macd_cross"
        },
        {
            "trade_id": "T3",
            "symbol": "ETH/USDT",
            "side": "buy",
            "entry_price": 3000,
            "exit_price": 2900,
            "amount": 1.0,
            "entry_time": datetime.datetime(2025, 1, 3, 10, 0, 0),
            "exit_time": datetime.datetime(2025, 1, 3, 15, 0, 0),
            "profit_loss": -100.0,
            "profit_loss_pct": -3.33,
            "strategy": "rsi"
        },
        {
            "trade_id": "T4",
            "symbol": "ETH/USDT",
            "side": "buy",
            "entry_price": 2900,
            "exit_price": 3100,
            "amount": 1.0,
            "entry_time": datetime.datetime(2025, 1, 4, 10, 0, 0),
            "exit_time": datetime.datetime(2025, 1, 4, 12, 0, 0),
            "profit_loss": 200.0,
            "profit_loss_pct": 6.9,
            "strategy": "rsi"
        },
        {
            "trade_id": "T5",
            "symbol": "BTC/USDT",
            "side": "buy",
            "entry_price": 41000,
            "exit_price": 40500,
            "amount": 0.1,
            "entry_time": datetime.datetime(2025, 1, 5, 10, 0, 0),
            "exit_time": datetime.datetime(2025, 1, 5, 11, 0, 0),
            "profit_loss": -50.0,
            "profit_loss_pct": -1.22,
            "strategy": "bollinger_bands"
        }
    ]


@pytest.fixture
def sample_balance_history():
    """Generar datos de histórico de saldo para pruebas."""
    dates = pd.date_range(start="2025-01-01", end="2025-01-10", freq="D")
    balances = [10000, 10200, 10300, 10250, 10450, 10400, 10600, 10700, 10650, 10800]
    return pd.DataFrame({"date": dates, "balance": balances})


@pytest.fixture
def performance_analyzer():
    """Crear instancia de PerformanceAnalyzer para pruebas."""
    event_bus = EventBus()
    analyzer = PerformanceAnalyzer(event_bus=event_bus)
    return analyzer


def test_performance_analyzer_calculate_metrics(performance_analyzer, sample_trades_data):
    """Probar el cálculo de métricas básicas de rendimiento."""
    # Convertir los datos a un DataFrame
    trades_df = pd.DataFrame(sample_trades_data)
    
    # Calcular métricas básicas
    metrics = performance_analyzer.calculate_basic_metrics(trades_df)
    
    # Verificar las métricas calculadas
    assert metrics["total_trades"] == 5
    assert metrics["winning_trades"] == 3
    assert metrics["losing_trades"] == 2
    assert metrics["win_rate"] == 0.6
    assert abs(metrics["total_profit_loss"] - 350.0) < 0.01
    assert abs(metrics["avg_profit_pct"] - 1.946) < 0.01  # (5.0 + 2.38 + (-3.33) + 6.9 + (-1.22)) / 5
    assert abs(metrics["avg_win_pct"] - 4.76) < 0.01  # (5.0 + 2.38 + 6.9) / 3
    assert abs(metrics["avg_loss_pct"] - (-2.275)) < 0.01  # ((-3.33) + (-1.22)) / 2
    assert abs(metrics["profit_factor"] - 5.0) < 0.01  # (200 + 100 + 200) / (100 + 50)


def test_performance_analyzer_calculate_advanced_metrics(performance_analyzer, sample_trades_data, sample_balance_history):
    """Probar el cálculo de métricas avanzadas de rendimiento."""
    # Convertir los datos a un DataFrame
    trades_df = pd.DataFrame(sample_trades_data)
    
    # Calcular métricas avanzadas
    metrics = performance_analyzer.calculate_advanced_metrics(trades_df, sample_balance_history)
    
    # Verificar las métricas calculadas
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "max_drawdown_pct" in metrics
    assert "volatility" in metrics
    assert "calmar_ratio" in metrics
    
    # Verificar valores específicos (los valores dependen de la implementación exacta)
    assert metrics["sharpe_ratio"] > 0  # Debería ser positivo para una estrategia rentable
    assert metrics["max_drawdown"] > 0  # Debería ser positivo (es una pérdida)
    assert metrics["max_drawdown_pct"] > 0  # Debería ser positivo (es un porcentaje de pérdida)
    assert metrics["volatility"] > 0  # Debería ser positivo


def test_performance_analyzer_analyze_by_symbol(performance_analyzer, sample_trades_data):
    """Probar el análisis de rendimiento por símbolo."""
    # Convertir los datos a un DataFrame
    trades_df = pd.DataFrame(sample_trades_data)
    
    # Analizar por símbolo
    symbol_performance = performance_analyzer.analyze_by_symbol(trades_df)
    
    # Verificar resultados
    assert "BTC/USDT" in symbol_performance
    assert "ETH/USDT" in symbol_performance
    
    # Verificar métricas para BTC/USDT
    btc_metrics = symbol_performance["BTC/USDT"]
    assert btc_metrics["total_trades"] == 3
    assert btc_metrics["winning_trades"] == 2
    assert btc_metrics["losing_trades"] == 1
    assert btc_metrics["win_rate"] == 2/3
    assert abs(btc_metrics["total_profit_loss"] - 250.0) < 0.01
    
    # Verificar métricas para ETH/USDT
    eth_metrics = symbol_performance["ETH/USDT"]
    assert eth_metrics["total_trades"] == 2
    assert eth_metrics["winning_trades"] == 1
    assert eth_metrics["losing_trades"] == 1
    assert eth_metrics["win_rate"] == 0.5
    assert abs(eth_metrics["total_profit_loss"] - 100.0) < 0.01


def test_performance_analyzer_analyze_by_strategy(performance_analyzer, sample_trades_data):
    """Probar el análisis de rendimiento por estrategia."""
    # Convertir los datos a un DataFrame
    trades_df = pd.DataFrame(sample_trades_data)
    
    # Analizar por estrategia
    strategy_performance = performance_analyzer.analyze_by_strategy(trades_df)
    
    # Verificar resultados
    assert "macd_cross" in strategy_performance
    assert "rsi" in strategy_performance
    assert "bollinger_bands" in strategy_performance
    
    # Verificar métricas para macd_cross
    macd_metrics = strategy_performance["macd_cross"]
    assert macd_metrics["total_trades"] == 2
    assert macd_metrics["winning_trades"] == 2
    assert macd_metrics["losing_trades"] == 0
    assert macd_metrics["win_rate"] == 1.0
    assert abs(macd_metrics["total_profit_loss"] - 300.0) < 0.01
    
    # Verificar métricas para rsi
    rsi_metrics = strategy_performance["rsi"]
    assert rsi_metrics["total_trades"] == 2
    assert rsi_metrics["winning_trades"] == 1
    assert rsi_metrics["losing_trades"] == 1
    assert rsi_metrics["win_rate"] == 0.5
    assert abs(rsi_metrics["total_profit_loss"] - 100.0) < 0.01
    
    # Verificar métricas para bollinger_bands
    bb_metrics = strategy_performance["bollinger_bands"]
    assert bb_metrics["total_trades"] == 1
    assert bb_metrics["winning_trades"] == 0
    assert bb_metrics["losing_trades"] == 1
    assert bb_metrics["win_rate"] == 0.0
    assert abs(bb_metrics["total_profit_loss"] - (-50.0)) < 0.01


def test_performance_analyzer_analyze_by_time(performance_analyzer, sample_trades_data):
    """Probar el análisis de rendimiento por tiempo."""
    # Convertir los datos a un DataFrame
    trades_df = pd.DataFrame(sample_trades_data)
    
    # Analizar por tiempo (diario)
    time_performance = performance_analyzer.analyze_by_time(trades_df, timeframe="D")
    
    # Verificar resultados
    assert len(time_performance) == 5  # 5 días diferentes
    
    # Verificar que las fechas estén en el formato correcto
    for date_str in time_performance.keys():
        assert isinstance(date_str, str)
        # Asegurar que la fecha se puede parsear
        datetime.datetime.strptime(date_str, "%Y-%m-%d")


@patch("matplotlib.pyplot.figure")
@patch("matplotlib.pyplot.savefig")
def test_performance_analyzer_generate_report(mock_savefig, mock_figure, 
                                             performance_analyzer, sample_trades_data, 
                                             sample_balance_history):
    """Probar la generación de un informe completo."""
    # Convertir los datos a un DataFrame
    trades_df = pd.DataFrame(sample_trades_data)
    
    # Generar un informe
    report = performance_analyzer.generate_report(trades_df, sample_balance_history, 
                                                include_plots=True, save_plots=True)
    
    # Verificar que el informe contenga las secciones esperadas
    assert "overall_metrics" in report
    assert "symbol_performance" in report
    assert "strategy_performance" in report
    assert "time_performance" in report
    assert "advanced_metrics" in report
    
    # Verificar que se llamó a las funciones de gráficos
    mock_figure.assert_called()
    mock_savefig.assert_called()


@pytest.mark.asyncio
async def test_performance_analyzer_handle_events(performance_analyzer, sample_trades_data):
    """Probar el manejo de eventos en el analizador de rendimiento."""
    # Configurar el analizador para recibir eventos
    await performance_analyzer.start()
    
    # Convertir los datos a un DataFrame
    trade_data = sample_trades_data[0]
    
    # Crear un mock para el método analyze_trade
    performance_analyzer.analyze_trade = MagicMock()
    
    # Simular un evento de operación cerrada
    await performance_analyzer.handle_event("trade.closed", trade_data, "exchange_manager")
    
    # Verificar que se llamó al método analyze_trade con los datos correctos
    performance_analyzer.analyze_trade.assert_called_once_with(trade_data)
    
    # Detener el analizador
    await performance_analyzer.stop()