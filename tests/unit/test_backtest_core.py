"""
Pruebas unitarias básicas para el módulo de backtesting.

Este módulo prueba las funcionalidades esenciales del motor de backtesting
con tests simplificados para evitar timeouts y problemas de recursión.
"""
import asyncio
import logging
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from genesis.backtesting.engine import BacktestEngine
from genesis.core.base import Component
from genesis.strategies.base import Strategy
from verify_strategies import SignalType


class TestStrategy(Strategy):
    """Estrategia de prueba simplificada para backtesting."""
    
    def __init__(self, name="test_strategy"):
        super().__init__(name)
        self.params = {}
        self.signal_index = 0  # Contador para llevar un registro de las llamadas
        
    async def generate_signal(self, symbol, data):
        """
        Generar señales predefinidas para testeo.
        
        En lugar de depender de equity_curve u otras variables internas,
        usa un contador simple para devolver señales en secuencia.
        """
        # Lista de señales predefinidas para posición
        signals = [
            {"symbol": symbol, "signal_type": SignalType.BUY, "price": 40000},    # Abrir long
            {"symbol": symbol, "signal_type": SignalType.HOLD, "price": 40100},   # Mantener
            {"symbol": symbol, "signal_type": SignalType.SELL, "price": 41000},   # Cerrar long
            {"symbol": symbol, "signal_type": SignalType.SELL, "price": 41000},   # Abrir short
            {"symbol": symbol, "signal_type": SignalType.HOLD, "price": 40900},   # Mantener
            {"symbol": symbol, "signal_type": SignalType.BUY, "price": 40500}     # Cerrar short
        ]
        
        # Obtener la siguiente señal y avanzar el contador
        index = min(self.signal_index, len(signals) - 1)
        signal = signals[index].copy()
        signal["timestamp"] = data.index[-1] if not data.empty else pd.Timestamp.now()
        
        # Incrementar el índice para la próxima llamada
        self.signal_index += 1
        
        return signal


class RiskStrategy(Strategy):
    """Estrategia para probar gestión de riesgos."""
    
    def __init__(self, name="risk_strategy"):
        super().__init__(name)
        self.params = {}
        self.signal_index = 0  # Contador para llevar un registro de las llamadas
        
    async def generate_signal(self, symbol, data):
        """Generar señales para probar stop loss."""
        # Lista de señales predefinidas para probar stop loss
        signals = [
            {"symbol": symbol, "signal_type": SignalType.BUY, "price": 40000},     # Abrir long
            {"symbol": symbol, "signal_type": SignalType.HOLD, "price": 40100},    # Subir precio
            {"symbol": symbol, "signal_type": SignalType.HOLD, "price": 39800},    # Bajar hacia stop loss
            {"symbol": symbol, "signal_type": SignalType.EXIT, "price": 39600}     # Activar stop loss
        ]
        
        # Obtener la siguiente señal y avanzar el contador
        index = min(self.signal_index, len(signals) - 1)
        signal = signals[index].copy()
        signal["timestamp"] = data.index[-1] if not data.empty else pd.Timestamp.now()
        
        # Incrementar el índice para la próxima llamada
        self.signal_index += 1
        
        return signal


@pytest.fixture
def sample_ohlcv_data():
    """Generar datos OHLCV de ejemplo para pruebas de backtesting."""
    # Crear un DataFrame con pocas filas para que las pruebas sean rápidas
    np.random.seed(42)  # Para reproducibilidad
    
    dates = pd.date_range("2025-01-01", periods=20, freq="1h")
    
    # Crear un camino aleatorio para los precios
    close_prices = np.random.normal(0, 1, 20).cumsum() + 40000
    
    # Calcular otros campos OHLCV basados en los precios de cierre
    df = pd.DataFrame({
        "open": close_prices[:-1].tolist() + [close_prices[-1]],
        "high": close_prices + np.random.uniform(50, 200, 20),
        "low": close_prices - np.random.uniform(50, 200, 20),
        "close": close_prices,
        "volume": np.random.lognormal(10, 1, 20)
    }, index=dates)
    
    return df


@pytest.fixture
def backtest_engine():
    """Crear una instancia de motor de backtesting para pruebas."""
    engine = BacktestEngine(
        name="test_engine",
        initial_capital=10000,
        commission=0.001,  # Usamos commission en lugar de fee_rate
        slippage=0.0005
    )
    # Establecer propiedades para tests
    engine.use_stop_loss = False
    engine.use_trailing_stop = False
    return engine


@pytest.mark.asyncio
async def test_backtest_engine_initialization(backtest_engine):
    """Probar la inicialización del motor de backtesting."""
    assert backtest_engine.initial_capital == 10000
    assert backtest_engine.commission == 0.001  # Se usa commission en lugar de fee_rate
    assert backtest_engine.current_capital == 10000  # Se usa current_capital en lugar de current_balance


@pytest.mark.asyncio
async def test_backtest_simple_run(backtest_engine, sample_ohlcv_data):
    """Probar la ejecución básica de un backtest."""
    strategy = TestStrategy()
    
    # Ejecutar el backtest
    symbol = "BTC/USDT"
    
    # Usar timeout para evitar bloqueos
    try:
        results, stats = await asyncio.wait_for(
            backtest_engine.run_backtest(
                strategy=strategy,
                data={symbol: sample_ohlcv_data},
                symbol=symbol,
                timeframe="1h"
            ),
            timeout=5  # 5 segundos máximo
        )
        
        # Verificar resultados básicos
        assert "trades" in results
        assert "equity_curve" in results
        assert "total_trades" in stats
        assert "win_rate" in stats
        
    except asyncio.TimeoutError:
        pytest.fail("Timeout en la ejecución del backtest")


@pytest.mark.asyncio
async def test_backtest_position_management(backtest_engine, sample_ohlcv_data):
    """Probar la gestión de posiciones durante el backtesting."""
    # Estrategia con señales predefinidas
    strategy = TestStrategy()
    
    # Ejecutar el backtest con timeout
    symbol = "BTC/USDT"
    
    try:
        results, stats = await asyncio.wait_for(
            backtest_engine.run_backtest(
                strategy=strategy,
                data={symbol: sample_ohlcv_data},
                symbol=symbol,
                timeframe="1h"
            ),
            timeout=5  # 5 segundos máximo
        )
        
        # Verificar resultados básicos
        trades = results["trades"]
        assert len(trades) > 0  # Debe haber operaciones
        
        # Verificar que hay operaciones de compra y venta
        buy_trades = [t for t in trades if t["side"] == "buy"]
        sell_trades = [t for t in trades if t["side"] == "sell"]
        
        assert len(buy_trades) > 0
        assert len(sell_trades) > 0
        
    except asyncio.TimeoutError:
        pytest.fail("Timeout en la ejecución del backtest de posiciones")


@pytest.mark.asyncio
async def test_backtest_risk_management(backtest_engine, sample_ohlcv_data):
    """Probar la gestión de riesgos durante el backtesting."""
    # Estrategia con señales predefinidas para activar stop loss
    strategy = RiskStrategy()
    
    # Configurar parámetros de gestión de riesgos
    backtest_engine.risk_per_trade = 0.02  # 2% de riesgo por operación
    backtest_engine.use_stop_loss = True
    backtest_engine.use_trailing_stop = True
    
    # Configurar un stop loss calculator simulado
    mock_stop_loss = Mock()
    mock_stop_loss.calculate.return_value = {"price": 39500, "percentage": 0.0125}
    mock_stop_loss.calculate_trailing_stop.return_value = {"price": 39700, "activated": True}
    
    backtest_engine.stop_loss_calculator = mock_stop_loss
    
    # Ejecutar el backtest con timeout
    symbol = "BTC/USDT"
    
    try:
        results, stats = await asyncio.wait_for(
            backtest_engine.run_backtest(
                strategy=strategy,
                data={symbol: sample_ohlcv_data},
                symbol=symbol,
                timeframe="1h"
            ),
            timeout=5  # 5 segundos máximo
        )
        
        # Verificar que se llamó al calculador de stop loss
        assert mock_stop_loss.calculate.called
        
        # Comprobar resultados básicos
        assert "trades" in results
        assert "total_trades" in stats
        
    except asyncio.TimeoutError:
        pytest.fail("Timeout en la ejecución del backtest de riesgos")