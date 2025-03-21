"""
Tests básicos para el módulo de backtesting.

Este módulo contiene tests simples y directos para verificar 
las funcionalidades básicas del motor de backtesting sin depender
de datos complejos o interacciones con otras partes del sistema.
"""

import unittest
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Importar componentes de backtesting
from genesis.backtesting.engine import BacktestEngine
from genesis.strategies.base import Strategy
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLoss


class TestStrategy(Strategy):
    """Estrategia de prueba simple para backtesting."""
    
    def __init__(self, name="test_strategy"):
        """Inicializar la estrategia de prueba."""
        super().__init__(name=name)
        self.signal_sequence = ["BUY", "HOLD", "HOLD", "SELL", "HOLD", "HOLD"]
        self.call_count = 0
    
    async def generate_signal(self, symbol, data):
        """
        Genera señales predefinidas para testing.
        
        Retorna una secuencia fija de señales para poder crear un escenario
        de prueba controlado.
        """
        if self.call_count < len(self.signal_sequence):
            signal = self.signal_sequence[self.call_count]
            self.call_count += 1
            return {
                "type": signal,
                "symbol": symbol,
                "price": data["close"].iloc[-1],
                "timestamp": data.index[-1]
            }
        return {"type": "HOLD", "symbol": symbol, "price": data["close"].iloc[-1], "timestamp": data.index[-1]}


class TestBacktestBasic(unittest.TestCase):
    """Tests básicos para backtesting."""
    
    def setUp(self):
        """Configuración inicial para los tests."""
        # Crear datos OHLCV simples para pruebas
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(10)]
        
        self.simple_data = pd.DataFrame({
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0],
            "high": [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 106.0, 105.0, 104.0, 103.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0],
            "close": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 105.0, 104.0, 103.0, 102.0],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1400, 1300, 1200, 1100]
        }, index=dates)
        
        # Componentes para backtesting
        self.strategy = TestStrategy()
        self.position_sizer = PositionSizer(risk_per_trade=0.02)
        self.stop_loss = StopLoss(risk_percentage=0.05)
        
        # Motor de backtesting
        self.backtest_engine = BacktestEngine(
            initial_capital=10000.0,
            position_sizer=self.position_sizer,
            stop_loss_calculator=self.stop_loss
        )
    
    def test_backtest_initialization(self):
        """Verifica que el motor de backtesting se inicializa correctamente."""
        # Verificar valores iniciales
        self.assertEqual(self.backtest_engine.initial_capital, 10000.0, 
                         "Capital inicial incorrecto")
        self.assertEqual(self.backtest_engine.current_capital, 10000.0, 
                         "Capital actual inicial debe ser igual al capital inicial")
        self.assertEqual(len(self.backtest_engine.trades), 0, 
                         "No debe haber operaciones al inicio")
        self.assertEqual(len(self.backtest_engine.positions), 0, 
                         "No debe haber posiciones abiertas al inicio")
    
    async def test_backtest_simple_run(self):
        """Prueba una ejecución simple del backtesting con señales predefinidas."""
        # Configurar
        symbol = "BTCUSDT"
        
        # Ejecutar
        results = await self.backtest_engine.run_backtest(
            strategy=self.strategy,
            data={symbol: self.simple_data},
            timeframe="1d"
        )
        
        # Verificar resultados básicos
        self.assertIsNotNone(results, "Los resultados no deben ser None")
        self.assertIn("trades", results, "Los resultados deben incluir las operaciones")
        self.assertIn("equity_curve", results, "Los resultados deben incluir la curva de capital")
        self.assertIn("metrics", results, "Los resultados deben incluir métricas")
        
        # Verificar que se ejecutaron operaciones según las señales predefinidas
        trades = results["trades"]
        self.assertTrue(len(trades) >= 1, "Debe haber al menos una operación")
    
    async def test_backtest_capital_changes(self):
        """Verifica que el capital cambia correctamente después de las operaciones."""
        # Configurar
        symbol = "BTCUSDT"
        
        # Ejecutar
        results = await self.backtest_engine.run_backtest(
            strategy=self.strategy,
            data={symbol: self.simple_data},
            timeframe="1d"
        )
        
        # Verificar
        final_capital = results["equity_curve"][-1] if len(results["equity_curve"]) > 0 else 10000.0
        
        # El capital final debe ser diferente al inicial
        self.assertNotEqual(final_capital, 10000.0, 
                            "El capital debe cambiar después de ejecutar operaciones")
        
        # Verificar que la curva de capital tiene la longitud correcta
        expected_length = len(self.simple_data)
        self.assertEqual(len(results["equity_curve"]), expected_length, 
                         f"La curva de capital debe tener {expected_length} puntos")
    
    async def test_backtest_metrics_calculation(self):
        """Verifica que las métricas de rendimiento se calculan correctamente."""
        # Configurar
        symbol = "BTCUSDT"
        
        # Ejecutar
        results = await self.backtest_engine.run_backtest(
            strategy=self.strategy,
            data={symbol: self.simple_data},
            timeframe="1d"
        )
        
        # Verificar la presencia de métricas básicas
        metrics = results["metrics"]
        
        self.assertIn("total_return", metrics, "Debe incluir retorno total")
        self.assertIn("win_rate", metrics, "Debe incluir tasa de aciertos")
        self.assertIn("max_drawdown", metrics, "Debe incluir drawdown máximo")
        
        # Verificar que las métricas tienen valores válidos
        self.assertIsInstance(metrics["total_return"], float, "Retorno total debe ser un float")
        self.assertTrue(0 <= metrics["win_rate"] <= 1, "Tasa de aciertos debe estar entre 0 y 1")
        self.assertTrue(metrics["max_drawdown"] <= 0, "Drawdown máximo debe ser negativo o cero")


@pytest.fixture
def sample_ohlcv_data():
    """Genera datos OHLCV de ejemplo para pruebas de backtesting."""
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(10)]
    
    return pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0],
        "high": [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 106.0, 105.0, 104.0, 103.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0],
        "close": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 105.0, 104.0, 103.0, 102.0],
        "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1400, 1300, 1200, 1100]
    }, index=dates)


@pytest.fixture
def backtest_engine():
    """Crea una instancia de BacktestEngine para pruebas."""
    position_sizer = PositionSizer(risk_per_trade=0.02)
    stop_loss = StopLoss(risk_percentage=0.05)
    
    return BacktestEngine(
        initial_capital=10000.0,
        position_sizer=position_sizer,
        stop_loss_calculator=stop_loss
    )


@pytest.mark.asyncio
async def test_backtest_simple_execution(backtest_engine, sample_ohlcv_data):
    """Prueba básica de ejecución del backtest con estrategia simple."""
    # Configurar
    strategy = TestStrategy()
    symbol = "BTCUSDT"
    
    # Ejecutar
    results = await backtest_engine.run_backtest(
        strategy=strategy,
        data={symbol: sample_ohlcv_data},
        timeframe="1d"
    )
    
    # Verificar
    assert results is not None, "Los resultados no deben ser None"
    assert "trades" in results, "Los resultados deben incluir operaciones"
    assert len(results["trades"]) > 0, "Debe haber al menos una operación"
    assert results["metrics"]["total_return"] != 0, "Debe haber algún retorno (positivo o negativo)"


if __name__ == "__main__":
    unittest.main()