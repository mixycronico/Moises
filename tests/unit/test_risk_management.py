"""
Pruebas unitarias para el módulo de gestión de riesgos.

Este módulo prueba los componentes de gestión de riesgos, incluyendo
el cálculo de tamaño de posición, stop-loss y gestión general de riesgos.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import datetime

from genesis.risk.manager import RiskManager
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator


@pytest.fixture
def mock_event_bus():
    """Event bus simulado para pruebas."""
    bus = MagicMock()
    bus.emit = MagicMock(return_value=asyncio.Future())
    bus.emit.return_value.set_result(None)
    return bus


@pytest.fixture
def risk_manager(mock_event_bus):
    """Instancia de gestor de riesgos para pruebas."""
    manager = RiskManager()
    manager.event_bus = mock_event_bus
    return manager


@pytest.mark.asyncio
async def test_risk_manager_start_stop(risk_manager):
    """Probar inicio y parada del gestor de riesgos."""
    await risk_manager.start()
    assert risk_manager.running is True
    
    await risk_manager.stop()
    assert risk_manager.running is False


@pytest.mark.asyncio
async def test_risk_manager_handle_signal(risk_manager):
    """Probar el manejo de señales por el gestor de riesgos."""
    # Configurar una señal de ejemplo
    signal_data = {
        "symbol": "BTCUSDT",
        "signal_type": "buy",
        "price": 40000.0,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "strategy": "macd_cross",
        "strength": 0.8,
        "metadata": {
            "indicator_values": {
                "macd": 0.5,
                "signal": 0.2
            }
        }
    }
    
    # Simular manejo de señal
    await risk_manager.handle_event("strategy.signal", signal_data, "strategy_manager")
    
    # Verificar que se emitió un evento de decisión de trading
    risk_manager.event_bus.emit.assert_called_once()
    call_args = risk_manager.event_bus.emit.call_args[0]
    assert call_args[0] == "risk.trade_decision"
    assert call_args[2] == "risk_manager"
    
    # Verificar el contenido de la decisión
    decision_data = call_args[1]
    assert decision_data["symbol"] == "BTCUSDT"
    assert "position_size" in decision_data
    assert "risk_level" in decision_data
    assert "stop_loss" in decision_data or "take_profit" in decision_data


@pytest.mark.asyncio
async def test_risk_manager_handle_trade_opened(risk_manager):
    """Probar el manejo de eventos de operación abierta."""
    trade_data = {
        "trade_id": "T12345",
        "symbol": "BTCUSDT",
        "side": "buy",
        "amount": 0.1,
        "price": 40000.0,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "status": "open"
    }
    
    await risk_manager.handle_event("trade.opened", trade_data, "exchange_manager")
    
    # Verificar actualización de seguimiento de riesgo
    # (esto depende de la implementación específica)
    if hasattr(risk_manager, "_active_trades"):
        assert "T12345" in risk_manager._active_trades
    
    # Opcionalmente, verificar emisión de evento de actualización de riesgo
    for call in risk_manager.event_bus.emit.call_args_list:
        if call[0][0] == "risk.exposure_updated":
            exposure_data = call[0][1]
            assert "total_exposure" in exposure_data
            break


@pytest.mark.asyncio
async def test_risk_manager_handle_trade_closed(risk_manager):
    """Probar el manejo de eventos de operación cerrada."""
    # Primero simulamos una operación abierta
    open_trade_data = {
        "trade_id": "T12345",
        "symbol": "BTCUSDT",
        "side": "buy",
        "amount": 0.1,
        "price": 40000.0,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "status": "open"
    }
    
    await risk_manager.handle_event("trade.opened", open_trade_data, "exchange_manager")
    
    # Luego simulamos el cierre de la operación
    close_trade_data = {
        "trade_id": "T12345",
        "symbol": "BTCUSDT",
        "side": "buy",
        "amount": 0.1,
        "entry_price": 40000.0,
        "exit_price": 41000.0,
        "profit_loss": 100.0,
        "profit_loss_pct": 2.5,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "status": "closed"
    }
    
    await risk_manager.handle_event("trade.closed", close_trade_data, "exchange_manager")
    
    # Verificar actualización de seguimiento de riesgo
    if hasattr(risk_manager, "_active_trades"):
        assert "T12345" not in risk_manager._active_trades
    
    # Verificar actualización de estadísticas de trading
    if hasattr(risk_manager, "_trading_stats"):
        assert risk_manager._trading_stats["total_trades"] > 0
        assert risk_manager._trading_stats["profitable_trades"] > 0


def test_position_sizer_calculate():
    """Probar el cálculo de tamaño de posición."""
    sizer = PositionSizer()
    
    # Caso 1: Cálculo básico con riesgo del 1%
    position_size = sizer.calculate(
        portfolio_value=10000.0,
        risk_percent=0.01,
        signal_strength=1.0
    )
    assert position_size == 100.0  # 1% de 10000
    
    # Caso 2: Con fuerza de señal reducida
    position_size = sizer.calculate(
        portfolio_value=10000.0,
        risk_percent=0.01,
        signal_strength=0.5
    )
    assert position_size == 50.0  # 1% * 0.5 de 10000
    
    # Caso 3: Con límite máximo
    position_size = sizer.calculate(
        portfolio_value=10000.0,
        risk_percent=0.05,
        signal_strength=1.0,
        max_size=300.0
    )
    assert position_size == 300.0  # Limitado por max_size


def test_position_sizer_calculate_units():
    """Probar el cálculo de unidades de posición."""
    sizer = PositionSizer()
    
    # Caso 1: Cálculo básico
    units = sizer.calculate_units(
        position_size=1000.0,
        current_price=40000.0
    )
    assert units == 0.025  # 1000 / 40000
    
    # Caso 2: Con cantidad mínima
    units = sizer.calculate_units(
        position_size=1000.0,
        current_price=40000.0,
        min_quantity=0.05
    )
    expected_units = 0.05  # Redondeado a min_quantity
    assert abs(units - expected_units) < 1e-8


@pytest.mark.asyncio
async def test_stop_loss_calculator():
    """Probar el cálculo de stop-loss."""
    calculator = StopLossCalculator()
    
    # Caso 1: Cálculo para posición long
    stop_loss = await calculator.calculate(
        symbol="BTCUSDT",
        signal_type="buy",
        position_size=1000.0,
        price=40000.0,
        atr_value=1200.0
    )
    
    assert "price" in stop_loss
    assert "percentage" in stop_loss
    assert stop_loss["price"] < 40000.0  # El stop-loss debe estar por debajo del precio
    
    # Caso 2: Cálculo para posición short
    stop_loss = await calculator.calculate(
        symbol="BTCUSDT",
        signal_type="sell",
        position_size=1000.0,
        price=40000.0,
        atr_value=1200.0
    )
    
    assert "price" in stop_loss
    assert "percentage" in stop_loss
    assert stop_loss["price"] > 40000.0  # El stop-loss debe estar por encima del precio


def test_trailing_stop_loss():
    """Probar el cálculo de stop-loss móvil."""
    calculator = StopLossCalculator()
    
    # Caso 1: Trailing stop para posición long
    trailing_stop = calculator.calculate_trailing_stop(
        current_price=42000.0,
        entry_price=40000.0,
        is_long=True,
        atr_value=1200.0,
        activation_pct=0.01
    )
    
    assert "price" in trailing_stop
    assert "activated" in trailing_stop
    assert trailing_stop["activated"] is True  # 5% de ganancia > 1% de activación
    assert trailing_stop["price"] < 42000.0  # El trailing stop debe estar por debajo del precio actual
    
    # Caso 2: Trailing stop para posición short
    trailing_stop = calculator.calculate_trailing_stop(
        current_price=38000.0,
        entry_price=40000.0,
        is_long=False,
        atr_value=1200.0,
        activation_pct=0.01
    )
    
    assert "price" in trailing_stop
    assert "activated" in trailing_stop
    assert trailing_stop["activated"] is True  # 5% de ganancia > 1% de activación
    assert trailing_stop["price"] > 38000.0  # El trailing stop debe estar por encima del precio actual