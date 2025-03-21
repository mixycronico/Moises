"""
Tests de integración para el módulo de gestión de riesgos.

Este módulo contiene tests que verifican la interacción entre
los diferentes componentes del módulo de gestión de riesgos.
"""

import unittest
import pytest
import logging
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from genesis.risk.risk_manager import RiskManager
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator

# Configurar logging para los tests
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def risk_components():
    """Fixture que proporciona componentes configurados para testing."""
    # Inicializar componentes
    position_sizer = PositionSizer(default_risk_percentage=2.0)
    position_sizer.set_account_balance(10000.0)
    
    stop_loss_calculator = StopLossCalculator(default_multiplier=2.0)
    
    event_bus_mock = Mock()
    event_bus_mock.emit = AsyncMock()
    event_bus_mock.subscribe = AsyncMock()
    
    risk_manager = RiskManager(event_bus=event_bus_mock)
    
    # Reemplazar componentes del risk_manager con nuestras versiones
    risk_manager._position_sizer = position_sizer
    risk_manager._stop_loss_calculator = stop_loss_calculator
    
    return {
        "position_sizer": position_sizer,
        "stop_loss_calculator": stop_loss_calculator,
        "risk_manager": risk_manager,
        "event_bus": event_bus_mock
    }


@pytest.mark.asyncio
async def test_risk_manager_with_position_sizer(risk_components):
    """
    Verificar la integración entre RiskManager y PositionSizer.
    
    Este test comprueba que el RiskManager usa correctamente el PositionSizer
    para calcular el tamaño de posición.
    """
    risk_manager = risk_components["risk_manager"]
    event_bus = risk_components["event_bus"]
    
    # Iniciar el risk manager
    await risk_manager.start()
    
    # Crear datos de prueba
    signal_data = {
        "symbol": "BTC/USDT",
        "signal": "buy",
        "price": 50000.0
    }
    
    # Procesar señal
    await risk_manager._handle_signal(signal_data)
    
    # Verificar que se emitió un evento de validación
    assert event_bus.emit.called
    call_args = event_bus.emit.call_args
    
    # Verificar el tipo de evento
    assert call_args[0][0] == "signal.validated"
    
    # Verificar que se calculó el tamaño de posición
    event_data = call_args[0][1]
    assert "position_size" in event_data
    assert event_data["position_size"] > 0


@pytest.mark.asyncio
async def test_risk_manager_with_stop_loss_calculator(risk_components):
    """
    Verificar la integración entre RiskManager y StopLossCalculator.
    
    Este test comprueba que el RiskManager usa correctamente el StopLossCalculator
    para calcular el nivel de stop-loss.
    """
    risk_manager = risk_components["risk_manager"]
    event_bus = risk_components["event_bus"]
    
    # Iniciar el risk manager
    await risk_manager.start()
    
    # Crear datos de prueba
    trade_data = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": 50000.0,
        "order_id": "test123",
        "exchange": "binance"
    }
    
    # Manejar apertura de operación (debería calcular stop-loss)
    await risk_manager._handle_trade_opened(trade_data)
    
    # Verificar que se emitió un evento de stop loss
    assert event_bus.emit.called
    call_args = event_bus.emit.call_args
    
    # Verificar el tipo de evento
    assert call_args[0][0] == "trade.stop_loss_set"
    
    # Verificar que el precio de stop-loss se calculó y está dentro del rango esperado
    event_data = call_args[0][1]
    assert "price" in event_data
    # Para operación de compra, el stop debe estar por debajo del precio de entrada
    assert event_data["price"] < trade_data["price"]


@pytest.mark.asyncio
async def test_full_risk_workflow(risk_components):
    """
    Verificar el flujo completo de gestión de riesgos.
    
    Este test simula el ciclo completo de:
    1. Recepción de señal
    2. Validación y cálculo de tamaño
    3. Apertura de operación y cálculo de stop-loss
    4. Cierre de operación y actualización de métricas
    """
    risk_manager = risk_components["risk_manager"]
    event_bus = risk_components["event_bus"]
    
    # Iniciar el risk manager
    await risk_manager.start()
    
    # 1. Simular recepción de señal
    signal_data = {
        "symbol": "BTC/USDT",
        "signal": "buy",
        "price": 50000.0
    }
    await risk_manager._handle_signal(signal_data)
    
    # Verificar primera emisión (validación)
    assert event_bus.emit.call_count >= 1
    first_call = event_bus.emit.call_args_list[0]
    assert first_call[0][0] == "signal.validated"
    
    # Reiniciar contador de llamadas del mock
    event_bus.emit.reset_mock()
    
    # 2. Simular apertura de operación
    trade_data = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": 50000.0,
        "order_id": "test123",
        "exchange": "binance"
    }
    await risk_manager._handle_trade_opened(trade_data)
    
    # Verificar segunda emisión (stop-loss)
    assert event_bus.emit.call_count >= 1
    second_call = event_bus.emit.call_args_list[0]
    assert second_call[0][0] == "trade.stop_loss_set"
    
    # Reiniciar contador de llamadas del mock
    event_bus.emit.reset_mock()
    
    # 3. Simular cierre de operación
    close_data = {
        "symbol": "BTC/USDT",
        "profit": 1000,
        "profit_percentage": 2.0
    }
    await risk_manager._handle_trade_closed(close_data)
    
    # Verificar tercera emisión (actualización de métricas)
    assert event_bus.emit.call_count >= 1
    third_call = event_bus.emit.call_args_list[0]
    assert third_call[0][0] == "risk.metrics_updated"
    
    # Verificar que las métricas se actualizaron
    metrics = risk_manager._risk_metrics["BTC/USDT"]
    assert metrics["total_trades"] == 1
    assert metrics["winning_trades"] == 1
    assert metrics["profit"] == 1000


if __name__ == "__main__":
    pytest.main()