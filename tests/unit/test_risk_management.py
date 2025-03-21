"""
Pruebas unitarias para el módulo de gestión de riesgos.

Este módulo prueba los componentes de gestión de riesgos, incluyendo
el cálculo de tamaño de posición, stop-loss y gestión general de riesgos.
"""

import pytest
from unittest.mock import Mock, AsyncMock
import logging
import sys
import os
import asyncio

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_risk_management")

# Asegurar que podemos importar los módulos de Genesis
sys.path.insert(0, os.getcwd())

# Importar componentes del sistema
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator
from genesis.risk.risk_manager import RiskManager
from genesis.core.event_bus import EventBus

# Fixture para EventBus mockeado
@pytest.fixture
def mock_event_bus():
    """Event bus simulado para pruebas."""
    bus = Mock(spec=EventBus)
    bus.emit = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus

# Fixture para RiskManager con EventBus mockeado
@pytest.fixture
def risk_manager(mock_event_bus):
    """Instancia de gestor de riesgos para pruebas."""
    manager = RiskManager(event_bus=mock_event_bus)
    return manager

# Pruebas para RiskManager
@pytest.mark.asyncio
async def test_risk_manager_start_stop(risk_manager):
    """Probar inicio y parada del gestor de riesgos."""
    # Iniciar el gestor
    await risk_manager.start()
    risk_manager._event_bus.subscribe.assert_called_with(risk_manager)
    
    # Detener el gestor
    await risk_manager.stop()
    # No hay manera directa de verificar el unsuscribe con este mock,
    # pero podemos verificar que el estado del gestor ha cambiado
    assert not risk_manager._running

@pytest.mark.asyncio
async def test_risk_manager_handle_signal(risk_manager):
    """Probar el manejo de señales por el gestor de riesgos."""
    # Simular una señal de compra
    await risk_manager.handle_event("signal.generated", {
        "signal": "BUY",
        "symbol": "BTC/USDT",
        "strength": 0.8,
        "price": 40000
    }, "signal_generator")
    
    # Verificar que se emitió un evento de validación de señal
    risk_manager._event_bus.emit.assert_called_with(
        "signal.validated",
        {
            "signal": "BUY",
            "symbol": "BTC/USDT",
            "approved": True,  # Asumimos que la señal es aprobada
            "risk_metrics": {"risk_score": risk_manager._calculate_risk_score("BTC/USDT")},
            "position_size": risk_manager._position_sizer.calculate_position_size(40000, "BTC/USDT")
        }
    )

@pytest.mark.asyncio
async def test_risk_manager_handle_trade_opened(risk_manager):
    """Probar el manejo de eventos de operación abierta."""
    # Configurar el gestor para la prueba
    risk_manager._stop_loss_calculator.calculate_stop_loss = Mock(return_value=38000)
    
    # Simular un evento de operación abierta
    await risk_manager.handle_event("trade.opened", {
        "symbol": "BTC/USDT",
        "side": "buy",
        "amount": 0.1,
        "price": 40000,
        "order_id": "123456",
        "exchange": "binance"
    }, "exchange_manager")
    
    # Verificar que se emitió un evento de establecimiento de stop loss
    risk_manager._event_bus.emit.assert_called_with(
        "trade.stop_loss_set",
        {
            "symbol": "BTC/USDT",
            "price": 38000,
            "trade_id": "123456",
            "exchange": "binance"
        }
    )

@pytest.mark.asyncio
async def test_risk_manager_handle_trade_closed(risk_manager):
    """Probar el manejo de eventos de operación cerrada."""
    # Simular un evento de operación cerrada
    await risk_manager.handle_event("trade.closed", {
        "symbol": "BTC/USDT",
        "side": "buy",
        "amount": 0.1,
        "entry_price": 40000,
        "exit_price": 42000,
        "profit": 200,
        "profit_percentage": 5,
        "order_id": "123456",
        "exchange": "binance"
    }, "exchange_manager")
    
    # Verificar que se emitió un evento de actualización de métricas de riesgo
    risk_manager._event_bus.emit.assert_called_with(
        "risk.metrics_updated",
        {
            "symbol": "BTC/USDT",
            "risk_score": risk_manager._calculate_risk_score("BTC/USDT"),
            "updated_metrics": {
                "total_trades": risk_manager._risk_metrics["BTC/USDT"]["total_trades"],
                "winning_trades": risk_manager._risk_metrics["BTC/USDT"]["winning_trades"],
                "profit": 200,
                "profit_percentage": 5
            }
        }
    )

# Pruebas para PositionSizer
def test_position_sizer_calculate():
    """Probar el cálculo de tamaño de posición."""
    sizer = PositionSizer()
    
    # Configurar el tamaño de la posición
    sizer.set_risk_percentage(2)  # 2% de riesgo por operación
    sizer.set_account_balance(10000)  # $10,000 de balance
    
    # Calcular tamaño de posición para un trade con stop loss a 5% de distancia
    position_size = sizer.calculate_position_size(
        entry_price=50000,
        symbol="BTC/USDT",
        stop_loss_percentage=5
    )
    
    # Verificar que el tamaño es correcto: (10000 * 0.02) / 0.05 = 4000
    expected_size = 4000  # Dólares
    assert position_size == expected_size
    
    # También podríamos verificar el cálculo de unidades
    units = sizer.calculate_units(position_size, 50000)
    expected_units = 0.08  # 4000 / 50000 = 0.08 BTC
    assert units == expected_units

# Pruebas para StopLossCalculator
@pytest.mark.asyncio
async def test_stop_loss_calculator():
    """Probar el cálculo de stop-loss."""
    calculator = StopLossCalculator()
    
    # Configurar el calculador
    calculator.set_default_multiplier(1.5)
    
    # Calcular stop-loss basado en ATR (Average True Range)
    entry_price = 50000
    atr = 1000
    
    stop_loss = calculator.calculate_stop_loss(
        entry_price=entry_price,
        atr=atr,
        side="buy"  # Para posiciones largas, el stop está por debajo
    )
    
    # Verificar que el stop-loss está correctamente calculado: 50000 - (1000 * 1.5) = 48500
    expected_stop_loss = 48500
    assert stop_loss == expected_stop_loss
    
    # Para posiciones cortas, el stop está por encima
    stop_loss_short = calculator.calculate_stop_loss(
        entry_price=entry_price,
        atr=atr,
        side="sell"
    )
    
    # Verificar que el stop-loss está correctamente calculado: 50000 + (1000 * 1.5) = 51500
    expected_stop_loss_short = 51500
    assert stop_loss_short == expected_stop_loss_short

def test_trailing_stop_loss():
    """Probar el cálculo de stop-loss móvil."""
    calculator = StopLossCalculator()
    
    # Configurar el calculador
    calculator.set_trailing_percentage(1)  # 1% de trailing stop
    
    # Calcular trailing stop para una posición larga
    entry_price = 50000
    current_price = 55000  # Precio ha subido un 10%
    
    trailing_stop = calculator.calculate_trailing_stop(
        entry_price=entry_price,
        current_price=current_price,
        side="buy"
    )
    
    # Verificar que el trailing stop está correctamente calculado: 55000 * 0.99 = 54450
    expected_trailing_stop = 54450
    assert trailing_stop == expected_trailing_stop
    
    # Para una posición corta
    trailing_stop_short = calculator.calculate_trailing_stop(
        entry_price=entry_price,
        current_price=45000,  # Precio ha bajado un 10%
        side="sell"
    )
    
    # Verificar que el trailing stop está correctamente calculado: 45000 * 1.01 = 45450
    expected_trailing_stop_short = 45450
    assert trailing_stop_short == expected_trailing_stop_short