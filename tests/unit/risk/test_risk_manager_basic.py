"""
Tests básicos para RiskManager.

Este módulo contiene tests unitarios para la funcionalidad básica
del gestor de riesgos.
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


class TestRiskManagerBasic(unittest.TestCase):
    """Tests básicos para RiskManager."""
    
    def setUp(self):
        """Configurar el entorno de prueba."""
        # Crear un mock para el EventBus
        self.event_bus_mock = Mock()
        self.event_bus_mock.emit = AsyncMock()
        self.event_bus_mock.subscribe = AsyncMock()
        
        # Crear instancia del RiskManager con el mock
        self.risk_manager = RiskManager(event_bus=self.event_bus_mock)
    
    def test_calculate_risk_score(self):
        """Verificar cálculo de puntuación de riesgo."""
        # Caso: Símbolo sin historial
        score = self.risk_manager._calculate_risk_score("ETH/USDT")
        self.assertEqual(score, 50.0)  # Riesgo neutral
        
        # Caso: Símbolo con buen historial (100% de éxito)
        self.risk_manager._risk_metrics["BTC/USDT"] = {
            "total_trades": 10,
            "winning_trades": 10,
            "losing_trades": 0,
            "profit": 1000,
            "drawdown": 0,
            "max_drawdown": 0
        }
        score = self.risk_manager._calculate_risk_score("BTC/USDT")
        self.assertEqual(score, 100.0)
        
        # Caso: Símbolo con mal historial (0% de éxito)
        self.risk_manager._risk_metrics["XRP/USDT"] = {
            "total_trades": 10,
            "winning_trades": 0,
            "losing_trades": 10,
            "profit": -1000,
            "drawdown": 10,
            "max_drawdown": 10
        }
        score = self.risk_manager._calculate_risk_score("XRP/USDT")
        self.assertEqual(score, 0.0)


# Configurar tests para pytest con asyncio
@pytest.mark.asyncio
async def test_risk_manager_event_handling():
    """Test que verifica el manejo de eventos por RiskManager."""
    # Crear mocks
    event_bus_mock = Mock()
    event_bus_mock.emit = AsyncMock()
    event_bus_mock.subscribe = AsyncMock()
    
    # Crear instancia del RiskManager
    risk_manager = RiskManager(event_bus=event_bus_mock)
    
    # Iniciar el risk manager
    await risk_manager.start()
    
    # Verificar que se haya suscrito al bus de eventos
    event_bus_mock.subscribe.assert_called_once_with(risk_manager)
    
    # Simular recepción de evento de señal
    await risk_manager.handle_event(
        "signal.generated",
        {
            "symbol": "BTC/USDT",
            "signal": "buy",
            "price": 50000.0
        },
        "Strategy"
    )
    
    # Verificar que se haya emitido un evento de validación
    assert event_bus_mock.emit.called
    call_args = event_bus_mock.emit.call_args_list[0]
    
    # Verificar el tipo de evento
    assert call_args[0][0] == "signal.validated"


if __name__ == "__main__":
    unittest.main()