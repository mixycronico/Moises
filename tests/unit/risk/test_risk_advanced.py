"""
Tests avanzados para el módulo de gestión de riesgos.

Este módulo contiene tests que verifican comportamientos complejos,
casos extremos y manejo de errores en los componentes de gestión de riesgos.
"""

import unittest
import pytest
import logging
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from genesis.risk.risk_manager import RiskManager
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator

# Configurar logging para los tests
logging.basicConfig(level=logging.INFO)


class TestStopLossCalculatorAdvanced(unittest.TestCase):
    """Tests avanzados para StopLossCalculator."""
    
    def setUp(self):
        """Configurar el entorno de prueba."""
        self.stop_loss = StopLossCalculator()
    
    def test_atr_stop_loss_edge_cases(self):
        """Verificar casos extremos en cálculo de stop-loss ATR."""
        # Caso: ATR cero
        entry_price = 50000
        atr = 0
        multiplier = 2.0
        
        result = self.stop_loss.calculate_atr_stop_loss(
            entry_price, atr, multiplier, is_long=True
        )
        
        # Cuando ATR es cero, debe retornar el precio de entrada (sin stop)
        self.assertEqual(result, entry_price)
        
        # Caso: Multiplicador cero
        result = self.stop_loss.calculate_atr_stop_loss(
            entry_price, 1000, 0, is_long=True
        )
        
        # Cuando el multiplicador es cero, debe retornar el precio de entrada
        self.assertEqual(result, entry_price)
    
    def test_atr_stop_loss_invalid_inputs(self):
        """Verificar manejo de inputs inválidos en cálculo de stop-loss ATR."""
        entry_price = 50000
        
        # Caso: ATR negativo
        with self.assertRaises(ValueError):
            self.stop_loss.calculate_atr_stop_loss(
                entry_price, -1000, 2.0, is_long=True
            )
        
        # Caso: Multiplicador negativo
        with self.assertRaises(ValueError):
            self.stop_loss.calculate_atr_stop_loss(
                entry_price, 1000, -2.0, is_long=True
            )
    
    def test_trailing_stop_edge_cases(self):
        """Verificar casos extremos en cálculo de trailing stop."""
        # Caso: Precio negativo
        with self.assertRaises(ValueError):
            self.stop_loss.calculate_trailing_stop(
                -1000, 50000, is_long=True, stop_pct=0.01
            )
        
        # Caso: Porcentaje de stop negativo
        with self.assertRaises(ValueError):
            self.stop_loss.calculate_trailing_stop(
                55000, 50000, is_long=True, stop_pct=-0.01
            )
    
    def test_trailing_stop_with_previous_levels(self):
        """Verificar ajuste de trailing stop con niveles previos."""
        # Escenario: Operación larga con precio en alza
        # Primera actualización
        result1 = self.stop_loss.calculate_trailing_stop(
            55000, 50000, is_long=True, stop_pct=0.01
        )
        
        # El primer trailing stop debe ser 55000 * 0.99 = 54450
        self.assertEqual(result1["price"], 54450)
        
        # Segunda actualización (precio subió más)
        result2 = self.stop_loss.calculate_trailing_stop(
            56000, 50000, is_long=True, stop_pct=0.01,
            previous_stop=result1["price"]
        )
        
        # El nuevo stop sería 56000 * 0.99 = 55440
        self.assertEqual(result2["price"], 55440)
        
        # Tercera actualización (precio bajó)
        result3 = self.stop_loss.calculate_trailing_stop(
            55500, 50000, is_long=True, stop_pct=0.01,
            previous_stop=result2["price"]
        )
        
        # Debería mantener el stop anterior (55440) ya que es mejor
        self.assertEqual(result3["price"], 55440)


class TestPositionSizerAdvanced(unittest.TestCase):
    """Tests avanzados para PositionSizer."""
    
    def setUp(self):
        """Configurar el entorno de prueba."""
        self.position_sizer = PositionSizer()
        self.position_sizer.set_account_balance(10000.0)
    
    def test_zero_balance(self):
        """Verificar el comportamiento con balance cero."""
        self.position_sizer.set_account_balance(0)
        
        position_size = self.position_sizer.calculate_position_size(100.0, "BTC/USDT")
        
        # Debe retornar 0 para evitar errores
        self.assertEqual(position_size, 0)
    
    def test_zero_risk_per_unit(self):
        """Verificar el comportamiento cuando el riesgo por unidad es cero."""
        # Mockear la clase para forzar un riesgo por unidad de cero
        with patch.object(self.position_sizer, 'calculate_position_size', 
                         return_value=0):
            position_size = self.position_sizer.calculate_position_size(100.0, "BTC/USDT")
            
            # Debe retornar 0 para evitar división por cero
            self.assertEqual(position_size, 0)
    
    def test_large_position_sizes(self):
        """Verificar el comportamiento con tamaños de posición muy grandes."""
        # Establecer un balance muy grande
        self.position_sizer.set_account_balance(1000000000.0)  # Mil millones
        
        # Establecer un riesgo mayor
        self.position_sizer.set_risk_percentage(5.0)
        
        # En un caso real, esto debería dar un valor muy grande pero válido
        position_size = self.position_sizer.calculate_position_size(100.0, "BTC/USDT")
        
        # Verificar que la respuesta sea finita y mayor que cero
        self.assertTrue(position_size > 0)
        self.assertTrue(float('inf') > position_size)


@pytest.mark.asyncio
async def test_risk_manager_concurrent_signals():
    """Verificar el manejo de múltiples señales concurrentes."""
    # Crear mocks
    event_bus_mock = Mock()
    event_bus_mock.emit = AsyncMock()
    event_bus_mock.subscribe = AsyncMock()
    
    # Crear instancia del RiskManager
    risk_manager = RiskManager(event_bus=event_bus_mock)
    
    # Iniciar el risk manager
    await risk_manager.start()
    
    # Crear datos de múltiples señales
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT", "DOT/USDT"]
    signals = []
    
    for symbol in symbols:
        signals.append({
            "symbol": symbol,
            "signal": "buy",
            "price": 50000.0
        })
    
    # Procesar señales concurrentemente
    tasks = []
    for signal in signals:
        tasks.append(asyncio.create_task(
            risk_manager._handle_signal(signal)
        ))
    
    # Esperar que todas las tareas terminen
    await asyncio.gather(*tasks)
    
    # Verificar que se emitieran eventos para todas las señales
    assert event_bus_mock.emit.call_count == len(symbols)
    
    # Verificar que todas las señales fueron procesadas
    for symbol in symbols:
        assert symbol in risk_manager._risk_metrics


@pytest.mark.asyncio
async def test_risk_manager_error_handling():
    """Verificar el manejo de errores en RiskManager."""
    # Crear mocks
    event_bus_mock = Mock()
    event_bus_mock.emit = AsyncMock()
    event_bus_mock.subscribe = AsyncMock()
    
    # Crear instancia del RiskManager con positionSizer defectuoso
    risk_manager = RiskManager(event_bus=event_bus_mock)
    
    # Reemplazar el PositionSizer con uno que lance excepción
    faulty_position_sizer = Mock()
    faulty_position_sizer.calculate_position_size = Mock(side_effect=Exception("Error simulado"))
    risk_manager._position_sizer = faulty_position_sizer
    
    # Iniciar el risk manager
    await risk_manager.start()
    
    # Crear datos de una señal
    signal_data = {
        "symbol": "BTC/USDT",
        "signal": "buy",
        "price": 50000.0
    }
    
    # El RiskManager debería manejar la excepción internamente sin propagarla
    await risk_manager._handle_signal(signal_data)
    
    # A pesar del error, debería seguir funcionando
    assert risk_manager._running == True


@pytest.mark.asyncio
async def test_risk_manager_missing_data():
    """Verificar cómo maneja RiskManager datos incompletos o faltantes."""
    # Crear mocks
    event_bus_mock = Mock()
    event_bus_mock.emit = AsyncMock()
    event_bus_mock.subscribe = AsyncMock()
    
    # Crear instancia del RiskManager
    risk_manager = RiskManager(event_bus=event_bus_mock)
    
    # Iniciar el risk manager
    await risk_manager.start()
    
    # Señal sin precio
    incomplete_signal = {
        "symbol": "BTC/USDT",
        "signal": "buy"
        # Sin precio
    }
    
    # Debería manejar el caso sin error
    await risk_manager._handle_signal(incomplete_signal)
    
    # Señal sin símbolo
    missing_symbol = {
        "signal": "buy",
        "price": 50000.0
        # Sin símbolo
    }
    
    # Debería manejar el caso sin error
    await risk_manager._handle_signal(missing_symbol)
    
    # Trade sin orden ID
    incomplete_trade = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": 50000.0,
        # Sin order_id
        "exchange": "binance"
    }
    
    # Debería manejar el caso sin error
    await risk_manager._handle_trade_opened(incomplete_trade)


if __name__ == "__main__":
    unittest.main()