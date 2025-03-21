"""
Tests básicos para las funcionalidades de gestión de riesgos.

Este módulo contiene tests simples y directos para verificar 
las funcionalidades básicas de los componentes de gestión de riesgos.
"""

import unittest
import pytest
import numpy as np

# Importar componentes de gestión de riesgos
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator as StopLoss  # Renombrar para facilitar la transición
from genesis.risk.risk_manager import RiskManager


class TestStopLossBasic(unittest.TestCase):
    """Tests básicos para el cálculo de stop-loss."""
    
    def test_stop_loss_percentage_long(self):
        """Verifica que el stop-loss basado en porcentaje se calcula correctamente para posición larga."""
        # Configurar
        stop_loss = StopLoss(risk_percentage=0.05)
        entry_price = 100.0
        expected_stop_loss_price = 95.0  # 5% debajo del precio de entrada
        
        # Ejecutar
        stop_price = stop_loss.calculate_percentage_stop_loss(entry_price, is_long=True)
        
        # Verificar
        self.assertEqual(stop_price, expected_stop_loss_price, 
                         f"Stop-loss incorrecto: esperado {expected_stop_loss_price}, obtenido {stop_price}")
    
    def test_stop_loss_percentage_short(self):
        """Verifica que el stop-loss basado en porcentaje se calcula correctamente para posición corta."""
        # Configurar
        stop_loss = StopLoss(risk_percentage=0.05)
        entry_price = 100.0
        expected_stop_loss_price = 105.0  # 5% por encima del precio de entrada
        
        # Ejecutar
        stop_price = stop_loss.calculate_percentage_stop_loss(entry_price, is_long=False)
        
        # Verificar
        self.assertEqual(stop_price, expected_stop_loss_price, 
                         f"Stop-loss incorrecto: esperado {expected_stop_loss_price}, obtenido {stop_price}")
    
    def test_atr_stop_loss_long(self):
        """Verifica que el stop-loss basado en ATR se calcula correctamente para posición larga."""
        # Configurar
        stop_loss = StopLoss()
        entry_price = 100.0
        atr = 2.0
        atr_multiplier = 2.0
        expected_stop_loss_price = 100.0 - (atr * atr_multiplier)  # 2 * ATR debajo del precio de entrada
        
        # Ejecutar
        stop_price = stop_loss.calculate_atr_stop_loss(entry_price, atr, atr_multiplier, is_long=True)
        
        # Verificar
        self.assertEqual(stop_price, expected_stop_loss_price, 
                         f"Stop-loss ATR incorrecto: esperado {expected_stop_loss_price}, obtenido {stop_price}")


class TestPositionSizerBasic(unittest.TestCase):
    """Tests básicos para el cálculo de tamaño de posición."""
    
    def test_position_size_percentage_risk(self):
        """Verifica el cálculo del tamaño de posición basado en porcentaje de riesgo."""
        # Configurar
        position_sizer = PositionSizer(risk_per_trade=0.02)  # 2% de riesgo por operación
        account_balance = 10000.0
        entry_price = 100.0
        stop_loss_price = 95.0  # 5% de distancia al stop-loss
        
        # Riesgo en unidades de cuenta = 10000 * 0.02 = 200
        # Riesgo por unidad = 100 - 95 = 5
        # Tamaño de posición = 200 / 5 = 40 unidades
        expected_position_size = 40.0
        
        # Ejecutar
        position_size = position_sizer.calculate_position_size(account_balance, entry_price, stop_loss_price)
        
        # Verificar
        self.assertEqual(position_size, expected_position_size, 
                         f"Tamaño de posición incorrecto: esperado {expected_position_size}, obtenido {position_size}")
    
    def test_position_size_fixed_risk(self):
        """Verifica el cálculo del tamaño de posición con una cantidad fija de riesgo."""
        # Configurar
        position_sizer = PositionSizer()
        fixed_risk_amount = 200.0  # Arriesgar $200 fijos
        entry_price = 100.0
        stop_loss_price = 90.0  # $10 de distancia al stop-loss
        
        # Tamaño de posición = 200 / 10 = 20 unidades
        expected_position_size = 20.0
        
        # Ejecutar
        position_size = position_sizer.calculate_position_size_fixed_risk(fixed_risk_amount, entry_price, stop_loss_price)
        
        # Verificar
        self.assertEqual(position_size, expected_position_size, 
                         f"Tamaño de posición (riesgo fijo) incorrecto: esperado {expected_position_size}, obtenido {position_size}")
    
    def test_position_size_max_capital(self):
        """Verifica que el tamaño de posición no exceda el máximo porcentaje de capital permitido."""
        # Configurar
        position_sizer = PositionSizer(risk_per_trade=0.02, max_position_size_percentage=0.1)  # Máx 10% del capital
        account_balance = 10000.0
        entry_price = 100.0
        stop_loss_price = 99.0  # Solo 1% de distancia al stop-loss, lo que daría un tamaño muy grande
        
        # Con 2% de riesgo y 1% de distancia, normalmente sería 10000 * 0.02 / 1 = 200 unidades
        # Pero el límite de 10% significa máximo 10000 * 0.1 / 100 = 10 unidades
        expected_position_size = 10.0
        
        # Ejecutar
        position_size = position_sizer.calculate_position_size(account_balance, entry_price, stop_loss_price)
        
        # Verificar
        self.assertEqual(position_size, expected_position_size, 
                         f"Limitación de tamaño incorrecta: esperado {expected_position_size}, obtenido {position_size}")


@pytest.fixture
def risk_components():
    """Fixture que proporciona los componentes básicos de gestión de riesgos."""
    position_sizer = PositionSizer(risk_per_trade=0.02)
    stop_loss = StopLoss(risk_percentage=0.05)
    risk_manager = RiskManager(
        max_risk_per_trade=0.02,
        max_total_risk=0.1,
        position_sizer=position_sizer,
        stop_loss_calculator=stop_loss
    )
    return {
        "position_sizer": position_sizer,
        "stop_loss": stop_loss,
        "risk_manager": risk_manager
    }


def test_risk_manager_evaluate_trade_basic(risk_components):
    """Test básico para verificar la evaluación de riesgo de una operación."""
    # Configurar
    risk_manager = risk_components["risk_manager"]
    account_balance = 10000.0
    entry_price = 100.0
    
    # Ejecutar
    success, position_info = risk_manager.evaluate_trade(account_balance, entry_price, is_long=True)
    
    # Verificar
    assert success is True
    assert position_info["entry_price"] == entry_price
    assert position_info["stop_loss_price"] == 95.0  # 5% por debajo
    assert position_info["position_size"] == 40.0  # 2% de riesgo con 5% de distancia


def test_risk_manager_exceeds_maximum_risk(risk_components):
    """Test para verificar que se rechaza una operación que excede el riesgo máximo."""
    # Configurar
    risk_manager = risk_components["risk_manager"]
    account_balance = 10000.0
    entry_price = 100.0
    
    # Modificar el gestor de riesgos para un límite más bajo
    risk_manager.max_risk_per_trade = 0.01  # Reducir a 1%
    
    # Ejecutar
    success, position_info = risk_manager.evaluate_trade(account_balance, entry_price, is_long=True)
    
    # Verificar
    assert success is False
    assert "reason" in position_info
    assert "excede" in position_info["reason"].lower()  # Debe mencionar que excede el riesgo


if __name__ == "__main__":
    unittest.main()