"""
Pruebas unitarias para cálculos de riesgo en trading.

Este módulo prueba los cálculos fundamentales de gestión de riesgos,
incluyendo cálculo de stop-loss, trailing stop y otros parámetros de riesgo.
"""

import pytest
from unittest.mock import Mock

from genesis.risk.manager import RiskManager
from genesis.risk.stop_loss import StopLossCalculator


# Fixture para RiskManagement
@pytest.fixture
def stop_loss_calculator():
    """Fixture que proporciona una instancia de StopLossCalculator."""
    return StopLossCalculator()


@pytest.fixture
def risk_manager():
    """Fixture que proporciona una instancia de RiskManager."""
    return RiskManager()


# Pruebas para calculate_stop_loss
def test_stop_loss_calculation_basic(stop_loss_calculator):
    """Prueba el cálculo básico de stop-loss basado en ATR."""
    atr = 50  # ATR de 50 puntos
    entry_price = 1000  # Precio de entrada
    stop_loss_multiplier = 1.5

    # Resultado esperado: entry_price - (atr * multiplier)
    expected_stop_loss = entry_price - (atr * stop_loss_multiplier)  # 1000 - (50 * 1.5) = 925
    calculated_stop_loss = stop_loss_calculator.calculate_atr_stop_loss(
        entry_price, atr, stop_loss_multiplier, is_long=True
    )
    
    assert calculated_stop_loss == expected_stop_loss


def test_stop_loss_calculation_zero_multiplier(stop_loss_calculator):
    """Prueba el cálculo de stop-loss con un multiplicador de 0."""
    atr = 50
    entry_price = 1000
    stop_loss_multiplier = 0

    # Resultado esperado: entry_price sin cambios
    expected_stop_loss = entry_price  # 1000 - (50 * 0) = 1000
    calculated_stop_loss = stop_loss_calculator.calculate_atr_stop_loss(
        entry_price, atr, stop_loss_multiplier, is_long=True
    )
    
    assert calculated_stop_loss == expected_stop_loss


def test_stop_loss_calculation_negative_values(stop_loss_calculator):
    """Prueba el manejo de valores negativos en ATR o multiplicador."""
    entry_price = 1000
    
    with pytest.raises(ValueError, match="ATR and multiplier must be non-negative"):
        stop_loss_calculator.calculate_atr_stop_loss(
            entry_price, atr=-50, multiplier=1.5, is_long=True
        )
    
    with pytest.raises(ValueError, match="ATR and multiplier must be non-negative"):
        stop_loss_calculator.calculate_atr_stop_loss(
            entry_price, atr=50, multiplier=-1.5, is_long=True
        )


def test_stop_loss_calculation_zero_atr(stop_loss_calculator):
    """Prueba el cálculo de stop-loss con ATR igual a 0."""
    atr = 0
    entry_price = 1000
    stop_loss_multiplier = 1.5

    # Resultado esperado: entry_price sin cambios
    expected_stop_loss = entry_price  # 1000 - (0 * 1.5) = 1000
    calculated_stop_loss = stop_loss_calculator.calculate_atr_stop_loss(
        entry_price, atr, stop_loss_multiplier, is_long=True
    )
    
    assert calculated_stop_loss == expected_stop_loss


def test_stop_loss_calculation_short_position(stop_loss_calculator):
    """Prueba el cálculo de stop-loss para posiciones cortas."""
    atr = 50
    entry_price = 1000
    stop_loss_multiplier = 1.5

    # Para posiciones cortas, el stop-loss debería estar por encima del precio de entrada
    # Resultado esperado: entry_price + (atr * multiplier)
    expected_stop_loss = entry_price + (atr * stop_loss_multiplier)  # 1000 + (50 * 1.5) = 1075
    calculated_stop_loss = stop_loss_calculator.calculate_atr_stop_loss(
        entry_price, atr, stop_loss_multiplier, is_long=False
    )
    
    assert calculated_stop_loss == expected_stop_loss


# Pruebas para calculate_trailing_stop y update_trailing_stop
def test_trailing_stop_calculation_basic(stop_loss_calculator):
    """Prueba el cálculo inicial del trailing stop."""
    initial_price = 1000
    trailing_stop_percentage = 0.01  # 1%

    # Resultado esperado: initial_price * (1 - percentage)
    expected_trailing_stop = initial_price * (1 - trailing_stop_percentage)  # 1000 * 0.99 = 990
    trailing_stop = stop_loss_calculator.calculate_trailing_stop(
        current_price=initial_price, 
        entry_price=initial_price,
        is_long=True, 
        activation_pct=0,  # Activado inmediatamente
        atr_value=None,
        stop_pct=trailing_stop_percentage
    )
    
    assert trailing_stop["price"] == expected_trailing_stop
    assert trailing_stop["activated"] is True


def test_trailing_stop_update_price_increase(stop_loss_calculator):
    """Prueba que el trailing stop se ajuste al alza cuando el precio sube."""
    initial_price = 1000
    trailing_stop_percentage = 0.01  # 1%
    
    # Trailing stop inicial
    trailing_stop = stop_loss_calculator.calculate_trailing_stop(
        current_price=initial_price, 
        entry_price=initial_price,
        is_long=True, 
        activation_pct=0,
        atr_value=None,
        stop_pct=trailing_stop_percentage
    )
    
    initial_stop_price = trailing_stop["price"]  # Debería ser 990
    
    # Precio sube
    new_price = 1050
    updated_trailing_stop = stop_loss_calculator.calculate_trailing_stop(
        current_price=new_price, 
        entry_price=initial_price,
        is_long=True, 
        activation_pct=0,
        atr_value=None,
        stop_pct=trailing_stop_percentage
    )
    
    # Resultado esperado: new_price * (1 - percentage)
    expected_updated_stop = new_price * (1 - trailing_stop_percentage)  # 1050 * 0.99 = 1039.5
    assert updated_trailing_stop["price"] == expected_updated_stop
    assert updated_trailing_stop["price"] > initial_stop_price


def test_trailing_stop_no_update_price_decrease(stop_loss_calculator):
    """Prueba que el trailing stop no cambie cuando el precio baja (emulando un escenario real)."""
    initial_price = 1000
    trailing_stop_percentage = 0.01  # 1%
    
    # Primer trailing stop con precio inicial
    trailing_stop1 = stop_loss_calculator.calculate_trailing_stop(
        current_price=initial_price, 
        entry_price=initial_price,
        is_long=True, 
        activation_pct=0,
        atr_value=None,
        stop_pct=trailing_stop_percentage
    )
    
    # Precio sube primero
    higher_price = 1050
    trailing_stop2 = stop_loss_calculator.calculate_trailing_stop(
        current_price=higher_price, 
        entry_price=initial_price,
        is_long=True, 
        activation_pct=0,
        atr_value=None,
        stop_pct=trailing_stop_percentage,
        previous_stop=trailing_stop1["price"]
    )
    
    updated_stop_price = trailing_stop2["price"]  # Debería ser 1039.5
    
    # Precio baja
    lower_price = 1020
    trailing_stop3 = stop_loss_calculator.calculate_trailing_stop(
        current_price=lower_price, 
        entry_price=initial_price,
        is_long=True, 
        activation_pct=0,
        atr_value=None,
        stop_pct=trailing_stop_percentage,
        previous_stop=updated_stop_price
    )
    
    # El trailing stop no debe bajar
    assert trailing_stop3["price"] == updated_stop_price  # Sigue siendo 1039.5


def test_trailing_stop_calculation_zero_percentage(stop_loss_calculator):
    """Prueba el cálculo del trailing stop con porcentaje 0."""
    initial_price = 1000
    trailing_stop_percentage = 0
    
    # Resultado esperado: initial_price sin cambios
    expected_trailing_stop = initial_price  # 1000 * (1 - 0) = 1000
    trailing_stop = stop_loss_calculator.calculate_trailing_stop(
        current_price=initial_price, 
        entry_price=initial_price,
        is_long=True, 
        activation_pct=0,
        atr_value=None,
        stop_pct=trailing_stop_percentage
    )
    
    assert trailing_stop["price"] == expected_trailing_stop


def test_trailing_stop_invalid_percentage(stop_loss_calculator):
    """Prueba el manejo de porcentajes inválidos."""
    initial_price = 1000
    
    with pytest.raises(ValueError, match="Trailing stop percentage must be non-negative"):
        stop_loss_calculator.calculate_trailing_stop(
            current_price=initial_price, 
            entry_price=initial_price,
            is_long=True, 
            activation_pct=0,
            atr_value=None,
            stop_pct=-0.01
        )


def test_trailing_stop_negative_price(stop_loss_calculator):
    """Prueba el manejo de precios negativos."""
    with pytest.raises(ValueError, match="Price must be positive"):
        stop_loss_calculator.calculate_trailing_stop(
            current_price=-1000, 
            entry_price=1000,
            is_long=True, 
            activation_pct=0,
            atr_value=None,
            stop_pct=0.01
        )


def test_trailing_stop_precision(stop_loss_calculator):
    """Prueba la precisión del trailing stop con valores decimales pequeños."""
    initial_price = 1000.55
    trailing_stop_percentage = 0.015  # 1.5%
    
    # Resultado esperado con precisión
    expected_trailing_stop = initial_price * (1 - trailing_stop_percentage)  # 1000.55 * 0.985 = ~985.54175
    trailing_stop = stop_loss_calculator.calculate_trailing_stop(
        current_price=initial_price, 
        entry_price=initial_price,
        is_long=True, 
        activation_pct=0,
        atr_value=None,
        stop_pct=trailing_stop_percentage
    )
    
    assert pytest.approx(trailing_stop["price"], rel=1e-5) == expected_trailing_stop  # Tolerancia para flotantes


def test_trailing_stop_activation_threshold(stop_loss_calculator):
    """Prueba que el trailing stop se active solo cuando se alcanza el umbral de activación."""
    initial_price = 1000
    activation_percentage = 0.02  # 2% de ganancia para activar
    trailing_stop_percentage = 0.01  # 1% de stop
    
    # Con precio igual al de entrada, no debería activarse
    trailing_stop1 = stop_loss_calculator.calculate_trailing_stop(
        current_price=initial_price, 
        entry_price=initial_price,
        is_long=True, 
        activation_pct=activation_percentage,
        atr_value=None,
        stop_pct=trailing_stop_percentage
    )
    
    assert trailing_stop1["activated"] is False
    
    # Con precio ligeramente mayor, pero por debajo del umbral, sigue sin activarse
    price_below_threshold = initial_price * 1.01  # 1% de ganancia, por debajo del umbral
    trailing_stop2 = stop_loss_calculator.calculate_trailing_stop(
        current_price=price_below_threshold, 
        entry_price=initial_price,
        is_long=True, 
        activation_pct=activation_percentage,
        atr_value=None,
        stop_pct=trailing_stop_percentage
    )
    
    assert trailing_stop2["activated"] is False
    
    # Con precio por encima del umbral, debería activarse
    price_above_threshold = initial_price * 1.03  # 3% de ganancia, por encima del umbral
    trailing_stop3 = stop_loss_calculator.calculate_trailing_stop(
        current_price=price_above_threshold, 
        entry_price=initial_price,
        is_long=True, 
        activation_pct=activation_percentage,
        atr_value=None,
        stop_pct=trailing_stop_percentage
    )
    
    assert trailing_stop3["activated"] is True
    # El stop debería ser el precio actual menos el porcentaje
    expected_stop = price_above_threshold * (1 - trailing_stop_percentage)
    assert trailing_stop3["price"] == expected_stop


def test_trailing_stop_for_short_position(stop_loss_calculator):
    """Prueba el cálculo y actualización del trailing stop para posiciones cortas."""
    initial_price = 1000
    trailing_stop_percentage = 0.01  # 1%
    
    # Para una posición corta, el trailing stop inicial debería estar por encima del precio
    trailing_stop = stop_loss_calculator.calculate_trailing_stop(
        current_price=initial_price, 
        entry_price=initial_price,
        is_long=False,  # Posición corta
        activation_pct=0,
        atr_value=None,
        stop_pct=trailing_stop_percentage
    )
    
    expected_stop = initial_price * (1 + trailing_stop_percentage)  # 1000 * 1.01 = 1010
    assert trailing_stop["price"] == expected_stop
    
    # Cuando el precio baja (ganancia para posición corta), el stop debería bajar
    lower_price = 950
    updated_trailing_stop = stop_loss_calculator.calculate_trailing_stop(
        current_price=lower_price, 
        entry_price=initial_price,
        is_long=False,
        activation_pct=0,
        atr_value=None,
        stop_pct=trailing_stop_percentage,
        previous_stop=trailing_stop["price"]
    )
    
    expected_updated_stop = lower_price * (1 + trailing_stop_percentage)  # 950 * 1.01 = 959.5
    assert updated_trailing_stop["price"] == expected_updated_stop
    assert updated_trailing_stop["price"] < trailing_stop["price"]


def test_trailing_stop_with_atr(stop_loss_calculator):
    """Prueba el cálculo del trailing stop basado en ATR en lugar de porcentaje."""
    initial_price = 1000
    atr_value = 50
    atr_multiplier = 2.0
    
    # El stop debería estar a una distancia de (ATR * multiplicador) del precio
    trailing_stop = stop_loss_calculator.calculate_trailing_stop(
        current_price=initial_price, 
        entry_price=initial_price,
        is_long=True,
        activation_pct=0,
        atr_value=atr_value,
        atr_multiplier=atr_multiplier,
        stop_pct=None  # No usar porcentaje
    )
    
    expected_stop = initial_price - (atr_value * atr_multiplier)  # 1000 - (50 * 2) = 900
    assert trailing_stop["price"] == expected_stop
    
    # Cuando el precio sube, el stop también debería subir manteniendo la distancia ATR
    higher_price = 1100
    updated_trailing_stop = stop_loss_calculator.calculate_trailing_stop(
        current_price=higher_price, 
        entry_price=initial_price,
        is_long=True,
        activation_pct=0,
        atr_value=atr_value,
        atr_multiplier=atr_multiplier,
        stop_pct=None,
        previous_stop=trailing_stop["price"]
    )
    
    expected_updated_stop = higher_price - (atr_value * atr_multiplier)  # 1100 - (50 * 2) = 1000
    assert updated_trailing_stop["price"] == expected_updated_stop