"""
Tests básicos para StopLossCalculator.

Este módulo contiene tests unitarios para la funcionalidad básica
del calculador de stop-loss.
"""

import unittest
import pytest
import logging

from genesis.risk.stop_loss import StopLossCalculator

# Configurar logging para los tests
logging.basicConfig(level=logging.INFO)


class TestStopLossCalculatorBasic(unittest.TestCase):
    """Tests básicos para StopLossCalculator."""
    
    def setUp(self):
        """Configurar el entorno de prueba."""
        self.stop_loss = StopLossCalculator(default_multiplier=1.5)
    
    def test_calculate_stop_loss_long(self):
        """Verificar cálculo de stop-loss para posición larga."""
        # Caso: Posición larga con precio de entrada 50000 y ATR 1000
        entry_price = 50000
        atr = 1000
        side = "buy"
        
        # Ejecutar
        stop_price = self.stop_loss.calculate_stop_loss(entry_price, atr, side)
        
        # Verificar (esperamos 50000 - 1000*1.5 = 48500)
        self.assertEqual(stop_price, 48500)
    
    def test_calculate_stop_loss_short(self):
        """Verificar cálculo de stop-loss para posición corta."""
        # Caso: Posición corta con precio de entrada 50000 y ATR 1000
        entry_price = 50000
        atr = 1000
        side = "sell"
        
        # Ejecutar
        stop_price = self.stop_loss.calculate_stop_loss(entry_price, atr, side)
        
        # Verificar (esperamos 50000 + 1000*1.5 = 51500)
        self.assertEqual(stop_price, 51500)
    
    def test_calculate_atr_stop_loss_long(self):
        """Verificar cálculo de stop-loss ATR para posición larga."""
        entry_price = 50000
        atr = 1000
        multiplier = 2.0
        
        # Ejecutar
        stop_price = self.stop_loss.calculate_atr_stop_loss(
            entry_price, atr, multiplier, is_long=True
        )
        
        # Verificar (50000 - 1000*2 = 48000)
        self.assertEqual(stop_price, 48000)
    
    def test_calculate_atr_stop_loss_short(self):
        """Verificar cálculo de stop-loss ATR para posición corta."""
        entry_price = 50000
        atr = 1000
        multiplier = 2.0
        
        # Ejecutar
        stop_price = self.stop_loss.calculate_atr_stop_loss(
            entry_price, atr, multiplier, is_long=False
        )
        
        # Verificar (50000 + 1000*2 = 52000)
        self.assertEqual(stop_price, 52000)
    
    def test_calculate_trailing_stop_long(self):
        """Verificar cálculo de trailing stop para posición larga."""
        current_price = 55000
        entry_price = 50000
        is_long = True
        stop_pct = 0.01  # 1%
        
        # Ejecutar
        result = self.stop_loss.calculate_trailing_stop(
            current_price, entry_price, is_long, 
            stop_pct=stop_pct
        )
        
        # Verificar
        self.assertTrue(result["activated"])  # Debe estar activado
        # 55000 * (1 - 0.01) = 54450
        self.assertEqual(result["price"], 54450)
    
    def test_calculate_trailing_stop_short(self):
        """Verificar cálculo de trailing stop para posición corta."""
        current_price = 45000
        entry_price = 50000
        is_long = False
        stop_pct = 0.01  # 1%
        
        # Ejecutar
        result = self.stop_loss.calculate_trailing_stop(
            current_price, entry_price, is_long, 
            stop_pct=stop_pct
        )
        
        # Verificar
        self.assertTrue(result["activated"])  # Debe estar activado
        # 45000 * (1 + 0.01) = 45450
        self.assertEqual(result["price"], 45450)
    
    def test_trailing_stop_not_activated(self):
        """Verificar que el trailing stop no se active si no se alcanza el umbral."""
        current_price = 50500  # Sólo 1% de ganancia
        entry_price = 50000
        is_long = True
        activation_pct = 0.02  # 2% para activación
        stop_pct = 0.01  # 1%
        
        # Ejecutar
        result = self.stop_loss.calculate_trailing_stop(
            current_price, entry_price, is_long, 
            activation_pct=activation_pct,
            stop_pct=stop_pct
        )
        
        # Verificar
        self.assertFalse(result["activated"])  # No debe estar activado
        self.assertEqual(result["price"], entry_price)  # Debe retornar precio de entrada


if __name__ == "__main__":
    unittest.main()