#!/usr/bin/env python3
"""
Pruebas para el Adaptador ARMAGEDÓN Ultra-Divino.

Este módulo implementa un conjunto completo de pruebas para verificar
todas las capacidades del Adaptador ARMAGEDÓN, asegurando su correcto
funcionamiento en la simulación de condiciones extremas y su integración
con APIs externas como DeepSeek, AlphaVantage y CoinMarketCap.
"""

import os
import sys
import json
import asyncio
import pytest
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from unittest.mock import Mock, patch, AsyncMock

# Ajustar path para importar desde directorio raíz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importar los módulos a probar
from genesis.oracle.quantum_oracle import QuantumOracle, OracleState
from genesis.oracle.armageddon_adapter import ArmageddonAdapter, ArmageddonPattern


# Mock para el oráculo cuántico en casos donde queremos aislar el adaptador
class MockQuantumOracle:
    """Versión simulada del Oráculo Cuántico para pruebas aisladas."""
    
    def __init__(self, config=None):
        self._state = {"state": "ACTIVE", "dimensional_spaces": 5, "tracked_assets": ["BTC/USDT", "ETH/USDT"]}
        self._metrics = {
            "oracle_metrics": {
                "coherence_level": 0.85,
                "dimensional_stability": 0.9,
                "resonance_frequency": 0.8
            },
            "api_calls": {
                "ALPHA_VANTAGE": 0,
                "COINMARKETCAP": 0,
                "DEEPSEEK": 0
            }
        }
        self._initialized = True
    
    def get_state(self):
        return self._state
    
    def get_metrics(self):
        return self._metrics
    
    async def initialize(self):
        return True
    
    async def dimensional_shift(self):
        return {"success": True, "new_coherence_level": 0.92, "old_coherence_level": 0.85}
    
    async def update_market_data(self, use_apis=False):
        return True
    
    async def generate_predictions(self, symbols, use_apis=False):
        result = {}
        for symbol in symbols:
            result[symbol] = {
                "current_price": 50000.0 if symbol == "BTC/USDT" else 3000.0,
                "price_predictions": [50100.0, 50200.0, 50500.0],
                "overall_confidence": 0.85,
                "confidence_category": "ALTA",
                "generated_at": datetime.now().isoformat(),
                "dimensional_state": "ACTIVE"
            }
        return result
    
    async def analyze_market_sentiment(self, symbol=None):
        return {
            "sentiments": {
                "positivo": 0.6,
                "neutral": 0.3,
                "negativo": 0.1
            },
            "dominant_sentiment": "positivo",
            "analyzed_at": datetime.now().isoformat(),
            "confidence": 0.85
        }


class TestArmageddonAdapter:
    """Pruebas para el Adaptador ARMAGEDÓN."""
    
    def setup_method(self):
        """Configuración para cada prueba."""
        # Usar oráculo real para pruebas integradas
        self.oracle = QuantumOracle({"dimensional_spaces": 5})
        self.adapter = ArmageddonAdapter(self.oracle)
        
        # También preparar un mock para pruebas aisladas
        self.mock_oracle = MockQuantumOracle()
        self.mock_adapter = ArmageddonAdapter(self.mock_oracle)
    
    async def test_initialization(self):
        """Probar inicialización básica."""
        # Inicializar adaptador
        result = await self.adapter.initialize()
        assert result is True
        
        # Verificar estado
        state = self.adapter.get_state()
        assert state["initialized"] is True
        assert state["armageddon_mode"] is False
        assert 0.7 <= state["armageddon_readiness"] <= 0.9
        assert 7.0 <= state["resilience_rating"] <= 9.0
        
        # Verificar APIs
        for api_state in state["api_states"].values():
            # El estado dependerá de si las claves están configuradas
            # (Podría ser True o False dependiendo del entorno)
            assert isinstance(api_state, bool)
    
    async def test_enable_armageddon_mode(self):
        """Probar activación del modo ARMAGEDÓN."""
        # Inicializar primero
        await self.adapter.initialize()
        
        # Activar modo ARMAGEDÓN
        result = await self.adapter.enable_armageddon_mode()
        assert result is True
        
        # Verificar cambios
        state = self.adapter.get_state()
        assert state["armageddon_mode"] is True
        assert state["armageddon_readiness"] > 0.7  # Debería aumentar tras la activación
    
    async def test_simulate_armageddon_pattern(self):
        """Probar simulación de patrones ARMAGEDÓN."""
        # Usar adaptador con mock para esta prueba
        await self.mock_adapter.initialize()
        await self.mock_adapter.enable_armageddon_mode()
        
        # Probar patrón básico
        result = await self.mock_adapter.simulate_armageddon_pattern(
            ArmageddonPattern.TSUNAMI_OPERACIONES,
            intensity=0.5,
            duration_seconds=1.0
        )
        
        # Verificar estructura del resultado
        assert "success" in result
        assert "pattern" in result
        assert "intensity" in result
        assert "duration_seconds" in result
        assert "recovery_needed" in result
        assert "resilience_impact" in result
        
        # También probar el patrón más severo
        result = await self.mock_adapter.simulate_armageddon_pattern(
            ArmageddonPattern.DEVASTADOR_TOTAL,
            intensity=0.8,
            duration_seconds=1.0
        )
        
        # Verificar si requiere recuperación
        if result["recovery_needed"]:
            # Desactivar modo para forzar recuperación
            await self.mock_adapter.disable_armageddon_mode()
    
    async def test_enhanced_update_market_data(self):
        """Probar actualización mejorada de datos de mercado."""
        # Inicializar adaptador
        await self.adapter.initialize()
        
        # Actualizar datos de mercado
        result = await self.adapter.enhanced_update_market_data(use_apis=True)
        assert result is True
        
        # Verificar métricas para ver si se usó alguna API
        metrics = self.adapter.get_metrics()
        
        # Las llamadas API dependerán de las claves configuradas
        # Al menos las métricas deben existir
        assert "api_calls" in metrics
        assert "ALPHA_VANTAGE" in metrics["api_calls"]
        assert "COINMARKETCAP" in metrics["api_calls"]
        assert "DEEPSEEK" in metrics["api_calls"]
    
    async def test_enhanced_generate_predictions(self):
        """Probar generación mejorada de predicciones."""
        # Inicializar adaptador
        await self.adapter.initialize()
        
        # Generar predicciones
        symbols = ["BTC/USDT"]
        predictions = await self.adapter.enhanced_generate_predictions(symbols, use_deepseek=True)
        
        # Verificar estructura
        if predictions:  # Podría estar vacío si el símbolo no está disponible
            assert symbols[0] in predictions
            prediction = predictions[symbols[0]]
            assert "current_price" in prediction
            assert "price_predictions" in prediction
            assert "overall_confidence" in prediction
    
    async def test_analyze_pattern_resilience(self):
        """Probar análisis de resiliencia ante patrones."""
        # Inicializar adaptador
        await self.adapter.initialize()
        
        # Analizar resiliencia ante patrones específicos
        patterns = [ArmageddonPattern.TSUNAMI_OPERACIONES, ArmageddonPattern.INYECCION_CAOS]
        analysis = await self.adapter.analyze_pattern_resilience(patterns)
        
        # Verificar estructura
        assert "overall_resilience" in analysis
        assert "armageddon_readiness" in analysis
        assert "pattern_results" in analysis
        
        # Verificar detalles de patrones
        for pattern_name in [p.name for p in patterns]:
            assert pattern_name in analysis["pattern_results"]
            pattern_data = analysis["pattern_results"][pattern_name]
            assert "resistance_score" in pattern_data
            assert "pattern_severity" in pattern_data
            assert "pattern_complexity" in pattern_data
    
    async def test_metrics(self):
        """Probar obtención de métricas."""
        # Inicializar adaptador y realizar algunas operaciones
        await self.adapter.initialize()
        await self.adapter.enhanced_update_market_data()
        await self.adapter.enhanced_generate_predictions(["BTC/USDT"])
        
        # Simular un patrón para generar más métricas
        await self.adapter.enable_armageddon_mode()
        await self.adapter.simulate_armageddon_pattern(ArmageddonPattern.OSCILACION_EXTREMA, 0.5, 1.0)
        await self.adapter.disable_armageddon_mode()
        
        # Obtener métricas
        metrics = self.adapter.get_metrics()
        
        # Verificar estructura
        assert "api_calls" in metrics
        assert "patterns_executed" in metrics
        assert "recoveries_performed" in metrics
        assert "enhanced_predictions" in metrics
        assert "resilience" in metrics
        
        # Verificar algunas actualizaciones específicas
        assert metrics["patterns_executed"] > 0


# Ejecutar pruebas
if __name__ == "__main__":
    # Función para ejecutar pruebas asíncronas
    async def run_tests():
        test = TestArmageddonAdapter()
        test.setup_method()
        
        print("\n=== Pruebas del Adaptador ARMAGEDÓN ===\n")
        
        tests = [
            test.test_initialization,
            test.test_enable_armageddon_mode,
            test.test_simulate_armageddon_pattern,
            test.test_enhanced_update_market_data,
            test.test_enhanced_generate_predictions,
            test.test_analyze_pattern_resilience,
            test.test_metrics
        ]
        
        for test_func in tests:
            try:
                print(f"Ejecutando {test_func.__name__}...")
                await test_func()
                print(f"✅ {test_func.__name__} completado con éxito\n")
            except Exception as e:
                print(f"❌ {test_func.__name__} falló: {e}\n")
                raise
        
        print("\n=== Todas las pruebas completadas ===\n")
    
    # Ejecutar suite de pruebas
    asyncio.run(run_tests())