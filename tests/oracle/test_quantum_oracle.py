#!/usr/bin/env python3
"""
Pruebas para el Oráculo Cuántico Predictivo Ultra-Divino Definitivo.

Este módulo implementa un conjunto completo de pruebas para verificar
todas las capacidades del Oráculo Cuántico, asegurando su correcto
funcionamiento en todos los aspectos:
- Inicialización y configuración
- Generación de predicciones
- Cambios dimensionales
- Análisis de mercado
- Integración con APIs externas
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

# Ajustar path para importar desde directorio raíz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importar el módulo a probar
from genesis.oracle.quantum_oracle import (
    QuantumOracle, OracleState, ConfidenceCategory
)


class TestQuantumOracle:
    """Pruebas para el Oráculo Cuántico."""
    
    def setup_method(self):
        """Configuración para cada prueba."""
        self.oracle = QuantumOracle({"dimensional_spaces": 5})
    
    async def test_initialization(self):
        """Probar inicialización básica."""
        # Verificar estado inicial
        state = self.oracle.get_state()
        assert state["state"] == "INACTIVE"
        assert not state["initialization_time"]
        assert state["dimensional_spaces"] == 5
        assert not state["tracked_assets"]
        
        # Inicializar
        result = await self.oracle.initialize()
        assert result is True
        
        # Verificar cambios
        state = self.oracle.get_state()
        assert state["state"] in ["ACTIVE", "ENHANCED"]
        assert state["tracked_assets"]
        assert len(state["tracked_assets"]) > 0
    
    async def test_update_market_data(self):
        """Probar actualización de datos de mercado."""
        # Inicializar oráculo
        await self.oracle.initialize()
        
        # Actualizar datos
        result = await self.oracle.update_market_data()
        assert result is True
        
        # Verificar datos
        assert self.oracle.get_state()["tracked_assets"]
        assert len(self.oracle.get_state()["tracked_assets"]) > 0
    
    async def test_generate_predictions(self):
        """Probar generación de predicciones."""
        # Inicializar y actualizar datos
        await self.oracle.initialize()
        await self.oracle.update_market_data()
        
        # Generar predicciones
        symbols = ["BTC/USDT", "ETH/USDT"]
        predictions = await self.oracle.generate_predictions(symbols)
        
        # Verificar estructura
        assert len(predictions) > 0
        for symbol in predictions:
            assert "current_price" in predictions[symbol]
            assert "price_predictions" in predictions[symbol]
            assert "overall_confidence" in predictions[symbol]
            assert "confidence_category" in predictions[symbol]
            assert "generated_at" in predictions[symbol]
            assert "dimensional_state" in predictions[symbol]
    
    async def test_dimensional_shift(self):
        """Probar cambio dimensional."""
        # Inicializar oráculo
        await self.oracle.initialize()
        
        # Registrar métricas iniciales
        initial_coherence = self.oracle.get_metrics()["oracle_metrics"]["coherence_level"]
        
        # Ejecutar cambio dimensional
        result = await self.oracle.dimensional_shift()
        
        # Verificar resultado
        assert result["success"] is True
        assert result["old_coherence_level"] == initial_coherence
        assert result["new_coherence_level"] > initial_coherence
        assert "coherence_improvement" in result
    
    async def test_analyze_market_sentiment(self):
        """Probar análisis de sentimiento de mercado."""
        # Inicializar y actualizar datos
        await self.oracle.initialize()
        await self.oracle.update_market_data()
        
        # Analizar mercado general
        sentiment = await self.oracle.analyze_market_sentiment()
        
        # Verificar estructura
        assert "sentiments" in sentiment
        assert "dominant_sentiment" in sentiment
        assert "indicators" in sentiment
        assert "analyzed_at" in sentiment
        assert "confidence" in sentiment
        
        # Analizar símbolo específico
        symbol = "BTC/USDT"
        symbol_sentiment = await self.oracle.analyze_market_sentiment(symbol)
        
        # Verificar datos específicos del símbolo
        assert "symbol" in symbol_sentiment
        assert symbol_sentiment["symbol"] == symbol
        assert "price" in symbol_sentiment
        assert "trend" in symbol_sentiment
    
    async def test_get_prediction_accuracy(self):
        """Probar cálculo de precisión de predicciones."""
        # Inicializar y preparar datos
        await self.oracle.initialize()
        await self.oracle.update_market_data()
        
        # Generar algunas predicciones para tener historial
        symbols = ["BTC/USDT"]
        await self.oracle.generate_predictions(symbols)
        
        # Simular paso del tiempo modificando tiempo de creación en el historial
        if hasattr(self.oracle, "_prediction_history") and symbols[0] in self.oracle._prediction_history:
            for entry in self.oracle._prediction_history[symbols[0]]:
                # Modificar timestamp para simular que fue hace una hora
                timestamp = datetime.fromisoformat(entry["timestamp"])
                entry["timestamp"] = (timestamp - timedelta(hours=2)).isoformat()
        
        # Verificar precisión
        accuracy = await self.oracle.get_prediction_accuracy(symbols[0])
        
        # El resultado puede variar, pero debe tener estructura válida
        assert accuracy["symbol"] == symbols[0]
        if accuracy["status"] == "success":
            assert "average_error_pct" in accuracy
            assert "accuracy_categories" in accuracy
            assert "accuracy_percentage" in accuracy
    
    async def test_diagnose(self):
        """Probar diagnóstico del oráculo."""
        # Inicializar oráculo
        await self.oracle.initialize()
        
        # Ejecutar diagnóstico
        diagnosis = await self.oracle.diagnose()
        
        # Verificar estructura
        assert "oracle_state" in diagnosis
        assert "coherence" in diagnosis
        assert "dimensional_stability" in diagnosis
        assert "time_metrics" in diagnosis
        assert "api_access" in diagnosis
        assert "enhanced_capabilities" in diagnosis
        assert "recommended_actions" in diagnosis
    
    async def test_metrics(self):
        """Probar obtención de métricas."""
        # Inicializar y realizar algunas operaciones
        await self.oracle.initialize()
        await self.oracle.update_market_data()
        await self.oracle.generate_predictions(["BTC/USDT"])
        
        # Obtener métricas
        metrics = self.oracle.get_metrics()
        
        # Verificar estructura
        assert "oracle_metrics" in metrics
        assert "api_calls" in metrics
        assert "performance" in metrics
        
        # Verificar actualización de métricas
        assert metrics["oracle_metrics"]["predictions_generated"] > 0
        assert metrics["performance"]["successful_predictions"] > 0


# Ejecutar pruebas
if __name__ == "__main__":
    # Función para ejecutar pruebas asíncronas
    async def run_tests():
        test = TestQuantumOracle()
        test.setup_method()
        
        print("\n=== Pruebas del Oráculo Cuántico ===\n")
        
        tests = [
            test.test_initialization,
            test.test_update_market_data,
            test.test_generate_predictions,
            test.test_dimensional_shift,
            test.test_analyze_market_sentiment,
            test.test_get_prediction_accuracy,
            test.test_diagnose,
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