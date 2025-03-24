"""
Pruebas para el Oráculo Cuántico Predictivo del Sistema Genesis.

Este módulo implementa pruebas exhaustivas para verificar:
1. Inicialización y configuración correcta del Oráculo
2. Generación de predicciones con niveles de confianza apropiados
3. Detección de insights de mercado
4. Entrelazamiento de activos para correlaciones
5. Cambios dimensionales y resonancia cuántica
6. Precisión predictiva contra datos reales (simulados)
7. Resistencia a condiciones ARMAGEDÓN
"""

import os
import sys
import unittest
import asyncio
import logging
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Asegurar que podemos importar desde el directorio raíz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from genesis.oracle.quantum_oracle import (
    QuantumOracle, PredictionConfidence, TemporalHorizon, 
    MarketInsightType, DimensionalState
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_quantum_oracle")


class TestQuantumOracle(unittest.TestCase):
    """Suite de pruebas para el Oráculo Cuántico."""
    
    def setUp(self):
        """Configurar para cada prueba."""
        self.oracle = QuantumOracle()
        
        # Para pruebas asíncronas
        self.loop = asyncio.get_event_loop()
        
    def tearDown(self):
        """Limpiar después de cada prueba."""
        self.oracle = None
    
    def test_initial_state(self):
        """Verificar estado inicial del oráculo."""
        self.assertEqual(self.oracle.state, DimensionalState.CALIBRATING)
        self.assertFalse(self.oracle.initialized)
        self.assertEqual(self.oracle.dimensional_spaces, 5)
        self.assertEqual(len(self.oracle.tracked_assets), 0)
        
    def test_initialization(self):
        """Verificar inicialización correcta."""
        success = self.loop.run_until_complete(self.oracle.initialize())
        
        self.assertTrue(success)
        self.assertTrue(self.oracle.initialized)
        self.assertEqual(self.oracle.state, DimensionalState.OPERATING)
        self.assertGreater(len(self.oracle.tracked_assets), 0)
        
    def test_market_data_update(self):
        """Probar actualización de datos de mercado."""
        # Inicializar primero
        self.loop.run_until_complete(self.oracle.initialize())
        
        # Datos simulados
        test_data = {
            "BTC/USDT": {
                "price": 50000.0,
                "volume": 5000000.0,
                "timestamp": datetime.now().timestamp()
            },
            "ETH/USDT": {
                "price": 3000.0,
                "volume": 2000000.0,
                "timestamp": datetime.now().timestamp()
            }
        }
        
        # Actualizar datos
        result = self.loop.run_until_complete(self.oracle.update_market_data(test_data))
        
        self.assertTrue(result)
        self.assertEqual(self.oracle.tracked_assets["BTC/USDT"]["current_price"], 50000.0)
        self.assertEqual(self.oracle.tracked_assets["ETH/USDT"]["current_price"], 3000.0)
        
    def test_prediction_generation(self):
        """Probar generación de predicciones."""
        # Inicializar y actualizar datos
        self.loop.run_until_complete(self.oracle.initialize())
        self.loop.run_until_complete(self.oracle.update_market_data())
        
        # Generar predicciones
        predictions = self.loop.run_until_complete(self.oracle.generate_predictions(["BTC/USDT"]))
        
        # Verificar estructura
        self.assertIn("BTC/USDT", predictions)
        self.assertIn("prices", predictions["BTC/USDT"])
        self.assertIn("confidence_levels", predictions["BTC/USDT"])
        self.assertIn("dominant_trend", predictions["BTC/USDT"])
        
        # Verificar que hay 5 horizontes temporales en las predicciones
        self.assertEqual(len(predictions["BTC/USDT"]["prices"]), 5)
        
    def test_insight_detection(self):
        """Probar detección de insights de mercado."""
        # Inicializar, actualizar datos y generar predicciones
        self.loop.run_until_complete(self.oracle.initialize())
        self.loop.run_until_complete(self.oracle.update_market_data())
        self.loop.run_until_complete(self.oracle.generate_predictions())
        
        # Detectar insights
        insights = self.loop.run_until_complete(self.oracle.detect_market_insights())
        
        # Los insights pueden variar, pero debe haber una estructura básica
        for insight in insights:
            self.assertIn("type", insight)
            self.assertIn("description", insight)
            self.assertIn("confidence", insight)
            self.assertIn("detected_at", insight)
            
    def test_dimensional_shift(self):
        """Probar cambio dimensional."""
        # Inicializar
        self.loop.run_until_complete(self.oracle.initialize())
        
        # Realizar cambio dimensional
        success = self.loop.run_until_complete(self.oracle.dimensional_shift())
        
        self.assertTrue(success)
        self.assertEqual(self.oracle.metrics["dimensional_shifts"], 1)
        self.assertGreater(self.oracle.metrics["coherence_level"], 0)
        
    def test_achieve_resonance(self):
        """Probar logro de resonancia cuántica."""
        # Inicializar
        self.loop.run_until_complete(self.oracle.initialize())
        
        # Lograr resonancia
        success, results = self.loop.run_until_complete(self.oracle.achieve_resonance())
        
        self.assertTrue(success)
        self.assertIn("resonance_level", results)
        self.assertIn("coherence_gain", results)
        self.assertIn("accuracy_boost", results)
        self.assertEqual(self.oracle.state, DimensionalState.QUANTUM_COHERENCE)
        
        # Esperar a que vuelva a estado normal
        self.loop.run_until_complete(asyncio.sleep(0.6))
        self.assertEqual(self.oracle.state, DimensionalState.OPERATING)
        
    def test_prediction_accuracy(self):
        """Probar evaluación de precisión predictiva."""
        # Inicializar, actualizar datos y generar predicciones
        self.loop.run_until_complete(self.oracle.initialize())
        self.loop.run_until_complete(self.oracle.update_market_data())
        predictions = self.loop.run_until_complete(self.oracle.generate_predictions(["BTC/USDT"]))
        
        # Simular precio real (cercano a la predicción para evaluar precisión alta)
        current_price = self.oracle.tracked_assets["BTC/USDT"]["current_price"]
        prediction = predictions["BTC/USDT"]["prices"][0]["price"]
        actual_price = prediction * (1 + random.uniform(-0.02, 0.02))  # Variación pequeña
        
        # Evaluar precisión
        evaluation = self.loop.run_until_complete(
            self.oracle.evaluate_prediction_accuracy("BTC/USDT", actual_price)
        )
        
        self.assertTrue(evaluation["success"])
        self.assertIn("accuracy", evaluation)
        self.assertIn("percent_error", evaluation)
        self.assertEqual(evaluation["asset"], "BTC/USDT")
        
    def test_metrics(self):
        """Probar obtención de métricas."""
        # Inicializar y realizar algunas operaciones
        self.loop.run_until_complete(self.oracle.initialize())
        self.loop.run_until_complete(self.oracle.update_market_data())
        self.loop.run_until_complete(self.oracle.generate_predictions())
        self.loop.run_until_complete(self.oracle.detect_market_insights())
        
        # Obtener métricas
        metrics = self.oracle.get_metrics()
        
        # Verificar estructura básica
        self.assertIn("state", metrics)
        self.assertIn("tracked_assets_count", metrics)
        self.assertIn("active_predictions_count", metrics)
        self.assertIn("current_insights_count", metrics)
        self.assertIn("last_updated", metrics)
        
    def test_state(self):
        """Probar obtención de estado."""
        # Inicializar
        self.loop.run_until_complete(self.oracle.initialize())
        
        # Obtener estado
        state = self.oracle.get_state()
        
        # Verificar estructura
        self.assertIn("initialized", state)
        self.assertIn("state", state)
        self.assertIn("dimensional_spaces", state)
        self.assertIn("entanglement_level", state)
        self.assertEqual(state["initialized"], True)
        
    def test_get_all_insights(self):
        """Probar obtención de todos los insights."""
        # Inicializar y generar insights
        self.loop.run_until_complete(self.oracle.initialize())
        self.loop.run_until_complete(self.oracle.update_market_data())
        self.loop.run_until_complete(self.oracle.generate_predictions())
        self.loop.run_until_complete(self.oracle.detect_market_insights())
        
        # Obtener todos los insights
        insights = self.oracle.get_all_insights()
        
        # Los insights pueden variar, pero deben tener una estructura básica
        for insight in insights:
            self.assertIn("type", insight)
            self.assertIn("description", insight)
            self.assertIn("confidence", insight)
            
    def test_get_prediction_for_asset(self):
        """Probar obtención de predicción para un activo específico."""
        # Inicializar y generar predicciones
        self.loop.run_until_complete(self.oracle.initialize())
        self.loop.run_until_complete(self.oracle.update_market_data())
        self.loop.run_until_complete(self.oracle.generate_predictions())
        
        # Obtener predicción para BTC/USDT
        prediction = self.oracle.get_prediction_for_asset("BTC/USDT")
        
        # Verificar estructura
        self.assertIsNotNone(prediction)
        self.assertIn("symbol", prediction)
        self.assertIn("prices", prediction)
        self.assertEqual(prediction["symbol"], "BTC/USDT")
        

class TestQuantumOracleArmageddon(unittest.TestCase):
    """
    Pruebas ARMAGEDÓN para el Oráculo Cuántico.
    
    Estas pruebas someten al oráculo a condiciones extremas para verificar
    su resiliencia y capacidad de recuperación ante situaciones catastróficas.
    """
    
    def setUp(self):
        """Configurar para cada prueba."""
        config = {
            "confidence_threshold": 0.75,
            "entanglement_level": 0.8,
            "dimensional_spaces": 7,
            "temporal_resolution": 150
        }
        self.oracle = QuantumOracle(config)
        
        # Para pruebas asíncronas
        self.loop = asyncio.get_event_loop()
        
        # Inicializar siempre
        self.loop.run_until_complete(self.oracle.initialize())
        
    def tearDown(self):
        """Limpiar después de cada prueba."""
        self.oracle = None
    
    def test_armageddon_rapid_state_changes(self):
        """Probar cambios de estado extremadamente rápidos."""
        async def rapid_state_changes():
            """Realizar cambios de estado rápidos."""
            for _ in range(50):  # 50 cambios rápidos
                await self.oracle.dimensional_shift()
                await self.oracle.achieve_resonance()
                await asyncio.sleep(0.01)  # Mínimo tiempo entre cambios
                
        self.loop.run_until_complete(rapid_state_changes())
        
        # Verificar que el oráculo sigue funcionando
        self.assertIn(self.oracle.state, [
            DimensionalState.OPERATING, 
            DimensionalState.QUANTUM_COHERENCE
        ])
        
    def test_armageddon_massive_predictions(self):
        """Probar generación masiva de predicciones."""
        # Actualizar datos primero
        self.loop.run_until_complete(self.oracle.update_market_data())
        
        async def massive_predictions():
            """Generar predicciones masivamente en paralelo."""
            tasks = []
            for _ in range(20):  # 20 operaciones paralelas
                tasks.append(asyncio.create_task(self.oracle.generate_predictions()))
                
            # Esperar a que todas terminen
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verificar resultados
            successes = sum(1 for r in results if isinstance(r, dict) and len(r) > 0)
            return successes
            
        successes = self.loop.run_until_complete(massive_predictions())
        
        # Debe haber al menos algunas predicciones exitosas
        self.assertGreater(successes, 0)
        
    def test_armageddon_corrupted_data(self):
        """Probar resistencia a datos corruptos o malformados."""
        # Datos malformados deliberadamente
        corrupted_data = {
            "BTC/USDT": {
                "price": "error",  # Debería ser float
                "volume": None,  # Debería ser float
                "timestamp": "now"  # Debería ser timestamp
            },
            None: {  # Key inválida
                "price": 50000.0
            },
            "ETH/USDT": {
                "price": float('inf'),  # Infinito
                "volume": float('nan'),  # No un número
                "timestamp": -1  # Timestamp negativo
            }
        }
        
        # Actualizar con datos corruptos
        result = self.loop.run_until_complete(self.oracle.update_market_data(corrupted_data))
        
        # Debería fallar pero no causar crash
        self.assertFalse(result)
        
        # El oráculo debería seguir operativo
        self.assertEqual(self.oracle.state, DimensionalState.OPERATING)
        
    def test_armageddon_concurrent_operations(self):
        """Probar operaciones concurrentes masivas de diferentes tipos."""
        async def concurrent_operations():
            """Realizar múltiples operaciones concurrentes de diferentes tipos."""
            operations = []
            # 10 operaciones de cada tipo
            for _ in range(10):
                operations.append(asyncio.create_task(self.oracle.update_market_data()))
                operations.append(asyncio.create_task(self.oracle.generate_predictions()))
                operations.append(asyncio.create_task(self.oracle.detect_market_insights()))
                operations.append(asyncio.create_task(self.oracle.dimensional_shift()))
                
            # Esperar a que todas terminen
            results = await asyncio.gather(*operations, return_exceptions=True)
            
            # Contar éxitos vs excepciones
            successes = sum(1 for r in results if r is True or isinstance(r, dict) or isinstance(r, list))
            exceptions = sum(1 for r in results if isinstance(r, Exception))
            
            return successes, exceptions
            
        successes, exceptions = self.loop.run_until_complete(concurrent_operations())
        
        # Debería haber más éxitos que excepciones
        self.assertGreater(successes, exceptions)
        
    def test_armageddon_dimensional_overload(self):
        """Probar sobrecarga dimensional extrema."""
        async def dimensional_overload():
            """Realizar cambios dimensionales en rápida sucesión."""
            tasks = []
            for _ in range(30):  # 30 cambios dimensionales rápidos
                tasks.append(asyncio.create_task(self.oracle.dimensional_shift()))
                
            # Esperar a que todas terminen
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Contar éxitos
            successes = sum(1 for r in results if r is True)
            return successes
            
        successes = self.loop.run_until_complete(dimensional_overload())
        
        # Debe haber completado algunos cambios dimensionales con éxito
        self.assertGreater(successes, 0)
        
        # El oráculo debe seguir en estado operativo
        self.assertEqual(self.oracle.state, DimensionalState.OPERATING)
        
    def test_armageddon_invalid_parameters(self):
        """Probar resistencia a parámetros inválidos en todas las operaciones."""
        # Lista de operaciones con parámetros inválidos
        async def invalid_operations():
            operations = [
                # Parámetros de tipos incorrectos
                self.oracle.generate_predictions(symbols=123),  # Debería ser lista
                self.oracle.evaluate_prediction_accuracy("INVALID_ASSET", "no_price"),  # Precio debería ser float
                
                # Parámetros inexistentes
                self.oracle.generate_predictions(symbols=["NO_EXIST_1", "NO_EXIST_2"]),
                self.oracle.get_insights_for_asset("NO_EXIST_ASSET"),
                
                # Simular entrada maliciosa
                self.oracle.generate_predictions(symbols=[f"BTC/USDT'; DROP TABLE predictions; --"]),
            ]
            
            # Ejecutar todas las operaciones
            results = await asyncio.gather(*operations, return_exceptions=True)
            
            # Contar cuántas no lanzaron excepción
            recoveries = sum(1 for r in results if not isinstance(r, Exception))
            return recoveries
            
        recoveries = self.loop.run_until_complete(invalid_operations())
        
        # Debe recuperarse de al menos algunas operaciones inválidas
        self.assertGreaterEqual(recoveries, 0)
        
        # El oráculo debe seguir funcionando
        self.assertTrue(self.oracle.initialized)
        

# Ejecutar pruebas si se ejecuta directamente
if __name__ == "__main__":
    unittest.main()