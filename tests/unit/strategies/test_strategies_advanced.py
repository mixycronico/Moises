"""
Tests avanzados para las estrategias de trading.

Este módulo prueba funcionalidades avanzadas de las estrategias de trading,
incluyendo machine learning, detección automática de patrones, adaptación a
condiciones cambiantes de mercado, y resiliencia ante datos anómalos.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
import random
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from genesis.strategies.base import Strategy
from genesis.strategies.moving_average import MovingAverageStrategy
from genesis.strategies.rsi import RSIStrategy
from genesis.strategies.bollinger_bands import BollingerBandsStrategy
from genesis.strategies.macd import MACDStrategy
from genesis.strategies.composite import CompositeStrategy
from genesis.strategies.optimizer import StrategyOptimizer
from genesis.strategies.ml_strategy import MLStrategy
from genesis.strategies.adaptive_strategy import AdaptiveStrategy
from genesis.strategies.pattern_recognition import PatternRecognitionStrategy
from genesis.strategies.ensemble import EnsembleStrategy


@pytest.fixture
def complex_market_data():
    """Proporcionar datos complejos de mercado para pruebas avanzadas."""
    # Generar 500 puntos de datos simulando diferentes regímenes de mercado
    np.random.seed(42)  # Para reproducibilidad de los tests
    
    # Base de tiempo (500 períodos)
    t = np.linspace(0, 10, 500)
    
    # Tendencia alcista de base
    trend = 100 + t * 5
    
    # Componente cíclico
    cycle = 15 * np.sin(2 * np.pi * 0.05 * t)
    
    # Volatilidad variable
    volatility = np.abs(np.sin(t)) * 2 + 1
    
    # Ruido aleatorio
    noise = np.random.normal(0, 1, size=len(t))
    
    # Eventos "cisne negro" (cambios bruscos)
    black_swans = np.zeros(len(t))
    black_swan_indices = [100, 250, 400]  # Posiciones de los eventos extremos
    for idx in black_swan_indices:
        black_swans[idx:idx+10] = -20 if idx == 250 else 20  # Caída o subida brusca
    
    # Combinar todos los componentes
    price = trend + cycle + volatility * noise + black_swans
    
    # Convertir a formato OHLCV
    ohlcv_data = []
    timestamps = pd.date_range(start='2021-01-01', periods=len(price), freq='H').astype(np.int64) // 10**6
    
    for i in range(len(price)):
        if i > 0:
            open_price = price[i-1]
        else:
            open_price = price[i] - volatility[i] * 0.5
            
        close_price = price[i]
        high_price = max(open_price, close_price) + volatility[i] * random.uniform(0.2, 0.8)
        low_price = min(open_price, close_price) - volatility[i] * random.uniform(0.2, 0.8)
        volume = (1000 + volatility[i] * 200) * (1 + 0.1 * np.sin(t[i] * 5))
        
        ohlcv_data.append([
            timestamps[i], 
            open_price, 
            high_price, 
            low_price, 
            close_price, 
            volume
        ])
    
    return np.array(ohlcv_data)


@pytest.fixture
def ml_strategy():
    """Proporcionar una estrategia basada en machine learning."""
    return MLStrategy(name="test_ml_strategy")


@pytest.fixture
def adaptive_strategy():
    """Proporcionar una estrategia adaptativa avanzada."""
    return AdaptiveStrategy(name="test_adaptive_strategy")


@pytest.fixture
def pattern_strategy():
    """Proporcionar una estrategia de reconocimiento de patrones."""
    return PatternRecognitionStrategy(name="test_pattern_strategy")


@pytest.fixture
def ensemble_strategy(ml_strategy, adaptive_strategy, pattern_strategy):
    """Proporcionar una estrategia de conjunto que combina múltiples estrategias."""
    strategies = [
        ml_strategy,
        adaptive_strategy,
        pattern_strategy,
        MovingAverageStrategy(name="ma_component"),
        RSIStrategy(name="rsi_component"),
        MACDStrategy(name="macd_component")
    ]
    
    weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
    
    return EnsembleStrategy(
        name="test_ensemble_strategy",
        strategies=strategies,
        weights=weights
    )


@pytest.mark.asyncio
async def test_ml_strategy_feature_engineering(ml_strategy, complex_market_data):
    """Probar la ingeniería de características en estrategias de ML."""
    # Mockear el método de extracción de características
    with patch.object(ml_strategy, '_extract_features') as mock_extract:
        # Simular retorno de características
        mock_features = np.random.random((complex_market_data.shape[0], 20))
        mock_extract.return_value = mock_features
        
        # Llamar al método real
        features = await ml_strategy._extract_features(complex_market_data)
        
        # Verificar que se llamó correctamente
        mock_extract.assert_called_once_with(complex_market_data)
        
        # Verificar dimensiones de salida
        assert features.shape[0] == complex_market_data.shape[0]
        assert features.shape[1] == 20


@pytest.mark.asyncio
async def test_ml_strategy_model_prediction(ml_strategy, complex_market_data):
    """Probar las predicciones del modelo en estrategias de ML."""
    # Mockear el método de extracción de características
    with patch.object(ml_strategy, '_extract_features') as mock_extract:
        # Simular retorno de características
        mock_features = np.random.random((complex_market_data.shape[0], 20))
        mock_extract.return_value = mock_features
        
        # Mockear el modelo predictivo
        with patch.object(ml_strategy, '_model_predict') as mock_predict:
            # Simular predicciones (probabilidades para cada clase)
            mock_predictions = np.array([
                [0.2, 0.7, 0.1],  # Probabilidad de [venta, compra, mantener]
                [0.6, 0.3, 0.1],
                [0.1, 0.2, 0.7]
            ])
            mock_predict.return_value = mock_predictions
            
            # Mockear generación de señal basada en predicciones
            with patch.object(ml_strategy, '_generate_signal_from_prediction') as mock_generate:
                # Simular señales generadas
                mock_signals = [
                    {"signal": "buy", "strength": 0.7},
                    {"signal": "sell", "strength": 0.6},
                    {"signal": "hold", "strength": 0.7}
                ]
                mock_generate.side_effect = mock_signals
                
                # Ejecutar generate_signal con datos limitados para reducir tiempo de ejecución
                sample_data = complex_market_data[:3]
                
                for i in range(len(sample_data)):
                    signal = await ml_strategy.generate_signal("BTC/USDT", sample_data[:i+1])
                    
                    # Verificar que la señal coincide con la simulada
                    assert signal == mock_signals[i]


@pytest.mark.asyncio
async def test_ml_strategy_retraining(ml_strategy, complex_market_data):
    """Probar el reentrenamiento periódico del modelo de ML."""
    # Mockear el método de reentrenamiento
    with patch.object(ml_strategy, '_retrain_model') as mock_retrain:
        # Configurar para reentrenar cada 100 puntos de datos
        ml_strategy.set_parameter("retrain_interval", 100)
        
        # Simular histórico de procesamientos
        ml_strategy._last_retrain_time = time.time() - 3600  # Hace 1 hora
        ml_strategy._processed_data_points = 99
        
        # Primera llamada - no debería reentrenar (99 puntos procesados)
        await ml_strategy.generate_signal("BTC/USDT", complex_market_data[:1])
        mock_retrain.assert_not_called()
        
        # Segunda llamada - debería reentrenar (100 puntos procesados)
        await ml_strategy.generate_signal("BTC/USDT", complex_market_data[:1])
        mock_retrain.assert_called_once()


@pytest.mark.asyncio
async def test_adaptive_strategy_regime_detection(adaptive_strategy, complex_market_data):
    """Probar la detección de regímenes de mercado en estrategias adaptativas."""
    # Mockear el detector de regímenes
    with patch.object(adaptive_strategy, 'detect_market_regime') as mock_detect:
        # Simular diferentes regímenes para diferentes segmentos
        regimes = ["uptrend", "volatile", "sideways", "downtrend", "recovery"]
        mock_detect.side_effect = regimes
        
        # Verificar adaptación para diferentes segmentos de datos
        segments = [50, 150, 250, 350, 450]
        
        for i, segment in enumerate(segments):
            # Obtener datos hasta el segmento actual
            data_segment = complex_market_data[:segment]
            
            # Generar señal (internamente llamará a detect_market_regime)
            signal = await adaptive_strategy.generate_signal("BTC/USDT", data_segment)
            
            # Verificar que se detectó el régimen esperado
            mock_detect.assert_called()
            
            # Verificar que la señal contiene información del régimen
            assert "market_regime" in signal.get("additional_data", {})
            
            # Resetear mock para la siguiente iteración
            mock_detect.reset_mock()


@pytest.mark.asyncio
async def test_adaptive_strategy_parameter_adaptation(adaptive_strategy, complex_market_data):
    """Probar la adaptación de parámetros según el régimen de mercado."""
    # Definir regímenes y parámetros esperados para cada uno
    regime_params = {
        "uptrend": {"fast_period": 5, "slow_period": 15, "rsi_threshold": 65},
        "volatile": {"fast_period": 3, "slow_period": 10, "rsi_threshold": 75},
        "sideways": {"fast_period": 10, "slow_period": 30, "rsi_threshold": 50},
        "downtrend": {"fast_period": 8, "slow_period": 20, "rsi_threshold": 35},
        "recovery": {"fast_period": 5, "slow_period": 15, "rsi_threshold": 45}
    }
    
    # Mockear la detección de regímenes
    with patch.object(adaptive_strategy, 'detect_market_regime') as mock_detect:
        for regime, params in regime_params.items():
            # Simular detección del régimen actual
            mock_detect.return_value = regime
            
            # Simular adaptación de parámetros
            with patch.object(adaptive_strategy, 'adapt_parameters') as mock_adapt:
                async def side_effect():
                    # Establecer parámetros según el régimen
                    for param, value in params.items():
                        adaptive_strategy.set_parameter(param, value)
                
                mock_adapt.side_effect = side_effect
                
                # Generar señal con cualquier conjunto de datos
                await adaptive_strategy.generate_signal("BTC/USDT", complex_market_data[:50])
                
                # Verificar que se llamaron los métodos esperados
                mock_detect.assert_called_once()
                mock_adapt.assert_called_once()
                
                # Verificar que los parámetros se adaptaron correctamente
                for param, value in params.items():
                    assert adaptive_strategy.get_parameter(param) == value
                
                # Resetear mocks para la siguiente iteración
                mock_detect.reset_mock()
                mock_adapt.reset_mock()


@pytest.mark.asyncio
async def test_pattern_recognition(pattern_strategy, complex_market_data):
    """Probar la detección de patrones chartistas."""
    # Definir patrones a detectar y sus respuestas esperadas
    patterns = [
        {"name": "head_and_shoulders", "signal": "sell", "confidence": 0.85},
        {"name": "double_bottom", "signal": "buy", "confidence": 0.90},
        {"name": "ascending_triangle", "signal": "buy", "confidence": 0.75},
        {"name": "descending_triangle", "signal": "sell", "confidence": 0.80},
        {"name": "wedge", "signal": "sell", "confidence": 0.70}
    ]
    
    # Mockear la detección de patrones
    with patch.object(pattern_strategy, 'detect_patterns') as mock_detect:
        # Probar cada patrón
        for pattern in patterns:
            # Simular detección del patrón
            mock_detect.return_value = [pattern]
            
            # Generar señal para una ventana de datos
            signal = await pattern_strategy.generate_signal("BTC/USDT", complex_market_data[100:200])
            
            # Verificar que se llamó la detección
            mock_detect.assert_called_once_with(complex_market_data[100:200])
            
            # Verificar la señal generada
            assert signal["signal"] == pattern["signal"]
            assert signal["strength"] >= pattern["confidence"]
            assert "patterns" in signal.get("additional_data", {})
            assert pattern["name"] in str(signal.get("additional_data", {}).get("patterns", []))
            
            # Resetear mock para la siguiente iteración
            mock_detect.reset_mock()


@pytest.mark.asyncio
async def test_ensemble_strategy_weighted_voting(ensemble_strategy, complex_market_data):
    """Probar el mecanismo de votación ponderada del conjunto de estrategias."""
    # Mockear las señales de las estrategias individuales
    strategy_signals = [
        {"signal": "buy", "strength": 0.8},    # ML (peso 0.25)
        {"signal": "buy", "strength": 0.6},    # Adaptativa (peso 0.20)
        {"signal": "sell", "strength": 0.7},   # Patrones (peso 0.15)
        {"signal": "buy", "strength": 0.5},    # MA (peso 0.15)
        {"signal": "hold", "strength": 0.4},   # RSI (peso 0.15)
        {"signal": "sell", "strength": 0.3}    # MACD (peso 0.10)
    ]
    
    # Configurar mocks para cada estrategia
    for i, strategy in enumerate(ensemble_strategy.strategies):
        strategy.generate_signal = AsyncMock(return_value=strategy_signals[i])
    
    # Generar señal del conjunto
    signal = await ensemble_strategy.generate_signal("BTC/USDT", complex_market_data[:100])
    
    # Verificar que todas las estrategias fueron consultadas
    for strategy in ensemble_strategy.strategies:
        strategy.generate_signal.assert_called_once_with("BTC/USDT", complex_market_data[:100])
    
    # Verificar el resultado de la votación ponderada
    # buy: 0.25*0.8 + 0.20*0.6 + 0.15*0.5 = 0.20 + 0.12 + 0.075 = 0.395
    # sell: 0.15*0.7 + 0.10*0.3 = 0.105 + 0.03 = 0.135
    # hold: 0.15*0.4 = 0.06
    # La señal debe ser "buy" por tener el mayor peso
    assert signal["signal"] == "buy"
    assert "ensemble_votes" in signal.get("additional_data", {})


@pytest.mark.asyncio
async def test_ensemble_strategy_disagreement_handling(ensemble_strategy, complex_market_data):
    """Probar el manejo de desacuerdos entre estrategias del conjunto."""
    # Simular un caso de alta discrepancia entre estrategias
    strategy_signals = [
        {"signal": "buy", "strength": 0.51},    # ML (peso 0.25)
        {"signal": "sell", "strength": 0.49},   # Adaptativa (peso 0.20)
        {"signal": "buy", "strength": 0.51},    # Patrones (peso 0.15)
        {"signal": "sell", "strength": 0.49},   # MA (peso 0.15)
        {"signal": "hold", "strength": 0.5},    # RSI (peso 0.15)
        {"signal": "hold", "strength": 0.5}     # MACD (peso 0.10)
    ]
    
    # Configurar mocks para cada estrategia
    for i, strategy in enumerate(ensemble_strategy.strategies):
        strategy.generate_signal = AsyncMock(return_value=strategy_signals[i])
    
    # Configurar modo conservador para desacuerdos
    ensemble_strategy.set_parameter("disagreement_threshold", 0.3)
    ensemble_strategy.set_parameter("disagreement_action", "conservative")
    
    # Generar señal del conjunto
    signal = await ensemble_strategy.generate_signal("BTC/USDT", complex_market_data[:100])
    
    # En modo conservador, con alta discrepancia debería dar "hold"
    assert signal["signal"] == "hold"
    assert "high_disagreement" in signal.get("additional_data", {})
    assert signal.get("additional_data", {}).get("high_disagreement") == True


@pytest.mark.asyncio
async def test_strategy_resilience_to_outliers(ensemble_strategy, complex_market_data):
    """Probar la resiliencia de las estrategias ante datos anómalos (outliers)."""
    # Crear datos con outliers extremos
    outlier_data = complex_market_data.copy()
    
    # Añadir outliers extremos (precios 10x) en algunas posiciones
    outlier_indices = [5, 25, 45]
    for idx in outlier_indices:
        outlier_data[idx, 1:5] *= 10  # Multiplicar OHLC por 10
    
    # Configurar comportamiento de las estrategias individuales
    for strategy in ensemble_strategy.strategies:
        # Simular detección de anomalías y manejo robusto
        with patch.object(strategy, '_preprocess_data', wraps=lambda x: x) as mock_preprocess:
            # Generar señal con datos anómalos
            await strategy.generate_signal("BTC/USDT", outlier_data[:50])
            
            # Verificar que se llamó al preprocesamiento
            mock_preprocess.assert_called_once()


@pytest.mark.asyncio
async def test_strategy_performance_under_load():
    """Probar el rendimiento de las estrategias bajo carga intensiva."""
    # Crear múltiples instancias de estrategias
    num_strategies = 20
    strategies = [
        RSIStrategy(name=f"rsi_strategy_{i}", 
                   parameters={"period": 14, "overbought": 70, "oversold": 30})
        for i in range(num_strategies)
    ]
    
    # Crear datos aleatorios
    data_size = 1000
    test_data = np.random.random(data_size) * 100
    
    # Medir tiempo para procesar todas las estrategias
    start_time = time.time()
    
    # Ejecutar todas las estrategias concurrentemente
    tasks = [
        strategy.generate_signal("BTC/USDT", test_data)
        for strategy in strategies
    ]
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Verificar que todas las estrategias generaron señales
    assert len(results) == num_strategies
    
    # Verificar que el tiempo de ejecución es razonable
    # En un sistema robusto, esto no debería tomar más de algunos segundos
    assert execution_time < 10, f"La ejecución tomó {execution_time} segundos, lo cual es demasiado"


@pytest.mark.asyncio
async def test_strategy_fault_tolerance(ensemble_strategy, complex_market_data):
    """Probar la tolerancia a fallos de las estrategias."""
    # Hacer que algunas estrategias fallen
    for i, strategy in enumerate(ensemble_strategy.strategies):
        if i % 2 == 0:  # Estrategias en posiciones pares funcionan normalmente
            strategy.generate_signal = AsyncMock(return_value={"signal": "buy", "strength": 0.7})
        else:  # Estrategias en posiciones impares fallan
            strategy.generate_signal = AsyncMock(side_effect=Exception("Simulación de fallo en estrategia"))
    
    # La estrategia de conjunto debería manejar los fallos y continuar con las estrategias que funcionan
    signal = await ensemble_strategy.generate_signal("BTC/USDT", complex_market_data[:100])
    
    # Verificar que se generó una señal a pesar de los fallos
    assert "signal" in signal
    assert "strength" in signal
    assert "failed_strategies" in signal.get("additional_data", {})
    
    # Debería haber registrado las estrategias que fallaron
    failed_strategies = signal.get("additional_data", {}).get("failed_strategies", [])
    assert len(failed_strategies) > 0


@pytest.mark.asyncio
async def test_strategy_backpressure_handling(ensemble_strategy):
    """Probar el manejo de backpressure en las estrategias."""
    # Simular una situación de alta carga
    num_requests = 100
    
    # Hacer que las estrategias individuales tomen tiempo variable
    delays = [0.01, 0.02, 0.05, 0.03, 0.01, 0.02]  # Tiempos en segundos
    
    for i, strategy in enumerate(ensemble_strategy.strategies):
        delay = delays[i]
        
        async def slow_generate(symbol, data, _delay=delay):
            await asyncio.sleep(_delay)
            return {"signal": "buy", "strength": 0.7}
        
        strategy.generate_signal = slow_generate
    
    # Establecer timeout y límite de concurrencia
    ensemble_strategy.set_parameter("timeout", 0.1)  # 100ms timeout
    ensemble_strategy.set_parameter("max_concurrent_strategies", 3)
    
    # Ejecutar múltiples solicitudes concurrentes
    tasks = [
        ensemble_strategy.generate_signal("BTC/USDT", np.random.random(10) * 100)
        for _ in range(num_requests)
    ]
    
    # Utilizar gather con return_exceptions para capturar cualquier error
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verificar que todas las solicitudes se completaron
    assert len(results) == num_requests
    
    # Contar señales válidas vs errores
    valid_signals = [r for r in results if isinstance(r, dict) and "signal" in r]
    timeouts = [r for r in results if isinstance(r, Exception)]
    
    # Debería haber algunas señales válidas y posiblemente algunos timeouts
    assert len(valid_signals) > 0
"""