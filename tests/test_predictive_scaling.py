"""
Test para el sistema de escalabilidad adaptativa.

Este módulo verifica la funcionalidad del motor predictivo y los modelos
de eficiencia para diferentes niveles de capital.
"""

import asyncio
import sys
import os
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Configurar path para importar módulos de Genesis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genesis.accounting.predictive_scaling import PredictiveScalingEngine, PredictiveModel, EfficiencyPrediction

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_predictive_scaling')

# Función para generar datos sintéticos que simulan curvas de eficiencia
def generate_synthetic_efficiency_data(
    symbol: str,
    capital_points: List[float],
    curve_type: str = 'polynomial',
    noise_level: float = 0.05,
    saturation_point: float = 20000,
    seed: int = None
) -> List[Tuple[float, float]]:
    """
    Generar datos sintéticos de eficiencia para pruebas.
    
    Args:
        symbol: Símbolo del instrumento
        capital_points: Puntos de capital para generar datos
        curve_type: Tipo de curva ('linear', 'polynomial', 'exponential')
        noise_level: Nivel de ruido aleatorio
        saturation_point: Punto donde comienza la saturación
        seed: Semilla para reproducibilidad
        
    Returns:
        Lista de tuplas (capital, eficiencia)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    data = []
    
    for capital in capital_points:
        # Base efficiency calculation
        if curve_type == 'linear':
            # Línea descendente: comienza alto y declina
            efficiency = max(0, min(1, 0.95 - (capital / saturation_point) * 0.5))
        elif curve_type == 'exponential':
            # Decae exponencialmente con el capital
            decay_rate = 1.5 / saturation_point
            efficiency = 0.2 + 0.75 * np.exp(-decay_rate * capital)
        else:  # polynomial (default)
            # Curva parabólica invertida
            a = -0.7 / (saturation_point ** 2)
            b = 0.7 / saturation_point
            c = 0.6
            efficiency = max(0, min(1, a * (capital ** 2) + b * capital + c))
        
        # Añadir ruido aleatorio
        noise = (random.random() - 0.5) * 2 * noise_level
        efficiency = max(0.1, min(0.98, efficiency + noise))
        
        data.append((capital, efficiency))
    
    return data

async def test_model_training():
    """
    Probar el entrenamiento de modelos predictivos con diferentes tipos.
    """
    logger.info("Prueba: Entrenamiento de modelos predictivos")
    
    # Definir símbolos y tipos de modelos para pruebas
    test_cases = [
        ("BTC/USDT", "polynomial"),
        ("ETH/USDT", "exponential"),
        ("XRP/USDT", "linear")
    ]
    
    for symbol, model_type in test_cases:
        # Generar puntos de capital de prueba
        capital_points = [1000, 2000, 5000, 10000, 15000, 20000, 30000, 50000]
        
        # Generar datos sintéticos
        synthetic_data = generate_synthetic_efficiency_data(
            symbol=symbol,
            capital_points=capital_points,
            curve_type=model_type,
            seed=42  # Para reproducibilidad
        )
        
        # Crear y entrenar modelo
        model = PredictiveModel(symbol, model_type)
        for capital, efficiency in synthetic_data:
            model.add_data_point(capital, efficiency)
        
        success = model.train()
        
        logger.info(f"Modelo {symbol} ({model_type}): {'Entrenado correctamente' if success else 'Falló entrenamiento'}")
        logger.info(f"  - R²: {model.r_squared:.4f}")
        logger.info(f"  - Error medio: {model.mean_error:.4f}")
        logger.info(f"  - Parámetros: {model.parameters}")
        
        # Verificar predicciones
        if success:
            test_points = [500, 7500, 25000, 75000]
            logger.info(f"  - Predicciones de prueba para {symbol}:")
            for point in test_points:
                pred = model.predict(point)
                conf = model.get_confidence(point)
                logger.info(f"    * Capital: ${point:,} - Eficiencia: {pred:.4f} (Confianza: {conf:.4f})")

async def test_scaling_engine():
    """
    Probar la funcionalidad del motor de escalabilidad predictiva.
    """
    logger.info("\nPrueba: Motor de escalabilidad predictiva")
    
    # Configuración del motor
    engine = PredictiveScalingEngine(
        config={
            "default_model_type": "polynomial",
            "cache_ttl": 60,
            "auto_train": True,
            "confidence_threshold": 0.5
        }
    )
    
    # Símbolos para pruebas
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
    
    # Alimentar datos al motor
    for symbol in symbols:
        capital_points = np.linspace(1000, 100000, 10).tolist()
        curve_type = random.choice(["polynomial", "exponential", "linear"])
        data = generate_synthetic_efficiency_data(
            symbol=symbol,
            capital_points=capital_points,
            curve_type=curve_type,
            seed=hash(symbol) % 10000  # Semilla diferente para cada símbolo
        )
        
        for capital, efficiency in data:
            await engine.add_efficiency_record(symbol, capital, efficiency)
        
        logger.info(f"Datos cargados para {symbol} con curva {curve_type}")
    
    # Probar predicciones
    test_points = [2000, 15000, 50000, 200000]
    for symbol in symbols:
        logger.info(f"Predicciones para {symbol}:")
        for point in test_points:
            prediction = await engine.predict_efficiency(symbol, point)
            logger.info(f"  * Capital: ${point:,} - Eficiencia: {prediction.efficiency:.4f} " + 
                      f"(Confianza: {prediction.confidence:.4f}, Tipo: {prediction.prediction_type})")
    
    # Probar optimización de asignación
    total_capital = 500000
    logger.info(f"\nOptimización de asignación con capital total: ${total_capital:,}")
    allocation = await engine.optimize_allocation(symbols, total_capital)
    
    logger.info("Resultados de asignación:")
    total_assigned = 0
    for symbol, amount in allocation.items():
        prediction = await engine.predict_efficiency(symbol, amount)
        total_assigned += amount
        logger.info(f"  * {symbol}: ${amount:,.2f} (Eficiencia esperada: {prediction.efficiency:.4f})")
    
    logger.info(f"Capital total asignado: ${total_assigned:,.2f} ({total_assigned/total_capital:.1%} del disponible)")
    logger.info(f"Estadísticas del motor: {engine.stats}")

async def visualize_predictions():
    """
    Visualizar predicciones del motor con gráficos.
    """
    logger.info("\nGenerando visualizaciones de predicciones")
    
    # Definir símbolos y tipos de modelos para visualización
    test_cases = [
        ("BTC/USDT", "polynomial"),
        ("ETH/USDT", "exponential"),
        ("XRP/USDT", "linear")
    ]
    
    # Crear una figura con subgráficos
    fig, axes = plt.subplots(len(test_cases), 1, figsize=(10, 4 * len(test_cases)))
    
    for i, (symbol, model_type) in enumerate(test_cases):
        # Generar datos de entrenamiento
        train_points = np.linspace(1000, 50000, 8).tolist()
        train_data = generate_synthetic_efficiency_data(
            symbol=symbol,
            capital_points=train_points,
            curve_type=model_type,
            seed=42  # Para reproducibilidad
        )
        
        # Crear y entrenar modelo
        model = PredictiveModel(symbol, model_type)
        for capital, efficiency in train_data:
            model.add_data_point(capital, efficiency)
        
        model.train()
        
        # Generar puntos para curva predicha
        test_x = np.linspace(500, 150000, 100)
        test_y = [model.predict(x) for x in test_x]
        confidence = [model.get_confidence(x) for x in test_x]
        
        # Graficar
        ax = axes[i] if len(test_cases) > 1 else axes
        
        # Datos de entrenamiento
        train_x, train_y = zip(*train_data)
        ax.scatter(train_x, train_y, color='blue', label='Datos de entrenamiento')
        
        # Curva de predicción
        ax.plot(test_x, test_y, 'r-', label=f'Modelo {model_type}')
        
        # Región de confianza (sombreada)
        ax.fill_between(
            test_x, 
            [max(0, y - (1-c)*0.3) for y, c in zip(test_y, confidence)],
            [min(1, y + (1-c)*0.3) for y, c in zip(test_y, confidence)],
            color='red', alpha=0.2, label='Rango de confianza'
        )
        
        # Punto de saturación si existe
        if model.saturation_point:
            ax.axvline(x=model.saturation_point, color='purple', linestyle='--', 
                      label=f'Punto de saturación: ${model.saturation_point:,.0f}')
        
        # Configuración del gráfico
        ax.set_title(f'{symbol} - Eficiencia vs Capital (R²: {model.r_squared:.4f})')
        ax.set_xlabel('Capital (USD)')
        ax.set_ylabel('Eficiencia')
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, 150000)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('efficiency_predictions.png')
    logger.info("Visualización guardada como 'efficiency_predictions.png'")

async def run_tests():
    """Ejecutar todas las pruebas."""
    try:
        await test_model_training()
        await test_scaling_engine()
        await visualize_predictions()
        logger.info("\n✅ Todas las pruebas completadas con éxito")
    except Exception as e:
        logger.error(f"❌ Error en las pruebas: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Iniciando pruebas del sistema de escalabilidad adaptativa")
    asyncio.run(run_tests())