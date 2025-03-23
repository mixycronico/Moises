"""
Ejemplo de uso del sistema de escalabilidad adaptativa.

Este script muestra cómo utilizar el sistema de escalabilidad adaptativa
para modelar la eficiencia de las criptomonedas en función del capital
y optimizar la asignación de capital entre ellas.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Añadir directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genesis.accounting.predictive_scaling import PredictiveScalingEngine, ModelFactory
from genesis.init.init_scaling import optimize_capital_allocation, quick_optimize

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("adaptive_scaling_example")

async def generate_synthetic_efficiency_data(
    symbol: str,
    capital_points: List[float],
    model_type: str = "polynomial"
) -> List[Tuple[float, float]]:
    """
    Generar datos sintéticos de eficiencia para demostración.
    
    Args:
        symbol: Símbolo del instrumento
        capital_points: Puntos de capital para generar datos
        model_type: Tipo de modelo para generar datos
        
    Returns:
        Lista de tuplas (capital, eficiencia)
    """
    # Parámetros para los diferentes modelos
    params = {
        "linear": {
            "BTC/USDT": (-0.00002, 0.9),
            "ETH/USDT": (-0.00003, 0.85),
            "SOL/USDT": (-0.00005, 0.82),
            "ADA/USDT": (-0.00008, 0.75),
            "DOT/USDT": (-0.00007, 0.78),
            "AVAX/USDT": (-0.00009, 0.8),
            "MATIC/USDT": (-0.00006, 0.77),
            "LINK/USDT": (-0.00004, 0.76),
            "XRP/USDT": (-0.00003, 0.74),
            "LTC/USDT": (-0.00005, 0.73)
        },
        "polynomial": {
            "BTC/USDT": (-0.0000000025, 0.000015, 0.85),
            "ETH/USDT": (-0.0000000035, 0.000020, 0.82),
            "SOL/USDT": (-0.0000000045, 0.000025, 0.78),
            "ADA/USDT": (-0.0000000050, 0.000030, 0.75),
            "DOT/USDT": (-0.0000000040, 0.000022, 0.77),
            "AVAX/USDT": (-0.0000000055, 0.000032, 0.76),
            "MATIC/USDT": (-0.0000000060, 0.000035, 0.73),
            "LINK/USDT": (-0.0000000045, 0.000027, 0.74),
            "XRP/USDT": (-0.0000000040, 0.000023, 0.79),
            "LTC/USDT": (-0.0000000035, 0.000021, 0.76)
        },
        "exponential": {
            "BTC/USDT": (0.6, 0.0001, 0.3),
            "ETH/USDT": (0.65, 0.00015, 0.25),
            "SOL/USDT": (0.7, 0.0002, 0.2),
            "ADA/USDT": (0.6, 0.00025, 0.15),
            "DOT/USDT": (0.55, 0.00018, 0.2),
            "AVAX/USDT": (0.58, 0.00022, 0.17),
            "MATIC/USDT": (0.62, 0.00019, 0.18),
            "LINK/USDT": (0.59, 0.00017, 0.21),
            "XRP/USDT": (0.57, 0.00016, 0.22),
            "LTC/USDT": (0.54, 0.00015, 0.24)
        }
    }
    
    # Usar modelo por defecto si no existe configuración para el símbolo
    if symbol not in params[model_type]:
        symbol = "BTC/USDT"  # Usar BTC como fallback
    
    # Generar datos
    data = []
    
    if model_type == "linear":
        # y = ax + b
        a, b = params[model_type][symbol]
        for capital in capital_points:
            efficiency = a * capital + b
            # Añadir un poco de ruido aleatorio
            efficiency += np.random.normal(0, 0.02)
            # Limitar a rango [0, 1]
            efficiency = max(0.0, min(1.0, efficiency))
            data.append((capital, efficiency))
    
    elif model_type == "polynomial":
        # y = ax² + bx + c
        a, b, c = params[model_type][symbol]
        for capital in capital_points:
            efficiency = a * capital * capital + b * capital + c
            # Añadir un poco de ruido aleatorio
            efficiency += np.random.normal(0, 0.02)
            # Limitar a rango [0, 1]
            efficiency = max(0.0, min(1.0, efficiency))
            data.append((capital, efficiency))
    
    elif model_type == "exponential":
        # y = a * exp(-b * x) + c
        a, b, c = params[model_type][symbol]
        for capital in capital_points:
            efficiency = a * np.exp(-b * capital) + c
            # Añadir un poco de ruido aleatorio
            efficiency += np.random.normal(0, 0.02)
            # Limitar a rango [0, 1]
            efficiency = max(0.0, min(1.0, efficiency))
            data.append((capital, efficiency))
    
    return data

async def example_train_model():
    """Ejemplo de entrenamiento de modelo y predicción."""
    logger.info("Ejemplo 1: Entrenamiento de modelo y predicción")
    
    # 1. Crear motor predictivo
    engine = PredictiveScalingEngine()
    
    # 2. Generar datos sintéticos para BTC
    symbol = "BTC/USDT"
    capital_points = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    data = await generate_synthetic_efficiency_data(symbol, capital_points, "polynomial")
    
    # 3. Alimentar datos al motor
    for capital, efficiency in data:
        await engine.add_efficiency_record(symbol, capital, efficiency)
    
    # 4. Entrenar modelo
    model = engine.models[symbol]
    trained = model.train()
    
    if not trained:
        logger.error("Error entrenando modelo")
        return
    
    # 5. Imprimir estadísticas y parámetros
    stats = model.get_stats()
    logger.info(f"Modelo entrenado para {symbol}:")
    logger.info(f"  Tipo: {stats['model_type']}")
    logger.info(f"  R²: {stats['r_squared']:.4f}")
    logger.info(f"  Muestras: {stats['samples_count']}")
    logger.info(f"  Error medio: {stats['mean_error']:.4f}")
    logger.info(f"  Error máximo: {stats['max_error']:.4f}")
    logger.info(f"  Punto de saturación: {stats['saturation_point']}")
    logger.info(f"  Parámetros: {stats['parameters']}")
    
    # 6. Realizar predicciones
    test_capitals = [1000, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000]
    
    print("\nPredicciones:")
    print(f"{'Capital':>10} | {'Eficiencia':>10} | {'Confianza':>10}")
    print(f"{'-'*10} | {'-'*10} | {'-'*10}")
    
    for capital in test_capitals:
        prediction = await engine.predict_efficiency(symbol, capital)
        print(f"{capital:>10.0f} | {prediction.efficiency:>10.4f} | {prediction.confidence:>10.4f}")

async def example_multiple_models():
    """Ejemplo de entrenamiento de múltiples modelos y comparación."""
    logger.info("Ejemplo 2: Comparación de modelos predictivos")
    
    # 1. Crear modelos para BTC/USDT
    symbol = "BTC/USDT"
    linear_model = ModelFactory.create_model(symbol, "linear")
    poly_model = ModelFactory.create_model(symbol, "polynomial")
    exp_model = ModelFactory.create_model(symbol, "exponential")
    
    # 2. Generar datos para entrenamiento
    capital_points = list(range(1000, 100001, 5000))  # 1K a 100K en pasos de 5K
    
    linear_data = await generate_synthetic_efficiency_data(symbol, capital_points, "linear")
    poly_data = await generate_synthetic_efficiency_data(symbol, capital_points, "polynomial")
    exp_data = await generate_synthetic_efficiency_data(symbol, capital_points, "exponential")
    
    # 3. Entrenar cada modelo con sus datos correspondientes
    for capital, efficiency in linear_data:
        linear_model.add_data_point(capital, efficiency)
    
    for capital, efficiency in poly_data:
        poly_model.add_data_point(capital, efficiency)
    
    for capital, efficiency in exp_data:
        exp_model.add_data_point(capital, efficiency)
    
    linear_model.train()
    poly_model.train()
    exp_model.train()
    
    # 4. Calcular métricas y puntos de saturación
    logger.info(f"Modelos entrenados para {symbol}:")
    
    logger.info(f"  Modelo Lineal:")
    logger.info(f"    R²: {linear_model.r_squared:.4f}")
    logger.info(f"    Error medio: {linear_model.mean_error:.4f}")
    logger.info(f"    Punto de saturación: {linear_model.saturation_point}")
    
    logger.info(f"  Modelo Polinomial:")
    logger.info(f"    R²: {poly_model.r_squared:.4f}")
    logger.info(f"    Error medio: {poly_model.mean_error:.4f}")
    logger.info(f"    Punto de saturación: {poly_model.saturation_point}")
    
    logger.info(f"  Modelo Exponencial:")
    logger.info(f"    R²: {exp_model.r_squared:.4f}")
    logger.info(f"    Error medio: {exp_model.mean_error:.4f}")
    logger.info(f"    Punto de saturación: {exp_model.saturation_point}")
    
    # 5. Visualizar modelos (si se ejecuta en entorno con GUI)
    try:
        plt.figure(figsize=(12, 8))
        
        # Datos originales
        plt.scatter([x for x, _ in linear_data], [y for _, y in linear_data], 
                   alpha=0.5, label='Datos Lineales')
        plt.scatter([x for x, _ in poly_data], [y for _, y in poly_data], 
                   alpha=0.5, label='Datos Polinomiales')
        plt.scatter([x for x, _ in exp_data], [y for _, y in exp_data], 
                   alpha=0.5, label='Datos Exponenciales')
        
        # Curvas de predicción
        x_curve = np.linspace(0, 200000, 1000)
        y_linear = [linear_model.predict(x)[0] for x in x_curve]
        y_poly = [poly_model.predict(x)[0] for x in x_curve]
        y_exp = [exp_model.predict(x)[0] for x in x_curve]
        
        plt.plot(x_curve, y_linear, 'r-', label=f'Lineal (R²={linear_model.r_squared:.4f})')
        plt.plot(x_curve, y_poly, 'g-', label=f'Polinomial (R²={poly_model.r_squared:.4f})')
        plt.plot(x_curve, y_exp, 'b-', label=f'Exponencial (R²={exp_model.r_squared:.4f})')
        
        # Marcar puntos de saturación
        if linear_model.saturation_point:
            plt.axvline(x=linear_model.saturation_point, color='r', linestyle='--', 
                       label=f'Saturación Lineal ({linear_model.saturation_point:.0f})')
        
        if poly_model.saturation_point:
            plt.axvline(x=poly_model.saturation_point, color='g', linestyle='--', 
                       label=f'Saturación Polinomial ({poly_model.saturation_point:.0f})')
        
        if exp_model.saturation_point:
            plt.axvline(x=exp_model.saturation_point, color='b', linestyle='--', 
                       label=f'Saturación Exponencial ({exp_model.saturation_point:.0f})')
        
        plt.title(f'Comparación de Modelos Predictivos para {symbol}')
        plt.xlabel('Capital')
        plt.ylabel('Eficiencia')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        
        # Guardar la figura
        plt.savefig('model_comparison.png')
        logger.info("Gráfico guardado como 'model_comparison.png'")
    except Exception as e:
        logger.warning(f"Error generando visualización: {e}")

async def example_optimize_allocation():
    """Ejemplo de optimización de asignación de capital."""
    logger.info("Ejemplo 3: Optimización de asignación de capital")
    
    # 1. Crear motor predictivo
    engine = PredictiveScalingEngine()
    
    # 2. Generar datos sintéticos para varios símbolos
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
    capital_points = list(range(1000, 100001, 10000))  # 1K a 100K en pasos de 10K
    
    for symbol in symbols:
        # Generar y alimentar datos
        data = await generate_synthetic_efficiency_data(symbol, capital_points, "polynomial")
        for capital, efficiency in data:
            await engine.add_efficiency_record(symbol, capital, efficiency)
        
        # Entrenar modelo
        engine.models[symbol].train()
    
    # 3. Optimizar asignación para diferentes niveles de capital
    test_capitals = [10000, 25000, 50000, 100000, 200000, 500000]
    
    for total_capital in test_capitals:
        allocations = await engine.optimize_allocation(
            symbols=symbols,
            total_capital=total_capital,
            min_efficiency=0.5
        )
        
        # Mostrar asignaciones
        logger.info(f"\nAsignación óptima para capital total: ${total_capital:,.2f}")
        logger.info(f"{'Símbolo':<10} | {'Asignación':>12} | {'% del Total':>12}")
        logger.info(f"{'-'*10} | {'-'*12} | {'-'*12}")
        
        total_alloc = sum(allocations.values())
        
        for symbol, amount in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
            if amount > 0:
                percent = amount / total_capital * 100
                logger.info(f"{symbol:<10} | ${amount:>10,.2f} | {percent:>11.2f}%")
        
        logger.info(f"{'-'*10} | {'-'*12} | {'-'*12}")
        logger.info(f"{'TOTAL':<10} | ${total_alloc:>10,.2f} | {100:>11.2f}%")
        
        # Obtener predicciones para los modelos con asignación
        logger.info("\nEficiencia predicha con asignación:")
        logger.info(f"{'Símbolo':<10} | {'Asignación':>12} | {'Eficiencia':>12} | {'Confianza':>12}")
        logger.info(f"{'-'*10} | {'-'*12} | {'-'*12} | {'-'*12}")
        
        for symbol, amount in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
            if amount > 0:
                prediction = await engine.predict_efficiency(symbol, amount)
                logger.info(f"{symbol:<10} | ${amount:>10,.2f} | {prediction.efficiency:>11.4f} | {prediction.confidence:>11.4f}")

async def example_quick_optimize():
    """Ejemplo de optimización rápida de asignación."""
    logger.info("Ejemplo 4: Optimización rápida de asignación")
    
    # 1. Definir símbolos y capital total
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT", 
              "AVAX/USDT", "MATIC/USDT", "LINK/USDT", "XRP/USDT", "LTC/USDT"]
    total_capital = 100000.0
    
    # 2. Realizar optimización rápida
    allocations = await quick_optimize(symbols, total_capital)
    
    # 3. Mostrar resultados
    logger.info(f"\nAsignación rápida para capital total: ${total_capital:,.2f}")
    logger.info(f"{'Símbolo':<10} | {'Asignación':>12} | {'% del Total':>12}")
    logger.info(f"{'-'*10} | {'-'*12} | {'-'*12}")
    
    total_alloc = sum(allocations.values())
    
    for symbol, amount in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
        if amount > 0:
            percent = amount / total_capital * 100
            logger.info(f"{symbol:<10} | ${amount:>10,.2f} | {percent:>11.2f}%")
    
    logger.info(f"{'-'*10} | {'-'*12} | {'-'*12}")
    logger.info(f"{'TOTAL':<10} | ${total_alloc:>10,.2f} | {100:>11.2f}%")

async def run_examples():
    """Ejecutar todos los ejemplos."""
    # Header
    print("=" * 80)
    print(" SISTEMA DE ESCALABILIDAD ADAPTATIVA - EJEMPLOS DE USO ".center(80))
    print("=" * 80)
    print()
    
    # Ejemplo 1: Entrenamiento básico
    await example_train_model()
    print("\n" + "-" * 80 + "\n")
    
    # Ejemplo 2: Comparación de modelos
    await example_multiple_models()
    print("\n" + "-" * 80 + "\n")
    
    # Ejemplo 3: Optimización de asignación
    await example_optimize_allocation()
    print("\n" + "-" * 80 + "\n")
    
    # Ejemplo 4: Optimización rápida
    await example_quick_optimize()
    print("\n" + "-" * 80 + "\n")
    
    # Footer
    print("=" * 80)
    print(" FIN DE LOS EJEMPLOS ".center(80))
    print("=" * 80)

if __name__ == "__main__":
    # Ejecutar ejemplos
    asyncio.run(run_examples())