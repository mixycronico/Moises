#!/usr/bin/env python
"""
Ejemplo de uso del Sistema de Escalabilidad Adaptativa.

Este script demuestra cómo utilizar el motor de escalabilidad
para modelar la eficiencia, predecir saturación y optimizar
asignaciones a medida que crece el capital.
"""

import asyncio
import sys
import os
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Configurar path para importar módulos de Genesis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genesis.accounting.predictive_scaling import PredictiveScalingEngine, EfficiencyPrediction

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('adaptive_scaling_example')

# Datos reales observados para diferentes instrumentos
# (capital, eficiencia) - Simulados pero basados en patrones reales
OBSERVED_DATA = {
    "BTC/USDT": [
        (1000, 0.95), (2000, 0.94), (5000, 0.92), 
        (10000, 0.89), (20000, 0.85), (50000, 0.78),
        (100000, 0.68), (200000, 0.57), (500000, 0.42)
    ],
    "ETH/USDT": [
        (1000, 0.93), (2000, 0.92), (5000, 0.90),
        (10000, 0.86), (20000, 0.81), (50000, 0.72),
        (100000, 0.62), (200000, 0.50), (500000, 0.35)
    ],
    "SOL/USDT": [
        (1000, 0.91), (2000, 0.89), (5000, 0.84),
        (10000, 0.77), (20000, 0.68), (50000, 0.53),
        (100000, 0.40), (200000, 0.28), (500000, 0.18)
    ],
    "ADA/USDT": [
        (1000, 0.89), (2000, 0.87), (5000, 0.82),
        (10000, 0.75), (20000, 0.67), (50000, 0.55),
        (100000, 0.42), (200000, 0.31), (500000, 0.20)
    ],
    "DOT/USDT": [
        (1000, 0.92), (2000, 0.90), (5000, 0.86),
        (10000, 0.80), (20000, 0.72), (50000, 0.61),
        (100000, 0.49), (200000, 0.37), (500000, 0.25)
    ]
}

# Función para visualizar curvas de eficiencia
def plot_efficiency_curves(
    predictions: Dict[str, List[Tuple[float, float]]],
    title: str = "Curvas de Eficiencia vs Capital"
):
    plt.figure(figsize=(12, 8))
    
    # Colores para cada símbolo
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    
    # Graficar cada símbolo
    for i, (symbol, data) in enumerate(predictions.items()):
        capital_values, efficiency_values = zip(*data)
        plt.plot(capital_values, efficiency_values, 
                marker='o', linestyle='-', color=colors[i % len(colors)],
                label=f"{symbol}")
    
    # Configurar gráfico
    plt.title(title, fontsize=16)
    plt.xlabel("Capital (USD)", fontsize=14)
    plt.ylabel("Eficiencia", fontsize=14)
    plt.xscale('log')  # Escala logarítmica para capital
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Agregar anotaciones para regiones de eficiencia
    plt.axhspan(0.8, 1.0, alpha=0.2, color='green', label='Alta eficiencia')
    plt.axhspan(0.5, 0.8, alpha=0.2, color='yellow', label='Media eficiencia')
    plt.axhspan(0, 0.5, alpha=0.2, color='red', label='Baja eficiencia')
    
    plt.text(1500, 0.9, "ALTA EFICIENCIA", fontsize=10, color='darkgreen')
    plt.text(1500, 0.65, "MEDIA EFICIENCIA", fontsize=10, color='darkgoldenrod')
    plt.text(1500, 0.25, "BAJA EFICIENCIA", fontsize=10, color='darkred')
    
    plt.tight_layout()
    plt.savefig("efficiency_curves.png")
    logger.info(f"Gráfico guardado como 'efficiency_curves.png'")
    plt.close()

# Función para visualizar asignaciones óptimas
def plot_allocations(
    allocations: Dict[str, Dict[float, float]],
    title: str = "Asignación Óptima a Diferentes Niveles de Capital"
):
    capital_levels = sorted(list(next(iter(allocations.values())).keys()))
    
    # Preparar datos para gráfico de barras apiladas
    symbols = list(allocations.keys())
    data = np.zeros((len(symbols), len(capital_levels)))
    
    for i, symbol in enumerate(symbols):
        for j, capital in enumerate(capital_levels):
            data[i, j] = allocations[symbol][capital]
    
    # Crear gráfico
    plt.figure(figsize=(14, 8))
    
    # Crear barras apiladas
    bottom = np.zeros(len(capital_levels))
    colors = plt.cm.viridis(np.linspace(0, 1, len(symbols)))
    
    for i, symbol in enumerate(symbols):
        plt.bar(
            [f"${c/1000:.0f}K" if c < 1000000 else f"${c/1000000:.1f}M" for c in capital_levels],
            data[i],
            bottom=bottom,
            label=symbol,
            color=colors[i]
        )
        bottom += data[i]
    
    # Configurar gráfico
    plt.title(title, fontsize=16)
    plt.xlabel("Capital Total", fontsize=14)
    plt.ylabel("Asignación (%)", fontsize=14)
    plt.ylim(0, 100)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Agregar etiquetas de porcentaje
    for i in range(len(capital_levels)):
        total = 0
        for j in range(len(symbols)):
            height = data[j, i]
            if height > 5:  # Solo mostrar etiquetas para segmentos grandes
                plt.text(
                    i, 
                    total + height/2, 
                    f"{height:.0f}%",
                    ha='center', 
                    va='center',
                    fontsize=9,
                    color='white'
                )
            total += height
    
    plt.tight_layout()
    plt.savefig("optimal_allocations.png")
    logger.info(f"Gráfico guardado como 'optimal_allocations.png'")
    plt.close()

async def run_example():
    """Ejecutar ejemplo completo del sistema de escalabilidad adaptativa."""
    try:
        logger.info("Inicializando motor de escalabilidad adaptativa...")
        
        # Crear motor predictivo
        engine = PredictiveScalingEngine(
            config={
                "default_model_type": "polynomial",  # Modelo por defecto
                "cache_ttl": 300,                   # TTL de caché (segundos)
                "auto_train": True,                 # Entrenar automáticamente
                "confidence_threshold": 0.6         # Umbral de confianza
            }
        )
        
        # 1. Cargar datos observados
        logger.info("Cargando datos históricos observados...")
        for symbol, data_points in OBSERVED_DATA.items():
            for capital, efficiency in data_points:
                await engine.add_efficiency_record(
                    symbol=symbol,
                    capital=capital,
                    efficiency=efficiency
                )
            logger.info(f"  Cargados {len(data_points)} puntos para {symbol}")
        
        # 2. Analizar modelos y puntos de saturación
        logger.info("\nAnalizando modelos y puntos de saturación...")
        saturation_points = {}
        model_quality = {}
        
        for symbol in OBSERVED_DATA.keys():
            if symbol in engine.models and engine.models[symbol].is_trained:
                model = engine.models[symbol]
                saturation_point = model.saturation_point
                
                saturation_points[symbol] = saturation_point
                model_quality[symbol] = {
                    "r_squared": model.r_squared,
                    "model_type": model.model_type,
                    "parameters": model.parameters
                }
                
                logger.info(f"  {symbol}:")
                logger.info(f"    - Tipo de modelo: {model.model_type}")
                logger.info(f"    - Calidad (R²): {model.r_squared:.4f}")
                
                if saturation_point:
                    logger.info(f"    - Punto de saturación: ${saturation_point:,.2f}")
                else:
                    logger.info(f"    - Punto de saturación: No detectado")
        
        # 3. Realizar predicciones para curvas completas
        logger.info("\nGenerando curvas de eficiencia completas...")
        
        # Definir puntos para curvas detalladas
        capital_points = [
            1000, 2000, 5000, 10000, 20000, 50000, 
            100000, 200000, 500000, 1000000, 2000000
        ]
        
        # Obtener predicciones para todos los símbolos
        prediction_curves = {}
        
        for symbol in OBSERVED_DATA.keys():
            predictions = []
            for capital in capital_points:
                prediction = await engine.predict_efficiency(symbol, capital)
                predictions.append((capital, prediction.efficiency))
            
            prediction_curves[symbol] = predictions
            
            logger.info(f"  {symbol}: Predicciones generadas para {len(capital_points)} niveles de capital")
        
        # Visualizar curvas de eficiencia
        plot_efficiency_curves(prediction_curves)
        
        # 4. Realizar optimizaciones a diferentes niveles de capital
        logger.info("\nOptimizando asignaciones a diferentes niveles de capital...")
        
        # Definir niveles de capital total para optimizar
        total_capital_levels = [
            10000, 50000, 100000, 500000, 1000000, 5000000
        ]
        
        # Optimizar para cada nivel
        symbols = list(OBSERVED_DATA.keys())
        allocations = {symbol: {} for symbol in symbols}
        efficiency_by_capital = {}
        
        for total_capital in total_capital_levels:
            # Optimizar asignación
            allocation = await engine.optimize_allocation(
                symbols=symbols,
                total_capital=total_capital,
                min_efficiency=0.3  # Permitir baja eficiencia para ver comportamiento
            )
            
            # Calcular eficiencia promedio
            total_efficiency = 0
            for symbol, amount in allocation.items():
                # Convertir a porcentaje para gráficos
                percentage = (amount / total_capital) * 100
                allocations[symbol][total_capital] = percentage
                
                # Obtener eficiencia predicha
                if amount > 0:
                    prediction = await engine.predict_efficiency(symbol, amount)
                    total_efficiency += prediction.efficiency
            
            # Calcular promedio
            avg_efficiency = total_efficiency / len([a for a in allocation.values() if a > 0])
            efficiency_by_capital[total_capital] = avg_efficiency
            
            logger.info(f"  Capital total: ${total_capital:,.2f}")
            logger.info(f"  Eficiencia promedio: {avg_efficiency:.4f}")
            logger.info(f"  Instrumentos utilizados: {len([a for a in allocation.values() if a > 0])}/{len(symbols)}")
        
        # Visualizar asignaciones óptimas
        plot_allocations(allocations)
        
        # 5. Guardar resultados
        results = {
            "saturation_points": {s: float(p) if p else None for s, p in saturation_points.items()},
            "model_quality": model_quality,
            "efficiency_by_capital": {float(k): float(v) for k, v in efficiency_by_capital.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        with open("adaptive_scaling_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("\nResultados guardados en 'adaptive_scaling_results.json'")
        logger.info("\nEjemplo completado con éxito")
        
    except Exception as e:
        logger.error(f"Error en el ejemplo: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Iniciando ejemplo del Sistema de Escalabilidad Adaptativa")
    asyncio.run(run_example())