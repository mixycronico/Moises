#!/usr/bin/env python
"""
Analizador de eficiencia de capital y optimizador de asignación.

Este script proporciona una interfaz de línea de comandos para interactuar
con el motor de escalabilidad adaptativa, permitiendo predecir la eficiencia
a diferentes niveles de capital y optimizar la asignación entre instrumentos.
"""

import asyncio
import sys
import os
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configurar path para importar módulos de Genesis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genesis.accounting.predictive_scaling import PredictiveScalingEngine
from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.utils.helpers import format_timestamp

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('capital_scaling_analyzer')

# Clase para formatear la salida JSON
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

async def load_database_records(db_url: Optional[str] = None) -> Tuple[PredictiveScalingEngine, List[str]]:
    """
    Cargar registros históricos desde la base de datos.
    
    Args:
        db_url: URL de conexión a la base de datos (opcional)
        
    Returns:
        Tuple (motor_predictivo, símbolos_disponibles)
    """
    engine = PredictiveScalingEngine(
        config={
            "default_model_type": "polynomial",
            "cache_ttl": 300,
            "auto_train": True,
            "confidence_threshold": 0.7
        }
    )
    
    symbols = []
    
    # Si se proporciona URL de BD, cargar registros históricos
    if db_url:
        try:
            logger.info(f"Conectando a la base de datos: {db_url}")
            db = TranscendentalDatabase(db_url)
            await db.connect()
            
            # Cargar registros de eficiencia
            rows = await db.fetch(
                "SELECT symbol, capital_level, efficiency FROM efficiency_records"
            )
            
            # Agrupar por símbolo
            records_by_symbol = {}
            for row in rows:
                symbol = row['symbol']
                if symbol not in records_by_symbol:
                    records_by_symbol[symbol] = []
                    symbols.append(symbol)
                records_by_symbol[symbol].append(row)
            
            # Alimentar registros al motor
            for symbol, records in records_by_symbol.items():
                for record in records:
                    await engine.add_efficiency_record(
                        symbol=record['symbol'],
                        capital=record['capital_level'],
                        efficiency=record['efficiency']
                    )
                
                logger.info(f"Cargados {len(records)} registros para {symbol}")
            
            await db.disconnect()
        except Exception as e:
            logger.error(f"Error cargando datos de la base de datos: {e}")
            
    return engine, symbols

async def predict_efficiency_for_symbol(engine: PredictiveScalingEngine, symbol: str, capital_levels: List[float]) -> None:
    """
    Predecir eficiencia para un símbolo a diferentes niveles de capital.
    
    Args:
        engine: Motor predictivo
        symbol: Símbolo del instrumento
        capital_levels: Niveles de capital a predecir
    """
    logger.info(f"Predicciones de eficiencia para {symbol}:")
    
    if not capital_levels:
        capital_levels = [1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    
    results = []
    for level in sorted(capital_levels):
        prediction = await engine.predict_efficiency(symbol, level)
        
        # Imprimir resultado
        logger.info(f"  Capital: ${level:,.2f} - Eficiencia: {prediction.efficiency:.4f} " +
                   f"(Confianza: {prediction.confidence:.4f})")
        
        # Guardar para salida JSON
        results.append(prediction.to_dict())
    
    # Guardar resultados en archivo JSON
    output_file = f"prediccion_{symbol.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=EnhancedJSONEncoder)
    
    logger.info(f"Resultados guardados en {output_file}")

async def optimize_allocation(engine: PredictiveScalingEngine, symbols: List[str], total_capital: float) -> None:
    """
    Optimizar la asignación de capital entre instrumentos.
    
    Args:
        engine: Motor predictivo
        symbols: Lista de símbolos
        total_capital: Capital total a asignar
    """
    logger.info(f"Optimizando asignación para capital total: ${total_capital:,.2f}")
    
    # Verificar que hay símbolos
    if not symbols:
        logger.error("No hay símbolos disponibles para optimizar")
        return
    
    # Ejecutar optimización
    allocation = await engine.optimize_allocation(symbols, total_capital)
    
    # Crear estructura para resultados
    results = {
        "capital_total": total_capital,
        "fecha": datetime.now(),
        "asignacion": {},
        "resumen": {
            "instrumentos": len(allocation),
            "capital_asignado": sum(allocation.values()),
            "eficiencia_esperada": 0
        }
    }
    
    # Mostrar y almacenar resultados
    logger.info("Resultados de la optimización:")
    total_efficiency = 0
    for symbol, amount in allocation.items():
        # Obtener eficiencia esperada
        prediction = await engine.predict_efficiency(symbol, amount)
        
        # Acumular para promedio
        total_efficiency += prediction.efficiency
        
        # Imprimir
        logger.info(f"  {symbol}: ${amount:,.2f} (Eficiencia: {prediction.efficiency:.4f})")
        
        # Guardar en resultados
        results["asignacion"][symbol] = {
            "capital": amount,
            "porcentaje": amount / total_capital,
            "eficiencia_esperada": prediction.efficiency,
            "confianza": prediction.confidence
        }
    
    # Calcular promedio de eficiencia
    if allocation:
        avg_efficiency = total_efficiency / len(allocation)
        results["resumen"]["eficiencia_esperada"] = avg_efficiency
        logger.info(f"Eficiencia promedio esperada: {avg_efficiency:.4f}")
    
    # Imprimir utilización
    utilization = sum(allocation.values()) / total_capital
    results["resumen"]["utilizacion"] = utilization
    logger.info(f"Utilización de capital: {utilization:.2%}")
    
    # Guardar resultados en archivo JSON
    timestamp = format_timestamp(datetime.now(), "%Y%m%d_%H%M%S")
    output_file = f"optimizacion_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=EnhancedJSONEncoder)
    
    logger.info(f"Resultados guardados en {output_file}")

async def analyze_model_quality(engine: PredictiveScalingEngine, symbol: str) -> None:
    """
    Analizar la calidad del modelo predictivo para un símbolo.
    
    Args:
        engine: Motor predictivo
        symbol: Símbolo del instrumento
    """
    logger.info(f"Analizando modelo predictivo para {symbol}")
    
    # Verificar si el modelo existe
    if symbol not in engine.models:
        logger.error(f"No hay datos para el símbolo {symbol}")
        return
    
    model = engine.models[symbol]
    
    # Entrenar si es necesario
    if not model.is_trained:
        success = model.train()
        if not success:
            logger.error(f"No se pudo entrenar el modelo para {symbol}")
            return
    
    # Obtener estadísticas
    stats = {
        "symbol": symbol,
        "model_type": model.model_type,
        "r_squared": model.r_squared,
        "mean_error": model.mean_error,
        "max_error": model.max_error,
        "parameters": model.parameters,
        "data_points": len(model.data_points),
        "saturation_point": model.saturation_point,
        "valid_range": model.valid_range,
    }
    
    # Imprimir estadísticas
    logger.info(f"Estadísticas del modelo para {symbol}:")
    logger.info(f"  Tipo de modelo: {stats['model_type']}")
    logger.info(f"  R²: {stats['r_squared']:.4f}")
    logger.info(f"  Error medio: {stats['mean_error']:.4f}")
    logger.info(f"  Error máximo: {stats['max_error']:.4f}")
    logger.info(f"  Puntos de datos: {stats['data_points']}")
    
    if stats['saturation_point']:
        logger.info(f"  Punto de saturación: ${stats['saturation_point']:,.2f}")
    
    logger.info(f"  Rango válido: ${stats['valid_range'][0]:,.2f} - ${stats['valid_range'][1]:,.2f}")
    logger.info(f"  Parámetros: {stats['parameters']}")
    
    # Guardar resultados
    output_file = f"modelo_{symbol.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2, cls=EnhancedJSONEncoder)
    
    logger.info(f"Estadísticas guardadas en {output_file}")

async def main():
    """Función principal del script."""
    # Configuración del analizador de argumentos
    parser = argparse.ArgumentParser(description='Analizador de eficiencia de capital')
    
    # Modo de operación
    parser.add_argument('modo', choices=['predecir', 'optimizar', 'analizar'],
                        help='Modo de operación (predecir eficiencia/optimizar asignación/analizar modelo)')
    
    # Argumentos comunes
    parser.add_argument('--db', type=str, help='URL de la base de datos para cargar datos históricos')
    
    # Argumentos para modo predecir
    parser.add_argument('--symbol', type=str, help='Símbolo a analizar')
    parser.add_argument('--capital', type=float, nargs='+', 
                       help='Niveles de capital a predecir (múltiples valores separados por espacio)')
    
    # Argumentos para modo optimizar
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Lista de símbolos para optimizar (separados por espacio)')
    parser.add_argument('--total', type=float, default=100000,
                       help='Capital total a asignar (default: 100000)')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Cargar motor predictivo y datos históricos
    engine, available_symbols = await load_database_records(args.db)
    
    # Ejecutar modo seleccionado
    if args.modo == 'predecir':
        if not args.symbol:
            parser.error("--symbol es requerido para el modo 'predecir'")
        await predict_efficiency_for_symbol(engine, args.symbol, args.capital)
        
    elif args.modo == 'optimizar':
        # Determinar símbolos a usar
        symbols_to_use = args.symbols or available_symbols
        if not symbols_to_use:
            parser.error("No hay símbolos especificados ni disponibles para optimizar")
        await optimize_allocation(engine, symbols_to_use, args.total)
        
    elif args.modo == 'analizar':
        if not args.symbol:
            parser.error("--symbol es requerido para el modo 'analizar'")
        await analyze_model_quality(engine, args.symbol)
    
    logger.info("Análisis completado")

if __name__ == "__main__":
    asyncio.run(main())