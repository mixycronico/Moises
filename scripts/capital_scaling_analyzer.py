#!/usr/bin/env python
"""
Herramienta de línea de comandos para analizar la escalabilidad.

Este script proporciona una interfaz para interactuar con el sistema
de escalabilidad adaptativa, permitiendo predecir eficiencia, optimizar
asignaciones y analizar modelos desde la línea de comandos.
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Configurar rutas para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genesis.accounting.predictive_scaling import PredictiveScalingEngine
from genesis.db.transcendental_database import TranscendentalDatabase

# Clase personalizada para serializar resultados
class ScalingJSONEncoder(json.JSONEncoder):
    """Serializador JSON personalizado para objetos relacionados con escalabilidad."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        return super().default(obj)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('capital_scaling_analyzer')

class CapitalScalingAnalyzer:
    """
    Analizador para el sistema de escalabilidad adaptativa.
    
    Esta clase proporciona métodos para interactuar con el sistema
    desde la línea de comandos, facilitando análisis y operaciones.
    """
    
    def __init__(self, db_url: Optional[str] = None, cache_size: str = "500"):
        """
        Inicializar el analizador.
        
        Args:
            db_url: URL de conexión a la base de datos (opcional)
            cache_size: Tamaño de caché para el motor predictivo
        """
        self.db = TranscendentalDatabase() if db_url else None
        self.engine = None
        self.cache_size = int(cache_size)
        
        logger.info(f"CapitalScalingAnalyzer inicializado")
    
    async def initialize(self) -> bool:
        """
        Inicializar componentes del analizador.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            # Conectar a base de datos si está configurada
            if self.db:
                await self.db.connect()
            
            # Inicializar motor predictivo
            self.engine = PredictiveScalingEngine(
                config={
                    "default_model_type": "polynomial",
                    "cache_ttl": 300,
                    "auto_train": True,
                    "confidence_threshold": 0.6
                }
            )
            
            # Cargar datos históricos si hay conexión a BD
            if self.db:
                records = await self.db.fetch(
                    """
                    SELECT symbol, capital_level, efficiency, roi, sharpe, max_drawdown, win_rate
                    FROM efficiency_records
                    ORDER BY symbol, capital_level
                    """
                )
                
                for record in records:
                    await self.engine.add_efficiency_record(
                        symbol=record['symbol'],
                        capital=record['capital_level'],
                        efficiency=record['efficiency'],
                        metrics={
                            "roi": record.get('roi'),
                            "sharpe": record.get('sharpe'),
                            "max_drawdown": record.get('max_drawdown'),
                            "win_rate": record.get('win_rate')
                        }
                    )
                
                logger.info(f"Cargados {len(records)} registros históricos desde BD")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en inicialización: {str(e)}")
            return False
    
    async def cleanup(self):
        """Limpiar recursos al finalizar."""
        if self.db:
            await self.db.disconnect()
    
    async def get_available_symbols(self) -> List[str]:
        """
        Obtener símbolos disponibles para análisis.
        
        Returns:
            Lista de símbolos disponibles
        """
        if self.db:
            try:
                records = await self.db.fetch(
                    """
                    SELECT DISTINCT symbol FROM efficiency_records
                    ORDER BY symbol
                    """
                )
                return [record['symbol'] for record in records]
            except Exception as e:
                logger.error(f"Error obteniendo símbolos desde BD: {str(e)}")
        
        if self.engine and hasattr(self.engine, 'models'):
            return list(self.engine.models.keys())
        
        return []
    
    async def load_synthetic_data(self):
        """Cargar datos sintéticos para demostración."""
        # Datos para varios símbolos con diferentes patrones
        data = {
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
        
        # Cargar datos en el motor
        for symbol, points in data.items():
            for capital, efficiency in points:
                await self.engine.add_efficiency_record(
                    symbol=symbol,
                    capital=capital,
                    efficiency=efficiency
                )
        
        logger.info(f"Datos sintéticos cargados para {len(data)} símbolos")
    
    async def predict_efficiency(self, symbol: str, capital_levels: List[float]) -> Dict[str, Any]:
        """
        Predecir eficiencia para un símbolo a diferentes niveles de capital.
        
        Args:
            symbol: Símbolo a analizar
            capital_levels: Niveles de capital para predicción
            
        Returns:
            Diccionario con resultados
        """
        if not self.engine:
            return {"error": "Motor predictivo no inicializado"}
        
        results = []
        for capital in capital_levels:
            prediction = await self.engine.predict_efficiency(symbol, capital)
            results.append({
                "capital": capital,
                "efficiency": prediction.efficiency,
                "confidence": prediction.confidence
            })
        
        if symbol in self.engine.models:
            model = self.engine.models[symbol]
            model_info = {
                "model_type": model.model_type,
                "parameters": model.parameters,
                "r_squared": model.r_squared,
                "saturation_point": model.saturation_point
            }
        else:
            model_info = {"error": "Modelo no disponible"}
        
        response = {
            "symbol": symbol,
            "predictions": results,
            "model_info": model_info
        }
        
        return response
    
    async def optimize_allocation(self, symbols: List[str], total_capital: float) -> Dict[str, Any]:
        """
        Optimizar asignación de capital entre instrumentos.
        
        Args:
            symbols: Lista de símbolos a considerar
            total_capital: Capital total disponible
            
        Returns:
            Diccionario con resultados
        """
        if not self.engine:
            return {"error": "Motor predictivo no inicializado"}
        
        # Ejecutar optimización
        allocations = await self.engine.optimize_allocation(
            symbols=symbols,
            total_capital=total_capital,
            min_efficiency=0.3  # Umbral bajo para ver todo el espectro
        )
        
        # Obtener eficiencia por símbolo
        efficiency_by_symbol = {}
        total_efficiency = 0.0
        symbols_used = 0
        
        for symbol, amount in allocations.items():
            if amount > 0:
                prediction = await self.engine.predict_efficiency(symbol, amount)
                efficiency_by_symbol[symbol] = prediction.efficiency
                total_efficiency += prediction.efficiency
                symbols_used += 1
        
        # Calcular eficiencia promedio
        avg_efficiency = total_efficiency / symbols_used if symbols_used > 0 else 0.0
        
        # Calcular utilización de capital
        capital_utilization = sum(allocations.values()) / total_capital
        
        # Calcular distribución porcentual
        percentage_allocation = {
            symbol: (amount / total_capital * 100) for symbol, amount in allocations.items()
        }
        
        response = {
            "total_capital": total_capital,
            "allocations": allocations,
            "percentage_allocation": percentage_allocation,
            "efficiency_by_symbol": efficiency_by_symbol,
            "avg_efficiency": avg_efficiency,
            "capital_utilization": capital_utilization,
            "symbols_used": symbols_used,
            "symbols_total": len(symbols)
        }
        
        return response
    
    async def analyze_model(self, symbol: str) -> Dict[str, Any]:
        """
        Analizar el modelo predictivo para un símbolo.
        
        Args:
            symbol: Símbolo a analizar
            
        Returns:
            Diccionario con resultados
        """
        if not self.engine:
            return {"error": "Motor predictivo no inicializado"}
        
        if symbol not in self.engine.models:
            return {"error": f"No hay modelo disponible para {symbol}"}
        
        model = self.engine.models[symbol]
        
        # Obtener información del modelo
        model_info = {
            "model_type": model.model_type,
            "parameters": model.parameters,
            "r_squared": model.r_squared,
            "samples_count": model.samples_count if hasattr(model, 'samples_count') else None,
            "saturation_point": model.saturation_point
        }
        
        # Generar puntos para la curva completa
        capital_points = [
            1000, 2000, 5000, 10000, 20000, 50000, 
            100000, 200000, 500000, 1000000, 2000000
        ]
        
        # Obtener predicciones
        curve_points = []
        for capital in capital_points:
            prediction = await self.engine.predict_efficiency(symbol, capital)
            curve_points.append({
                "capital": capital,
                "efficiency": prediction.efficiency,
                "confidence": prediction.confidence
            })
        
        # Obtener datos históricos
        historical = []
        if hasattr(model, 'data_points'):
            for point in model.data_points:
                historical.append({
                    "capital": point[0],
                    "efficiency": point[1]
                })
        
        response = {
            "symbol": symbol,
            "model_info": model_info,
            "curve_points": curve_points,
            "historical_points": historical
        }
        
        return response
    
    # Métodos para visualización
    
    def plot_efficiency_curve(self, data: Dict[str, Any], output_file: str = None):
        """
        Generar gráfico de curva de eficiencia.
        
        Args:
            data: Datos del análisis de modelo
            output_file: Ruta para guardar el gráfico (opcional)
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Graficar puntos históricos
            historical = data.get('historical_points', [])
            if historical:
                x_hist = [p['capital'] for p in historical]
                y_hist = [p['efficiency'] for p in historical]
                plt.scatter(x_hist, y_hist, color='blue', marker='o', s=80, 
                           label='Datos históricos', alpha=0.7)
            
            # Graficar curva de predicción
            curve = data.get('curve_points', [])
            if curve:
                x_curve = [p['capital'] for p in curve]
                y_curve = [p['efficiency'] for p in curve]
                plt.plot(x_curve, y_curve, 'r-', linewidth=2, 
                        label='Modelo predictivo')
            
            # Marcar punto de saturación
            saturation = data.get('model_info', {}).get('saturation_point')
            if saturation:
                # Encontrar eficiencia en el punto de saturación
                efficiency_at_saturation = None
                for point in curve:
                    if abs(point['capital'] - saturation) / saturation < 0.1:  # Aproximadamente igual
                        efficiency_at_saturation = point['efficiency']
                        break
                
                if efficiency_at_saturation is not None:
                    plt.axvline(x=saturation, color='orange', linestyle='--', alpha=0.7,
                               label=f'Punto de saturación: ${saturation:,.0f}')
                    plt.plot([saturation], [efficiency_at_saturation], 'o', markersize=10,
                            color='orange')
            
            # Configuración del gráfico
            plt.title(f"Curva de Eficiencia vs Capital - {data.get('symbol', 'Desconocido')}", fontsize=16)
            plt.xlabel("Capital (USD)", fontsize=14)
            plt.ylabel("Eficiencia", fontsize=14)
            plt.xscale('log')  # Escala logarítmica para capital
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            
            # Agregar anotaciones para regiones de eficiencia
            plt.axhspan(0.8, 1.0, alpha=0.2, color='green')
            plt.axhspan(0.5, 0.8, alpha=0.2, color='yellow')
            plt.axhspan(0, 0.5, alpha=0.2, color='red')
            
            plt.text(x_curve[0] * 1.5, 0.9, "ALTA EFICIENCIA", fontsize=10, color='darkgreen')
            plt.text(x_curve[0] * 1.5, 0.65, "MEDIA EFICIENCIA", fontsize=10, color='darkgoldenrod')
            plt.text(x_curve[0] * 1.5, 0.25, "BAJA EFICIENCIA", fontsize=10, color='darkred')
            
            # Añadir información del modelo
            model_info = data.get('model_info', {})
            model_type = model_info.get('model_type', 'Desconocido')
            r_squared = model_info.get('r_squared', 0)
            
            plt.figtext(0.01, 0.01, 
                      f"Modelo: {model_type}\nR²: {r_squared:.4f}", 
                      fontsize=10)
            
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file)
                logger.info(f"Gráfico guardado como '{output_file}'")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generando gráfico: {str(e)}")
    
    def plot_allocation(self, data: Dict[str, Any], output_file: str = None):
        """
        Generar gráfico de asignación de capital.
        
        Args:
            data: Datos de la optimización
            output_file: Ruta para guardar el gráfico (opcional)
        """
        try:
            # Extraer datos de asignación porcentual
            allocations = data.get('percentage_allocation', {})
            
            # Filtrar solo asignaciones positivas y ordenar por valor
            allocations = {k: v for k, v in allocations.items() if v > 0}
            sorted_items = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
            
            if not sorted_items:
                logger.warning("No hay asignaciones positivas para graficar")
                return
            
            # Preparar datos para el gráfico
            labels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            # Crear gráfico
            plt.figure(figsize=(10, 8))
            
            # Crear gráfico de torta
            _, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%',
                                         startangle=90, shadow=False)
            
            # Establecer propiedades del texto
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
            
            # Añadir título y métricas
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title(f"Asignación Óptima de Capital - ${data.get('total_capital', 0):,.0f}", fontsize=14)
            
            # Añadir métricas clave
            metrics_text = (
                f"Eficiencia promedio: {data.get('avg_efficiency', 0):.2f}\n"
                f"Utilización de capital: {data.get('capital_utilization', 0):.1%}\n"
                f"Instrumentos utilizados: {data.get('symbols_used', 0)}/{data.get('symbols_total', 0)}"
            )
            
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file)
                logger.info(f"Gráfico guardado como '{output_file}'")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generando gráfico de asignación: {str(e)}")
    
    def plot_comparative_efficiency(self, predictions: Dict[str, List[Dict[str, Any]]], output_file: str = None):
        """
        Generar gráfico comparativo de eficiencia para múltiples símbolos.
        
        Args:
            predictions: Predicciones por símbolo
            output_file: Ruta para guardar el gráfico (opcional)
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Colores para cada símbolo
            colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown']
            
            # Graficar cada símbolo
            for i, (symbol, prediction_data) in enumerate(predictions.items()):
                capital_values = [p['capital'] for p in prediction_data]
                efficiency_values = [p['efficiency'] for p in prediction_data]
                
                plt.plot(capital_values, efficiency_values, 
                        marker='o', linestyle='-', color=colors[i % len(colors)],
                        label=f"{symbol}")
            
            # Configurar gráfico
            plt.title("Comparativa de Eficiencia vs Capital", fontsize=16)
            plt.xlabel("Capital (USD)", fontsize=14)
            plt.ylabel("Eficiencia", fontsize=14)
            plt.xscale('log')  # Escala logarítmica para capital
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            
            # Agregar anotaciones para regiones de eficiencia
            plt.axhspan(0.8, 1.0, alpha=0.2, color='green')
            plt.axhspan(0.5, 0.8, alpha=0.2, color='yellow')
            plt.axhspan(0, 0.5, alpha=0.2, color='red')
            
            plt.text(capital_values[0] * 1.5, 0.9, "ALTA EFICIENCIA", fontsize=10, color='darkgreen')
            plt.text(capital_values[0] * 1.5, 0.65, "MEDIA EFICIENCIA", fontsize=10, color='darkgoldenrod')
            plt.text(capital_values[0] * 1.5, 0.25, "BAJA EFICIENCIA", fontsize=10, color='darkred')
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file)
                logger.info(f"Gráfico comparativo guardado como '{output_file}'")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generando gráfico comparativo: {str(e)}")


async def main():
    """Función principal para línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Analizador de Escalabilidad de Capital - Herramienta de línea de comandos"
    )
    
    # Opciones globales
    parser.add_argument('--db', help='URL de conexión a base de datos (opcional)')
    parser.add_argument('--cache-size', default='500', help='Tamaño de caché para el motor predictivo')
    parser.add_argument('--format', choices=['json', 'pretty'], default='pretty',
                      help='Formato de salida (json o pretty)')
    parser.add_argument('--plot', action='store_true', help='Generar gráficos')
    parser.add_argument('--output', help='Ruta para guardar resultados y gráficos')
    
    # Subcomandos
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: predecir
    predict_parser = subparsers.add_parser('predecir', help='Predecir eficiencia para un símbolo')
    predict_parser.add_argument('--symbol', required=True, help='Símbolo a analizar')
    predict_parser.add_argument('--capital', nargs='+', type=float, required=True,
                              help='Niveles de capital para predicción')
    
    # Comando: optimizar
    optimize_parser = subparsers.add_parser('optimizar', help='Optimizar asignación de capital')
    optimize_parser.add_argument('--symbols', nargs='+', required=True,
                               help='Lista de símbolos a considerar')
    optimize_parser.add_argument('--total', type=float, required=True,
                               help='Capital total disponible')
    
    # Comando: analizar
    analyze_parser = subparsers.add_parser('analizar', help='Analizar modelo predictivo de un símbolo')
    analyze_parser.add_argument('--symbol', required=True, help='Símbolo a analizar')
    
    # Comando: comparar
    compare_parser = subparsers.add_parser('comparar', help='Comparar eficiencia de múltiples símbolos')
    compare_parser.add_argument('--symbols', nargs='+', required=True,
                              help='Lista de símbolos a comparar')
    compare_parser.add_argument('--capital', nargs='+', type=float, required=True,
                              help='Niveles de capital para comparación')
    
    # Comando: símbolos
    symbols_parser = subparsers.add_parser('simbolos', help='Listar símbolos disponibles')
    
    # Comando: sintético (para demostración)
    synthetic_parser = subparsers.add_parser('sintetico', help='Cargar datos sintéticos para demostración')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Inicializar analizador
    analyzer = CapitalScalingAnalyzer(db_url=args.db, cache_size=args.cache_size)
    
    try:
        # Inicializar componentes
        success = await analyzer.initialize()
        if not success:
            logger.error("Falló la inicialización del analizador")
            return 1
        
        # Ejecutar comando
        result = None
        
        if args.command == 'predecir':
            result = await analyzer.predict_efficiency(args.symbol, args.capital)
            if args.plot:
                # Convertir resultado a formato esperado por plot_efficiency_curve
                model_data = await analyzer.analyze_model(args.symbol)
                analyzer.plot_efficiency_curve(model_data, 
                                             output_file=f"{args.output or 'efficiency'}_{args.symbol}.png")
        
        elif args.command == 'optimizar':
            result = await analyzer.optimize_allocation(args.symbols, args.total)
            if args.plot:
                analyzer.plot_allocation(result, 
                                       output_file=f"{args.output or 'allocation'}_capital_{args.total:.0f}.png")
        
        elif args.command == 'analizar':
            result = await analyzer.analyze_model(args.symbol)
            if args.plot:
                analyzer.plot_efficiency_curve(result, 
                                             output_file=f"{args.output or 'model'}_{args.symbol}.png")
        
        elif args.command == 'comparar':
            # Recolectar predicciones para cada símbolo
            predictions = {}
            for symbol in args.symbols:
                prediction_result = await analyzer.predict_efficiency(symbol, args.capital)
                predictions[symbol] = prediction_result.get('predictions', [])
            
            result = {
                "comparative_predictions": predictions,
                "capital_levels": args.capital,
                "symbols": args.symbols
            }
            
            if args.plot:
                analyzer.plot_comparative_efficiency(predictions, 
                                                  output_file=f"{args.output or 'compare'}_symbols.png")
        
        elif args.command == 'simbolos':
            symbols = await analyzer.get_available_symbols()
            result = {"available_symbols": symbols, "count": len(symbols)}
        
        elif args.command == 'sintetico':
            await analyzer.load_synthetic_data()
            result = {"status": "success", "message": "Datos sintéticos cargados correctamente"}
        
        else:
            logger.error(f"Comando desconocido: {args.command}")
            return 1
        
        # Imprimir resultado
        if result:
            if args.format == 'json':
                print(json.dumps(result, cls=ScalingJSONEncoder, indent=2))
            else:
                # Formato más bonito para humanos
                print("\n" + "="*80)
                print(f"RESULTADO: {args.command.upper()}")
                print("="*80)
                
                if args.command == 'predecir':
                    print(f"\nSímbolo: {result['symbol']}")
                    if 'model_info' in result:
                        model = result['model_info']
                        print(f"Modelo: {model.get('model_type', 'N/A')}, R²: {model.get('r_squared', 0):.4f}")
                        if 'saturation_point' in model and model['saturation_point']:
                            print(f"Punto de saturación: ${model['saturation_point']:,.2f}")
                    
                    print("\nPredicciones:")
                    print(f"{'Capital':>12} | {'Eficiencia':>10} | {'Confianza':>10}")
                    print("-"*38)
                    for p in result.get('predictions', []):
                        print(f"${p['capital']:>10,.0f} | {p['efficiency']:>9.2f} | {p['confidence']:>9.2f}")
                
                elif args.command == 'optimizar':
                    print(f"\nCapital total: ${result['total_capital']:,.2f}")
                    print(f"Eficiencia promedio: {result.get('avg_efficiency', 0):.4f}")
                    print(f"Utilización de capital: {result.get('capital_utilization', 0):.1%}")
                    print(f"Instrumentos utilizados: {result.get('symbols_used', 0)}/{result.get('symbols_total', 0)}")
                    
                    print("\nAsignación óptima:")
                    print(f"{'Símbolo':^10} | {'Monto ($)':>12} | {'Porcentaje':>10} | {'Eficiencia':>10}")
                    print("-"*52)
                    
                    # Ordenar por monto, de mayor a menor
                    sorted_items = sorted(result.get('allocations', {}).items(), 
                                         key=lambda x: x[1], reverse=True)
                    
                    for symbol, amount in sorted_items:
                        if amount > 0:
                            percentage = result['percentage_allocation'].get(symbol, 0)
                            efficiency = result['efficiency_by_symbol'].get(symbol, 0)
                            print(f"{symbol:^10} | ${amount:>10,.0f} | {percentage:>9.1f}% | {efficiency:>9.2f}")
                
                elif args.command == 'analizar':
                    print(f"\nAnálisis de modelo para: {result['symbol']}")
                    
                    if 'model_info' in result:
                        model = result['model_info']
                        print(f"\nTipo de modelo: {model.get('model_type', 'N/A')}")
                        print(f"R²: {model.get('r_squared', 0):.4f}")
                        print(f"Muestras: {model.get('samples_count', 'N/A')}")
                        
                        if 'parameters' in model:
                            print("\nParámetros del modelo:")
                            for k, v in model['parameters'].items():
                                print(f"  {k}: {v}")
                        
                        if 'saturation_point' in model and model['saturation_point']:
                            print(f"\nPunto de saturación: ${model['saturation_point']:,.2f}")
                    
                    if 'historical_points' in result and result['historical_points']:
                        print("\nDatos históricos:")
                        print(f"{'Capital':>12} | {'Eficiencia':>10}")
                        print("-"*25)
                        for p in result['historical_points'][:10]:  # Mostrar solo los primeros 10
                            print(f"${p['capital']:>10,.0f} | {p['efficiency']:>9.2f}")
                        
                        if len(result['historical_points']) > 10:
                            print(f"... (y {len(result['historical_points']) - 10} puntos más)")
                
                elif args.command == 'simbolos':
                    print(f"\nSímbolos disponibles ({result['count']}):")
                    for i, symbol in enumerate(result['available_symbols']):
                        print(f"{i+1:3d}. {symbol}")
                
                elif args.command == 'comparar':
                    print(f"\nComparación de eficiencia para {len(args.symbols)} símbolos:")
                    print(f"{' ':^10} | " + " | ".join([f"{s:^10}" for s in args.symbols]))
                    print("-" * (12 + 14 * len(args.symbols)))
                    
                    predictions = result.get('comparative_predictions', {})
                    for i, capital in enumerate(args.capital):
                        row = [f"${capital:>8,.0f}"]
                        for symbol in args.symbols:
                            if symbol in predictions and i < len(predictions[symbol]):
                                efficiency = predictions[symbol][i]['efficiency']
                                row.append(f"{efficiency:>9.2f}")
                            else:
                                row.append(f"{'N/A':>9}")
                        print(" | ".join(row))
                
                elif args.command == 'sintetico':
                    print(f"\n{result['message']}")
                
                print("\n" + "="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error ejecutando comando: {str(e)}")
        return 1
    
    finally:
        # Limpiar recursos
        await analyzer.cleanup()
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)