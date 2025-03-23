"""
Gestor de simulaciones Monte Carlo para el Sistema Genesis.

Este módulo proporciona una interfaz para integrar las simulaciones Monte Carlo
con el resto de componentes del sistema, permitiendo análisis de riesgo,
evaluación de estrategias y optimización de parámetros.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

from genesis.simulation.monte_carlo import MonteCarloSimulator
from genesis.db.transcendental_database import TranscendentalDatabase

class MonteCarloManager:
    """
    Gestor de simulaciones Monte Carlo para el Sistema Genesis.
    
    Proporciona métodos para ejecutar simulaciones Monte Carlo, calcular VaR,
    optimizar tamaños de posición y evaluar estrategias con datos históricos.
    """
    
    def __init__(self, 
                 db: Optional[TranscendentalDatabase] = None,
                 log_returns: bool = True,
                 random_seed: Optional[int] = None,
                 num_cores: int = 4,
                 cache_results: bool = True,
                 cache_dir: str = './cache/monte_carlo'):
        """
        Inicializar gestor de simulaciones Monte Carlo.
        
        Args:
            db: Instancia de base de datos para persistencia (opcional)
            log_returns: Si es True, usa retornos logarítmicos para simulación (más preciso)
            random_seed: Semilla para el generador de números aleatorios (opcional)
            num_cores: Número de núcleos para procesamiento paralelo
            cache_results: Si es True, guarda resultados en caché para reutilización
            cache_dir: Directorio para la caché de resultados
        """
        self.logger = logging.getLogger(__name__)
        self.db = db
        self.cache_results = cache_results
        self.cache_dir = cache_dir
        
        # Crear directorio de caché si no existe
        if cache_results:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Inicializar simulador
        self.simulator = MonteCarloSimulator(
            log_returns=log_returns,
            random_seed=random_seed,
            num_cores=num_cores
        )
        
        # Caché en memoria
        self.result_cache = {}
        
        self.logger.info(f"Gestor de Monte Carlo inicializado con {num_cores} núcleos")
    
    async def analyze_risk(self, 
                     symbol: str,
                     price_data: pd.DataFrame,
                     position_size: float,
                     confidence_level: float = 0.95,
                     time_horizon: int = 1) -> Dict[str, Any]:
        """
        Analizar riesgo de una posición mediante VaR y simulaciones Monte Carlo.
        
        Args:
            symbol: Símbolo del activo
            price_data: DataFrame con precios históricos
            position_size: Tamaño de la posición
            confidence_level: Nivel de confianza para el VaR (0.95 = 95%)
            time_horizon: Horizonte temporal en días
            
        Returns:
            Diccionario con análisis de riesgo
        """
        self.logger.info(f"Analizando riesgo para {symbol} con posición de {position_size}")
        
        # Calcular VaR
        var_result = await self.simulator.calculate_var(
            price_data=price_data,
            position_size=position_size,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            num_simulations=10000
        )
        
        # Calcular pronóstico de precios
        price_forecast = await self.simulator.simulate_price_paths(
            price_data=price_data,
            num_simulations=1000,
            prediction_days=time_horizon
        )
        
        # Combinar resultados
        risk_analysis = {
            'symbol': symbol,
            'position_size': position_size,
            'position_value': var_result['position_value'],
            'var': {
                'absolute': var_result['var_absolute'],
                'percentage': var_result['var_percentage'],
                'confidence_level': var_result['confidence_level'],
                'time_horizon': var_result['time_horizon']
            },
            'cvar': {
                'absolute': var_result['cvar_absolute'],
                'percentage': var_result['cvar_percentage']
            },
            'price_forecast': {
                'initial_price': price_forecast['initial_price'],
                'mean_price': price_forecast['price_stats']['mean'],
                'min_price': price_forecast['price_stats']['min'],
                'max_price': price_forecast['price_stats']['max'],
                'prob_increase': price_forecast['prob_increase']
            },
            'var_histogram': var_result['histogram_base64'],
            'timestamp': int(time.time())
        }
        
        # Guardar en caché
        if self.cache_results:
            cache_key = f"{symbol}_risk_{confidence_level}_{time_horizon}"
            self.result_cache[cache_key] = risk_analysis
            
            # Guardar en archivo
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            with open(cache_file, 'w') as f:
                # No guardar imágenes en el archivo para ahorrar espacio
                risk_analysis_no_images = {k: v for k, v in risk_analysis.items() if k != 'var_histogram'}
                json.dump(risk_analysis_no_images, f)
            
            self.logger.info(f"Resultado guardado en caché: {cache_file}")
        
        # Guardar en base de datos si está disponible
        if self.db:
            try:
                await self.db.store('risk_analysis', {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'position_size': position_size,
                    'var_absolute': var_result['var_absolute'],
                    'var_percentage': var_result['var_percentage'],
                    'cvar_absolute': var_result['cvar_absolute'],
                    'cvar_percentage': var_result['cvar_percentage'],
                    'confidence_level': confidence_level,
                    'time_horizon': time_horizon,
                    'prob_increase': price_forecast['prob_increase']
                })
                self.logger.info(f"Análisis de riesgo guardado en base de datos para {symbol}")
            except Exception as e:
                self.logger.error(f"Error guardando análisis en base de datos: {str(e)}")
        
        return risk_analysis
    
    async def optimize_portfolio_allocation(self,
                                      symbols: List[str],
                                      price_data_dict: Dict[str, pd.DataFrame],
                                      total_capital: float,
                                      max_iterations: int = 1000,
                                      target_risk: float = 0.05,
                                      risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Optimizar la asignación de capital entre múltiples activos usando Monte Carlo.
        
        Args:
            symbols: Lista de símbolos en el portafolio
            price_data_dict: Diccionario con DataFrames de precios por símbolo
            total_capital: Capital total a asignar
            max_iterations: Número máximo de iteraciones para la optimización
            target_risk: Volatilidad objetivo del portafolio (anualizada)
            risk_free_rate: Tasa libre de riesgo anualizada
            
        Returns:
            Diccionario con la asignación óptima y métricas
        """
        self.logger.info(f"Optimizando asignación para portafolio de {len(symbols)} activos")
        
        # Calcular retornos diarios para cada activo
        returns_dict = {}
        for symbol, df in price_data_dict.items():
            if 'close' not in df.columns:
                raise ValueError(f"DataFrame para {symbol} debe contener columna 'close'")
            
            # Calcular retornos logarítmicos
            returns_dict[symbol] = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        # Asegurar que todos los retornos tengan la misma longitud
        min_length = min(len(returns) for returns in returns_dict.values())
        aligned_returns = {}
        for symbol, returns in returns_dict.items():
            aligned_returns[symbol] = returns.iloc[-min_length:].values
        
        # Crear matriz de retornos
        returns_matrix = np.column_stack([aligned_returns[symbol] for symbol in symbols])
        
        # Calcular matriz de covarianza y vector de retornos esperados
        cov_matrix = np.cov(returns_matrix.T)
        expected_returns = np.mean(returns_matrix, axis=0)
        
        # Simulación de Monte Carlo para encontrar la asignación óptima
        np.random.seed(42)  # Para reproducibilidad
        
        n_assets = len(symbols)
        best_sharpe = -np.inf
        best_allocation = np.ones(n_assets) / n_assets  # Asignación uniforme inicial
        
        results = []
        
        for i in range(max_iterations):
            # Generar asignación aleatoria
            allocation = np.random.random(n_assets)
            allocation = allocation / np.sum(allocation)
            
            # Calcular retorno esperado del portafolio
            portfolio_return = np.sum(expected_returns * allocation) * 252  # Anualizado
            
            # Calcular riesgo del portafolio
            portfolio_risk = np.sqrt(np.dot(allocation.T, np.dot(cov_matrix, allocation))) * np.sqrt(252)  # Anualizado
            
            # Calcular Sharpe Ratio
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
            
            # Guardar resultado
            results.append({
                'allocation': allocation,
                'return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe': sharpe_ratio
            })
            
            # Actualizar mejor asignación
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_allocation = allocation
        
        # Ordenar resultados por Sharpe Ratio
        results.sort(key=lambda x: x['sharpe'], reverse=True)
        
        # Preparar asignaciones monetarias
        best_allocation_dict = {}
        allocation_amounts = {}
        
        for i, symbol in enumerate(symbols):
            best_allocation_dict[symbol] = float(best_allocation[i])
            allocation_amounts[symbol] = float(best_allocation[i] * total_capital)
        
        # Calcular estadísticas del mejor portafolio
        best_return = float(np.sum(expected_returns * best_allocation) * 252)
        best_risk = float(np.sqrt(np.dot(best_allocation.T, np.dot(cov_matrix, best_allocation))) * np.sqrt(252))
        
        # Preparar resultado
        optimization_result = {
            'allocations': best_allocation_dict,
            'allocation_amounts': allocation_amounts,
            'portfolio_metrics': {
                'expected_return': best_return,
                'risk': best_risk,
                'sharpe_ratio': float(best_sharpe)
            },
            'simulation_params': {
                'iterations': max_iterations,
                'target_risk': target_risk,
                'risk_free_rate': risk_free_rate,
                'total_capital': total_capital
            },
            'efficient_frontier': [
                {
                    'risk': float(result['risk']),
                    'return': float(result['return']),
                    'sharpe': float(result['sharpe'])
                }
                for result in results[:100]  # Top 100 resultados para la frontera eficiente
            ],
            'timestamp': int(time.time())
        }
        
        # Guardar en base de datos si está disponible
        if self.db:
            try:
                await self.db.store('portfolio_optimization', {
                    'symbols': symbols,
                    'timestamp': datetime.now().isoformat(),
                    'total_capital': total_capital,
                    'allocations': best_allocation_dict,
                    'expected_return': best_return,
                    'risk': best_risk,
                    'sharpe_ratio': float(best_sharpe)
                })
                self.logger.info(f"Optimización de portafolio guardada en base de datos")
            except Exception as e:
                self.logger.error(f"Error guardando optimización en base de datos: {str(e)}")
        
        return optimization_result
    
    async def evaluate_strategy(self,
                          symbol: str,
                          price_data: pd.DataFrame,
                          strategy_func: Callable,
                          initial_capital: float = 10000.0,
                          position_size_pct: float = 0.1,
                          strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluar una estrategia con backtesting y simulaciones Monte Carlo.
        
        Args:
            symbol: Símbolo del activo
            price_data: DataFrame con datos OHLCV
            strategy_func: Función que implementa la estrategia
            initial_capital: Capital inicial
            position_size_pct: Porcentaje del capital por posición
            strategy_params: Parámetros adicionales para la estrategia
            
        Returns:
            Diccionario con resultados de la evaluación
        """
        self.logger.info(f"Evaluando estrategia para {symbol}")
        
        # Realizar backtest con Monte Carlo
        backtest_result = await self.simulator.backtest_with_monte_carlo(
            price_data=price_data,
            strategy_func=strategy_func,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            num_simulations=100,
            strategy_params=strategy_params
        )
        
        # Calcular criterio de Kelly para optimizar tamaño de posición
        kelly_result = await self.simulator.calculate_kelly_criterion(backtest_result)
        
        # Preparar resultado completo
        evaluation_result = {
            'symbol': symbol,
            'backtest': backtest_result['historical'],
            'monte_carlo': backtest_result['monte_carlo'],
            'kelly_criterion': kelly_result,
            'recommended_position': {
                'percentage': float(kelly_result.get('kelly_quarter', 0.025)),
                'amount': float(kelly_result.get('kelly_quarter', 0.025) * initial_capital)
            },
            'charts': {
                'backtest': backtest_result['chart_base64']
            },
            'timestamp': int(time.time())
        }
        
        # Guardar en base de datos si está disponible
        if self.db:
            try:
                await self.db.store('strategy_evaluation', {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'initial_capital': initial_capital,
                    'total_return': backtest_result['historical']['total_return'],
                    'max_drawdown': backtest_result['historical']['max_drawdown'],
                    'sharpe_ratio': backtest_result['historical']['sharpe_ratio'],
                    'sortino_ratio': backtest_result['historical']['sortino_ratio'],
                    'kelly_full': kelly_result.get('kelly_full'),
                    'kelly_half': kelly_result.get('kelly_half'),
                    'kelly_quarter': kelly_result.get('kelly_quarter'),
                    'probability_of_win': kelly_result.get('probability_of_win'),
                    'strategy_params': strategy_params
                })
                self.logger.info(f"Evaluación de estrategia guardada en base de datos para {symbol}")
            except Exception as e:
                self.logger.error(f"Error guardando evaluación en base de datos: {str(e)}")
        
        return evaluation_result
    
    async def optimize_strategy_parameters(self,
                                     symbol: str,
                                     price_data: pd.DataFrame,
                                     strategy_func: Callable,
                                     param_grid: Dict[str, List[Any]],
                                     initial_capital: float = 10000.0,
                                     position_size_pct: float = 0.1,
                                     max_combinations: int = 50,
                                     optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimizar los parámetros de una estrategia mediante búsqueda de cuadrícula (grid search).
        
        Args:
            symbol: Símbolo del activo
            price_data: DataFrame con datos OHLCV
            strategy_func: Función que implementa la estrategia
            param_grid: Diccionario con parámetros a probar y sus posibles valores
            initial_capital: Capital inicial
            position_size_pct: Porcentaje del capital por posición
            max_combinations: Número máximo de combinaciones a probar
            optimization_metric: Métrica a optimizar ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Diccionario con los mejores parámetros y resultados
        """
        self.logger.info(f"Optimizando parámetros para estrategia en {symbol}")
        
        # Generar todas las combinaciones de parámetros
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Calcular número total de combinaciones
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        # Limitar número de combinaciones si es necesario
        if total_combinations > max_combinations:
            self.logger.warning(f"Número de combinaciones ({total_combinations}) excede el máximo ({max_combinations}). Se reducirá aleatoriamente.")
            
            # Reducir tamaño de los valores
            for i, values in enumerate(param_values):
                if len(values) > 3 and total_combinations > max_combinations:
                    # Reducir este parámetro
                    new_size = max(3, len(values) // 2)
                    indices = np.linspace(0, len(values) - 1, new_size, dtype=int)
                    param_values[i] = [values[idx] for idx in indices]
                    
                    # Recalcular total
                    total_combinations = 1
                    for values in param_values:
                        total_combinations *= len(values)
        
        # Generar lista de combinaciones a probar
        param_combinations = []
        
        # Función recursiva para generar combinaciones
        def generate_combinations(current_idx=0, current_params={}):
            if current_idx == len(param_keys):
                param_combinations.append(current_params.copy())
                return
            
            key = param_keys[current_idx]
            for value in param_values[current_idx]:
                current_params[key] = value
                generate_combinations(current_idx + 1, current_params)
        
        generate_combinations()
        
        # Limitar aleatoriamente si sigue siendo demasiado grande
        if len(param_combinations) > max_combinations:
            self.logger.warning(f"Seleccionando {max_combinations} combinaciones aleatorias de {len(param_combinations)}")
            indices = np.random.choice(len(param_combinations), max_combinations, replace=False)
            param_combinations = [param_combinations[i] for i in indices]
        
        self.logger.info(f"Probando {len(param_combinations)} combinaciones de parámetros")
        
        # Evaluar cada combinación
        results = []
        
        for i, params in enumerate(param_combinations):
            self.logger.info(f"Probando combinación {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Evaluar estrategia con estos parámetros
                backtest_result = await self.simulator.backtest_with_monte_carlo(
                    price_data=price_data,
                    strategy_func=strategy_func,
                    initial_capital=initial_capital,
                    position_size_pct=position_size_pct,
                    num_simulations=50,  # Menos simulaciones para optimización
                    strategy_params=params
                )
                
                # Extraer métrica a optimizar
                if optimization_metric == 'sharpe_ratio':
                    metric_value = backtest_result['historical']['sharpe_ratio']
                elif optimization_metric == 'sortino_ratio':
                    metric_value = backtest_result['historical']['sortino_ratio']
                elif optimization_metric == 'total_return':
                    metric_value = backtest_result['historical']['total_return']
                elif optimization_metric == 'max_drawdown':
                    # Para drawdown, menor es mejor (negamos para ordenar)
                    metric_value = -backtest_result['historical']['max_drawdown']
                else:
                    # Métrica por defecto: Sharpe
                    metric_value = backtest_result['historical']['sharpe_ratio']
                
                # Almacenar resultado
                results.append({
                    'params': params,
                    'metric_value': metric_value,
                    'backtest_result': {
                        'total_return': backtest_result['historical']['total_return'],
                        'max_drawdown': backtest_result['historical']['max_drawdown'],
                        'sharpe_ratio': backtest_result['historical']['sharpe_ratio'],
                        'sortino_ratio': backtest_result['historical']['sortino_ratio']
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Error evaluando combinación {i+1}: {str(e)}")
        
        # Ordenar resultados por la métrica de optimización
        results.sort(key=lambda x: x['metric_value'], reverse=True)
        
        # Extraer mejores parámetros
        best_params = results[0]['params'] if results else {}
        
        # Preparar resultado
        optimization_result = {
            'symbol': symbol,
            'best_params': best_params,
            'best_metrics': results[0]['backtest_result'] if results else {},
            'optimization_metric': optimization_metric,
            'all_results': [
                {
                    'params': r['params'],
                    'metrics': r['backtest_result'],
                    'metric_value': float(r['metric_value'])
                }
                for r in results
            ],
            'param_grid': param_grid,
            'combinations_tested': len(param_combinations),
            'timestamp': int(time.time())
        }
        
        # Guardar en base de datos si está disponible
        if self.db and results:
            try:
                await self.db.store('strategy_optimization', {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'best_params': best_params,
                    'optimization_metric': optimization_metric,
                    'best_metrics': results[0]['backtest_result'],
                    'combinations_tested': len(param_combinations),
                })
                self.logger.info(f"Optimización de parámetros guardada en base de datos para {symbol}")
            except Exception as e:
                self.logger.error(f"Error guardando optimización en base de datos: {str(e)}")
        
        return optimization_result
    
    async def stress_test_strategy(self,
                             symbol: str,
                             price_data: pd.DataFrame,
                             strategy_func: Callable,
                             strategy_params: Dict[str, Any],
                             initial_capital: float = 10000.0,
                             position_size_pct: float = 0.1,
                             num_scenarios: int = 5,
                             stress_scenarios: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Realizar stress testing de una estrategia bajo escenarios extremos.
        
        Args:
            symbol: Símbolo del activo
            price_data: DataFrame con datos OHLCV
            strategy_func: Función que implementa la estrategia
            strategy_params: Parámetros para la estrategia
            initial_capital: Capital inicial
            position_size_pct: Porcentaje del capital por posición
            num_scenarios: Número de escenarios a generar (si no se proporcionan)
            stress_scenarios: Diccionario con escenarios personalizados (opcional)
            
        Returns:
            Diccionario con resultados del stress testing
        """
        self.logger.info(f"Realizando stress testing para estrategia en {symbol}")
        
        # Si no se proporcionan escenarios, generar algunos típicos
        if not stress_scenarios:
            stress_scenarios = {
                "crash_50_percent": {
                    "description": "Caída del 50% en 5 días",
                    "transform": lambda df: self._transform_crash(df, percent=0.5, days=5)
                },
                "crash_20_percent": {
                    "description": "Caída del 20% en 1 día",
                    "transform": lambda df: self._transform_crash(df, percent=0.2, days=1)
                },
                "high_volatility": {
                    "description": "Volatilidad 3x durante 20 días",
                    "transform": lambda df: self._transform_volatility(df, multiplier=3.0, days=20)
                },
                "whipsaw": {
                    "description": "Movimientos bruscos en direcciones opuestas",
                    "transform": lambda df: self._transform_whipsaw(df, intensity=0.15, days=10)
                },
                "no_volume": {
                    "description": "Volumen extremadamente bajo",
                    "transform": lambda df: self._transform_volume(df, multiplier=0.1, days=15)
                }
            }
        
        # Resultados por escenario
        scenario_results = {}
        
        # Evaluar cada escenario
        for scenario_name, scenario_config in stress_scenarios.items():
            self.logger.info(f"Probando escenario: {scenario_name} - {scenario_config['description']}")
            
            try:
                # Aplicar transformación al dataframe
                scenario_data = scenario_config['transform'](price_data.copy())
                
                # Evaluar estrategia bajo este escenario
                backtest_result = await self.simulator.backtest_with_monte_carlo(
                    price_data=scenario_data,
                    strategy_func=strategy_func,
                    initial_capital=initial_capital,
                    position_size_pct=position_size_pct,
                    num_simulations=50,
                    strategy_params=strategy_params
                )
                
                # Guardar resultados
                scenario_results[scenario_name] = {
                    'description': scenario_config['description'],
                    'metrics': {
                        'total_return': backtest_result['historical']['total_return'],
                        'max_drawdown': backtest_result['historical']['max_drawdown'],
                        'sharpe_ratio': backtest_result['historical']['sharpe_ratio'],
                        'sortino_ratio': backtest_result['historical']['sortino_ratio']
                    },
                    'monte_carlo': {
                        'prob_profit': backtest_result['monte_carlo']['prob_profit'],
                        'return_percentiles': backtest_result['monte_carlo']['return_percentiles'],
                        'drawdown_percentiles': backtest_result['monte_carlo']['drawdown_percentiles']
                    },
                    'chart_base64': backtest_result.get('chart_base64')
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluando escenario {scenario_name}: {str(e)}")
                scenario_results[scenario_name] = {
                    'description': scenario_config['description'],
                    'error': str(e)
                }
        
        # Preparar resultado completo
        stress_test_result = {
            'symbol': symbol,
            'strategy_params': strategy_params,
            'initial_capital': initial_capital,
            'position_size_pct': position_size_pct,
            'scenarios': scenario_results,
            'timestamp': int(time.time())
        }
        
        # Guardar en base de datos si está disponible
        if self.db:
            try:
                # Solo guardar métricas principales
                await self.db.store('stress_test_results', {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'strategy_params': strategy_params,
                    'scenarios': {
                        name: {
                            'description': data['description'],
                            'total_return': data.get('metrics', {}).get('total_return'),
                            'max_drawdown': data.get('metrics', {}).get('max_drawdown')
                        }
                        for name, data in scenario_results.items()
                    }
                })
                self.logger.info(f"Resultados de stress testing guardados en base de datos para {symbol}")
            except Exception as e:
                self.logger.error(f"Error guardando stress test en base de datos: {str(e)}")
        
        return stress_test_result
    
    def _transform_crash(self, df: pd.DataFrame, percent: float = 0.5, days: int = 5) -> pd.DataFrame:
        """Transformar precios para simular un crash."""
        # Crear copia del dataframe
        transformed_df = df.copy()
        
        # Calcular factor por día
        daily_factor = (1 - percent) ** (1/days)
        
        # Aplicar crash a los últimos 'days' filas
        for i in range(1, days + 1):
            idx = len(transformed_df) - i
            if idx >= 0:
                transformed_df.loc[transformed_df.index[idx], 'open'] *= daily_factor ** i
                transformed_df.loc[transformed_df.index[idx], 'high'] *= daily_factor ** i
                transformed_df.loc[transformed_df.index[idx], 'low'] *= daily_factor ** i
                transformed_df.loc[transformed_df.index[idx], 'close'] *= daily_factor ** i
        
        return transformed_df
    
    def _transform_volatility(self, df: pd.DataFrame, multiplier: float = 3.0, days: int = 20) -> pd.DataFrame:
        """Transformar precios para simular alta volatilidad."""
        # Crear copia del dataframe
        transformed_df = df.copy()
        
        # Aplicar volatilidad aumentada a los últimos 'days' filas
        for i in range(days):
            idx = len(transformed_df) - i - 1
            if idx >= 0:
                # Calcular rango de precios ampliado
                close = transformed_df.loc[transformed_df.index[idx], 'close']
                open_price = transformed_df.loc[transformed_df.index[idx], 'open']
                
                # Ajustar high y low
                current_range = transformed_df.loc[transformed_df.index[idx], 'high'] - transformed_df.loc[transformed_df.index[idx], 'low']
                new_range = current_range * multiplier
                
                mid_price = (close + open_price) / 2
                transformed_df.loc[transformed_df.index[idx], 'high'] = mid_price + new_range/2
                transformed_df.loc[transformed_df.index[idx], 'low'] = mid_price - new_range/2
        
        return transformed_df
    
    def _transform_whipsaw(self, df: pd.DataFrame, intensity: float = 0.15, days: int = 10) -> pd.DataFrame:
        """Transformar precios para simular movimientos bruscos en direcciones opuestas."""
        # Crear copia del dataframe
        transformed_df = df.copy()
        
        # Aplicar movimientos de whipsaw a los últimos 'days' filas
        for i in range(days):
            idx = len(transformed_df) - i - 1
            if idx >= 0:
                # Alternar entre movimientos hacia arriba y hacia abajo
                direction = 1 if i % 2 == 0 else -1
                factor = 1 + (direction * intensity)
                
                transformed_df.loc[transformed_df.index[idx], 'open'] *= factor
                transformed_df.loc[transformed_df.index[idx], 'high'] *= factor
                transformed_df.loc[transformed_df.index[idx], 'low'] *= factor
                transformed_df.loc[transformed_df.index[idx], 'close'] *= factor
        
        return transformed_df
    
    def _transform_volume(self, df: pd.DataFrame, multiplier: float = 0.1, days: int = 15) -> pd.DataFrame:
        """Transformar volumen para simular baja liquidez."""
        # Crear copia del dataframe
        transformed_df = df.copy()
        
        # Aplicar volumen reducido a los últimos 'days' filas
        for i in range(days):
            idx = len(transformed_df) - i - 1
            if idx >= 0 and 'volume' in transformed_df.columns:
                transformed_df.loc[transformed_df.index[idx], 'volume'] *= multiplier
        
        return transformed_df