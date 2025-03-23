"""
Simulaciones Monte Carlo para evaluación de estrategias y análisis de riesgo.

Este módulo implementa simulaciones Monte Carlo para generar múltiples escenarios
de precios y evaluar el rendimiento y riesgo de estrategias de trading bajo
diferentes condiciones de mercado.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class MonteCarloSimulator:
    """
    Simulador de Monte Carlo para análisis de estrategias y riesgo.
    
    Permite generar escenarios de precios basados en datos históricos
    y evaluar estrategias bajo diferentes condiciones.
    """
    
    def __init__(self, 
                 log_returns: bool = True,
                 random_seed: Optional[int] = None,
                 num_cores: int = 4):
        """
        Inicializar simulador de Monte Carlo.
        
        Args:
            log_returns: Si es True, usa retornos logarítmicos para simulación (más preciso)
            random_seed: Semilla para el generador de números aleatorios (opcional)
            num_cores: Número de núcleos para procesamiento paralelo
        """
        self.logger = logging.getLogger(__name__)
        self.log_returns = log_returns
        self.executor = ThreadPoolExecutor(max_workers=num_cores)
        self.last_simulation = {}
        
        # Inicializar generador aleatorio
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.logger.info(f"Simulador Monte Carlo inicializado con {num_cores} núcleos")
    
    async def simulate_price_paths(self, 
                             price_data: pd.DataFrame, 
                             num_simulations: int = 1000,
                             prediction_days: int = 30,
                             percentiles: List[int] = [5, 25, 50, 75, 95]) -> Dict[str, Any]:
        """
        Simular múltiples trayectorias de precios basadas en datos históricos.
        
        Args:
            price_data: DataFrame con precios históricos (debe incluir columna 'close')
            num_simulations: Número de simulaciones a generar
            prediction_days: Número de días a proyectar en el futuro
            percentiles: Percentiles a calcular para los caminos de precios
            
        Returns:
            Diccionario con resultados de la simulación
        """
        self.logger.info(f"Iniciando simulación Monte Carlo con {num_simulations} escenarios")
        start_time = time.time()
        
        # Ejecutar en un thread aparte para evitar bloquear el asyncio
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._simulate_price_paths_sync,
                price_data,
                num_simulations,
                prediction_days,
                percentiles
            )
            
            self.last_simulation = result
            
            duration = time.time() - start_time
            self.logger.info(f"Simulación Monte Carlo completada en {duration:.2f} segundos")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en simulación Monte Carlo: {str(e)}")
            raise
    
    def _simulate_price_paths_sync(self, 
                              price_data: pd.DataFrame,
                              num_simulations: int,
                              prediction_days: int,
                              percentiles: List[int]) -> Dict[str, Any]:
        """
        Versión sincrónica de simulate_price_paths para ejecutar en un thread.
        
        Args:
            price_data: DataFrame con precios históricos
            num_simulations: Número de simulaciones
            prediction_days: Días a proyectar
            percentiles: Percentiles a calcular
            
        Returns:
            Diccionario con resultados
        """
        # Validar datos de entrada
        if 'close' not in price_data.columns:
            raise ValueError("price_data debe contener una columna 'close' con precios de cierre")
        
        if price_data.empty:
            raise ValueError("No hay datos de precios para simular")
        
        if not isinstance(price_data.index[0], (datetime, np.datetime64, pd.Timestamp)):
            self.logger.warning("El índice no es de tipo fecha; se asumirá que son días consecutivos")
            
        # Extraer precios de cierre
        close_prices = price_data['close'].values
        last_price = close_prices[-1]
        
        # Calcular retornos diarios
        if self.log_returns:
            # Retornos logarítmicos para distribución más realista
            returns = np.log(close_prices[1:] / close_prices[:-1])
        else:
            # Retornos porcentuales simples
            returns = close_prices[1:] / close_prices[:-1] - 1
        
        # Estadísticas de los retornos
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generar caminos de precios
        simulation_df = pd.DataFrame()
        simulation_df['actual'] = price_data['close'].values
        
        # Matriz para almacenar todos los caminos simulados
        price_paths = np.zeros((prediction_days, num_simulations))
        
        # Precio inicial para todas las simulaciones
        price_paths[0, :] = last_price
        
        # Simular caminos futuros
        for i in range(1, prediction_days):
            if self.log_returns:
                # Muestrear de la distribución normal para retornos log
                random_returns = np.random.normal(mean_return, std_return, num_simulations)
                # Calcular nuevos precios con retornos logarítmicos
                price_paths[i, :] = price_paths[i-1, :] * np.exp(random_returns)
            else:
                # Muestrear de la distribución normal para retornos simples
                random_returns = np.random.normal(mean_return, std_return, num_simulations)
                # Calcular nuevos precios con retornos porcentuales
                price_paths[i, :] = price_paths[i-1, :] * (1 + random_returns)
        
        # Calcular percentiles
        percentile_paths = {}
        for p in percentiles:
            percentile_paths[f'p{p}'] = np.percentile(price_paths, p, axis=1)
        
        # Construir fechas futuras
        last_date = price_data.index[-1]
        if isinstance(last_date, (datetime, np.datetime64, pd.Timestamp)):
            future_dates = [last_date + timedelta(days=i) for i in range(prediction_days)]
        else:
            # Si no son fechas, usar enteros incrementales
            last_idx = len(price_data)
            future_dates = [last_idx + i for i in range(prediction_days)]
        
        # Crear DataFrame con resultados
        future_df = pd.DataFrame(index=future_dates)
        
        # Añadir percentiles al DataFrame
        for p, values in percentile_paths.items():
            future_df[p] = values
        
        # Calcular estadísticas adicionales
        final_prices = price_paths[-1, :]
        price_stats = {
            'mean': np.mean(final_prices),
            'median': np.median(final_prices),
            'min': np.min(final_prices),
            'max': np.max(final_prices),
            'std': np.std(final_prices)
        }
        
        # Calcular probabilidad de que el precio suba/baje
        prob_increase = np.sum(final_prices > last_price) / num_simulations
        
        # Almacenar algunos caminos para visualización
        num_paths_to_store = min(200, num_simulations)  # Almacenar máximo 200 caminos
        stored_paths = price_paths[:, :num_paths_to_store]
        
        # Resultados finales
        results = {
            'initial_price': last_price,
            'price_paths': stored_paths,
            'future_df': future_df,
            'price_stats': price_stats,
            'prob_increase': prob_increase,
            'simulation_params': {
                'mean_return': mean_return,
                'std_return': std_return,
                'num_simulations': num_simulations,
                'prediction_days': prediction_days,
                'percentiles': percentiles,
                'log_returns': self.log_returns
            },
            'future_dates': future_dates
        }
        
        return results
    
    async def calculate_var(self, 
                      price_data: pd.DataFrame,
                      position_size: float,
                      confidence_level: float = 0.95,
                      time_horizon: int = 1,
                      num_simulations: int = 10000) -> Dict[str, Any]:
        """
        Calcular Value at Risk (VaR) mediante simulación Monte Carlo.
        
        Args:
            price_data: DataFrame con precios históricos
            position_size: Tamaño de la posición en unidades base (ej. BTC)
            confidence_level: Nivel de confianza para el VaR (ej. 0.95 para 95%)
            time_horizon: Horizonte temporal en días
            num_simulations: Número de simulaciones
            
        Returns:
            Diccionario con resultados del VaR
        """
        self.logger.info(f"Calculando VaR con nivel de confianza {confidence_level*100}%")
        
        # Ejecutar en un thread aparte para evitar bloquear el asyncio
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._calculate_var_sync,
                price_data,
                position_size,
                confidence_level,
                time_horizon,
                num_simulations
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculando VaR: {str(e)}")
            raise
    
    def _calculate_var_sync(self,
                       price_data: pd.DataFrame,
                       position_size: float,
                       confidence_level: float,
                       time_horizon: int,
                       num_simulations: int) -> Dict[str, Any]:
        """
        Versión sincrónica de calculate_var para ejecutar en un thread.
        
        Args:
            price_data: DataFrame con precios históricos
            position_size: Tamaño de la posición
            confidence_level: Nivel de confianza
            time_horizon: Horizonte temporal
            num_simulations: Número de simulaciones
            
        Returns:
            Diccionario con resultados del VaR
        """
        # Validar datos de entrada
        if 'close' not in price_data.columns:
            raise ValueError("price_data debe contener una columna 'close' con precios de cierre")
        
        # Extraer precios de cierre
        close_prices = price_data['close'].values
        last_price = close_prices[-1]
        
        # Calcular retornos diarios
        if self.log_returns:
            returns = np.log(close_prices[1:] / close_prices[:-1])
        else:
            returns = close_prices[1:] / close_prices[:-1] - 1
        
        # Estadísticas de los retornos
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Simular retornos futuros
        if self.log_returns:
            # Para horizon > 1, ajustar media y varianza
            horizon_mean = mean_return * time_horizon
            horizon_std = std_return * np.sqrt(time_horizon)
            
            # Simular retornos logarítmicos
            simulated_returns = np.random.normal(horizon_mean, horizon_std, num_simulations)
            
            # Convertir a precios
            simulated_prices = last_price * np.exp(simulated_returns)
        else:
            # Para retornos simples
            horizon_mean = (1 + mean_return) ** time_horizon - 1
            horizon_std = std_return * np.sqrt(time_horizon)
            
            # Simular retornos porcentuales
            simulated_returns = np.random.normal(horizon_mean, horizon_std, num_simulations)
            
            # Convertir a precios
            simulated_prices = last_price * (1 + simulated_returns)
        
        # Calcular P&L de la posición
        position_value_now = position_size * last_price
        simulated_values = position_size * simulated_prices
        simulated_pnl = simulated_values - position_value_now
        
        # Ordenar P&L de menor a mayor
        sorted_pnl = np.sort(simulated_pnl)
        
        # Calcular VaR
        var_index = int(num_simulations * (1 - confidence_level))
        var_absolute = abs(sorted_pnl[var_index])
        var_percentage = var_absolute / position_value_now * 100
        
        # Calcular Expected Shortfall (CVaR)
        cvar_absolute = abs(np.mean(sorted_pnl[:var_index]))
        cvar_percentage = cvar_absolute / position_value_now * 100
        
        # Histograma de P&L para visualización
        plt.figure(figsize=(10, 6))
        
        n, bins, patches = plt.hist(simulated_pnl, bins=50, alpha=0.75, density=True)
        
        # Añadir línea vertical para el VaR
        plt.axvline(x=sorted_pnl[var_index], color='r', linestyle='dashed', linewidth=2, 
                   label=f'VaR {confidence_level*100}%: ${var_absolute:.2f}')
        
        # Añadir línea vertical para el CVaR
        plt.axvline(x=np.mean(sorted_pnl[:var_index]), color='g', linestyle='dashed', linewidth=2,
                   label=f'CVaR {confidence_level*100}%: ${cvar_absolute:.2f}')
        
        plt.title(f'Distribución de P&L Simulados - Horizonte: {time_horizon} días')
        plt.xlabel('Profit & Loss ($)')
        plt.ylabel('Densidad')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Guardar el gráfico en base64 para enviar como resultado
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Preparar resultados
        results = {
            'var_absolute': float(var_absolute),
            'var_percentage': float(var_percentage),
            'cvar_absolute': float(cvar_absolute),
            'cvar_percentage': float(cvar_percentage),
            'position_value': float(position_value_now),
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'histogram_base64': img_str,
            'params': {
                'mean_return': float(mean_return),
                'std_return': float(std_return),
                'num_simulations': num_simulations,
                'log_returns': self.log_returns
            }
        }
        
        return results
    
    async def backtest_with_monte_carlo(self,
                                  price_data: pd.DataFrame,
                                  strategy_func: Callable,
                                  initial_capital: float = 10000.0,
                                  position_size_pct: float = 0.1,
                                  num_simulations: int = 100,
                                  strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Realizar backtesting de una estrategia con datos históricos y simulaciones Monte Carlo.
        
        Args:
            price_data: DataFrame con datos OHLCV históricos
            strategy_func: Función que implementa la estrategia, debe aceptar (df, params) y devolver señales
            initial_capital: Capital inicial para el backtest
            position_size_pct: Porcentaje del capital para cada posición
            num_simulations: Número de simulaciones Monte Carlo
            strategy_params: Parámetros adicionales para la estrategia
            
        Returns:
            Diccionario con resultados del backtest y simulaciones
        """
        self.logger.info(f"Iniciando backtest con Monte Carlo usando {num_simulations} simulaciones")
        
        # Ejecutar en un thread aparte para evitar bloquear el asyncio
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._backtest_with_monte_carlo_sync,
                price_data,
                strategy_func,
                initial_capital,
                position_size_pct,
                num_simulations,
                strategy_params
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en backtest con Monte Carlo: {str(e)}")
            raise
    
    def _backtest_with_monte_carlo_sync(self,
                                   price_data: pd.DataFrame,
                                   strategy_func: Callable,
                                   initial_capital: float,
                                   position_size_pct: float,
                                   num_simulations: int,
                                   strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Versión sincrónica de backtest_with_monte_carlo para ejecutar en un thread.
        
        Args:
            price_data: DataFrame con datos OHLCV
            strategy_func: Función de estrategia
            initial_capital: Capital inicial
            position_size_pct: Porcentaje del capital por posición
            num_simulations: Número de simulaciones
            strategy_params: Parámetros adicionales
            
        Returns:
            Diccionario con resultados
        """
        # Validar datos de entrada
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in price_data.columns:
                raise ValueError(f"price_data debe contener la columna '{col}'")
        
        if strategy_params is None:
            strategy_params = {}
        
        # Realizar backtest con datos históricos originales
        signals = strategy_func(price_data.copy(), strategy_params)
        
        if not isinstance(signals, pd.Series) and not isinstance(signals, np.ndarray):
            raise ValueError("strategy_func debe devolver una Serie o array con señales")
        
        # Asegurar que signals tenga el mismo índice que price_data
        if isinstance(signals, pd.Series) and len(signals) != len(price_data):
            raise ValueError("Las señales devueltas deben tener la misma longitud que price_data")
        elif isinstance(signals, np.ndarray) and len(signals) != len(price_data):
            raise ValueError("Las señales devueltas deben tener la misma longitud que price_data")
        
        # Convertir a Serie si es un array
        if isinstance(signals, np.ndarray):
            signals = pd.Series(signals, index=price_data.index)
        
        # Calcular retornos del precio
        price_returns = price_data['close'].pct_change().fillna(0)
        
        # Calcular equity curve con las señales (1 = long, 0 = neutral, -1 = short)
        strategy_returns = signals.shift(1) * price_returns
        strategy_returns.fillna(0, inplace=True)
        
        equity_curve = (1 + strategy_returns).cumprod() * initial_capital
        
        # Calcular métricas de rendimiento
        total_return = (equity_curve.iloc[-1] / initial_capital - 1)
        
        # Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (anualizado)
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        
        # Sortino Ratio (anualizado)
        negative_returns = strategy_returns[strategy_returns < 0]
        sortino_ratio = strategy_returns.mean() / negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else np.nan
        
        # Generar escenarios con Monte Carlo
        mc_results = []
        
        # Estadísticas de los retornos de la estrategia
        strategy_mean = strategy_returns.mean()
        strategy_std = strategy_returns.std()
        
        # Generar múltiples caminos para la equity curve
        for sim in range(num_simulations):
            # Perturbar los retornos históricos
            random_returns = np.random.normal(strategy_mean, strategy_std, len(strategy_returns))
            
            # Calcular equity curve
            sim_equity = initial_capital
            sim_equity_curve = [sim_equity]
            
            for ret in random_returns:
                sim_equity *= (1 + ret)
                sim_equity_curve.append(sim_equity)
            
            # Calcular métricas
            sim_return = (sim_equity_curve[-1] / initial_capital - 1)
            
            # Drawdown
            sim_equity_array = np.array(sim_equity_curve)
            sim_rolling_max = np.maximum.accumulate(sim_equity_array)
            sim_drawdown = (sim_equity_array - sim_rolling_max) / sim_rolling_max
            sim_max_drawdown = np.min(sim_drawdown)
            
            mc_results.append({
                'final_equity': sim_equity_curve[-1],
                'total_return': sim_return,
                'max_drawdown': sim_max_drawdown
            })
        
        # Calcular percentiles
        mc_returns = [res['total_return'] for res in mc_results]
        mc_drawdowns = [res['max_drawdown'] for res in mc_results]
        mc_final_equity = [res['final_equity'] for res in mc_results]
        
        percentiles = [5, 25, 50, 75, 95]
        return_percentiles = {}
        drawdown_percentiles = {}
        equity_percentiles = {}
        
        for p in percentiles:
            return_percentiles[f'p{p}'] = np.percentile(mc_returns, p)
            drawdown_percentiles[f'p{p}'] = np.percentile(mc_drawdowns, p)
            equity_percentiles[f'p{p}'] = np.percentile(mc_final_equity, p)
        
        # Probabilidad de ganancia
        prob_profit = np.sum(np.array(mc_returns) > 0) / num_simulations
        
        # Crear gráficos
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Gráfico de equity curve
        ax1.plot(equity_curve, label='Equity Original')
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Capital ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histograma de retornos de Monte Carlo
        n, bins, patches = ax2.hist(mc_returns, bins=30, alpha=0.75, density=True)
        ax2.axvline(x=total_return, color='r', linestyle='dashed', linewidth=2, 
                   label=f'Retorno Original: {total_return:.2%}')
        ax2.axvline(x=return_percentiles['p5'], color='g', linestyle='dashed', linewidth=2,
                   label=f'Retorno P5: {return_percentiles["p5"]:.2%}')
        ax2.axvline(x=return_percentiles['p95'], color='g', linestyle='dashed', linewidth=2,
                   label=f'Retorno P95: {return_percentiles["p95"]:.2%}')
        ax2.set_title('Distribución de Retornos Simulados')
        ax2.set_xlabel('Retorno (%)')
        ax2.set_ylabel('Densidad')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar el gráfico en base64 para enviar como resultado
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Preparar resultados
        results = {
            'historical': {
                'equity_curve': equity_curve.to_list(),
                'total_return': float(total_return),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio)
            },
            'monte_carlo': {
                'return_percentiles': return_percentiles,
                'drawdown_percentiles': drawdown_percentiles,
                'equity_percentiles': equity_percentiles,
                'prob_profit': float(prob_profit)
            },
            'params': {
                'initial_capital': initial_capital,
                'position_size_pct': position_size_pct,
                'num_simulations': num_simulations
            },
            'chart_base64': img_str,
            'dates': price_data.index.to_list() if isinstance(price_data.index[0], (datetime, np.datetime64, pd.Timestamp)) else None
        }
        
        return results
    
    def generate_price_chart(self, simulation_result: Dict[str, Any], show_paths: bool = True, num_paths: int = 50) -> str:
        """
        Generar gráfico de los caminos de precios simulados.
        
        Args:
            simulation_result: Resultado previo de simulate_price_paths
            show_paths: Si es True, muestra los caminos individuales
            num_paths: Número de caminos a mostrar (si show_paths es True)
            
        Returns:
            Imagen en formato base64
        """
        if not simulation_result:
            self.logger.error("No hay resultado de simulación disponible")
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Mostrar caminos individuales si se solicita
        if show_paths and 'price_paths' in simulation_result:
            paths = simulation_result['price_paths']
            future_dates = simulation_result['future_dates']
            
            # Limitar el número de caminos a mostrar para mantener el gráfico limpio
            paths_to_show = min(num_paths, paths.shape[1])
            
            for i in range(paths_to_show):
                plt.plot(future_dates, paths[:, i], 'b-', alpha=0.1)
        
        # Mostrar percentiles
        if 'future_df' in simulation_result:
            future_df = simulation_result['future_df']
            
            if 'p5' in future_df.columns:
                plt.plot(future_df.index, future_df['p5'], 'r--', linewidth=2, label='5%')
            
            if 'p25' in future_df.columns:
                plt.plot(future_df.index, future_df['p25'], 'y--', linewidth=2, label='25%')
            
            if 'p50' in future_df.columns:
                plt.plot(future_df.index, future_df['p50'], 'g-', linewidth=2, label='Mediana')
            
            if 'p75' in future_df.columns:
                plt.plot(future_df.index, future_df['p75'], 'y--', linewidth=2, label='75%')
            
            if 'p95' in future_df.columns:
                plt.plot(future_df.index, future_df['p95'], 'r--', linewidth=2, label='95%')
        
        # Mostrar precio inicial
        initial_price = simulation_result.get('initial_price')
        if initial_price:
            plt.axhline(y=initial_price, color='k', linestyle='-', linewidth=1, label='Precio Actual')
        
        plt.title('Simulación Monte Carlo de Precios')
        plt.xlabel('Fecha')
        plt.ylabel('Precio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Formatear eje X si son fechas
        if isinstance(future_df.index[0], (datetime, np.datetime64, pd.Timestamp)):
            plt.gcf().autofmt_xdate()
        
        # Guardar el gráfico en base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    async def generate_equity_chart(self, 
                              backtest_result: Dict[str, Any], 
                              num_paths: int = 20) -> str:
        """
        Generar gráfico comparativo de curvas de equity para el backtest.
        
        Args:
            backtest_result: Resultado de backtest_with_monte_carlo
            num_paths: Número de caminos simulados a mostrar
            
        Returns:
            Imagen en formato base64
        """
        try:
            # Obtener datos
            historical_equity = backtest_result['historical']['equity_curve']
            initial_capital = backtest_result['params']['initial_capital']
            
            # Generar gráfico
            plt.figure(figsize=(12, 8))
            
            # Mostrar equity curve histórica
            plt.plot(historical_equity, 'b-', linewidth=2, label='Equity Histórica')
            
            # Agregar línea de capital inicial
            plt.axhline(y=initial_capital, color='k', linestyle='--', linewidth=1, label='Capital Inicial')
            
            plt.title('Curva de Equity y Simulaciones Monte Carlo')
            plt.xlabel('Tiempo')
            plt.ylabel('Capital ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Guardar el gráfico en base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error generando gráfico de equity: {str(e)}")
            return None
    
    async def calculate_kelly_criterion(self, 
                                  backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcular el Criterio de Kelly para optimizar el tamaño de posición.
        
        Args:
            backtest_result: Resultado de backtest_with_monte_carlo
            
        Returns:
            Diccionario con resultados del criterio de Kelly
        """
        try:
            # Obtener retornos históricos de la estrategia
            total_return = backtest_result['historical']['total_return']
            
            # Para simplificar, usamos una aproximación con probabilidad de ganancia
            prob_win = backtest_result['monte_carlo']['prob_profit']
            
            # Para calcular Kelly completo, necesitaríamos datos de operaciones individuales
            # Esta es una aproximación simple
            avg_win = max(0.01, total_return * 2)  # Asumimos que la ganancia promedio es el doble del retorno total
            avg_loss = max(0.01, -backtest_result['historical']['max_drawdown'])  # Usamos el drawdown como pérdida
            
            # Fórmula clásica de Kelly
            kelly_full = (prob_win / avg_loss) - ((1 - prob_win) / avg_win)
            
            # Kelly fraccional (más conservador)
            kelly_half = kelly_full * 0.5
            kelly_quarter = kelly_full * 0.25
            
            return {
                'kelly_full': float(kelly_full),
                'kelly_half': float(kelly_half),
                'kelly_quarter': float(kelly_quarter),
                'probability_of_win': float(prob_win),
                'average_win': float(avg_win),
                'average_loss': float(avg_loss)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando criterio de Kelly: {str(e)}")
            return {
                'error': str(e),
                'kelly_quarter': 0.025  # Valor seguro por defecto (2.5% del capital)
            }