"""
Simulaciones Monte Carlo avanzadas para evaluación de estrategias.

Este módulo proporciona funcionalidades para realizar simulaciones Monte Carlo
avanzadas, incluyendo escenarios extremos, pruebas de estrés y optimización
de parámetros de estrategias.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime, timedelta
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import random
import scipy.stats as stats

class MonteCarloAdvanced:
    """
    Simulador avanzado de Monte Carlo para evaluación de estrategias.
    
    Proporciona herramientas para generar escenarios de mercado realistas,
    probar estrategias con miles de simulaciones y evaluar riesgos extremos.
    """
    
    def __init__(self, 
                 n_simulations: int = 1000,
                 confidence_level: float = 0.95,
                 max_processes: Optional[int] = None,
                 seed: Optional[int] = None,
                 output_dir: str = './results/monte_carlo'):
        """
        Inicializar simulador Monte Carlo avanzado.
        
        Args:
            n_simulations: Número de simulaciones a generar
            confidence_level: Nivel de confianza para cálculos de VaR y ES
            max_processes: Número máximo de procesos paralelos
            seed: Semilla para reproducibilidad
            output_dir: Directorio para resultados
        """
        self.logger = logging.getLogger(__name__)
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.max_processes = max_processes or max(1, multiprocessing.cpu_count() - 1)
        self.seed = seed
        self.output_dir = output_dir
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Establecer semilla si se proporciona
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Resultados de simulaciones
        self.simulation_results = None
        
        self.logger.info(f"MonteCarloAdvanced inicializado con {n_simulations} simulaciones")
    
    def generate_gbm_paths(self, 
                          initial_price: float, 
                          mu: float, 
                          sigma: float, 
                          days: int, 
                          steps_per_day: int = 1) -> np.ndarray:
        """
        Generar caminos de precio según Movimiento Browniano Geométrico.
        
        Args:
            initial_price: Precio inicial
            mu: Rendimiento esperado anualizado
            sigma: Volatilidad anualizada
            days: Número de días a simular
            steps_per_day: Pasos por día
            
        Returns:
            Array de caminos de precio simulados
        """
        # Parámetros
        total_steps = days * steps_per_day
        dt = 1.0 / (252 * steps_per_day)  # Tiempo en años por paso
        
        # Convertir parámetros anualizados a escala por paso
        mu_dt = mu * dt
        sigma_dt = sigma * np.sqrt(dt)
        
        # Generar innovaciones
        Z = np.random.normal(0, 1, (self.n_simulations, total_steps))
        
        # Calcular rendimientos
        returns = mu_dt + sigma_dt * Z
        
        # Calcular precio acumulado (en forma logarítmica y luego exponenciar)
        log_returns = np.cumsum(returns, axis=1)
        paths = initial_price * np.exp(log_returns)
        
        # Añadir precio inicial
        paths_with_initial = np.hstack([
            np.full((self.n_simulations, 1), initial_price),
            paths
        ])
        
        return paths_with_initial
    
    def generate_gbm_with_jumps(self, 
                               initial_price: float, 
                               mu: float, 
                               sigma: float, 
                               jump_intensity: float,
                               jump_mean: float,
                               jump_std: float,
                               days: int, 
                               steps_per_day: int = 1) -> np.ndarray:
        """
        Generar caminos de precio con Movimiento Browniano Geométrico y saltos.
        
        Args:
            initial_price: Precio inicial
            mu: Rendimiento esperado anualizado
            sigma: Volatilidad anualizada
            jump_intensity: Intensidad anualizada de saltos (lambda)
            jump_mean: Media del tamaño de los saltos
            jump_std: Desviación estándar del tamaño de los saltos
            days: Número de días a simular
            steps_per_day: Pasos por día
            
        Returns:
            Array de caminos de precio simulados
        """
        # Parámetros
        total_steps = days * steps_per_day
        dt = 1.0 / (252 * steps_per_day)  # Tiempo en años por paso
        
        # Intensidad de saltos por paso
        jump_intensity_dt = jump_intensity * dt
        
        # Convertir parámetros anualizados a escala por paso
        mu_dt = mu * dt
        sigma_dt = sigma * np.sqrt(dt)
        
        # Generar número de saltos para cada paso usando distribución de Poisson
        n_jumps = np.random.poisson(jump_intensity_dt, (self.n_simulations, total_steps))
        
        # Generar tamaños de saltos
        jump_sizes = np.random.normal(jump_mean, jump_std, (self.n_simulations, total_steps))
        
        # Calcular efecto de los saltos
        jumps = n_jumps * jump_sizes
        
        # Generar innovaciones para GBM
        Z = np.random.normal(0, 1, (self.n_simulations, total_steps))
        
        # Calcular rendimientos con GBM y saltos
        returns = mu_dt + sigma_dt * Z + jumps
        
        # Calcular precio acumulado (en forma logarítmica y luego exponenciar)
        log_returns = np.cumsum(returns, axis=1)
        paths = initial_price * np.exp(log_returns)
        
        # Añadir precio inicial
        paths_with_initial = np.hstack([
            np.full((self.n_simulations, 1), initial_price),
            paths
        ])
        
        return paths_with_initial
    
    def generate_heston_paths(self, 
                             initial_price: float, 
                             initial_vol: float,
                             kappa: float,
                             theta: float,
                             sigma_v: float,
                             rho: float,
                             mu: float,
                             days: int, 
                             steps_per_day: int = 1) -> np.ndarray:
        """
        Generar caminos de precio según el modelo de Heston (volatilidad estocástica).
        
        Args:
            initial_price: Precio inicial
            initial_vol: Volatilidad inicial (anualizada)
            kappa: Velocidad de reversión a la media de la volatilidad
            theta: Nivel medio de volatilidad a largo plazo
            sigma_v: Volatilidad de la volatilidad
            rho: Correlación entre el precio y la volatilidad
            mu: Rendimiento esperado anualizado
            days: Número de días a simular
            steps_per_day: Pasos por día
            
        Returns:
            Array de caminos de precio simulados
        """
        # Parámetros
        total_steps = days * steps_per_day
        dt = 1.0 / (252 * steps_per_day)  # Tiempo en años por paso
        
        # Convertir volatilidad inicial a varianza
        initial_var = initial_vol**2
        
        # Matrices para almacenar resultados
        prices = np.zeros((self.n_simulations, total_steps + 1))
        prices[:, 0] = initial_price
        
        variances = np.zeros((self.n_simulations, total_steps + 1))
        variances[:, 0] = initial_var
        
        # Simulación de Heston
        for i in range(total_steps):
            # Generar ruido correlacionado
            Z1 = np.random.normal(0, 1, self.n_simulations)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, self.n_simulations)
            
            # Actualizar varianza (con una restricción para mantenerla positiva)
            variances[:, i+1] = np.maximum(
                variances[:, i] + kappa * (theta - variances[:, i]) * dt + sigma_v * np.sqrt(variances[:, i] * dt) * Z1,
                1e-10  # Evitar valores negativos
            )
            
            # Actualizar precios
            prices[:, i+1] = prices[:, i] * np.exp(
                (mu - 0.5 * variances[:, i]) * dt + np.sqrt(variances[:, i] * dt) * Z2
            )
        
        return prices
    
    def generate_regime_switching_paths(self, 
                                       initial_price: float, 
                                       mu_regimes: List[float], 
                                       sigma_regimes: List[float],
                                       transition_matrix: np.ndarray,
                                       days: int, 
                                       steps_per_day: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generar caminos con cambios de régimen de mercado.
        
        Args:
            initial_price: Precio inicial
            mu_regimes: Lista de rendimientos esperados para cada régimen
            sigma_regimes: Lista de volatilidades para cada régimen
            transition_matrix: Matriz de transición entre regímenes
            days: Número de días a simular
            steps_per_day: Pasos por día
            
        Returns:
            Tupla de (caminos de precio, regímenes)
        """
        # Verificar coherencia de parámetros
        n_regimes = len(mu_regimes)
        if len(sigma_regimes) != n_regimes:
            raise ValueError("La longitud de mu_regimes y sigma_regimes debe ser igual")
        
        if transition_matrix.shape != (n_regimes, n_regimes):
            raise ValueError(f"La matriz de transición debe ser de tamaño {n_regimes}x{n_regimes}")
        
        # Parámetros
        total_steps = days * steps_per_day
        dt = 1.0 / (252 * steps_per_day)  # Tiempo en años por paso
        
        # Matrices para almacenar resultados
        prices = np.zeros((self.n_simulations, total_steps + 1))
        prices[:, 0] = initial_price
        
        regimes = np.zeros((self.n_simulations, total_steps + 1), dtype=int)
        
        # Seleccionar régimen inicial aleatoriamente
        initial_regimes = np.random.choice(n_regimes, size=self.n_simulations)
        regimes[:, 0] = initial_regimes
        
        # Simulación con cambios de régimen
        for i in range(total_steps):
            for sim in range(self.n_simulations):
                # Régimen actual
                current_regime = regimes[sim, i]
                
                # Determinar próximo régimen según matriz de transición
                next_regime = np.random.choice(n_regimes, p=transition_matrix[current_regime])
                regimes[sim, i+1] = next_regime
                
                # Parámetros del régimen actual
                mu = mu_regimes[current_regime]
                sigma = sigma_regimes[current_regime]
                
                # Calcular rendimiento
                Z = np.random.normal(0, 1)
                returns = mu * dt + sigma * np.sqrt(dt) * Z
                
                # Actualizar precio
                prices[sim, i+1] = prices[sim, i] * np.exp(returns)
        
        return prices, regimes
    
    def generate_agent_based_paths(self, 
                                  initial_price: float,
                                  trend_followers_weight: float = 0.3,
                                  value_traders_weight: float = 0.3,
                                  noise_traders_weight: float = 0.4,
                                  fundamental_vol: float = 0.01,
                                  noise_vol: float = 0.02,
                                  days: int = 252,
                                  steps_per_day: int = 1) -> np.ndarray:
        """
        Generar caminos usando modelo basado en agentes.
        
        Args:
            initial_price: Precio inicial
            trend_followers_weight: Peso de traders siguiendo tendencia
            value_traders_weight: Peso de traders siguiendo valor fundamental
            noise_traders_weight: Peso de traders aleatorios
            fundamental_vol: Volatilidad del valor fundamental
            noise_vol: Volatilidad del ruido
            days: Número de días a simular
            steps_per_day: Pasos por día
            
        Returns:
            Array de caminos de precio simulados
        """
        # Verificar que los pesos sumen 1
        total_weight = trend_followers_weight + value_traders_weight + noise_traders_weight
        if abs(total_weight - 1.0) > 1e-6:
            self.logger.warning("Los pesos de los agentes no suman 1. Normalizando.")
            trend_followers_weight /= total_weight
            value_traders_weight /= total_weight
            noise_traders_weight /= total_weight
        
        # Parámetros
        total_steps = days * steps_per_day
        dt = 1.0 / (252 * steps_per_day)  # Tiempo en años por paso
        
        # Matrices para almacenar resultados
        prices = np.zeros((self.n_simulations, total_steps + 1))
        prices[:, 0] = initial_price
        
        # Generar valores fundamentales
        fundamental_values = np.zeros((self.n_simulations, total_steps + 1))
        fundamental_values[:, 0] = initial_price
        
        # Parámetros para la memoria de los trend followers
        trend_memory = 10  # Días que consideran los trend followers
        
        # Simulación basada en agentes
        for sim in range(self.n_simulations):
            for i in range(total_steps):
                # Actualizar valor fundamental
                fundamental_shock = np.random.normal(0, fundamental_vol * np.sqrt(dt))
                fundamental_values[sim, i+1] = fundamental_values[sim, i] * np.exp(fundamental_shock)
                
                # Cálculo de rendimiento de cada tipo de agente
                
                # 1. Trend followers
                if i >= trend_memory:
                    # Calcular tendencia reciente
                    recent_trend = (prices[sim, i] / prices[sim, i-trend_memory]) - 1
                    trend_return = 0.5 * recent_trend  # Siguen la tendencia reciente
                else:
                    trend_return = 0
                
                # 2. Value traders
                fundamental_gap = fundamental_values[sim, i] / prices[sim, i] - 1
                value_return = 0.3 * fundamental_gap  # Compran si subvalorado, venden si sobrevalorado
                
                # 3. Noise traders
                noise = np.random.normal(0, noise_vol * np.sqrt(dt))
                noise_return = noise
                
                # Rendimiento combinado según pesos
                combined_return = (
                    trend_followers_weight * trend_return +
                    value_traders_weight * value_return +
                    noise_traders_weight * noise_return
                )
                
                # Actualizar precio
                prices[sim, i+1] = prices[sim, i] * (1 + combined_return)
        
        return prices
    
    def simulate_strategy(self, 
                         price_paths: np.ndarray, 
                         strategy_func: Callable,
                         initial_balance: float = 10000.0,
                         commission: float = 0.001,
                         days: int = 252,
                         steps_per_day: int = 1,
                         parallel: bool = True) -> Dict[str, np.ndarray]:
        """
        Simular una estrategia de trading en múltiples caminos de precio.
        
        Args:
            price_paths: Caminos de precio simulados
            strategy_func: Función que implementa la estrategia
            initial_balance: Balance inicial
            commission: Comisión por operación
            days: Número de días simulados
            steps_per_day: Pasos por día
            parallel: Si es True, usa procesamiento paralelo
            
        Returns:
            Diccionario con resultados de las simulaciones
        """
        if price_paths.shape[0] != self.n_simulations:
            raise ValueError("El número de caminos de precio no coincide con n_simulations")
        
        # Total de pasos en la simulación
        total_steps = days * steps_per_day + 1  # +1 para incluir el precio inicial
        
        if price_paths.shape[1] != total_steps:
            raise ValueError(f"Los caminos de precio deben tener {total_steps} pasos")
        
        # Función para ejecutar una simulación individual
        def run_single_simulation(sim_idx):
            # Extraer camino de precio para esta simulación
            prices = price_paths[sim_idx]
            
            # Inicializar variables de seguimiento
            balance = initial_balance
            position = 0.0
            portfolio_values = np.zeros(total_steps)
            portfolio_values[0] = balance
            trades = []
            
            # Simular la estrategia
            for i in range(1, total_steps):
                # Datos de precio actuales e históricos
                current_price = prices[i]
                price_history = prices[:i+1]
                
                # Ejecutar estrategia
                action = strategy_func(price_history, balance, position)
                
                # Procesar acción (1: comprar, -1: vender, 0: mantener)
                if action == 1 and balance > 0:  # Comprar
                    # Calcular cantidad a comprar
                    buy_amount = balance / current_price
                    # Aplicar comisión
                    commission_cost = buy_amount * current_price * commission
                    buy_amount_after_commission = (balance - commission_cost) / current_price
                    
                    # Actualizar posición y balance
                    position += buy_amount_after_commission
                    balance = 0
                    
                    # Registrar operación
                    trades.append({
                        'time_idx': i,
                        'price': current_price,
                        'action': 'buy',
                        'amount': buy_amount_after_commission,
                        'balance': balance,
                        'position': position
                    })
                    
                elif action == -1 and position > 0:  # Vender
                    # Calcular valor de venta
                    sell_value = position * current_price
                    # Aplicar comisión
                    commission_cost = sell_value * commission
                    sell_value_after_commission = sell_value - commission_cost
                    
                    # Actualizar posición y balance
                    balance += sell_value_after_commission
                    position = 0
                    
                    # Registrar operación
                    trades.append({
                        'time_idx': i,
                        'price': current_price,
                        'action': 'sell',
                        'amount': position,
                        'balance': balance,
                        'position': 0
                    })
                
                # Actualizar valor del portafolio
                portfolio_values[i] = balance + position * current_price
            
            # Calcular métricas
            final_value = portfolio_values[-1]
            total_return = (final_value / initial_balance) - 1
            
            # Calcular drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown)
            
            return {
                'portfolio_values': portfolio_values,
                'trades': trades,
                'final_value': final_value,
                'total_return': total_return,
                'max_drawdown': max_drawdown
            }
        
        # Ejecutar simulaciones
        if parallel and self.n_simulations > 1:
            # Ejecución paralela
            with ProcessPoolExecutor(max_workers=self.max_processes) as executor:
                results = list(executor.map(run_single_simulation, range(self.n_simulations)))
        else:
            # Ejecución secuencial
            results = [run_single_simulation(i) for i in range(self.n_simulations)]
        
        # Recopilar resultados
        portfolio_values = np.array([r['portfolio_values'] for r in results])
        trades = [r['trades'] for r in results]
        final_values = np.array([r['final_value'] for r in results])
        total_returns = np.array([r['total_return'] for r in results])
        max_drawdowns = np.array([r['max_drawdown'] for r in results])
        
        # Guardar resultados
        self.simulation_results = {
            'portfolio_values': portfolio_values,
            'trades': trades,
            'final_values': final_values,
            'total_returns': total_returns,
            'max_drawdowns': max_drawdowns,
            'price_paths': price_paths,
            'initial_balance': initial_balance,
            'commission': commission,
            'days': days,
            'steps_per_day': steps_per_day
        }
        
        return self.simulation_results
    
    def calculate_risk_metrics(self, results: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calcular métricas de riesgo a partir de los resultados de simulación.
        
        Args:
            results: Resultados de simulación (None = usar resultados actuales)
            
        Returns:
            Diccionario con métricas de riesgo
        """
        if results is None:
            if self.simulation_results is None:
                raise ValueError("No hay resultados de simulación disponibles")
            results = self.simulation_results
        
        # Extraer datos relevantes
        final_values = results['final_values']
        total_returns = results['total_returns']
        max_drawdowns = results['max_drawdowns']
        portfolio_values = results['portfolio_values']
        initial_balance = results['initial_balance']
        
        # Calcular métricas de rendimiento
        mean_return = np.mean(total_returns)
        median_return = np.median(total_returns)
        std_return = np.std(total_returns)
        
        # Calcular VaR (Value at Risk)
        var_percent = 1 - self.confidence_level
        var = np.percentile(total_returns, var_percent * 100)
        
        # Calcular ES (Expected Shortfall / Conditional VaR)
        es = np.mean(total_returns[total_returns <= var])
        
        # Calcular probabilidad de pérdida
        prob_loss = np.mean(total_returns < 0)
        
        # Calcular promedio de drawdown máximo
        mean_max_drawdown = np.mean(max_drawdowns)
        
        # Calcular volatilidad diaria y anualizada
        # Primero convertir los valores del portafolio a retornos diarios
        daily_returns = np.diff(portfolio_values, axis=1) / portfolio_values[:, :-1]
        # Calcular volatilidad diaria (promedio de las desviaciones estándar de cada simulación)
        daily_vol = np.mean([np.std(daily_returns[i]) for i in range(self.n_simulations)])
        # Anualizar (asumiendo 252 días de trading al año)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Calcular Sharpe Ratio (asumiendo tasa libre de riesgo = 0)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Calcular Sortino Ratio (solo considera volatilidad de retornos negativos)
        negative_returns = total_returns[total_returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
        
        # Calcular Calmar Ratio (rendimiento / máximo drawdown)
        calmar_ratio = mean_return / mean_max_drawdown if mean_max_drawdown > 0 else 0
        
        # Calcular omega ratio (probabilidad ponderada de ganancia vs pérdida)
        threshold = 0  # Umbral de retorno (0 = retorno positivo)
        returns_above = total_returns[total_returns > threshold] - threshold
        returns_below = threshold - total_returns[total_returns <= threshold]
        
        omega_ratio = (np.sum(returns_above) / len(returns_above)) / (np.sum(returns_below) / len(returns_below)) if len(returns_below) > 0 and np.sum(returns_below) > 0 else float('inf')
        
        # Métricas adicionales para evaluación de robustez
        min_return = np.min(total_returns)
        max_return = np.max(total_returns)
        
        # Calcular percentiles
        percentiles = {
            'p1': np.percentile(total_returns, 1),
            'p5': np.percentile(total_returns, 5),
            'p10': np.percentile(total_returns, 10),
            'p25': np.percentile(total_returns, 25),
            'p50': np.percentile(total_returns, 50),
            'p75': np.percentile(total_returns, 75),
            'p90': np.percentile(total_returns, 90),
            'p95': np.percentile(total_returns, 95),
            'p99': np.percentile(total_returns, 99)
        }
        
        # Calcular estadísticas adicionales
        skewness = stats.skew(total_returns)
        kurtosis = stats.kurtosis(total_returns)
        
        # Preparar resultado
        risk_metrics = {
            'mean_return': float(mean_return),
            'median_return': float(median_return),
            'std_return': float(std_return),
            'var': float(var),
            'expected_shortfall': float(es),
            'prob_loss': float(prob_loss),
            'mean_max_drawdown': float(mean_max_drawdown),
            'daily_volatility': float(daily_vol),
            'annual_volatility': float(annual_vol),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'omega_ratio': float(omega_ratio),
            'min_return': float(min_return),
            'max_return': float(max_return),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis)
        }
        
        # Añadir percentiles
        for k, v in percentiles.items():
            risk_metrics[k] = float(v)
        
        return risk_metrics
    
    def plot_simulation_results(self, 
                               results: Optional[Dict[str, Any]] = None,
                               plot_paths: bool = True,
                               plot_histogram: bool = True,
                               plot_drawdowns: bool = True,
                               max_paths: int = 100,
                               save_path: Optional[str] = None) -> str:
        """
        Generar gráficos de los resultados de simulación.
        
        Args:
            results: Resultados de simulación (None = usar resultados actuales)
            plot_paths: Si es True, grafica los caminos de portafolio
            plot_histogram: Si es True, grafica histograma de retornos
            plot_drawdowns: Si es True, grafica distribución de drawdowns
            max_paths: Número máximo de caminos a graficar
            save_path: Ruta donde guardar el gráfico (opcional)
            
        Returns:
            Imagen en formato base64
        """
        if results is None:
            if self.simulation_results is None:
                raise ValueError("No hay resultados de simulación disponibles")
            results = self.simulation_results
        
        # Extraer datos relevantes
        portfolio_values = results['portfolio_values']
        total_returns = results['total_returns']
        max_drawdowns = results['max_drawdowns']
        price_paths = results['price_paths']
        initial_balance = results['initial_balance']
        days = results['days']
        steps_per_day = results['steps_per_day']
        
        # Limitar número de caminos a graficar
        plot_n_simulations = min(max_paths, self.n_simulations)
        
        # Determinar número de subplots
        n_plots = sum([plot_paths, plot_histogram, plot_drawdowns])
        
        # Crear figura
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots))
        
        # Asegurar que axes sea una lista
        if n_plots == 1:
            axes = [axes]
        
        # Índice para el subplot actual
        ax_idx = 0
        
        # Gráfico de caminos de portafolio
        if plot_paths:
            ax = axes[ax_idx]
            
            # Graficar caminos de portafolio
            x = np.arange(portfolio_values.shape[1])
            
            # Graficar una muestra de caminos
            for i in range(plot_n_simulations):
                ax.plot(x, portfolio_values[i], 'b-', alpha=0.1)
            
            # Graficar percentiles 10, 50 (mediana) y 90
            p10 = np.percentile(portfolio_values, 10, axis=0)
            p50 = np.percentile(portfolio_values, 50, axis=0)
            p90 = np.percentile(portfolio_values, 90, axis=0)
            
            ax.plot(x, p10, 'r--', label='Percentil 10', linewidth=2)
            ax.plot(x, p50, 'g-', label='Mediana', linewidth=2)
            ax.plot(x, p90, 'r--', label='Percentil 90', linewidth=2)
            
            # Formato del gráfico
            ax.set_title('Simulación Monte Carlo de Valores del Portafolio')
            ax.set_xlabel('Pasos')
            ax.set_ylabel('Valor del Portafolio')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax_idx += 1
        
        # Histograma de retornos
        if plot_histogram:
            ax = axes[ax_idx]
            
            # Graficar histograma de retornos
            ax.hist(total_returns, bins=50, alpha=0.7, density=True)
            
            # Graficar líneas para la media, VaR y ES
            mean_return = np.mean(total_returns)
            var = np.percentile(total_returns, (1 - self.confidence_level) * 100)
            es = np.mean(total_returns[total_returns <= var])
            
            ax.axvline(mean_return, color='g', linestyle='-', linewidth=2, label=f'Media: {mean_return:.2%}')
            ax.axvline(var, color='r', linestyle='--', linewidth=2, label=f'VaR {self.confidence_level:.0%}: {var:.2%}')
            ax.axvline(es, color='m', linestyle='-.', linewidth=2, label=f'ES {self.confidence_level:.0%}: {es:.2%}')
            
            # Graficar distribución normal ajustada
            x = np.linspace(min(total_returns), max(total_returns), 100)
            mean = np.mean(total_returns)
            std = np.std(total_returns)
            normal_pdf = stats.norm.pdf(x, mean, std)
            ax.plot(x, normal_pdf, 'k--', linewidth=2, label='Normal')
            
            # Formato del gráfico
            ax.set_title('Distribución de Retornos')
            ax.set_xlabel('Retorno Total')
            ax.set_ylabel('Densidad')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax_idx += 1
        
        # Gráfico de drawdowns
        if plot_drawdowns:
            ax = axes[ax_idx]
            
            # Graficar histograma de drawdowns máximos
            ax.hist(max_drawdowns, bins=50, alpha=0.7, density=True)
            
            # Graficar línea para el drawdown medio
            mean_max_drawdown = np.mean(max_drawdowns)
            median_max_drawdown = np.median(max_drawdowns)
            p95_max_drawdown = np.percentile(max_drawdowns, 95)
            
            ax.axvline(mean_max_drawdown, color='g', linestyle='-', linewidth=2, label=f'Media: {mean_max_drawdown:.2%}')
            ax.axvline(median_max_drawdown, color='b', linestyle='--', linewidth=2, label=f'Mediana: {median_max_drawdown:.2%}')
            ax.axvline(p95_max_drawdown, color='r', linestyle='-.', linewidth=2, label=f'Percentil 95: {p95_max_drawdown:.2%}')
            
            # Formato del gráfico
            ax.set_title('Distribución de Drawdowns Máximos')
            ax.set_xlabel('Drawdown Máximo')
            ax.set_ylabel('Densidad')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Ajustar espaciado
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Guardar gráfico si se solicita
        if save_path:
            plt.savefig(save_path)
            
        plt.close()
        
        return img_str
    
    def run_stress_test(self, 
                       strategy_func: Callable,
                       initial_balance: float = 10000.0,
                       commission: float = 0.001,
                       scenarios: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Ejecutar pruebas de estrés con escenarios predefinidos.
        
        Args:
            strategy_func: Función que implementa la estrategia
            initial_balance: Balance inicial
            commission: Comisión por operación
            scenarios: Diccionario con escenarios de estrés (None = usar por defecto)
            
        Returns:
            Diccionario con resultados por escenario
        """
        # Definir escenarios por defecto si no se proporcionan
        if scenarios is None:
            scenarios = {
                'baseline': {
                    'mu': 0.05,  # 5% rendimiento anualizado
                    'sigma': 0.15,  # 15% volatilidad anualizada
                    'days': 252,  # 1 año
                    'initial_price': 100.0
                },
                'bear_market': {
                    'mu': -0.20,  # -20% rendimiento anualizado
                    'sigma': 0.30,  # 30% volatilidad anualizada
                    'days': 126,  # 6 meses
                    'initial_price': 100.0
                },
                'sharp_crash': {
                    'initial_price': 100.0,
                    'mu': 0.05,
                    'sigma': 0.15,
                    'jump_intensity': 10,  # Alta intensidad de saltos
                    'jump_mean': -0.10,  # Saltos negativos
                    'jump_std': 0.05,
                    'days': 63  # 3 meses
                },
                'volatile_sideways': {
                    'initial_price': 100.0,
                    'mu': 0.0,  # Sin tendencia
                    'sigma': 0.40,  # Alta volatilidad
                    'days': 126  # 6 meses
                },
                'regime_switching': {
                    'initial_price': 100.0,
                    'mu_regimes': [0.10, -0.20, 0.0],  # Tendencia, corrección, lateral
                    'sigma_regimes': [0.10, 0.30, 0.15],
                    'transition_matrix': np.array([
                        [0.98, 0.01, 0.01],  # Probabilidades de transición
                        [0.01, 0.95, 0.04],
                        [0.02, 0.03, 0.95]
                    ]),
                    'days': 252  # 1 año
                }
            }
        
        # Resultados por escenario
        stress_results = {}
        
        # Ejecutar cada escenario
        for scenario_name, params in scenarios.items():
            self.logger.info(f"Ejecutando escenario de estrés: {scenario_name}")
            
            # Generar caminos de precio según el escenario
            if 'jump_intensity' in params:
                # Escenario con saltos
                price_paths = self.generate_gbm_with_jumps(
                    initial_price=params['initial_price'],
                    mu=params['mu'],
                    sigma=params['sigma'],
                    jump_intensity=params['jump_intensity'],
                    jump_mean=params['jump_mean'],
                    jump_std=params['jump_std'],
                    days=params['days']
                )
            elif 'mu_regimes' in params:
                # Escenario con cambios de régimen
                price_paths, _ = self.generate_regime_switching_paths(
                    initial_price=params['initial_price'],
                    mu_regimes=params['mu_regimes'],
                    sigma_regimes=params['sigma_regimes'],
                    transition_matrix=params['transition_matrix'],
                    days=params['days']
                )
            else:
                # Escenario con GBM simple
                price_paths = self.generate_gbm_paths(
                    initial_price=params['initial_price'],
                    mu=params['mu'],
                    sigma=params['sigma'],
                    days=params['days']
                )
            
            # Simular estrategia
            results = self.simulate_strategy(
                price_paths=price_paths,
                strategy_func=strategy_func,
                initial_balance=initial_balance,
                commission=commission,
                days=params['days']
            )
            
            # Calcular métricas de riesgo
            risk_metrics = self.calculate_risk_metrics(results)
            
            # Guardar resultados
            stress_results[scenario_name] = {
                'simulation_results': results,
                'risk_metrics': risk_metrics
            }
            
            # Generar gráfico
            chart_base64 = self.plot_simulation_results(results)
            stress_results[scenario_name]['chart'] = chart_base64
            
            # Guardar gráfico
            chart_path = os.path.join(self.output_dir, f"stress_test_{scenario_name}.png")
            with open(chart_path, 'wb') as f:
                f.write(base64.b64decode(chart_base64))
            
            # Guardar métricas
            metrics_path = os.path.join(self.output_dir, f"stress_test_{scenario_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(risk_metrics, f, indent=2)
        
        return stress_results
    
    def optimize_strategy_parameters(self, 
                                   strategy_factory: Callable,
                                   param_grid: Dict[str, List[Any]],
                                   initial_price: float = 100.0,
                                   mu: float = 0.05,
                                   sigma: float = 0.15,
                                   days: int = 252,
                                   initial_balance: float = 10000.0,
                                   commission: float = 0.001,
                                   optimization_metric: str = 'sharpe_ratio',
                                   n_simulations_per_param: int = 100) -> Dict[str, Any]:
        """
        Optimizar parámetros de una estrategia mediante búsqueda en grid.
        
        Args:
            strategy_factory: Función que genera una estrategia a partir de parámetros
            param_grid: Diccionario con parámetros a optimizar
            initial_price: Precio inicial para simulaciones
            mu: Rendimiento esperado anualizado
            sigma: Volatilidad anualizada
            days: Número de días a simular
            initial_balance: Balance inicial
            commission: Comisión por operación
            optimization_metric: Métrica a optimizar
            n_simulations_per_param: Número de simulaciones por combinación de parámetros
            
        Returns:
            Diccionario con resultados de la optimización
        """
        # Verificar métrica
        valid_metrics = [
            'mean_return', 'sharpe_ratio', 'sortino_ratio', 
            'calmar_ratio', 'omega_ratio'
        ]
        
        if optimization_metric not in valid_metrics:
            raise ValueError(f"Métrica de optimización no válida. Use una de: {valid_metrics}")
        
        # Generar todas las combinaciones de parámetros
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"Optimizando con {len(param_combinations)} combinaciones de parámetros")
        
        # Resultados por combinación de parámetros
        optimization_results = []
        
        # Guardar número de simulaciones original
        original_n_simulations = self.n_simulations
        self.n_simulations = n_simulations_per_param
        
        # Generar caminos de precio una vez para todas las pruebas
        price_paths = self.generate_gbm_paths(
            initial_price=initial_price,
            mu=mu,
            sigma=sigma,
            days=days
        )
        
        # Evaluar cada combinación
        for i, param_values in enumerate(param_combinations):
            # Crear diccionario de parámetros
            params = dict(zip(param_names, param_values))
            self.logger.info(f"Evaluando combinación {i+1}/{len(param_combinations)}: {params}")
            
            # Crear estrategia con estos parámetros
            strategy_func = strategy_factory(**params)
            
            # Simular estrategia
            results = self.simulate_strategy(
                price_paths=price_paths,
                strategy_func=strategy_func,
                initial_balance=initial_balance,
                commission=commission,
                days=days
            )
            
            # Calcular métricas de riesgo
            risk_metrics = self.calculate_risk_metrics(results)
            
            # Guardar resultados
            optimization_results.append({
                'params': params,
                'risk_metrics': risk_metrics,
                'metric_value': risk_metrics[optimization_metric]
            })
        
        # Restaurar número de simulaciones original
        self.n_simulations = original_n_simulations
        
        # Ordenar resultados por la métrica de optimización (mayor primero)
        optimization_results.sort(key=lambda x: x['metric_value'], reverse=True)
        
        # Mejores parámetros
        best_params = optimization_results[0]['params']
        best_metric = optimization_results[0]['metric_value']
        
        self.logger.info(f"Optimización completada. Mejores parámetros: {best_params}, {optimization_metric}: {best_metric}")
        
        # Preparar resultado final
        result = {
            'best_params': best_params,
            'best_metric_value': best_metric,
            'optimization_metric': optimization_metric,
            'all_results': optimization_results
        }
        
        # Guardar resultados
        results_path = os.path.join(self.output_dir, "parameter_optimization.json")
        with open(results_path, 'w') as f:
            # Eliminar arrays grandes para el guardado
            save_results = {
                'best_params': best_params,
                'best_metric_value': best_metric,
                'optimization_metric': optimization_metric,
                'all_results': [
                    {
                        'params': r['params'],
                        'metric_value': r['metric_value'],
                        'risk_metrics': r['risk_metrics']
                    }
                    for r in optimization_results
                ]
            }
            json.dump(save_results, f, indent=2)
        
        # Generar gráfico de comparación
        self._plot_optimization_results(optimization_results, optimization_metric)
        
        return result
    
    def _plot_optimization_results(self, 
                                 optimization_results: List[Dict[str, Any]], 
                                 metric: str) -> None:
        """
        Generar gráfico de resultados de optimización.
        
        Args:
            optimization_results: Lista de resultados por combinación de parámetros
            metric: Métrica optimizada
        """
        # Extraer parámetros y valores de métrica
        params = [r['params'] for r in optimization_results]
        metric_values = [r['metric_value'] for r in optimization_results]
        
        # Obtener nombres de parámetros
        param_names = list(params[0].keys())
        
        # Si hay muchos parámetros, mostrar solo los más relevantes
        if len(param_names) > 2:
            # Identificar los parámetros más influyentes
            from sklearn.ensemble import RandomForestRegressor
            
            # Preparar datos
            X = pd.DataFrame(params)
            y = np.array(metric_values)
            
            # Entrenar Random Forest para identificar importancia
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Obtener importancia de características
            importances = rf.feature_importances_
            
            # Seleccionar los 2 parámetros más importantes
            top_indices = importances.argsort()[-2:]
            param_names = [param_names[i] for i in top_indices]
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Si hay un solo parámetro relevante
        if len(param_names) == 1:
            param_name = param_names[0]
            param_values = [p[param_name] for p in params]
            
            # Crear gráfico de dispersión
            ax.scatter(param_values, metric_values)
            
            # Ajustar curva suavizada si hay suficientes puntos
            if len(param_values) >= 5:
                from scipy.interpolate import interp1d
                
                # Ordenar datos para interpolación
                sorted_indices = np.argsort(param_values)
                sorted_x = np.array(param_values)[sorted_indices]
                sorted_y = np.array(metric_values)[sorted_indices]
                
                # Eliminar duplicados para interpolación
                unique_x, unique_indices = np.unique(sorted_x, return_index=True)
                unique_y = sorted_y[unique_indices]
                
                if len(unique_x) >= 3:  # Necesitamos al menos 3 puntos para interpolación cúbica
                    # Interpolación cúbica
                    f = interp1d(unique_x, unique_y, kind='cubic', bounds_error=False, fill_value="extrapolate")
                    
                    # Generar puntos suavizados
                    x_smooth = np.linspace(min(param_values), max(param_values), 100)
                    y_smooth = f(x_smooth)
                    
                    # Graficar curva suavizada
                    ax.plot(x_smooth, y_smooth, 'r-')
            
            # Formato del gráfico
            ax.set_title(f'Optimización de {metric}')
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            
        # Si hay dos parámetros relevantes
        elif len(param_names) == 2:
            param1_name, param2_name = param_names
            param1_values = [p[param1_name] for p in params]
            param2_values = [p[param2_name] for p in params]
            
            # Crear mapa de calor
            # Primero identificar valores únicos de cada parámetro
            unique_param1 = sorted(set(param1_values))
            unique_param2 = sorted(set(param2_values))
            
            # Crear matriz para el mapa de calor
            heatmap_data = np.zeros((len(unique_param2), len(unique_param1)))
            heatmap_counts = np.zeros((len(unique_param2), len(unique_param1)))
            
            # Llenar matriz con valores de la métrica
            for p1, p2, mv in zip(param1_values, param2_values, metric_values):
                i = unique_param2.index(p2)
                j = unique_param1.index(p1)
                heatmap_data[i, j] += mv
                heatmap_counts[i, j] += 1
            
            # Calcular promedio
            mask = heatmap_counts > 0
            heatmap_data[mask] /= heatmap_counts[mask]
            
            # Crear mapa de calor
            im = ax.imshow(heatmap_data, cmap='viridis', interpolation='nearest', aspect='auto')
            
            # Etiquetas de ejes
            ax.set_xticks(np.arange(len(unique_param1)))
            ax.set_yticks(np.arange(len(unique_param2)))
            ax.set_xticklabels(unique_param1)
            ax.set_yticklabels(unique_param2)
            
            # Rotar etiquetas del eje x
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Añadir valores en cada celda
            for i in range(len(unique_param2)):
                for j in range(len(unique_param1)):
                    if mask[i, j]:
                        ax.text(j, i, f"{heatmap_data[i, j]:.2f}", ha="center", va="center", color="w" if heatmap_data[i, j] < np.max(heatmap_data) * 0.7 else "black")
            
            # Añadir colorbar
            plt.colorbar(im, ax=ax)
            
            # Formato del gráfico
            ax.set_title(f'Optimización de {metric}')
            ax.set_xlabel(param1_name)
            ax.set_ylabel(param2_name)
        
        plt.tight_layout()
        
        # Guardar gráfico
        save_path = os.path.join(self.output_dir, f"optimization_{metric}.png")
        plt.savefig(save_path)
        plt.close()