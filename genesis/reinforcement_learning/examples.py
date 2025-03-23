"""
Ejemplos de uso del módulo de Reinforcement Learning.

Este script proporciona ejemplos prácticos sobre cómo utilizar los componentes
de Reinforcement Learning para trading en el sistema Genesis.
"""

import pandas as pd
import numpy as np
import logging
import time
import os
import asyncio
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

# Importar componentes de RL
from genesis.reinforcement_learning.environments import TradingEnvironment, MultiAssetTradingEnvironment
from genesis.reinforcement_learning.agents import RLAgent, DQNAgent, PPOAgent, SACAgent, RLAgentManager
from genesis.reinforcement_learning.evaluation import BacktestAgent, HyperparameterOptimizer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Función para generar datos sintéticos para ejemplos
def generate_example_data(start_date: str = '2020-01-01', 
                          end_date: str = '2021-01-01',
                          n_assets: int = 1,
                          freq: str = 'D',
                          volatility: float = 0.02,
                          with_trend: bool = True,
                          seed: int = 42) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Generar datos sintéticos de precios para ejemplos.
    
    Args:
        start_date: Fecha de inicio en formato 'YYYY-MM-DD'
        end_date: Fecha de fin en formato 'YYYY-MM-DD'
        n_assets: Número de activos a generar
        freq: Frecuencia de datos ('D'=diaria, 'H'=horaria)
        volatility: Volatilidad de los precios
        with_trend: Si es True, agrega tendencia
        seed: Semilla para reproducibilidad
        
    Returns:
        DataFrame o diccionario de DataFrames con datos sintéticos
    """
    np.random.seed(seed)
    
    # Generar fechas
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_periods = len(date_range)
    
    # Función para generar un activo
    def generate_asset(asset_id: int) -> pd.DataFrame:
        # Generar retornos aleatorios
        returns = np.random.normal(0, volatility, n_periods)
        
        # Agregar tendencia si se solicita
        if with_trend:
            # Tendencia principal (creciente, decreciente o lateral)
            trend_type = np.random.choice(['up', 'down', 'sideways'])
            
            if trend_type == 'up':
                trend = np.linspace(0, 0.5, n_periods) * volatility
            elif trend_type == 'down':
                trend = np.linspace(0, -0.5, n_periods) * volatility
            else:
                trend = np.zeros(n_periods)
            
            # Agregar ciclos y regímenes de mercado
            cycles = 0.5 * volatility * np.sin(np.linspace(0, np.random.randint(2, 5) * np.pi, n_periods))
            
            # Combinar
            returns = returns + trend + cycles
        
        # Convertir retornos a precios
        close = 100 * np.cumprod(1 + returns)
        
        # Generar OHLCV a partir del cierre
        high = close * np.random.uniform(1.0, 1.0 + volatility, n_periods)
        low = close * np.random.uniform(1.0 - volatility, 1.0, n_periods)
        open_price = low + np.random.random(n_periods) * (high - low)
        volume = np.random.lognormal(10, 1, n_periods)
        
        # Crear DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        # Calcular algunos indicadores
        df['rsi'] = calculate_rsi(df['close'])
        df['sma_short'] = df['close'].rolling(window=5).mean()
        df['sma_long'] = df['close'].rolling(window=20).mean()
        df['atr'] = calculate_atr(df)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # Rellenar NaN
        df.fillna(method='bfill', inplace=True)
        
        return df
    
    # Generar activos
    if n_assets == 1:
        return generate_asset(0)
    else:
        dfs = {}
        for i in range(n_assets):
            symbol = f"ASSET_{i+1}"
            dfs[symbol] = generate_asset(i)
        return dfs

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calcular RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calcular ATR."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr

# Ejemplo 1: Entrenamiento básico de un agente RL
async def example_single_asset_training():
    """
    Ejemplo de entrenamiento de un agente RL para un solo activo.
    """
    logger.info("Ejemplo 1: Entrenamiento básico de un agente RL para un solo activo")
    
    # Generar datos de ejemplo
    df = generate_example_data(
        start_date='2020-01-01',
        end_date='2021-01-01',
        freq='D',
        with_trend=True
    )
    
    # Crear entorno de trading
    env = TradingEnvironment(
        df=df,
        initial_balance=10000.0,
        commission=0.001,
        window_size=20,
        features=['rsi', 'sma_short', 'sma_long', 'atr', 'volatility']
    )
    
    # Crear un agente PPO
    agent = PPOAgent(
        env=env,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )
    
    # Entrenar agente (versión corta para ejemplo)
    training_results = agent.train(
        total_timesteps=10000,  # Pocas iteraciones para ejemplo
        log_dir='./logs/examples',
        eval_freq=5000,
        n_eval_episodes=3,
        save_path='./cache/rl_models/example_ppo'
    )
    
    logger.info(f"Entrenamiento completado: {training_results}")
    
    # Crear backtester para evaluar rendimiento
    backtester = BacktestAgent(
        env=env,
        agent=agent,
        output_dir='./results/examples'
    )
    
    # Ejecutar backtest
    backtest_results = backtester.run_backtest(
        n_episodes=3,
        deterministic=True,
        render=True
    )
    
    logger.info(f"Backtest completado: {backtest_results}")
    
    return agent, backtest_results

# Ejemplo 2: Optimización de hiperparámetros
async def example_hyperparameter_optimization():
    """
    Ejemplo de optimización de hiperparámetros para un agente RL.
    """
    logger.info("Ejemplo 2: Optimización de hiperparámetros para un agente RL")
    
    # Función para crear entorno con datos sintéticos
    def create_env():
        df = generate_example_data(
            start_date='2020-01-01',
            end_date='2021-01-01',
            freq='D',
            with_trend=True,
            seed=np.random.randint(0, 10000)  # Diferentes semillas para variabilidad
        )
        
        return TradingEnvironment(
            df=df,
            initial_balance=10000.0,
            commission=0.001,
            window_size=20,
            features=['rsi', 'sma_short', 'sma_long', 'atr', 'volatility']
        )
    
    # Crear optimizador
    optimizer = HyperparameterOptimizer(
        env_creator=create_env,
        agent_type='ppo',
        study_name='example_optimization',
        n_trials=5,  # Pocas pruebas para ejemplo
        n_eval_episodes=2,
        eval_timesteps=5000,  # Pocas iteraciones para ejemplo
        output_dir='./results/examples'
    )
    
    # Ejecutar optimización
    optimization_results = optimizer.optimize()
    
    logger.info(f"Optimización completada: {optimization_results}")
    
    # Crear agente con los mejores hiperparámetros
    best_agent = optimizer.create_best_agent()
    
    # Entrenar brevemente
    best_agent.train(
        total_timesteps=5000,  # Pocas iteraciones para ejemplo
        log_dir='./logs/examples',
        save_path='./cache/rl_models/example_best_ppo'
    )
    
    # Crear entorno para evaluación
    eval_env = create_env()
    
    # Crear backtester
    backtester = BacktestAgent(
        env=eval_env,
        agent=best_agent,
        output_dir='./results/examples'
    )
    
    # Ejecutar backtest
    backtest_results = backtester.run_backtest(
        n_episodes=2,
        deterministic=True,
        render=True
    )
    
    logger.info(f"Backtest completado con agente optimizado: {backtest_results}")
    
    return best_agent, backtest_results

# Ejemplo 3: Trading con múltiples activos
async def example_multi_asset_trading():
    """
    Ejemplo de entrenamiento de un agente RL para múltiples activos.
    """
    logger.info("Ejemplo 3: Trading con múltiples activos")
    
    # Generar datos de ejemplo para múltiples activos
    dfs = generate_example_data(
        start_date='2020-01-01',
        end_date='2021-01-01',
        n_assets=3,  # 3 activos diferentes
        freq='D',
        with_trend=True
    )
    
    # Crear entorno de trading multi-activo
    env = MultiAssetTradingEnvironment(
        dfs=dfs,
        initial_balance=10000.0,
        commission=0.001,
        window_size=20,
        features=['rsi', 'sma_short', 'sma_long', 'atr', 'volatility']
    )
    
    # Crear gestor de agentes
    agent_manager = RLAgentManager(
        cache_dir='./cache/rl_models',
        logs_dir='./logs/examples'
    )
    
    # Crear agentes diferentes
    dqn_id = await agent_manager.create_agent(
        symbol="MULTI_DQN",
        env=env,
        agent_type='dqn',
        agent_params={
            'learning_rate': 0.0001,
            'buffer_size': 50000,
            'batch_size': 64
        }
    )
    
    ppo_id = await agent_manager.create_agent(
        symbol="MULTI_PPO",
        env=env,
        agent_type='ppo',
        agent_params={
            'learning_rate': 0.0001,
            'n_steps': 2048,
            'batch_size': 64
        }
    )
    
    # Entrenar brevemente los agentes
    for agent_id in [dqn_id, ppo_id]:
        await agent_manager.train_agent(
            agent_id=agent_id,
            total_timesteps=5000,  # Pocas iteraciones para ejemplo
            eval_freq=0,
            save_model=True
        )
    
    # Crear backtester
    backtester = BacktestAgent(
        env=env,
        agent=agent_manager.agents[ppo_id],  # Usar PPO para el backtest
        output_dir='./results/examples'
    )
    
    # Ejecutar backtest con ambos agentes
    dqn_agent = agent_manager.agents[dqn_id]
    ppo_agent = agent_manager.agents[ppo_id]
    
    # Comparar agentes
    comparison = backtester.compare_agents(
        agents=[dqn_agent, ppo_agent],
        agent_names=["DQN", "PPO"],
        n_episodes=2,
        deterministic=True,
        plot_results=True
    )
    
    logger.info(f"Comparación completada: {comparison}")
    
    return agent_manager, comparison

# Función para ejecutar todos los ejemplos
async def run_all_examples():
    """
    Ejecutar todos los ejemplos.
    """
    # Crear directorios necesarios
    os.makedirs('./logs/examples', exist_ok=True)
    os.makedirs('./cache/rl_models', exist_ok=True)
    os.makedirs('./results/examples', exist_ok=True)
    
    logger.info("Ejecutando todos los ejemplos...")
    
    # Ejemplo 1
    agent, results = await example_single_asset_training()
    logger.info(f"Ejemplo 1 completado: Retorno medio = {results['mean_return']:.2f}")
    
    # Ejemplo 2
    best_agent, opt_results = await example_hyperparameter_optimization()
    logger.info(f"Ejemplo 2 completado: Retorno optimizado = {opt_results['mean_return']:.2f}")
    
    # Ejemplo 3
    agent_manager, comparison = await example_multi_asset_trading()
    logger.info(f"Ejemplo 3 completado: {len(comparison['results'])} agentes comparados")
    
    logger.info("Todos los ejemplos completados correctamente")

# Función principal para ejecutar desde línea de comandos
if __name__ == "__main__":
    asyncio.run(run_all_examples())