"""
Utilidades para la evaluación y experimentación con modelos de Reinforcement Learning.

Este módulo proporciona herramientas para evaluar agentes RL, ejecutar experimentos
de hiperparámetros y visualizar resultados de entrenamiento.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from io import BytesIO
import base64
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
import optuna
from optuna.visualization import plot_contour, plot_optimization_history, plot_param_importances
import copy
import pickle

# Importar componentes de RL
from genesis.reinforcement_learning.environments import TradingEnvironment, MultiAssetTradingEnvironment
from genesis.reinforcement_learning.agents import RLAgent, DQNAgent, PPOAgent, SACAgent, RLAgentManager

# Verificar disponibilidad de estable-baselines3
try:
    import stable_baselines3 as sb3
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

class BacktestAgent:
    """
    Ejecutor de backtests para agentes RL entrenados.
    
    Esta clase permite realizar backtests de agentes RL en entornos históricos
    y calcular métricas de rendimiento detalladas.
    """
    
    def __init__(self, 
                 env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
                 agent: Optional[RLAgent] = None,
                 output_dir: str = './results/rl_backtests',
                 verbose: bool = True):
        """
        Inicializar executor de backtests.
        
        Args:
            env: Entorno de trading para backtest
            agent: Agente RL entrenado (opcional)
            output_dir: Directorio para resultados
            verbose: Si es True, muestra mensajes de progreso
        """
        self.logger = logging.getLogger(__name__)
        self.env = env
        self.agent = agent
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Resultados de backtests
        self.backtest_results = []
        
        self.logger.info("Inicializado ejecutor de backtests para agentes RL")
    
    def run_backtest(self, 
                     agent: Optional[RLAgent] = None,
                     n_episodes: int = 1,
                     deterministic: bool = True,
                     render: bool = False,
                     save_results: bool = True) -> Dict[str, Any]:
        """
        Ejecutar backtest del agente en el entorno.
        
        Args:
            agent: Agente RL a evaluar (si None, usa self.agent)
            n_episodes: Número de episodios para backtest
            deterministic: Si es True, usa política determinista
            render: Si es True, renderiza el entorno al final
            save_results: Si es True, guarda resultados
            
        Returns:
            Diccionario con resultados del backtest
        """
        # Verificar agente
        if agent is None:
            agent = self.agent
        
        if agent is None or agent.model is None:
            raise ValueError("Se requiere un agente entrenado para el backtest")
        
        # Verificar disponibilidad de SB3
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 no está disponible. Instala con: pip install stable-baselines3")
        
        # Resultados por episodio
        episode_returns = []
        episode_lengths = []
        win_rates = []
        drawdowns = []
        trades_per_episode = []
        
        # Progreso
        episodes_range = range(n_episodes)
        if self.verbose:
            episodes_range = tqdm(episodes_range, desc="Ejecutando backtests")
        
        for episode in episodes_range:
            # Resetear entorno
            obs, info = self.env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            steps = 0
            
            # Ejecutar episodio
            while not done and not truncated:
                # Predecir acción
                action, _ = agent.predict(obs, deterministic=deterministic)
                
                # Ejecutar acción en el entorno
                obs, reward, done, truncated, info = self.env.step(action)
                
                # Actualizar estadísticas
                total_reward += reward
                steps += 1
            
            # Calcular métricas del episodio
            episode_returns.append(total_reward)
            episode_lengths.append(steps)
            
            # Obtener resultados detallados del entorno
            if hasattr(self.env, 'get_trading_results'):
                trading_results = self.env.get_trading_results()
                win_rates.append(trading_results['win_rate'])
                drawdowns.append(trading_results['max_drawdown'])
                trades_per_episode.append(trading_results['num_trades'])
                
                # Guardar resultados del último episodio para visualización
                if episode == n_episodes - 1 and render:
                    if hasattr(self.env, 'plot_results'):
                        chart_base64 = self.env.plot_results()
                        
                        # Guardar gráfico si se solicita
                        if save_results and chart_base64:
                            timestamp = int(time.time())
                            chart_path = os.path.join(self.output_dir, f"backtest_chart_{timestamp}.png")
                            with open(chart_path, 'wb') as f:
                                f.write(base64.b64decode(chart_base64))
                            
                            self.logger.info(f"Gráfico de backtest guardado en {chart_path}")
        
        # Calcular métricas agregadas
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        mean_length = np.mean(episode_lengths)
        
        # Métricas de trading si están disponibles
        mean_win_rate = np.mean(win_rates) if win_rates else None
        mean_drawdown = np.mean(drawdowns) if drawdowns else None
        mean_trades = np.mean(trades_per_episode) if trades_per_episode else None
        
        # Construir resultados
        results = {
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'mean_episode_length': float(mean_length),
            'mean_win_rate': float(mean_win_rate) if mean_win_rate is not None else None,
            'mean_drawdown': float(mean_drawdown) if mean_drawdown is not None else None,
            'mean_trades': float(mean_trades) if mean_trades is not None else None,
            'agent_type': agent.model_type,
            'deterministic': deterministic,
            'n_episodes': n_episodes,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar resultados
        if save_results:
            timestamp = int(time.time())
            results_path = os.path.join(self.output_dir, f"backtest_results_{timestamp}.json")
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Resultados de backtest guardados en {results_path}")
        
        # Agregar a lista de resultados
        self.backtest_results.append(results)
        
        return results
    
    def compare_agents(self, 
                       agents: List[RLAgent],
                       agent_names: List[str],
                       n_episodes: int = 5,
                       deterministic: bool = True,
                       plot_results: bool = True) -> Dict[str, Any]:
        """
        Comparar rendimiento de múltiples agentes.
        
        Args:
            agents: Lista de agentes RL a comparar
            agent_names: Nombres para los agentes
            n_episodes: Número de episodios para cada agente
            deterministic: Si es True, usa política determinista
            plot_results: Si es True, genera gráficos comparativos
            
        Returns:
            Diccionario con resultados de la comparación
        """
        if len(agents) != len(agent_names):
            raise ValueError("El número de agentes y nombres debe coincidir")
        
        # Resultados por agente
        all_results = []
        
        self.logger.info(f"Comparando {len(agents)} agentes con {n_episodes} episodios cada uno...")
        
        # Ejecutar backtest para cada agente
        for i, (agent, name) in enumerate(zip(agents, agent_names)):
            if self.verbose:
                print(f"Evaluando agente {i+1}/{len(agents)}: {name}")
            
            # Guardar agente actual
            current_agent = self.agent
            
            # Configurar agente para backtest
            self.agent = agent
            
            # Ejecutar backtest
            results = self.run_backtest(
                n_episodes=n_episodes,
                deterministic=deterministic,
                render=False,
                save_results=False
            )
            
            # Agregar nombre del agente
            results['agent_name'] = name
            all_results.append(results)
            
            # Restaurar agente original
            self.agent = current_agent
        
        # Comparación de resultados
        comparison = {
            'results': all_results,
            'n_episodes': n_episodes,
            'deterministic': deterministic,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generar gráficos comparativos si se solicita
        if plot_results:
            # Gráfico de retornos
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Datos para el gráfico
            names = [r['agent_name'] for r in all_results]
            returns = [r['mean_return'] for r in all_results]
            errors = [r['std_return'] for r in all_results]
            
            # Crear gráfico de barras con error
            ax.bar(names, returns, yerr=errors, alpha=0.7, capsize=10)
            ax.set_ylabel('Retorno Medio')
            ax.set_title('Comparación de Retornos entre Agentes')
            ax.grid(axis='y', alpha=0.3)
            
            # Guardar gráfico
            comparison_chart_path = os.path.join(self.output_dir, f"agent_comparison_{int(time.time())}.png")
            plt.tight_layout()
            plt.savefig(comparison_chart_path)
            plt.close()
            
            self.logger.info(f"Gráfico comparativo guardado en {comparison_chart_path}")
            
            # Si hay métricas de trading, crear otro gráfico
            if all(r.get('mean_win_rate') is not None for r in all_results):
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Datos para el gráfico
                win_rates = [r['mean_win_rate'] * 100 for r in all_results]
                drawdowns = [r['mean_drawdown'] * 100 for r in all_results]
                trades = [r['mean_trades'] for r in all_results]
                
                # Índice para agentes
                x = np.arange(len(names))
                width = 0.25
                
                # Gráfico con múltiples métricas
                ax.bar(x - width, win_rates, width, label='Win Rate (%)')
                ax.bar(x, drawdowns, width, label='Max Drawdown (%)')
                ax.bar(x + width, trades, width, label='Trades')
                
                # Formato
                ax.set_ylabel('Valor')
                ax.set_title('Métricas de Trading por Agente')
                ax.set_xticks(x)
                ax.set_xticklabels(names)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                # Guardar gráfico
                trading_chart_path = os.path.join(self.output_dir, f"trading_metrics_comparison_{int(time.time())}.png")
                plt.tight_layout()
                plt.savefig(trading_chart_path)
                plt.close()
                
                self.logger.info(f"Gráfico de métricas de trading guardado en {trading_chart_path}")
        
        # Guardar comparación
        comparison_path = os.path.join(self.output_dir, f"agent_comparison_{int(time.time())}.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def get_last_backtest_results(self) -> Dict[str, Any]:
        """
        Obtener resultados del último backtest.
        
        Returns:
            Diccionario con resultados
        """
        if not self.backtest_results:
            return {}
        
        return self.backtest_results[-1]
    
    def plot_backtest_results(self, 
                              results: Optional[Dict[str, Any]] = None,
                              save_path: Optional[str] = None) -> str:
        """
        Generar gráfico de resultados de backtest.
        
        Args:
            results: Resultados de backtest (si None, usa el último backtest)
            save_path: Ruta donde guardar el gráfico
            
        Returns:
            Imagen en formato base64
        """
        if results is None:
            results = self.get_last_backtest_results()
        
        if not results:
            return ""
        
        # Resultado final del trading (de la instancia del entorno)
        if hasattr(self.env, 'plot_results'):
            chart_base64 = self.env.plot_results()
            
            # Guardar gráfico si se solicita
            if save_path and chart_base64:
                with open(save_path, 'wb') as f:
                    f.write(base64.b64decode(chart_base64))
                
                self.logger.info(f"Gráfico de backtest guardado en {save_path}")
            
            return chart_base64
        
        # Si el entorno no tiene método plot_results, crear un gráfico básico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Datos básicos
        metrics = {
            'Retorno Medio': results.get('mean_return', 0),
            'Desviación Estándar': results.get('std_return', 0)
        }
        
        # Agregar métricas de trading si están disponibles
        if results.get('mean_win_rate') is not None:
            metrics['Win Rate (%)'] = results.get('mean_win_rate', 0) * 100
        
        if results.get('mean_drawdown') is not None:
            metrics['Max Drawdown (%)'] = results.get('mean_drawdown', 0) * 100
        
        if results.get('mean_trades') is not None:
            metrics['Operaciones'] = results.get('mean_trades', 0)
        
        # Crear gráfico de barras
        ax.bar(list(metrics.keys()), list(metrics.values()), alpha=0.7)
        ax.set_ylabel('Valor')
        ax.set_title(f"Resultados de Backtest - {results.get('agent_type', 'Agente RL')}")
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Guardar gráfico si se solicita
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(base64.b64decode(img_str))
            
            self.logger.info(f"Gráfico de backtest guardado en {save_path}")
        
        return img_str


class HyperparameterOptimizer:
    """
    Optimizador de hiperparámetros para agentes RL.
    
    Esta clase utiliza Optuna para realizar búsqueda de hiperparámetros
    óptimos para agentes RL aplicados a trading.
    """
    
    def __init__(self, 
                 env_creator: Callable[[], Union[TradingEnvironment, MultiAssetTradingEnvironment]],
                 agent_type: str = 'ppo',
                 study_name: str = 'rl_optimization',
                 n_trials: int = 20,
                 timeout: Optional[int] = None,
                 n_eval_episodes: int = 3,
                 eval_timesteps: int = 50000,
                 db_storage: Optional[str] = None,
                 output_dir: str = './results/rl_optimization',
                 verbose: bool = True):
        """
        Inicializar optimizador de hiperparámetros.
        
        Args:
            env_creator: Función que crea una instancia del entorno
            agent_type: Tipo de agente RL ('dqn', 'ppo', 'sac')
            study_name: Nombre del estudio de optimización
            n_trials: Número de pruebas para optimización
            timeout: Tiempo límite en segundos (opcional)
            n_eval_episodes: Número de episodios para evaluación
            eval_timesteps: Pasos de entrenamiento para evaluación
            db_storage: URL de almacenamiento para Optuna (opcional)
            output_dir: Directorio para resultados
            verbose: Si es True, muestra mensajes de progreso
        """
        # Verificar disponibilidad de requisitos
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 no está disponible. Instala con: pip install stable-baselines3")
        
        self.logger = logging.getLogger(__name__)
        self.env_creator = env_creator
        self.agent_type = agent_type.lower()
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_eval_episodes = n_eval_episodes
        self.eval_timesteps = eval_timesteps
        self.db_storage = db_storage
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Crear entorno inicial para validación
        self.env = env_creator()
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Verificar tipo de agente
        if agent_type not in ['dqn', 'ppo', 'sac']:
            raise ValueError(f"Tipo de agente no soportado: {agent_type}. Use 'dqn', 'ppo' o 'sac'")
        
        # Estudio de Optuna
        self.study = None
        
        self.logger.info(f"Inicializado optimizador de hiperparámetros para agentes {agent_type}")
    
    def _create_agent(self, trial: optuna.Trial) -> RLAgent:
        """
        Crear agente RL con hiperparámetros sugeridos por Optuna.
        
        Args:
            trial: Prueba de Optuna
            
        Returns:
            Agente RL
        """
        # Crear entorno nuevo para este trial
        env = self.env_creator()
        
        # Hiperparámetros comunes
        policy_type = 'MlpPolicy'
        seed = 42
        
        # Hiperparámetros específicos según el tipo de agente
        if self.agent_type == 'dqn':
            # Hiperparámetros para DQN
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            buffer_size = trial.suggest_categorical('buffer_size', [10000, 50000, 100000])
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
            gamma = trial.suggest_float('gamma', 0.9, 0.9999)
            exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.5)
            exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.1)
            target_update_interval = trial.suggest_int('target_update_interval', 100, 1000)
            
            # Crear agente DQN
            agent = DQNAgent(
                env=env,
                policy_type=policy_type,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                gamma=gamma,
                exploration_fraction=exploration_fraction,
                exploration_final_eps=exploration_final_eps,
                target_update_interval=target_update_interval,
                seed=seed,
                verbose=0
            )
            
        elif self.agent_type == 'ppo':
            # Hiperparámetros para PPO
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            n_steps = trial.suggest_categorical('n_steps', [64, 128, 256, 512, 1024, 2048])
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
            n_epochs = trial.suggest_int('n_epochs', 3, 30)
            gamma = trial.suggest_float('gamma', 0.9, 0.9999)
            gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.999)
            clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
            ent_coef = trial.suggest_float('ent_coef', 0.0, 0.01)
            
            # Crear agente PPO
            agent = PPOAgent(
                env=env,
                policy_type=policy_type,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                seed=seed,
                verbose=0
            )
            
        elif self.agent_type == 'sac':
            # Hiperparámetros para SAC
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            buffer_size = trial.suggest_categorical('buffer_size', [10000, 50000, 100000])
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
            gamma = trial.suggest_float('gamma', 0.9, 0.9999)
            tau = trial.suggest_float('tau', 0.001, 0.01)
            train_freq = trial.suggest_categorical('train_freq', [1, 4, 8])
            gradient_steps = trial.suggest_categorical('gradient_steps', [1, 2, 4])
            
            # Crear agente SAC
            agent = SACAgent(
                env=env,
                policy_type=policy_type,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                gamma=gamma,
                tau=tau,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                seed=seed,
                verbose=0
            )
        
        return agent
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Función objetivo para optimización de hiperparámetros.
        
        Args:
            trial: Prueba de Optuna
            
        Returns:
            Recompensa media negativa (para minimizar)
        """
        # Crear agente con los hiperparámetros sugeridos
        agent = self._create_agent(trial)
        
        try:
            # Entrenar agente
            if self.verbose:
                print(f"Trial {trial.number}: Entrenando agente con hiperparámetros: {trial.params}")
            
            agent.train(
                total_timesteps=self.eval_timesteps,
                log_dir=None,
                eval_freq=0,
                save_path=None
            )
            
            # Evaluar agente
            env = self.env_creator()  # Nuevo entorno para evaluación
            mean_reward, std_reward = evaluate_policy(
                agent.model, 
                env, 
                n_eval_episodes=self.n_eval_episodes, 
                deterministic=True
            )
            
            if self.verbose:
                print(f"Trial {trial.number}: Recompensa media = {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Guardar información adicional
            trial.set_user_attr('std_reward', float(std_reward))
            trial.set_user_attr('agent_type', self.agent_type)
            trial.set_user_attr('n_eval_episodes', self.n_eval_episodes)
            
            # Liberar recursos
            del agent
            
            # Retornar recompensa negativa (para minimizar)
            return -float(mean_reward)
            
        except Exception as e:
            self.logger.error(f"Error en trial {trial.number}: {str(e)}")
            # Si hay error, una recompensa muy negativa
            return -float('-inf')
    
    def optimize(self) -> Dict[str, Any]:
        """
        Ejecutar optimización de hiperparámetros.
        
        Returns:
            Diccionario con resultados de la optimización
        """
        self.logger.info(f"Iniciando optimización con {self.n_trials} trials")
        
        # Crear estudio de Optuna
        if self.db_storage:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.db_storage,
                load_if_exists=True,
                direction='maximize'
            )
        else:
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction='maximize'
            )
        
        # Ejecutar optimización
        try:
            self.study.optimize(
                lambda trial: -self._objective(trial),  # Negamos para maximizar
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=self.verbose
            )
        except KeyboardInterrupt:
            self.logger.info("Optimización interrumpida por el usuario")
        
        # Obtener mejores hiperparámetros
        best_params = self.study.best_params
        best_value = -self.study.best_value  # Convertir a valor real
        
        self.logger.info(f"Optimización completada. Mejor recompensa: {best_value:.2f}")
        self.logger.info(f"Mejores hiperparámetros: {best_params}")
        
        # Generar visualizaciones
        try:
            # Historia de optimización
            fig = plot_optimization_history(self.study)
            fig.write_image(os.path.join(self.output_dir, f"{self.study_name}_history.png"))
            
            # Importancia de parámetros
            fig = plot_param_importances(self.study)
            fig.write_image(os.path.join(self.output_dir, f"{self.study_name}_param_importance.png"))
            
            # Contorno para pares seleccionados (si hay suficientes parámetros)
            if len(best_params) >= 2:
                param_names = list(best_params.keys())[:2]
                fig = plot_contour(self.study, params=param_names)
                fig.write_image(os.path.join(self.output_dir, f"{self.study_name}_contour.png"))
        except Exception as e:
            self.logger.warning(f"Error generando visualizaciones: {str(e)}")
        
        # Guardar resultados
        results = {
            'best_params': best_params,
            'best_reward': float(best_value),
            'agent_type': self.agent_type,
            'n_trials': self.n_trials,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar en archivo
        results_path = os.path.join(self.output_dir, f"{self.study_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Resultados guardados en {results_path}")
        
        return results
    
    def create_best_agent(self, env: Optional[Union[TradingEnvironment, MultiAssetTradingEnvironment]] = None) -> RLAgent:
        """
        Crear agente con los mejores hiperparámetros encontrados.
        
        Args:
            env: Entorno para el agente (si None, crea uno nuevo)
            
        Returns:
            Agente RL con los mejores hiperparámetros
        """
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")
        
        # Crear entorno si no se proporciona
        if env is None:
            env = self.env_creator()
        
        # Obtener mejores hiperparámetros
        best_params = self.study.best_params
        
        # Hiperparámetros comunes
        policy_type = 'MlpPolicy'
        seed = 42
        
        # Crear agente según el tipo
        if self.agent_type == 'dqn':
            agent = DQNAgent(
                env=env,
                policy_type=policy_type,
                learning_rate=best_params.get('learning_rate'),
                buffer_size=best_params.get('buffer_size'),
                batch_size=best_params.get('batch_size'),
                gamma=best_params.get('gamma'),
                exploration_fraction=best_params.get('exploration_fraction'),
                exploration_final_eps=best_params.get('exploration_final_eps'),
                target_update_interval=best_params.get('target_update_interval'),
                seed=seed,
                verbose=1
            )
        elif self.agent_type == 'ppo':
            agent = PPOAgent(
                env=env,
                policy_type=policy_type,
                learning_rate=best_params.get('learning_rate'),
                n_steps=best_params.get('n_steps'),
                batch_size=best_params.get('batch_size'),
                n_epochs=best_params.get('n_epochs'),
                gamma=best_params.get('gamma'),
                gae_lambda=best_params.get('gae_lambda'),
                clip_range=best_params.get('clip_range'),
                ent_coef=best_params.get('ent_coef'),
                seed=seed,
                verbose=1
            )
        elif self.agent_type == 'sac':
            agent = SACAgent(
                env=env,
                policy_type=policy_type,
                learning_rate=best_params.get('learning_rate'),
                buffer_size=best_params.get('buffer_size'),
                batch_size=best_params.get('batch_size'),
                gamma=best_params.get('gamma'),
                tau=best_params.get('tau'),
                train_freq=best_params.get('train_freq'),
                gradient_steps=best_params.get('gradient_steps'),
                seed=seed,
                verbose=1
            )
        
        return agent
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """
        Obtener resultados de la optimización.
        
        Returns:
            Diccionario con resultados
        """
        if self.study is None:
            return {}
        
        # Resultados básicos
        results = {
            'best_params': self.study.best_params,
            'best_reward': -self.study.best_value,  # Convertir a valor real
            'agent_type': self.agent_type,
            'n_trials': len(self.study.trials),
            'timestamp': datetime.now().isoformat()
        }
        
        # Obtener estadísticas de todos los trials
        rewards = [-trial.value for trial in self.study.trials if trial.value is not None]
        
        if rewards:
            results['mean_reward'] = float(np.mean(rewards))
            results['std_reward'] = float(np.std(rewards))
            results['min_reward'] = float(np.min(rewards))
            results['max_reward'] = float(np.max(rewards))
        
        return results
    
    def plot_optimization_results(self, save_path: Optional[str] = None) -> str:
        """
        Generar gráfico de resultados de optimización.
        
        Args:
            save_path: Ruta donde guardar el gráfico
            
        Returns:
            Imagen en formato base64
        """
        if self.study is None:
            return ""
        
        # Crear figura con múltiples subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        # Historia de optimización
        trial_numbers = [t.number for t in self.study.trials if t.value is not None]
        rewards = [-t.value for t in self.study.trials if t.value is not None]
        
        axs[0].plot(trial_numbers, rewards, 'o-')
        axs[0].set_xlabel('Número de Trial')
        axs[0].set_ylabel('Recompensa')
        axs[0].set_title('Historia de Optimización')
        axs[0].axhline(y=-self.study.best_value, color='r', linestyle='--', label=f'Mejor ({-self.study.best_value:.2f})')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Importancia de parámetros
        param_importances = optuna.importance.get_param_importances(self.study)
        
        # Ordenar por importancia
        sorted_importances = dict(sorted(param_importances.items(), key=lambda x: x[1], reverse=True))
        
        if sorted_importances:
            # Graficar importancias
            axs[1].bar(list(sorted_importances.keys()), list(sorted_importances.values()))
            axs[1].set_xlabel('Parámetro')
            axs[1].set_ylabel('Importancia')
            axs[1].set_title('Importancia de Parámetros')
            plt.setp(axs[1].get_xticklabels(), rotation=45, ha='right')
            axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Guardar gráfico si se solicita
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(base64.b64decode(img_str))
            
            self.logger.info(f"Gráfico de optimización guardado en {save_path}")
        
        return img_str
    
    def save_study(self, filepath: str) -> str:
        """
        Guardar estudio de optimización en disco.
        
        Args:
            filepath: Ruta donde guardar el estudio
            
        Returns:
            Ruta donde se guardó el estudio
        """
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")
        
        # Guardar estudio
        with open(filepath, 'wb') as f:
            pickle.dump(self.study, f)
        
        self.logger.info(f"Estudio guardado en {filepath}")
        
        return filepath
    
    def load_study(self, filepath: str) -> bool:
        """
        Cargar estudio de optimización desde disco.
        
        Args:
            filepath: Ruta desde donde cargar el estudio
            
        Returns:
            True si se cargó correctamente
        """
        try:
            # Cargar estudio
            with open(filepath, 'rb') as f:
                self.study = pickle.load(f)
            
            self.logger.info(f"Estudio cargado desde {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando estudio: {str(e)}")
            return False