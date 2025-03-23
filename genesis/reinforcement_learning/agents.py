"""
Agentes de Aprendizaje por Refuerzo para trading.

Este módulo implementa agentes RL para la toma de decisiones de trading,
incluyendo agentes DQN, PPO y SAC optimizados para mercados financieros.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import gymnasium as gym
from gymnasium import spaces
import asyncio
from concurrent.futures import ThreadPoolExecutor
import random
from datetime import datetime
from pathlib import Path

from genesis.reinforcement_learning.environments import TradingEnvironment, MultiAssetTradingEnvironment

# Verificar disponibilidad de estable-baselines3
try:
    import stable_baselines3 as sb3
    from stable_baselines3 import DQN, PPO, SAC, A2C
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.logger import configure
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

class TrainingCallback(BaseCallback):
    """
    Callback personalizado para monitorear el entrenamiento de agentes RL.
    
    Este callback registra métricas durante el entrenamiento y puede
    guardar checkpoints periódicos.
    """
    
    def __init__(self, 
                 eval_env: Optional[gym.Env] = None,
                 log_dir: str = './logs/rl_training',
                 eval_freq: int = 10000,
                 n_eval_episodes: int = 5,
                 save_path: Optional[str] = None,
                 save_freq: int = 10000,
                 verbose: int = 1):
        """
        Inicializar callback de entrenamiento.
        
        Args:
            eval_env: Entorno para evaluación periódica
            log_dir: Directorio para logs
            eval_freq: Frecuencia de evaluación (pasos)
            n_eval_episodes: Número de episodios para evaluación
            save_path: Ruta donde guardar modelos (None = no guardar)
            save_freq: Frecuencia de guardado (pasos)
            verbose: Nivel de verbosidad
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.save_freq = save_freq
        
        # Crear directorio de logs si no existe
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Crear directorio de guardado si no existe
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Métricas para monitoreo
        self.ep_rewards = []
        self.ep_lengths = []
        self.last_mean_reward = -float('inf')
        self.eval_results = []
        
        self.logger = logging.getLogger(__name__)
    
    def _on_training_start(self) -> None:
        """Configuración al iniciar el entrenamiento."""
        # Registrar hiperparámetros
        self.logger.info("Iniciando entrenamiento de RL")
        self.logger.info(f"Modelo: {self.model.__class__.__name__}")
        self.logger.info(f"Pasos totales: {self.model._total_timesteps}")
        
        # Guardar hiperparámetros en archivo si hay log_dir
        if self.log_dir:
            hyperparams = {
                'model_class': self.model.__class__.__name__,
                'total_timesteps': self.model._total_timesteps,
                'learning_rate': self.model.learning_rate,
                'policy_class': str(self.model.policy.__class__.__name__),
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                # Intentar extraer más hiperparámetros según el tipo de modelo
                if isinstance(self.model, DQN):
                    hyperparams.update({
                        'buffer_size': self.model.buffer_size,
                        'learning_starts': self.model.learning_starts,
                        'batch_size': self.model.batch_size,
                        'target_update_interval': self.model.target_update_interval,
                        'exploration_fraction': self.model.exploration_fraction,
                        'exploration_initial_eps': self.model.exploration_initial_eps,
                        'exploration_final_eps': self.model.exploration_final_eps
                    })
                elif isinstance(self.model, PPO):
                    hyperparams.update({
                        'n_steps': self.model.n_steps,
                        'batch_size': self.model.batch_size,
                        'n_epochs': self.model.n_epochs,
                        'gae_lambda': self.model.gae_lambda,
                        'clip_range': self.model.clip_range,
                        'normalize_advantage': self.model.normalize_advantage
                    })
                elif isinstance(self.model, SAC):
                    hyperparams.update({
                        'buffer_size': self.model.buffer_size,
                        'learning_starts': self.model.learning_starts,
                        'batch_size': self.model.batch_size,
                        'tau': self.model.tau,
                        'gamma': self.model.gamma,
                        'ent_coef': self.model.ent_coef
                    })
            except Exception as e:
                self.logger.warning(f"Error extrayendo hiperparámetros: {str(e)}")
            
            # Guardar hiperparámetros
            hyperparams_path = os.path.join(self.log_dir, "hyperparams.json")
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparams, f, indent=2)
    
    def _on_step(self) -> bool:
        """
        Ejecutado en cada paso de entrenamiento.
        
        Returns:
            True para continuar entrenamiento, False para detener
        """
        # Verificar si es tiempo de guardar el modelo
        if self.save_path and self.n_calls % self.save_freq == 0:
            save_path = f"{self.save_path}_step{self.n_calls}"
            self.model.save(save_path)
            self.logger.info(f"Modelo guardado en {save_path}")
        
        # Verificar si es tiempo de evaluar el modelo
        if self.eval_env is not None and self.n_calls % self.eval_freq == 0:
            # Evaluar modelo actual
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.n_eval_episodes, 
                deterministic=True
            )
            
            # Registrar métrica en el logger de stable-baselines
            self.logger.info(f"Evaluación en paso {self.n_calls}: Recompensa media = {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Guardar resultados
            self.eval_results.append({
                'step': self.n_calls,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward)
            })
            
            # Guardar resultados en archivo
            if self.log_dir:
                eval_path = os.path.join(self.log_dir, "eval_results.json")
                with open(eval_path, 'w') as f:
                    json.dump(self.eval_results, f, indent=2)
            
            # Guardar mejor modelo
            if self.save_path and mean_reward > self.last_mean_reward:
                best_path = f"{self.save_path}_best"
                self.model.save(best_path)
                self.logger.info(f"Mejor modelo actualizado (recompensa: {mean_reward:.2f}) y guardado en {best_path}")
                self.last_mean_reward = mean_reward
        
        return True
    
    def _on_training_end(self) -> None:
        """Ejecutado al finalizar el entrenamiento."""
        # Guardar modelo final
        if self.save_path:
            final_path = f"{self.save_path}_final"
            self.model.save(final_path)
            self.logger.info(f"Modelo final guardado en {final_path}")
        
        # Guardar resultados finales
        if self.log_dir:
            # Guardar resultados de evaluación si hay
            if self.eval_results:
                eval_path = os.path.join(self.log_dir, "eval_results_final.json")
                with open(eval_path, 'w') as f:
                    json.dump(self.eval_results, f, indent=2)
            
            # Guardar recompensas por episodio si hay
            if self.ep_rewards:
                rewards_path = os.path.join(self.log_dir, "episode_rewards.json")
                with open(rewards_path, 'w') as f:
                    json.dump(self.ep_rewards, f, indent=2)
        
        self.logger.info("Entrenamiento finalizado")
    
    def plot_training_progress(self) -> str:
        """
        Generar gráfico de progreso del entrenamiento.
        
        Returns:
            Imagen en formato base64
        """
        if not self.eval_results:
            return ""
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Datos
        steps = [r['step'] for r in self.eval_results]
        rewards = [r['mean_reward'] for r in self.eval_results]
        stds = [r['std_reward'] for r in self.eval_results]
        
        # Gráfico de recompensas
        ax.plot(steps, rewards, 'o-', label='Mean Reward')
        ax.fill_between(
            steps, 
            [r - s for r, s in zip(rewards, stds)], 
            [r + s for r, s in zip(rewards, stds)], 
            alpha=0.3
        )
        
        # Formato
        ax.set_title('Training Progress')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        fig.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return img_str


class RLAgent:
    """
    Clase base para agentes de Reinforcement Learning.
    
    Define la interfaz común para los diferentes algoritmos RL
    que pueden ser aplicados en trading.
    """
    
    def __init__(self, 
                 env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
                 model_type: str = 'ppo',
                 policy_type: str = 'MlpPolicy',
                 model_params: Optional[Dict[str, Any]] = None,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 1,
                 seed: int = 42,
                 device: str = 'auto'):
        """
        Inicializar agente RL.
        
        Args:
            env: Entorno de trading
            model_type: Tipo de modelo ('dqn', 'ppo', 'sac', 'a2c')
            policy_type: Tipo de política ('MlpPolicy', 'CnnPolicy', etc.)
            model_params: Parámetros específicos del modelo
            tensorboard_log: Directorio para logs de TensorBoard
            verbose: Nivel de verbosidad
            seed: Semilla para reproducibilidad
            device: Dispositivo ('auto', 'cpu', 'cuda')
        """
        # Verificar disponibilidad de stable-baselines3
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 no está disponible. Instala con: pip install stable-baselines3")
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Guardar parámetros
        self.env = env
        self.model_type = model_type.lower()
        self.policy_type = policy_type
        self.seed = seed
        self.device = device
        
        # Configurar parámetros del modelo por defecto
        self.model_params = {
            'verbose': verbose,
            'seed': seed,
            'device': device
        }
        
        # Añadir TensorBoard log si está especificado
        if tensorboard_log:
            self.model_params['tensorboard_log'] = tensorboard_log
            # Crear directorio si no existe
            os.makedirs(tensorboard_log, exist_ok=True)
        
        # Actualizar con parámetros específicos
        if model_params:
            self.model_params.update(model_params)
        
        # Modelo interno
        self.model = None
        
        # Dimensionalidad
        if isinstance(env.observation_space, spaces.Dict):
            self.is_dict_obs = True
        else:
            self.is_dict_obs = False
        
        # Vectorizar entorno si es necesario
        self.vec_env = None
        
        self.logger.info(f"Agente RL inicializado con modelo {model_type}")
    
    def _create_model(self) -> Any:
        """
        Crear instancia del modelo específico.
        
        Returns:
            Modelo de RL
        """
        # Determinar la clase de modelo según el tipo
        if self.model_type == 'dqn':
            model_class = DQN
        elif self.model_type == 'ppo':
            model_class = PPO
        elif self.model_type == 'sac':
            model_class = SAC
        elif self.model_type == 'a2c':
            model_class = A2C
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}. Use 'dqn', 'ppo', 'sac' o 'a2c'.")
        
        # Crear modelo
        model = model_class(
            policy=self.policy_type,
            env=self.env if self.vec_env is None else self.vec_env,
            **self.model_params
        )
        
        return model
    
    def train(self, 
              total_timesteps: int = 100000,
              callback: Optional[BaseCallback] = None,
              log_dir: Optional[str] = None,
              eval_freq: int = 10000,
              n_eval_episodes: int = 5,
              save_path: Optional[str] = None,
              save_freq: int = 10000) -> Dict[str, Any]:
        """
        Entrenar el agente de RL.
        
        Args:
            total_timesteps: Número total de pasos para entrenamiento
            callback: Callback personalizado (opcional)
            log_dir: Directorio para logs
            eval_freq: Frecuencia de evaluación (pasos)
            n_eval_episodes: Número de episodios para evaluación
            save_path: Ruta donde guardar modelos (None = no guardar)
            save_freq: Frecuencia de guardado (pasos)
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        # Crear directorio de logs si está especificado
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            # Configurar logger
            configure(log_dir, ["stdout", "csv", "tensorboard"])
        
        # Preparar entorno de evaluación si se requiere
        eval_env = None
        if eval_freq > 0 and n_eval_episodes > 0:
            # Crear copia del entorno para evaluación
            if isinstance(self.env, gym.Env):
                # Para entornos no vectorizados
                eval_env = self.env
            elif self.vec_env is not None:
                # Para entornos vectorizados
                eval_env = self.vec_env
        
        # Crear callback si no se proporciona
        if callback is None and (eval_freq > 0 or save_freq > 0):
            callback = TrainingCallback(
                eval_env=eval_env,
                log_dir=log_dir,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                save_path=save_path,
                save_freq=save_freq
            )
        
        # Crear modelo si no existe
        if self.model is None:
            self.model = self._create_model()
        
        # Entrenar modelo
        start_time = time.time()
        self.logger.info(f"Iniciando entrenamiento por {total_timesteps} pasos")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=f"{self.model_type}_{int(time.time())}"
        )
        
        training_time = time.time() - start_time
        self.logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        # Evaluar modelo final si hay entorno de evaluación
        final_eval_result = None
        if eval_env is not None:
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                eval_env, 
                n_eval_episodes=n_eval_episodes, 
                deterministic=True
            )
            
            final_eval_result = {
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward)
            }
            
            self.logger.info(f"Evaluación final: Recompensa media = {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Guardar modelo final si se especifica ruta
        if save_path:
            final_path = f"{save_path}_final"
            self.model.save(final_path)
            self.logger.info(f"Modelo final guardado en {final_path}")
        
        # Resultados del entrenamiento
        results = {
            'model_type': self.model_type,
            'policy_type': self.policy_type,
            'training_time': training_time,
            'total_timesteps': total_timesteps,
            'final_evaluation': final_eval_result,
            'training_success': True
        }
        
        return results
    
    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[Any, Any]:
        """
        Realizar predicción con el modelo entrenado.
        
        Args:
            observation: Observación del entorno
            deterministic: Si es True, usa política determinista
            
        Returns:
            Tupla de (acción, estado)
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Predicción
        action, state = self.model.predict(observation, deterministic=deterministic)
        
        return action, state
    
    def save_model(self, filepath: str) -> str:
        """
        Guardar modelo entrenado en disco.
        
        Args:
            filepath: Ruta donde guardar el modelo
            
        Returns:
            Ruta donde se guardó el modelo
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Guardar modelo
        self.model.save(filepath)
        self.logger.info(f"Modelo guardado en {filepath}")
        
        # Guardar metadatos
        metadata_path = f"{filepath}_metadata.json"
        metadata = {
            'model_type': self.model_type,
            'policy_type': self.policy_type,
            'model_params': self.model_params,
            'timestamp': datetime.now().isoformat(),
            'framework': 'stable-baselines3'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
    
    def load_model(self, filepath: str) -> bool:
        """
        Cargar modelo entrenado desde disco.
        
        Args:
            filepath: Ruta al archivo del modelo
            
        Returns:
            True si se cargó correctamente
        """
        try:
            # Determinar la clase de modelo según los metadatos o el tipo
            metadata_path = f"{filepath}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                model_type = metadata.get('model_type', self.model_type)
            else:
                model_type = self.model_type
            
            # Cargar modelo según el tipo
            if model_type == 'dqn':
                self.model = DQN.load(filepath, env=self.env if self.vec_env is None else self.vec_env)
            elif model_type == 'ppo':
                self.model = PPO.load(filepath, env=self.env if self.vec_env is None else self.vec_env)
            elif model_type == 'sac':
                self.model = SAC.load(filepath, env=self.env if self.vec_env is None else self.vec_env)
            elif model_type == 'a2c':
                self.model = A2C.load(filepath, env=self.env if self.vec_env is None else self.vec_env)
            else:
                raise ValueError(f"Tipo de modelo no soportado: {model_type}")
            
            self.logger.info(f"Modelo cargado correctamente desde {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo: {str(e)}")
            return False


class DQNAgent(RLAgent):
    """
    Agente DQN (Deep Q-Network) para trading.
    
    DQN es un algoritmo de Q-learning basado en redes neuronales
    adecuado para espacios de acción discretos.
    """
    
    def __init__(self, 
                 env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
                 policy_type: str = 'MlpPolicy',
                 learning_rate: float = 0.0001,
                 buffer_size: int = 100000,
                 learning_starts: int = 1000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 target_update_interval: int = 100,
                 exploration_fraction: float = 0.1,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.05,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 1,
                 seed: int = 42,
                 device: str = 'auto'):
        """
        Inicializar agente DQN.
        
        Args:
            env: Entorno de trading
            policy_type: Tipo de política ('MlpPolicy', 'CnnPolicy', etc.)
            learning_rate: Tasa de aprendizaje
            buffer_size: Tamaño del buffer de experiencia
            learning_starts: Número de pasos antes de empezar a entrenar
            batch_size: Tamaño del lote para entrenamiento
            gamma: Factor de descuento
            target_update_interval: Intervalo para actualizar la red objetivo
            exploration_fraction: Fracción de entrenamiento para exploración
            exploration_initial_eps: Epsilon inicial para exploración
            exploration_final_eps: Epsilon final para exploración
            tensorboard_log: Directorio para logs de TensorBoard
            verbose: Nivel de verbosidad
            seed: Semilla para reproducibilidad
            device: Dispositivo ('auto', 'cpu', 'cuda')
        """
        # Parámetros específicos de DQN
        model_params = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'learning_starts': learning_starts,
            'batch_size': batch_size,
            'gamma': gamma,
            'target_update_interval': target_update_interval,
            'exploration_fraction': exploration_fraction,
            'exploration_initial_eps': exploration_initial_eps,
            'exploration_final_eps': exploration_final_eps
        }
        
        super().__init__(
            env=env,
            model_type='dqn',
            policy_type=policy_type,
            model_params=model_params,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device
        )


class PPOAgent(RLAgent):
    """
    Agente PPO (Proximal Policy Optimization) para trading.
    
    PPO es un algoritmo de optimización de política que equilibra
    exploración y explotación, adecuado para problemas de alta dimensionalidad.
    """
    
    def __init__(self, 
                 env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
                 policy_type: str = 'MlpPolicy',
                 learning_rate: float = 0.0003,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 clip_range_vf: Optional[float] = None,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 1,
                 seed: int = 42,
                 device: str = 'auto'):
        """
        Inicializar agente PPO.
        
        Args:
            env: Entorno de trading
            policy_type: Tipo de política ('MlpPolicy', 'CnnPolicy', etc.)
            learning_rate: Tasa de aprendizaje
            n_steps: Número de pasos por actualización
            batch_size: Tamaño del lote para entrenamiento
            n_epochs: Número de épocas
            gamma: Factor de descuento
            gae_lambda: Factor para Generalized Advantage Estimation
            clip_range: Rango de recorte para PPO
            clip_range_vf: Rango de recorte para función de valor
            normalize_advantage: Si es True, normaliza ventajas
            ent_coef: Coeficiente de entropía
            vf_coef: Coeficiente de función de valor
            max_grad_norm: Norma máxima de gradiente
            use_sde: Si es True, usa State Dependent Exploration
            sde_sample_freq: Frecuencia de muestreo para SDE
            tensorboard_log: Directorio para logs de TensorBoard
            verbose: Nivel de verbosidad
            seed: Semilla para reproducibilidad
            device: Dispositivo ('auto', 'cpu', 'cuda')
        """
        # Parámetros específicos de PPO
        model_params = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'clip_range_vf': clip_range_vf,
            'normalize_advantage': normalize_advantage,
            'ent_coef': ent_coef,
            'vf_coef': vf_coef,
            'max_grad_norm': max_grad_norm,
            'use_sde': use_sde,
            'sde_sample_freq': sde_sample_freq
        }
        
        super().__init__(
            env=env,
            model_type='ppo',
            policy_type=policy_type,
            model_params=model_params,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device
        )


class SACAgent(RLAgent):
    """
    Agente SAC (Soft Actor-Critic) para trading.
    
    SAC es un algoritmo de actor-crítico que maximiza tanto el rendimiento
    como la entropía, equilibrando exploración y explotación.
    """
    
    def __init__(self, 
                 env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
                 policy_type: str = 'MlpPolicy',
                 learning_rate: float = 0.0003,
                 buffer_size: int = 100000,
                 learning_starts: int = 1000,
                 batch_size: int = 256,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 train_freq: int = 1,
                 gradient_steps: int = 1,
                 action_noise: Optional[Any] = None,
                 optimize_memory_usage: bool = False,
                 ent_coef: Union[str, float] = 'auto',
                 target_update_interval: int = 1,
                 target_entropy: Union[str, float] = 'auto',
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 use_sde_at_warmup: bool = False,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 1,
                 seed: int = 42,
                 device: str = 'auto'):
        """
        Inicializar agente SAC.
        
        Args:
            env: Entorno de trading
            policy_type: Tipo de política ('MlpPolicy', 'CnnPolicy', etc.)
            learning_rate: Tasa de aprendizaje
            buffer_size: Tamaño del buffer de experiencia
            learning_starts: Número de pasos antes de empezar a entrenar
            batch_size: Tamaño del lote para entrenamiento
            gamma: Factor de descuento
            tau: Tasa de actualización para promedio polinómico
            train_freq: Frecuencia de entrenamiento
            gradient_steps: Número de pasos de gradiente por actualización
            action_noise: Ruido para la acción
            optimize_memory_usage: Si es True, optimiza uso de memoria
            ent_coef: Coeficiente de entropía ('auto' o valor)
            target_update_interval: Intervalo para actualizar red objetivo
            target_entropy: Entropía objetivo ('auto' o valor)
            use_sde: Si es True, usa State Dependent Exploration
            sde_sample_freq: Frecuencia de muestreo para SDE
            use_sde_at_warmup: Si es True, usa SDE durante warmup
            tensorboard_log: Directorio para logs de TensorBoard
            verbose: Nivel de verbosidad
            seed: Semilla para reproducibilidad
            device: Dispositivo ('auto', 'cpu', 'cuda')
        """
        # Parámetros específicos de SAC
        model_params = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'learning_starts': learning_starts,
            'batch_size': batch_size,
            'gamma': gamma,
            'tau': tau,
            'train_freq': train_freq,
            'gradient_steps': gradient_steps,
            'action_noise': action_noise,
            'optimize_memory_usage': optimize_memory_usage,
            'ent_coef': ent_coef,
            'target_update_interval': target_update_interval,
            'target_entropy': target_entropy,
            'use_sde': use_sde,
            'sde_sample_freq': sde_sample_freq,
            'use_sde_at_warmup': use_sde_at_warmup
        }
        
        super().__init__(
            env=env,
            model_type='sac',
            policy_type=policy_type,
            model_params=model_params,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device
        )


class RLAgentManager:
    """
    Gestor de agentes de Reinforcement Learning para el Sistema Genesis.
    
    Proporciona una interfaz unificada para crear, entrenar y usar
    diferentes tipos de agentes RL en el contexto de trading.
    """
    
    def __init__(self, 
                 db: Optional[Any] = None,
                 cache_dir: str = './cache/rl_models',
                 logs_dir: str = './logs/rl',
                 num_cores: int = 4):
        """
        Inicializar gestor de agentes RL.
        
        Args:
            db: Conexión a base de datos (opcional)
            cache_dir: Directorio para caché de modelos
            logs_dir: Directorio para logs
            num_cores: Número de núcleos para procesamiento paralelo
        """
        self.logger = logging.getLogger(__name__)
        self.db = db
        self.cache_dir = cache_dir
        self.logs_dir = logs_dir
        self.num_cores = num_cores
        self.executor = ThreadPoolExecutor(max_workers=num_cores)
        
        # Crear directorios si no existen
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Registro de agentes/modelos
        self.agents = {}
        
        self.logger.info(f"Gestor de agentes RL inicializado")
    
    def _get_agent_class(self, agent_type: str) -> Type[RLAgent]:
        """
        Obtener clase de agente según el tipo.
        
        Args:
            agent_type: Tipo de agente ('dqn', 'ppo', 'sac')
            
        Returns:
            Clase del agente
        """
        agent_type = agent_type.lower()
        
        if agent_type == 'dqn':
            return DQNAgent
        elif agent_type == 'ppo':
            return PPOAgent
        elif agent_type == 'sac':
            return SACAgent
        else:
            raise ValueError(f"Tipo de agente no soportado: {agent_type}. Use 'dqn', 'ppo' o 'sac'")
    
    async def create_agent(self, 
                     symbol: str,
                     env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
                     agent_type: str = 'ppo',
                     policy_type: str = 'MlpPolicy',
                     agent_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Crear un agente RL para un símbolo/entorno específico.
        
        Args:
            symbol: Símbolo/identificador del modelo
            env: Entorno de trading
            agent_type: Tipo de agente ('dqn', 'ppo', 'sac')
            policy_type: Tipo de política ('MlpPolicy', 'CnnPolicy', etc.)
            agent_params: Parámetros específicos del agente
            
        Returns:
            ID del agente creado
        """
        self.logger.info(f"Creando agente {agent_type} para {symbol}")
        
        # Determinar clase de agente
        agent_class = self._get_agent_class(agent_type)
        
        # Parámetros específicos del agente
        params = agent_params or {}
        
        # Configurar TensorBoard log
        tb_log_dir = os.path.join(self.logs_dir, 'tensorboard', f"{symbol}_{agent_type}_{int(time.time())}")
        params['tensorboard_log'] = tb_log_dir
        
        # Crear agente
        agent = agent_class(
            env=env,
            policy_type=policy_type,
            **params
        )
        
        # Registrar agente
        agent_id = f"{symbol}_{agent_type}_{int(time.time())}"
        self.agents[agent_id] = agent
        
        self.logger.info(f"Agente creado con ID: {agent_id}")
        
        return agent_id
    
    async def train_agent(self, 
                    agent_id: str,
                    total_timesteps: int = 100000,
                    eval_freq: int = 10000,
                    n_eval_episodes: int = 5,
                    save_freq: int = 10000,
                    save_model: bool = True) -> Dict[str, Any]:
        """
        Entrenar un agente RL.
        
        Args:
            agent_id: ID del agente a entrenar
            total_timesteps: Número total de pasos para entrenamiento
            eval_freq: Frecuencia de evaluación (pasos)
            n_eval_episodes: Número de episodios para evaluación
            save_freq: Frecuencia de guardado (pasos)
            save_model: Si es True, guarda el modelo entrenado
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        # Verificar si el agente existe
        if agent_id not in self.agents:
            raise ValueError(f"Agente no encontrado: {agent_id}")
        
        agent = self.agents[agent_id]
        
        # Configurar directorios para logs y modelos
        log_dir = os.path.join(self.logs_dir, agent_id)
        save_path = os.path.join(self.cache_dir, agent_id) if save_model else None
        
        # Crear directorios si no existen
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Entrenar agente en un hilo aparte para no bloquear
        loop = asyncio.get_event_loop()
        
        try:
            # Entrenamiento asíncrono
            training_results = await loop.run_in_executor(
                self.executor,
                lambda: agent.train(
                    total_timesteps=total_timesteps,
                    log_dir=log_dir,
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    save_path=save_path,
                    save_freq=save_freq
                )
            )
            
            # Agregar información adicional
            training_results['agent_id'] = agent_id
            
            # Guardar en base de datos si está disponible
            if self.db:
                try:
                    await self.db.store('rl_training_results', {
                        'agent_id': agent_id,
                        'training_time': training_results['training_time'],
                        'total_timesteps': training_results['total_timesteps'],
                        'final_evaluation': training_results.get('final_evaluation'),
                        'timestamp': datetime.now().isoformat()
                    })
                    self.logger.info(f"Resultados de entrenamiento guardados en base de datos: {agent_id}")
                except Exception as e:
                    self.logger.error(f"Error guardando resultados en base de datos: {str(e)}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error entrenando agente: {str(e)}")
            raise
    
    async def predict(self, 
                agent_id: str,
                observation: Any,
                deterministic: bool = True) -> Any:
        """
        Realizar predicción con un agente entrenado.
        
        Args:
            agent_id: ID del agente
            observation: Observación del entorno
            deterministic: Si es True, usa política determinista
            
        Returns:
            Acción predicha
        """
        # Verificar si el agente existe
        if agent_id not in self.agents:
            raise ValueError(f"Agente no encontrado: {agent_id}")
        
        agent = self.agents[agent_id]
        
        # Ejecutar en un hilo aparte para no bloquear
        loop = asyncio.get_event_loop()
        
        try:
            # Predicción asíncrona
            action, _ = await loop.run_in_executor(
                self.executor,
                lambda: agent.predict(observation, deterministic)
            )
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error realizando predicción: {str(e)}")
            raise
    
    async def evaluate_agent(self, 
                       agent_id: str,
                       n_eval_episodes: int = 10,
                       deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluar un agente entrenado.
        
        Args:
            agent_id: ID del agente
            n_eval_episodes: Número de episodios para evaluación
            deterministic: Si es True, usa política determinista
            
        Returns:
            Diccionario con resultados de la evaluación
        """
        # Verificar si el agente existe
        if agent_id not in self.agents:
            raise ValueError(f"Agente no encontrado: {agent_id}")
        
        agent = self.agents[agent_id]
        
        if agent.model is None:
            raise ValueError("El agente no ha sido entrenado")
        
        # Ejecutar en un hilo aparte para no bloquear
        loop = asyncio.get_event_loop()
        
        try:
            # Evaluación asíncrona
            mean_reward, std_reward = await loop.run_in_executor(
                self.executor,
                lambda: evaluate_policy(
                    agent.model, 
                    agent.env if agent.vec_env is None else agent.vec_env, 
                    n_eval_episodes=n_eval_episodes, 
                    deterministic=deterministic
                )
            )
            
            # Resultados
            results = {
                'agent_id': agent_id,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'n_eval_episodes': n_eval_episodes,
                'deterministic': deterministic,
                'timestamp': datetime.now().isoformat()
            }
            
            # Guardar en base de datos si está disponible
            if self.db:
                try:
                    await self.db.store('rl_evaluation_results', results)
                    self.logger.info(f"Resultados de evaluación guardados en base de datos: {agent_id}")
                except Exception as e:
                    self.logger.error(f"Error guardando resultados en base de datos: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluando agente: {str(e)}")
            raise
    
    async def save_agent(self, agent_id: str, filepath: Optional[str] = None) -> str:
        """
        Guardar un agente entrenado en disco.
        
        Args:
            agent_id: ID del agente
            filepath: Ruta donde guardar el agente (opcional)
            
        Returns:
            Ruta donde se guardó el agente
        """
        # Verificar si el agente existe
        if agent_id not in self.agents:
            raise ValueError(f"Agente no encontrado: {agent_id}")
        
        agent = self.agents[agent_id]
        
        # Generar ruta si no se proporciona
        if filepath is None:
            filepath = os.path.join(self.cache_dir, f"{agent_id}_model")
        
        # Guardar agente en un hilo aparte
        loop = asyncio.get_event_loop()
        
        try:
            # Guardado asíncrono
            filepath = await loop.run_in_executor(
                self.executor,
                lambda: agent.save_model(filepath)
            )
            
            self.logger.info(f"Agente guardado en: {filepath}")
            
            # Guardar en base de datos si está disponible
            if self.db:
                try:
                    await self.db.store('rl_models', {
                        'agent_id': agent_id,
                        'model_path': filepath,
                        'model_type': agent.model_type,
                        'policy_type': agent.policy_type,
                        'timestamp': datetime.now().isoformat()
                    })
                    self.logger.info(f"Registro del modelo guardado en base de datos: {agent_id}")
                except Exception as e:
                    self.logger.error(f"Error guardando registro en base de datos: {str(e)}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error guardando agente: {str(e)}")
            raise
    
    async def load_agent(self, 
                   agent_id: str,
                   filepath: str,
                   env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
                   agent_type: str = 'ppo') -> bool:
        """
        Cargar un agente desde disco.
        
        Args:
            agent_id: ID para el agente cargado
            filepath: Ruta al archivo del modelo
            env: Entorno de trading
            agent_type: Tipo de agente ('dqn', 'ppo', 'sac')
            
        Returns:
            True si se cargó correctamente
        """
        try:
            # Determinar clase de agente
            agent_class = self._get_agent_class(agent_type)
            
            # Crear agente temporal
            agent = agent_class(env=env)
            
            # Cargar modelo en un thread aparte
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor,
                lambda: agent.load_model(filepath)
            )
            
            if not success:
                raise ValueError(f"Error al cargar modelo desde: {filepath}")
            
            # Registrar agente
            self.agents[agent_id] = agent
            
            self.logger.info(f"Agente cargado correctamente con ID: {agent_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando agente: {str(e)}")
            return False
    
    def get_registered_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener lista de agentes registrados.
        
        Returns:
            Diccionario con información de los agentes
        """
        result = {}
        
        for agent_id, agent in self.agents.items():
            result[agent_id] = {
                'model_type': agent.model_type,
                'policy_type': agent.policy_type,
                'is_trained': agent.model is not None
            }
        
        return result
    
    async def get_stored_models(self) -> List[Dict[str, Any]]:
        """
        Obtener lista de modelos almacenados en base de datos.
        
        Returns:
            Lista de diccionarios con información de los modelos
        """
        if not self.db:
            return []
        
        try:
            # Consultar base de datos
            records = await self.db.retrieve('rl_models', None)
            
            if not records:
                return []
            
            # Formatear resultados
            result = []
            for record in records:
                result.append({
                    'agent_id': record.get('agent_id'),
                    'model_path': record.get('model_path'),
                    'model_type': record.get('model_type'),
                    'policy_type': record.get('policy_type'),
                    'timestamp': record.get('timestamp')
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error obteniendo modelos almacenados: {str(e)}")
            return []