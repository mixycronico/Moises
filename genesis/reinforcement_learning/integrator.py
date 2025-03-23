"""
Integrador de Reinforcement Learning (RL) para el Sistema Genesis.

Este módulo proporciona la integración entre los componentes de RL (agentes,
entornos, evaluación) y el resto del sistema Genesis, permitiendo utilizar
modelos de RL para tomar decisiones de trading.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class RLIntegrator:
    """
    Integrador para conectar Reinforcement Learning con el Sistema Genesis.
    
    Proporciona interfaces para:
    - Inicializar agentes y entornos
    - Procesar datos de mercado en tiempo real
    - Generar señales de trading basadas en modelos de RL
    - Realizar entrenamiento continuo y optimización
    - Integrar con otros componentes (sentiment, onchain, etc.)
    """
    
    def __init__(self, 
                 models_path: str = './models/rl',
                 use_ensemble: bool = True,
                 max_parallel_inference: int = 4):
        """
        Inicializar integrador de RL.
        
        Args:
            models_path: Ruta para guardar/cargar modelos
            use_ensemble: Si es True, usa un conjunto de agentes
            max_parallel_inference: Máximo de inferencias paralelas
        """
        self.logger = logging.getLogger(__name__)
        self.models_path = models_path
        self.use_ensemble = use_ensemble
        self.max_parallel_inference = max_parallel_inference
        
        # Crear directorio si no existe
        os.makedirs(models_path, exist_ok=True)
        
        # Componentes de RL
        self.agents = {}
        self.environments = {}
        self.evaluators = {}
        
        # Estadísticas y métricas
        self.stats = {
            'inferences': 0,
            'successful_trades': 0,
            'unsuccessful_trades': 0,
            'total_reward': 0.0,
            'last_train_time': None,
            'model_versions': {}
        }
        
        # Configuración
        self.config = {
            'training_frequency': 24 * 3600,  # 24 horas en segundos
            'window_size': 60,  # Tamaño de la ventana temporal para features
            'portfolio_features': True,  # Incluir features de portfolio
            'use_sentiment': True,  # Incluir datos de sentimiento
            'use_onchain': True,  # Incluir datos on-chain
            'ensemble_weights': {
                'dqn': 0.3,
                'ppo': 0.4,
                'sac': 0.3
            }
        }
        
        # Thread pool para inferencia
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_inference)
        
        self.logger.info("RLIntegrator inicializado")
    
    async def load_agents(self, symbols: List[str], agent_types: List[str] = ['dqn', 'ppo', 'sac']) -> Dict[str, Any]:
        """
        Cargar agentes de RL para símbolos específicos.
        
        Args:
            symbols: Lista de símbolos (ej. 'BTC/USDT')
            agent_types: Tipos de agentes a cargar
            
        Returns:
            Diccionario con resultados
        """
        results = {}
        
        # Importaciones condicionadas aquí para evitar cargar módulos pesados innecesariamente
        try:
            # Importar solo cuando sea necesario
            from .agents import RLAgentFactory, DQNAgent, PPOAgent, SACAgent, AgentConfig
            from .environments import TradingEnvironment, EnvironmentConfig
        except ImportError as e:
            self.logger.error(f"Error importando módulos de RL: {str(e)}")
            return {"error": f"Error importando módulos de RL: {str(e)}"}
        
        for symbol in symbols:
            self.logger.info(f"Cargando agentes para {symbol}")
            
            # Normalizar símbolo para uso en rutas
            symbol_norm = symbol.replace('/', '_')
            
            # Configurar entorno
            env_config = {
                'symbol': symbol,
                'window_size': self.config['window_size'],
                'include_portfolio': self.config['portfolio_features'],
                'reward_function': 'sharpe',  # sharpe, sortino, pnl, etc.
                'feature_set': 'full'  # basic, technical, full
            }
            
            try:
                # Crear entorno compartido para los agentes
                env = TradingEnvironment(env_config)
                self.environments[symbol] = env
                
                # Cargar/crear agentes
                symbol_agents = {}
                
                for agent_type in agent_types:
                    # Configuración específica según tipo de agente
                    if agent_type == 'dqn':
                        agent_config = {
                            'learning_rate': 1e-4,
                            'batch_size': 64,
                            'memory_size': 10000,
                            'gamma': 0.99,
                            'epsilon_start': 1.0,
                            'epsilon_end': 0.05,
                            'epsilon_decay': 0.995
                        }
                    elif agent_type == 'ppo':
                        agent_config = {
                            'learning_rate': 3e-4,
                            'gamma': 0.99,
                            'gae_lambda': 0.95,
                            'clip_ratio': 0.2,
                            'value_coeff': 0.5,
                            'entropy_coeff': 0.01
                        }
                    elif agent_type == 'sac':
                        agent_config = {
                            'learning_rate': 3e-4,
                            'alpha': 0.2,
                            'gamma': 0.99,
                            'tau': 0.005,
                            'buffer_size': 10000,
                            'batch_size': 64
                        }
                    else:
                        self.logger.warning(f"Tipo de agente no soportado: {agent_type}")
                        continue
                    
                    # Ruta del modelo
                    model_path = os.path.join(self.models_path, f"{symbol_norm}_{agent_type}")
                    
                    # Crear configuración de agente
                    config = AgentConfig(
                        agent_type=agent_type,
                        model_path=model_path,
                        env=env,
                        **agent_config
                    )
                    
                    # Crear agente
                    agent = RLAgentFactory.create_agent(config)
                    
                    # Verificar si existe un modelo guardado
                    if os.path.exists(model_path) or os.path.exists(f"{model_path}.zip"):
                        # Cargar modelo existente
                        try:
                            agent.load(model_path)
                            self.logger.info(f"Modelo {agent_type} cargado para {symbol}")
                            self.stats['model_versions'][f"{symbol}_{agent_type}"] = agent.get_version()
                        except Exception as e:
                            self.logger.error(f"Error cargando modelo {agent_type} para {symbol}: {str(e)}")
                            # Crear nuevo modelo si no se puede cargar
                            agent.save(model_path)
                            self.stats['model_versions'][f"{symbol}_{agent_type}"] = "initial"
                    else:
                        # Crear y guardar nuevo modelo
                        agent.save(model_path)
                        self.logger.info(f"Nuevo modelo {agent_type} creado para {symbol}")
                        self.stats['model_versions'][f"{symbol}_{agent_type}"] = "initial"
                    
                    # Añadir a la colección
                    symbol_agents[agent_type] = agent
                
                # Guardar agentes
                self.agents[symbol] = symbol_agents
                
                # Añadir al resultado
                results[symbol] = {
                    'status': 'loaded',
                    'agent_types': list(symbol_agents.keys())
                }
                
            except Exception as e:
                self.logger.error(f"Error cargando agentes para {symbol}: {str(e)}")
                results[symbol] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    async def process_market_data(self, 
                           symbol: str, 
                           data: pd.DataFrame,
                           portfolio_state: Optional[Dict[str, Any]] = None,
                           sentiment_data: Optional[Dict[str, Any]] = None,
                           onchain_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesar datos de mercado para generar señales de trading con RL.
        
        Args:
            symbol: Símbolo (ej. 'BTC/USDT')
            data: DataFrame con datos OHLCV
            portfolio_state: Estado actual del portfolio (opcional)
            sentiment_data: Datos de sentimiento (opcional)
            onchain_data: Datos on-chain (opcional)
            
        Returns:
            Diccionario con señales y metadatos
        """
        start_time = time.time()
        self.logger.info(f"Procesando datos de {symbol} con RL ({len(data)} registros)")
        
        # Verificar si tenemos agentes para este símbolo
        if symbol not in self.agents:
            self.logger.warning(f"No hay agentes cargados para {symbol}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': 'No hay agentes cargados para este símbolo'
            }
        
        # Verificar si hay suficientes datos
        if len(data) < self.config['window_size']:
            self.logger.warning(f"Datos insuficientes para {symbol}: {len(data)} < {self.config['window_size']}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': f"Datos insuficientes: {len(data)} < {self.config['window_size']}"
            }
        
        try:
            # Preparar datos para el entorno
            env = self.environments[symbol]
            
            # Actualizar datos en el entorno
            env.update_data(data)
            
            # Añadir datos de portfolio si están disponibles
            if portfolio_state and self.config['portfolio_features']:
                env.update_portfolio_state(portfolio_state)
            
            # Añadir datos de sentimiento si están disponibles y configurados
            if sentiment_data and self.config['use_sentiment']:
                env.add_sentiment_features(sentiment_data)
            
            # Añadir datos on-chain si están disponibles y configurados
            if onchain_data and self.config['use_onchain']:
                env.add_onchain_features(onchain_data)
            
            # Obtener estado actual
            state = env.get_current_state()
            
            # Si usamos ensemble, obtener acción de cada agente y combinar
            if self.use_ensemble and len(self.agents[symbol]) > 1:
                actions = {}
                confidence = {}
                
                # Obtener acciones de cada agente
                for agent_type, agent in self.agents[symbol].items():
                    action_result = await self._get_agent_action(agent, state)
                    actions[agent_type] = action_result['action']
                    confidence[agent_type] = action_result['confidence']
                
                # Combinar acciones según pesos configurados
                final_action, action_probs = self._combine_actions(actions, confidence)
                
                signal = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'action': final_action,
                    'action_name': self._action_to_name(final_action),
                    'confidence': float(action_probs[final_action]),
                    'action_distribution': {k: float(v) for k, v in action_probs.items()},
                    'agent_actions': {a_type: self._action_to_name(a) for a_type, a in actions.items()},
                    'agent_confidence': {a_type: float(c) for a_type, c in confidence.items()},
                    'processing_time': time.time() - start_time,
                    'status': 'success'
                }
            
            # Si no usamos ensemble, usar el agente principal (por defecto PPO)
            else:
                # Determinar agente principal
                primary_agent_type = 'ppo' if 'ppo' in self.agents[symbol] else list(self.agents[symbol].keys())[0]
                agent = self.agents[symbol][primary_agent_type]
                
                # Obtener acción
                action_result = await self._get_agent_action(agent, state)
                
                signal = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'action': action_result['action'],
                    'action_name': self._action_to_name(action_result['action']),
                    'confidence': float(action_result['confidence']),
                    'agent_type': primary_agent_type,
                    'processing_time': time.time() - start_time,
                    'status': 'success'
                }
            
            # Incrementar contador de inferencias
            self.stats['inferences'] += 1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error procesando datos de {symbol} con RL: {str(e)}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_agent_action(self, agent, state) -> Dict[str, Any]:
        """
        Obtener acción de un agente de forma asíncrona.
        
        Args:
            agent: Instancia del agente
            state: Estado actual
            
        Returns:
            Diccionario con acción y confianza
        """
        # Ejecutar inferencia en un hilo separado para no bloquear
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self.executor,
            lambda: agent.predict(state)
        )
        
        # Desempaquetar resultado (acción, probabilidades)
        action, probs = result
        
        # Si las probabilidades están disponibles, usarlas como confianza
        confidence = max(probs) if probs is not None else 1.0
        
        return {
            'action': action,
            'confidence': confidence,
            'probabilities': probs
        }
    
    def _combine_actions(self, 
                       actions: Dict[str, int], 
                       confidence: Dict[str, float]) -> Tuple[int, np.ndarray]:
        """
        Combinar acciones de múltiples agentes usando pesos.
        
        Args:
            actions: Acciones por tipo de agente
            confidence: Confianza por tipo de agente
            
        Returns:
            Tupla (acción final, distribución de probabilidad)
        """
        # Inicializar array de votos ponderados
        action_votes = np.zeros(3)  # 0: HOLD, 1: BUY, 2: SELL
        
        # Para cada agente
        for agent_type, action in actions.items():
            # Obtener peso del agente
            weight = self.config['ensemble_weights'].get(agent_type, 1.0 / len(actions))
            
            # Ajustar peso por confianza
            adjusted_weight = weight * confidence[agent_type]
            
            # Añadir voto
            action_votes[action] += adjusted_weight
        
        # Normalizar votos
        if action_votes.sum() > 0:
            action_probs = action_votes / action_votes.sum()
        else:
            # Si no hay votos (poco probable), distribuir uniformemente
            action_probs = np.ones(3) / 3
        
        # Seleccionar acción con más votos
        final_action = np.argmax(action_probs)
        
        return final_action, action_probs
    
    def _action_to_name(self, action: int) -> str:
        """
        Convertir acción numérica a nombre.
        
        Args:
            action: Índice de acción (0, 1, 2)
            
        Returns:
            Nombre de la acción
        """
        actions = {
            0: "HOLD",
            1: "BUY",
            2: "SELL"
        }
        return actions.get(action, "UNKNOWN")
    
    async def train_agents(self, symbol: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Entrenar agentes de RL con datos históricos.
        
        Args:
            symbol: Símbolo (ej. 'BTC/USDT')
            historical_data: DataFrame con datos históricos
            
        Returns:
            Diccionario con resultados de entrenamiento
        """
        start_time = time.time()
        self.logger.info(f"Iniciando entrenamiento para {symbol} con {len(historical_data)} registros")
        
        # Verificar si tenemos agentes para este símbolo
        if symbol not in self.agents:
            self.logger.warning(f"No hay agentes cargados para {symbol}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': 'No hay agentes cargados para este símbolo'
            }
        
        # Verificar si hay suficientes datos
        if len(historical_data) < 1000:
            self.logger.warning(f"Datos históricos insuficientes para {symbol}: {len(historical_data)} < 1000")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': f"Datos históricos insuficientes: {len(historical_data)} < 1000"
            }
        
        # Normalizar símbolo para uso en rutas
        symbol_norm = symbol.replace('/', '_')
        
        try:
            # Preparar entorno para entrenamiento
            env = self.environments[symbol]
            
            # Actualizar datos históricos en el entorno
            env.update_data(historical_data, is_training=True)
            
            # Resultados por agente
            results = {}
            
            # Entrenar cada agente
            for agent_type, agent in self.agents[symbol].items():
                agent_start_time = time.time()
                
                # Determinar parámetros de entrenamiento según tipo
                if agent_type == 'dqn':
                    train_params = {
                        'total_timesteps': 50000,
                        'eval_freq': 5000
                    }
                elif agent_type == 'ppo':
                    train_params = {
                        'total_timesteps': 100000,
                        'eval_freq': 10000
                    }
                elif agent_type == 'sac':
                    train_params = {
                        'total_timesteps': 100000,
                        'eval_freq': 10000
                    }
                else:
                    train_params = {
                        'total_timesteps': 50000,
                        'eval_freq': 5000
                    }
                
                # Ejecutar entrenamiento en un hilo separado
                loop = asyncio.get_running_loop()
                train_result = await loop.run_in_executor(
                    self.executor,
                    lambda: agent.train(**train_params)
                )
                
                # Guardar modelo entrenado
                model_path = os.path.join(self.models_path, f"{symbol_norm}_{agent_type}")
                agent.save(model_path)
                
                # Actualizar versión del modelo
                self.stats['model_versions'][f"{symbol}_{agent_type}"] = agent.get_version()
                
                # Registrar tiempo de entrenamiento
                training_time = time.time() - agent_start_time
                
                # Añadir a resultados
                results[agent_type] = {
                    'status': 'success',
                    'reward_mean': float(train_result.get('mean_reward', 0.0)),
                    'reward_std': float(train_result.get('std_reward', 0.0)),
                    'success_rate': float(train_result.get('success_rate', 0.0)),
                    'total_timesteps': train_result.get('total_timesteps', 0),
                    'training_time': training_time
                }
                
                self.logger.info(f"Entrenamiento de {agent_type} para {symbol} completado en {training_time:.1f}s")
            
            # Actualizar estadísticas
            self.stats['last_train_time'] = datetime.now().isoformat()
            
            return {
                'symbol': symbol,
                'status': 'success',
                'agents': results,
                'total_time': time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error entrenando agentes para {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': str(e)
            }
    
    async def backtest_agents(self, 
                        symbol: str, 
                        test_data: pd.DataFrame,
                        initial_balance: float = 10000.0,
                        commission: float = 0.001) -> Dict[str, Any]:
        """
        Realizar backtest de agentes con datos históricos.
        
        Args:
            symbol: Símbolo (ej. 'BTC/USDT')
            test_data: DataFrame con datos para backtest
            initial_balance: Balance inicial
            commission: Comisión por operación
            
        Returns:
            Diccionario con resultados de backtest
        """
        start_time = time.time()
        self.logger.info(f"Iniciando backtest para {symbol} con {len(test_data)} registros")
        
        # Verificar si tenemos agentes para este símbolo
        if symbol not in self.agents:
            self.logger.warning(f"No hay agentes cargados para {symbol}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': 'No hay agentes cargados para este símbolo'
            }
        
        # Verificar si hay suficientes datos
        if len(test_data) < 100:
            self.logger.warning(f"Datos de test insuficientes para {symbol}: {len(test_data)} < 100")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': f"Datos de test insuficientes: {len(test_data)} < 100"
            }
        
        try:
            # Obtener entorno
            env = self.environments[symbol]
            
            # Actualizar datos en el entorno
            env.update_data(test_data, is_training=False)
            
            # Configurar backtest
            env.configure_backtest(
                initial_balance=initial_balance,
                commission=commission
            )
            
            # Resultados por agente
            results = {}
            
            # Ejecutar backtest para cada agente
            for agent_type, agent in self.agents[symbol].items():
                agent_start_time = time.time()
                
                # Ejecutar backtest
                backtest_result = await self._run_backtest(agent, env)
                
                # Añadir a resultados
                results[agent_type] = {
                    'status': 'success',
                    'metrics': backtest_result['metrics'],
                    'trades': len(backtest_result['trades']),
                    'backtest_time': time.time() - agent_start_time
                }
                
                self.logger.info(f"Backtest de {agent_type} para {symbol} completado")
            
            # Si usamos ensemble, ejecutar backtest con strategy combinada
            if self.use_ensemble and len(self.agents[symbol]) > 1:
                # Configurar backtest de ensemble
                ensemble_result = await self._run_ensemble_backtest(self.agents[symbol], env)
                
                # Añadir a resultados
                results['ensemble'] = {
                    'status': 'success',
                    'metrics': ensemble_result['metrics'],
                    'trades': len(ensemble_result['trades']),
                    'backtest_time': ensemble_result['execution_time']
                }
            
            # Generar gráfico de backtest
            plot_data = await self._generate_backtest_plot(env, results)
            
            return {
                'symbol': symbol,
                'status': 'success',
                'agents': results,
                'total_time': time.time() - start_time,
                'chart': plot_data
            }
            
        except Exception as e:
            self.logger.error(f"Error en backtest para {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': str(e)
            }
    
    async def _run_backtest(self, agent, env) -> Dict[str, Any]:
        """
        Ejecutar backtest con un agente.
        
        Args:
            agent: Agente de RL
            env: Entorno de trading
            
        Returns:
            Diccionario con resultados
        """
        # Resetear entorno
        state = env.reset(mode='backtest')
        done = False
        
        # Ejecutar episodio completo
        while not done:
            # Tomar acción
            action_result = await self._get_agent_action(agent, state)
            action = action_result['action']
            
            # Aplicar acción
            next_state, reward, done, info = env.step(action)
            state = next_state
        
        # Obtener resultados
        backtest_summary = env.get_backtest_summary()
        trades = env.get_backtest_trades()
        
        return {
            'metrics': backtest_summary,
            'trades': trades
        }
    
    async def _run_ensemble_backtest(self, agents: Dict[str, Any], env) -> Dict[str, Any]:
        """
        Ejecutar backtest con estrategia ensemble.
        
        Args:
            agents: Diccionario de agentes
            env: Entorno de trading
            
        Returns:
            Diccionario con resultados
        """
        start_time = time.time()
        
        # Resetear entorno
        state = env.reset(mode='backtest')
        done = False
        
        # Ejecutar episodio completo
        while not done:
            # Obtener acción de cada agente
            actions = {}
            confidence = {}
            
            for agent_type, agent in agents.items():
                action_result = await self._get_agent_action(agent, state)
                actions[agent_type] = action_result['action']
                confidence[agent_type] = action_result['confidence']
            
            # Combinar acciones
            final_action, _ = self._combine_actions(actions, confidence)
            
            # Aplicar acción
            next_state, reward, done, info = env.step(final_action)
            state = next_state
        
        # Obtener resultados
        backtest_summary = env.get_backtest_summary()
        trades = env.get_backtest_trades()
        
        return {
            'metrics': backtest_summary,
            'trades': trades,
            'execution_time': time.time() - start_time
        }
    
    async def _generate_backtest_plot(self, env, results) -> str:
        """
        Generar gráfico de backtest.
        
        Args:
            env: Entorno de trading
            results: Resultados de backtest
            
        Returns:
            Imagen en formato base64
        """
        # Obtener datos para graficar
        equity_curves = env.get_backtest_equity_curves()
        price_data = env.get_price_data()
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Graficar precio
        ax1.plot(price_data.index, price_data['close'], 'k-', label='Precio', alpha=0.7)
        
        # Graficar equity curves
        for agent_type, curve in equity_curves.items():
            ax1.plot(curve.index, curve['equity'], label=f'{agent_type.upper()}')
        
        # Formato del gráfico de precio/equity
        ax1.set_title('Backtest RL - Precio y Equity')
        ax1.set_ylabel('Valor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graficar drawdown
        for agent_type, curve in equity_curves.items():
            drawdown = curve['drawdown']
            ax2.fill_between(curve.index, 0, -drawdown, alpha=0.5, label=f'{agent_type.upper()} Drawdown')
        
        # Formato del gráfico de drawdown
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown %')
        ax2.set_xlabel('Fecha')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    async def evaluate_and_optimize(self, 
                             symbol: str, 
                             historical_data: pd.DataFrame,
                             num_trials: int = 20) -> Dict[str, Any]:
        """
        Evaluar y optimizar agentes con búsqueda de hiperparámetros.
        
        Args:
            symbol: Símbolo (ej. 'BTC/USDT')
            historical_data: DataFrame con datos históricos
            num_trials: Número de pruebas
            
        Returns:
            Diccionario con resultados de optimización
        """
        start_time = time.time()
        self.logger.info(f"Iniciando optimización para {symbol} con {num_trials} pruebas")
        
        # Aquí se implementaría la optimización de hiperparámetros
        # Esto es un placeholder para el diseño inicial
        
        results = {
            'symbol': symbol,
            'status': 'success',
            'message': 'Optimización de hiperparámetros pendiente de implementación',
            'trials': num_trials,
            'total_time': time.time() - start_time
        }
        
        return results
    
    def get_signal_description(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtener descripción textual de una señal RL.
        
        Args:
            signal_data: Señal generada por process_market_data
            
        Returns:
            Diccionario con descripción
        """
        # Extraer datos principales
        symbol = signal_data.get('symbol', 'Unknown')
        action = signal_data.get('action_name', 'UNKNOWN')
        confidence = signal_data.get('confidence', 0.0)
        
        # Determinar nivel de confianza
        if confidence > 0.8:
            confidence_level = "alta"
        elif confidence > 0.6:
            confidence_level = "media-alta"
        elif confidence > 0.4:
            confidence_level = "media"
        elif confidence > 0.2:
            confidence_level = "media-baja"
        else:
            confidence_level = "baja"
        
        # Generar mensaje de recomendación
        if action == "BUY":
            recommendation = f"El sistema recomienda COMPRAR {symbol} con confianza {confidence_level} ({confidence:.2f})."
        elif action == "SELL":
            recommendation = f"El sistema recomienda VENDER {symbol} con confianza {confidence_level} ({confidence:.2f})."
        else:  # HOLD
            recommendation = f"El sistema recomienda MANTENER posición en {symbol} con confianza {confidence_level} ({confidence:.2f})."
        
        # Añadir información sobre fuentes (si está disponible)
        if 'agent_actions' in signal_data:
            agents_info = []
            for agent_type, agent_action in signal_data['agent_actions'].items():
                agent_conf = signal_data['agent_confidence'].get(agent_type, 0.0)
                agents_info.append(f"{agent_type.upper()}: {agent_action} ({agent_conf:.2f})")
            
            agents_text = "\nRecomendaciones por agente:\n" + "\n".join(agents_info)
        else:
            agent_type = signal_data.get('agent_type', 'rl')
            agents_text = f"\nRecomendación basada en agente {agent_type.upper()}."
        
        # Construir respuesta completa
        description = {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'recommendation': recommendation,
            'details': agents_text,
            'timestamp': datetime.now().isoformat()
        }
        
        return description
    
    async def update_from_execution_feedback(self, 
                                      symbol: str,
                                      signal: Dict[str, Any],
                                      execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualizar agentes con feedback de ejecución.
        
        Args:
            symbol: Símbolo (ej. 'BTC/USDT')
            signal: Señal original
            execution_result: Resultado de ejecución
            
        Returns:
            Diccionario con resultado de actualización
        """
        self.logger.info(f"Actualizando agentes con feedback de ejecución para {symbol}")
        
        # Verificar si tenemos agentes para este símbolo
        if symbol not in self.agents:
            self.logger.warning(f"No hay agentes cargados para {symbol}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': 'No hay agentes cargados para este símbolo'
            }
        
        try:
            # Extraer datos relevantes
            action = signal.get('action', 0)
            result = execution_result.get('result', 'unknown')
            reward = execution_result.get('realized_pnl', 0.0)
            
            # Actualizar estadísticas
            if result == 'success':
                self.stats['successful_trades'] += 1
            else:
                self.stats['unsuccessful_trades'] += 1
            
            self.stats['total_reward'] += reward
            
            # Actualizar agentes con la experiencia
            # Esto requeriría implementación adicional para cada tipo de agente
            
            return {
                'symbol': symbol,
                'status': 'success',
                'updated_agents': list(self.agents[symbol].keys()),
                'reward': reward
            }
            
        except Exception as e:
            self.logger.error(f"Error actualizando agentes con feedback: {str(e)}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del integrador RL.
        
        Returns:
            Diccionario con estadísticas
        """
        # Calcular métricas adicionales
        total_trades = self.stats['successful_trades'] + self.stats['unsuccessful_trades']
        success_rate = self.stats['successful_trades'] / total_trades if total_trades > 0 else 0.0
        
        # Añadir información sobre modelos
        num_models = len(self.stats['model_versions'])
        
        # Añadir configuración actual
        stats = {
            'statistics': {
                'inferences': self.stats['inferences'],
                'successful_trades': self.stats['successful_trades'],
                'unsuccessful_trades': self.stats['unsuccessful_trades'],
                'total_trades': total_trades,
                'success_rate': success_rate,
                'total_reward': float(self.stats['total_reward']),
                'last_train_time': self.stats['last_train_time'],
                'num_models': num_models
            },
            'configuration': self.config,
            'models': self.stats['model_versions'],
            'timestamp': datetime.now().isoformat()
        }
        
        return stats