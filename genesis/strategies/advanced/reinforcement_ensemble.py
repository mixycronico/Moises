"""
Estrategia avanzada basada en Reinforcement Learning con ensemble de modelos.

Esta implementación completa integra múltiples modelos de RL (DQN, PPO, SAC)
con indicadores técnicos avanzados, análisis de sentimiento y capacidades
de meta-aprendizaje para adaptarse dinámicamente a las condiciones del mercado.

Requiere las bibliotecas gymnasium y stable-baselines3 para funcionar completamente.
"""
import logging
import asyncio
import random
from typing import Dict, Any, List, Optional, Tuple, Union, cast
import time
import json
import os
from datetime import datetime

# Importar NumPy con manejo de errores
try:
    import numpy as np
except ImportError:
    logging.warning("NumPy no encontrado. Algunas funciones no estarán disponibles.")
    # Crear un módulo alternativo mínimo para np
    class NumpyReplacement:
        # Constantes
        inf = float('inf')
        nan = float('nan')
        
        # Tipos de datos
        float32 = float
        float64 = float
        int32 = int
        int64 = int
        
        def __init__(self):
            pass
        
        def array(self, data, dtype=None):
            return data
            
        def astype(self, dtype):
            """Método para simular conversión de tipos."""
            return self
            
        def zeros_like(self, data):
            if isinstance(data, list):
                return [0] * len(data)
            return 0
            
        def zeros(self, shape):
            """Crear array de ceros."""
            if isinstance(shape, tuple):
                if len(shape) == 1:
                    return [0] * shape[0]
                # Simplificado para arrays 2D
                return [[0] * shape[1] for _ in range(shape[0])]
            else:
                return [0] * shape
                
        def ones(self, shape):
            """Crear array de unos."""
            if isinstance(shape, tuple):
                if len(shape) == 1:
                    return [1] * shape[0]
                # Simplificado para arrays 2D
                return [[1] * shape[1] for _ in range(shape[0])]
            else:
                return [1] * shape
            
        def mean(self, data):
            if not data:
                return 0
            return sum(data) / len(data)
            
        def std(self, data):
            if not data:
                return 0
            mean = self.mean(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return variance ** 0.5
            
        def diff(self, data):
            return [data[i+1] - data[i] for i in range(len(data)-1)]
            
        def concatenate(self, arrays, axis=0):
            result = []
            for arr in arrays:
                if isinstance(arr, list):
                    result.extend(arr)
                else:
                    result.append(arr)
            return result
            
        def cumprod(self, data):
            result = [1]
            for i in range(len(data)):
                result.append(result[-1] * (1 + data[i]))
            return result[1:]
            
        def maximum(self):
            class MaxAccumulate:
                def accumulate(self, data):
                    result = []
                    current_max = data[0] if data else 0
                    for val in data:
                        current_max = max(current_max, val)
                        result.append(current_max)
                    return result
            return MaxAccumulate()
            
        def max(self, data):
            if not data:
                return 0
            return max(data)
            
        def sqrt(self, value):
            return value ** 0.5
            
        def convolve(self, a, v, mode='full'):
            """Implementación básica de convolución."""
            return a  # Simplificado
            
        # Clase random para generar números aleatorios
        class random:
            @staticmethod
            def normal(loc=0.0, scale=1.0, size=None):
                import random
                if size is None:
                    return random.normalvariate(loc, scale)
                if isinstance(size, int):
                    return [random.normalvariate(loc, scale) for _ in range(size)]
                return [random.normalvariate(loc, scale) for _ in range(size[0])]
                    
            @staticmethod
            def uniform(low=0.0, high=1.0, size=None):
                import random
                if size is None:
                    return random.uniform(low, high)
                if isinstance(size, int):
                    return [random.uniform(low, high) for _ in range(size)]
                return [random.uniform(low, high) for _ in range(size[0])]
            
        def random(self):
            class RandomGen:
                def normal(self, mean, std, size):
                    import random
                    return [random.normalvariate(mean, std) for _ in range(size)]
                    
                def uniform(self, low, high, size):
                    import random
                    return [random.uniform(low, high) for _ in range(size)]
            return RandomGen()
            
        def abs(self, value):
            return abs(value)
            
        def inf(self):
            return float('inf')
            
    np = NumpyReplacement()

# Intentar importar pandas
try:
    import pandas as pd
except ImportError:
    logging.warning("Pandas no encontrado. Algunas funciones no estarán disponibles.")

# Definir clases de espacio para RL
class GymSpaceReplacement:
    class Box:
        def __init__(self, low, high, shape, dtype=None):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype
            
    class Discrete:
        def __init__(self, n):
            self.n = n

# Clases de reemplazo para RL
class GymEnvReplacement:
    """Clase base para entornos de gymnasium."""
    class Env:
        """Clase base para entornos."""
        def __init__(self):
            pass
            
        def reset(self, **kwargs):
            pass
            
        def step(self, action):
            pass
    
    def __init__(self):
        pass
        
    def reset(self, **kwargs):
        pass
        
    def step(self, action):
        pass

class DQNReplacement:
    def __init__(self, policy, env, verbose=0, tensorboard_log=None):
        self.policy = policy
        self.env = env
        self.verbose = verbose
        
    @staticmethod
    def load(path, env=None):
        return DQNReplacement("MlpPolicy", env)
        
    def predict(self, observation, deterministic=True):
        import random
        return random.randint(0, 2), None
        
    def save(self, path):
        pass

class PPOReplacement:
    def __init__(self, policy, env, verbose=0, tensorboard_log=None):
        self.policy = policy
        self.env = env
        self.verbose = verbose
        
    @staticmethod
    def load(path, env=None):
        return PPOReplacement("MlpPolicy", env)
        
    def predict(self, observation, deterministic=True):
        import random
        return random.randint(0, 2), None
        
    def save(self, path):
        pass

class SACReplacement:
    def __init__(self, policy, env, verbose=0, tensorboard_log=None):
        self.policy = policy
        self.env = env
        self.verbose = verbose
        
    @staticmethod
    def load(path, env=None):
        return SACReplacement("MlpPolicy", env)
        
    def predict(self, observation, deterministic=True):
        import random
        return random.randint(0, 2), None
        
    def save(self, path):
        pass

class BaseCallbackReplacement:
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.n_calls = 0
        self.model = None
        
    def _init_callback(self):
        pass
        
    def _on_step(self):
        self.n_calls += 1
        return True

# Intentar importar biblioteca de RL
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import DQN, PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    RL_LIBRARIES_AVAILABLE = True
except ImportError:
    logging.warning("Bibliotecas gymnasium o stable-baselines3 no encontradas. Usando versión simplificada.")
    RL_LIBRARIES_AVAILABLE = False
    gym = cast(Any, GymEnvReplacement)
    spaces = cast(Any, GymSpaceReplacement)
    DQN = cast(Any, DQNReplacement)
    PPO = cast(Any, PPOReplacement)
    SAC = cast(Any, SACReplacement)
    BaseCallback = cast(Any, BaseCallbackReplacement)

# Importar componentes de DeepSeek si están disponibles
try:
    from genesis.lsml.deepseek_integrator import DeepSeekIntegrator
    from genesis.lsml import deepseek_config
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logging.warning("Módulos DeepSeek no encontrados. La estrategia funcionará sin capacidades avanzadas de análisis.")
    
    # Crear clases de reemplazo para DeepSeek
    class DeepSeekConfigReplacement:
        @staticmethod
        def is_enabled():
            return False
    
    class DeepSeekIntegratorReplacement:
        def __init__(self, api_key=None, intelligence_factor=1.0):
            self.api_key = api_key
            self.intelligence_factor = intelligence_factor
            
        async def initialize(self):
            return True
            
        async def analyze_trading_opportunities(self, market_data, news_data=None):
            return {"error": "DeepSeek no disponible"}
            
        async def enhance_trading_signals(self, signals, market_data):
            return signals
    
    deepseek_config = cast(Any, DeepSeekConfigReplacement())
    DeepSeekIntegrator = cast(Any, DeepSeekIntegratorReplacement)

# Componentes del Sistema Genesis
from genesis.strategies.base import Strategy

# Importar indicadores con manejo de errores
try:
    from genesis.analysis.advanced_indicators import (
        calculate_ichimoku_cloud,
        calculate_dynamic_bollinger_bands,
        calculate_directional_movement_index,
    )
except ImportError:
    logging.warning("Módulos de indicadores avanzados no encontrados. Usando funciones de reemplazo.")
    
    def calculate_ichimoku_cloud(high, low, close, tenkan_period=9, kijun_period=26, 
                               senkou_span_b_period=52, displacement=26):
        """Función de reemplazo para ichimoku."""
        return {
            "tenkan": [0] * len(close),
            "kijun": [0] * len(close),
            "senkou_span_a": [0] * len(close),
            "senkou_span_b": [0] * len(close),
            "chikou_span": [0] * len(close),
            "close": close
        }
        
    def calculate_dynamic_bollinger_bands(close, window=20, num_std_dev=2.0):
        """Función de reemplazo para bandas de bollinger."""
        middle = [0] * len(close)
        upper = [0] * len(close)
        lower = [0] * len(close)
        return {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "close": close
        }
        
    def calculate_directional_movement_index(high, low, close, window=14, period=14):
        """Función de reemplazo para DMI."""
        return {
            "adx": [0] * len(close),
            "di_plus": [0] * len(close),
            "di_minus": [0] * len(close)
        }
        
    def calculate_vwap(high, low, close, volume):
        """Función de reemplazo para VWAP."""
        return [0] * len(close)
        
    def calculate_volume_profile(close, volume, num_bins=10):
        """Función de reemplazo para perfil de volumen."""
        return {
            "bins": [0] * num_bins,
            "volumes": [0] * num_bins
        }
        
    def calculate_market_profile(high, low, close, num_bins=10):
        """Función de reemplazo para perfil de mercado."""
        return {
            "bins": [0] * num_bins,
            "frequencies": [0] * num_bins
        }
else:
    # Definir funciones que podrían faltar
    def calculate_vwap(high, low, close, volume):
        """Calcular VWAP (Volume Weighted Average Price)."""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
        
    def calculate_volume_profile(close, volume, num_bins=10):
        """Calcular perfil de volumen."""
        # Manejo defensivo para arrays vacíos o datos inválidos
        if len(close) == 0 or len(volume) == 0:
            return {"bins": [0] * num_bins, "volumes": [0] * num_bins}
            
        # Verificar si hay NaN y usar valores válidos solamente
        valid_indices = [i for i, (p, v) in enumerate(zip(close, volume)) 
                         if not (np.isnan(p) or np.isnan(v))]
        
        if not valid_indices:
            return {"bins": [0] * num_bins, "volumes": [0] * num_bins}
            
        valid_prices = [close[i] for i in valid_indices]
        valid_volumes = [volume[i] for i in valid_indices]
        
        # Simplificado
        min_price = min(valid_prices)
        max_price = max(valid_prices)
        
        # Evitar división por cero
        if min_price == max_price:
            bins = [min_price] * num_bins
            volumes = [sum(valid_volumes) / num_bins] * num_bins
            return {"bins": bins, "volumes": volumes}
            
        bin_size = (max_price - min_price) / num_bins
        
        bins = [min_price + i * bin_size for i in range(num_bins)]
        volumes = [0] * num_bins
        
        for i, price in enumerate(valid_prices):
            # Asegurar que el cálculo del índice es seguro
            try:
                bin_idx = min(int((price - min_price) / bin_size), num_bins - 1)
                volumes[bin_idx] += valid_volumes[i]
            except (ValueError, TypeError, OverflowError):
                # Ignorar valores que no se pueden procesar
                continue
            
        return {"bins": bins, "volumes": volumes}
        
    def calculate_market_profile(high, low, close, num_bins=10):
        """Calcular perfil de mercado."""
        # Manejo defensivo para arrays vacíos o datos inválidos
        if len(high) == 0 or len(low) == 0 or len(close) == 0:
            return {"bins": [0] * num_bins, "frequencies": [0] * num_bins}
        
        # Verificar si hay NaN y usar valores válidos solamente
        valid_indices = [i for i, (h, l, c) in enumerate(zip(high, low, close)) 
                         if not (np.isnan(h) or np.isnan(l) or np.isnan(c))]
        
        if not valid_indices:
            return {"bins": [0] * num_bins, "frequencies": [0] * num_bins}
            
        valid_high = [high[i] for i in valid_indices]
        valid_low = [low[i] for i in valid_indices]
        valid_close = [close[i] for i in valid_indices]
        
        # Simplificado
        try:
            min_price = min(valid_low)
            max_price = max(valid_high)
            
            # Evitar división por cero
            if min_price == max_price:
                bins = [min_price] * num_bins
                return {"bins": bins, "frequencies": [len(valid_close) // num_bins] * num_bins}
                
            bin_size = (max_price - min_price) / num_bins
            
            bins = [min_price + i * bin_size for i in range(num_bins)]
            frequencies = [0] * num_bins
            
            for i in range(len(valid_close)):
                for bin_idx, bin_price in enumerate(bins):
                    try:
                        if valid_low[i] <= bin_price <= valid_high[i]:
                            frequencies[bin_idx] += 1
                    except (IndexError, TypeError):
                        continue
                        
            return {"bins": bins, "frequencies": frequencies}
        except (ValueError, TypeError) as e:
            # En caso de error, devolver datos vacíos
            return {"bins": [0] * num_bins, "frequencies": [0] * num_bins}

# Importar análisis de sentimiento con manejo de errores
try:
    from genesis.analysis.sentiment import (
        calculate_sentiment_score,
        analyze_social_volume
    )
except ImportError:
    logging.warning("Módulos de análisis de sentimiento no encontrados. Usando funciones de reemplazo.")
    
    def calculate_sentiment_score(text_data, source="news"):
        """Función de reemplazo para cálculo de sentimiento."""
        return {"score": 0.0, "magnitude": 0.0, "label": "neutral"}
        
    def analyze_social_volume(symbol, timeframe="1d", sources=None):
        """Función de reemplazo para análisis de volumen social."""
        return {"volume": 0, "change": 0.0, "sentiment": "neutral"}

# Importar gestor de APIs con manejo de errores
try:
    from genesis.api_integration import api_manager
except ImportError:
    logging.warning("Módulo de integración de APIs no encontrado.")
    
    class ApiManagerReplacement:
        def __init__(self):
            self.initialized = False
            
        async def get_alpha_vantage_data(self, **kwargs):
            return {"error": "API manager no disponible"}
            
        async def get_news_api_data(self, **kwargs):
            return {"error": "API manager no disponible"}
            
        async def get_coinmarketcap_data(self, **kwargs):
            return {"error": "API manager no disponible"}
            
        async def call_deepseek_api(self, **kwargs):
            return {"error": "API manager no disponible"}
            
    api_manager = ApiManagerReplacement()

# Logger para esta estrategia
logger = logging.getLogger('genesis.strategies.advanced.reinforcement_ensemble')

class TradingEnvironment(gym.Env):
    """
    Entorno de trading para Reinforcement Learning.
    
    Esta clase proporciona una interfaz compatible con gymnasium para
    que los agentes de RL puedan interactuar con el entorno de trading.
    """
    
    def __init__(self, 
                 data: Dict[str, Any],
                 symbol: str,
                 initial_balance: float = 10000.0,
                 commission: float = 0.001,
                 window_size: int = 30,
                 reward_scaling: float = 1.0):
        """
        Inicializar entorno de trading.
        
        Args:
            data: Datos históricos OHLCV y otros indicadores
            symbol: Símbolo del instrumento a negociar
            initial_balance: Balance inicial para simulación
            commission: Comisión por operación (porcentaje)
            window_size: Tamaño de la ventana de observación
            reward_scaling: Factor para escalar la recompensa
        """
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        
        # Estado del entorno
        self.position = 0  # 0 = sin posición, 1 = long, -1 = short
        self.position_price = 0.0
        self.current_step = window_size
        self.trade_history = []
        self.returns = []
        
        # Definir espacio de observación (estado)
        # Incluye OHLCV, indicadores técnicos y estado de la posición
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size * 10 + 2,), 
            dtype=np.float32
        )
        
        # Definir espacio de acción: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
    def reset(self, **kwargs):
        """
        Reiniciar el entorno al estado inicial.
        
        Returns:
            Estado inicial del entorno
        """
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0.0
        self.current_step = self.window_size
        self.trade_history = []
        self.returns = []
        
        initial_observation = self._get_observation()
        return initial_observation, {}
    
    def step(self, action):
        """
        Ejecutar una acción en el entorno.
        
        Args:
            action: Acción a ejecutar (0 = hold, 1 = buy, 2 = sell)
            
        Returns:
            Tupla (observación, recompensa, terminado, truncado, info)
        """
        # Guardar estado anterior para calcular recompensa
        previous_balance = self.balance
        previous_position = self.position
        
        # Procesar acción
        current_price = self.data['close'][self.current_step]
        
        # Acción 1: Comprar
        if action == 1 and self.position <= 0:
            if self.position == -1:  # Cerrar short existente
                profit = self.position_price - current_price
                self.balance += self.balance * abs(self.position) * profit / self.position_price
                self.balance -= self.balance * abs(self.position) * self.commission
                self.trade_history.append({
                    'step': self.current_step,
                    'type': 'close_short',
                    'price': current_price,
                    'balance': self.balance
                })
            
            # Abrir long
            self.position = 1
            self.position_price = current_price
            self.balance -= self.balance * self.commission
            self.trade_history.append({
                'step': self.current_step,
                'type': 'buy',
                'price': current_price,
                'balance': self.balance
            })
            
        # Acción 2: Vender
        elif action == 2 and self.position >= 0:
            if self.position == 1:  # Cerrar long existente
                profit = current_price - self.position_price
                self.balance += self.balance * abs(self.position) * profit / self.position_price
                self.balance -= self.balance * abs(self.position) * self.commission
                self.trade_history.append({
                    'step': self.current_step,
                    'type': 'close_long',
                    'price': current_price,
                    'balance': self.balance
                })
            
            # Abrir short
            self.position = -1
            self.position_price = current_price
            self.balance -= self.balance * self.commission
            self.trade_history.append({
                'step': self.current_step,
                'type': 'sell',
                'price': current_price,
                'balance': self.balance
            })
        
        # Actualizar posición actual
        if self.position != 0:
            if self.position == 1:
                unrealized_profit = (current_price - self.position_price) / self.position_price
            else:
                unrealized_profit = (self.position_price - current_price) / self.position_price
            
            virtual_balance = self.balance * (1 + abs(self.position) * unrealized_profit)
        else:
            virtual_balance = self.balance
        
        # Calcular recompensa
        reward = ((virtual_balance - previous_balance) / previous_balance) * self.reward_scaling
        
        # Actualizar contador de pasos
        self.current_step += 1
        
        # Verificar si el episodio ha terminado
        done = self.current_step >= len(self.data['close']) - 1
        
        # Obtener nueva observación
        observation = self._get_observation()
        
        # Información adicional para debugging
        info = {
            'balance': self.balance,
            'virtual_balance': virtual_balance,
            'position': self.position,
            'position_price': self.position_price,
            'step': self.current_step,
            'current_price': current_price
        }
        
        # Guardar retorno para métricas
        self.returns.append((virtual_balance - previous_balance) / previous_balance)
        
        return observation, reward, done, False, info
    
    def _get_observation(self):
        """
        Obtener observación del estado actual.
        
        Returns:
            Vector de características para el RL
        """
        # Obtener ventana de datos
        obs_window = {
            'close': self.data['close'][self.current_step - self.window_size:self.current_step],
            'high': self.data['high'][self.current_step - self.window_size:self.current_step],
            'low': self.data['low'][self.current_step - self.window_size:self.current_step],
            'open': self.data['open'][self.current_step - self.window_size:self.current_step],
            'volume': self.data['volume'][self.current_step - self.window_size:self.current_step]
        }
        
        # Normalizar datos para que estén en rango similar
        for key in obs_window:
            # Evitar división por cero
            if np.std(obs_window[key]) > 0:
                obs_window[key] = (obs_window[key] - np.mean(obs_window[key])) / np.std(obs_window[key])
            else:
                obs_window[key] = obs_window[key] - np.mean(obs_window[key]) if np.mean(obs_window[key]) > 0 else obs_window[key]
        
        # Agregar indicadores técnicos
        rsi = self.data.get('rsi', np.zeros(self.window_size))[self.current_step - self.window_size:self.current_step]
        macd = self.data.get('macd', np.zeros(self.window_size))[self.current_step - self.window_size:self.current_step]
        macd_signal = self.data.get('macd_signal', np.zeros(self.window_size))[self.current_step - self.window_size:self.current_step]
        bb_upper = self.data.get('bb_upper', np.zeros(self.window_size))[self.current_step - self.window_size:self.current_step]
        bb_lower = self.data.get('bb_lower', np.zeros(self.window_size))[self.current_step - self.window_size:self.current_step]
        
        # Normalizar indicadores
        for indicator in [rsi, macd, macd_signal, bb_upper, bb_lower]:
            if np.std(indicator) > 0:
                indicator = (indicator - np.mean(indicator)) / np.std(indicator)
            else:
                indicator = indicator - np.mean(indicator) if np.mean(indicator) > 0 else indicator
        
        # Aplanar todos los datos en un solo vector
        flattened = np.concatenate([
            obs_window['close'],
            obs_window['high'],
            obs_window['low'],
            obs_window['open'],
            obs_window['volume'],
            rsi,
            macd,
            macd_signal,
            bb_upper,
            bb_lower,
            [self.position],  # Estado de la posición actual
            [self.balance / self.initial_balance]  # Balance normalizado
        ])
        
        return flattened.astype(np.float32)
    
    def render(self, mode='human'):
        """
        Visualizar el estado del entorno.
        """
        pass  # Implementación opcional para visualización
    
    def get_performance_metrics(self):
        """
        Obtener métricas de rendimiento del agente.
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        total_return = (self.balance / self.initial_balance) - 1.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        win_rate = 0.0
        
        if len(self.returns) > 0:
            # Sharpe ratio (asumiendo retornos diarios)
            sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9) * np.sqrt(252)
            
            # Máximo drawdown
            cumulative = np.cumprod(1 + np.array(self.returns))
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / peak
            max_drawdown = np.max(drawdown)
            
            # Ratio de victorias
            if len(self.trade_history) > 1:
                profitable_trades = sum(1 for i in range(1, len(self.trade_history))
                                      if self.trade_history[i]['balance'] > self.trade_history[i-1]['balance'])
                win_rate = profitable_trades / (len(self.trade_history) - 1)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trade_history) - 1
        }


class TradingCallback(BaseCallback):
    """
    Callback para monitorear y guardar modelos durante el entrenamiento.
    """
    
    def __init__(self, check_freq=1000, save_path="models", verbose=1):
        """
        Inicializar callback.
        
        Args:
            check_freq: Frecuencia de verificación (pasos)
            save_path: Ruta para guardar modelos
            verbose: Nivel de verbosidad
        """
        super(TradingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.best_model_path = ""
    
    def _init_callback(self):
        """Inicialización al comienzo del entrenamiento."""
        # Crear directorio si no existe
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        """
        Llamada en cada paso del entrenamiento.
        
        Returns:
            True para continuar entrenamiento
        """
        if self.n_calls % self.check_freq == 0:
            # Obtener recompensa media
            x, y = self.model.logger.get_x(), self.model.logger.get_y()
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                
                # Guardar el modelo si mejora
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    model_path = os.path.join(self.save_path, f"model_{self.n_calls}_steps")
                    self.model.save(model_path)
                    
                    if self.verbose > 0:
                        print(f"Guardando nuevo mejor modelo en {model_path}")
                        print(f"Nueva mejor recompensa media: {mean_reward:.2f}")
                    
                    # Guardar ruta del mejor modelo
                    if os.path.exists(model_path + ".zip"):
                        self.best_model_path = model_path + ".zip"
        
        return True


class ReinforcementEnsembleStrategy(Strategy):
    """
    Estrategia de trading avanzada basada en ensemble de modelos de RL.
    
    Combina múltiples agentes de RL con diferentes arquitecturas y parámetros,
    junto con indicadores técnicos tradicionales y análisis de sentimiento.
    Incluye capacidades de meta-aprendizaje para adaptarse dinámicamente.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializar estrategia avanzada de RL con ensemble.
        
        Args:
            config: Configuración de la estrategia
        """
        strategy_name = config.get('name', "Reinforcement Ensemble Strategy")
        super().__init__(strategy_name)
        
        self.description = "Estrategia avanzada que combina modelos RL, indicadores técnicos y análisis DeepSeek"
        
        # Parámetros de configuración
        self.symbols = config.get('symbols', ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
        self.timeframe = config.get('timeframe', '1h')
        self.lookback_period = config.get('lookback_period', 100)
        self.initial_capital = config.get('initial_capital', 10000.0)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% del capital
        self.meta_learning_enabled = config.get('meta_learning_enabled', True)
        self.voting_threshold = config.get('voting_threshold', 0.67)  # 2/3 para consenso
        
        # Configuración de DeepSeek
        self.use_deepseek = config.get('use_deepseek', DEEPSEEK_AVAILABLE)
        self.deepseek_intelligence_factor = config.get('deepseek_intelligence_factor', 1.0)
        self.deepseek_integrator = None
        
        # Comprobar disponibilidad de bibliotecas RL
        self.use_real_rl = config.get('use_real_rl', RL_LIBRARIES_AVAILABLE)
        
        # Métricas de rendimiento
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'average_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'current_capital': self.initial_capital
        }
        
        # Estado actual de posiciones
        self.positions = {}
        
        # Historial de predicciones de cada agente
        self.agent_predictions_history = []
        
        # Modelo de rendimiento histórico por símbolo
        self.symbol_performance = {symbol: {"success_rate": 0.5} for symbol in self.symbols}
        
        # Flags para seguimiento
        self.is_initialized = False
        self.needs_retraining = False
        
        # Inicializar modelos si están disponibles las bibliotecas
        self.models = {}
        if self.use_real_rl and RL_LIBRARIES_AVAILABLE:
            self.envs = {}  # Entornos para cada símbolo
            # Se inicializarán completamente en initialize()
        else:
            # Modelos simulados si no hay bibliotecas RL
            self.models = {
                "dqn": {"weight": 0.3, "success_rate": 0.6},
                "ppo": {"weight": 0.4, "success_rate": 0.65},
                "sac": {"weight": 0.3, "success_rate": 0.55}
            }
        
        # Inicializar DeepSeek si está disponible
        if self.use_deepseek and DEEPSEEK_AVAILABLE:
            try:
                self.deepseek_integrator = DeepSeekIntegrator(
                    intelligence_factor=self.deepseek_intelligence_factor
                )
                logger.info(f"Integrador DeepSeek inicializado para la estrategia {self.name}")
            except Exception as e:
                logger.warning(f"No se pudo inicializar DeepSeek: {str(e)}. La estrategia funcionará sin análisis avanzado.")
                self.use_deepseek = False
        
        log_msg = f"Estrategia {self.name} creada con {len(self.symbols)} símbolos"
        if self.use_real_rl:
            log_msg += ", modelos RL reales"
        else:
            log_msg += ", modelos RL simulados"
            
        if self.use_deepseek:
            log_msg += " y capacidades DeepSeek activadas"
        
        logger.info(log_msg)
    
    async def initialize(self) -> bool:
        """
        Inicializar todos los componentes de la estrategia.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Crear directorios para modelos
            os.makedirs("models", exist_ok=True)
            
            # Inicializar modelos si están disponibles las bibliotecas
            if self.use_real_rl and RL_LIBRARIES_AVAILABLE:
                logger.info("Inicializando modelos RL reales...")
                
                # Inicializar con datos de ejemplo
                for symbol in self.symbols:
                    # Crear datos de ejemplo
                    sample_data = self._create_sample_data(symbol)
                    
                    # Crear entorno para este símbolo
                    env = TradingEnvironment(
                        data=sample_data,
                        symbol=symbol,
                        initial_balance=self.initial_capital,
                        window_size=20
                    )
                    self.envs[symbol] = env
                    
                    # Crear modelos para este símbolo
                    self.models[symbol] = {
                        "dqn": DQN("MlpPolicy", env, verbose=0, tensorboard_log="./dqn_logs/"),
                        "ppo": PPO("MlpPolicy", env, verbose=0, tensorboard_log="./ppo_logs/"),
                        "sac": SAC("MlpPolicy", env, verbose=0, tensorboard_log="./sac_logs/")
                    }
                    
                    # Cargar modelos pre-entrenados si existen
                    model_paths = {
                        "dqn": f"models/{symbol.replace('/', '_')}_dqn.zip",
                        "ppo": f"models/{symbol.replace('/', '_')}_ppo.zip",
                        "sac": f"models/{symbol.replace('/', '_')}_sac.zip"
                    }
                    
                    for model_type, path in model_paths.items():
                        if os.path.exists(path):
                            try:
                                if model_type == "dqn":
                                    self.models[symbol][model_type] = DQN.load(path, env=env)
                                elif model_type == "ppo":
                                    self.models[symbol][model_type] = PPO.load(path, env=env)
                                elif model_type == "sac":
                                    self.models[symbol][model_type] = SAC.load(path, env=env)
                                
                                logger.info(f"Modelo {model_type} para {symbol} cargado desde {path}")
                            except Exception as e:
                                logger.warning(f"Error al cargar modelo {model_type} para {symbol}: {str(e)}")
                
                logger.info(f"Modelos RL inicializados para {len(self.symbols)} símbolos")
            else:
                await asyncio.sleep(0.5)  # Simular tiempo de carga
            
            # Guardar configuración
            config_path = os.path.join("models", "ensemble_config.json")
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    json.dump({
                        "symbols": self.symbols,
                        "timeframe": self.timeframe,
                        "meta_learning_enabled": self.meta_learning_enabled,
                        "use_deepseek": self.use_deepseek,
                        "use_real_rl": self.use_real_rl
                    }, f)
            
            # Inicializar DeepSeek si está configurado
            if self.use_deepseek and DEEPSEEK_AVAILABLE and self.deepseek_integrator:
                try:
                    logger.info("Inicializando integrador DeepSeek...")
                    deepseek_initialized = await self.deepseek_integrator.initialize()
                    if not deepseek_initialized:
                        logger.warning("No se pudo inicializar el integrador DeepSeek. La estrategia funcionará sin capacidades avanzadas de análisis.")
                        self.use_deepseek = False
                    else:
                        logger.info("Integrador DeepSeek inicializado correctamente.")
                except Exception as e:
                    logger.warning(f"Error al inicializar DeepSeek: {str(e)}. La estrategia funcionará sin capacidades avanzadas de análisis.")
                    self.use_deepseek = False
            
            self.is_initialized = True
            log_msg = f"Estrategia {self.name} inicializada correctamente"
            if self.use_real_rl:
                log_msg += " con modelos RL reales"
            if self.use_deepseek:
                log_msg += " y con integración DeepSeek"
            logger.info(log_msg)
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar estrategia {self.name}: {str(e)}")
            import traceback
            logger.debug(f"Detalle del error: {traceback.format_exc()}")
            return False
    
    async def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar señal de trading basada en el estado actual del mercado.
        
        Utiliza una combinación de modelos RL, análisis técnico tradicional,
        y análisis avanzado mediante DeepSeek si está disponible.
        
        Args:
            symbol: Símbolo de trading
            data: Datos del mercado actual (OHLCV, indicadores, etc.)
            
        Returns:
            Diccionario con señal y metadatos
        """
        if not self.is_initialized:
            logger.warning("Estrategia no inicializada, inicializando...")
            await self.initialize()
        
        if not symbol or symbol not in self.symbols:
            return {'signal': 'NONE', 'reason': 'Symbol not supported'}
        
        try:
            # Extraer OHLCV y otros datos
            ohlcv = data.get('ohlcv', {})
            news_data = data.get('news', [])
            sentiment_data = data.get('sentiment', {})
            
            # Calcular indicadores avanzados
            indicators = await self._calculate_indicators(symbol, ohlcv)
            
            # Análisis técnico tradicional
            analysis_results = self._analyze_technical_indicators(indicators)
            
            # Predicciones de modelos RL
            if self.use_real_rl and RL_LIBRARIES_AVAILABLE and symbol in self.models:
                # Usar modelos reales de RL
                agent_predictions = await self._get_real_rl_predictions(symbol, ohlcv, indicators)
            else:
                # Usar simulación de predicciones
                agent_predictions = self._simulate_agent_predictions(analysis_results, symbol)
            
            # Consenso de agentes
            consensus_signal, confidence = self._get_ensemble_consensus(agent_predictions)
            
            # Evaluar riesgo
            risk_assessment = self._assess_risk(symbol, consensus_signal, confidence, ohlcv, indicators)
            
            # Integración con DeepSeek si está disponible
            deepseek_analysis = None
            if self.use_deepseek and DEEPSEEK_AVAILABLE and self.deepseek_integrator:
                # Verificar si DeepSeek está habilitado en la configuración global
                if not deepseek_config.is_enabled():
                    logger.info(f"DeepSeek está desactivado en la configuración global. Usando análisis tradicional para {symbol}")
                else:
                    try:
                        # Preparar datos para análisis DeepSeek
                        # Convertir arrays numpy a listas para evitar problemas de serialización
                        serializable_ohlcv = {}
                        for k, v in ohlcv.items():
                            if hasattr(v, 'tolist'):  # Si es un array numpy
                                serializable_ohlcv[k] = v.tolist()
                            else:
                                serializable_ohlcv[k] = v
                        
                        # Serializar indicadores técnicos
                        serializable_indicators = {}
                        for k, v in indicators.items():
                            if hasattr(v, 'tolist'):  # Si es un array numpy
                                serializable_indicators[k] = v.tolist()
                            elif isinstance(v, dict):  # Si es un diccionario anidado
                                serializable_indicators[k] = {}
                                for k2, v2 in v.items():
                                    if hasattr(v2, 'tolist'):
                                        serializable_indicators[k][k2] = v2.tolist()
                                    else:
                                        serializable_indicators[k][k2] = v2
                            else:
                                serializable_indicators[k] = v
                        
                        market_data = {
                            'symbol': symbol,
                            'ohlcv': serializable_ohlcv,
                            'indicators': serializable_indicators,
                            'technical_analysis': analysis_results,
                            'agent_predictions': agent_predictions,
                            'sentiment': sentiment_data,
                            'consensus': {
                                'signal': consensus_signal,
                                'confidence': confidence
                            }
                        }
                        
                        # Solicitar análisis avanzado a DeepSeek
                        logger.info(f"Solicitando análisis DeepSeek para {symbol}")
                        deepseek_analysis = await self.deepseek_integrator.analyze_trading_opportunities(
                            market_data=market_data,
                            news_data=news_data
                        )
                        
                        # Mejorar señales con DeepSeek
                        if deepseek_analysis and 'error' not in deepseek_analysis:
                            # Transformar la decisión original a formato de señal
                            original_signal = [{
                                'symbol': symbol,
                                'signal': consensus_signal,
                                'confidence': confidence,
                                'risk_assessment': risk_assessment
                            }]
                            
                            # Obtener señales mejoradas
                            enhanced_signals = await self.deepseek_integrator.enhance_trading_signals(
                                signals=original_signal,
                                market_data=market_data
                            )
                            
                            if enhanced_signals and len(enhanced_signals) > 0:
                                # Usar la primera señal mejorada (solo tenemos una)
                                enhanced = enhanced_signals[0]
                                logger.info(f"DeepSeek mejoró la señal para {symbol}: confianza ajustada de {confidence} a {enhanced.get('deepseek_confidence', confidence)}")
                                
                                # Actualizar con las mejoras de DeepSeek
                                consensus_signal = enhanced.get('signal', consensus_signal)
                                confidence = enhanced.get('deepseek_confidence', confidence)
                                
                                # Considerar niveles mejorados en la evaluación de riesgo
                                if 'enhanced_stop_loss' in enhanced or 'enhanced_take_profit' in enhanced:
                                    risk_assessment = self._adjust_risk_with_deepseek(
                                        risk_assessment, 
                                        enhanced.get('enhanced_stop_loss'), 
                                        enhanced.get('enhanced_take_profit')
                                    )
                        else:
                            logger.warning(f"Análisis DeepSeek no disponible o con error: {deepseek_analysis.get('error', 'Error desconocido')}")
                    
                    except Exception as e:
                        logger.warning(f"Error en análisis DeepSeek para {symbol}: {str(e)}")
                        # Continuar con la estrategia normal si falla DeepSeek
            
            # Tomar decisión final
            final_decision = self._make_final_decision(
                {"signal": consensus_signal, "confidence": confidence},
                risk_assessment
            )
            
            # Añadir información de DeepSeek si está disponible
            if deepseek_analysis and 'error' not in deepseek_analysis:
                final_decision['deepseek'] = {
                    'analysis': deepseek_analysis.get('market_analysis', {}),
                    'sentiment': deepseek_analysis.get('sentiment_analysis', {}),
                    'recommendations': deepseek_analysis.get('combined_recommendations', [])
                }
            
            # Registrar para análisis posterior
            self._update_metrics(final_decision)
            
            # Guardar historial de predicciones
            self.agent_predictions_history.append({
                'timestamp': datetime.now().timestamp(),
                'symbol': symbol,
                'predictions': agent_predictions,
                'deepseek_enhanced': deepseek_analysis is not None
            })
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Error al generar señal para {symbol}: {str(e)}")
            import traceback
            logger.debug(f"Detalle del error: {traceback.format_exc()}")
            return {'signal': 'ERROR', 'reason': str(e)}
    
    async def _calculate_indicators(self, symbol: str, ohlcv: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calcular todos los indicadores técnicos necesarios.
        
        Args:
            symbol: Símbolo de trading
            ohlcv: Datos OHLCV
            
        Returns:
            Diccionario con todos los indicadores calculados
        """
        indicators = {}
        
        try:
            # Convertir OHLCV a formato adecuado
            high = np.array(ohlcv.get('high', []), dtype=float)
            low = np.array(ohlcv.get('low', []), dtype=float)
            close = np.array(ohlcv.get('close', []), dtype=float)
            open_prices = np.array(ohlcv.get('open', []), dtype=float)
            volume = np.array(ohlcv.get('volume', []), dtype=float)
            
            # Calcular Ichimoku Cloud
            indicators['ichimoku'] = calculate_ichimoku_cloud(
                high=high,
                low=low,
                close=close,
                tenkan_period=9,
                kijun_period=26,
                senkou_span_b_period=52,
                displacement=26
            )
            
            # Calcular Bollinger Bands dinámicas
            indicators['bollinger'] = calculate_dynamic_bollinger_bands(
                close=close,
                window=20,
                num_std_dev=2.0
            )
            
            # Calcular DMI (ADX)
            indicators['dmi'] = calculate_directional_movement_index(
                high=high,
                low=low,
                close=close,
                period=14
            )
            
            # Calcular VWAP
            indicators['vwap'] = calculate_vwap(
                high=high,
                low=low,
                close=close,
                volume=volume
            )
            
            # Calcular perfil de volumen
            indicators['volume_profile'] = calculate_volume_profile(
                close=close,
                volume=volume,
                num_bins=10
            )
            
            # Calcular Market Profile (simulado)
            indicators['market_profile'] = calculate_market_profile(
                high=high,
                low=low,
                close=close,
                num_bins=10
            )
            
            # Calcular indicadores básicos
            indicators['rsi'] = self._calculate_rsi(close, 14)
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self._calculate_macd(close)
            indicators['sma_short'] = self._calculate_sma(close, 10)
            indicators['sma_medium'] = self._calculate_sma(close, 20)
            indicators['sma_long'] = self._calculate_sma(close, 50)
            indicators['ema_short'] = self._calculate_ema(close, 10)
            indicators['ema_medium'] = self._calculate_ema(close, 20)
            indicators['ema_long'] = self._calculate_ema(close, 50)
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Error al calcular indicadores para {symbol}: {str(e)}")
            return {}
    
    def _calculate_rsi(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Calcular RSI (Relative Strength Index)."""
        deltas = np.diff(prices)
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum()/window
        down = -seed[seed < 0].sum()/window
        rs = up/down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:window] = 100. - 100./(1. + rs)
        
        for i in range(window, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window
            rs = up/down if down != 0 else 0
            rsi[i] = 100. - 100./(1. + rs)
            
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcular MACD (Moving Average Convergence Divergence)."""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calcular SMA (Simple Moving Average)."""
        return np.array([np.mean(prices[max(0, i-window+1):i+1]) if i >= window-1 else np.nan for i in range(len(prices))])
    
    def _calculate_ema(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calcular EMA (Exponential Moving Average)."""
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        multiplier = 2 / (window + 1)
        
        for i in range(1, len(prices)):
            ema[i] = prices[i] * multiplier + ema[i-1] * (1 - multiplier)
            
        return ema
    
    def _analyze_technical_indicators(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizar indicadores técnicos para determinar señales.
        
        Args:
            indicators: Diccionario con indicadores calculados
            
        Returns:
            Diccionario con resultados del análisis
        """
        analysis = {
            'trend': {'bullish': 0, 'bearish': 0, 'neutral': 0},
            'momentum': {'bullish': 0, 'bearish': 0, 'neutral': 0},
            'volatility': {'high': 0, 'medium': 0, 'low': 0},
            'strength': {'strong': 0, 'medium': 0, 'weak': 0},
            'signals': {
                'buy': 0,
                'sell': 0,
                'hold': 0
            }
        }
        
        # Verificar si el diccionario está vacío
        if not indicators or not isinstance(indicators, dict):
            logger.warning("Indicadores no disponibles o con formato inválido")
            analysis['dominant_signal'] = 'hold'
            return analysis
        
        try:
            # Función auxiliar para obtener valores de indicadores de forma segura
            def safe_get_value(indicator_dict, key, default=0):
                """Obtener valor de indicador de forma segura"""
                if key not in indicator_dict:
                    return default
                
                value = indicator_dict[key]
                if not isinstance(value, (list, np.ndarray)) or len(value) == 0:
                    return default
                
                try:
                    return value[-1]  # Último valor
                except (IndexError, TypeError):
                    return default
            
            # Analizar tendencia con EMAs
            have_emas = ('ema_short' in indicators and 
                        'ema_medium' in indicators and 
                        'ema_long' in indicators)
            
            if have_emas:
                try:
                    ema_short = safe_get_value(indicators, 'ema_short')
                    ema_medium = safe_get_value(indicators, 'ema_medium')
                    ema_long = safe_get_value(indicators, 'ema_long')
                    
                    if ema_short > ema_medium > ema_long:
                        analysis['trend']['bullish'] += 2
                        analysis['signals']['buy'] += 1
                    elif ema_short < ema_medium < ema_long:
                        analysis['trend']['bearish'] += 2
                        analysis['signals']['sell'] += 1
                    else:
                        analysis['trend']['neutral'] += 1
                        analysis['signals']['hold'] += 0.5
                except (TypeError, ValueError):
                    analysis['trend']['neutral'] += 1
            else:
                analysis['trend']['neutral'] += 1
            
            # Analizar MACD
            have_macd = ('macd' in indicators and 'macd_signal' in indicators)
            if have_macd:
                try:
                    macd = safe_get_value(indicators, 'macd')
                    macd_signal = safe_get_value(indicators, 'macd_signal')
                    macd_hist = safe_get_value(indicators, 'macd_hist')
                    
                    if macd > macd_signal and macd_hist > 0:
                        analysis['momentum']['bullish'] += 1.5
                        analysis['signals']['buy'] += 1
                    elif macd < macd_signal and macd_hist < 0:
                        analysis['momentum']['bearish'] += 1.5
                        analysis['signals']['sell'] += 1
                    else:
                        analysis['momentum']['neutral'] += 1
                except (TypeError, ValueError):
                    analysis['momentum']['neutral'] += 1
            else:
                analysis['momentum']['neutral'] += 1
            
            # Analizar RSI
            if 'rsi' in indicators:
                try:
                    rsi = safe_get_value(indicators, 'rsi')
                    
                    if rsi < 30:
                        analysis['momentum']['bullish'] += 1.5  # Sobrevendido
                        analysis['signals']['buy'] += 1.5
                    elif rsi > 70:
                        analysis['momentum']['bearish'] += 1.5  # Sobrecomprado
                        analysis['signals']['sell'] += 1.5
                    else:
                        analysis['momentum']['neutral'] += 1
                    
                    # Fuerza del RSI
                    if 30 <= rsi <= 70:
                        if 45 <= rsi <= 55:
                            analysis['strength']['medium'] += 1
                        elif rsi < 45:
                            analysis['strength']['weak'] += 1
                        else:
                            analysis['strength']['strong'] += 1
                    elif rsi < 30:
                        analysis['strength']['weak'] += 1.5
                    else:  # rsi > 70
                        analysis['strength']['strong'] += 1.5
                except (TypeError, ValueError):
                    analysis['momentum']['neutral'] += 1
                    analysis['strength']['medium'] += 1
            
            # Analizar Bollinger Bands
            if 'bollinger' in indicators and isinstance(indicators['bollinger'], dict):
                try:
                    bb = indicators['bollinger']
                    price = safe_get_value(bb, 'close')
                    upper = safe_get_value(bb, 'upper')
                    lower = safe_get_value(bb, 'lower')
                    middle = safe_get_value(bb, 'middle')
                    
                    # Medir volatilidad
                    bandwidth = (upper - lower) / middle if middle > 0 else 0
                    if bandwidth > 0.1:  # Alto
                        analysis['volatility']['high'] += 2
                    elif bandwidth > 0.05:  # Medio
                        analysis['volatility']['medium'] += 2
                    else:  # Bajo
                        analysis['volatility']['low'] += 2
                    
                    # Señales de Bollinger
                    if price <= lower:
                        analysis['signals']['buy'] += 1.5  # Posible rebote
                    elif price >= upper:
                        analysis['signals']['sell'] += 1.5  # Posible caída
                    elif price > middle:
                        analysis['signals']['hold'] += 0.5
                        analysis['trend']['bullish'] += 0.5
                    elif price < middle:
                        analysis['signals']['hold'] += 0.5
                        analysis['trend']['bearish'] += 0.5
                except (TypeError, ValueError, ZeroDivisionError):
                    analysis['volatility']['medium'] += 1
            else:
                analysis['volatility']['medium'] += 1
            
            # Ichimoku Cloud
            if 'ichimoku' in indicators and isinstance(indicators['ichimoku'], dict):
                try:
                    cloud = indicators['ichimoku']
                    tenkan = safe_get_value(cloud, 'tenkan')
                    kijun = safe_get_value(cloud, 'kijun')
                    senkou_a = safe_get_value(cloud, 'senkou_span_a')
                    senkou_b = safe_get_value(cloud, 'senkou_span_b')
                    price = safe_get_value(cloud, 'close')
                    
                    # Tendencia
                    if price > senkou_a and price > senkou_b:  # Precio sobre la nube
                        analysis['trend']['bullish'] += 2
                        analysis['signals']['buy'] += 1
                    elif price < senkou_a and price < senkou_b:  # Precio bajo la nube
                        analysis['trend']['bearish'] += 2
                        analysis['signals']['sell'] += 1
                    
                    # Señal de cruce
                    if tenkan > kijun:  # Cruce alcista
                        analysis['signals']['buy'] += 1
                        analysis['momentum']['bullish'] += 1
                    elif tenkan < kijun:  # Cruce bajista
                        analysis['signals']['sell'] += 1
                        analysis['momentum']['bearish'] += 1
                except (TypeError, ValueError):
                    pass  # No ajustamos nada
            
            # DMI (ADX)
            if 'dmi' in indicators and isinstance(indicators['dmi'], dict):
                try:
                    dmi = indicators['dmi']
                    adx = safe_get_value(dmi, 'adx')
                    di_plus = safe_get_value(dmi, 'di_plus')
                    di_minus = safe_get_value(dmi, 'di_minus')
                    
                    # Fuerza de la tendencia
                    if adx > 25:
                        analysis['strength']['strong'] += 1
                    elif adx > 20:
                        analysis['strength']['medium'] += 1
                    else:
                        analysis['strength']['weak'] += 1
                    
                    # Dirección
                    if di_plus > di_minus:
                        analysis['trend']['bullish'] += 1
                        analysis['signals']['buy'] += adx / 50  # Ponderado por fuerza
                    else:
                        analysis['trend']['bearish'] += 1
                        analysis['signals']['sell'] += adx / 50  # Ponderado por fuerza
                except (TypeError, ValueError, ZeroDivisionError):
                    analysis['strength']['medium'] += 1
            
            # Determinar resultados agregados
            for key in ['trend', 'momentum', 'volatility', 'strength']:
                try:
                    max_val = max(analysis[key].values())
                    if max_val > 0:
                        max_keys = [k for k, v in analysis[key].items() if v == max_val]
                        analysis[key]['result'] = max_keys[0]
                    else:
                        analysis[key]['result'] = list(analysis[key].keys())[0]
                except (ValueError, TypeError, IndexError):
                    analysis[key]['result'] = list(analysis[key].keys())[0]
            
            # Normalizar señales
            try:
                total_signals = sum(analysis['signals'].values())
                if total_signals > 0:
                    for signal in analysis['signals']:
                        analysis['signals'][signal] /= total_signals
                    
                    # Determinar señal más fuerte
                    max_signal = max(analysis['signals'], key=analysis['signals'].get)
                    analysis['dominant_signal'] = max_signal
                else:
                    analysis['dominant_signal'] = 'hold'
            except (ValueError, TypeError, ZeroDivisionError):
                analysis['dominant_signal'] = 'hold'
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error en análisis de indicadores: {str(e)}")
            analysis['dominant_signal'] = 'hold'
            return analysis
    
    async def _get_real_rl_predictions(self, symbol: str, ohlcv: Dict[str, List[float]], 
                                 indicators: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Obtener predicciones de los modelos RL reales.
        
        Args:
            symbol: Símbolo de trading
            ohlcv: Datos OHLCV
            indicators: Indicadores técnicos
            
        Returns:
            Predicciones de cada agente
        """
        agent_predictions = {}
        
        try:
            if not RL_LIBRARIES_AVAILABLE or symbol not in self.models:
                # Usar simulación si no hay acceso a modelos reales
                return self._simulate_agent_predictions(self._analyze_technical_indicators(indicators), symbol)
            
            # Preparar entorno para predicciones
            env = self.envs[symbol]
            
            # Actualizar datos del entorno
            env.data = {
                'close': np.array(ohlcv['close']),
                'high': np.array(ohlcv['high']),
                'low': np.array(ohlcv['low']),
                'open': np.array(ohlcv['open']),
                'volume': np.array(ohlcv['volume']),
                'rsi': indicators.get('rsi', np.zeros_like(ohlcv['close'])),
                'macd': indicators.get('macd', np.zeros_like(ohlcv['close'])),
                'macd_signal': indicators.get('macd_signal', np.zeros_like(ohlcv['close'])),
                'bb_upper': indicators.get('bollinger', {}).get('upper', np.zeros_like(ohlcv['close'])),
                'bb_lower': indicators.get('bollinger', {}).get('lower', np.zeros_like(ohlcv['close']))
            }
            
            # Reset del entorno para la nueva predicción
            observation, _ = env.reset()
            
            # Obtener predicción de cada modelo
            for model_name, model in self.models[symbol].items():
                # Tomar acción según el modelo
                action, _ = model.predict(observation, deterministic=True)
                
                # Traducir acción a señal
                if action == 0:  # Hold
                    signal = "hold"
                elif action == 1:  # Buy
                    signal = "buy"
                else:  # Sell
                    signal = "sell"
                
                # Calcular confianza basada en datos históricos
                history_confidence = self.models.get(model_name, {}).get('success_rate', 0.5)
                
                # Agregar predicción
                agent_predictions[model_name] = {
                    "signal": signal,
                    "confidence": history_confidence,
                    "timestamp": datetime.now().timestamp()
                }
            
            return agent_predictions
            
        except Exception as e:
            logger.warning(f"Error en predicciones RL para {symbol}: {str(e)}. Usando simulación.")
            return self._simulate_agent_predictions(self._analyze_technical_indicators(indicators), symbol)
    
    def _simulate_agent_predictions(self, analysis_results: Dict[str, Any], 
                                    symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Simular predicciones de diferentes agentes RL.
        
        Args:
            analysis_results: Resultados del análisis técnico
            symbol: Símbolo de trading
            
        Returns:
            Diccionario con predicciones de cada agente
        """
        agent_predictions = {}
        
        try:
            # Obtener señales dominantes
            dominant_signal = analysis_results.get('dominant_signal', 'hold')
            signals = analysis_results.get('signals', {})
            trend = analysis_results.get('trend', {}).get('result', 'neutral')
            momentum = analysis_results.get('momentum', {}).get('result', 'neutral')
            
            # Convertir señales a probabilidades
            probabilities = {
                'buy': signals.get('buy', 0.33),
                'sell': signals.get('sell', 0.33),
                'hold': signals.get('hold', 0.34)
            }
            
            # Normalizar probabilidades
            total_prob = sum(probabilities.values())
            if total_prob > 0:
                for signal in probabilities:
                    probabilities[signal] /= total_prob
            
            # DQN - Sensible a tendencias
            dqn_probs = probabilities.copy()
            if trend == 'bullish':
                dqn_probs['buy'] *= 1.3
                dqn_probs['sell'] *= 0.7
            elif trend == 'bearish':
                dqn_probs['buy'] *= 0.7
                dqn_probs['sell'] *= 1.3
                
            # Normalizar nuevamente
            dqn_total = sum(dqn_probs.values())
            for signal in dqn_probs:
                dqn_probs[signal] /= dqn_total
                
            # PPO - Sensible a momentum
            ppo_probs = probabilities.copy()
            if momentum == 'bullish':
                ppo_probs['buy'] *= 1.5
                ppo_probs['sell'] *= 0.5
            elif momentum == 'bearish':
                ppo_probs['buy'] *= 0.5
                ppo_probs['sell'] *= 1.5
                
            # Normalizar
            ppo_total = sum(ppo_probs.values())
            for signal in ppo_probs:
                ppo_probs[signal] /= ppo_total
                
            # SAC - Más conservador, prefiere hold en incertidumbre
            sac_probs = probabilities.copy()
            sac_probs['hold'] *= 1.2
            if max(probabilities.values()) - min(probabilities.values()) < 0.2:
                # Mucha incertidumbre, aumentar hold
                sac_probs['hold'] *= 1.5
                sac_probs['buy'] *= 0.8
                sac_probs['sell'] *= 0.8
                
            # Normalizar
            sac_total = sum(sac_probs.values())
            for signal in sac_probs:
                sac_probs[signal] /= sac_total
            
            # Asegurar que valores específicos de éxito estén presentes
            model_configs = {
                "dqn": {"weight": 0.3, "success_rate": 0.6},
                "ppo": {"weight": 0.4, "success_rate": 0.65},
                "sac": {"weight": 0.3, "success_rate": 0.55}
            }
            
            # Simular toma de decisiones de cada agente
            timestamp = datetime.now().timestamp()
            
            # DQN - Favorece tendencia de largo plazo
            dqn_signal = max(dqn_probs, key=dqn_probs.get)
            dqn_confidence = dqn_probs[dqn_signal] * model_configs["dqn"]["success_rate"]
            agent_predictions["dqn"] = {
                "signal": dqn_signal,
                "confidence": dqn_confidence,
                "raw_confidence": dqn_probs[dqn_signal],
                "reasoning": f"Basado en tendencia {trend}",
                "timestamp": timestamp
            }
            
            # PPO - Favorece momentum
            ppo_signal = max(ppo_probs, key=ppo_probs.get)
            ppo_confidence = ppo_probs[ppo_signal] * model_configs["ppo"]["success_rate"]
            agent_predictions["ppo"] = {
                "signal": ppo_signal,
                "confidence": ppo_confidence,
                "raw_confidence": ppo_probs[ppo_signal],
                "reasoning": f"Basado en momentum {momentum}",
                "timestamp": timestamp
            }
            
            # SAC - Más conservador
            sac_signal = max(sac_probs, key=sac_probs.get)
            sac_confidence = sac_probs[sac_signal] * model_configs["sac"]["success_rate"]
            agent_predictions["sac"] = {
                "signal": sac_signal,
                "confidence": sac_confidence,
                "raw_confidence": sac_probs[sac_signal],
                "reasoning": "Enfoque conservador",
                "timestamp": timestamp
            }
            
            return agent_predictions
            
        except Exception as e:
            logger.warning(f"Error al simular predicciones de agentes: {str(e)}")
            
            # Retornar predicciones por defecto en caso de error
            timestamp = datetime.now().timestamp()
            return {
                "dqn": {"signal": "hold", "confidence": 0.5, "timestamp": timestamp},
                "ppo": {"signal": "hold", "confidence": 0.5, "timestamp": timestamp},
                "sac": {"signal": "hold", "confidence": 0.5, "timestamp": timestamp}
            }
    
    def _get_ensemble_consensus(self, agent_predictions: Dict[str, Dict[str, Any]]) -> Tuple[str, float]:
        """
        Obtener consenso del ensemble de agentes RL.
        
        Args:
            agent_predictions: Predicciones de cada agente
            
        Returns:
            Tupla de (señal consenso, nivel de confianza)
        """
        if not agent_predictions:
            return "hold", 0.5
        
        try:
            # Contar votos ponderados por confianza y peso del modelo
            weighted_votes = {
                "buy": 0.0,
                "sell": 0.0,
                "hold": 0.0
            }
            
            model_weights = {
                "dqn": 0.3,
                "ppo": 0.4,
                "sac": 0.3
            }
            
            # Calcular votos ponderados
            for model, prediction in agent_predictions.items():
                signal = prediction["signal"]
                confidence = prediction["confidence"]
                weight = model_weights.get(model, 0.33)
                
                weighted_votes[signal] += confidence * weight
            
            # Normalizar votos
            total_weighted = sum(weighted_votes.values())
            if total_weighted > 0:
                for signal in weighted_votes:
                    weighted_votes[signal] /= total_weighted
            
            # Obtener señal con mayor voto ponderado
            max_signal = max(weighted_votes, key=weighted_votes.get)
            max_confidence = weighted_votes[max_signal]
            
            # Verificar si hay una señal dominante (supera el umbral de votación)
            if max_confidence >= self.voting_threshold:
                return max_signal, max_confidence
            else:
                # No hay consenso fuerte, ser conservador
                return "hold", max(weighted_votes["hold"], 0.5)
                
        except Exception as e:
            logger.warning(f"Error al obtener consenso de ensemble: {str(e)}")
            return "hold", 0.5
    
    def _assess_risk(self, symbol: str, signal: str, confidence: float, 
                     ohlcv: Dict[str, List[float]], indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluar riesgo de la operación.
        
        Args:
            symbol: Símbolo de trading
            signal: Señal de trading
            confidence: Nivel de confianza
            ohlcv: Datos OHLCV
            indicators: Indicadores técnicos
            
        Returns:
            Evaluación de riesgo con niveles recomendados
        """
        try:
            # Extraer datos recientes
            close_prices = ohlcv.get('close', [])
            if not close_prices:
                return {"risk_level": "high", "take_profit": None, "stop_loss": None}
            
            current_price = close_prices[-1]
            
            # Volatilidad reciente (20 periodos)
            recent_prices = close_prices[-20:] if len(close_prices) >= 20 else close_prices
            volatility = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0.01
            
            # Determinar nivel de riesgo base
            base_risk_level = "medium"
            if volatility > 0.05:  # 5% de volatilidad es alta
                base_risk_level = "high"
            elif volatility < 0.01:  # 1% de volatilidad es baja
                base_risk_level = "low"
            
            # Ajustar por confianza
            if confidence < 0.6:
                # Baja confianza aumenta riesgo
                if base_risk_level == "low":
                    risk_level = "medium"
                else:
                    risk_level = "high"
            elif confidence > 0.8:
                # Alta confianza reduce riesgo
                if base_risk_level == "high":
                    risk_level = "medium"
                else:
                    risk_level = "low"
            else:
                risk_level = base_risk_level
            
            # Establecer niveles de take profit y stop loss basados en volatilidad y tendencia
            if signal == "buy":
                # Stop loss: 1-3 veces la volatilidad diaria
                stop_loss_pct = min(max(volatility * 2, 0.01), 0.05)  # Entre 1% y 5%
                take_profit_pct = stop_loss_pct * 1.5  # Ratio ganancia/pérdida 1.5:1
                
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            elif signal == "sell":
                stop_loss_pct = min(max(volatility * 2, 0.01), 0.05)
                take_profit_pct = stop_loss_pct * 1.5
                
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
            else:  # hold
                stop_loss = None
                take_profit = None
            
            # Ajustar por soporte/resistencia si están disponibles
            if 'bollinger' in indicators:
                bb = indicators['bollinger']
                upper = bb.get('upper', [0])[-1]
                lower = bb.get('lower', [0])[-1]
                
                if signal == "buy" and take_profit > upper:
                    # Ajustar take profit a nivel de resistencia
                    take_profit = min(take_profit, upper)
                    
                elif signal == "sell" and take_profit < lower:
                    # Ajustar take profit a nivel de soporte
                    take_profit = max(take_profit, lower)
            
            # Redondear valores
            if stop_loss:
                stop_loss = round(stop_loss, 8)
            if take_profit:
                take_profit = round(take_profit, 8)
                
            return {
                "risk_level": risk_level,
                "volatility": volatility,
                "confidence_impact": confidence,
                "take_profit": take_profit,
                "stop_loss": stop_loss
            }
            
        except Exception as e:
            logger.warning(f"Error al evaluar riesgo para {symbol}: {str(e)}")
            return {"risk_level": "high", "take_profit": None, "stop_loss": None}
    
    def _adjust_risk_with_deepseek(self, 
                                  risk_assessment: Dict[str, Any],
                                  enhanced_stop_loss: Optional[float] = None,
                                  enhanced_take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Ajustar evaluación de riesgo con datos de DeepSeek.
        
        Args:
            risk_assessment: Evaluación de riesgo original
            enhanced_stop_loss: Nivel de stop loss mejorado
            enhanced_take_profit: Nivel de take profit mejorado
            
        Returns:
            Evaluación de riesgo ajustada
        """
        adjusted_risk = risk_assessment.copy()
        
        if enhanced_stop_loss is not None:
            adjusted_risk["stop_loss"] = enhanced_stop_loss
            adjusted_risk["deepseek_adjusted_stop"] = True
        
        if enhanced_take_profit is not None:
            adjusted_risk["take_profit"] = enhanced_take_profit
            adjusted_risk["deepseek_adjusted_tp"] = True
        
        # Ajustar nivel de riesgo si DeepSeek mejoró ambos niveles
        if enhanced_stop_loss is not None and enhanced_take_profit is not None:
            # Calcular ratio riesgo/recompensa
            current_price = (enhanced_take_profit + enhanced_stop_loss) / 2  # Estimación
            
            if enhanced_take_profit > enhanced_stop_loss:  # Posición larga
                risk = (current_price - enhanced_stop_loss) / current_price
                reward = (enhanced_take_profit - current_price) / current_price
            else:  # Posición corta
                risk = (enhanced_stop_loss - current_price) / current_price
                reward = (current_price - enhanced_take_profit) / current_price
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Ajustar nivel de riesgo
            if risk_reward_ratio >= 2.0:
                adjusted_risk["risk_level"] = "low"
            elif risk_reward_ratio >= 1.5:
                adjusted_risk["risk_level"] = "medium"
            else:
                adjusted_risk["risk_level"] = "high"
                
            adjusted_risk["risk_reward_ratio"] = risk_reward_ratio
            
        return adjusted_risk
    
    def _make_final_decision(self, prediction: Dict[str, Any], 
                            risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tomar decisión final combinando predicción y riesgo.
        
        Args:
            prediction: Predicción de la señal
            risk_assessment: Evaluación de riesgo
            
        Returns:
            Decisión final con todos los metadatos
        """
        signal = prediction.get("signal", "hold")
        confidence = prediction.get("confidence", 0.5)
        risk_level = risk_assessment.get("risk_level", "medium")
        
        # Por defecto no hacer nada si el riesgo es alto y la confianza baja
        if risk_level == "high" and confidence < 0.7 and signal != "hold":
            final_signal = "hold"
            reason = f"Señal original {signal} rechazada: riesgo alto ({risk_level}) y confianza baja ({confidence:.2f})"
        else:
            final_signal = signal
            reason = f"Señal {signal} aceptada con confianza {confidence:.2f} y nivel de riesgo {risk_level}"
        
        # Ajustar tamaño de la posición según confianza y riesgo
        position_size = 0.0
        if final_signal in ["buy", "sell"]:
            # Base: usar el riesgo por operación configurado
            base_size = self.risk_per_trade
            
            # Ajustar por confianza (0.5-1.0)
            confidence_factor = (confidence - 0.5) * 2 if confidence > 0.5 else 0
            
            # Ajustar por riesgo
            risk_factor = 1.0
            if risk_level == "high":
                risk_factor = 0.5  # Reducir tamaño en riesgo alto
            elif risk_level == "low":
                risk_factor = 1.5  # Aumentar tamaño en riesgo bajo
            
            # Calcular tamaño final
            position_size = base_size * (1 + confidence_factor) * risk_factor
            
            # Limitar a máximo 10% del capital
            position_size = min(position_size, 0.1)
        
        final_decision = {
            "signal": final_signal,
            "original_signal": signal,
            "confidence": confidence,
            "risk_level": risk_level,
            "position_size": position_size,
            "reason": reason,
            "timestamp": datetime.now().timestamp()
        }
        
        # Añadir niveles de stop loss y take profit si existen
        if risk_assessment.get("stop_loss") is not None:
            final_decision["stop_loss"] = risk_assessment["stop_loss"]
        
        if risk_assessment.get("take_profit") is not None:
            final_decision["take_profit"] = risk_assessment["take_profit"]
            
        # Incluir información adicional
        final_decision["risk_assessment"] = risk_assessment
        
        return final_decision
    
    def _update_metrics(self, decision: Dict[str, Any]) -> None:
        """
        Actualizar métricas de rendimiento.
        
        Args:
            decision: Decisión tomada
        """
        # Solo actualizar métricas para operaciones ejecutadas
        if decision["signal"] in ["buy", "sell"]:
            self.performance_metrics["total_trades"] += 1
        
        # Otras métricas se actualizarían cuando se cierren operaciones
        
    def _create_sample_data(self, symbol: str) -> Dict[str, np.ndarray]:
        """
        Crear datos de ejemplo para inicializar modelos RL.
        
        Args:
            symbol: Símbolo de trading
            
        Returns:
            Datos OHLCV simulados
        """
        # Generar serie simulada
        n_points = 200
        base = 100 + np.random.normal(0, 1, n_points).cumsum()
        
        # OHLCV
        close = np.abs(base)
        high = close * (1 + np.random.uniform(0, 0.03, n_points))
        low = close * (1 - np.random.uniform(0, 0.03, n_points))
        open_prices = low + np.random.uniform(0, 1, n_points) * (high - low)
        volume = np.random.uniform(50, 500, n_points) * (1 + 0.1 * np.random.normal(0, 1, n_points))
        
        # Indicadores adicionales
        rsi = np.random.uniform(20, 80, n_points)
        macd = np.random.normal(0, 1, n_points).cumsum()
        macd_signal = np.convolve(macd, np.ones(5)/5, mode='same')
        macd_hist = macd - macd_signal
        bb_upper = close * (1 + 0.02)
        bb_lower = close * (1 - 0.02)
        
        return {
            'close': close,
            'high': high,
            'low': low,
            'open': open_prices,
            'volume': volume,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower
        }
        
    async def optimize_allocation(self) -> Dict[str, Any]:
        """
        Optimizar la asignación de capital entre diferentes símbolos.
        
        Returns:
            Asignación optimizada
        """
        try:
            allocation = {}
            
            # Obtener rendimiento por símbolo
            performance = self.symbol_performance
            
            # Método simple: asignar capital proporcional al éxito histórico
            total_success = sum(data["success_rate"] for data in performance.values())
            
            if total_success > 0:
                for symbol, data in performance.items():
                    allocation[symbol] = data["success_rate"] / total_success
            else:
                # División equitativa si no hay datos
                equal_share = 1.0 / len(self.symbols)
                allocation = {symbol: equal_share for symbol in self.symbols}
            
            # Simular optimización
            await asyncio.sleep(0.2)
            
            return {
                "allocation": allocation,
                "capital": self.initial_capital,
                "timestamp": datetime.now().timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error al optimizar asignación: {str(e)}")
            return {"error": str(e)}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de rendimiento de la estrategia.
        
        Returns:
            Estadísticas de rendimiento
        """
        return {
            **self.performance_metrics,
            "timestamp": self._get_current_timestamp()
        }
    
    def _get_current_timestamp(self) -> float:
        """Obtener timestamp actual."""
        return datetime.now().timestamp()