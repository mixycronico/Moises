"""
Entornos de Aprendizaje por Refuerzo para trading de criptomonedas.

Este módulo implementa entornos compatibles con OpenAI Gym para el
aprendizaje por refuerzo en contextos de trading. Los entornos proporcionan
interfaces estandarizadas para entrenar agentes RL.
"""

import numpy as np
import pandas as pd
import logging
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from io import BytesIO
import base64

class TradingEnvironment(gym.Env):
    """
    Entorno base para trading con gimnasio (gymnasium) RL.
    
    Este entorno soporta acciones discretas (comprar, vender, mantener)
    y utiliza datos históricos de precios para simular un entorno de trading.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array', 'none'], 'render_fps': 4}
    
    def __init__(self, 
                 df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 commission: float = 0.001,
                 reward_scaling: float = 1.0,
                 max_position: float = 1.0,
                 window_size: int = 20,
                 features: Optional[List[str]] = None,
                 render_mode: Optional[str] = None,
                 log_returns: bool = True,
                 reward_function: Optional[Callable] = None):
        """
        Inicializar entorno de trading.
        
        Args:
            df: DataFrame con datos históricos (debe tener columnas 'timestamp', 'open', 'high', 'low', 'close', 'volume')
            initial_balance: Saldo inicial en unidades de la moneda base (ej. USDT)
            commission: Comisión de trading en porcentaje (0.001 = 0.1%)
            reward_scaling: Factor de escala para las recompensas
            max_position: Tamaño máximo de posición como fracción del balance (1.0 = 100%)
            window_size: Tamaño de la ventana de observación
            features: Lista de características adicionales a incluir en la observación
            render_mode: Modo de renderizado ('human', 'rgb_array', 'none')
            log_returns: Si es True, usa log-returns en lugar de returns simples
            reward_function: Función personalizada para calcular recompensas
        """
        self.logger = logging.getLogger(__name__)
        
        # Validar DataFrame
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"DataFrame debe contener columnas {required_columns}. Faltan: {missing}")
        
        # Datos y parámetros
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_scaling = reward_scaling
        self.max_position = max_position
        self.window_size = window_size
        self.render_mode = render_mode
        self.log_returns = log_returns
        self.reward_function = reward_function
        
        # Características a utilizar
        self.price_features = ['open', 'high', 'low', 'close', 'volume']
        self.features = features if features is not None else []
        
        # Si las características especificadas no están en el DataFrame, iniciarlas
        for feature in self.features:
            if feature not in self.df.columns:
                self.df[feature] = 0.0
        
        # Calcular retornos
        self.df['returns'] = self.df['close'].pct_change()
        if self.log_returns:
            self.df['returns'] = np.log1p(self.df['returns'])
        
        # Convertir NaN a 0
        self.df.fillna(0, inplace=True)
        
        # Espacio de acciones: 0 (mantener), 1 (comprar), 2 (vender)
        self.action_space = spaces.Discrete(3)
        
        # Espacio de observación: [ventana de precios, indicadores, posición, balance]
        num_price_features = len(self.price_features)
        num_features = len(self.features)
        
        # Observación incluye:
        # - Ventana histórica para cada característica de precio
        # - Ventana histórica para cada característica adicional
        # - Estado actual del agente (posición, balance, etc.)
        obs_shape = (self.window_size, num_price_features + num_features)
        
        # Rango bajo/alto estimado para valores de observación
        if np.any(self.df['close'] > 0):
            price_scale = np.max(self.df['close']) * 10
        else:
            price_scale = 100000.0
            
        low_values = np.array([-price_scale] * (num_price_features + num_features))
        high_values = np.array([price_scale] * (num_price_features + num_features))
        
        self.observation_space = spaces.Dict({
            # Ventana histórica
            'market_history': spaces.Box(
                low=np.tile(low_values, (self.window_size, 1)),
                high=np.tile(high_values, (self.window_size, 1)),
                shape=obs_shape,
                dtype=np.float32
            ),
            # Estado del agente
            'account_state': spaces.Box(
                low=np.array([-1.0, 0.0, -1.0]),  # posición, balance, PnL
                high=np.array([1.0, price_scale, 1.0]),
                shape=(3,),
                dtype=np.float32
            )
        })
        
        # Variables de estado
        self.current_step = None
        self.current_price = None
        self.current_position = None
        self.current_balance = None
        self.trades = None
        self.account_history = None
        
        # Para renderizado
        self.fig = None
        self.axes = None
        
        self.logger.info("Entorno de trading inicializado con %d pasos y ventana de %d", 
                        len(self.df), self.window_size)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Obtener observación actual del entorno.
        
        Returns:
            Diccionario con observación completa
        """
        # Obtener ventana de datos históricos
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # Si no hay suficientes datos históricos, rellenar con primeros valores
        if start_idx < 0:
            pad_size = abs(start_idx)
            start_idx = 0
        else:
            pad_size = 0
        
        # Matriz de histórico
        history_window = []
        # Primero características de precio
        for feature in self.price_features:
            # Extraer ventana
            feature_window = self.df[feature].iloc[start_idx:end_idx].values
            
            # Normalizar para características conocidas
            if feature == 'close' or feature == 'open' or feature == 'high' or feature == 'low':
                # Normalizar precio sobre precio de cierre actual
                if self.current_price > 0:
                    feature_window = feature_window / self.current_price - 1.0
                else:
                    feature_window = feature_window * 0.0
            elif feature == 'volume':
                # Normalizar volumen sobre volumen máximo
                vol_max = np.max(self.df['volume']) if np.max(self.df['volume']) > 0 else 1.0
                feature_window = feature_window / vol_max
            
            # Rellenar si es necesario
            if pad_size > 0:
                feature_window = np.concatenate([np.repeat(feature_window[0], pad_size), feature_window])
            
            history_window.append(feature_window)
        
        # Luego características adicionales
        for feature in self.features:
            # Extraer ventana
            feature_window = self.df[feature].iloc[start_idx:end_idx].values
            
            # Normalizar si valores son grandes
            if np.any(np.abs(feature_window) > 10):
                # Normalizar sobre valor máximo absoluto
                max_val = np.max(np.abs(self.df[feature])) if np.max(np.abs(self.df[feature])) > 0 else 1.0
                feature_window = feature_window / max_val
            
            # Rellenar si es necesario
            if pad_size > 0:
                feature_window = np.concatenate([np.repeat(feature_window[0], pad_size), feature_window])
            
            history_window.append(feature_window)
        
        # Transponemos para tener la forma (window_size, n_features)
        market_history = np.array(history_window).T.astype(np.float32)
        
        # Estado de la cuenta
        account_state = np.array([
            self.current_position,  # Posición actual normalizada
            self.current_balance / self.initial_balance,  # Balance normalizado
            self._get_unrealized_pnl() / self.initial_balance  # PnL no realizada normalizada
        ], dtype=np.float32)
        
        return {
            'market_history': market_history,
            'account_state': account_state
        }
    
    def _get_unrealized_pnl(self) -> float:
        """
        Calcular PnL no realizada de la posición actual.
        
        Returns:
            PnL no realizada
        """
        if self.current_position == 0:
            return 0.0
        
        # Si hay posición, calcular PnL no realizada
        if self.current_position > 0:
            # Posición larga
            entry_price = self.account_history[-1]['entry_price'] if self.account_history else self.current_price
            unrealized_pnl = self.current_position * (self.current_price - entry_price)
        else:
            # Posición corta
            entry_price = self.account_history[-1]['entry_price'] if self.account_history else self.current_price
            unrealized_pnl = -self.current_position * (entry_price - self.current_price)
        
        return unrealized_pnl
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calcular recompensa basada en la acción y el movimiento del mercado.
        
        Args:
            action: Acción tomada (0=mantener, 1=comprar, 2=vender)
            
        Returns:
            Recompensa
        """
        # Si se proporciona una función de recompensa personalizada, usarla
        if self.reward_function is not None:
            return self.reward_function(
                self, action, self.current_step, self.trades, self.account_history
            ) * self.reward_scaling
        
        # Recompensa base: cambio en el valor del portafolio
        prev_portfolio_value = self.account_history[-2]['portfolio_value'] if len(self.account_history) > 1 else self.initial_balance
        current_portfolio_value = self.account_history[-1]['portfolio_value']
        
        # Calcular rendimiento del portafolio
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Escalar recompensa
        reward = portfolio_return * self.reward_scaling
        
        # Penalizar por comisiones excesivas
        if len(self.trades) > 0 and self.trades[-1]['step'] == self.current_step:
            # Operación en este paso, incluir penalización por comisión
            reward -= self.commission * self.reward_scaling
        
        return reward
    
    def _take_action(self, action: int) -> Tuple[float, Dict[str, Any]]:
        """
        Ejecutar una acción en el entorno.
        
        Args:
            action: Acción a tomar (0=mantener, 1=comprar, 2=vender)
            
        Returns:
            Tupla de (recompensa, información)
        """
        prev_portfolio_value = self._get_portfolio_value()
        
        if action == 1:  # Comprar
            if self.current_position < self.max_position:
                # Calcular cantidad a comprar
                target_position = self.max_position
                amount_to_buy = target_position - self.current_position
                
                # Verificar balance disponible
                cost = amount_to_buy * self.current_price * (1 + self.commission)
                if cost > self.current_balance:
                    # Ajustar si no hay suficiente balance
                    amount_to_buy = self.current_balance / (self.current_price * (1 + self.commission))
                
                if amount_to_buy > 0:
                    # Registrar operación
                    commission = amount_to_buy * self.current_price * self.commission
                    self.current_balance -= (amount_to_buy * self.current_price + commission)
                    
                    # Actualizar posición
                    entry_price = self.current_price
                    self.current_position += amount_to_buy
                    
                    # Registrar operación
                    self.trades.append({
                        'step': self.current_step,
                        'timestamp': self.df['timestamp'].iloc[self.current_step],
                        'type': 'buy',
                        'price': self.current_price,
                        'amount': amount_to_buy,
                        'commission': commission,
                        'balance': self.current_balance
                    })
                    
                    # Actualizar historial
                    self.account_history.append({
                        'step': self.current_step,
                        'timestamp': self.df['timestamp'].iloc[self.current_step],
                        'balance': self.current_balance,
                        'position': self.current_position,
                        'price': self.current_price,
                        'entry_price': entry_price,
                        'portfolio_value': self._get_portfolio_value()
                    })
        
        elif action == 2:  # Vender
            if self.current_position > -self.max_position:
                # Calcular cantidad a vender
                target_position = -self.max_position
                amount_to_sell = self.current_position - target_position
                
                if amount_to_sell > 0:
                    # Registrar operación
                    commission = amount_to_sell * self.current_price * self.commission
                    self.current_balance += (amount_to_sell * self.current_price - commission)
                    
                    # Actualizar posición
                    entry_price = self.current_price
                    self.current_position -= amount_to_sell
                    
                    # Registrar operación
                    self.trades.append({
                        'step': self.current_step,
                        'timestamp': self.df['timestamp'].iloc[self.current_step],
                        'type': 'sell',
                        'price': self.current_price,
                        'amount': amount_to_sell,
                        'commission': commission,
                        'balance': self.current_balance
                    })
                    
                    # Actualizar historial
                    self.account_history.append({
                        'step': self.current_step,
                        'timestamp': self.df['timestamp'].iloc[self.current_step],
                        'balance': self.current_balance,
                        'position': self.current_position,
                        'price': self.current_price,
                        'entry_price': entry_price,
                        'portfolio_value': self._get_portfolio_value()
                    })
        
        else:  # Mantener
            # Actualizar historial
            self.account_history.append({
                'step': self.current_step,
                'timestamp': self.df['timestamp'].iloc[self.current_step],
                'balance': self.current_balance,
                'position': self.current_position,
                'price': self.current_price,
                'entry_price': self.account_history[-1]['entry_price'] if self.account_history else self.current_price,
                'portfolio_value': self._get_portfolio_value()
            })
        
        # Calcular recompensa
        reward = self._calculate_reward(action)
        
        # Preparar información adicional
        info = {
            'portfolio_value': self._get_portfolio_value(),
            'portfolio_return': (self._get_portfolio_value() - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0,
            'price': self.current_price,
            'position': self.current_position,
            'balance': self.current_balance
        }
        
        return reward, info
    
    def _get_portfolio_value(self) -> float:
        """
        Calcular valor total del portafolio.
        
        Returns:
            Valor del portafolio (balance + posición)
        """
        position_value = self.current_position * self.current_price if self.current_position > 0 else 0
        
        # Para posiciones cortas, el valor no está en la posición sino en el balance
        if self.current_position < 0:
            return self.current_balance
        
        return self.current_balance + position_value
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reiniciar el entorno al estado inicial.
        
        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales
            
        Returns:
            Tupla de (observación, información)
        """
        super().reset(seed=seed)
        
        # Inicializar variables de estado
        self.current_step = self.window_size - 1
        self.current_price = self.df['close'].iloc[self.current_step]
        self.current_position = 0.0
        self.current_balance = self.initial_balance
        self.trades = []
        self.account_history = [{
            'step': self.current_step,
            'timestamp': self.df['timestamp'].iloc[self.current_step],
            'balance': self.current_balance,
            'position': self.current_position,
            'price': self.current_price,
            'entry_price': self.current_price,
            'portfolio_value': self._get_portfolio_value()
        }]
        
        # Obtener observación inicial
        observation = self._get_observation()
        
        # Información inicial
        info = {
            'portfolio_value': self._get_portfolio_value(),
            'price': self.current_price,
            'position': self.current_position,
            'balance': self.current_balance
        }
        
        if self.render_mode == 'human':
            self._render_frame()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Avanzar el entorno un paso con la acción dada.
        
        Args:
            action: Acción a ejecutar (0=mantener, 1=comprar, 2=vender)
            
        Returns:
            Tupla de (observación, recompensa, terminado, truncado, información)
        """
        # Ejecutar acción y calcular recompensa
        reward, info = self._take_action(action)
        
        # Avanzar al siguiente paso
        self.current_step += 1
        
        # Verificar si la simulación ha terminado
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        if not done:
            # Actualizar precio actual
            self.current_price = self.df['close'].iloc[self.current_step]
            
            # Actualizar observación
            observation = self._get_observation()
            
            # Actualizar información
            info.update({
                'portfolio_value': self._get_portfolio_value(),
                'price': self.current_price,
                'position': self.current_position,
                'balance': self.current_balance
            })
        else:
            # Final de la simulación, liquidar posición
            if self.current_position != 0:
                # Cerrar posición y calcular PnL final
                final_return = 0.0
                if self.current_position > 0:
                    # Vender posición larga
                    commission = self.current_position * self.current_price * self.commission
                    self.current_balance += (self.current_position * self.current_price - commission)
                    
                    # Registrar operación final
                    self.trades.append({
                        'step': self.current_step,
                        'timestamp': self.df['timestamp'].iloc[self.current_step],
                        'type': 'sell',
                        'price': self.current_price,
                        'amount': self.current_position,
                        'commission': commission,
                        'balance': self.current_balance
                    })
                    
                    self.current_position = 0.0
                
                elif self.current_position < 0:
                    # Cerrar posición corta
                    commission = abs(self.current_position) * self.current_price * self.commission
                    self.current_balance -= (abs(self.current_position) * self.current_price + commission)
                    
                    # Registrar operación final
                    self.trades.append({
                        'step': self.current_step,
                        'timestamp': self.df['timestamp'].iloc[self.current_step],
                        'type': 'buy',
                        'price': self.current_price,
                        'amount': abs(self.current_position),
                        'commission': commission,
                        'balance': self.current_balance
                    })
                    
                    self.current_position = 0.0
            
            # Observación final
            observation = self._get_observation()
            
            # Actualizar historial final
            self.account_history.append({
                'step': self.current_step,
                'timestamp': self.df['timestamp'].iloc[self.current_step],
                'balance': self.current_balance,
                'position': self.current_position,
                'price': self.current_price,
                'entry_price': self.current_price,
                'portfolio_value': self._get_portfolio_value()
            })
            
            # Información final
            info.update({
                'portfolio_value': self._get_portfolio_value(),
                'final_balance': self.current_balance,
                'return': (self._get_portfolio_value() - self.initial_balance) / self.initial_balance,
                'num_trades': len(self.trades)
            })
        
        if self.render_mode == 'human':
            self._render_frame()
        
        return observation, reward, done, truncated, info
    
    def render(self) -> Optional[Union[np.ndarray, Figure]]:
        """
        Renderizar el estado actual del entorno.
        
        Returns:
            None, array RGB o figura según el modo de renderizado
        """
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        
        return None
    
    def _render_frame(self) -> Optional[Union[np.ndarray, Figure]]:
        """
        Renderizar un frame del entorno.
        
        Returns:
            None, array RGB o figura según el modo de renderizado
        """
        if self.render_mode not in ['human', 'rgb_array']:
            return None
        
        # Crear figura y ejes si no existen
        if self.fig is None or self.axes is None:
            self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        else:
            for ax in self.axes:
                ax.clear()
        
        ax1, ax2 = self.axes
        
        # Preparar datos hasta el paso actual
        data = self.df.iloc[:self.current_step+1]
        
        # Gráfico de precio
        ax1.plot(data['timestamp'], data['close'], label='Close')
        
        # Marcar operaciones
        buys = [t for t in self.trades if t['step'] <= self.current_step and t['type'] == 'buy']
        sells = [t for t in self.trades if t['step'] <= self.current_step and t['type'] == 'sell']
        
        buy_times = [self.df['timestamp'].iloc[t['step']] for t in buys]
        buy_prices = [t['price'] for t in buys]
        
        sell_times = [self.df['timestamp'].iloc[t['step']] for t in sells]
        sell_prices = [t['price'] for t in sells]
        
        ax1.scatter(buy_times, buy_prices, marker='^', color='green', label='Buy')
        ax1.scatter(sell_times, sell_prices, marker='v', color='red', label='Sell')
        
        # Marcar posición actual
        if self.current_position != 0:
            ax1.axhline(y=self.account_history[-1]['entry_price'], color='orange', linestyle='--', alpha=0.7, label='Entry')
        
        # Formato del gráfico de precio
        ax1.set_title(f'Trading Simulation - Step {self.current_step} / {len(self.df)-1}')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de valor del portafolio
        portfolio_values = [h['portfolio_value'] for h in self.account_history if h['step'] <= self.current_step]
        portfolio_times = [self.df['timestamp'].iloc[h['step']] for h in self.account_history if h['step'] <= self.current_step]
        
        ax2.plot(portfolio_times, portfolio_values, label='Portfolio Value', color='blue')
        ax2.axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.7, label='Initial Balance')
        
        # Formato del gráfico de portafolio
        ax2.set_ylabel('Portfolio Value')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Formatear eje X para mostrar fechas correctamente
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Añadir información adicional en el título
        percent_change = ((self._get_portfolio_value() - self.initial_balance) / self.initial_balance) * 100
        info_text = (
            f"Balance: {self.current_balance:.2f} | "
            f"Position: {self.current_position:.4f} | "
            f"Price: {self.current_price:.2f} | "
            f"P&L: {percent_change:.2f}% | "
            f"# Trades: {len(self.trades)}"
        )
        self.fig.suptitle(info_text, fontsize=12)
        
        self.fig.tight_layout()
        
        # Mostrar o devolver según el modo
        if self.render_mode == 'human':
            plt.pause(0.01)
            return None
        elif self.render_mode == 'rgb_array':
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img
    
    def close(self) -> None:
        """Cerrar el entorno y liberar recursos."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
    
    def get_trading_results(self) -> Dict[str, Any]:
        """
        Obtener resultados de la simulación de trading.
        
        Returns:
            Diccionario con resultados de la simulación
        """
        if not self.account_history:
            return {
                'initial_balance': self.initial_balance,
                'final_balance': self.initial_balance,
                'return': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Cálculos básicos
        initial_balance = self.initial_balance
        final_balance = self.current_balance + (self.current_position * self.current_price if self.current_position > 0 else 0)
        total_return = (final_balance - initial_balance) / initial_balance
        num_trades = len(self.trades)
        
        # Cálculo de WinRate
        profits = []
        for i in range(0, len(self.trades), 2):
            if i + 1 < len(self.trades):
                # Analizar pares de operaciones (entrada + salida)
                entry = self.trades[i]
                exit = self.trades[i+1]
                
                if entry['type'] == 'buy' and exit['type'] == 'sell':
                    # Long trade
                    profit = exit['price'] - entry['price']
                    profits.append(profit)
                elif entry['type'] == 'sell' and exit['type'] == 'buy':
                    # Short trade
                    profit = entry['price'] - exit['price']
                    profits.append(profit)
        
        win_rate = 0.0
        avg_profit = 0.0
        avg_loss = 0.0
        
        if profits:
            win_count = sum(1 for p in profits if p > 0)
            win_rate = win_count / len(profits) if len(profits) > 0 else 0.0
            
            profit_trades = [p for p in profits if p > 0]
            loss_trades = [p for p in profits if p <= 0]
            
            avg_profit = np.mean(profit_trades) if profit_trades else 0.0
            avg_loss = np.mean(loss_trades) if loss_trades else 0.0
        
        # Cálculo de máximo drawdown
        portfolio_values = [h['portfolio_value'] for h in self.account_history]
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Cálculo de Sharpe Ratio
        if len(portfolio_values) > 1:
            returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] for i in range(1, len(portfolio_values))]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def plot_results(self) -> str:
        """
        Generar gráfico de resultados de trading.
        
        Returns:
            Imagen en formato base64
        """
        if not self.account_history:
            return ""
        
        # Crear figura
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Datos completos
        data = self.df.copy()
        
        # Gráfico de precio y operaciones
        ax1.plot(data['timestamp'], data['close'], label='Close')
        
        # Marcar operaciones
        buys = [t for t in self.trades if t['type'] == 'buy']
        sells = [t for t in self.trades if t['type'] == 'sell']
        
        buy_times = [self.df['timestamp'].iloc[t['step']] for t in buys]
        buy_prices = [t['price'] for t in buys]
        
        sell_times = [self.df['timestamp'].iloc[t['step']] for t in sells]
        sell_prices = [t['price'] for t in sells]
        
        ax1.scatter(buy_times, buy_prices, marker='^', color='green', label='Buy')
        ax1.scatter(sell_times, sell_prices, marker='v', color='red', label='Sell')
        
        # Formato del gráfico de precio
        ax1.set_title('Trading Simulation Results')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de valor del portafolio
        portfolio_values = [h['portfolio_value'] for h in self.account_history]
        portfolio_times = [self.df['timestamp'].iloc[h['step']] for h in self.account_history]
        
        ax2.plot(portfolio_times, portfolio_values, label='Portfolio Value', color='blue')
        ax2.axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.7, label='Initial Balance')
        
        # Formato del gráfico de portafolio
        ax2.set_ylabel('Portfolio Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico de posiciones
        positions = [h['position'] for h in self.account_history]
        position_times = [self.df['timestamp'].iloc[h['step']] for h in self.account_history]
        
        ax3.fill_between(position_times, positions, 0, where=np.array(positions) > 0, color='green', alpha=0.3, label='Long')
        ax3.fill_between(position_times, positions, 0, where=np.array(positions) < 0, color='red', alpha=0.3, label='Short')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Formato del gráfico de posiciones
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Formatear eje X para mostrar fechas correctamente
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Añadir información de resultados
        results = self.get_trading_results()
        info_text = (
            f"Initial Balance: {results['initial_balance']:.2f} | "
            f"Final Balance: {results['final_balance']:.2f} | "
            f"Return: {results['return'] * 100:.2f}% | "
            f"Trades: {results['num_trades']} | "
            f"Win Rate: {results['win_rate'] * 100:.2f}% | "
            f"Max DD: {results['max_drawdown'] * 100:.2f}% | "
            f"Sharpe: {results['sharpe_ratio']:.2f}"
        )
        fig.suptitle(info_text, fontsize=12)
        
        fig.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return img_str


class MultiAssetTradingEnvironment(gym.Env):
    """
    Entorno de trading para múltiples activos.
    
    Este entorno permite operar con varios activos simultáneamente,
    balanceando un portafolio entre ellos.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array', 'none'], 'render_fps': 4}
    
    def __init__(self, 
                 dfs: Dict[str, pd.DataFrame],
                 initial_balance: float = 10000.0,
                 commission: float = 0.001,
                 reward_scaling: float = 1.0,
                 max_position: float = 1.0,
                 window_size: int = 20,
                 features: Optional[List[str]] = None,
                 render_mode: Optional[str] = None,
                 log_returns: bool = True,
                 reward_function: Optional[Callable] = None):
        """
        Inicializar entorno de trading multi-activo.
        
        Args:
            dfs: Diccionario de DataFrames con datos históricos para cada activo
            initial_balance: Saldo inicial en unidades de la moneda base (ej. USDT)
            commission: Comisión de trading en porcentaje (0.001 = 0.1%)
            reward_scaling: Factor de escala para las recompensas
            max_position: Tamaño máximo de posición como fracción del balance (1.0 = 100%)
            window_size: Tamaño de la ventana de observación
            features: Lista de características adicionales a incluir en la observación
            render_mode: Modo de renderizado ('human', 'rgb_array', 'none')
            log_returns: Si es True, usa log-returns en lugar de returns simples
            reward_function: Función personalizada para calcular recompensas
        """
        self.logger = logging.getLogger(__name__)
        
        # Validar DataFrames
        self.assets = list(dfs.keys())
        self.num_assets = len(self.assets)
        
        if self.num_assets == 0:
            raise ValueError("Debe proporcionarse al menos un activo")
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for asset, df in dfs.items():
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"DataFrame de {asset} debe contener columnas {required_columns}. Faltan: {missing}")
        
        # Datos y parámetros
        self.dfs = {asset: df.copy() for asset, df in dfs.items()}
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_scaling = reward_scaling
        self.max_position = max_position / self.num_assets  # Posición máxima por activo
        self.window_size = window_size
        self.render_mode = render_mode
        self.log_returns = log_returns
        self.reward_function = reward_function
        
        # Características a utilizar
        self.price_features = ['open', 'high', 'low', 'close', 'volume']
        self.features = features if features is not None else []
        
        # Calcular retornos para cada activo
        for asset, df in self.dfs.items():
            # Si las características especificadas no están en el DataFrame, iniciarlas
            for feature in self.features:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # Calcular retornos
            df['returns'] = df['close'].pct_change()
            if self.log_returns:
                df['returns'] = np.log1p(df['returns'])
            
            # Convertir NaN a 0
            df.fillna(0, inplace=True)
        
        # Espacio de acciones: Para cada activo: 0 (mantener), 1 (comprar), 2 (vender)
        self.action_space = spaces.Discrete(3**self.num_assets)
        
        # Espacio de observación
        num_price_features = len(self.price_features)
        num_features = len(self.features)
        
        # Observación incluye para cada activo:
        # - Ventana histórica para cada característica de precio
        # - Ventana histórica para cada característica adicional
        # - Estado actual del agente (posición, balance, etc.)
        
        # Detectar scale_range
        max_price = 0.0
        for df in self.dfs.values():
            if np.any(df['close'] > 0):
                max_price = max(max_price, np.max(df['close']))
        
        if max_price == 0.0:
            max_price = 100000.0
            
        price_scale = max_price * 10
        
        # Para cada activo, crear una observación
        obs_dict = {}
        for asset in self.assets:
            low_values = np.array([-price_scale] * (num_price_features + num_features))
            high_values = np.array([price_scale] * (num_price_features + num_features))
            
            obs_dict[f'{asset}_history'] = spaces.Box(
                low=np.tile(low_values, (self.window_size, 1)),
                high=np.tile(high_values, (self.window_size, 1)),
                shape=(self.window_size, num_price_features + num_features),
                dtype=np.float32
            )
        
        # Estado del agente para todos los activos
        # [posición_1, ..., posición_n, balance, portfolio_value]
        obs_dict['account_state'] = spaces.Box(
            low=np.array([-1.0] * self.num_assets + [0.0, 0.0]),
            high=np.array([1.0] * self.num_assets + [price_scale, price_scale]),
            shape=(self.num_assets + 2,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict(obs_dict)
        
        # Variables de estado
        self.current_step = None
        self.current_prices = None
        self.current_positions = None
        self.current_balance = None
        self.trades = None
        self.account_history = None
        
        # Para renderizado
        self.fig = None
        self.axes = None
        
        self.logger.info("Entorno de trading multi-activo inicializado con %d activos", self.num_assets)
    
    def _decode_action(self, action: int) -> List[int]:
        """
        Decodificar acción compuesta en acciones individuales para cada activo.
        
        Args:
            action: Acción compuesta (0 a 3^num_assets - 1)
            
        Returns:
            Lista de acciones individuales (0=mantener, 1=comprar, 2=vender)
        """
        actions = []
        remaining = action
        
        for _ in range(self.num_assets):
            actions.append(remaining % 3)
            remaining //= 3
        
        return actions
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Obtener observación actual del entorno.
        
        Returns:
            Diccionario con observación completa
        """
        observation = {}
        
        # Para cada activo, obtener ventana de observación
        for i, asset in enumerate(self.assets):
            df = self.dfs[asset]
            
            # Obtener ventana de datos históricos
            start_idx = max(0, self.current_step - self.window_size + 1)
            end_idx = self.current_step + 1
            
            # Si no hay suficientes datos históricos, rellenar con primeros valores
            if start_idx < 0:
                pad_size = abs(start_idx)
                start_idx = 0
            else:
                pad_size = 0
            
            # Matriz de histórico
            history_window = []
            
            # Primero características de precio
            for feature in self.price_features:
                # Extraer ventana
                feature_window = df[feature].iloc[start_idx:end_idx].values
                
                # Normalizar para características conocidas
                if feature in ['close', 'open', 'high', 'low']:
                    # Normalizar precio sobre precio de cierre actual
                    if self.current_prices[i] > 0:
                        feature_window = feature_window / self.current_prices[i] - 1.0
                    else:
                        feature_window = feature_window * 0.0
                elif feature == 'volume':
                    # Normalizar volumen sobre volumen máximo
                    vol_max = np.max(df['volume']) if np.max(df['volume']) > 0 else 1.0
                    feature_window = feature_window / vol_max
                
                # Rellenar si es necesario
                if pad_size > 0:
                    feature_window = np.concatenate([np.repeat(feature_window[0], pad_size), feature_window])
                
                history_window.append(feature_window)
            
            # Luego características adicionales
            for feature in self.features:
                # Extraer ventana
                feature_window = df[feature].iloc[start_idx:end_idx].values
                
                # Normalizar si valores son grandes
                if np.any(np.abs(feature_window) > 10):
                    # Normalizar sobre valor máximo absoluto
                    max_val = np.max(np.abs(df[feature])) if np.max(np.abs(df[feature])) > 0 else 1.0
                    feature_window = feature_window / max_val
                
                # Rellenar si es necesario
                if pad_size > 0:
                    feature_window = np.concatenate([np.repeat(feature_window[0], pad_size), feature_window])
                
                history_window.append(feature_window)
            
            # Transponemos para tener la forma (window_size, n_features)
            observation[f'{asset}_history'] = np.array(history_window).T.astype(np.float32)
        
        # Estado de la cuenta
        # [posición_1, ..., posición_n, balance, portfolio_value]
        account_state = np.zeros(self.num_assets + 2, dtype=np.float32)
        account_state[:self.num_assets] = self.current_positions
        account_state[self.num_assets] = self.current_balance / self.initial_balance  # Balance normalizado
        account_state[self.num_assets + 1] = self._get_portfolio_value() / self.initial_balance  # Portfolio normalizado
        
        observation['account_state'] = account_state
        
        return observation
    
    def _calculate_reward(self, actions: List[int]) -> float:
        """
        Calcular recompensa basada en las acciones y el movimiento del mercado.
        
        Args:
            actions: Lista de acciones tomadas para cada activo
            
        Returns:
            Recompensa
        """
        # Si se proporciona una función de recompensa personalizada, usarla
        if self.reward_function is not None:
            return self.reward_function(
                self, actions, self.current_step, self.trades, self.account_history
            ) * self.reward_scaling
        
        # Recompensa base: cambio en el valor del portafolio
        prev_portfolio_value = self.account_history[-2]['portfolio_value'] if len(self.account_history) > 1 else self.initial_balance
        current_portfolio_value = self.account_history[-1]['portfolio_value']
        
        # Calcular rendimiento del portafolio
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Escalar recompensa
        reward = portfolio_return * self.reward_scaling
        
        # Penalizar por comisiones excesivas (si hubo operaciones en este paso)
        operations_this_step = sum(1 for t in self.trades if t['step'] == self.current_step)
        if operations_this_step > 0:
            reward -= self.commission * operations_this_step * self.reward_scaling
        
        return reward
    
    def _take_action(self, actions: List[int]) -> Tuple[float, Dict[str, Any]]:
        """
        Ejecutar acciones en el entorno.
        
        Args:
            actions: Lista de acciones para cada activo (0=mantener, 1=comprar, 2=vender)
            
        Returns:
            Tupla de (recompensa, información)
        """
        prev_portfolio_value = self._get_portfolio_value()
        operations_count = 0
        
        # Procesar acción para cada activo
        for i, (asset, action) in enumerate(zip(self.assets, actions)):
            current_price = self.current_prices[i]
            
            if action == 1:  # Comprar
                if self.current_positions[i] < self.max_position:
                    # Calcular cantidad a comprar
                    target_position = self.max_position
                    amount_to_buy = target_position - self.current_positions[i]
                    
                    # Verificar balance disponible
                    cost = amount_to_buy * current_price * (1 + self.commission)
                    if cost > self.current_balance:
                        # Ajustar si no hay suficiente balance
                        amount_to_buy = self.current_balance / (current_price * (1 + self.commission))
                    
                    if amount_to_buy > 0:
                        # Registrar operación
                        commission = amount_to_buy * current_price * self.commission
                        self.current_balance -= (amount_to_buy * current_price + commission)
                        
                        # Actualizar posición
                        self.current_positions[i] += amount_to_buy
                        
                        # Registrar operación
                        self.trades.append({
                            'step': self.current_step,
                            'timestamp': self.dfs[asset]['timestamp'].iloc[self.current_step],
                            'asset': asset,
                            'type': 'buy',
                            'price': current_price,
                            'amount': amount_to_buy,
                            'commission': commission,
                            'balance': self.current_balance
                        })
                        
                        operations_count += 1
            
            elif action == 2:  # Vender
                if self.current_positions[i] > -self.max_position:
                    # Calcular cantidad a vender
                    target_position = -self.max_position
                    amount_to_sell = self.current_positions[i] - target_position
                    
                    if amount_to_sell > 0:
                        # Registrar operación
                        commission = amount_to_sell * current_price * self.commission
                        self.current_balance += (amount_to_sell * current_price - commission)
                        
                        # Actualizar posición
                        self.current_positions[i] -= amount_to_sell
                        
                        # Registrar operación
                        self.trades.append({
                            'step': self.current_step,
                            'timestamp': self.dfs[asset]['timestamp'].iloc[self.current_step],
                            'asset': asset,
                            'type': 'sell',
                            'price': current_price,
                            'amount': amount_to_sell,
                            'commission': commission,
                            'balance': self.current_balance
                        })
                        
                        operations_count += 1
        
        # Actualizar historial
        self.account_history.append({
            'step': self.current_step,
            'timestamp': self.dfs[self.assets[0]]['timestamp'].iloc[self.current_step],
            'balance': self.current_balance,
            'positions': self.current_positions.copy(),
            'prices': self.current_prices.copy(),
            'portfolio_value': self._get_portfolio_value()
        })
        
        # Calcular recompensa
        reward = self._calculate_reward(actions)
        
        # Preparar información adicional
        info = {
            'portfolio_value': self._get_portfolio_value(),
            'portfolio_return': (self._get_portfolio_value() - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0,
            'prices': self.current_prices.copy(),
            'positions': self.current_positions.copy(),
            'balance': self.current_balance,
            'operations': operations_count
        }
        
        return reward, info
    
    def _get_portfolio_value(self) -> float:
        """
        Calcular valor total del portafolio.
        
        Returns:
            Valor del portafolio (balance + posiciones)
        """
        value = self.current_balance
        
        # Sumar valor de posiciones largas
        for i, price in enumerate(self.current_prices):
            position = self.current_positions[i]
            if position > 0:
                value += position * price
        
        return value
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reiniciar el entorno al estado inicial.
        
        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales
            
        Returns:
            Tupla de (observación, información)
        """
        super().reset(seed=seed)
        
        # Inicializar variables de estado
        self.current_step = self.window_size - 1
        self.current_prices = np.array([df['close'].iloc[self.current_step] for df in self.dfs.values()])
        self.current_positions = np.zeros(self.num_assets)
        self.current_balance = self.initial_balance
        self.trades = []
        self.account_history = [{
            'step': self.current_step,
            'timestamp': self.dfs[self.assets[0]]['timestamp'].iloc[self.current_step],
            'balance': self.current_balance,
            'positions': self.current_positions.copy(),
            'prices': self.current_prices.copy(),
            'portfolio_value': self._get_portfolio_value()
        }]
        
        # Obtener observación inicial
        observation = self._get_observation()
        
        # Información inicial
        info = {
            'portfolio_value': self._get_portfolio_value(),
            'prices': self.current_prices.copy(),
            'positions': self.current_positions.copy(),
            'balance': self.current_balance
        }
        
        if self.render_mode == 'human':
            self._render_frame()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Avanzar el entorno un paso con la acción dada.
        
        Args:
            action: Acción compuesta a ejecutar
            
        Returns:
            Tupla de (observación, recompensa, terminado, truncado, información)
        """
        # Decodificar acción compuesta en acciones individuales
        actions = self._decode_action(action)
        
        # Ejecutar acciones y calcular recompensa
        reward, info = self._take_action(actions)
        
        # Avanzar al siguiente paso
        self.current_step += 1
        
        # Verificar si la simulación ha terminado
        done = self.current_step >= len(self.dfs[self.assets[0]]) - 1
        truncated = False
        
        if not done:
            # Actualizar precios actuales
            self.current_prices = np.array([df['close'].iloc[self.current_step] for df in self.dfs.values()])
            
            # Actualizar observación
            observation = self._get_observation()
            
            # Actualizar información
            info.update({
                'portfolio_value': self._get_portfolio_value(),
                'prices': self.current_prices.copy(),
                'positions': self.current_positions.copy(),
                'balance': self.current_balance
            })
        else:
            # Final de la simulación, liquidar posiciones
            for i, asset in enumerate(self.assets):
                position = self.current_positions[i]
                
                if position != 0:
                    # Cerrar posición
                    price = self.current_prices[i]
                    
                    if position > 0:
                        # Vender posición larga
                        commission = position * price * self.commission
                        self.current_balance += (position * price - commission)
                        
                        # Registrar operación final
                        self.trades.append({
                            'step': self.current_step,
                            'timestamp': self.dfs[asset]['timestamp'].iloc[self.current_step],
                            'asset': asset,
                            'type': 'sell',
                            'price': price,
                            'amount': position,
                            'commission': commission,
                            'balance': self.current_balance
                        })
                        
                        self.current_positions[i] = 0.0
                    
                    elif position < 0:
                        # Cerrar posición corta
                        commission = abs(position) * price * self.commission
                        self.current_balance -= (abs(position) * price + commission)
                        
                        # Registrar operación final
                        self.trades.append({
                            'step': self.current_step,
                            'timestamp': self.dfs[asset]['timestamp'].iloc[self.current_step],
                            'asset': asset,
                            'type': 'buy',
                            'price': price,
                            'amount': abs(position),
                            'commission': commission,
                            'balance': self.current_balance
                        })
                        
                        self.current_positions[i] = 0.0
            
            # Actualizar historial final
            self.account_history.append({
                'step': self.current_step,
                'timestamp': self.dfs[self.assets[0]]['timestamp'].iloc[self.current_step],
                'balance': self.current_balance,
                'positions': self.current_positions.copy(),
                'prices': self.current_prices.copy(),
                'portfolio_value': self._get_portfolio_value()
            })
            
            # Observación final
            observation = self._get_observation()
            
            # Información final
            info.update({
                'portfolio_value': self._get_portfolio_value(),
                'final_balance': self.current_balance,
                'return': (self._get_portfolio_value() - self.initial_balance) / self.initial_balance,
                'num_trades': len(self.trades)
            })
        
        if self.render_mode == 'human':
            self._render_frame()
        
        return observation, reward, done, truncated, info
    
    def render(self) -> Optional[Union[np.ndarray, Figure]]:
        """
        Renderizar el estado actual del entorno.
        
        Returns:
            None, array RGB o figura según el modo de renderizado
        """
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        
        return None
    
    def _render_frame(self) -> Optional[Union[np.ndarray, Figure]]:
        """
        Renderizar un frame del entorno.
        
        Returns:
            None, array RGB o figura según el modo de renderizado
        """
        if self.render_mode not in ['human', 'rgb_array']:
            return None
        
        # Crear figura y ejes si no existen
        if self.fig is None or self.axes is None:
            # Un gráfico por activo, más uno para el portafolio
            self.fig, self.axes = plt.subplots(
                self.num_assets + 1, 1, 
                figsize=(12, 4 * (self.num_assets + 1)), 
                sharex=True, 
                gridspec_kw={'height_ratios': [3] * self.num_assets + [2]}
            )
            if self.num_assets == 1:
                self.axes = [self.axes[0], self.axes[1]]
        else:
            for ax in self.axes:
                ax.clear()
        
        # Preparar datos hasta el paso actual
        timestamp = self.dfs[self.assets[0]]['timestamp'].iloc[:self.current_step+1]
        
        # Gráficos de precio para cada activo
        for i, asset in enumerate(self.assets):
            data = self.dfs[asset].iloc[:self.current_step+1]
            
            # Gráfico de precio
            self.axes[i].plot(timestamp, data['close'], label=f'{asset} Close')
            
            # Marcar operaciones
            asset_buys = [t for t in self.trades if t['step'] <= self.current_step and t['type'] == 'buy' and t['asset'] == asset]
            asset_sells = [t for t in self.trades if t['step'] <= self.current_step and t['type'] == 'sell' and t['asset'] == asset]
            
            buy_times = [self.dfs[asset]['timestamp'].iloc[t['step']] for t in asset_buys]
            buy_prices = [t['price'] for t in asset_buys]
            
            sell_times = [self.dfs[asset]['timestamp'].iloc[t['step']] for t in asset_sells]
            sell_prices = [t['price'] for t in asset_sells]
            
            self.axes[i].scatter(buy_times, buy_prices, marker='^', color='green', label='Buy')
            self.axes[i].scatter(sell_times, sell_prices, marker='v', color='red', label='Sell')
            
            # Formato del gráfico
            self.axes[i].set_title(f'{asset} - Position: {self.current_positions[i]:.4f}')
            self.axes[i].set_ylabel('Price')
            self.axes[i].legend()
            self.axes[i].grid(True, alpha=0.3)
        
        # Gráfico de valor del portafolio
        portfolio_values = [h['portfolio_value'] for h in self.account_history if h['step'] <= self.current_step]
        portfolio_times = [self.dfs[self.assets[0]]['timestamp'].iloc[h['step']] for h in self.account_history if h['step'] <= self.current_step]
        
        self.axes[-1].plot(portfolio_times, portfolio_values, label='Portfolio Value', color='blue')
        self.axes[-1].axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.7, label='Initial Balance')
        
        # Formato del gráfico de portafolio
        self.axes[-1].set_ylabel('Portfolio Value')
        self.axes[-1].set_xlabel('Time')
        self.axes[-1].legend()
        self.axes[-1].grid(True, alpha=0.3)
        
        # Formatear eje X para mostrar fechas correctamente
        self.axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.fig.autofmt_xdate()
        
        # Añadir información adicional en el título
        percent_change = ((self._get_portfolio_value() - self.initial_balance) / self.initial_balance) * 100
        info_text = (
            f"Balance: {self.current_balance:.2f} | "
            f"P&L: {percent_change:.2f}% | "
            f"# Trades: {len(self.trades)}"
        )
        self.fig.suptitle(info_text, fontsize=12)
        
        self.fig.tight_layout()
        
        # Mostrar o devolver según el modo
        if self.render_mode == 'human':
            plt.pause(0.01)
            return None
        elif self.render_mode == 'rgb_array':
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img
    
    def close(self) -> None:
        """Cerrar el entorno y liberar recursos."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
    
    def get_trading_results(self) -> Dict[str, Any]:
        """
        Obtener resultados de la simulación de trading.
        
        Returns:
            Diccionario con resultados de la simulación
        """
        if not self.account_history:
            return {
                'initial_balance': self.initial_balance,
                'final_balance': self.initial_balance,
                'return': 0.0,
                'num_trades': 0,
                'trades_per_asset': {asset: 0 for asset in self.assets},
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Cálculos básicos
        initial_balance = self.initial_balance
        final_balance = self.current_balance
        total_return = (final_balance - initial_balance) / initial_balance
        num_trades = len(self.trades)
        
        # Trades por activo
        trades_per_asset = {}
        for asset in self.assets:
            asset_trades = [t for t in self.trades if t['asset'] == asset]
            trades_per_asset[asset] = len(asset_trades)
        
        # Cálculo de Win Rate global
        wins = 0
        losses = 0
        
        # Agrupar trades por (asset, tipo, precio)
        for asset in self.assets:
            asset_trades = sorted([t for t in self.trades if t['asset'] == asset], key=lambda x: x['step'])
            
            # Analizar secuencias de trades
            for i in range(0, len(asset_trades) - 1, 2):
                if i + 1 < len(asset_trades):
                    entry = asset_trades[i]
                    exit = asset_trades[i+1]
                    
                    if entry['type'] == 'buy' and exit['type'] == 'sell':
                        # Long trade
                        profit = exit['price'] - entry['price']
                        if profit > 0:
                            wins += 1
                        else:
                            losses += 1
                    elif entry['type'] == 'sell' and exit['type'] == 'buy':
                        # Short trade
                        profit = entry['price'] - exit['price']
                        if profit > 0:
                            wins += 1
                        else:
                            losses += 1
        
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
        
        # Cálculo de máximo drawdown
        portfolio_values = [h['portfolio_value'] for h in self.account_history]
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Cálculo de Sharpe Ratio
        if len(portfolio_values) > 1:
            returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] for i in range(1, len(portfolio_values))]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'return': total_return,
            'num_trades': num_trades,
            'trades_per_asset': trades_per_asset,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def plot_results(self) -> str:
        """
        Generar gráfico de resultados de trading.
        
        Returns:
            Imagen en formato base64
        """
        if not self.account_history:
            return ""
        
        # Crear figura
        fig, axes = plt.subplots(
            self.num_assets + 1, 1, 
            figsize=(12, 4 * (self.num_assets + 1)), 
            sharex=True, 
            gridspec_kw={'height_ratios': [3] * self.num_assets + [2]}
        )
        
        if self.num_assets == 1:
            axes = [axes[0], axes[1]]
        
        # Gráficos de precio para cada activo
        for i, asset in enumerate(self.assets):
            data = self.dfs[asset]
            
            # Gráfico de precio
            axes[i].plot(data['timestamp'], data['close'], label=f'{asset} Close')
            
            # Marcar operaciones
            asset_buys = [t for t in self.trades if t['type'] == 'buy' and t['asset'] == asset]
            asset_sells = [t for t in self.trades if t['type'] == 'sell' and t['asset'] == asset]
            
            buy_times = [self.dfs[asset]['timestamp'].iloc[t['step']] for t in asset_buys]
            buy_prices = [t['price'] for t in asset_buys]
            
            sell_times = [self.dfs[asset]['timestamp'].iloc[t['step']] for t in asset_sells]
            sell_prices = [t['price'] for t in asset_sells]
            
            axes[i].scatter(buy_times, buy_prices, marker='^', color='green', label='Buy')
            axes[i].scatter(sell_times, sell_prices, marker='v', color='red', label='Sell')
            
            # Formato del gráfico
            axes[i].set_title(f'{asset}')
            axes[i].set_ylabel('Price')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Gráfico de valor del portafolio
        portfolio_values = [h['portfolio_value'] for h in self.account_history]
        portfolio_times = [self.dfs[self.assets[0]]['timestamp'].iloc[h['step']] for h in self.account_history]
        
        axes[-1].plot(portfolio_times, portfolio_values, label='Portfolio Value', color='blue')
        axes[-1].axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.7, label='Initial Balance')
        
        # Formato del gráfico de portafolio
        axes[-1].set_ylabel('Portfolio Value')
        axes[-1].set_xlabel('Time')
        axes[-1].legend()
        axes[-1].grid(True, alpha=0.3)
        
        # Formatear eje X para mostrar fechas correctamente
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Añadir información de resultados
        results = self.get_trading_results()
        info_text = (
            f"Initial Balance: {results['initial_balance']:.2f} | "
            f"Final Balance: {results['final_balance']:.2f} | "
            f"Return: {results['return'] * 100:.2f}% | "
            f"Trades: {results['num_trades']} | "
            f"Win Rate: {results['win_rate'] * 100:.2f}% | "
            f"Max DD: {results['max_drawdown'] * 100:.2f}% | "
            f"Sharpe: {results['sharpe_ratio']:.2f}"
        )
        fig.suptitle(info_text, fontsize=12)
        
        fig.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return img_str