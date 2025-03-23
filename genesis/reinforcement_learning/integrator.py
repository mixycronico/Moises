"""
Integrador de Reinforcement Learning con el Sistema Genesis.

Este módulo proporciona la integración entre los agentes RL entrenados
y el sistema de trading Genesis, permitiendo la toma de decisiones
en tiempo real basada en políticas optimizadas mediante RL.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Importaciones para RL
from genesis.reinforcement_learning.environments import TradingEnvironment, MultiAssetTradingEnvironment
from genesis.reinforcement_learning.agents import RLAgent, DQNAgent, PPOAgent, SACAgent, RLAgentManager
from genesis.reinforcement_learning.evaluation import BacktestAgent, HyperparameterOptimizer

class RLStrategyIntegrator:
    """
    Integrador de estrategias de RL con el Sistema Genesis.
    
    Esta clase permite utilizar agentes RL entrenados para tomar decisiones
    de trading en el contexto del sistema Genesis, adaptando estados del
    sistema a observaciones para el agente y decisiones del agente a acciones
    en el sistema.
    """
    
    def __init__(self, 
                 agent_manager: RLAgentManager,
                 feature_columns: List[str],
                 window_size: int = 20,
                 position_scaling: float = 1.0,
                 risk_per_trade: float = 0.02,
                 max_position_value: Optional[float] = None,
                 use_deterministic_policy: bool = True,
                 cache_dir: str = './cache/rl_strategies'):
        """
        Inicializar integrador de estrategias RL.
        
        Args:
            agent_manager: Gestor de agentes RL
            feature_columns: Columnas de características para observaciones
            window_size: Tamaño de la ventana para observaciones
            position_scaling: Factor de escala para tamaños de posición
            risk_per_trade: Riesgo por operación (porcentaje del capital)
            max_position_value: Valor máximo de posición (None = sin límite)
            use_deterministic_policy: Si es True, usa política determinista
            cache_dir: Directorio para caché
        """
        self.logger = logging.getLogger(__name__)
        self.agent_manager = agent_manager
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.position_scaling = position_scaling
        self.risk_per_trade = risk_per_trade
        self.max_position_value = max_position_value
        self.use_deterministic_policy = use_deterministic_policy
        self.cache_dir = cache_dir
        
        # Crear directorio de caché si no existe
        os.makedirs(cache_dir, exist_ok=True)
        
        # Estado interno
        self.active_agent_id = None
        self.market_history = {}
        self.current_positions = {}
        self.current_balance = 0.0
        self.trade_history = []
        
        # Contadores de señales
        self.signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        # Cache de observaciones
        self.observation_cache = {}
        
        self.logger.info("Integrador de estrategias RL inicializado")
    
    async def set_active_agent(self, agent_id: str) -> bool:
        """
        Establecer agente activo para toma de decisiones.
        
        Args:
            agent_id: ID del agente a activar
            
        Returns:
            True si se activó correctamente
        """
        if agent_id not in self.agent_manager.agents:
            raise ValueError(f"Agente no encontrado: {agent_id}")
        
        agent = self.agent_manager.agents[agent_id]
        
        if agent.model is None:
            raise ValueError(f"El agente {agent_id} no ha sido entrenado")
        
        self.active_agent_id = agent_id
        self.logger.info(f"Agente activo establecido: {agent_id}")
        
        return True
    
    def _prepare_observation_single_asset(self, 
                                       symbol: str, 
                                       data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Preparar observación para un solo activo.
        
        Args:
            symbol: Símbolo del activo
            data: DataFrame con datos históricos
            
        Returns:
            Observación en formato compatible con el agente
        """
        # Verificar si tenemos suficientes datos
        if len(data) < self.window_size:
            raise ValueError(f"No hay suficientes datos para {symbol}. Se requieren al menos {self.window_size} registros.")
        
        # Obtener últimos datos
        df_window = data.tail(self.window_size).copy()
        
        # Verificar columnas requeridas
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df_window.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas en los datos: {missing}")
        
        # Verificar columnas de características
        missing_features = [col for col in self.feature_columns if col not in df_window.columns]
        if missing_features:
            self.logger.warning(f"Faltan características: {missing_features}. Se usarán ceros.")
            for col in missing_features:
                df_window[col] = 0.0
        
        # Normalizar datos
        current_price = df_window['close'].iloc[-1]
        
        # Precios normalizados
        df_window['open'] = df_window['open'] / current_price - 1.0
        df_window['high'] = df_window['high'] / current_price - 1.0
        df_window['low'] = df_window['low'] / current_price - 1.0
        df_window['close'] = df_window['close'] / current_price - 1.0
        
        # Volumen normalizado
        vol_max = df_window['volume'].max() if df_window['volume'].max() > 0 else 1.0
        df_window['volume'] = df_window['volume'] / vol_max
        
        # Normalizar características adicionales
        for feature in self.feature_columns:
            if feature in df_window.columns:
                if np.any(np.abs(df_window[feature]) > 10):
                    # Normalizar sobre valor máximo absoluto
                    max_val = np.max(np.abs(df_window[feature])) if np.max(np.abs(df_window[feature])) > 0 else 1.0
                    df_window[feature] = df_window[feature] / max_val
        
        # Preparar observación para el entorno
        # Primero características de precio
        price_features = ['open', 'high', 'low', 'close', 'volume']
        market_history = df_window[price_features + self.feature_columns].values
        
        # Posición actual para este símbolo
        position = self.current_positions.get(symbol, 0.0)
        # Normalizar posición (-1 a 1)
        position_norm = np.clip(position / (self.max_position_value or 1.0), -1.0, 1.0)
        
        # Estado de la cuenta
        account_state = np.array([
            position_norm,  # Posición normalizada
            1.0,  # Balance normalizado (siempre 1 para simplicidad)
            0.0   # PnL normalizada (siempre 0 para simplicidad)
        ], dtype=np.float32)
        
        # Observación final
        observation = {
            'market_history': market_history.astype(np.float32),
            'account_state': account_state
        }
        
        return observation
    
    def _prepare_observation_multi_asset(self, 
                                      data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Preparar observación para múltiples activos.
        
        Args:
            data: Diccionario con DataFrames de datos históricos
            
        Returns:
            Observación en formato compatible con el agente
        """
        observation = {}
        
        # Procesar cada activo
        for symbol, df in data.items():
            # Verificar si tenemos suficientes datos
            if len(df) < self.window_size:
                raise ValueError(f"No hay suficientes datos para {symbol}. Se requieren al menos {self.window_size} registros.")
            
            # Obtener últimos datos
            df_window = df.tail(self.window_size).copy()
            
            # Verificar columnas requeridas
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required_columns if col not in df_window.columns]
            if missing:
                raise ValueError(f"Faltan columnas requeridas en los datos para {symbol}: {missing}")
            
            # Verificar columnas de características
            missing_features = [col for col in self.feature_columns if col not in df_window.columns]
            if missing_features:
                self.logger.warning(f"Faltan características para {symbol}: {missing_features}. Se usarán ceros.")
                for col in missing_features:
                    df_window[col] = 0.0
            
            # Normalizar datos
            current_price = df_window['close'].iloc[-1]
            
            # Precios normalizados
            df_window['open'] = df_window['open'] / current_price - 1.0
            df_window['high'] = df_window['high'] / current_price - 1.0
            df_window['low'] = df_window['low'] / current_price - 1.0
            df_window['close'] = df_window['close'] / current_price - 1.0
            
            # Volumen normalizado
            vol_max = df_window['volume'].max() if df_window['volume'].max() > 0 else 1.0
            df_window['volume'] = df_window['volume'] / vol_max
            
            # Normalizar características adicionales
            for feature in self.feature_columns:
                if feature in df_window.columns:
                    if np.any(np.abs(df_window[feature]) > 10):
                        # Normalizar sobre valor máximo absoluto
                        max_val = np.max(np.abs(df_window[feature])) if np.max(np.abs(df_window[feature])) > 0 else 1.0
                        df_window[feature] = df_window[feature] / max_val
            
            # Preparar observación para este activo
            price_features = ['open', 'high', 'low', 'close', 'volume']
            observation[f'{symbol}_history'] = df_window[price_features + self.feature_columns].values.astype(np.float32)
        
        # Posiciones actuales para todos los activos
        positions = [self.current_positions.get(symbol, 0.0) for symbol in data.keys()]
        # Normalizar posiciones (-1 a 1)
        positions_norm = [np.clip(pos / (self.max_position_value or 1.0), -1.0, 1.0) for pos in positions]
        
        # Estado de la cuenta
        # [posición_1, ..., posición_n, balance, portfolio_value]
        account_state = np.array(
            positions_norm + [1.0, 1.0],  # Balance y portfolio normalizados (siempre 1 para simplicidad)
            dtype=np.float32
        )
        
        observation['account_state'] = account_state
        
        return observation
    
    def _decode_multi_asset_action(self, action: int, n_assets: int) -> List[int]:
        """
        Decodificar acción compuesta en acciones individuales para cada activo.
        
        Args:
            action: Acción compuesta (0 a 3^num_assets - 1)
            n_assets: Número de activos
            
        Returns:
            Lista de acciones individuales (0=mantener, 1=comprar, 2=vender)
        """
        actions = []
        remaining = action
        
        for _ in range(n_assets):
            actions.append(remaining % 3)
            remaining //= 3
        
        return actions
    
    async def get_trading_decision(self, 
                             symbols: Union[str, List[str]],
                             data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                             balance: float) -> Dict[str, Any]:
        """
        Obtener decisión de trading del agente RL.
        
        Args:
            symbols: Símbolo o lista de símbolos
            data: DataFrame o diccionario de DataFrames con datos históricos
            balance: Balance disponible
            
        Returns:
            Diccionario con decisión de trading
        """
        if self.active_agent_id is None:
            raise ValueError("No hay un agente activo establecido")
        
        agent = self.agent_manager.agents[self.active_agent_id]
        
        # Actualizar balance
        self.current_balance = balance
        
        # Determinar si es single o multi-asset
        is_multi_asset = isinstance(symbols, list) and len(symbols) > 1
        
        # Preparar observación
        if is_multi_asset:
            if not isinstance(data, dict):
                raise ValueError("Para múltiples activos, data debe ser un diccionario")
            
            observation = self._prepare_observation_multi_asset(data)
            
            # Cachear datos por símbolo
            for symbol, df in data.items():
                self.market_history[symbol] = df.copy()
        else:
            # Para un solo activo
            symbol = symbols if isinstance(symbols, str) else symbols[0]
            
            if isinstance(data, dict):
                df = data[symbol]
            else:
                df = data
            
            observation = self._prepare_observation_single_asset(symbol, df)
            
            # Cachear datos
            self.market_history[symbol] = df.copy()
        
        # Realizar predicción con el agente
        action, _ = await self.agent_manager.predict(
            agent_id=self.active_agent_id,
            observation=observation,
            deterministic=self.use_deterministic_policy
        )
        
        # Procesar acción
        decision = {}
        
        if is_multi_asset:
            # Decodificar acción para múltiples activos
            individual_actions = self._decode_multi_asset_action(action, len(symbols))
            
            # Interpretar cada acción
            for i, (symbol, act) in enumerate(zip(symbols, individual_actions)):
                current_price = data[symbol]['close'].iloc[-1]
                
                if act == 1:  # Comprar
                    signal = 'BUY'
                    # Calcular cantidad según riesgo y balance
                    amount = self._calculate_position_size(balance, current_price, self.risk_per_trade)
                elif act == 2:  # Vender
                    signal = 'SELL'
                    amount = self._calculate_position_size(balance, current_price, self.risk_per_trade)
                else:  # Mantener
                    signal = 'HOLD'
                    amount = 0.0
                
                # Guardar decisión para este símbolo
                decision[symbol] = {
                    'signal': signal,
                    'amount': amount,
                    'price': current_price,
                    'confidence': 1.0  # Siempre 1.0 para simplicidad
                }
                
                # Actualizar conteo de señales
                self.signal_counts[signal] = self.signal_counts.get(signal, 0) + 1
        else:
            # Para un solo activo
            symbol = symbols if isinstance(symbols, str) else symbols[0]
            current_price = df['close'].iloc[-1]
            
            if action == 1:  # Comprar
                signal = 'BUY'
                # Calcular cantidad según riesgo y balance
                amount = self._calculate_position_size(balance, current_price, self.risk_per_trade)
            elif action == 2:  # Vender
                signal = 'SELL'
                amount = self._calculate_position_size(balance, current_price, self.risk_per_trade)
            else:  # Mantener
                signal = 'HOLD'
                amount = 0.0
            
            # Guardar decisión
            decision[symbol] = {
                'signal': signal,
                'amount': amount,
                'price': current_price,
                'confidence': 1.0  # Siempre 1.0 para simplicidad
            }
            
            # Actualizar conteo de señales
            self.signal_counts[signal] = self.signal_counts.get(signal, 0) + 1
        
        # Registrar decisión
        timestamp = datetime.now().isoformat()
        decision_record = {
            'timestamp': timestamp,
            'symbols': symbols if isinstance(symbols, list) else [symbols],
            'decision': decision,
            'agent_id': self.active_agent_id,
            'balance': balance
        }
        
        # Guardar en historial
        self.trade_history.append(decision_record)
        
        # Si hay muchos registros, limitar
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        
        return decision
    
    def _calculate_position_size(self, 
                               balance: float, 
                               current_price: float, 
                               risk_per_trade: float) -> float:
        """
        Calcular tamaño de posición basado en riesgo.
        
        Args:
            balance: Balance disponible
            current_price: Precio actual
            risk_per_trade: Riesgo por operación (porcentaje del capital)
            
        Returns:
            Tamaño de posición
        """
        # Calcular valor monetario del riesgo
        risk_amount = balance * risk_per_trade
        
        # Calcular tamaño en unidades
        position_size = risk_amount / current_price
        
        # Aplicar escalado
        position_size *= self.position_scaling
        
        # Limitar por valor máximo si está definido
        if self.max_position_value is not None and position_size * current_price > self.max_position_value:
            position_size = self.max_position_value / current_price
        
        return position_size
    
    async def update_position(self, symbol: str, position: float) -> None:
        """
        Actualizar posición actual para un símbolo.
        
        Args:
            symbol: Símbolo del activo
            position: Nueva posición
        """
        self.current_positions[symbol] = position
    
    def get_signal_stats(self) -> Dict[str, int]:
        """
        Obtener estadísticas de señales generadas.
        
        Returns:
            Diccionario con conteo de señales
        """
        return self.signal_counts
    
    async def save_state(self, filepath: Optional[str] = None) -> str:
        """
        Guardar estado del integrador.
        
        Args:
            filepath: Ruta donde guardar el estado (opcional)
            
        Returns:
            Ruta donde se guardó el estado
        """
        # Generar ruta si no se proporciona
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.cache_dir, f"strategy_state_{timestamp}.json")
        
        # Preparar estado
        state = {
            'active_agent_id': self.active_agent_id,
            'current_positions': self.current_positions,
            'current_balance': self.current_balance,
            'signal_counts': self.signal_counts,
            'trade_history': self.trade_history[-100:],  # Últimos 100 para no hacer archivo muy grande
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar estado
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Estado guardado en {filepath}")
        
        return filepath
    
    async def load_state(self, filepath: str) -> bool:
        """
        Cargar estado del integrador.
        
        Args:
            filepath: Ruta desde donde cargar el estado
            
        Returns:
            True si se cargó correctamente
        """
        try:
            # Cargar estado
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restaurar estado
            self.active_agent_id = state.get('active_agent_id')
            self.current_positions = state.get('current_positions', {})
            self.current_balance = state.get('current_balance', 0.0)
            self.signal_counts = state.get('signal_counts', {'BUY': 0, 'SELL': 0, 'HOLD': 0})
            self.trade_history = state.get('trade_history', [])
            
            self.logger.info(f"Estado cargado desde {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando estado: {str(e)}")
            return False
    
    def plot_decision_history(self, symbol: Optional[str] = None, save_path: Optional[str] = None) -> str:
        """
        Generar gráfico de historial de decisiones.
        
        Args:
            symbol: Símbolo específico a graficar (None = todos)
            save_path: Ruta donde guardar el gráfico
            
        Returns:
            Imagen en formato base64
        """
        # Filtrar decisiones por símbolo si se especifica
        if symbol:
            decisions = [
                d for d in self.trade_history 
                if symbol in d['symbols'] and symbol in d['decision']
            ]
            
            if not decisions:
                return ""
            
            # Extraer datos relevantes
            timestamps = [datetime.fromisoformat(d['timestamp']) for d in decisions]
            signals = [d['decision'][symbol]['signal'] for d in decisions]
            prices = [d['decision'][symbol]['price'] for d in decisions]
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Gráfico de precio
            ax.plot(timestamps, prices, label='Precio', color='blue')
            
            # Marcar señales
            buy_times = [t for t, s in zip(timestamps, signals) if s == 'BUY']
            buy_prices = [p for p, s in zip(prices, signals) if s == 'BUY']
            
            sell_times = [t for t, s in zip(timestamps, signals) if s == 'SELL']
            sell_prices = [p for p, s in zip(prices, signals) if s == 'SELL']
            
            ax.scatter(buy_times, buy_prices, marker='^', color='green', label='Compra')
            ax.scatter(sell_times, sell_prices, marker='v', color='red', label='Venta')
            
            # Formato
            ax.set_title(f'Historial de Decisiones para {symbol}')
            ax.set_xlabel('Fecha')
            ax.set_ylabel('Precio')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
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
                
                self.logger.info(f"Gráfico guardado en {save_path}")
            
            return img_str
            
        else:
            # Para todos los símbolos, gráfico de conteo de señales
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Datos
            signals = ['BUY', 'SELL', 'HOLD']
            counts = [self.signal_counts.get(s, 0) for s in signals]
            
            # Gráfico de barras
            ax.bar(signals, counts, color=['green', 'red', 'blue'])
            
            # Formato
            ax.set_title('Distribución de Señales')
            ax.set_xlabel('Señal')
            ax.set_ylabel('Conteo')
            ax.grid(True, alpha=0.3, axis='y')
            
            fig.tight_layout()
            
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
                
                self.logger.info(f"Gráfico guardado en {save_path}")
            
            return img_str
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de rendimiento del integrador.
        
        Returns:
            Diccionario con estadísticas
        """
        # Estadísticas básicas
        stats = {
            'total_decisions': len(self.trade_history),
            'signal_counts': self.signal_counts.copy(),
            'active_symbols': len(self.current_positions),
            'active_positions': self.current_positions.copy()
        }
        
        # Calcular ratios
        total_signals = sum(self.signal_counts.values())
        if total_signals > 0:
            stats['buy_ratio'] = self.signal_counts.get('BUY', 0) / total_signals
            stats['sell_ratio'] = self.signal_counts.get('SELL', 0) / total_signals
            stats['hold_ratio'] = self.signal_counts.get('HOLD', 0) / total_signals
        
        return stats


class RLStrategyManager:
    """
    Gestor de estrategias de Reinforcement Learning.
    
    Esta clase permite gestionar múltiples estrategias basadas en RL,
    facilitando su entrenamiento, evaluación y distribución de capital.
    """
    
    def __init__(self, 
                 agent_manager: RLAgentManager,
                 db: Optional[Any] = None,
                 cache_dir: str = './cache/rl_strategies',
                 logs_dir: str = './logs/rl_strategies'):
        """
        Inicializar gestor de estrategias RL.
        
        Args:
            agent_manager: Gestor de agentes RL
            db: Conexión a base de datos (opcional)
            cache_dir: Directorio para caché
            logs_dir: Directorio para logs
        """
        self.logger = logging.getLogger(__name__)
        self.agent_manager = agent_manager
        self.db = db
        self.cache_dir = cache_dir
        self.logs_dir = logs_dir
        
        # Crear directorios si no existen
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Integradores de estrategia
        self.integrators = {}
        
        # Asignación de capital
        self.capital_allocation = {}
        
        self.logger.info("Gestor de estrategias RL inicializado")
    
    async def create_strategy(self, 
                        strategy_id: str,
                        feature_columns: List[str],
                        window_size: int = 20,
                        position_scaling: float = 1.0,
                        risk_per_trade: float = 0.02,
                        max_position_value: Optional[float] = None) -> str:
        """
        Crear una nueva estrategia RL.
        
        Args:
            strategy_id: Identificador para la estrategia
            feature_columns: Columnas de características para observaciones
            window_size: Tamaño de la ventana para observaciones
            position_scaling: Factor de escala para tamaños de posición
            risk_per_trade: Riesgo por operación (porcentaje del capital)
            max_position_value: Valor máximo de posición (None = sin límite)
            
        Returns:
            ID de la estrategia creada
        """
        # Verificar si la estrategia ya existe
        if strategy_id in self.integrators:
            raise ValueError(f"La estrategia {strategy_id} ya existe")
        
        # Crear integrador de estrategia
        integrator = RLStrategyIntegrator(
            agent_manager=self.agent_manager,
            feature_columns=feature_columns,
            window_size=window_size,
            position_scaling=position_scaling,
            risk_per_trade=risk_per_trade,
            max_position_value=max_position_value,
            cache_dir=os.path.join(self.cache_dir, strategy_id)
        )
        
        # Registrar integrador
        self.integrators[strategy_id] = integrator
        
        # Asignación inicial de capital
        self.capital_allocation[strategy_id] = 0.0
        
        self.logger.info(f"Estrategia {strategy_id} creada")
        
        return strategy_id
    
    async def set_strategy_agent(self, 
                           strategy_id: str, 
                           agent_id: str) -> bool:
        """
        Establecer agente para una estrategia.
        
        Args:
            strategy_id: ID de la estrategia
            agent_id: ID del agente
            
        Returns:
            True si se estableció correctamente
        """
        # Verificar si la estrategia existe
        if strategy_id not in self.integrators:
            raise ValueError(f"Estrategia no encontrada: {strategy_id}")
        
        # Obtener integrador
        integrator = self.integrators[strategy_id]
        
        # Establecer agente activo
        success = await integrator.set_active_agent(agent_id)
        
        if success:
            self.logger.info(f"Agente {agent_id} establecido para estrategia {strategy_id}")
        
        return success
    
    async def allocate_capital(self, 
                         allocation: Dict[str, float],
                         total_capital: Optional[float] = None) -> Dict[str, float]:
        """
        Asignar capital entre estrategias.
        
        Args:
            allocation: Diccionario con porcentajes de asignación por estrategia
            total_capital: Capital total disponible (opcional)
            
        Returns:
            Diccionario con asignación monetaria por estrategia
        """
        # Verificar si todas las estrategias existen
        for strategy_id in allocation:
            if strategy_id not in self.integrators:
                raise ValueError(f"Estrategia no encontrada: {strategy_id}")
        
        # Normalizar porcentajes
        total_percentage = sum(allocation.values())
        
        if total_percentage <= 0:
            raise ValueError("La suma de porcentajes debe ser mayor que 0")
        
        normalized_allocation = {
            k: v / total_percentage for k, v in allocation.items()
        }
        
        # Asignar capital monetario si se proporciona
        if total_capital is not None:
            monetary_allocation = {
                k: v * total_capital for k, v in normalized_allocation.items()
            }
            
            # Actualizar asignación
            self.capital_allocation = monetary_allocation
            
            self.logger.info(f"Capital asignado: {self.capital_allocation}")
            
            return monetary_allocation
        else:
            # Solo actualizar porcentajes
            self.capital_allocation = normalized_allocation
            
            self.logger.info(f"Porcentajes asignados: {self.capital_allocation}")
            
            return normalized_allocation
    
    async def get_trading_decision(self, 
                             strategy_id: str,
                             symbols: Union[str, List[str]],
                             data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Obtener decisión de trading de una estrategia.
        
        Args:
            strategy_id: ID de la estrategia
            symbols: Símbolo o lista de símbolos
            data: DataFrame o diccionario de DataFrames con datos históricos
            
        Returns:
            Diccionario con decisión de trading
        """
        # Verificar si la estrategia existe
        if strategy_id not in self.integrators:
            raise ValueError(f"Estrategia no encontrada: {strategy_id}")
        
        # Obtener integrador
        integrator = self.integrators[strategy_id]
        
        # Obtener balance asignado
        balance = self.capital_allocation.get(strategy_id, 0.0)
        
        # Obtener decisión
        decision = await integrator.get_trading_decision(
            symbols=symbols,
            data=data,
            balance=balance
        )
        
        return decision
    
    async def get_ensemble_decision(self, 
                              symbols: Union[str, List[str]],
                              data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                              strategy_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Obtener decisión combinada de múltiples estrategias.
        
        Args:
            symbols: Símbolo o lista de símbolos
            data: DataFrame o diccionario de DataFrames con datos históricos
            strategy_weights: Pesos para cada estrategia (opcional)
            
        Returns:
            Diccionario con decisión de trading combinada
        """
        if not self.integrators:
            raise ValueError("No hay estrategias registradas")
        
        # Determinar pesos
        if strategy_weights is None:
            # Usar asignación de capital como pesos
            weights = self.capital_allocation
        else:
            # Verificar si todas las estrategias existen
            for strategy_id in strategy_weights:
                if strategy_id not in self.integrators:
                    raise ValueError(f"Estrategia no encontrada: {strategy_id}")
            
            weights = strategy_weights
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        
        if total_weight <= 0:
            raise ValueError("La suma de pesos debe ser mayor que 0")
        
        normalized_weights = {
            k: v / total_weight for k, v in weights.items()
        }
        
        # Obtener decisiones individuales
        decisions = {}
        
        for strategy_id, weight in normalized_weights.items():
            if weight > 0:
                # Obtener decisión
                strategy_decision = await self.get_trading_decision(
                    strategy_id=strategy_id,
                    symbols=symbols,
                    data=data
                )
                
                decisions[strategy_id] = strategy_decision
        
        # Combinar decisiones
        combined_decision = {}
        
        # Determinar si es single o multi-asset
        if isinstance(symbols, str):
            symbols_list = [symbols]
        else:
            symbols_list = symbols
        
        # Procesar cada símbolo
        for symbol in symbols_list:
            # Recopilar señales para este símbolo
            buy_weight = 0.0
            sell_weight = 0.0
            hold_weight = 0.0
            total_amount = 0.0
            
            for strategy_id, decision in decisions.items():
                if symbol in decision:
                    signal = decision[symbol]['signal']
                    strategy_weight = normalized_weights[strategy_id]
                    
                    if signal == 'BUY':
                        buy_weight += strategy_weight
                        total_amount += decision[symbol]['amount'] * strategy_weight
                    elif signal == 'SELL':
                        sell_weight += strategy_weight
                        total_amount += decision[symbol]['amount'] * strategy_weight
                    else:  # HOLD
                        hold_weight += strategy_weight
            
            # Determinar señal final
            if buy_weight > sell_weight and buy_weight > hold_weight:
                final_signal = 'BUY'
                final_amount = total_amount
            elif sell_weight > buy_weight and sell_weight > hold_weight:
                final_signal = 'SELL'
                final_amount = total_amount
            else:
                final_signal = 'HOLD'
                final_amount = 0.0
            
            # Obtener precio del primer decision (debería ser igual en todos)
            price = decisions[list(decisions.keys())[0]][symbol]['price']
            
            # Guardar decisión final
            combined_decision[symbol] = {
                'signal': final_signal,
                'amount': final_amount,
                'price': price,
                'confidence': max(buy_weight, sell_weight, hold_weight)
            }
        
        return combined_decision
    
    async def update_position(self, 
                        strategy_id: str, 
                        symbol: str, 
                        position: float) -> None:
        """
        Actualizar posición actual para una estrategia.
        
        Args:
            strategy_id: ID de la estrategia
            symbol: Símbolo del activo
            position: Nueva posición
        """
        # Verificar si la estrategia existe
        if strategy_id not in self.integrators:
            raise ValueError(f"Estrategia no encontrada: {strategy_id}")
        
        # Obtener integrador
        integrator = self.integrators[strategy_id]
        
        # Actualizar posición
        await integrator.update_position(symbol, position)
    
    async def save_state(self, filepath: Optional[str] = None) -> str:
        """
        Guardar estado del gestor de estrategias.
        
        Args:
            filepath: Ruta donde guardar el estado (opcional)
            
        Returns:
            Ruta donde se guardó el estado
        """
        # Generar ruta si no se proporciona
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.cache_dir, f"strategy_manager_state_{timestamp}.json")
        
        # Guardar estado de cada integrador
        integrator_states = {}
        
        for strategy_id, integrator in self.integrators.items():
            # Crear directorio para cada estrategia
            strategy_dir = os.path.join(self.cache_dir, strategy_id)
            os.makedirs(strategy_dir, exist_ok=True)
            
            # Guardar estado del integrador
            integrator_path = os.path.join(strategy_dir, f"state_{timestamp}.json")
            await integrator.save_state(integrator_path)
            
            integrator_states[strategy_id] = integrator_path
        
        # Preparar estado
        state = {
            'integrator_states': integrator_states,
            'capital_allocation': self.capital_allocation,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar estado
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Estado guardado en {filepath}")
        
        return filepath
    
    async def load_state(self, filepath: str) -> bool:
        """
        Cargar estado del gestor de estrategias.
        
        Args:
            filepath: Ruta desde donde cargar el estado
            
        Returns:
            True si se cargó correctamente
        """
        try:
            # Cargar estado
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restaurar asignación de capital
            self.capital_allocation = state.get('capital_allocation', {})
            
            # Restaurar estado de cada integrador
            integrator_states = state.get('integrator_states', {})
            
            for strategy_id, integrator_path in integrator_states.items():
                # Verificar si el integrador existe
                if strategy_id not in self.integrators:
                    self.logger.warning(f"Estrategia {strategy_id} no encontrada, se omitirá su carga")
                    continue
                
                # Cargar estado del integrador
                await self.integrators[strategy_id].load_state(integrator_path)
            
            self.logger.info(f"Estado cargado desde {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando estado: {str(e)}")
            return False
    
    def get_strategy_stats(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas de una o todas las estrategias.
        
        Args:
            strategy_id: ID de la estrategia (None = todas)
            
        Returns:
            Diccionario con estadísticas
        """
        if strategy_id:
            # Verificar si la estrategia existe
            if strategy_id not in self.integrators:
                raise ValueError(f"Estrategia no encontrada: {strategy_id}")
            
            # Obtener integrador
            integrator = self.integrators[strategy_id]
            
            # Obtener estadísticas
            stats = integrator.get_performance_stats()
            
            # Agregar información adicional
            stats['strategy_id'] = strategy_id
            stats['capital_allocation'] = self.capital_allocation.get(strategy_id, 0.0)
            
            return stats
        else:
            # Estadísticas de todas las estrategias
            all_stats = {}
            
            for s_id, integrator in self.integrators.items():
                # Obtener estadísticas
                stats = integrator.get_performance_stats()
                
                # Agregar información adicional
                stats['strategy_id'] = s_id
                stats['capital_allocation'] = self.capital_allocation.get(s_id, 0.0)
                
                all_stats[s_id] = stats
            
            return all_stats
    
    def plot_strategy_performance(self, 
                                 strategy_id: Optional[str] = None, 
                                 symbol: Optional[str] = None,
                                 save_path: Optional[str] = None) -> str:
        """
        Generar gráfico de rendimiento de estrategia.
        
        Args:
            strategy_id: ID de la estrategia (None = comparar todas)
            symbol: Símbolo específico a graficar (opcional)
            save_path: Ruta donde guardar el gráfico
            
        Returns:
            Imagen en formato base64
        """
        if strategy_id:
            # Gráfico de una estrategia específica
            if strategy_id not in self.integrators:
                raise ValueError(f"Estrategia no encontrada: {strategy_id}")
            
            # Obtener integrador
            integrator = self.integrators[strategy_id]
            
            # Generar gráfico
            return integrator.plot_decision_history(symbol, save_path)
            
        else:
            # Comparación de todas las estrategias
            # Conteo de señales por estrategia
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Datos
            strategy_ids = list(self.integrators.keys())
            buy_counts = []
            sell_counts = []
            hold_counts = []
            
            for s_id in strategy_ids:
                stats = self.integrators[s_id].get_signal_stats()
                buy_counts.append(stats.get('BUY', 0))
                sell_counts.append(stats.get('SELL', 0))
                hold_counts.append(stats.get('HOLD', 0))
            
            # Configurar gráfico de barras apiladas
            bar_width = 0.5
            indices = np.arange(len(strategy_ids))
            
            p1 = ax.bar(indices, buy_counts, bar_width, label='Compra')
            p2 = ax.bar(indices, sell_counts, bar_width, bottom=buy_counts, label='Venta')
            p3 = ax.bar(indices, hold_counts, bar_width, bottom=np.array(buy_counts) + np.array(sell_counts), label='Mantener')
            
            # Formato
            ax.set_title('Distribución de Señales por Estrategia')
            ax.set_xlabel('Estrategia')
            ax.set_ylabel('Conteo')
            ax.set_xticks(indices)
            ax.set_xticklabels(strategy_ids)
            ax.legend()
            
            fig.tight_layout()
            
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
                
                self.logger.info(f"Gráfico guardado en {save_path}")
            
            return img_str