"""
Estrategia avanzada basada en Reinforcement Learning con ensemble de modelos.

Esta estrategia integra múltiples agentes de Reinforcement Learning (DQN, PPO, SAC)
con indicadores técnicos avanzados (Ichimoku, Bollinger Bands) y análisis de sentimiento
para tomar decisiones de trading óptimas.
"""
import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Componentes del Sistema Genesis
from genesis.strategies.base import Strategy
from genesis.reinforcement_learning.agents import (
    RLAgentFactory, DQNAgent, PPOAgent, SACAgent, AgentConfig, 
    MultiAgentEnsemble, AgentType
)
from genesis.reinforcement_learning.environments import TradingEnv, EnvironmentConfig
from genesis.reinforcement_learning.integrator import RLIntegrator
from genesis.analysis.advanced_indicators import (
    calculate_ichimoku_cloud,
    calculate_dynamic_bollinger_bands,
    calculate_directional_movement_index
)
from genesis.data.sentiment_analyzer import SentimentAnalyzer
from genesis.data.onchain_analyzer import OnChainAnalyzer
from genesis.simulation.monte_carlo_advanced import MonteCarloAdvancedSimulator

# Logger para esta estrategia
logger = logging.getLogger('genesis.strategies.advanced.reinforcement_ensemble')

class ReinforcementEnsembleStrategy(Strategy):
    """
    Estrategia de trading avanzada que utiliza un ensemble de agentes de RL.
    
    Esta estrategia combina:
    1. Múltiples agentes de RL (DQN, PPO, SAC)
    2. Indicadores técnicos avanzados (Ichimoku, Bollinger Bands, DMI)
    3. Análisis de sentimiento de mercado
    4. Datos on-chain para criptomonedas
    5. Simulaciones de Monte Carlo para gestión de riesgos
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializar estrategia avanzada con ensemble de RL.
        
        Args:
            config: Configuración de la estrategia
        """
        super().__init__(config)
        
        self.name = "Reinforcement Ensemble Strategy"
        self.description = "Estrategia avanzada que combina RL, indicadores técnicos y sentimiento"
        
        # Parámetros de configuración
        self.symbols = config.get('symbols', ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
        self.timeframe = config.get('timeframe', '1h')
        self.lookback_period = config.get('lookback_period', 100)
        self.initial_capital = config.get('initial_capital', 10000.0)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% del capital
        self.meta_learning_enabled = config.get('meta_learning_enabled', True)
        self.voting_threshold = config.get('voting_threshold', 0.67)  # 2/3 para consenso
        
        # Inicializar componentes
        self.integrator = None
        self.ensemble = None
        self.sentiment_analyzer = None
        self.onchain_analyzer = None
        self.monte_carlo = None
        
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
        
        # Flags para seguimiento
        self.is_initialized = False
        self.needs_retraining = False
        
        logger.info(f"Estrategia {self.name} creada con {len(self.symbols)} símbolos")
    
    async def initialize(self) -> bool:
        """
        Inicializar todos los componentes de la estrategia.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Inicializar analizador de sentimiento
            self.sentiment_analyzer = SentimentAnalyzer()
            await self.sentiment_analyzer.initialize()
            
            # Inicializar analizador on-chain
            self.onchain_analyzer = OnChainAnalyzer()
            await self.onchain_analyzer.initialize()
            
            # Inicializar simulador de Monte Carlo
            self.monte_carlo = MonteCarloAdvancedSimulator(
                initial_capital=self.initial_capital,
                risk_per_trade=self.risk_per_trade,
                confidence_level=0.95
            )
            
            # Configurar entorno de trading
            env_config = EnvironmentConfig(
                symbols=self.symbols,
                timeframe=self.timeframe,
                lookback_period=self.lookback_period,
                initial_balance=self.initial_capital,
                commission=0.001,  # 0.1%
                use_sentiment=True,
                use_onchain=True
            )
            
            # Inicializar fábrica de agentes y crear ensemble
            agent_factory = RLAgentFactory()
            
            # Configuraciones para diferentes agentes
            dqn_config = AgentConfig(
                agent_type=AgentType.DQN,
                learning_rate=0.0001,
                gamma=0.99,
                batch_size=64,
                update_target_every=100,
                hidden_layers=[128, 64],
                model_path="models/dqn_trading"
            )
            
            ppo_config = AgentConfig(
                agent_type=AgentType.PPO,
                learning_rate=0.0003,
                gamma=0.99,
                clip_ratio=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                model_path="models/ppo_trading"
            )
            
            sac_config = AgentConfig(
                agent_type=AgentType.SAC,
                learning_rate=0.0003,
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                model_path="models/sac_trading"
            )
            
            # Crear ensemble de agentes
            self.ensemble = MultiAgentEnsemble(
                agent_configs=[dqn_config, ppo_config, sac_config],
                voting_threshold=self.voting_threshold,
                meta_learning_enabled=self.meta_learning_enabled
            )
            
            # Inicializar integrador de RL
            self.integrator = RLIntegrator(
                environment_config=env_config,
                ensemble=self.ensemble
            )
            
            await self.integrator.initialize()
            
            # Cargar modelos pre-entrenados si existen
            await self.load_models()
            
            self.is_initialized = True
            logger.info(f"Estrategia {self.name} inicializada correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar estrategia {self.name}: {str(e)}")
            import traceback
            logger.debug(f"Detalle del error: {traceback.format_exc()}")
            return False
    
    async def load_models(self) -> bool:
        """
        Cargar modelos pre-entrenados para los agentes.
        
        Returns:
            True si se cargaron correctamente
        """
        try:
            result = await self.ensemble.load_models()
            logger.info(f"Modelos cargados: {result}")
            return result
        except Exception as e:
            logger.warning(f"No se pudieron cargar modelos pre-entrenados: {str(e)}")
            logger.info("Se utilizarán modelos nuevos")
            return False
    
    async def process_tick(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar un nuevo tick de datos y generar señales de trading.
        
        Args:
            tick_data: Datos del tick actual
            
        Returns:
            Resultado del procesamiento con señales
        """
        if not self.is_initialized:
            logger.warning("Estrategia no inicializada, inicializando...")
            await self.initialize()
        
        symbol = tick_data.get('symbol')
        if not symbol or symbol not in self.symbols:
            return {'signal': 'NONE', 'reason': 'Symbol not supported'}
        
        try:
            # Extraer OHLCV
            ohlcv = tick_data.get('ohlcv', {})
            
            # Calcular indicadores avanzados
            ichimoku = await self._calculate_ichimoku(symbol, ohlcv)
            bollinger = await self._calculate_bollinger(symbol, ohlcv)
            dmi = await self._calculate_dmi(symbol, ohlcv)
            
            # Obtener sentimiento del mercado
            sentiment = await self.sentiment_analyzer.analyze_sentiment(symbol)
            
            # Obtener datos on-chain para criptomonedas
            onchain_data = await self.onchain_analyzer.get_metrics(symbol.split('/')[0])
            
            # Preparar estado actual para el entorno RL
            current_state = {
                'ohlcv': ohlcv,
                'ichimoku': ichimoku,
                'bollinger': bollinger,
                'dmi': dmi,
                'sentiment': sentiment,
                'onchain': onchain_data
            }
            
            # Obtener predicción del ensemble de RL
            prediction = await self.integrator.predict(current_state)
            
            # Guardar predicciones individuales para análisis
            self.agent_predictions_history.append({
                'timestamp': tick_data.get('timestamp'),
                'symbol': symbol,
                'predictions': prediction.get('agent_predictions', {})
            })
            
            # Evaluar riesgo con Monte Carlo
            risk_assessment = await self._assess_risk(symbol, prediction, current_state)
            
            # Ajustar predicción según riesgo
            final_decision = self._make_final_decision(prediction, risk_assessment)
            
            # Registrar para análisis posterior
            self._update_metrics(final_decision)
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Error al procesar tick para {symbol}: {str(e)}")
            import traceback
            logger.debug(f"Detalle del error: {traceback.format_exc()}")
            return {'signal': 'ERROR', 'reason': str(e)}
    
    async def _calculate_ichimoku(self, symbol: str, ohlcv: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calcular indicador Ichimoku Cloud."""
        try:
            # Convertir OHLCV a formato adecuado
            high = ohlcv.get('high', [])
            low = ohlcv.get('low', [])
            close = ohlcv.get('close', [])
            
            # Parámetros personalizados para Ichimoku
            tenkan_period = 9
            kijun_period = 26
            senkou_span_b_period = 52
            displacement = 26
            
            ichimoku = calculate_ichimoku_cloud(
                high=np.array(high, dtype=float),
                low=np.array(low, dtype=float),
                close=np.array(close, dtype=float),
                tenkan_period=tenkan_period,
                kijun_period=kijun_period,
                senkou_span_b_period=senkou_span_b_period,
                displacement=displacement
            )
            
            return ichimoku
        except Exception as e:
            logger.warning(f"Error al calcular Ichimoku para {symbol}: {str(e)}")
            return {}
    
    async def _calculate_bollinger(self, symbol: str, ohlcv: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calcular Bandas de Bollinger dinámicas."""
        try:
            close = ohlcv.get('close', [])
            
            # Parámetros personalizados para Bollinger
            window = 20
            num_std_dev = 2.0
            
            bollinger = calculate_dynamic_bollinger_bands(
                close=np.array(close, dtype=float),
                window=window,
                num_std_dev=num_std_dev
            )
            
            return bollinger
        except Exception as e:
            logger.warning(f"Error al calcular Bollinger para {symbol}: {str(e)}")
            return {}
    
    async def _calculate_dmi(self, symbol: str, ohlcv: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calcular Índice de Movimiento Direccional (DMI/ADX)."""
        try:
            high = ohlcv.get('high', [])
            low = ohlcv.get('low', [])
            close = ohlcv.get('close', [])
            
            # Parámetros personalizados para DMI
            period = 14
            
            dmi = calculate_directional_movement_index(
                high=np.array(high, dtype=float),
                low=np.array(low, dtype=float),
                close=np.array(close, dtype=float),
                period=period
            )
            
            return dmi
        except Exception as e:
            logger.warning(f"Error al calcular DMI para {symbol}: {str(e)}")
            return {}
    
    async def _assess_risk(self, symbol: str, prediction: Dict[str, Any], 
                          current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluar riesgo de la operación usando Monte Carlo.
        
        Args:
            symbol: Símbolo de trading
            prediction: Predicción del ensemble de RL
            current_state: Estado actual del mercado
            
        Returns:
            Evaluación de riesgo
        """
        try:
            # Extraer señal y probabilidad
            signal = prediction.get('signal', 'NONE')
            probability = prediction.get('confidence', 0.5)
            
            # Si no hay señal, no hay riesgo que evaluar
            if signal == 'NONE':
                return {
                    'risk_level': 'NONE',
                    'max_position_size': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0
                }
            
            # Preparar datos para simulación
            price_data = np.array(current_state['ohlcv'].get('close', []), dtype=float)
            
            # Ejecutar simulación de Monte Carlo
            simulation_result = await self.monte_carlo.run_simulation(
                price_data=price_data,
                signal=signal,
                confidence=probability,
                current_price=price_data[-1] if len(price_data) > 0 else 0.0,
                symbol=symbol
            )
            
            # Extraer recomendaciones de la simulación
            position_size = simulation_result.get('recommended_position_size', 0.0)
            stop_loss = simulation_result.get('recommended_stop_loss', 0.0)
            take_profit = simulation_result.get('recommended_take_profit', 0.0)
            risk_level = simulation_result.get('risk_assessment', 'MEDIUM')
            
            return {
                'risk_level': risk_level,
                'max_position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'simulation_details': simulation_result
            }
            
        except Exception as e:
            logger.warning(f"Error al evaluar riesgo para {symbol}: {str(e)}")
            return {
                'risk_level': 'HIGH',  # Por defecto, asumir riesgo alto si hay error
                'max_position_size': self.risk_per_trade * 0.5,  # Reducir tamaño por seguridad
                'stop_loss': 0.02,  # 2% por defecto
                'take_profit': 0.04  # 4% por defecto
            }
    
    def _make_final_decision(self, prediction: Dict[str, Any], 
                           risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tomar decisión final combinando predicción RL y evaluación de riesgo.
        
        Args:
            prediction: Predicción del ensemble de RL
            risk_assessment: Evaluación de riesgo de Monte Carlo
            
        Returns:
            Decisión final
        """
        signal = prediction.get('signal', 'NONE')
        symbol = prediction.get('symbol', '')
        confidence = prediction.get('confidence', 0.0)
        
        # Extraer evaluación de riesgo
        risk_level = risk_assessment.get('risk_level', 'HIGH')
        max_position_size = risk_assessment.get('max_position_size', 0.0)
        stop_loss = risk_assessment.get('stop_loss', 0.0)
        take_profit = risk_assessment.get('take_profit', 0.0)
        
        # Ajustar señal según riesgo
        adjusted_signal = signal
        reason = prediction.get('reason', '')
        
        # Si es señal de compra/venta pero riesgo muy alto, reducir a neutral
        if signal in ['BUY', 'SELL'] and risk_level == 'EXTREME':
            adjusted_signal = 'NONE'
            reason += f" | Señal cancelada por riesgo extremo ({risk_level})"
        
        # Si es señal de compra/venta pero confianza baja, reducir tamaño
        elif signal in ['BUY', 'SELL'] and confidence < 0.7:
            max_position_size *= (confidence / 0.7)  # Reducir proporcionalmente
            reason += f" | Tamaño reducido por confianza baja ({confidence:.2f})"
        
        # Construir decisión final
        final_decision = {
            'symbol': symbol,
            'signal': adjusted_signal,
            'original_signal': signal,
            'confidence': confidence,
            'risk_level': risk_level,
            'position_size': max_position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason,
            'timestamp': prediction.get('timestamp')
        }
        
        return final_decision
    
    def _update_metrics(self, decision: Dict[str, Any]) -> None:
        """
        Actualizar métricas de rendimiento con la última decisión.
        
        Args:
            decision: Decisión final tomada
        """
        # Por ahora solo registramos la decisión
        # En implementación real, se actualizarían métricas basadas en resultados
        signal = decision.get('signal', 'NONE')
        if signal in ['BUY', 'SELL']:
            self.performance_metrics['total_trades'] += 1
    
    async def rebalance_portfolio(self) -> Dict[str, Any]:
        """
        Rebalancear cartera basado en las últimas predicciones y rendimiento.
        
        Returns:
            Resultado del rebalanceo
        """
        try:
            # Recolectar rendimiento de cada símbolo
            symbol_performance = {}
            for symbol in self.symbols:
                # Simplificado - en producción se usarían métricas reales
                symbol_performance[symbol] = {
                    'return': 0.0,
                    'volatility': 0.0,
                    'sharpe': 0.0
                }
            
            # Optimizar asignación de capital
            if self.integrator:
                allocation = await self.integrator.optimize_allocation(symbol_performance)
                
                return {
                    'success': True,
                    'new_allocation': allocation,
                    'timestamp': self._get_current_timestamp()
                }
            else:
                return {
                    'success': False,
                    'reason': 'Integrador RL no inicializado',
                    'timestamp': self._get_current_timestamp()
                }
                
        except Exception as e:
            logger.error(f"Error al rebalancear cartera: {str(e)}")
            return {
                'success': False,
                'reason': str(e),
                'timestamp': self._get_current_timestamp()
            }
    
    async def train_models(self, force: bool = False) -> Dict[str, Any]:
        """
        Entrenar o reentrenar modelos de RL.
        
        Args:
            force: Si True, forzar reentrenamiento aunque no sea necesario
            
        Returns:
            Resultado del entrenamiento
        """
        if not self.needs_retraining and not force:
            logger.info("No es necesario reentrenar modelos")
            return {'success': True, 'message': 'No se requiere reentrenamiento'}
        
        try:
            # Configurar parámetros de entrenamiento
            training_params = {
                'episodes': 100,
                'evaluation_episodes': 10,
                'save_best_only': True,
                'tensorboard_log': True
            }
            
            # Iniciar entrenamiento
            logger.info("Iniciando entrenamiento de modelos RL")
            result = await self.integrator.train_ensemble(training_params)
            
            # Resetear flag
            self.needs_retraining = False
            
            return {
                'success': True,
                'training_results': result,
                'message': 'Entrenamiento completado exitosamente'
            }
            
        except Exception as e:
            logger.error(f"Error durante entrenamiento: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    async def evaluate_strategy_performance(self) -> Dict[str, Any]:
        """
        Evaluar rendimiento general de la estrategia.
        
        Returns:
            Métricas de rendimiento
        """
        # Actualizar métricas actuales
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades']
            )
        
        # En implementación real, calcularía Sharpe, Drawdown, etc.
        
        return self.performance_metrics
    
    def _get_current_timestamp(self) -> float:
        """Obtener timestamp actual."""
        import time
        return time.time()

    async def shutdown(self) -> bool:
        """
        Cerrar la estrategia y liberar recursos.
        
        Returns:
            True si se cerró correctamente
        """
        try:
            # Guardar estado actual para futura recuperación
            if self.integrator:
                await self.integrator.save_state()
            
            # Liberar recursos
            if self.ensemble:
                await self.ensemble.close()
            
            if self.monte_carlo:
                self.monte_carlo.close()
            
            logger.info(f"Estrategia {self.name} cerrada correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al cerrar estrategia {self.name}: {str(e)}")
            return False