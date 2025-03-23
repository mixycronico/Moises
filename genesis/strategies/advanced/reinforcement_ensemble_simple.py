"""
Estrategia avanzada basada en Reinforcement Learning con ensemble de modelos (versión simplificada).

Esta implementación es una versión simplificada que no depende de las bibliotecas
gymnasium o stable-baselines3, permitiendo el funcionamiento básico sin estas dependencias.

Incorpora además integración con DeepSeek para análisis avanzado y mejora del proceso
de toma de decisiones mediante técnicas de Large Language Models.
"""
import logging
import asyncio
import numpy as np
import random
from typing import Dict, Any, List, Optional, Tuple
import time
import json
import os
from datetime import datetime

# Importar componentes de DeepSeek si están disponibles
try:
    from genesis.lsml.deepseek_integrator import DeepSeekIntegrator
    from genesis.lsml import deepseek_config
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logging.warning("Módulos DeepSeek no encontrados. La estrategia funcionará sin capacidades avanzadas de análisis.")

# Componentes del Sistema Genesis
from genesis.strategies.base import Strategy
from genesis.analysis.advanced_indicators import (
    calculate_ichimoku_cloud,
    calculate_dynamic_bollinger_bands,
    calculate_directional_movement_index
)

# Logger para esta estrategia
logger = logging.getLogger('genesis.strategies.advanced.reinforcement_ensemble_simple')

class ReinforcementEnsembleStrategy(Strategy):
    """
    Estrategia de trading avanzada que simula un ensemble de agentes de RL.
    
    Esta es una versión simplificada que implementa la misma interfaz pero
    sin dependencias de gymnasium o stable-baselines3.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializar estrategia avanzada con ensemble simulado de RL.
        
        Args:
            config: Configuración de la estrategia
        """
        super().__init__(config)
        
        self.name = "Reinforcement Ensemble Strategy"
        self.description = "Estrategia avanzada que combina RL, indicadores técnicos, sentimiento y análisis DeepSeek"
        
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
        self.deepseek_api_key = config.get('deepseek_api_key', None)
        self.deepseek_intelligence_factor = config.get('deepseek_intelligence_factor', 1.0)
        self.deepseek_integrator = None
        
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
        
        # Historial de predicciones (simuladas) de cada agente
        self.agent_predictions_history = []
        
        # Modelo de rendimiento histórico por símbolo (simulado)
        self.symbol_performance = {symbol: {"success_rate": 0.5} for symbol in self.symbols}
        
        # Flags para seguimiento
        self.is_initialized = False
        self.needs_retraining = False
        
        # Modelos simulados
        self.models = {
            "dqn": {"weight": 0.3, "success_rate": 0.6},
            "ppo": {"weight": 0.4, "success_rate": 0.65},
            "sac": {"weight": 0.3, "success_rate": 0.55}
        }
        
        # Inicializar DeepSeek si está disponible
        if self.use_deepseek and DEEPSEEK_AVAILABLE:
            try:
                self.deepseek_integrator = DeepSeekIntegrator(
                    api_key=self.deepseek_api_key,
                    intelligence_factor=self.deepseek_intelligence_factor
                )
                logger.info(f"Integrador DeepSeek inicializado para la estrategia {self.name}")
            except Exception as e:
                logger.warning(f"No se pudo inicializar DeepSeek: {str(e)}. La estrategia funcionará sin análisis avanzado.")
                self.use_deepseek = False
        
        log_msg = f"Estrategia {self.name} (versión simplificada) creada con {len(self.symbols)} símbolos"
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
            # Crear directorios para modelos simulados
            os.makedirs("models", exist_ok=True)
            
            # Simular carga de modelos
            await asyncio.sleep(0.5)  # Simular tiempo de carga
            
            # Guardar configuración
            config_path = os.path.join("models", "ensemble_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "models": self.models,
                    "symbols": self.symbols,
                    "timeframe": self.timeframe,
                    "meta_learning_enabled": self.meta_learning_enabled,
                    "use_deepseek": self.use_deepseek
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
            log_msg = f"Estrategia {self.name} (versión simplificada) inicializada correctamente"
            if self.use_deepseek:
                log_msg += " con integración DeepSeek"
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
        
        Utiliza una combinación de análisis técnico tradicional, ensemble de agentes RL
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
            # Extraer OHLCV
            ohlcv = data.get('ohlcv', {})
            news_data = data.get('news', [])
            
            # Calcular indicadores avanzados
            ichimoku = await self._calculate_ichimoku(symbol, ohlcv)
            bollinger = await self._calculate_bollinger(symbol, ohlcv)
            dmi = await self._calculate_dmi(symbol, ohlcv)
            
            # Análisis técnico simplificado
            analysis_results = self._analyze_indicators(ichimoku, bollinger, dmi)
            
            # Simulación de predicciones de diferentes agentes
            agent_predictions = self._simulate_agent_predictions(
                analysis_results, 
                symbol
            )
            
            # Consenso de agentes
            consensus_signal, confidence = self._get_ensemble_consensus(agent_predictions)
            
            # Evaluar riesgo (simulación simplificada)
            risk_assessment = self._assess_risk_simple(symbol, consensus_signal, confidence, ohlcv)
            
            # Integración con DeepSeek si está disponible
            deepseek_analysis = None
            if self.use_deepseek and DEEPSEEK_AVAILABLE and self.deepseek_integrator:
                # Verificar si DeepSeek está habilitado en la configuración global
                if not deepseek_config.is_enabled():
                    logger.info(f"DeepSeek está desactivado en la configuración global. Usando análisis tradicional para {symbol}")
                else:
                    try:
                        # Preparar datos para análisis DeepSeek
                        market_data = {
                            'symbol': symbol,
                            'ohlcv': ohlcv,
                            'indicators': {
                                'ichimoku': ichimoku,
                                'bollinger': bollinger,
                                'dmi': dmi
                            },
                            'technical_analysis': analysis_results,
                            'agent_predictions': agent_predictions,
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
    
    def _analyze_indicators(self, ichimoku: Dict[str, Any], 
                          bollinger: Dict[str, Any], 
                          dmi: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizar indicadores técnicos y generar señales.
        
        Args:
            ichimoku: Indicadores de Ichimoku
            bollinger: Indicadores de Bollinger
            dmi: Indicadores de DMI/ADX
            
        Returns:
            Resultados del análisis
        """
        signals = {
            "ichimoku": 0,
            "bollinger": 0,
            "dmi": 0
        }
        
        # Analizar Ichimoku
        if ichimoku:
            # Señal basada en cruces y posición respecto a la nube
            if ichimoku.get('tk_cross') == 1 and ichimoku.get('price_above_cloud', False):
                signals["ichimoku"] = 1  # Señal alcista fuerte
            elif ichimoku.get('tk_cross') == 1:
                signals["ichimoku"] = 0.5  # Señal alcista moderada
            elif ichimoku.get('tk_cross') == -1 and not ichimoku.get('price_above_cloud', True):
                signals["ichimoku"] = -1  # Señal bajista fuerte
            elif ichimoku.get('tk_cross') == -1:
                signals["ichimoku"] = -0.5  # Señal bajista moderada
        
        # Analizar Bollinger Bands
        if bollinger:
            # Señal basada en posición respecto a las bandas
            if bollinger.get('signal') == 1:
                signals["bollinger"] = -0.7  # Sobrecompra, posible reversión
            elif bollinger.get('signal') == -1:
                signals["bollinger"] = 0.7  # Sobreventa, posible reversión
            elif bollinger.get('percent_b') is not None:
                pb = bollinger.get('percent_b')[-1] if isinstance(bollinger.get('percent_b'), (list, np.ndarray)) else bollinger.get('percent_b')
                if pb > 0.8:
                    signals["bollinger"] = -0.5  # Acercándose a sobrecompra
                elif pb < 0.2:
                    signals["bollinger"] = 0.5  # Acercándose a sobreventa
                
        # Analizar DMI/ADX
        if dmi:
            trend_strength = dmi.get('trend_strength', 0)
            trend_direction = dmi.get('trend_direction', 0)
            
            # Señal basada en fuerza y dirección de la tendencia
            if trend_strength >= 2:  # Tendencia fuerte
                signals["dmi"] = trend_direction * 0.8
            elif trend_strength >= 1:  # Tendencia moderada
                signals["dmi"] = trend_direction * 0.5
            else:  # Tendencia débil o inexistente
                signals["dmi"] = trend_direction * 0.2
        
        # Calcular señal compuesta
        composite_signal = (
            signals["ichimoku"] * 0.4 + 
            signals["bollinger"] * 0.3 + 
            signals["dmi"] * 0.3
        )
        
        return {
            "signals": signals,
            "composite_signal": composite_signal
        }
    
    def _simulate_agent_predictions(self, analysis_results: Dict[str, Any], 
                                  symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Simular predicciones de diferentes agentes de RL.
        
        Args:
            analysis_results: Resultados del análisis de indicadores
            symbol: Símbolo de trading
            
        Returns:
            Predicciones de cada agente
        """
        composite_signal = analysis_results.get("composite_signal", 0)
        
        # Factor de ajuste específico por símbolo basado en rendimiento histórico
        symbol_factor = (self.symbol_performance.get(symbol, {}).get("success_rate", 0.5) - 0.5) * 2
        
        # Predicciones de cada agente con variaciones
        predictions = {}
        
        # DQN (más sensible a señales fuertes)
        dqn_confidence = min(0.9, max(0.1, 0.5 + (composite_signal * 0.8)))
        dqn_signal = "BUY" if composite_signal > 0.3 else "SELL" if composite_signal < -0.3 else "HOLD"
        
        # Añadir algo de aleatoriedad controlada
        dqn_confidence += (random.random() - 0.5) * 0.1
        if random.random() < 0.1:  # 10% de posibilidad de señal contraria (exploración)
            dqn_signal = "BUY" if dqn_signal != "BUY" else "SELL"
            dqn_confidence = 0.5 + (random.random() - 0.5) * 0.2
        
        predictions["dqn"] = {
            "signal": dqn_signal,
            "confidence": dqn_confidence,
            "weight": self.models["dqn"]["weight"]
        }
        
        # PPO (más equilibrado)
        ppo_confidence = min(0.9, max(0.1, 0.5 + (composite_signal * 0.6)))
        ppo_signal = "BUY" if composite_signal > 0.2 else "SELL" if composite_signal < -0.2 else "HOLD"
        
        # Añadir algo de aleatoriedad controlada
        ppo_confidence += (random.random() - 0.5) * 0.15
        if random.random() < 0.05:  # 5% de posibilidad de señal contraria
            ppo_signal = "BUY" if ppo_signal != "BUY" else "SELL"
            ppo_confidence = 0.5 + (random.random() - 0.5) * 0.1
            
        predictions["ppo"] = {
            "signal": ppo_signal,
            "confidence": ppo_confidence,
            "weight": self.models["ppo"]["weight"]
        }
        
        # SAC (más conservador)
        sac_confidence = min(0.9, max(0.1, 0.5 + (composite_signal * 0.4)))
        sac_signal = "BUY" if composite_signal > 0.4 else "SELL" if composite_signal < -0.4 else "HOLD"
        
        # Añadir algo de aleatoriedad controlada
        sac_confidence += (random.random() - 0.5) * 0.1
        if random.random() < 0.03:  # 3% de posibilidad de señal contraria
            sac_signal = "BUY" if sac_signal != "BUY" else "SELL"
            sac_confidence = 0.5 + (random.random() - 0.5) * 0.05
            
        predictions["sac"] = {
            "signal": sac_signal,
            "confidence": sac_confidence,
            "weight": self.models["sac"]["weight"]
        }
        
        return predictions
    
    def _get_ensemble_consensus(self, agent_predictions: Dict[str, Dict[str, Any]]) -> Tuple[str, float]:
        """
        Obtener consenso del ensemble de agentes.
        
        Args:
            agent_predictions: Predicciones de cada agente
            
        Returns:
            Tupla (señal de consenso, confianza)
        """
        # Contar votos ponderados
        weighted_votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        total_weight = 0
        
        for agent_id, prediction in agent_predictions.items():
            signal = prediction.get("signal", "HOLD")
            confidence = prediction.get("confidence", 0.5)
            weight = prediction.get("weight", 1.0)
            
            weighted_votes[signal] += weight * confidence
            total_weight += weight
        
        # Normalizar votos
        if total_weight > 0:
            for signal in weighted_votes:
                weighted_votes[signal] /= total_weight
        
        # Encontrar señal ganadora
        max_signal = max(weighted_votes.items(), key=lambda x: x[1])
        winning_signal = max_signal[0]
        confidence = max_signal[1]
        
        # Verificar si supera el umbral
        if confidence < self.voting_threshold:
            return "HOLD", confidence
        
        return winning_signal, confidence
    
    def _assess_risk_simple(self, symbol: str, signal: str, confidence: float, 
                         ohlcv: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Evaluación simplificada de riesgo.
        
        Args:
            symbol: Símbolo de trading
            signal: Señal de trading
            confidence: Confianza en la señal
            ohlcv: Datos OHLCV
            
        Returns:
            Evaluación de riesgo
        """
        if signal == "HOLD" or signal == "NONE":
            return {
                'risk_level': 'NONE',
                'max_position_size': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0
            }
        
        # Calcular volatilidad reciente
        close_prices = ohlcv.get('close', [])
        if len(close_prices) > 20:
            recent_close = np.array(close_prices[-20:])
            volatility = np.std(recent_close) / np.mean(recent_close)
        else:
            volatility = 0.02  # Valor por defecto
        
        # Ajustar posición según volatilidad y confianza
        position_size = self.risk_per_trade * confidence * (1.0 - (volatility * 10))
        position_size = max(0.01, min(position_size, self.risk_per_trade * 1.5))
        
        # Calcular stop loss y take profit
        stop_loss_pct = volatility * 2.0  # 2x la volatilidad
        take_profit_pct = stop_loss_pct * 1.5  # Ratio riesgo:recompensa de 1:1.5
        
        # Ajustar según la señal
        if signal == "SELL":
            stop_loss_pct, take_profit_pct = take_profit_pct, stop_loss_pct
        
        # Determinar nivel de riesgo
        if volatility > 0.03:
            risk_level = "HIGH"
        elif volatility > 0.015:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'risk_level': risk_level,
            'max_position_size': position_size,
            'stop_loss': stop_loss_pct,
            'take_profit': take_profit_pct,
            'volatility': volatility
        }
    
    def _adjust_risk_with_deepseek(self, 
                           risk_assessment: Dict[str, Any],
                           enhanced_stop_loss: Optional[float] = None,
                           enhanced_take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Ajustar la evaluación de riesgo con recomendaciones de DeepSeek.
        
        Args:
            risk_assessment: Evaluación de riesgo original
            enhanced_stop_loss: Nivel de stop loss mejorado por DeepSeek (opcional)
            enhanced_take_profit: Nivel de take profit mejorado por DeepSeek (opcional)
            
        Returns:
            Evaluación de riesgo ajustada
        """
        # Crear una copia para no modificar el original directamente
        adjusted_risk = risk_assessment.copy()
        
        # Aplicar ajustes si están disponibles
        if enhanced_stop_loss is not None:
            adjusted_risk['stop_loss'] = enhanced_stop_loss
            logger.info(f"Stop loss ajustado a {enhanced_stop_loss} por recomendación de DeepSeek")
            
        if enhanced_take_profit is not None:
            adjusted_risk['take_profit'] = enhanced_take_profit
            logger.info(f"Take profit ajustado a {enhanced_take_profit} por recomendación de DeepSeek")
            
        # Si se ajustaron los niveles, posiblemente ajustar también el tamaño de posición
        if enhanced_stop_loss is not None and enhanced_take_profit is not None:
            # Calcular nuevo ratio riesgo/recompensa
            original_risk = risk_assessment.get('stop_loss', 0)
            original_reward = risk_assessment.get('take_profit', 0)
            
            if original_risk > 0 and original_reward > 0:
                original_rr_ratio = original_reward / original_risk
                new_rr_ratio = enhanced_take_profit / enhanced_stop_loss
                
                # Si el nuevo ratio es mejor, podemos ajustar el tamaño
                if new_rr_ratio > original_rr_ratio:
                    # Ajustar tamaño de posición manteniendo el mismo riesgo absoluto
                    position_adjustment = min(1.2, max(0.8, new_rr_ratio / original_rr_ratio))
                    adjusted_risk['max_position_size'] = risk_assessment.get('max_position_size', 0) * position_adjustment
                    logger.info(f"Tamaño de posición ajustado por factor {position_adjustment:.2f} debido a mejor ratio riesgo/recompensa")
        
        return adjusted_risk
    
    def _make_final_decision(self, prediction: Dict[str, Any], 
                           risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tomar decisión final combinando predicción y evaluación de riesgo.
        
        Args:
            prediction: Predicción del ensemble de RL
            risk_assessment: Evaluación de riesgo
            
        Returns:
            Decisión final con todos los metadatos
        """
        signal = prediction.get('signal', 'NONE')
        confidence = prediction.get('confidence', 0.5)
        
        risk_level = risk_assessment.get('risk_level', 'MEDIUM')
        position_size = risk_assessment.get('max_position_size', 0.0)
        
        # Ajustar según nivel de riesgo
        if risk_level == "HIGH" and confidence < 0.75:
            # Reducir exposición si riesgo alto y confianza no es muy alta
            position_size *= 0.5
            if confidence < 0.6:
                signal = "HOLD"  # No operar si confianza baja en entorno de riesgo alto
        
        # Calcular factores técnicos
        stop_loss = risk_assessment.get('stop_loss', 0.02)
        take_profit = risk_assessment.get('take_profit', 0.04)
        
        # Construir decisión final con todos los metadatos
        final_decision = {
            'signal': signal,
            'confidence': confidence,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_level': risk_level,
            'timestamp': datetime.now().timestamp(),
            'analysis': {
                'prediction': prediction,
                'risk_assessment': risk_assessment
            }
        }
        
        return final_decision
    
    def _update_metrics(self, decision: Dict[str, Any]) -> None:
        """
        Actualizar métricas de rendimiento con nueva decisión.
        
        Args:
            decision: Decisión tomada
        """
        # En un sistema real, esto se actualizaría con los resultados reales de trading
        # Aquí solo lo simulamos
        
        if decision.get('signal') in ['BUY', 'SELL']:
            self.performance_metrics['total_trades'] += 1
            
            # Simular éxito/fracaso basado en confianza
            confidence = decision.get('confidence', 0.5)
            
            # Calcular probabilidad de éxito basado en confianza
            success_prob = 0.3 + (confidence * 0.5)  # Entre 30% y 80%
            
            if random.random() < success_prob:
                self.performance_metrics['winning_trades'] += 1
                
                # Simular ganancia
                profit_factor = 1.0 + (random.random() * 0.5)  # Entre 1.0 y 1.5
                profit = decision.get('take_profit', 0.04) * profit_factor
                
                # Actualizar métricas
                self.performance_metrics['current_capital'] *= (1.0 + (profit * decision.get('position_size', 0.0)))
            else:
                self.performance_metrics['losing_trades'] += 1
                
                # Simular pérdida
                loss_factor = 0.7 + (random.random() * 0.3)  # Entre 0.7 y 1.0
                loss = decision.get('stop_loss', 0.02) * loss_factor
                
                # Actualizar métricas
                self.performance_metrics['current_capital'] *= (1.0 - (loss * decision.get('position_size', 0.0)))
            
            # Actualizar métricas agregadas
            total_trades = self.performance_metrics['total_trades']
            if total_trades > 0:
                self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / total_trades
                
                # Calcular rendimiento promedio (simulado)
                pnl_ratio = self.performance_metrics['current_capital'] / self.initial_capital - 1.0
                self.performance_metrics['average_profit'] = pnl_ratio / max(1, total_trades)
    
    async def optimize_allocation(self) -> Dict[str, Any]:
        """
        Optimizar la asignación de capital entre instrumentos.
        
        Returns:
            Resultado de la optimización
        """
        # Simulación de optimización
        await asyncio.sleep(0.2)  # Simular proceso
        
        optimized_allocation = {}
        remaining_allocation = 1.0
        
        # Asignar según rendimiento simulado
        for symbol in self.symbols:
            # Asignar entre 10% y 50% según rendimiento pasado
            symbol_success = self.symbol_performance.get(symbol, {}).get("success_rate", 0.5)
            allocation = 0.1 + (symbol_success - 0.5) * 0.8
            allocation = min(allocation, remaining_allocation)
            remaining_allocation -= allocation
            
            optimized_allocation[symbol] = allocation
        
        # Distribuir cualquier remanente
        if remaining_allocation > 0 and self.symbols:
            for symbol in self.symbols:
                optimized_allocation[symbol] += remaining_allocation / len(self.symbols)
        
        return {
            "success": True,
            "allocation": optimized_allocation,
            "timestamp": datetime.now().timestamp()
        }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de rendimiento de la estrategia.
        
        Returns:
            Métricas de rendimiento
        """
        return {
            "success": True,
            "metrics": self.performance_metrics,
            "timestamp": datetime.now().timestamp()
        }
    
    def _get_current_timestamp(self) -> float:
        """Obtener timestamp actual."""
        return datetime.now().timestamp()