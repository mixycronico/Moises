"""
Módulo de Análisis para el Pipeline de Genesis.

Este módulo se encarga de analizar los datos procesados y generar señales
de trading con capacidades trascendentales de alta precisión.
"""
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta

from genesis.base import GenesisComponent, GenesisSingleton, validate_mode
from genesis.db.transcendental_database import TranscendentalDatabase

# Configuración de logging
logger = logging.getLogger("genesis.pipeline.analysis")

# Enumeración de tipos de señal
class SignalType:
    """Tipos de señales de trading."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"
    CLOSE = "close"
    
    @staticmethod
    def all_types() -> List[str]:
        """Obtener todos los tipos de señal."""
        return [SignalType.BUY, SignalType.SELL, 
                SignalType.HOLD, SignalType.EXIT, 
                SignalType.CLOSE]

class Analyzer(GenesisComponent):
    """Analizador base con capacidades trascendentales."""
    
    def __init__(self, analyzer_id: str, analyzer_name: str, mode: str = "SINGULARITY_V4"):
        """
        Inicializar analizador.
        
        Args:
            analyzer_id: Identificador único del analizador
            analyzer_name: Nombre descriptivo
            mode: Modo trascendental
        """
        super().__init__(f"analyzer_{analyzer_id}", mode)
        self.analyzer_id = analyzer_id
        self.analyzer_name = analyzer_name
        self.last_analysis = 0
        self.analysis_count = 0
        self.db = TranscendentalDatabase()
        
        logger.info(f"Analizador {analyzer_name} ({analyzer_id}) inicializado")
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizar datos.
        
        Args:
            data: Datos a analizar
            
        Returns:
            Datos analizados con señales
        """
        raise NotImplementedError("Las subclases deben implementar analyze")
    
    async def analyze_with_resilience(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizar datos con mecanismos de resiliencia.
        
        Args:
            data: Datos a analizar
            
        Returns:
            Datos analizados con señales o datos originales en caso de error
        """
        self.analysis_count += 1
        self.last_analysis = time.time()
        
        try:
            start_time = time.time()
            result = await self.analyze(data)
            analysis_time = time.time() - start_time
            
            # Actualizar métricas
            self.update_metric("analysis_time", analysis_time)
            self.update_metric("success_rate", 1.0 - (self.error_count / max(1, self.operation_count)))
            
            logger.debug(f"Análisis {self.analyzer_id} exitoso en {analysis_time:.3f}s")
            self.register_operation(True)
            return result
        
        except Exception as e:
            self.register_operation(False)
            logger.error(f"Error en analizador {self.analyzer_id}: {str(e)}")
            
            # En caso de error, devolver los datos originales
            return data

class TechnicalAnalyzer(Analyzer):
    """Analizador técnico con capacidades trascendentales."""
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar analizador técnico.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("technical", "Análisis Técnico", mode)
        self.signal_thresholds = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_signal": 0.0,
            "bb_position_low": 0.2,
            "bb_position_high": 0.8,
            "volume_surge": 2.0  # Multiplicador sobre el promedio
        }
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar señales técnicas basadas en indicadores.
        
        Args:
            data: Datos a analizar
            
        Returns:
            Datos con señales técnicas
        """
        analyzed_data = data.copy()
        
        # Inicializar estructura de señales si no existe
        if "signals" not in analyzed_data:
            analyzed_data["signals"] = {}
        
        # Obtener datos de mercado
        market_data = analyzed_data.get("market_data", {})
        
        # Generar señales para cada símbolo
        for symbol, symbol_data in market_data.items():
            symbol_signals = self._generate_signals_for_symbol(symbol, symbol_data)
            analyzed_data["signals"][symbol] = symbol_signals
        
        # Agregar timestamp de análisis
        analyzed_data["technical_analysis_timestamp"] = time.time()
        
        return analyzed_data
    
    def _generate_signals_for_symbol(self, symbol: str, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar señales técnicas para un símbolo específico.
        
        Args:
            symbol: Símbolo a analizar
            symbol_data: Datos del símbolo
            
        Returns:
            Señales técnicas
        """
        # Estructuras para almacenar señales
        signals = {
            "technical": {
                "indicators": {},
                "combined": None,
                "strength": 0,
                "timestamp": time.time()
            }
        }
        
        # Extraer datos técnicos si existen
        technical_signals = symbol_data.get("technical_signals", {})
        latest_candle = symbol_data.get("latest", {})
        
        # 1. Procesar señales de RSI
        if "rsi" in technical_signals:
            rsi_signal = technical_signals["rsi"]
            rsi_value = latest_candle.get("rsi", 50)
            
            signals["technical"]["indicators"]["rsi"] = {
                "signal": rsi_signal,
                "value": rsi_value,
                "timestamp": time.time()
            }
            
            # Ajustar fuerza según el valor de RSI
            if rsi_signal == "oversold":
                # Más extremo = señal más fuerte
                strength = (self.signal_thresholds["rsi_oversold"] - rsi_value) / 10
                signals["technical"]["strength"] += min(max(strength, 0), 1)
            elif rsi_signal == "overbought":
                strength = (rsi_value - self.signal_thresholds["rsi_overbought"]) / 10
                signals["technical"]["strength"] -= min(max(strength, 0), 1)
        
        # 2. Procesar señales de MACD
        if "macd" in technical_signals:
            macd_signal = technical_signals["macd"]
            macd_value = latest_candle.get("macd", 0)
            macd_hist = latest_candle.get("macd_hist", 0)
            
            signals["technical"]["indicators"]["macd"] = {
                "signal": macd_signal,
                "value": macd_value,
                "histogram": macd_hist,
                "timestamp": time.time()
            }
            
            # Ajustar fuerza según magnitud del histograma
            if macd_signal == "bullish":
                strength = min(abs(macd_hist) / 10, 1)
                signals["technical"]["strength"] += strength
            elif macd_signal == "bearish":
                strength = min(abs(macd_hist) / 10, 1)
                signals["technical"]["strength"] -= strength
        
        # 3. Procesar señales de Bandas de Bollinger
        if "bbands" in technical_signals:
            bb_signal = technical_signals["bbands"]
            bb_position = latest_candle.get("bb_position", 0.5)
            bb_width = latest_candle.get("bb_width", 0.1)
            
            signals["technical"]["indicators"]["bbands"] = {
                "signal": bb_signal,
                "position": bb_position,
                "width": bb_width,
                "timestamp": time.time()
            }
            
            # Ajustar fuerza según posición en la banda
            if bb_signal == "oversold":
                strength = (self.signal_thresholds["bb_position_low"] - bb_position) / self.signal_thresholds["bb_position_low"]
                signals["technical"]["strength"] += min(max(strength, 0), 1)
            elif bb_signal == "overbought":
                strength = (bb_position - self.signal_thresholds["bb_position_high"]) / (1 - self.signal_thresholds["bb_position_high"])
                signals["technical"]["strength"] -= min(max(strength, 0), 1)
        
        # 4. Procesar señales de EMA Cross
        if "ema_cross" in technical_signals:
            ema_signal = technical_signals["ema_cross"]
            ema_changed = latest_candle.get("ema_cross_changed", 0)
            
            signals["technical"]["indicators"]["ema_cross"] = {
                "signal": ema_signal,
                "changed": bool(ema_changed),
                "timestamp": time.time()
            }
            
            # Cruce reciente = señal más fuerte
            if ema_changed:
                if ema_signal == "bullish":
                    signals["technical"]["strength"] += 0.7
                elif ema_signal == "bearish":
                    signals["technical"]["strength"] -= 0.7
            else:
                # Tendencia establecida
                if ema_signal == "bullish":
                    signals["technical"]["strength"] += 0.3
                elif ema_signal == "bearish":
                    signals["technical"]["strength"] -= 0.3
        
        # 5. Procesar señales de volumen
        if "volume" in latest_candle and "volume_sma" in latest_candle:
            volume = latest_candle["volume"]
            volume_sma = latest_candle["volume_sma"]
            volume_ratio = volume / volume_sma if volume_sma > 0 else 1.0
            
            volume_signal = "neutral"
            if volume_ratio > self.signal_thresholds["volume_surge"]:
                volume_signal = "surge"
            
            signals["technical"]["indicators"]["volume"] = {
                "signal": volume_signal,
                "ratio": volume_ratio,
                "timestamp": time.time()
            }
            
            # Volumen alto potencia la señal actual
            if volume_signal == "surge":
                # Amplificar la fuerza actual, no cambiar dirección
                current_strength = signals["technical"]["strength"]
                amplifier = min(volume_ratio / self.signal_thresholds["volume_surge"], 2.0)
                signals["technical"]["strength"] = current_strength * amplifier
        
        # Determinar señal combinada basada en fuerza acumulada
        strength = signals["technical"]["strength"]
        if strength > 0.5:
            signals["technical"]["combined"] = SignalType.BUY
        elif strength < -0.5:
            signals["technical"]["combined"] = SignalType.SELL
        else:
            signals["technical"]["combined"] = SignalType.HOLD
        
        # Normalizar fuerza entre -1 y 1
        signals["technical"]["strength"] = max(min(strength, 1.0), -1.0)
        
        return signals

class SentimentAnalyzer(Analyzer):
    """Analizador de sentimiento con capacidades trascendentales."""
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar analizador de sentimiento.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("sentiment", "Análisis de Sentimiento", mode)
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar señales basadas en sentimiento de mercado.
        
        Args:
            data: Datos a analizar
            
        Returns:
            Datos con señales de sentimiento
        """
        analyzed_data = data.copy()
        
        # Inicializar estructura de señales si no existe
        if "signals" not in analyzed_data:
            analyzed_data["signals"] = {}
        
        # Obtener datos de sentimiento y mercado
        sentiment_data = analyzed_data.get("sentiment", {})
        market_data = analyzed_data.get("market_data", {})
        sentiment_by_currency = sentiment_data.get("sentiment", {})
        
        # Generar señales para cada símbolo
        for symbol, symbol_data in market_data.items():
            # Si no hay señales previas para este símbolo, inicializar
            if symbol not in analyzed_data["signals"]:
                analyzed_data["signals"][symbol] = {}
            
            # Extraer moneda base del símbolo
            base_currency = None
            if symbol.endswith("USDT"):
                base_currency = symbol[:-4]
            elif symbol.endswith("USD"):
                base_currency = symbol[:-3]
            
            # Generar señales de sentimiento
            sentiment_signal = {
                "sentiment": {
                    "score": 0,
                    "signal": SignalType.HOLD,
                    "strength": 0,
                    "timestamp": time.time()
                }
            }
            
            # Si tenemos sentimiento para esta moneda, procesarlo
            if base_currency and base_currency in sentiment_by_currency:
                sentiment_score = sentiment_by_currency[base_currency]
                sentiment_signal["sentiment"]["score"] = sentiment_score
                
                # Convertir score a señal
                if sentiment_score > 3:
                    sentiment_signal["sentiment"]["signal"] = SignalType.BUY
                    sentiment_signal["sentiment"]["strength"] = min(sentiment_score / 10, 1.0)
                elif sentiment_score < -3:
                    sentiment_signal["sentiment"]["signal"] = SignalType.SELL
                    sentiment_signal["sentiment"]["strength"] = min(abs(sentiment_score) / 10, 1.0)
                else:
                    sentiment_signal["sentiment"]["signal"] = SignalType.HOLD
                    sentiment_signal["sentiment"]["strength"] = 0
            
            # Actualizar señales para este símbolo
            analyzed_data["signals"][symbol].update(sentiment_signal)
        
        # Agregar timestamp de análisis
        analyzed_data["sentiment_analysis_timestamp"] = time.time()
        
        return analyzed_data

class CombinedAnalyzer(Analyzer):
    """Analizador combinado que integra señales técnicas y de sentimiento."""
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar analizador combinado.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("combined", "Análisis Combinado", mode)
        self.weights = {
            "technical": 0.7,
            "sentiment": 0.3
        }
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combinar señales de diferentes fuentes y generar señal final.
        
        Args:
            data: Datos a analizar
            
        Returns:
            Datos con señales combinadas
        """
        analyzed_data = data.copy()
        
        # Asegurarnos que exista la estructura de señales
        if "signals" not in analyzed_data:
            return analyzed_data  # No hay señales para combinar
        
        signals = analyzed_data["signals"]
        
        # Para cada símbolo, combinar señales existentes
        for symbol, symbol_signals in signals.items():
            # Inicializar señal combinada final
            combined_signal = {
                "final": {
                    "signal": SignalType.HOLD,
                    "strength": 0,
                    "sources": [],
                    "timestamp": time.time()
                }
            }
            
            # Acumular fuerza ponderada
            weighted_strength = 0
            sources = []
            
            # Procesar señal técnica
            if "technical" in symbol_signals:
                tech_signal = symbol_signals["technical"]
                tech_type = tech_signal.get("combined")
                tech_strength = tech_signal.get("strength", 0)
                
                # Aplicar peso
                weighted_tech_strength = tech_strength * self.weights["technical"]
                weighted_strength += weighted_tech_strength
                
                # Registrar fuente
                sources.append({
                    "source": "technical",
                    "signal": tech_type,
                    "raw_strength": tech_strength,
                    "weighted_strength": weighted_tech_strength
                })
            
            # Procesar señal de sentimiento
            if "sentiment" in symbol_signals:
                sent_signal = symbol_signals["sentiment"]
                sent_type = sent_signal.get("signal")
                sent_strength = sent_signal.get("strength", 0)
                
                # El sentimiento SELL genera valor negativo
                if sent_type == SignalType.SELL:
                    sent_strength = -sent_strength
                
                # Aplicar peso
                weighted_sent_strength = sent_strength * self.weights["sentiment"]
                weighted_strength += weighted_sent_strength
                
                # Registrar fuente
                sources.append({
                    "source": "sentiment",
                    "signal": sent_type,
                    "raw_strength": sent_strength,
                    "weighted_strength": weighted_sent_strength
                })
            
            # Determinar señal final basada en fuerza combinada
            if weighted_strength > 0.3:
                final_signal = SignalType.BUY
            elif weighted_strength < -0.3:
                final_signal = SignalType.SELL
            else:
                final_signal = SignalType.HOLD
            
            # Asignar valores finales
            combined_signal["final"]["signal"] = final_signal
            combined_signal["final"]["strength"] = weighted_strength
            combined_signal["final"]["sources"] = sources
            
            # Actualizar señales del símbolo
            symbol_signals.update(combined_signal)
        
        # Agregar timestamp de análisis combinado
        analyzed_data["combined_analysis_timestamp"] = time.time()
        
        return analyzed_data

class ModelBasedAnalyzer(Analyzer):
    """Analizador basado en modelos de machine learning."""
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar analizador basado en modelos.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("model", "Análisis con Modelos", mode)
        self.models = {}
        self.prediction_horizon = 24  # Horas
        
        # Iniciar con un modelo simple por defecto
        self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Inicializar modelo simple por defecto."""
        # En una implementación real, aquí cargaríamos modelos entrenados
        # Para este ejemplo, usamos una función simple basada en reglas
        self.models["default"] = {
            "predict": self._simple_predict,
            "type": "rule_based",
            "version": "1.0"
        }
    
    def _simple_predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Predictor simple basado en reglas.
        
        Args:
            features: Características para predicción
            
        Returns:
            Tupla (señal, probabilidad)
        """
        # Extraer características relevantes
        rsi = features.get("rsi", 50)
        macd = features.get("macd", 0)
        macd_hist = features.get("macd_hist", 0)
        ema_cross = features.get("ema_cross", 0)
        bb_position = features.get("bb_position", 0.5)
        price_distance_ema21 = features.get("price_distance_ema21", 0)
        
        # Combinar características con pesos
        score = 0
        score += (50 - rsi) / 15  # RSI bajo -> score positivo
        score += macd_hist * 2  # MACD histograma positivo -> score positivo
        score += ema_cross * 0.5  # EMA cruce alcista -> score positivo
        score += (0.5 - bb_position) * 0.5  # Posición BB baja -> score positivo
        score -= price_distance_ema21 * 2  # Precio por encima de EMA21 -> score negativo
        
        # Normalizar score
        score = max(min(score, 1.0), -1.0)
        
        # Determinar señal y probabilidad
        if score > 0.3:
            return SignalType.BUY, abs(score)
        elif score < -0.3:
            return SignalType.SELL, abs(score)
        else:
            return SignalType.HOLD, 1 - abs(score)
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar señales basadas en modelos de predicción.
        
        Args:
            data: Datos a analizar
            
        Returns:
            Datos con señales de modelos
        """
        analyzed_data = data.copy()
        
        # Inicializar estructura de señales si no existe
        if "signals" not in analyzed_data:
            analyzed_data["signals"] = {}
        
        # Obtener datos de mercado
        market_data = analyzed_data.get("market_data", {})
        
        # Modelo por defecto
        model = self.models.get("default")
        if not model:
            logger.warning("No hay modelos disponibles para análisis")
            return analyzed_data
        
        # Generar señales para cada símbolo
        for symbol, symbol_data in market_data.items():
            # Si no hay señales previas para este símbolo, inicializar
            if symbol not in analyzed_data["signals"]:
                analyzed_data["signals"][symbol] = {}
            
            # Extraer último candle con características
            latest_candle = symbol_data.get("latest", {})
            if not latest_candle:
                continue
            
            # Preparar features para el modelo
            features = {}
            for key in ["rsi", "macd", "macd_hist", "bb_position", "price_distance_ema21", "ema_cross"]:
                if key in latest_candle:
                    features[key] = latest_candle[key]
            
            # Predecir con modelo
            signal, probability = model["predict"](features)
            
            # Crear señal del modelo
            model_signal = {
                "model": {
                    "signal": signal,
                    "probability": probability,
                    "strength": probability if signal != SignalType.HOLD else 0,
                    "model_type": model["type"],
                    "model_version": model["version"],
                    "horizon": self.prediction_horizon,
                    "timestamp": time.time()
                }
            }
            
            # Si es SELL, hacer negativa la fuerza
            if signal == SignalType.SELL:
                model_signal["model"]["strength"] = -model_signal["model"]["strength"]
            
            # Actualizar señales para este símbolo
            analyzed_data["signals"][symbol].update(model_signal)
        
        # Agregar timestamp de análisis
        analyzed_data["model_analysis_timestamp"] = time.time()
        
        return analyzed_data

class AnalysisEngine(GenesisComponent, GenesisSingleton):
    """
    Motor de análisis con capacidades trascendentales.
    
    Este componente coordina todos los analizadores y genera señales
    combinadas para toma de decisiones.
    """
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar motor de análisis.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("analysis_engine", mode)
        self.analyzers: Dict[str, Analyzer] = {}
        self.analysis_sequence = []
        self.db = TranscendentalDatabase()
        
        logger.info(f"Motor de análisis inicializado en modo {mode}")
    
    def register_analyzer(self, analyzer_id: str, analyzer: Analyzer) -> None:
        """
        Registrar analizador.
        
        Args:
            analyzer_id: Identificador único del analizador
            analyzer: Instancia del analizador
        """
        self.analyzers[analyzer_id] = analyzer
        logger.info(f"Analizador {analyzer_id} registrado")
    
    def set_analysis_sequence(self, sequence: List[str]) -> None:
        """
        Establecer secuencia de análisis.
        
        Args:
            sequence: Lista de IDs de analizadores en orden de ejecución
        """
        # Verificar que todos los analizadores existan
        for analyzer_id in sequence:
            if analyzer_id not in self.analyzers:
                raise ValueError(f"Analizador {analyzer_id} no encontrado")
        
        self.analysis_sequence = sequence
        logger.info(f"Secuencia de análisis establecida: {sequence}")
    
    async def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizar datos a través de la secuencia completa.
        
        Args:
            data: Datos a analizar
            
        Returns:
            Datos analizados con señales
        """
        analyzed_data = data.copy()
        
        # Si no hay secuencia establecida, usar todos en orden de registro
        if not self.analysis_sequence:
            self.analysis_sequence = list(self.analyzers.keys())
        
        logger.info(f"Iniciando análisis con {len(self.analysis_sequence)} analizadores")
        
        for analyzer_id in self.analysis_sequence:
            analyzer = self.analyzers[analyzer_id]
            logger.debug(f"Ejecutando analizador: {analyzer.analyzer_name} ({analyzer_id})")
            
            try:
                start_time = time.time()
                analyzed_data = await analyzer.analyze_with_resilience(analyzed_data)
                analysis_time = time.time() - start_time
                
                # Registrar estadísticas
                logger.debug(f"Analizador {analyzer_id} completado en {analysis_time:.3f}s")
                self.update_metric(f"{analyzer_id}_time", analysis_time)
                
            except Exception as e:
                logger.error(f"Error en analizador {analyzer_id}: {str(e)}")
                self.register_operation(False)
                # Continuar con el siguiente analizador para mantener resiliencia
        
        # Registrar operación exitosa
        self.register_operation(True)
        analyzed_data["analysis_timestamp"] = time.time()
        
        return analyzed_data
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor de análisis.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = super().get_stats()
        
        # Agregar estadísticas específicas
        engine_stats = {
            "analyzers": len(self.analyzers),
            "analysis_sequence": self.analysis_sequence,
            "analyzer_stats": {
                analyzer_id: analyzer.get_stats() 
                for analyzer_id, analyzer in self.analyzers.items()
            }
        }
        
        stats.update(engine_stats)
        return stats
    
    async def initialize(self) -> bool:
        """
        Inicializar motor de análisis con analizadores estándar.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Registrar analizadores estándar
            self.register_analyzer("technical", TechnicalAnalyzer(self.mode))
            self.register_analyzer("sentiment", SentimentAnalyzer(self.mode))
            self.register_analyzer("model", ModelBasedAnalyzer(self.mode))
            self.register_analyzer("combined", CombinedAnalyzer(self.mode))
            
            # Establecer secuencia por defecto
            self.set_analysis_sequence(["technical", "sentiment", "model", "combined"])
            
            logger.info(f"Motor de análisis inicializado con {len(self.analyzers)} analizadores")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar motor de análisis: {str(e)}")
            return False

# Función de análisis para el pipeline
async def process_data_analysis(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función de análisis de datos para el pipeline.
    
    Args:
        data: Datos procesados
        context: Contexto de ejecución
        
    Returns:
        Datos analizados con señales
    """
    engine = AnalysisEngine()
    
    # Inicializar si es necesario
    if not engine.analyzers:
        await engine.initialize()
    
    # Analizar datos
    analyzed_data = await engine.analyze_data(data)
    
    # Registrar analizadores en el contexto
    context["analyzers"] = engine.analysis_sequence
    
    # Contar señales por tipo
    signal_counts = {signal_type: 0 for signal_type in SignalType.all_types()}
    
    for symbol, signals in analyzed_data.get("signals", {}).items():
        if "final" in signals:
            signal_type = signals["final"]["signal"]
            signal_counts[signal_type] += 1
    
    # Registrar conteo en el contexto
    context["signal_counts"] = signal_counts
    
    logger.info(f"Análisis de datos completado. Señales: {signal_counts}")
    return analyzed_data

# Instancia global para uso directo
analysis_engine = AnalysisEngine()