"""
Módulo de Procesamiento para el Pipeline de Genesis.

Este módulo se encarga del preprocesamiento, limpieza y transformación
de datos con capacidades de resiliencia trascendental.
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
logger = logging.getLogger("genesis.pipeline.processing")

class DataProcessor(GenesisComponent):
    """Procesador de datos base con capacidades trascendentales."""
    
    def __init__(self, processor_id: str, processor_name: str, mode: str = "SINGULARITY_V4"):
        """
        Inicializar procesador de datos.
        
        Args:
            processor_id: Identificador único del procesador
            processor_name: Nombre descriptivo
            mode: Modo trascendental
        """
        super().__init__(f"processor_{processor_id}", mode)
        self.processor_id = processor_id
        self.processor_name = processor_name
        self.last_processing = 0
        self.processing_count = 0
        self.db = TranscendentalDatabase()
        
        logger.info(f"Procesador {processor_name} ({processor_id}) inicializado")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar datos.
        
        Args:
            data: Datos a procesar
            
        Returns:
            Datos procesados
        """
        raise NotImplementedError("Las subclases deben implementar process")
    
    async def process_with_resilience(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar datos con mecanismos de resiliencia.
        
        Args:
            data: Datos a procesar
            
        Returns:
            Datos procesados o datos originales en caso de error
        """
        self.processing_count += 1
        self.last_processing = time.time()
        
        try:
            start_time = time.time()
            result = await self.process(data)
            processing_time = time.time() - start_time
            
            # Actualizar métricas
            self.update_metric("processing_time", processing_time)
            self.update_metric("success_rate", 1.0 - (self.error_count / max(1, self.operation_count)))
            
            logger.debug(f"Procesamiento {self.processor_id} exitoso en {processing_time:.3f}s")
            self.register_operation(True)
            return result
        
        except Exception as e:
            self.register_operation(False)
            logger.error(f"Error en procesador {self.processor_id}: {str(e)}")
            
            # En caso de error, devolver los datos originales
            return data

class DataCleaner(DataProcessor):
    """Limpiador de datos con capacidades de resilencia trascendental."""
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar limpiador de datos.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("cleaner", "Limpieza de Datos", mode)
        self.nan_strategy = "ffill"  # forward fill
        self.outlier_detection = "zscore"
        self.outlier_threshold = 3.0
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Limpiar datos de mercado.
        
        Args:
            data: Datos a limpiar
            
        Returns:
            Datos limpios
        """
        cleaned_data = data.copy()
        
        # Obtener datos de mercado
        market_data = data.get("market_data", {})
        cleaned_market_data = {}
        
        for symbol, symbol_data in market_data.items():
            # Convertir datos a DataFrame para procesamiento
            candles = symbol_data.get("data", [])
            if not candles:
                cleaned_market_data[symbol] = symbol_data
                continue
            
            df = pd.DataFrame(candles)
            
            # 1. Eliminar duplicados
            df = df.drop_duplicates(subset=["timestamp"])
            
            # 2. Ordenar por timestamp
            df = df.sort_values("timestamp")
            
            # 3. Manejar NaN
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            if self.nan_strategy == "ffill":
                df[numeric_columns] = df[numeric_columns].fillna(method='ffill')
                df[numeric_columns] = df[numeric_columns].fillna(method='bfill')  # por si hay NaN al principio
            elif self.nan_strategy == "mean":
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
            
            # 4. Detectar y corregir outliers
            if self.outlier_detection == "zscore":
                for col in ["close", "volume"]:
                    if col in df.columns:
                        # Calcular Z-score
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        outliers = z_scores > self.outlier_threshold
                        
                        # Registrar outliers
                        n_outliers = outliers.sum()
                        if n_outliers > 0:
                            logger.debug(f"Detectados {n_outliers} outliers en {symbol} columna {col}")
                            
                            # Reemplazar outliers con media móvil
                            df.loc[outliers, col] = df[col].rolling(window=5, min_periods=1).mean()
            
            # 5. Asegurar que high >= open/close y low <= open/close
            df["high"] = df[["high", "open", "close"]].max(axis=1)
            df["low"] = df[["low", "open", "close"]].min(axis=1)
            
            # Convertir de nuevo a formato original
            clean_candles = df.to_dict(orient="records")
            
            # Actualizar datos limpios
            cleaned_symbol_data = symbol_data.copy()
            cleaned_symbol_data["data"] = clean_candles
            cleaned_symbol_data["cleaned"] = True
            cleaned_symbol_data["outliers_corrected"] = n_outliers
            
            cleaned_market_data[symbol] = cleaned_symbol_data
        
        # Actualizar datos de mercado limpios
        cleaned_data["market_data"] = cleaned_market_data
        cleaned_data["cleaning_timestamp"] = time.time()
        
        return cleaned_data

class FeatureEngineer(DataProcessor):
    """Ingeniero de características con capacidades trascendentales."""
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar ingeniero de características.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("feature_engineer", "Ingeniería de Características", mode)
        self.indicators_config = {
            "rsi": {"window": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bbands": {"window": 20, "num_std_dev": 2},
            "atr": {"window": 14},
            "ema": {"windows": [5, 8, 13, 21, 55]},
            "volume_sma": {"window": 20}
        }
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcular características e indicadores técnicos.
        
        Args:
            data: Datos a procesar
            
        Returns:
            Datos con características adicionales
        """
        enriched_data = data.copy()
        
        # Obtener datos de mercado
        market_data = data.get("market_data", {})
        enriched_market_data = {}
        
        for symbol, symbol_data in market_data.items():
            # Convertir datos a DataFrame para procesamiento
            candles = symbol_data.get("data", [])
            if not candles:
                enriched_market_data[symbol] = symbol_data
                continue
            
            df = pd.DataFrame(candles)
            
            # Ordenar por timestamp
            df = df.sort_values("timestamp")
            
            # Convertir columnas numéricas
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            # Calcular indicadores técnicos
            df_with_indicators = self._calculate_indicators(df)
            
            # Calcular características adicionales
            df_with_features = self._calculate_features(df_with_indicators)
            
            # Convertir de nuevo a formato original
            enriched_candles = df_with_features.to_dict(orient="records")
            
            # Actualizar datos enriquecidos
            enriched_symbol_data = symbol_data.copy()
            enriched_symbol_data["data"] = enriched_candles
            enriched_symbol_data["enriched"] = True
            enriched_symbol_data["indicators"] = list(self.indicators_config.keys())
            
            enriched_market_data[symbol] = enriched_symbol_data
        
        # Actualizar datos de mercado enriquecidos
        enriched_data["market_data"] = enriched_market_data
        enriched_data["feature_engineering_timestamp"] = time.time()
        
        return enriched_data
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcular indicadores técnicos.
        
        Args:
            df: DataFrame con datos de precio
            
        Returns:
            DataFrame con indicadores
        """
        # Crear copia para no modificar el original
        result_df = df.copy()
        
        # 1. RSI
        if "rsi" in self.indicators_config:
            window = self.indicators_config["rsi"]["window"]
            delta = result_df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss
            result_df["rsi"] = 100 - (100 / (1 + rs))
        
        # 2. MACD
        if "macd" in self.indicators_config:
            fast = self.indicators_config["macd"]["fast"]
            slow = self.indicators_config["macd"]["slow"]
            signal = self.indicators_config["macd"]["signal"]
            
            ema_fast = result_df["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = result_df["close"].ewm(span=slow, adjust=False).mean()
            result_df["macd"] = ema_fast - ema_slow
            result_df["macd_signal"] = result_df["macd"].ewm(span=signal, adjust=False).mean()
            result_df["macd_hist"] = result_df["macd"] - result_df["macd_signal"]
        
        # 3. Bandas de Bollinger
        if "bbands" in self.indicators_config:
            window = self.indicators_config["bbands"]["window"]
            num_std_dev = self.indicators_config["bbands"]["num_std_dev"]
            
            result_df["bb_middle"] = result_df["close"].rolling(window=window).mean()
            result_df["bb_std"] = result_df["close"].rolling(window=window).std()
            result_df["bb_upper"] = result_df["bb_middle"] + (result_df["bb_std"] * num_std_dev)
            result_df["bb_lower"] = result_df["bb_middle"] - (result_df["bb_std"] * num_std_dev)
            
            # Posición dentro de las bandas (0-1)
            result_df["bb_width"] = (result_df["bb_upper"] - result_df["bb_lower"]) / result_df["bb_middle"]
            result_df["bb_position"] = (result_df["close"] - result_df["bb_lower"]) / (result_df["bb_upper"] - result_df["bb_lower"])
        
        # 4. ATR (Average True Range)
        if "atr" in self.indicators_config:
            window = self.indicators_config["atr"]["window"]
            
            high_low = result_df["high"] - result_df["low"]
            high_close = np.abs(result_df["high"] - result_df["close"].shift())
            low_close = np.abs(result_df["low"] - result_df["close"].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            result_df["atr"] = true_range.rolling(window=window).mean()
        
        # 5. EMAs
        if "ema" in self.indicators_config:
            for window in self.indicators_config["ema"]["windows"]:
                result_df[f"ema_{window}"] = result_df["close"].ewm(span=window, adjust=False).mean()
        
        # 6. Volumen SMA
        if "volume_sma" in self.indicators_config:
            window = self.indicators_config["volume_sma"]["window"]
            result_df["volume_sma"] = result_df["volume"].rolling(window=window).mean()
            
            # Volumen relativo 
            result_df["volume_ratio"] = result_df["volume"] / result_df["volume_sma"]
        
        return result_df
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcular características adicionales derivadas.
        
        Args:
            df: DataFrame con indicadores
            
        Returns:
            DataFrame con características adicionales
        """
        # Crear copia para no modificar el original
        result_df = df.copy()
        
        # 1. Retornos logarítmicos
        result_df["log_return"] = np.log(result_df["close"] / result_df["close"].shift(1))
        
        # 2. Volatilidad móvil
        result_df["volatility"] = result_df["log_return"].rolling(window=20).std() * np.sqrt(365)
        
        # 3. Días desde máximo/mínimo
        result_df["price_max"] = result_df["close"].rolling(window=20).max()
        result_df["price_min"] = result_df["close"].rolling(window=20).min()
        result_df["days_since_max"] = (result_df["price_max"] != result_df["close"]).astype(int).cumsum()
        result_df["days_since_min"] = (result_df["price_min"] != result_df["close"]).astype(int).cumsum()
        
        # 4. Price change features
        result_df["price_change_1d"] = result_df["close"].pct_change(1)
        result_df["price_change_3d"] = result_df["close"].pct_change(3)
        result_df["price_change_5d"] = result_df["close"].pct_change(5)
        
        # 5. Cruces de medias móviles
        if all(f"ema_{window}" in result_df.columns for window in [8, 21]):
            result_df["ema_cross"] = (result_df["ema_8"] > result_df["ema_21"]).astype(int)
            result_df["ema_cross_changed"] = result_df["ema_cross"].diff().abs()
        
        # 6. Señales MACD
        if all(col in result_df.columns for col in ["macd", "macd_signal"]):
            result_df["macd_cross"] = (result_df["macd"] > result_df["macd_signal"]).astype(int)
            result_df["macd_cross_changed"] = result_df["macd_cross"].diff().abs()
        
        # 7. Señales RSI
        if "rsi" in result_df.columns:
            result_df["rsi_oversold"] = (result_df["rsi"] < 30).astype(int)
            result_df["rsi_overbought"] = (result_df["rsi"] > 70).astype(int)
        
        # 8. Media de Body y Sombras
        result_df["body"] = abs(result_df["close"] - result_df["open"])
        result_df["upper_shadow"] = result_df["high"] - result_df[["open", "close"]].max(axis=1)
        result_df["lower_shadow"] = result_df[["open", "close"]].min(axis=1) - result_df["low"]
        
        # 9. Price distance from moving averages (normalized)
        if "ema_21" in result_df.columns:
            result_df["price_distance_ema21"] = (result_df["close"] - result_df["ema_21"]) / result_df["ema_21"]
        
        # 10. Tendencia de volumen
        result_df["volume_trend"] = result_df["volume"].pct_change(3).rolling(window=5).mean()
        
        return result_df

class DataAggregator(DataProcessor):
    """Agregador de datos de múltiples fuentes."""
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar agregador de datos.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("aggregator", "Agregación de Datos", mode)
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agregar datos de múltiples fuentes.
        
        Args:
            data: Datos a agregar
            
        Returns:
            Datos agregados
        """
        aggregated_data = data.copy()
        
        # Asegurarnos que existan las claves necesarias
        if "market_data" not in aggregated_data:
            aggregated_data["market_data"] = {}
        
        if "sentiment" not in aggregated_data:
            aggregated_data["sentiment"] = {"data": [], "sentiment": {}}
        
        # Agregar datos de mercado con sentimiento
        market_data = aggregated_data["market_data"]
        sentiment_data = aggregated_data.get("sentiment", {})
        sentiment_by_currency = sentiment_data.get("sentiment", {})
        
        for symbol, symbol_data in market_data.items():
            # Extraer moneda base del símbolo
            base_currency = None
            if symbol.endswith("USDT"):
                base_currency = symbol[:-4]
            elif symbol.endswith("USD"):
                base_currency = symbol[:-3]
            
            # Agregar sentimiento si está disponible
            if base_currency and base_currency in sentiment_by_currency:
                symbol_data["sentiment_score"] = sentiment_by_currency[base_currency]
            else:
                symbol_data["sentiment_score"] = 0.0
            
            # Resumir indicadores técnicos
            candles = symbol_data.get("data", [])
            if candles:
                latest_candle = candles[-1]
                symbol_data["latest"] = latest_candle
                
                # Resumir señales técnicas
                tech_signals = {}
                
                # RSI
                if "rsi" in latest_candle:
                    rsi = latest_candle["rsi"]
                    if rsi < 30:
                        tech_signals["rsi"] = "oversold"
                    elif rsi > 70:
                        tech_signals["rsi"] = "overbought"
                    else:
                        tech_signals["rsi"] = "neutral"
                
                # MACD
                if all(k in latest_candle for k in ["macd", "macd_signal"]):
                    if latest_candle["macd"] > latest_candle["macd_signal"]:
                        tech_signals["macd"] = "bullish"
                    else:
                        tech_signals["macd"] = "bearish"
                
                # Bandas de Bollinger
                if all(k in latest_candle for k in ["bb_position"]):
                    bb_pos = latest_candle["bb_position"]
                    if bb_pos < 0.2:
                        tech_signals["bbands"] = "oversold"
                    elif bb_pos > 0.8:
                        tech_signals["bbands"] = "overbought"
                    else:
                        tech_signals["bbands"] = "neutral"
                
                # Cruces de EMAs
                if "ema_cross" in latest_candle:
                    if latest_candle["ema_cross"] == 1:
                        tech_signals["ema_cross"] = "bullish"
                    else:
                        tech_signals["ema_cross"] = "bearish"
                
                # Agregar resumen técnico
                symbol_data["technical_signals"] = tech_signals
                
                # Calcular señal combinada
                bullish_count = sum(1 for signal in tech_signals.values() 
                                 if signal in ["bullish", "oversold"])
                bearish_count = sum(1 for signal in tech_signals.values() 
                                  if signal in ["bearish", "overbought"])
                
                if bullish_count > bearish_count:
                    symbol_data["combined_signal"] = "bullish"
                elif bearish_count > bullish_count:
                    symbol_data["combined_signal"] = "bearish"
                else:
                    symbol_data["combined_signal"] = "neutral"
        
        # Actualizar timestamp
        aggregated_data["aggregation_timestamp"] = time.time()
        
        return aggregated_data

class ProcessingEngine(GenesisComponent, GenesisSingleton):
    """
    Motor de procesamiento de datos con capacidades trascendentales.
    
    Este componente coordina todos los procesadores y transforma los
    datos para su uso en análisis y toma de decisiones.
    """
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar motor de procesamiento.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("processing_engine", mode)
        self.processors: Dict[str, DataProcessor] = {}
        self.processing_sequence = []
        self.db = TranscendentalDatabase()
        
        logger.info(f"Motor de procesamiento inicializado en modo {mode}")
    
    def register_processor(self, processor_id: str, processor: DataProcessor) -> None:
        """
        Registrar procesador.
        
        Args:
            processor_id: Identificador único del procesador
            processor: Instancia del procesador
        """
        self.processors[processor_id] = processor
        logger.info(f"Procesador {processor_id} registrado")
    
    def set_processing_sequence(self, sequence: List[str]) -> None:
        """
        Establecer secuencia de procesamiento.
        
        Args:
            sequence: Lista de IDs de procesadores en orden de ejecución
        """
        # Verificar que todos los procesadores existan
        for processor_id in sequence:
            if processor_id not in self.processors:
                raise ValueError(f"Procesador {processor_id} no encontrado")
        
        self.processing_sequence = sequence
        logger.info(f"Secuencia de procesamiento establecida: {sequence}")
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar datos a través de la secuencia completa.
        
        Args:
            data: Datos a procesar
            
        Returns:
            Datos procesados
        """
        processed_data = data.copy()
        
        # Si no hay secuencia establecida, procesar con todos en orden de registro
        if not self.processing_sequence:
            self.processing_sequence = list(self.processors.keys())
        
        logger.info(f"Iniciando procesamiento con {len(self.processing_sequence)} procesadores")
        
        for processor_id in self.processing_sequence:
            processor = self.processors[processor_id]
            logger.debug(f"Ejecutando procesador: {processor.processor_name} ({processor_id})")
            
            try:
                start_time = time.time()
                processed_data = await processor.process_with_resilience(processed_data)
                processing_time = time.time() - start_time
                
                # Registrar estadísticas
                logger.debug(f"Procesador {processor_id} completado en {processing_time:.3f}s")
                self.update_metric(f"{processor_id}_time", processing_time)
                
            except Exception as e:
                logger.error(f"Error en procesador {processor_id}: {str(e)}")
                self.register_operation(False)
                # Continuar con el siguiente procesador para mantener resiliencia
        
        # Registrar operación exitosa
        self.register_operation(True)
        processed_data["processing_timestamp"] = time.time()
        
        return processed_data
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor de procesamiento.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = super().get_stats()
        
        # Agregar estadísticas específicas
        engine_stats = {
            "processors": len(self.processors),
            "processing_sequence": self.processing_sequence,
            "processor_stats": {
                processor_id: processor.get_stats() 
                for processor_id, processor in self.processors.items()
            }
        }
        
        stats.update(engine_stats)
        return stats
    
    async def initialize(self) -> bool:
        """
        Inicializar motor de procesamiento con procesadores estándar.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Registrar procesadores estándar
            self.register_processor("cleaner", DataCleaner(self.mode))
            self.register_processor("feature_engineer", FeatureEngineer(self.mode))
            self.register_processor("aggregator", DataAggregator(self.mode))
            
            # Establecer secuencia por defecto
            self.set_processing_sequence(["cleaner", "feature_engineer", "aggregator"])
            
            logger.info(f"Motor de procesamiento inicializado con {len(self.processors)} procesadores")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar motor de procesamiento: {str(e)}")
            return False

# Función de procesamiento para el pipeline
async def process_data_processing(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función de procesamiento de datos para el pipeline.
    
    Args:
        data: Datos de entrada
        context: Contexto de ejecución
        
    Returns:
        Datos procesados
    """
    engine = ProcessingEngine()
    
    # Inicializar si es necesario
    if not engine.processors:
        await engine.initialize()
    
    # Procesar datos
    processed_data = await engine.process_data(data)
    
    # Registrar procesadores en el contexto
    context["processors"] = engine.processing_sequence
    
    logger.info("Procesamiento de datos completado")
    return processed_data

# Instancia global para uso directo
processing_engine = ProcessingEngine()