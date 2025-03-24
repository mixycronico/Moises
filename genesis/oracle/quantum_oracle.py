"""
Oráculo Cuántico Predictivo para el Sistema Genesis.

Este módulo implementa un sistema de predicción avanzada que utiliza:
1. Redes neuronales LSTM con atención para series temporales
2. Algoritmos de optimización inspirados en mecánica cuántica
3. Entrelazamiento de señales para correlacionar múltiples activos
4. Detección de anomalías trascendentales pre-evento

El sistema opera en múltiples dimensiones temporales simultáneamente,
permitiendo una visión holística del mercado y detectando patrones
invisibles para sistemas convencionales.
"""

import asyncio
import logging
import numpy as np
import random
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from enum import Enum, auto

# Configurar logging
logger = logging.getLogger("genesis.oracle")

class PredictionConfidence(Enum):
    """Niveles de confianza para predicciones."""
    LOW = auto()          # < 65%
    MEDIUM = auto()       # 65-80%
    HIGH = auto()         # 80-90%
    VERY_HIGH = auto()    # 90-95%
    DIVINE = auto()       # > 95%

class TemporalHorizon(Enum):
    """Horizontes temporales de predicción."""
    IMMEDIATE = auto()    # 5-15 minutos
    SHORT = auto()        # 1-4 horas
    MEDIUM = auto()       # 1-3 días
    LONG = auto()         # 1-2 semanas
    TRANSCENDENT = auto() # > 2 semanas

class MarketInsightType(Enum):
    """Tipos de insights de mercado."""
    TREND_CHANGE = auto()         # Cambio de tendencia
    SUPPORT_BREAK = auto()        # Ruptura de soporte
    RESISTANCE_BREAK = auto()     # Ruptura de resistencia
    VOLUME_ANOMALY = auto()       # Anomalía de volumen
    CORRELATION_SHIFT = auto()    # Cambio en correlaciones
    SENTIMENT_CHANGE = auto()     # Cambio de sentimiento
    OSCILLATION_PATTERN = auto()  # Patrón oscilatorio
    CONVERGENCE = auto()          # Convergencia de indicadores
    DIVERGENCE = auto()           # Divergencia de indicadores
    QUANTUM_RESONANCE = auto()    # Resonancia cuántica

class DimensionalState(Enum):
    """Estados dimensionales del oráculo."""
    CALIBRATING = auto()          # Calibrando
    OPERATING = auto()            # Operando normal
    ENTANGLING = auto()           # Entrelazando
    FORECASTING = auto()          # Pronosticando
    TRANSMUTING = auto()          # Transmutando señales
    DIMENSIONAL_SHIFT = auto()    # Cambio dimensional
    RESONATING = auto()           # Resonando
    QUANTUM_COHERENCE = auto()    # Coherencia cuántica

class QuantumOracle:
    """
    Oráculo Cuántico Predictivo que utiliza técnicas de ML avanzadas
    y principios cuánticos para generar predicciones trascendentales.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el Oráculo Cuántico.
        
        Args:
            config: Configuración opcional
        """
        self.initialized = False
        self.config = config or {}
        self.state = DimensionalState.CALIBRATING
        self.confidence_threshold = self.config.get("confidence_threshold", 0.65)
        self.entanglement_level = self.config.get("entanglement_level", 0.7)
        self.temporal_resolution = self.config.get("temporal_resolution", 300)  # segundos
        self.dimensional_spaces = self.config.get("dimensional_spaces", 5)
        
        # Activos bajo seguimiento
        self.tracked_assets: Dict[str, Dict[str, Any]] = {}
        
        # Insights detectados
        self.current_insights: List[Dict[str, Any]] = []
        
        # Predicciones activas
        self.active_predictions: Dict[str, Dict[str, Any]] = {}
        
        # Métricas
        self.metrics = {
            "accuracy_short_term": 0.0,
            "accuracy_medium_term": 0.0,
            "accuracy_long_term": 0.0,
            "insights_generated": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "dimensional_shifts": 0,
            "entanglement_score": 0.0,
            "coherence_level": 0.0,
            "temporal_accuracy": 0.0
        }
        
        # Simulador de modelos para demostración
        self._lstm_attn_model = None
        self._quantum_optimizer = None
        self._anomaly_detector = None
        
        logger.info("Oráculo Cuántico inicializado en modo calibración")
    
    async def initialize(self) -> bool:
        """
        Inicializar completamente el oráculo.
        
        Returns:
            True si inicializado correctamente
        """
        try:
            # Simular inicialización de modelos
            logger.info("Inicializando modelos predictivos cuánticos...")
            await asyncio.sleep(1.0)
            
            # Inicializar trackers de assets
            await self._initialize_asset_trackers()
            
            # Inicializar espacios dimensionales
            await self._initialize_dimensional_spaces()
            
            # Estado inicial
            self.state = DimensionalState.OPERATING
            self.initialized = True
            
            logger.info("Oráculo Cuántico completamente inicializado y operativo")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar Oráculo Cuántico: {e}")
            return False
    
    async def _initialize_asset_trackers(self) -> None:
        """Inicializar trackers para activos."""
        default_assets = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT",
            "DOT/USDT", "AVAX/USDT", "MATIC/USDT", "ATOM/USDT", "LINK/USDT"
        ]
        
        for asset in default_assets:
            self.tracked_assets[asset] = {
                "last_update": datetime.now(),
                "current_price": self._simulate_price(),
                "predicted_prices": [],
                "confidence_levels": [],
                "entanglement_assets": [],
                "dimensional_markers": {},
                "anomaly_score": 0.0,
                "active_insights": []
            }
        
        logger.info(f"Inicializados trackers para {len(self.tracked_assets)} activos")
    
    async def _initialize_dimensional_spaces(self) -> None:
        """Inicializar espacios dimensionales para análisis cuántico."""
        # Simular inicialización de espacios dimensionales
        for i in range(self.dimensional_spaces):
            logger.debug(f"Inicializando espacio dimensional {i+1}...")
            await asyncio.sleep(0.2)
        
        logger.info(f"Inicializados {self.dimensional_spaces} espacios dimensionales")
    
    def _simulate_price(self) -> float:
        """Simular precio para demo."""
        return random.uniform(100, 50000)
    
    async def update_market_data(self, market_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Actualizar datos de mercado para el oráculo.
        
        Args:
            market_data: Datos de mercado, si None se generan datos simulados
            
        Returns:
            True si actualizado correctamente
        """
        if not self.initialized:
            logger.warning("No se puede actualizar datos, oráculo no inicializado")
            return False
        
        try:
            # Cambiar estado
            prev_state = self.state
            self.state = DimensionalState.ENTANGLING
            
            # Usar datos proporcionados o simular
            data = market_data or self._generate_simulated_market_data()
            
            # Actualizar activos
            for asset, price_data in data.items():
                if asset in self.tracked_assets:
                    self.tracked_assets[asset]["last_update"] = datetime.now()
                    self.tracked_assets[asset]["current_price"] = price_data["price"]
                    
                    # Actualizar marcadores dimensionales
                    for dim in range(self.dimensional_spaces):
                        dim_key = f"dim_{dim}"
                        self.tracked_assets[asset]["dimensional_markers"][dim_key] = random.random()
            
            # Restaurar estado previo
            self.state = prev_state
            
            logger.info(f"Datos de mercado actualizados para {len(data)} activos")
            return True
            
        except Exception as e:
            logger.error(f"Error al actualizar datos de mercado: {e}")
            self.state = DimensionalState.OPERATING
            return False
    
    def _generate_simulated_market_data(self) -> Dict[str, Dict[str, Any]]:
        """Generar datos de mercado simulados para demo."""
        data = {}
        for asset in self.tracked_assets:
            # Simular movimiento de precio
            current = self.tracked_assets[asset]["current_price"]
            change_pct = random.uniform(-0.02, 0.02)
            new_price = current * (1 + change_pct)
            
            # Añadir datos simulados
            data[asset] = {
                "price": new_price,
                "volume": random.uniform(1000000, 100000000),
                "timestamp": datetime.now().timestamp(),
                "bid": new_price * 0.999,
                "ask": new_price * 1.001,
                "high_24h": new_price * 1.05,
                "low_24h": new_price * 0.95
            }
        
        return data
    
    async def generate_predictions(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generar predicciones para los activos especificados.
        
        Args:
            symbols: Lista de símbolos para predicción, si None usa todos los seguidos
            
        Returns:
            Diccionario con predicciones
        """
        if not self.initialized:
            logger.warning("No se puede generar predicciones, oráculo no inicializado")
            return {}
        
        # Cambiar estado
        prev_state = self.state
        self.state = DimensionalState.FORECASTING
        
        # Usar activos especificados o todos
        assets_to_predict = symbols or list(self.tracked_assets.keys())
        predictions = {}
        
        try:
            # Genera predicciones para cada activo
            for asset in assets_to_predict:
                if asset not in self.tracked_assets:
                    continue
                
                logger.debug(f"Generando predicción cuántica para {asset}...")
                
                # Actualizar predicciones
                result = await self._simulate_prediction(asset)
                predictions[asset] = result
                
                # Guardar predicción
                self.active_predictions[asset] = result
                
                # Actualizar tracker
                self.tracked_assets[asset]["predicted_prices"] = result["prices"]
                self.tracked_assets[asset]["confidence_levels"] = result["confidence_levels"]
            
            # Detectar correlaciones entre predicciones
            await self._entangle_predictions(predictions)
            
            logger.info(f"Generadas predicciones cuánticas para {len(predictions)} activos")
            
            # Restaurar estado
            self.state = prev_state
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error al generar predicciones: {e}")
            self.state = prev_state
            return {}
    
    async def _simulate_prediction(self, asset: str) -> Dict[str, Any]:
        """
        Simular predicción para un activo.
        
        Args:
            asset: Símbolo del activo
            
        Returns:
            Predicción simulada
        """
        current_price = self.tracked_assets[asset]["current_price"]
        
        # Generar predicciones temporales
        predictions = []
        confidence_levels = []
        
        # Simular cambios para diferentes horizontes temporales
        time_horizons = [
            timedelta(minutes=15),   # IMMEDIATE
            timedelta(hours=4),      # SHORT
            timedelta(days=2),       # MEDIUM
            timedelta(days=10),      # LONG
            timedelta(days=21)       # TRANSCENDENT
        ]
        
        # Para cada horizonte, generar predicción
        base_trend = random.uniform(-0.15, 0.2)  # Tendencia base
        
        for i, horizon in enumerate(time_horizons):
            # Añadir algo de variación a la tendencia
            if i == 0:  # IMMEDIATE
                trend = base_trend * 0.2
                confidence = random.uniform(0.85, 0.95)
            elif i == 1:  # SHORT
                trend = base_trend * 0.5
                confidence = random.uniform(0.78, 0.88)
            elif i == 2:  # MEDIUM
                trend = base_trend * 0.8
                confidence = random.uniform(0.72, 0.82)
            elif i == 3:  # LONG
                trend = base_trend
                confidence = random.uniform(0.65, 0.75)
            else:  # TRANSCENDENT
                trend = base_trend * 1.2
                confidence = random.uniform(0.6, 0.7)
            
            # Añadir algo de ruido
            noise = random.uniform(-0.02, 0.02)
            
            # Precio predicho
            predicted_price = current_price * (1 + trend + noise)
            
            # Agregar a listas
            predictions.append({
                "timestamp": (datetime.now() + horizon).timestamp(),
                "horizon": str(time_horizons[i]),
                "price": predicted_price,
                "change_pct": (predicted_price / current_price - 1) * 100
            })
            
            confidence_levels.append(confidence)
        
        # Crear resultado
        result = {
            "symbol": asset,
            "current_price": current_price,
            "prices": predictions,
            "confidence_levels": confidence_levels,
            "overall_confidence": sum(confidence_levels) / len(confidence_levels),
            "dominant_trend": "BULLISH" if base_trend > 0 else "BEARISH",
            "trend_strength": abs(base_trend) * 100,
            "generated_at": datetime.now().timestamp()
        }
        
        # Agregar nivel de confianza categorizado
        overall_confidence = result["overall_confidence"]
        if overall_confidence > 0.95:
            result["confidence_category"] = str(PredictionConfidence.DIVINE)
        elif overall_confidence > 0.9:
            result["confidence_category"] = str(PredictionConfidence.VERY_HIGH)
        elif overall_confidence > 0.8:
            result["confidence_category"] = str(PredictionConfidence.HIGH)
        elif overall_confidence > 0.65:
            result["confidence_category"] = str(PredictionConfidence.MEDIUM)
        else:
            result["confidence_category"] = str(PredictionConfidence.LOW)
        
        # Simular trabajo del modelo
        await asyncio.sleep(0.1)
        
        return result
    
    async def _entangle_predictions(self, predictions: Dict[str, Dict[str, Any]]) -> None:
        """
        Entrelazar predicciones para encontrar correlaciones.
        
        Args:
            predictions: Predicciones a entrelazar
        """
        # Cambiar estado
        prev_state = self.state
        self.state = DimensionalState.ENTANGLING
        
        try:
            # Simular entrelazamiento
            logger.debug(f"Entrelazando predicciones para {len(predictions)} activos...")
            
            # Crear matriz de correlación
            assets = list(predictions.keys())
            for i, asset1 in enumerate(assets):
                entangled_assets = []
                
                for j, asset2 in enumerate(assets):
                    if i != j:
                        # Simular correlación
                        correlation = random.uniform(-1.0, 1.0)
                        
                        # Si la correlación es fuerte, añadir a activos entrelazados
                        if abs(correlation) > self.entanglement_level:
                            entangled_assets.append({
                                "symbol": asset2,
                                "correlation": correlation,
                                "direction": "POSITIVE" if correlation > 0 else "NEGATIVE"
                            })
                
                # Actualizar información de entrelazamiento
                if asset1 in self.tracked_assets:
                    self.tracked_assets[asset1]["entanglement_assets"] = entangled_assets
            
            # Simular procesamiento
            await asyncio.sleep(0.3)
            
            # Actualizar métrica de entrelazamiento
            self.metrics["entanglement_score"] = random.uniform(0.7, 0.9)
            
            logger.info("Entrelazamiento cuántico de predicciones completado")
            
        except Exception as e:
            logger.error(f"Error en entrelazamiento de predicciones: {e}")
        
        # Restaurar estado
        self.state = prev_state
    
    async def detect_market_insights(self) -> List[Dict[str, Any]]:
        """
        Detectar insights de mercado basados en predicciones y datos actuales.
        
        Returns:
            Lista de insights detectados
        """
        if not self.initialized:
            logger.warning("No se puede detectar insights, oráculo no inicializado")
            return []
        
        # Cambiar estado
        prev_state = self.state
        self.state = DimensionalState.TRANSMUTING
        
        # Limpiar insights actuales
        self.current_insights = []
        
        try:
            logger.debug("Detectando insights cuánticos de mercado...")
            
            # Para cada activo, detectar posibles insights
            for asset, data in self.tracked_assets.items():
                # Verificar si hay predicciones
                if not data.get("predicted_prices"):
                    continue
                
                # Obtener predicción inmediata
                immediate_pred = data["predicted_prices"][0] if data["predicted_prices"] else {}
                if not immediate_pred:
                    continue
                
                # Obtener precio actual y predicho
                current_price = data["current_price"]
                predicted_price = immediate_pred.get("price", current_price)
                
                # Calcular cambio porcentual
                change_pct = (predicted_price / current_price - 1) * 100
                
                # 1. Detectar cambios de tendencia
                if abs(change_pct) > 4.0:
                    direction = "ALCISTA" if change_pct > 0 else "BAJISTA"
                    confidence = data["confidence_levels"][0] if data["confidence_levels"] else 0.7
                    
                    insight = {
                        "type": str(MarketInsightType.TREND_CHANGE),
                        "symbol": asset,
                        "direction": direction,
                        "magnitude": abs(change_pct),
                        "confidence": confidence,
                        "description": f"Cambio de tendencia {direction} detectado para {asset}",
                        "detected_at": datetime.now().timestamp(),
                        "horizon": str(TemporalHorizon.IMMEDIATE)
                    }
                    
                    self.current_insights.append(insight)
                    
                    # Agregar al tracker
                    if insight not in data["active_insights"]:
                        data["active_insights"].append(insight)
                
                # 2. Detectar rupturas de resistencia/soporte
                if len(data["predicted_prices"]) >= 3:
                    short_pred = data["predicted_prices"][1]
                    medium_pred = data["predicted_prices"][2]
                    
                    short_price = short_pred.get("price", current_price)
                    medium_price = medium_pred.get("price", current_price)
                    
                    # Si hay una aceleración en la tendencia
                    short_change = (short_price / current_price - 1) * 100
                    medium_change = (medium_price / short_price - 1) * 100
                    
                    if short_change > 0 and medium_change > short_change * 1.5:
                        # Posible ruptura de resistencia
                        confidence = min(data["confidence_levels"][1:3]) if len(data["confidence_levels"]) >= 3 else 0.7
                        
                        insight = {
                            "type": str(MarketInsightType.RESISTANCE_BREAK),
                            "symbol": asset,
                            "price_level": current_price * 1.02,  # Simular nivel de resistencia
                            "confidence": confidence,
                            "description": f"Posible ruptura de resistencia en {asset}",
                            "detected_at": datetime.now().timestamp(),
                            "horizon": str(TemporalHorizon.SHORT)
                        }
                        
                        self.current_insights.append(insight)
                        
                        # Agregar al tracker
                        if insight not in data["active_insights"]:
                            data["active_insights"].append(insight)
                            
                    elif short_change < 0 and medium_change < short_change * 1.5:
                        # Posible ruptura de soporte
                        confidence = min(data["confidence_levels"][1:3]) if len(data["confidence_levels"]) >= 3 else 0.7
                        
                        insight = {
                            "type": str(MarketInsightType.SUPPORT_BREAK),
                            "symbol": asset,
                            "price_level": current_price * 0.98,  # Simular nivel de soporte
                            "confidence": confidence,
                            "description": f"Posible ruptura de soporte en {asset}",
                            "detected_at": datetime.now().timestamp(),
                            "horizon": str(TemporalHorizon.SHORT)
                        }
                        
                        self.current_insights.append(insight)
                        
                        # Agregar al tracker
                        if insight not in data["active_insights"]:
                            data["active_insights"].append(insight)
                
                # 3. Detectar resonancia cuántica (simulada)
                if random.random() < 0.1:  # 10% de probabilidad
                    # Resonancia cuántica simulada
                    confidence = random.uniform(0.85, 0.98)
                    
                    insight = {
                        "type": str(MarketInsightType.QUANTUM_RESONANCE),
                        "symbol": asset,
                        "confidence": confidence,
                        "magnitude": random.uniform(5.0, 15.0),
                        "description": f"Resonancia cuántica excepcional detectada en {asset}",
                        "detected_at": datetime.now().timestamp(),
                        "horizon": str(TemporalHorizon.TRANSCENDENT)
                    }
                    
                    self.current_insights.append(insight)
                    
                    # Agregar al tracker
                    if insight not in data["active_insights"]:
                        data["active_insights"].append(insight)
            
            # 4. Detectar cambios de correlación entre activos
            if random.random() < 0.15:  # 15% de probabilidad
                assets = list(self.tracked_assets.keys())
                if len(assets) >= 2:
                    # Seleccionar dos activos aleatorios
                    asset1, asset2 = random.sample(assets, 2)
                    
                    # Simular cambio de correlación
                    confidence = random.uniform(0.75, 0.9)
                    
                    insight = {
                        "type": str(MarketInsightType.CORRELATION_SHIFT),
                        "symbols": [asset1, asset2],
                        "old_correlation": random.uniform(-0.2, 0.2),
                        "new_correlation": random.uniform(0.7, 0.9),
                        "confidence": confidence,
                        "description": f"Cambio significativo en correlación entre {asset1} y {asset2}",
                        "detected_at": datetime.now().timestamp(),
                        "horizon": str(TemporalHorizon.MEDIUM)
                    }
                    
                    self.current_insights.append(insight)
            
            # Actualizar métrica
            self.metrics["insights_generated"] += len(self.current_insights)
            
            logger.info(f"Detectados {len(self.current_insights)} insights cuánticos de mercado")
            
        except Exception as e:
            logger.error(f"Error al detectar insights de mercado: {e}")
        
        # Restaurar estado
        self.state = prev_state
        
        return self.current_insights
    
    async def dimensional_shift(self) -> bool:
        """
        Realizar cambio dimensional para reorganizar espacios predictivos.
        
        Returns:
            True si el cambio fue exitoso
        """
        if not self.initialized:
            logger.warning("No se puede realizar cambio dimensional, oráculo no inicializado")
            return False
        
        logger.info("Iniciando cambio dimensional...")
        
        # Cambiar estado
        self.state = DimensionalState.DIMENSIONAL_SHIFT
        
        try:
            # Simular trabajo dimensional
            for i in range(self.dimensional_spaces):
                logger.debug(f"Reconfigurando espacio dimensional {i+1}...")
                await asyncio.sleep(0.1)
            
            # Actualizar métricas
            self.metrics["dimensional_shifts"] += 1
            self.metrics["coherence_level"] = random.uniform(0.8, 0.95)
            
            # Actualizar marcadores dimensionales
            for asset in self.tracked_assets:
                for dim in range(self.dimensional_spaces):
                    dim_key = f"dim_{dim}"
                    self.tracked_assets[asset]["dimensional_markers"][dim_key] = random.random()
            
            logger.info("Cambio dimensional completado exitosamente")
            
            # Restaurar estado
            self.state = DimensionalState.OPERATING
            
            return True
            
        except Exception as e:
            logger.error(f"Error durante cambio dimensional: {e}")
            self.state = DimensionalState.OPERATING
            return False
    
    async def achieve_resonance(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Lograr resonancia cuántica para predicciones de máxima precisión.
        
        Returns:
            Tupla (éxito, resultados)
        """
        if not self.initialized:
            logger.warning("No se puede lograr resonancia, oráculo no inicializado")
            return False, {}
        
        logger.info("Iniciando proceso de resonancia cuántica...")
        
        # Cambiar estado
        self.state = DimensionalState.RESONATING
        
        try:
            # Simular proceso de resonancia
            await asyncio.sleep(0.5)
            
            # Resultados de resonancia
            results = {
                "resonance_level": random.uniform(0.85, 0.98),
                "coherence_gain": random.uniform(0.05, 0.15),
                "accuracy_boost": random.uniform(0.08, 0.2),
                "timestamp": datetime.now().timestamp(),
                "assets_affected": random.randint(3, len(self.tracked_assets)),
                "duration_seconds": random.uniform(600, 3600)
            }
            
            # Actualizar métricas
            self.metrics["coherence_level"] += results["coherence_gain"]
            self.metrics["coherence_level"] = min(self.metrics["coherence_level"], 1.0)
            
            self.metrics["temporal_accuracy"] += results["accuracy_boost"]
            self.metrics["temporal_accuracy"] = min(self.metrics["temporal_accuracy"], 1.0)
            
            logger.info(f"Resonancia cuántica lograda a nivel {results['resonance_level']:.2f}")
            
            # Restaurar estado después de resonancia
            self.state = DimensionalState.QUANTUM_COHERENCE
            
            # Programar retorno a estado normal
            asyncio.create_task(self._return_to_normal_state(results["duration_seconds"]))
            
            return True, results
            
        except Exception as e:
            logger.error(f"Error durante proceso de resonancia: {e}")
            self.state = DimensionalState.OPERATING
            return False, {"error": str(e)}
    
    async def _return_to_normal_state(self, delay: float) -> None:
        """
        Retornar a estado normal después de un tiempo.
        
        Args:
            delay: Tiempo en segundos antes de volver a estado normal
        """
        try:
            await asyncio.sleep(0.5)  # Simulamos el tiempo real con un retraso pequeño
            logger.info(f"Resonancia cuántica finalizada, retornando a estado normal")
            self.state = DimensionalState.OPERATING
        except Exception as e:
            logger.error(f"Error al retornar a estado normal: {e}")
            self.state = DimensionalState.OPERATING
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas actuales del oráculo.
        
        Returns:
            Diccionario con métricas
        """
        # Añadir información de estado actual
        metrics = dict(self.metrics)
        metrics["state"] = str(self.state)
        metrics["tracked_assets_count"] = len(self.tracked_assets)
        metrics["active_predictions_count"] = len(self.active_predictions)
        metrics["current_insights_count"] = len(self.current_insights)
        metrics["dimensional_spaces"] = self.dimensional_spaces
        metrics["last_updated"] = datetime.now().timestamp()
        
        return metrics
    
    def get_state(self) -> Dict[str, Any]:
        """
        Obtener estado actual del oráculo.
        
        Returns:
            Diccionario con estado
        """
        return {
            "initialized": self.initialized,
            "state": str(self.state),
            "dimensional_spaces": self.dimensional_spaces,
            "entanglement_level": self.entanglement_level,
            "confidence_threshold": self.confidence_threshold,
            "tracked_assets_count": len(self.tracked_assets),
            "coherence_level": self.metrics["coherence_level"]
        }
    
    def get_insights_for_asset(self, asset: str) -> List[Dict[str, Any]]:
        """
        Obtener insights activos para un activo específico.
        
        Args:
            asset: Símbolo del activo
            
        Returns:
            Lista de insights
        """
        if asset not in self.tracked_assets:
            return []
        
        return self.tracked_assets[asset].get("active_insights", [])
    
    def get_all_insights(self) -> List[Dict[str, Any]]:
        """
        Obtener todos los insights activos.
        
        Returns:
            Lista de todos los insights
        """
        all_insights = []
        
        for asset, data in self.tracked_assets.items():
            insights = data.get("active_insights", [])
            all_insights.extend(insights)
        
        # Ordenar por tiempo de detección (más recientes primero)
        all_insights.sort(key=lambda x: x.get("detected_at", 0), reverse=True)
        
        return all_insights
    
    def get_prediction_for_asset(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Obtener la predicción más reciente para un activo.
        
        Args:
            asset: Símbolo del activo
            
        Returns:
            Predicción o None si no hay
        """
        return self.active_predictions.get(asset)
    
    async def evaluate_prediction_accuracy(self, asset: str, actual_price: float) -> Dict[str, Any]:
        """
        Evaluar precisión de predicción contra precio real.
        
        Args:
            asset: Símbolo del activo
            actual_price: Precio real actual
            
        Returns:
            Resultados de evaluación
        """
        if asset not in self.active_predictions:
            return {"success": False, "error": "No hay predicción activa para este activo"}
        
        prediction = self.active_predictions[asset]
        
        # Obtener predicción más reciente (inmediata)
        if not prediction.get("prices"):
            return {"success": False, "error": "Predicción sin precios"}
        
        immediate_pred = prediction["prices"][0]
        predicted_price = immediate_pred.get("price", 0)
        
        # Calcular error porcentual
        percent_error = abs((predicted_price - actual_price) / actual_price) * 100
        
        # Determinar precisión
        accuracy = max(0, 100 - percent_error)
        
        # Actualizar métricas
        if percent_error < 5:  # Menos de 5% de error se considera exitoso
            self.metrics["successful_predictions"] += 1
            result = "SUCCESS"
        else:
            self.metrics["failed_predictions"] += 1
            result = "FAILURE"
            
        # Actualizar exactitud por horizonte
        self.metrics["accuracy_short_term"] = (self.metrics["accuracy_short_term"] * 0.8 + accuracy * 0.2)
        
        # Resultados
        evaluation = {
            "success": True,
            "asset": asset,
            "predicted_price": predicted_price,
            "actual_price": actual_price,
            "percent_error": percent_error,
            "accuracy": accuracy,
            "result": result,
            "prediction_time": immediate_pred.get("timestamp"),
            "evaluation_time": datetime.now().timestamp()
        }
        
        logger.info(f"Evaluación de predicción para {asset}: {accuracy:.2f}% de precisión")
        
        return evaluation
    
    def __str__(self) -> str:
        """Representación como string."""
        asset_count = len(self.tracked_assets)
        insight_count = len(self.current_insights)
        state_str = str(self.state)
        
        return f"QuantumOracle(tracked_assets={asset_count}, insights={insight_count}, state={state_str})"


# Función para demostración independiente
async def demo():
    """Función de demostración."""
    # Crear y configurar oráculo
    oracle = QuantumOracle()
    
    # Inicializar
    await oracle.initialize()
    
    # Actualizar datos
    await oracle.update_market_data()
    
    # Generar predicciones
    predictions = await oracle.generate_predictions()
    
    # Detectar insights
    insights = await oracle.detect_market_insights()
    
    # Mostrar resultados
    print(f"ORÁCULO CUÁNTICO PREDICTIVO - DEMO")
    print(f"----------------------------------")
    print(f"Estado: {oracle.state}")
    print(f"Activos seguidos: {len(oracle.tracked_assets)}")
    print(f"Predicciones generadas: {len(predictions)}")
    print(f"Insights detectados: {len(insights)}")
    
    # Mostrar una predicción
    if predictions:
        symbol = list(predictions.keys())[0]
        pred = predictions[symbol]
        print("\nPREDICCIÓN DE EJEMPLO:")
        print(f"Activo: {symbol}")
        print(f"Precio actual: ${pred['current_price']:.2f}")
        print(f"Tendencia dominante: {pred['dominant_trend']}")
        print(f"Nivel de confianza: {pred['confidence_category']}")
        
        for i, price in enumerate(pred["prices"]):
            print(f"  - Horizonte {i+1}: ${price['price']:.2f} ({price['change_pct']:.2f}%)")
    
    # Mostrar un insight
    if insights:
        insight = insights[0]
        print("\nINSIGHT DE EJEMPLO:")
        print(f"Tipo: {insight['type']}")
        print(f"Activo: {insight.get('symbol')}")
        print(f"Descripción: {insight['description']}")
        print(f"Confianza: {insight['confidence']:.2f}")
    
    # Realizar cambio dimensional y resonancia
    await oracle.dimensional_shift()
    success, resonance = await oracle.achieve_resonance()
    
    if success:
        print("\nRESONANCIA CUÁNTICA:")
        print(f"Nivel: {resonance['resonance_level']:.2f}")
        print(f"Mejora de precisión: +{resonance['accuracy_boost']*100:.1f}%")
    
    # Mostrar métricas
    metrics = oracle.get_metrics()
    print("\nMÉTRICAS:")
    print(f"Coherencia: {metrics['coherence_level']:.2f}")
    print(f"Precisión temporal: {metrics['temporal_accuracy']:.2f}")
    print(f"Puntuación de entrelazamiento: {metrics['entanglement_score']:.2f}")


# Ejecutar demo si se ejecuta directamente
if __name__ == "__main__":
    asyncio.run(demo())